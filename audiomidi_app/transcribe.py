from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, resample_poly, stft, butter, sosfilt

from audiomidi_app.midi import NoteEvent
from audiomidi_app.postprocess import full_postprocess, PostProcessConfig, OnsetDetector

class NeuralTranscriber(Protocol):
    name: str
    
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]: ...
    
    def get_onset_times(self, samples: np.ndarray, sample_rate: int) -> list[float]:
        try:
            import librosa
            return librosa.onset.onset_detect(
                y=samples, sr=sample_rate, hop_length=256,
                backtrack=True, units="time"
            ).tolist()
        except Exception:
            return []


class DSPTranscriber(Protocol):
    name: str
    
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]: ...
from audiomidi_app.voice_separation import (
    separate_voices,
    VoiceSeparationConfig,
    VoiceSeparationResult,
)


class Transcriber(Protocol):
    name: str

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]: ...


@dataclass(frozen=True)
class SpectralPeaksConfig:
    target_sr: int = 22050
    n_fft: int = 4096
    hop_length: int = 512
    fmin_hz: float = 55.0
    fmax_hz: float = 1760.0
    peak_prominence: float = 4.0
    peak_height_db: float = -50.0
    min_note_s: float = 0.06
    release_s: float = 0.04
    frequency_interpolation: bool = True
    dynamic_threshold: bool = False
    use_onset_detection: bool = False
    use_temporal_smoothing: bool = False


@dataclass(frozen=True)
class HarmonicSalienceConfig:
    target_sr: int = 22050
    n_fft: int = 8192
    hop_length: int = 512
    fmin_hz: float = 27.5
    fmax_hz: float = 4200.0
    peak_prominence: float = 2.0
    peak_height_db: float = -55.0
    min_note_s: float = 0.05
    release_s: float = 0.03
    harmonic_weight: float = 1.5
    max_polyphony: int = 6
    frequency_interpolation: bool = True
    use_onset_detection: bool = True
    use_temporal_smoothing: bool = True
    use_cqt: bool = False


@dataclass(frozen=True)
class PedalConfig:
    enabled: bool = True
    min_sustained_notes: int = 2
    max_extension_ratio: float = 4.0
    min_extension_s: float = 0.05


def parabolic_interpolation(freqs: np.ndarray, mags: np.ndarray, peak_idx: int) -> float:
    if peak_idx <= 0 or peak_idx >= len(mags) - 1:
        return freqs[peak_idx]
    
    alpha = mags[peak_idx - 1]
    beta = mags[peak_idx]
    gamma = mags[peak_idx + 1]
    
    denom = alpha - 2 * beta + gamma
    if abs(denom) < 1e-10:
        return freqs[peak_idx]
    
    p = 0.5 * (alpha - gamma) / denom
    interpolated_freq = freqs[peak_idx] + p * (freqs[peak_idx + 1] - freqs[peak_idx])
    return interpolated_freq


def hz_to_midi_with_cents(freq_hz: float) -> tuple[int, float]:
    if freq_hz <= 0:
        return 0, 0.0
    midi_float = 69.0 + 12.0 * np.log2(freq_hz / 440.0)
    midi_note = int(np.clip(np.rint(midi_float), 0, 127))
    cents = (midi_float - midi_note) * 100.0
    return midi_note, cents


def compute_harmonic_salience(
    mag_linear: np.ndarray,
    f: np.ndarray,
    n_harmonics: int = 8,
    harmonic_decay: float = 0.8,
) -> np.ndarray:
    weights = harmonic_decay ** np.arange(n_harmonics)
    weights /= weights.sum()

    sal = np.zeros_like(mag_linear)
    for h_idx in range(n_harmonics):
        h = h_idx + 1
        target = f * h
        j = np.searchsorted(f, target, side="left")
        valid = (j >= 0) & (j < len(f))
        sal[valid, :] += mag_linear[j[valid], :] * weights[h_idx]

    return sal


def detect_onsets(samples: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    try:
        import librosa
        onset_frames = librosa.onset.onset_detect(
            y=samples,
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
            units="frames"
        )
        return onset_frames
    except ImportError:
        return np.array([], dtype=int)


def detect_bpm(samples: np.ndarray, sr: int) -> float:
    try:
        import librosa
        tempo, _ = librosa.beat.beat_track(y=samples, sr=sr)
        return float(tempo) if tempo > 0 else 120.0
    except ImportError:
        return 120.0


def _bandpass(samples: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    nyq = sr / 2.0
    sos = butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype="band", output="sos")
    return sosfilt(sos, samples)


def transcribe_drums(samples: np.ndarray, sr: int) -> list[NoteEvent]:
    try:
        import librosa
    except ImportError:
        return []

    bands = {
        36: (40,  200),
        38: (200, 2000),
        42: (6000, 16000),
    }

    events = []
    for gm_note, (flo, fhi) in bands.items():
        y_band = _bandpass(samples, sr, flo, min(fhi, sr / 2 - 1))
        times = librosa.onset.onset_detect(
            y=y_band, sr=sr, hop_length=256,
            backtrack=True, units="time"
        )
        for t in times:
            t = max(0.0, float(t))
            events.append(NoteEvent(note=gm_note, start_s=t,
                                    end_s=t + 0.05, velocity=90))

    events.sort(key=lambda e: e.start_s)
    return events


def apply_pedal_correction(
    events: list[NoteEvent],
    pedal_events: list[dict],
    config: PedalConfig | None = None,
) -> list[NoteEvent]:
    if config is None:
        config = PedalConfig()
    
    if not config.enabled or not pedal_events:
        return events
    
    if not events:
        return events
    
    events = [NoteEvent(
        note=e.note,
        start_s=e.start_s,
        end_s=e.end_s,
        velocity=e.velocity,
        confidence=getattr(e, 'confidence', 1.0),
    ) for e in events]
    
    notes_by_pedal: dict[int, list[int]] = {}
    
    for pedal_idx, pedal in enumerate(pedal_events):
        pedal_on = float(pedal["start_time"])
        pedal_off = float(pedal["end_time"])
        
        sustained_notes: list[int] = []
        
        for note_idx, note in enumerate(events):
            if note.start_s >= pedal_on and note.start_s <= pedal_off:
                sustained_notes.append(note_idx)
            elif note.start_s < pedal_on and note.end_s > pedal_on:
                sustained_notes.append(note_idx)
        
        if len(sustained_notes) >= config.min_sustained_notes:
            notes_by_pedal[pedal_idx] = sustained_notes
    
    for pedal_idx, note_indices in notes_by_pedal.items():
        if not note_indices:
            continue
            
        pedal = pedal_events[pedal_idx]
        pedal_off = float(pedal["end_time"])
        
        for note_idx in note_indices:
            note = events[note_idx]
            
            if note.end_s < pedal_off:
                extension = pedal_off - note.end_s
                original_duration = note.end_s - note.start_s
                max_extension = original_duration * config.max_extension_ratio
                
                if extension >= config.min_extension_s and extension <= max_extension:
                    events[note_idx] = NoteEvent(
                        note=note.note,
                        start_s=note.start_s,
                        end_s=pedal_off,
                        velocity=note.velocity,
                        confidence=getattr(note, 'confidence', 1.0),
                    )
    
    return events


class SpectralPeaksTranscriber:
    name = "Spectral Peaks [DEBUG ONLY]"

    def __init__(self, config: SpectralPeaksConfig | None = None) -> None:
        self._cfg = config or SpectralPeaksConfig()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
        x = samples.astype(np.float32, copy=False)
        if x.ndim != 1:
            x = x.mean(axis=-1)
        
        onset_frames = np.array([], dtype=int)
        onset_set = set()
        if self._cfg.use_onset_detection:
            onset_frames = detect_onsets(x, sample_rate, self._cfg.hop_length)
            onset_set = set(onset_frames.tolist())
        
        f, t, z = stft(
            x,
            fs=sample_rate,
            nperseg=self._cfg.n_fft,
            noverlap=self._cfg.n_fft - self._cfg.hop_length,
            padded=False,
            boundary=None,
        )
        
        mag = np.abs(z) + 1e-9
        mag_db = 20.0 * np.log10(mag)
        
        if self._cfg.use_temporal_smoothing:
            mag_db = median_filter(mag_db, size=(1, 3))
        
        band = (f >= self._cfg.fmin_hz) & (f <= self._cfg.fmax_hz)
        f_band = f[band]
        mag_db = mag_db[band, :]
        mag = mag[band, :]
        
        dt = float(self._cfg.hop_length) / float(sample_rate)
        
        active: dict[int, tuple[float, float, float]] = {}
        events: list[NoteEvent] = []
        
        for frame_i in range(mag_db.shape[1]):
            is_onset = frame_i in onset_set
            
            frame_db = mag_db[:, frame_i]
            frame_mag = mag[:, frame_i]
            peaks, props = find_peaks(
                frame_db,
                height=self._cfg.peak_height_db,
                prominence=self._cfg.peak_prominence,
            )
            
            present: dict[int, float] = {}
            if peaks.size:
                if self._cfg.frequency_interpolation:
                    for idx in peaks:
                        interp_freq = parabolic_interpolation(f_band, frame_db, idx)
                        midi_note, cents = hz_to_midi_with_cents(interp_freq)
                        
                        if 21 <= midi_note <= 108:
                            if abs(cents) < 50 or midi_note == 21 or midi_note == 108:
                                amp = frame_mag[idx]
                                if midi_note not in present or amp > present[midi_note]:
                                    present[midi_note] = float(amp)
                else:
                    peak_freq = f_band[peaks]
                    peak_amp = frame_mag[peaks]
                    notes = hz_to_midi(peak_freq)
                    for n, amp in zip(notes, peak_amp, strict=False):
                        if 21 <= n <= 108:
                            present[n] = max(present.get(n, 0.0), float(amp))
            
            now_s = frame_i * dt
            
            to_close: list[int] = []
            for n, (start_s, last_s, max_amp) in active.items():
                if is_onset and n in present:
                    dur = now_s - start_s
                    if dur >= self._cfg.min_note_s:
                        events.append(
                            NoteEvent(
                                note=n,
                                start_s=start_s,
                                end_s=now_s,
                                velocity=amp_to_velocity(max_amp),
                                confidence=0.5,
                            )
                        )
                    to_close.append(n)
                elif n in present:
                    active[n] = (start_s, now_s, max(max_amp, present[n]))
                else:
                    if now_s - last_s >= self._cfg.release_s:
                        dur = last_s - start_s
                        if dur >= self._cfg.min_note_s:
                            events.append(
                                NoteEvent(
                                    note=n,
                                    start_s=start_s,
                                    end_s=last_s,
                                    velocity=amp_to_velocity(max_amp),
                                    confidence=0.5,
                                )
                            )
                        to_close.append(n)
            
            for n in to_close:
                active.pop(n, None)
            
            for n, amp in present.items():
                if n not in active:
                    active[n] = (now_s, now_s, amp)
        
        end_s = mag_db.shape[1] * dt
        for n, (start_s, last_s, max_amp) in active.items():
            last = min(end_s, last_s)
            dur = last - start_s
            if dur >= self._cfg.min_note_s:
                events.append(
                    NoteEvent(
                        note=n,
                        start_s=start_s,
                        end_s=last,
                        velocity=amp_to_velocity(max_amp),
                        confidence=0.5,
                    )
                )
        
        events.sort(key=lambda e: (e.start_s, e.note))
        return merge_overlaps(events)


class HarmonicSalienceTranscriber:
    name = "Harmonic Salience [DEBUG ONLY]"

    def __init__(self, config: HarmonicSalienceConfig | None = None) -> None:
        self._cfg = config or HarmonicSalienceConfig()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
        x = samples.astype(np.float32, copy=False)
        if x.ndim != 1:
            x = x.mean(axis=-1)
        
        onset_frames = np.array([], dtype=int)
        onset_set = set()
        if self._cfg.use_onset_detection:
            onset_frames = detect_onsets(x, sample_rate, self._cfg.hop_length)
            onset_set = set(onset_frames.tolist())
        
        if self._cfg.use_cqt:
            try:
                import librosa
                n_bins = 288
                cqt = librosa.cqt(
                    x,
                    sr=sample_rate,
                    hop_length=self._cfg.hop_length,
                    fmin=librosa.note_to_hz('A0'),
                    n_bins=n_bins,
                    bins_per_octave=36
                )
                mag = np.abs(cqt) + 1e-12
                f = librosa.cqt_frequencies(n_bins, fmin=librosa.note_to_hz('A0'), bins_per_octave=36)
            except ImportError:
                mag, f = self._compute_stft_mag(x, sample_rate)
        else:
            mag, f = self._compute_stft_mag(x, sample_rate)
        
        harmonic_decay = 1.0 - (1.0 / (self._cfg.harmonic_weight + 0.5))
        salience = compute_harmonic_salience(mag, f, n_harmonics=8, harmonic_decay=harmonic_decay)
        sal_db = 20.0 * np.log10(salience + 1e-12)
        
        if self._cfg.use_temporal_smoothing:
            sal_db = median_filter(sal_db, size=(1, 3))
        
        band = (f >= self._cfg.fmin_hz) & (f <= self._cfg.fmax_hz)
        f_band = f[band]
        sal_band = sal_db[band, :]
        mag_band = mag[band, :]
        
        dt = float(self._cfg.hop_length) / float(sample_rate)
        
        active: dict[int, tuple[float, float, float]] = {}
        events: list[NoteEvent] = []
        
        for frame_i in range(sal_band.shape[1]):
            is_onset = frame_i in onset_set
            
            frame_sal = sal_band[:, frame_i]
            frame_mag = mag_band[:, frame_i]
            
            peaks, props = find_peaks(
                frame_sal,
                height=self._cfg.peak_height_db,
                prominence=self._cfg.peak_prominence,
            )
            
            present: dict[int, float] = {}
            if peaks.size:
                peak_freq = f_band[peaks]
                peak_sal = frame_sal[peaks]
                peak_amp = frame_mag[peaks]
                
                sorted_idx = np.argsort(peak_sal)[::-1]
                count = 0
                
                for idx in sorted_idx:
                    if count >= self._cfg.max_polyphony:
                        break
                    
                    freq = peak_freq[idx]
                    amp = peak_amp[idx]
                    midi_note, cents = hz_to_midi_with_cents(freq)
                    
                    if 21 <= midi_note <= 108:
                        if abs(cents) < 50 or midi_note == 21 or midi_note == 108:
                            if midi_note not in present or amp > present[midi_note]:
                                present[midi_note] = float(amp)
                                count += 1
            
            now_s = frame_i * dt
            
            to_close: list[int] = []
            for n, (start_s, last_s, max_amp) in active.items():
                if is_onset and n in present:
                    dur = now_s - start_s
                    if dur >= self._cfg.min_note_s:
                        events.append(
                            NoteEvent(
                                note=n,
                                start_s=start_s,
                                end_s=now_s,
                                velocity=amp_to_velocity(max_amp),
                                confidence=0.5,
                            )
                        )
                    to_close.append(n)
                elif n in present:
                    active[n] = (start_s, now_s, max(max_amp, present[n]))
                else:
                    if now_s - last_s >= self._cfg.release_s:
                        dur = last_s - start_s
                        if dur >= self._cfg.min_note_s:
                            events.append(
                                NoteEvent(
                                    note=n,
                                    start_s=start_s,
                                    end_s=last_s,
                                    velocity=amp_to_velocity(max_amp),
                                    confidence=0.5,
                                )
                            )
                        to_close.append(n)
            
            for n in to_close:
                active.pop(n, None)
            
            for n, amp in present.items():
                if n not in active:
                    active[n] = (now_s, now_s, amp)
        
        end_s = sal_band.shape[1] * dt
        for n, (start_s, last_s, max_amp) in active.items():
            last = min(end_s, last_s)
            dur = last_s - start_s
            if dur >= self._cfg.min_note_s:
                events.append(
                    NoteEvent(
                        note=n,
                        start_s=start_s,
                        end_s=last,
                        velocity=amp_to_velocity(max_amp),
                        confidence=0.5,
                    )
                )
        
        events.sort(key=lambda e: (e.start_s, e.note))
        return merge_overlaps(events)
    
    def _compute_stft_mag(self, x: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        f, t, z = stft(
            x,
            fs=sample_rate,
            nperseg=self._cfg.n_fft,
            noverlap=self._cfg.n_fft - self._cfg.hop_length,
            padded=True,
            boundary=None,
        )
        mag = np.abs(z) + 1e-12
        return mag, f


def amp_to_velocity(amp: float, mode: str = "linear") -> int:
    if mode == "piano":
        db = 20.0 * np.log10(max(1e-9, amp))
        normalized = np.clip((db + 70.0) / 70.0, 0.0, 1.0)
        v = int(normalized ** (1/3) * 126) + 1
    else:
        db = 20.0 * np.log10(max(1e-9, amp))
        v = int(np.clip((db + 60.0) * 2.2, 1.0, 127.0))
    return max(1, min(127, int(v)))


def hz_to_midi(freqs: np.ndarray) -> np.ndarray:
    midi = 69.0 + 12.0 * np.log2(np.maximum(freqs, 1e-9) / 440.0)
    return np.clip(np.rint(midi), 0, 127).astype(int)


def merge_overlaps(events: list[NoteEvent]) -> list[NoteEvent]:
    by_note: dict[int, list[NoteEvent]] = {}
    for e in events:
        by_note.setdefault(e.note, []).append(e)

    merged: list[NoteEvent] = []
    for note, es in by_note.items():
        es.sort(key=lambda x: x.start_s)
        cur = es[0]
        for nxt in es[1:]:
            if nxt.start_s <= cur.end_s + 1e-3:
                cur = NoteEvent(
                    note=note,
                    start_s=cur.start_s,
                    end_s=max(cur.end_s, nxt.end_s),
                    velocity=max(cur.velocity, nxt.velocity),
                    confidence=max(getattr(cur, 'confidence', 1.0), getattr(nxt, 'confidence', 1.0)),
                )
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)

    merged.sort(key=lambda e: (e.start_s, e.note))
    return merged


def available_transcribers() -> list[Transcriber]:
    transcribers: list[Transcriber] = []
    
    pt = try_piano_transcription_transcriber()
    if pt is not None:
        transcribers.append(pt)
    
    bp = try_basic_pitch_transcriber()
    if bp is not None:
        transcribers.append(bp)
    
    dsp_fallback: list[Transcriber] = [
        HarmonicSalienceTranscriber(),
        SpectralPeaksTranscriber(),
    ]
    transcribers.extend(dsp_fallback)
    
    return transcribers


def available_neural_transcribers() -> list[NeuralTranscriber]:
    neural: list[NeuralTranscriber] = []
    
    pt = try_piano_transcription_transcriber()
    if pt is not None:
        neural.append(pt)
    
    bp = try_basic_pitch_transcriber()
    if bp is not None and hasattr(bp, 'get_onset_times'):
        neural.append(bp)
    
    return neural


def available_dsp_transcribers() -> list[DSPTranscriber]:
    return [
        HarmonicSalienceTranscriber(),
        SpectralPeaksTranscriber(),
    ]


def available_voice_separation_transcribers(
    voice_config: VoiceSeparationConfig | None = None,
) -> list[Transcriber]:
    base_transcribers = available_transcribers()
    
    voice_transcribers: list[Transcriber] = []
    for base in base_transcribers:
        voice_transcribers.append(
            VoiceSeparationTranscriber(base, voice_config)
        )
    
    return voice_transcribers


class VoiceSeparationTranscriber:
    name = "Voice Separation"

    def __init__(
        self,
        base_transcriber: Transcriber,
        voice_config: VoiceSeparationConfig | None = None,
    ) -> None:
        self._base = base_transcriber
        self._voice_config = voice_config or VoiceSeparationConfig()
        self._name = f"{base_transcriber.name} + Voice Sep"

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
        events = self._base.transcribe(samples, sample_rate)
        result = separate_voices(events, self._voice_config)
        all_notes: list[NoteEvent] = []
        for voice in result.voices:
            all_notes.extend(voice.notes)
        return sorted(all_notes, key=lambda n: (n.start_s, n.note))
    
    def separate(self, samples: np.ndarray, sample_rate: int) -> VoiceSeparationResult:
        events = self._base.transcribe(samples, sample_rate)
        return separate_voices(events, self._voice_config)


def try_piano_transcription_transcriber() -> NeuralTranscriber | None:
    try:
        from piano_transcription_inference import PianoTranscription
    except Exception:
        return None

    class _PianoTranscriptionTranscriber:
        name = "Piano Transcription (Neural)"

        def __init__(self):
            from piano_transcription_inference import PianoTranscription
            self._model = PianoTranscription(device="cpu", duration=None)
            self._pedal_config = PedalConfig()

        def transcribe(self, samples: np.ndarray, sample_rate_in: int) -> list[NoteEvent]:
            import soundfile as sf
            from pathlib import Path
            from piano_transcription_inference import sample_rate as PT_SR

            if sample_rate_in != PT_SR:
                g = np.gcd(sample_rate_in, PT_SR)
                samples_resampled = resample_poly(
                    samples, 
                    PT_SR // g, 
                    sample_rate_in // g
                ).astype(np.float32)
                sample_rate_out = PT_SR
            else:
                samples_resampled = samples
                sample_rate_out = sample_rate_in
            
            onset_detector = OnsetDetector(sample_rate_out)
            onset_detector.detect(samples_resampled)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, mode="wb") as f:
                temp_wav = f.name
            
            try:
                sf.write(temp_wav, samples_resampled, sample_rate_out)
                result = self._model.transcribe(temp_wav, midi_path=None)
                
                events = []
                for note_info in result["notes"]:
                    events.append(NoteEvent(
                        note=int(note_info["midi_note"]),
                        start_s=float(note_info["onset_time"]),
                        end_s=float(note_info["offset_time"]),
                        velocity=int(np.clip(note_info["velocity"] * 127, 1, 127)),
                        confidence=1.0,
                    ))
                
                pedal_events = result.get("pedal", [])
                
                events = apply_pedal_correction(events, pedal_events, self._pedal_config)
                
                events = full_postprocess(
                    events, 
                    samples_resampled, 
                    sample_rate_out,
                    bpm=120.0,
                    onset_detector=onset_detector,
                    config=None
                )
                
                events.sort(key=lambda e: (e.start_s, e.note))
                return events
            finally:
                try:
                    Path(temp_wav).unlink(missing_ok=True)
                except Exception:
                    pass

    return _PianoTranscriptionTranscriber()


def try_basic_pitch_transcriber() -> Transcriber | None:
    try:
        from basic_pitch.inference import predict, predict_and_save
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except Exception:
        return None

    class _BasicPitchTranscriber:
        name = "Basic Pitch"

        def __init__(self):
            self._model_path = ICASSP_2022_MODEL_PATH

        def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
            import soundfile as sf
            from pathlib import Path

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, mode="wb") as f:
                temp_wav = f.name
            
            try:
                sf.write(temp_wav, samples, sample_rate)
                model_output, midi_data, note_events = predict(temp_wav, self._model_path)
                
                events = []
                for note in note_events:
                    events.append(NoteEvent(
                        note=int(note.pitch),
                        start_s=float(note.start_time),
                        end_s=float(note.end_time),
                        velocity=int(min(127, max(1, note.amplitude * 127))),
                        confidence=1.0,
                    ))
                return events
            finally:
                try:
                    Path(temp_wav).unlink(missing_ok=True)
                except Exception:
                    pass

        def transcribe_file(self, in_path: str, out_dir: str) -> str:
            import os
            from pathlib import Path
            from basic_pitch.inference import predict_and_save

            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            predict_and_save(
                [in_path],
                output_directory=str(out),
                save_midi=True,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=self._model_path,
            )
            midi_path = out / (Path(in_path).stem + ".mid")
            if not midi_path.exists():
                candidates = list(out.glob("*.mid"))
                if not candidates:
                    raise RuntimeError("Basic Pitch 未生成MIDI输出")
                midi_path = candidates[0]
            return os.fspath(midi_path)

    return _BasicPitchTranscriber()


class ProductionPipeline:
    """
    Modern Production Transcription Pipeline
    完整的现代 AI 扒谱系统架构
    """
    
    name = "Production Transcription Pipeline"
    
    def __init__(
        self,
        use_separation: bool = True,
        use_ensemble: bool = True,
        enable_symbolic_refinement: bool = True,
        enable_beat_tracking: bool = True,
        config: dict | None = None,
    ):
        self._config = config or {}
        
        self._separator = None
        self._router = None
        self._symbolic_refiner = None
        self._beat_tracker = None
        self._renderer = None
        
        self._use_separation = use_separation
        self._use_ensemble = use_ensemble
        self._enable_symbolic_refinement = enable_symbolic_refinement
        self._enable_beat_tracking = enable_beat_tracking
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所有组件"""
        if self._use_separation:
            try:
                from audiomidi_app.pipeline.separation import HTDemucsSeparator
                self._separator = HTDemucsSeparator()
            except Exception as e:
                print(f"Warning: Stem separation unavailable: {e}")
                self._separator = None
        
        if self._use_ensemble:
            try:
                from audiomidi_app.pipeline.transcription import EnsembleTranscriptionEngine
                from audiomidi_app.pipeline.transcription import TranscriptionConfig
                
                config = TranscriptionConfig(
                    use_separation=False,
                    use_mt3=True,
                    use_piano_transcription=True,
                    use_basic_pitch=True,
                )
                self._router = EnsembleTranscriptionEngine(config)
            except Exception as e:
                print(f"Warning: Ensemble transcription unavailable: {e}")
                self._router = None
        
        if self._enable_symbolic_refinement:
            try:
                from audiomidi_app.pipeline.symbolic_refinement import (
                    SymbolicRefiner,
                    SymbolicRefinementConfig,
                )
                
                config = SymbolicRefinementConfig(
                    enable_harmony_check=True,
                    enable_voice_leading=True,
                    enable_tempo_constraints=True,
                )
                self._symbolic_refiner = SymbolicRefiner(config)
            except Exception as e:
                print(f"Warning: Symbolic refinement unavailable: {e}")
                self._symbolic_refiner = None
        
        if self._enable_beat_tracking:
            try:
                from audiomidi_app.rhythm.beat_tracking import BeatTracker
                self._beat_tracker = BeatTracker()
            except Exception as e:
                print(f"Warning: Beat tracking unavailable: {e}")
                self._beat_tracker = None
    
    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int = 44100,
    ) -> list[NoteEvent]:
        """
        完整的 production transcription 流程
        
        Pipeline:
        Audio → Stem Separation → Multi-Engine Transcription 
                → Symbolic Refinement → Beat-Aligned Quantization → MIDI
        """
        try:
            if self._separator is not None:
                stems = self._separator.separate(samples, sample_rate)
            else:
                from audiomidi_app.pipeline.separation import SeparatedStems
                stems = SeparatedStems(
                    vocals=None,
                    drums=None,
                    bass=None,
                    other=samples,
                    mixture=samples,
                    sample_rate=sample_rate,
                )
            
            notes = self._transcribe_stems(stems)
            
            if self._beat_tracker is not None:
                tempo_map = self._beat_tracker.track(samples, sample_rate)
                notes = self._align_to_tempo(notes, tempo_map)
            
            if self._symbolic_refiner is not None:
                notes = self._symbolic_refiner.refine(notes)
            
            notes = self._postprocess(notes, samples, sample_rate)
            
            notes.sort(key=lambda n: (n.start_s, n.note))
            return notes
            
        except Exception as e:
            print(f"Production pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _transcribe_stems(self, stems) -> list[NoteEvent]:
        """转录分离后的各个 stem"""
        if self._router is not None:
            try:
                notes = self._router.transcribe(stems.other, stems.sample_rate)
                return notes
            except Exception as e:
                print(f"Router transcription failed: {e}")
        
        from audiomidi_app.pipeline.transcription import PianoTranscriptionEngine
        
        engine = PianoTranscriptionEngine()
        notes = engine.transcribe(stems.other, stems.sample_rate)
        return notes
    
    def _align_to_tempo(self, notes: list[NoteEvent], tempo_map) -> list[NoteEvent]:
        """将音符对齐到节拍"""
        if not notes or not tempo_map.beats:
            return notes
        
        aligned = []
        for note in notes:
            nearest_beat = tempo_map.get_nearest_beat(note.start_s)
            
            deviation = abs(note.start_s - nearest_beat.time)
            threshold = 60.0 / tempo_map.bpm / 8
            
            if deviation < threshold and nearest_beat.confidence > 0.7:
                note = NoteEvent(
                    note=note.note,
                    start_s=nearest_beat.time,
                    end_s=note.end_s,
                    velocity=note.velocity,
                    confidence=getattr(note, 'confidence', 1.0),
                )
            
            aligned.append(note)
        
        return aligned
    
    def _postprocess(
        self,
        notes: list[NoteEvent],
        samples: np.ndarray,
        sample_rate: int,
    ) -> list[NoteEvent]:
        """后处理"""
        try:
            from audiomidi_app.postprocess import full_postprocess
            
            notes = full_postprocess(
                notes,
                samples,
                sample_rate,
                bpm=120.0,
            )
        except Exception as e:
            print(f"Postprocessing failed: {e}")
        
        return notes
    
    def transcribe_with_metadata(
        self,
        samples: np.ndarray,
        sample_rate: int = 44100,
    ) -> dict:
        """
        带元数据的转录
        
        Returns:
            {
                'notes': list[NoteEvent],
                'tempo': float,
                'beats': list[BeatInfo],
                'stems_used': dict,
            }
        """
        tempo_map = None
        if self._beat_tracker is not None:
            tempo_map = self._beat_tracker.track(samples, sample_rate)
        
        stems_used = {}
        if self._separator is not None:
            stems = self._separator.separate(samples, sample_rate)
            stems_used = {
                'vocals': stems.vocals is not None,
                'drums': stems.drums is not None,
                'bass': stems.bass is not None,
                'other': stems.other is not None,
            }
        
        notes = self.transcribe(samples, sample_rate)
        
        return {
            'notes': notes,
            'tempo': tempo_map.bpm if tempo_map else 120.0,
            'beats': tempo_map.beats if tempo_map else [],
            'stems_used': stems_used,
        }


def create_modern_pipeline() -> ProductionPipeline:
    """创建现代 production pipeline"""
    return ProductionPipeline(
        use_separation=True,
        use_ensemble=True,
        enable_symbolic_refinement=True,
        enable_beat_tracking=True,
    )


def create_simple_pipeline() -> ProductionPipeline:
    """创建简化版 pipeline - 只用 Piano Transcription"""
    return ProductionPipeline(
        use_separation=False,
        use_ensemble=False,
        enable_symbolic_refinement=True,
        enable_beat_tracking=True,
    )
