from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, resample_poly, stft, butter, sosfilt

from audiomidi_app.midi import NoteEvent
from audiomidi_app.postprocess import full_postprocess, PostProcessConfig, OnsetDetector

from audiomidi_app.voice_separation import (
    separate_voices,
    VoiceSeparationConfig,
    VoiceSeparationResult,
)


class Transcriber(Protocol):
    @property
    def name(self) -> str: ...

    def transcribe(self, samples: np.ndarray, sample_rate: int, progress_callback=None) -> list[NoteEvent]: ...


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
        # 先判断哪些谐波在频率范围内，避免越界
        valid = target <= f[-1]
        if not np.any(valid):
            continue
        j = np.searchsorted(f, target[valid], side="left")
        j = np.clip(j, 0, len(f) - 1)
        sal[valid, :] += mag_linear[j, :] * weights[h_idx]

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
        pedal_on = float(pedal.get("onset_time") or pedal.get("start_time") or pedal.get("on_time") or 0.0)
        pedal_off = float(pedal.get("offset_time") or pedal.get("end_time") or pedal.get("off_time") or 0.0)

        if pedal_off <= pedal_on:
            continue
        
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
        pedal_off = float(pedal.get("offset_time") or pedal.get("end_time") or pedal.get("off_time") or 0.0)
        
        for note_idx in note_indices:
            note = events[note_idx]
            
            if note.end_s < pedal_off:
                extension = pedal_off - note.end_s
                original_duration = note.end_s - note.start_s
                max_extension = min(
                        original_duration * config.max_extension_ratio,
                        2.0,  # 绝对上限 2 秒，避免短音被过度延长
                    )
                
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

    def transcribe(self, samples: np.ndarray, sample_rate: int, progress_callback=None) -> list[NoteEvent]:
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

    def transcribe(self, samples: np.ndarray, sample_rate: int, progress_callback=None) -> list[NoteEvent]:
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
            if nxt.start_s < cur.start_s + 0.020:
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


def try_piano_transcription_transcriber() -> Transcriber | None:
    """Try to load PianoTranscription model - may fail without correct download!"""
    try:
        from piano_transcription_inference import PianoTranscription
    except Exception:
        return None

    # Also check for model file
    from pathlib import Path
    model_path = Path.home() / "piano_transcription_inference_data" / "note_F1=0.9677_pedal_F1=0.9186.pth"
    if not model_path.exists():
        return None

    class _PianoTranscriptionTranscriber:
        name = "Piano Transcription (Neural)"

        def __init__(self):
            from piano_transcription_inference import PianoTranscription
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else \
                         "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
            except Exception:
                device = "cpu"
            print(f"[Piano Transcription] 使用设备: {device}")
            self._model = PianoTranscription(device=device)
            self._pedal_config = PedalConfig()

        def _transcribe_segment(self, audio_arr: np.ndarray, sample_rate_out: int) -> tuple[list[NoteEvent], list[dict]]:
            result = self._model.transcribe(audio_arr, midi_path=None)

            note_list = result.get("notes") or result.get("est_note_events") or []
            pedal_list = result.get("pedals") or result.get("est_pedal_events") or []

            return note_list, pedal_list

        def transcribe(self, samples: np.ndarray, sample_rate_in: int, progress_callback=None) -> list[NoteEvent]:
            import soundfile as sf
            from piano_transcription_inference import sample_rate as PT_SR

            if sample_rate_in != PT_SR:
                from scipy.signal import resample_poly
                g = np.gcd(sample_rate_in, PT_SR)
                samples_resampled = resample_poly(samples, PT_SR // g, sample_rate_in // g).astype(np.float32)
                sample_rate_out = PT_SR
            else:
                samples_resampled = samples.astype(np.float32, copy=False)
                sample_rate_out = sample_rate_in

            rms = float(np.sqrt(np.mean(samples_resampled ** 2)))
            if rms < 0.001:
                print(f"[Piano Transcription] 音频 RMS={rms:.6f} 过低，跳过")
                return []

            duration = len(samples_resampled) / sample_rate_out
            segment_len = 150
            overlap = 10

            if duration <= 180:
                note_list, pedal_list = self._transcribe_segment(samples_resampled, sample_rate_out)
            else:
                print(f"[Piano Transcription] 长音频 {duration:.1f}s，分段处理")
                note_list = []
                pedal_list = []
                start = 0.0
                seg_idx = 0
                
                total_segments = int((duration + segment_len - overlap - 1) / (segment_len - overlap)) + 1
                
                while start < duration:
                    end = min(start + segment_len, duration)
                    s0 = int(start * sample_rate_out)
                    s1 = int(end * sample_rate_out)
                    seg_audio = samples_resampled[s0:s1]

                    print(f"[Piano Transcription] Segment {seg_idx} / {total_segments - 1}")
                    
                    if progress_callback:
                        progress_callback(seg_idx, total_segments)

                    seg_notes, seg_pedals = self._transcribe_segment(seg_audio, sample_rate_out)

                    offset = start
                    for n in seg_notes:
                        n["onset_time"] = float(n.get("onset_time", 0)) + offset
                        n["offset_time"] = float(n.get("offset_time", 0)) + offset
                    for p in seg_pedals:
                        for k_on, k_off in [("onset_time", "offset_time"), ("start_time", "end_time"), ("on_time", "off_time")]:
                            if k_on in p:
                                p[k_on] = float(p[k_on]) + offset
                            if k_off in p:
                                p[k_off] = float(p[k_off]) + offset

                    if seg_idx > 0:
                        overlap_start = start
                        note_list = [
                            n for n in note_list
                            if float(n.get("offset_time", n.get("end_time", 0))) <= overlap_start
                        ]
                        pedal_list = [
                            p for p in pedal_list
                            if float(p.get("offset_time", p.get("end_time", p.get("off_time", 0)))) <= overlap_start
                        ]

                    note_list.extend(seg_notes)
                    pedal_list.extend(seg_pedals)

                    start += segment_len - overlap
                    seg_idx += 1

            events = []
            for note_info in note_list:
                vel = int(np.clip(note_info.get("velocity", 64), 1, 127))
                events.append(NoteEvent(
                    note=int(note_info["midi_note"]),
                    start_s=float(note_info["onset_time"]),
                    end_s=float(note_info["offset_time"]),
                    velocity=vel,
                    confidence=1.0,
                ))

            events = apply_pedal_correction(events, pedal_list, self._pedal_config)
            events.sort(key=lambda e: (e.start_s, e.note))
            return events

    # Try to actually instantiate to be 100% sure!
    try:
        return _PianoTranscriptionTranscriber()
    except Exception:
        return None


def try_basic_pitch_transcriber() -> Transcriber | None:
    """Try to load BasicPitch model"""
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except Exception:
        return None

    class _BasicPitchTranscriber:
        name = "Basic Pitch"

        def __init__(self, onset_threshold: float = 0.35, frame_threshold: float = 0.20):
            from pathlib import Path
            tf_model_path = Path(ICASSP_2022_MODEL_PATH)
            onnx_model_path = tf_model_path.parent / "nmp.onnx"
            if onnx_model_path.exists():
                self._model_path = str(onnx_model_path)
            else:
                self._model_path = ICASSP_2022_MODEL_PATH
            self._onset_threshold = onset_threshold
            self._frame_threshold = frame_threshold

        def transcribe(self, samples: np.ndarray, sample_rate: int, progress_callback=None) -> list[NoteEvent]:
            import tempfile
            import soundfile as sf
            from pathlib import Path

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, mode="wb") as f:
                temp_wav = f.name

            try:
                sf.write(temp_wav, samples, sample_rate)
                model_output, midi_data, note_events = predict(
                    temp_wav,
                    self._model_path,
                    onset_threshold=self._onset_threshold,
                    frame_threshold=self._frame_threshold,
                    minimum_note_length=50,
                    minimum_frequency=27.5,
                    maximum_frequency=4186.0,
                )

                events = []
                for note in note_events:
                    if isinstance(note, tuple):
                        start_time, end_time, pitch, amplitude = note[0], note[1], note[2], note[3]
                    else:
                        start_time = note.start_time
                        end_time = note.end_time
                        pitch = note.pitch
                        amplitude = note.amplitude
                    raw = float(amplitude)
                    if raw >= 0.8:
                        velocity = int(100 + (raw - 0.8) / 0.2 * 27)
                    elif raw >= 0.4:
                        velocity = int(50 + (raw - 0.4) / 0.4 * 50)
                    else:
                        velocity = int(raw / 0.4 * 50)
                    velocity = int(np.clip(velocity, 1, 127))
                    events.append(NoteEvent(
                        note=int(pitch),
                        start_s=float(start_time),
                        end_s=float(end_time),
                        velocity=velocity,
                        confidence=1.0,
                    ))
                return events
            finally:
                try:
                    Path(temp_wav).unlink(missing_ok=True)
                except Exception:
                    pass

    return _BasicPitchTranscriber()


class EnsembleTranscriber:
    name = "Ensemble (PT + BP)"

    def __init__(self, pt, bp):
        self._pt = pt
        self._bp = bp

    def transcribe(self, samples: np.ndarray, sample_rate: int, progress_callback=None) -> list[NoteEvent]:
        pt_events = self._pt.transcribe(samples, sample_rate)
        bp_events = self._bp.transcribe(samples, sample_rate)

        PITCH_TOL = 0
        TIME_TOL = 0.08

        pt_lookup: dict[int, list[NoteEvent]] = {}
        for e in pt_events:
            pt_lookup.setdefault(e.note, []).append(e)

        added = []
        for bp_note in bp_events:
            pt_candidates = pt_lookup.get(bp_note.note, [])
            is_duplicate = any(
                abs(bp_note.start_s - pt.start_s) <= TIME_TOL
                for pt in pt_candidates
            )
            if not is_duplicate:
                added.append(bp_note)

        combined = pt_events + added
        combined.sort(key=lambda e: (e.start_s, e.note))
        return combined


def available_transcribers() -> list[Transcriber]:
    transcribers: list[Transcriber] = []

    pt = try_piano_transcription_transcriber()
    if pt is not None:
        transcribers.append(pt)

    bp = try_basic_pitch_transcriber()
    if bp is not None:
        transcribers.append(bp)

    if pt is not None and bp is not None:
        transcribers.append(EnsembleTranscriber(pt, bp))

    transcribers.extend([
        HarmonicSalienceTranscriber(),
        SpectralPeaksTranscriber(),
    ])

    return transcribers


def available_dsp_transcribers() -> list[Transcriber]:
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

    def transcribe(self, samples: np.ndarray, sample_rate: int, progress_callback=None) -> list[NoteEvent]:
        events = self._base.transcribe(samples, sample_rate)
        result = separate_voices(events, self._voice_config)
        all_notes: list[NoteEvent] = []
        for voice in result.voices:
            all_notes.extend(voice.notes)
        return sorted(all_notes, key=lambda n: (n.start_s, n.note))

    def separate(self, samples: np.ndarray, sample_rate: int) -> VoiceSeparationResult:
        events = self._base.transcribe(samples, sample_rate)
        return separate_voices(events, self._voice_config)


@dataclass(frozen=True)
class BatchJobItem:
    audio_path: str
    out_path: str


def batch_transcribe(
    items: list[BatchJobItem],
    engine_name: str,
    cfg: dict | None = None,
    progress_callback=None,
) -> list[str]:
    from audiomidi_app.audio import read_audio
    from audiomidi_app.midi import events_to_midi
    from audiomidi_app.postprocess import full_postprocess, PostProcessConfig, OnsetDetector

    transcribers = available_transcribers()
    transcriber = None
    for t in transcribers:
        if t.name == engine_name:
            transcriber = t
            break
    if transcriber is None:
        raise RuntimeError(f"未找到引擎: {engine_name}")

    if cfg and hasattr(transcriber, '_onset_threshold') and hasattr(transcriber, '_frame_threshold'):
        transcriber._onset_threshold = cfg.get("bp_onset_threshold", 0.35)
        transcriber._frame_threshold = cfg.get("bp_frame_threshold", 0.20)
    if cfg and hasattr(transcriber, '_bp') and hasattr(transcriber._bp, '_onset_threshold'):
        transcriber._bp._onset_threshold = cfg.get("bp_onset_threshold", 0.35)
        transcriber._bp._frame_threshold = cfg.get("bp_frame_threshold", 0.20)

    is_neural = engine_name in ("Piano Transcription (Neural)", "Basic Pitch", "Ensemble (PT + BP)")
    pp_config = PostProcessConfig(
        confidence_threshold=(cfg or {}).get("confidence_threshold", 0.2),
        enable_velocity_normalize=(cfg or {}).get("velocity_stretch", True),
    )

    results: list[str] = []
    total = len(items)
    for idx, item in enumerate(items):
        if progress_callback:
            progress_callback(idx, total, item.audio_path)

        audio = read_audio(
            item.audio_path,
            normalize=(cfg or {}).get("normalize", True),
            normalize_mode="rms" if is_neural else "peak",
            preemphasis=(cfg or {}).get("preemphasis", False) and not is_neural,
        )

        bpm = (cfg or {}).get("bpm", 120.0)
        if (cfg or {}).get("auto_bpm", False):
            bpm = detect_bpm(audio.samples, audio.sample_rate)

        events = transcriber.transcribe(audio.samples, audio.sample_rate)

        onset_detector = OnsetDetector(audio.sample_rate)
        onset_detector.detect(audio.samples)
        events = full_postprocess(
            events,
            samples=audio.samples,
            sample_rate=audio.sample_rate,
            bpm=bpm,
            onset_detector=onset_detector,
            config=pp_config,
            is_neural=is_neural,
        )

        Path(item.out_path).parent.mkdir(parents=True, exist_ok=True)
        mid = events_to_midi(events, bpm=bpm)
        mid.save(item.out_path)
        results.append(item.out_path)

    if progress_callback:
        progress_callback(total, total, "")

    return results
