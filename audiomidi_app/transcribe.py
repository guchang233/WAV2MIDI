from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, resample_poly, stft

from audiomidi_app.midi import NoteEvent


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
    mag: np.ndarray,
    f: np.ndarray,
    harmonic_weight: float = 1.5,
    n_harmonics: int = 8,
) -> np.ndarray:
    sal = mag.copy()
    band = (f >= 50) & (f <= 4200)
    f_band = f[band]
    mag_band = mag[band, :]
    
    for i, (freq, mag_col) in enumerate(zip(f_band, mag_band.T)):
        if freq < 50:
            continue
        for h in range(2, n_harmonics + 1):
            harmonic_freq = freq * h
            idx = np.argmin(np.abs(f_band - harmonic_freq))
            if abs(f_band[idx] - harmonic_freq) < harmonic_freq * 0.05:
                sal[band, i] += mag_band[idx, i] * (harmonic_weight / h)
    
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


def separate_audio(
    audio_path: str,
    model: str = "htdemucs",
    device: str = "cpu",
) -> dict[str, str] | None:
    try:
        from demucs.pretrained import get_model
        from demucs.separate import Separator
        from pathlib import Path
        import os
    except ImportError:
        return None
    
    try:
        model_instance = get_model(model)
        stem_paths = {}
        
        separator_instance = Separator(model_instance, device=device)
        wavs = separator_instance.separate(audio_path)
        
        for stem_name, stem_wav in wavs.items():
            if stem_wav is not None and len(stem_wav) > 0:
                temp_path = Path(tempfile.gettempdir()) / f"demucs_{stem_name}.wav"
                import soundfile as sf
                sf.write(str(temp_path), stem_wav, 44100)
                stem_paths[stem_name] = str(temp_path)
        
        return stem_paths if stem_paths else None
    except Exception:
        return None


class SpectralPeaksTranscriber:
    name = "Spectral Peaks"

    def __init__(self, config: SpectralPeaksConfig | None = None) -> None:
        self._cfg = config or SpectralPeaksConfig()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
        x = samples.astype(np.float32, copy=False)
        if x.ndim != 1:
            x = x.mean(axis=-1)
        
        onset_frames = np.array([], dtype=int)
        if self._cfg.use_onset_detection:
            onset_frames = detect_onsets(x, sample_rate, self._cfg.hop_length)
        
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
            is_onset = frame_i in onset_frames
            
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
                    )
                )
        
        events.sort(key=lambda e: (e.start_s, e.note))
        return merge_overlaps(events)


class HarmonicSalienceTranscriber:
    name = "Harmonic Salience"

    def __init__(self, config: HarmonicSalienceConfig | None = None) -> None:
        self._cfg = config or HarmonicSalienceConfig()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
        x = samples.astype(np.float32, copy=False)
        if x.ndim != 1:
            x = x.mean(axis=-1)
        
        onset_frames = np.array([], dtype=int)
        if self._cfg.use_onset_detection:
            onset_frames = detect_onsets(x, sample_rate, self._cfg.hop_length)
        
        if self._cfg.use_cqt:
            try:
                import librosa
                cqt = librosa.cqt(
                    x,
                    sr=sample_rate,
                    hop_length=self._cfg.hop_length,
                    fmin=librosa.note_to_hz('A0'),
                    n_bins=96,
                    bins_per_octave=12
                )
                mag = np.abs(cqt) + 1e-12
                f = librosa.cqt_frequencies(96, fmin=librosa.note_to_hz('A0'), bins_per_octave=12)
            except ImportError:
                mag, f = self._compute_stft_mag(x, sample_rate)
        else:
            mag, f = self._compute_stft_mag(x, sample_rate)
        
        mag_db = 20.0 * np.log10(mag)
        
        if self._cfg.use_temporal_smoothing:
            mag_db = median_filter(mag_db, size=(1, 3))
        
        salience = compute_harmonic_salience(mag_db, f, self._cfg.harmonic_weight)
        
        band = (f >= self._cfg.fmin_hz) & (f <= self._cfg.fmax_hz)
        f_band = f[band]
        salience = salience[band, :]
        mag_band = mag[band, :]
        
        dt = float(self._cfg.hop_length) / float(sample_rate)
        
        active: dict[int, tuple[float, float, float]] = {}
        events: list[NoteEvent] = []
        
        for frame_i in range(salience.shape[1]):
            is_onset = frame_i in onset_frames
            
            frame_sal = salience[:, frame_i]
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
                                )
                            )
                        to_close.append(n)
            
            for n in to_close:
                active.pop(n, None)
            
            for n, amp in present.items():
                if n not in active:
                    active[n] = (now_s, now_s, amp)
        
        end_s = salience.shape[1] * dt
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


def hz_to_midi(freq_hz: np.ndarray) -> np.ndarray:
    midi = 69.0 + 12.0 * np.log2(freq_hz / 440.0)
    return np.rint(midi).astype(int)


def amp_to_velocity(amp: float, mode: str = "linear") -> int:
    if mode == "piano":
        db = 20.0 * np.log10(max(1e-9, amp))
        normalized = np.clip((db + 70.0) / 70.0, 0.0, 1.0)
        v = int(normalized ** (1/3) * 126) + 1
    else:
        db = 20.0 * np.log10(max(1e-9, amp))
        v = int(np.clip((db + 60.0) * 2.2, 1.0, 127.0))
    return max(1, min(127, int(v)))


def transcribe_drums(samples: np.ndarray, sr: int) -> list[NoteEvent]:
    try:
        import librosa
    except ImportError:
        return []
    
    bands = {
        36: (20, 100),
        38: (100, 500),
        42: (5000, 16000),
    }
    
    events = []
    onset_times = set()
    
    for gm_note, (flo, fhi) in bands.items():
        onset_frames = librosa.onset.onset_detect(
            y=samples,
            sr=sr,
            hop_length=256,
            backtrack=True,
            units="time"
        )
        for t in onset_frames:
            if abs(t) not in onset_times:
                onset_times.add(abs(t))
                events.append(NoteEvent(
                    note=gm_note,
                    start_s=float(t) if t >= 0 else 0.0,
                    end_s=float(t) + 0.05 if t >= 0 else 0.05,
                    velocity=90
                ))
    
    events.sort(key=lambda e: e.start_s)
    return events


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
                )
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)

    merged.sort(key=lambda e: (e.start_s, e.note))
    return merged


def available_transcribers() -> list[Transcriber]:
    transcribers: list[Transcriber] = [
        HarmonicSalienceTranscriber(),
        SpectralPeaksTranscriber(),
    ]
    
    pt = try_piano_transcription_transcriber()
    if pt is not None:
        transcribers.insert(0, pt)
    
    bp = try_basic_pitch_transcriber()
    if bp is not None:
        transcribers.append(bp)
    
    return transcribers


def try_piano_transcription_transcriber() -> Transcriber | None:
    try:
        from piano_transcription_inference import PianoTranscription
    except Exception:
        return None

    class _PianoTranscriptionTranscriber:
        name = "Piano Transcription"

        def transcribe(self, samples: np.ndarray, sample_rate_in: int) -> list[NoteEvent]:
            import tempfile
            import soundfile as sf
            from pathlib import Path
            from piano_transcription_inference import PianoTranscription, sample_rate as PT_SR

            if sample_rate_in != PT_SR:
                g = np.gcd(sample_rate_in, PT_SR)
                samples = resample_poly(
                    samples, 
                    PT_SR // g, 
                    sample_rate_in // g
                ).astype(np.float32)
                sample_rate_out = PT_SR
            else:
                sample_rate_out = sample_rate_in
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, mode="wb") as f:
                temp_wav = f.name
            
            try:
                sf.write(temp_wav, samples, sample_rate_out)
                transcriptor = PianoTranscription(device="cpu", duration=None)
                result = transcriptor.transcribe(temp_wav, midi_path=None)
                
                events = []
                for note_info in result["notes"]:
                    events.append(NoteEvent(
                        note=int(note_info["midi_note"]),
                        start_s=float(note_info["onset_time"]),
                        end_s=float(note_info["offset_time"]),
                        velocity=int(np.clip(note_info["velocity"] * 127, 1, 127)),
                    ))
                return events
            finally:
                try:
                    Path(temp_wav).unlink(missing_ok=True)
                except Exception:
                    pass

    return _PianoTranscriptionTranscriber()


def try_basic_pitch_transcriber() -> Transcriber | None:
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except Exception:
        return None

    class _BasicPitchTranscriber:
        name = "Basic Pitch"

        def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
            import tempfile
            import soundfile as sf
            from pathlib import Path

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, mode="wb") as f:
                temp_wav = f.name
            
            try:
                sf.write(temp_wav, samples, sample_rate)
                model_output, midi_data, note_events = predict(temp_wav, ICASSP_2022_MODEL_PATH)
                
                events = []
                for note in note_events:
                    events.append(NoteEvent(
                        note=int(note.pitch),
                        start_s=float(note.start_time),
                        end_s=float(note.end_time),
                        velocity=int(min(127, max(1, note.amplitude * 127))),
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
                model_or_model_path=ICASSP_2022_MODEL_PATH,
            )
            midi_path = out / (Path(in_path).stem + ".mid")
            if not midi_path.exists():
                candidates = list(out.glob("*.mid"))
                if not candidates:
                    raise RuntimeError("Basic Pitch 未生成MIDI输出")
                midi_path = candidates[0]
            return os.fspath(midi_path)

    return _BasicPitchTranscriber()
