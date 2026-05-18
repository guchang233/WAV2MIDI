from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from audiomidi_app.midi import NoteEvent


@dataclass(frozen=True)
class PostProcessConfig:
    min_note_gap_s: float = 0.05
    min_note_duration_s: float = 0.05
    onset_refinement_threshold_s: float = 0.01
    repeated_note_max_gap_s: float = 0.15
    
    enable_quantization: bool = False
    quantize_division: float = 0.25
    
    enable_velocity_smooth: bool = True
    velocity_savgol_window: int = 5
    velocity_savgol_polyorder: int = 2
    
    enable_harmonic_masking: bool = True
    harmonic_order: int = 6


def detect_repeated_notes_onset_aware(
    events: list[NoteEvent],
    onset_times: list[float] | None = None,
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    if config is None:
        config = PostProcessConfig()
    
    if not events:
        return events
    
    onset_set = set()
    if onset_times:
        onset_set = {round(t, 3) for t in onset_times}
    
    events = sorted(events, key=lambda e: (e.start_s, e.note))
    result: list[NoteEvent] = []
    
    by_note: dict[int, list[NoteEvent]] = {}
    for e in events:
        by_note.setdefault(e.note, []).append(e)
    
    for note_num, note_events in by_note.items():
        note_events.sort(key=lambda e: e.start_s)
        
        active: list[NoteEvent] = []
        
        for note in note_events:
            should_split = False
            
            has_clear_onset = False
            for onset_t in onset_set:
                if abs(onset_t - note.start_s) <= 0.03:
                    has_clear_onset = True
                    break
            
            if has_clear_onset:
                gap_to_prev = note.start_s - active[-1].end_s if active else float('inf')
                
                if gap_to_prev > config.repeated_note_max_gap_s:
                    should_split = True
                elif gap_to_prev > config.min_note_gap_s:
                    should_split = True
                elif gap_to_prev <= config.min_note_gap_s and gap_to_prev > 0:
                    should_split = False
                else:
                    should_split = False
            
            for active_note in active:
                gap = note.start_s - active_note.end_s
                if gap < 0:
                    overlap = active_note.end_s - note.start_s
                    if overlap > 0.015:
                        if has_clear_onset:
                            should_split = True
                        break
            
            active = [n for n in active if n.end_s > note.start_s - 0.001]
            
            if should_split and has_clear_onset:
                result.append(note)
                active.append(note)
            else:
                if active:
                    merged_end = max(active[-1].end_s, note.end_s)
                    merged_velocity = max(active[-1].velocity, note.velocity)
                    
                    if note.start_s > active[-1].start_s:
                        active[-1] = NoteEvent(
                            note=note.note,
                            start_s=active[-1].start_s,
                            end_s=merged_end,
                            velocity=merged_velocity,
                        )
                else:
                    result.append(note)
                    active.append(note)
        
        for active_note in active:
            if active_note not in result:
                result.append(active_note)
    
    result.sort(key=lambda e: (e.start_s, e.note))
    return result


def refine_onsets_by_pitch(
    events: list[NoteEvent],
    samples: np.ndarray,
    sample_rate: int,
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    if config is None:
        config = PostProcessConfig()
    
    if not events:
        return events
    
    try:
        import librosa
    except ImportError:
        return events
    
    refined_events: list[NoteEvent] = []
    
    for event in events:
        onset_time = event.start_s
        pitch_hz = 440.0 * (2.0 ** ((event.note - 69) / 12.0))
        
        fmin = pitch_hz * 0.9
        fmax = pitch_hz * 1.1
        
        onset_frames = librosa.onset.onset_detect(
            y=samples,
            sr=sample_rate,
            hop_length=256,
            backtrack=True,
            units="time",
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.07,
            wait=10,
        )
        
        nearby_onsets = [t for t in onset_frames if abs(t - onset_time) <= 0.08]
        
        if nearby_onsets:
            best_onset = min(nearby_onsets, key=lambda t: abs(t - onset_time))
            
            if abs(best_onset - onset_time) <= config.onset_refinement_threshold_s:
                onset_time = best_onset
        
        refined_events.append(NoteEvent(
            note=event.note,
            start_s=onset_time,
            end_s=event.end_s,
            velocity=event.velocity,
        ))
    
    return refined_events


def smooth_velocities_savgol(
    events: list[NoteEvent],
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    if config is None:
        config = PostProcessConfig()
    
    if not events or not config.enable_velocity_smooth:
        return events
    
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return events
    
    events = sorted(events, key=lambda e: e.start_s)
    
    by_note: dict[int, list[tuple[int, NoteEvent]]] = {}
    for i, e in enumerate(events):
        by_note.setdefault(e.note, []).append((i, e))
    
    result = list(events)
    
    for note_num, indexed_events in by_note.items():
        if len(indexed_events) < 3:
            continue
        
        velocities = [e.velocity for _, e in indexed_events]
        times = [e.start_s for _, e in indexed_events]
        
        window = min(config.velocity_savgol_window, len(velocities) if len(velocities) % 2 == 1 else len(velocities) - 1)
        if window < 3:
            window = 3
        
        polyorder = min(config.velocity_savgol_polyorder, window - 1)
        
        try:
            smoothed = savgol_filter(velocities, window, polyorder)
            smoothed = np.clip(np.round(smoothed), 1, 127).astype(int)
            
            for idx, (orig_idx, event) in enumerate(indexed_events):
                result[orig_idx] = NoteEvent(
                    note=event.note,
                    start_s=event.start_s,
                    end_s=event.end_s,
                    velocity=int(smoothed[idx]),
                )
        except Exception:
            pass
    
    return result


def apply_harmonic_masking(
    events: list[NoteEvent],
    samples: np.ndarray | None = None,
    sample_rate: int = 44100,
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    if config is None:
        config = PostProcessConfig()
    
    if not events or not config.enable_harmonic_masking:
        return events
    
    events = sorted(events, key=lambda e: e.start_s)
    
    filtered: list[NoteEvent] = []
    active_notes: dict[int, tuple[float, float]] = {}
    
    for event in events:
        should_include = True
        
        note_hz = 440.0 * (2.0 ** ((event.note - 69) / 12.0))
        
        for active_note, (start, end) in list(active_notes.items()):
            if end < event.start_s - 0.01:
                continue
            
            active_hz = 440.0 * (2.0 ** ((active_note - 69) / 12.0))
            
            for h in range(2, config.harmonic_order + 1):
                harmonic_hz = active_hz * h
                
                if abs(harmonic_hz - note_hz) < note_hz * 0.05:
                    is_harmonic = True
                    
                    for other_note, (other_start, other_end) in active_notes.items():
                        if other_note == active_note:
                            continue
                        if other_end < event.start_s - 0.01:
                            continue
                        
                        other_hz = 440.0 * (2.0 ** ((other_note - 69) / 12.0))
                        
                        if abs(other_hz - harmonic_hz) < harmonic_hz * 0.02:
                            is_harmonic = False
                            break
                    
                    if is_harmonic:
                        should_include = False
                        break
            
            if not should_include:
                break
        
        active_notes = {
            n: (s, e) for n, (s, e) in active_notes.items()
            if e > event.start_s - 0.01
        }
        
        if should_include:
            filtered.append(event)
            active_notes[event.note] = (event.start_s, event.end_s)
    
    return filtered


def quantize_onsets_gentle(
    events: list[NoteEvent],
    bpm: float = 120.0,
    division: float = 0.25,
    threshold: float = 0.15,
) -> list[NoteEvent]:
    if not events or division <= 0 or threshold <= 0:
        return events
    
    beat_duration = 60.0 / bpm
    grid_duration = beat_duration * division
    
    quantized: list[NoteEvent] = []
    
    for event in events:
        nearest_grid = round(event.start_s / grid_duration) * grid_duration
        
        deviation = abs(event.start_s - nearest_grid) / grid_duration
        
        if deviation <= threshold:
            start_s = nearest_grid
        else:
            start_s = event.start_s
        
        quantized.append(NoteEvent(
            note=event.note,
            start_s=start_s,
            end_s=event.end_s,
            velocity=event.velocity,
        ))
    
    return quantized


def merge_overlaps_onset_aware(
    events: list[NoteEvent],
    onset_times: list[float] | None = None,
) -> list[NoteEvent]:
    if not events:
        return events
    
    onset_set = set()
    if onset_times:
        onset_set = {round(t, 3) for t in onset_times}
    
    events = sorted(events, key=lambda e: (e.note, e.start_s))
    
    merged: list[NoteEvent] = []
    current: Optional[NoteEvent] = None
    
    for event in events:
        if current is None:
            current = event
            continue
        
        if event.note != current.note:
            merged.append(current)
            current = event
            continue
        
        gap = event.start_s - current.end_s
        
        if gap < 0:
            overlap = current.end_s - event.start_s
            if overlap > 0.01:
                has_separate_onset = False
                for onset_t in onset_set:
                    if abs(onset_t - event.start_s) <= 0.025:
                        has_separate_onset = True
                        break
                
                if has_separate_onset:
                    merged.append(current)
                    current = event
                else:
                    current = NoteEvent(
                        note=current.note,
                        start_s=current.start_s,
                        end_s=max(current.end_s, event.end_s),
                        velocity=max(current.velocity, event.velocity),
                    )
            else:
                current = NoteEvent(
                    note=current.note,
                    start_s=current.start_s,
                    end_s=max(current.end_s, event.end_s),
                    velocity=max(current.velocity, event.velocity),
                )
        elif gap <= 0.03:
            current = NoteEvent(
                note=current.note,
                start_s=current.start_s,
                end_s=max(current.end_s, event.end_s),
                velocity=max(current.velocity, event.velocity),
            )
        else:
            merged.append(current)
            current = event
    
    if current is not None:
        merged.append(current)
    
    merged.sort(key=lambda e: (e.start_s, e.note))
    return merged


def full_postprocess(
    events: list[NoteEvent],
    samples: np.ndarray | None = None,
    sample_rate: int = 44100,
    bpm: float = 120.0,
    onset_times: list[float] | None = None,
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    if config is None:
        config = PostProcessConfig()
    
    if not events:
        return events
    
    events = apply_harmonic_masking(events, samples, sample_rate, config)
    
    events = detect_repeated_notes_onset_aware(events, onset_times, config)
    
    if samples is not None:
        events = refine_onsets_by_pitch(events, samples, sample_rate, config)
    
    events = [
        NoteEvent(
            note=e.note,
            start_s=max(0, e.start_s),
            end_s=max(e.start_s + config.min_note_duration_s, e.end_s),
            velocity=max(1, min(127, e.velocity)),
        )
        for e in events
    ]
    
    events = smooth_velocities_savgol(events, config)
    
    if config.enable_quantization:
        events = quantize_onsets_gentle(events, bpm, config.quantize_division, threshold=0.15)
    
    events = merge_overlaps_onset_aware(events, onset_times)
    
    events.sort(key=lambda e: (e.start_s, e.note))
    
    return events
