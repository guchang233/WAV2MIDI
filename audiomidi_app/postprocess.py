from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import bisect

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
    
    enable_harmonic_confidence: bool = True
    harmonic_confidence_factor: float = 0.7
    harmonic_order: int = 6


class OnsetDetector:
    """统一 onset 检测器"""
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 256):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self._onset_times: list[float] = []
        self._onset_set: set[float] = set()
    
    def detect(self, samples: np.ndarray) -> list[float]:
        """检测全局 onsets"""
        try:
            import librosa
            self._onset_times = librosa.onset.onset_detect(
                y=samples,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                backtrack=True,
                units="time",
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=5,
                delta=0.07,
                wait=10,
            ).tolist()
            
            self._onset_set = {round(t, 3) for t in self._onset_times}
            return self._onset_times
        except Exception:
            self._onset_times = []
            self._onset_set = set()
            return []
    
    def find_nearby_onset(self, time: float, threshold: float = 0.08) -> Optional[float]:
        """用 bisect 找最近的 onset"""
        if not self._onset_times:
            return None
        
        idx = bisect.bisect_left(self._onset_times, time)
        
        candidates = []
        if idx < len(self._onset_times):
            candidates.append(self._onset_times[idx])
        if idx > 0:
            candidates.append(self._onset_times[idx - 1])
        
        best = None
        min_dist = float('inf')
        
        for candidate in candidates:
            dist = abs(candidate - time)
            if dist <= threshold and dist < min_dist:
                best = candidate
                min_dist = dist
        
        return best
    
    def has_onset_near(self, time: float, threshold: float = 0.03) -> bool:
        """检查附近是否有 onset"""
        return self.find_nearby_onset(time, threshold=threshold) is not None


def apply_harmonic_confidence_adjustment(
    events: list[NoteEvent],
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    """不删除音符，只调整 harmonic candidate 的 confidence"""
    if config is None:
        config = PostProcessConfig()
    
    if not events or not config.enable_harmonic_confidence:
        return events
    
    events = sorted(events, key=lambda e: (e.start_s, -e.note))
    
    adjusted = []
    for event in events:
        e = NoteEvent(
            note=event.note,
            start_s=event.start_s,
            end_s=event.end_s,
            velocity=event.velocity,
            confidence=event.confidence,
        )
        
        for other in reversed(adjusted):
            if other.end_s < event.start_s - 0.5:
                break

            other_hz = 440.0 * (2.0 ** ((other.note - 69) / 12.0))
            event_hz = 440.0 * (2.0 ** ((event.note - 69) / 12.0))

            for h in range(2, config.harmonic_order + 1):
                harmonic_hz = other_hz * h

                if abs(harmonic_hz - event_hz) < event_hz * 0.05:
                    e.confidence *= config.harmonic_confidence_factor
                    break

            if e.confidence < 0.4:
                break
        
        adjusted.append(e)
    
    return adjusted


def refine_onsets_from_global_detector(
    events: list[NoteEvent],
    onset_detector: OnsetDetector,
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    """从全局 onset detector 中精调，而不是 per-note 检测"""
    if config is None:
        config = PostProcessConfig()
    
    if not events:
        return events
    
    refined_events: list[NoteEvent] = []
    
    for event in events:
        onset_time = event.start_s
        
        nearby_onset = onset_detector.find_nearby_onset(
            onset_time,
            threshold=0.04,
        )

        if nearby_onset is not None:
            onset_time = nearby_onset
        
        refined_events.append(NoteEvent(
            note=event.note,
            start_s=onset_time,
            end_s=event.end_s,
            velocity=event.velocity,
            confidence=event.confidence,
        ))
    
    return refined_events


def detect_repeated_notes_simple(
    events: list[NoteEvent],
    onset_detector: OnsetDetector,
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    """简化核心：new onset + existing sustain = repeated note"""
    if config is None:
        config = PostProcessConfig()
    
    if not events:
        return events
    
    events = sorted(events, key=lambda e: (e.note, e.start_s))
    result: list[NoteEvent] = []
    
    for note_num in sorted({e.note for e in events}):
        note_events = [e for e in events if e.note == note_num]
        note_events.sort(key=lambda e: e.start_s)
        
        current: Optional[NoteEvent] = None
        
        for event in note_events:
            has_clear_onset = onset_detector.has_onset_near(
                event.start_s, 
                threshold=0.025
            )
            
            if current is None:
                current = event
                continue
            
            gap = event.start_s - current.end_s
            
            if has_clear_onset:
                result.append(current)
                current = event
            else:
                if gap <= config.min_note_gap_s:
                    current = NoteEvent(
                        note=current.note,
                        start_s=current.start_s,
                        end_s=max(current.end_s, event.end_s),
                        velocity=max(current.velocity, event.velocity),
                        confidence=max(current.confidence, event.confidence),
                    )
                else:
                    result.append(current)
                    current = event
        
        if current is not None:
            result.append(current)
    
    result.sort(key=lambda e: (e.start_s, e.note))
    return result


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
                    confidence=event.confidence,
                )
        except Exception:
            pass
    
    return result


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
            confidence=event.confidence,
        ))
    
    return quantized


def merge_overlaps_onset_confidence_aware(
    events: list[NoteEvent],
    onset_detector: OnsetDetector,
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    """基于 confidence 和 onset 来 merge，而不是只看 gap"""
    if config is None:
        config = PostProcessConfig()
    
    if not events:
        return events
    
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
        
        has_new_onset = onset_detector.has_onset_near(event.start_s, threshold=0.025)
        
        if has_new_onset:
            merged.append(current)
            current = event
        else:
            if gap < 0:
                current = NoteEvent(
                    note=current.note,
                    start_s=current.start_s,
                    end_s=max(current.end_s, event.end_s),
                    velocity=max(current.velocity, event.velocity),
                    confidence=max(current.confidence, event.confidence),
                )
            elif gap <= 0.05:
                current = NoteEvent(
                    note=current.note,
                    start_s=current.start_s,
                    end_s=max(current.end_s, event.end_s),
                    velocity=max(current.velocity, event.velocity),
                    confidence=max(current.confidence, event.confidence),
                )
            else:
                merged.append(current)
                current = event
    
    if current is not None:
        merged.append(current)
    
    merged.sort(key=lambda e: (e.start_s, e.note))
    return merged


def normalize_velocity_percentile(
    events: list[NoteEvent],
    percentile_min: float = 5.0,
    percentile_max: float = 95.0,
) -> list[NoteEvent]:
    """基于 percentile 归一化 velocity，而不是固定 dB 映射"""
    if not events:
        return events
    
    velocities = np.array([e.velocity for e in events])
    
    v_min = np.percentile(velocities, percentile_min)
    v_max = np.percentile(velocities, percentile_max)
    
    if v_max - v_min < 1.0:
        return events
    
    normalized: list[NoteEvent] = []
    
    for event in events:
        norm = (event.velocity - v_min) / (v_max - v_min)
        norm = np.clip(norm, 0.0, 1.0)
        
        new_vel = int(round(norm * 116)) + 10
        new_vel = int(np.clip(new_vel, 1, 127))
        
        normalized.append(NoteEvent(
            note=event.note,
            start_s=event.start_s,
            end_s=event.end_s,
            velocity=new_vel,
            confidence=event.confidence,
        ))
    
    return normalized


def full_postprocess(
    events: list[NoteEvent],
    samples: np.ndarray | None = None,
    sample_rate: int = 44100,
    bpm: float = 120.0,
    onset_detector: OnsetDetector | None = None,
    config: PostProcessConfig | None = None,
    is_neural: bool = False,
) -> list[NoteEvent]:
    if config is None:
        config = PostProcessConfig()

    if not events:
        return events

    if is_neural:
        from audiomidi_app.transcribe import merge_overlaps
        events = merge_overlaps(events)
        events = [
            NoteEvent(
                note=e.note,
                start_s=max(0, e.start_s),
                end_s=max(e.start_s + 0.03, e.end_s),
                velocity=max(1, min(127, e.velocity)),
                confidence=e.confidence,
            )
            for e in events
        ]
        if config.enable_quantization:
            events = quantize_onsets_gentle(events, bpm, config.quantize_division, threshold=0.15)
        events.sort(key=lambda e: (e.start_s, e.note))
        return events

    if onset_detector is None and samples is not None:
        onset_detector = OnsetDetector(sample_rate)
        onset_detector.detect(samples)

    if onset_detector is None:
        onset_detector = OnsetDetector(sample_rate)

    events = refine_onsets_from_global_detector(events, onset_detector, config)

    events = detect_repeated_notes_simple(events, onset_detector, config)

    events = merge_overlaps_onset_confidence_aware(events, onset_detector, config)

    events = apply_harmonic_confidence_adjustment(events, config)

    events = [
        NoteEvent(
            note=e.note,
            start_s=max(0, e.start_s),
            end_s=max(e.start_s + config.min_note_duration_s, e.end_s),
            velocity=max(1, min(127, e.velocity)),
            confidence=e.confidence,
        )
        for e in events
    ]

    events = normalize_velocity_percentile(events)

    events = smooth_velocities_savgol(events, config)

    confidence_threshold = 0.35
    events = [e for e in events if e.confidence >= confidence_threshold]

    if config.enable_quantization:
        events = quantize_onsets_gentle(events, bpm, config.quantize_division, threshold=0.15)

    events.sort(key=lambda e: (e.start_s, e.note))

    return events
