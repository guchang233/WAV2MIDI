"""
Instrument Routing Module
根据分离的 stem 路由到相应的 transcription 模型
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Optional
import numpy as np

from audiomidi_app.midi import NoteEvent
from audiomidi_app.pipeline.separation import SeparatedStems


class StemType(Enum):
    """Stem 类型枚举"""
    PIANO = "piano"
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    GUITAR = "guitar"
    OTHER = "other"
    MIXED = "mixed"


@dataclass(frozen=True)
class RoutingConfig:
    """路由配置"""
    enable_piano: bool = True
    enable_vocals: bool = False
    enable_drums: bool = True
    enable_bass: bool = True
    enable_guitar: bool = False
    enable_other: bool = True
    
    fallback_to_mixed: bool = True
    priority_order: list[StemType] = None
    
    def __post_init__(self):
        if self.priority_order is None:
            object.__setattr__(
                self,
                'priority_order',
                [
                    StemType.PIANO,
                    StemType.VOCALS,
                    StemType.DRUMS,
                    StemType.BASS,
                    StemType.GUITAR,
                    StemType.OTHER,
                ]
            )


class TranscriptionEngine(Protocol):
    """Transcription 引擎协议"""
    name: str
    
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]: ...
    
    def transcribe_with_confidence(
        self, samples: np.ndarray, sample_rate: int
    ) -> tuple[list[NoteEvent], list[float]]: ...


class StemRouter:
    """根据不同 stem 路由到对应的 transcription 引擎"""
    
    def __init__(self, config: RoutingConfig | None = None):
        self._cfg = config or RoutingConfig()
        self._engines: dict[StemType, TranscriptionEngine] = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        """初始化所有引擎"""
        from audiomidi_app.pipeline.transcription import (
            PianoTranscriptionEngine,
            DrumsTranscriptionEngine,
            BasicPitchTranscriptionEngine,
            MT3TranscriptionEngine,
        )
        
        if self._cfg.enable_piano:
            self._engines[StemType.PIANO] = PianoTranscriptionEngine()
        
        if self._cfg.enable_vocals:
            self._engines[StemType.VOCALS] = BasicPitchTranscriptionEngine()
        
        if self._cfg.enable_drums:
            self._engines[StemType.DRUMS] = DrumsTranscriptionEngine()
        
        if self._cfg.enable_bass:
            self._engines[StemType.BASS] = BasicPitchTranscriptionEngine()
        
        if self._cfg.enable_guitar:
            self._engines[StemType.GUITAR] = MT3TranscriptionEngine()
        
        if self._cfg.enable_other:
            self._engines[StemType.OTHER] = MT3TranscriptionEngine()
    
    def route_and_transcribe(
        self, stems: SeparatedStems
    ) -> tuple[list[NoteEvent], dict[StemType, list[NoteEvent]]]:
        """
        路由并转录所有 stem
        
        Returns:
            (merged_notes, notes_by_stem)
        """
        all_notes: list[NoteEvent] = []
        notes_by_stem: dict[StemType, list[NoteEvent]] = {}
        
        for stem_type in self._cfg.priority_order:
            notes = self._transcribe_stem(stems, stem_type)
            if notes:
                notes_by_stem[stem_type] = notes
                all_notes.extend(notes)
        
        all_notes = self._merge_notes(all_notes)
        return all_notes, notes_by_stem
    
    def _transcribe_stem(
        self, stems: SeparatedStems, stem_type: StemType
    ) -> list[NoteEvent]:
        """转录单个 stem"""
        engine = self._engines.get(stem_type)
        if engine is None:
            return []
        
        samples = self._get_stem_samples(stems, stem_type)
        if samples is None:
            return []
        
        try:
            notes = engine.transcribe(samples, stems.sample_rate)
            return notes
        except Exception as e:
            print(f"Transcription failed for {stem_type}: {e}")
            return []
    
    def _get_stem_samples(
        self, stems: SeparatedStems, stem_type: StemType
    ) -> Optional[np.ndarray]:
        """获取对应 stem 的音频"""
        stem_map = {
            StemType.PIANO: stems.other,
            StemType.VOCALS: stems.vocals,
            StemType.DRUMS: stems.drums,
            StemType.BASS: stems.bass,
            StemType.GUITAR: stems.other,
            StemType.OTHER: stems.other,
        }
        
        samples = stem_map.get(stem_type)
        
        if samples is None and self._cfg.fallback_to_mixed:
            return stems.mixture
        
        return samples
    
    def _merge_notes(self, notes: list[NoteEvent]) -> list[NoteEvent]:
        """合并所有音符"""
        if not notes:
            return []
        
        notes.sort(key=lambda e: (e.start_s, e.note))
        
        merged: list[NoteEvent] = []
        current: Optional[NoteEvent] = None
        
        for note in notes:
            if current is None:
                current = note
                continue
            
            gap = note.start_s - current.end_s
            same_note = note.note == current.note
            
            if same_note and gap <= 0.05:
                current = NoteEvent(
                    note=current.note,
                    start_s=current.start_s,
                    end_s=max(current.end_s, note.end_s),
                    velocity=max(current.velocity, note.velocity),
                    confidence=max(getattr(current, 'confidence', 1.0), 
                                 getattr(note, 'confidence', 1.0)),
                )
            else:
                merged.append(current)
                current = note
        
        if current is not None:
            merged.append(current)
        
        return merged
