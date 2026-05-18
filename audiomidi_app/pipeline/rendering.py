"""
MIDI Rendering Module
将处理后的音符渲染为 MIDI 文件
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from audiomidi_app.midi import NoteEvent, events_to_midi


@dataclass(frozen=True)
class RenderingConfig:
    """渲染配置"""
    bpm: float = 120.0
    ticks_per_beat: int = 480
    
    velocity_scaling: bool = True
    velocity_min: int = 20
    velocity_max: int = 120
    
    apply_quantization: bool = False
    quantize_division: float = 0.25
    quantize_threshold: float = 0.15
    
    filter_confidence_threshold: float = 0.2
    
    sort_by_time: bool = True
    merge_overlaps: bool = True


class MIDIRenderer:
    """MIDI 渲染器"""
    
    def __init__(self, config: RenderingConfig | None = None):
        self._cfg = config or RenderingConfig()
    
    def render(
        self,
        notes: list[NoteEvent],
        bpm: float | None = None
    ) -> 'mido.MidiFile':
        """
        将音符渲染为 MIDI 文件
        
        Args:
            notes: NoteEvent 列表
            bpm: 拍子速度，如果为 None 使用配置中的值
        
        Returns:
            mido.MidiFile
        """
        if not notes:
            return self._create_empty_midi()
        
        notes = self._preprocess(notes)
        
        if bpm is not None:
            self._cfg.bpm = bpm
        
        midi = events_to_midi(
            notes,
            bpm=self._cfg.bpm,
            ticks_per_beat=self._cfg.ticks_per_beat
        )
        
        return midi
    
    def _preprocess(self, notes: list[NoteEvent]) -> list[NoteEvent]:
        """预处理音符"""
        processed: list[NoteEvent] = []
        
        for note in notes:
            confidence = getattr(note, 'confidence', 1.0)
            
            if confidence < self._cfg.filter_confidence_threshold:
                continue
            
            velocity = note.velocity
            
            if self._cfg.velocity_scaling:
                velocity = self._scale_velocity(velocity)
            
            start_s = max(0.0, note.start_s)
            end_s = max(note.start_s + 0.01, note.end_s)
            
            if self._cfg.apply_quantization:
                start_s = self._quantize_time(start_s)
            
            processed.append(NoteEvent(
                note=note.note,
                start_s=start_s,
                end_s=end_s,
                velocity=velocity,
                confidence=confidence,
            ))
        
        if self._cfg.sort_by_time:
            processed.sort(key=lambda n: (n.start_s, n.note))
        
        if self._cfg.merge_overlaps:
            processed = self._merge_overlaps(processed)
        
        return processed
    
    def _scale_velocity(self, velocity: int) -> int:
        """缩放 velocity 到配置范围"""
        v = float(velocity)
        
        v_scaled = (
            (v - 1) / 126.0 * (self._cfg.velocity_max - self._cfg.velocity_min)
            + self._cfg.velocity_min
        )
        
        return max(1, min(127, int(round(v_scaled))))
    
    def _quantize_time(self, time: float) -> float:
        """量化时间到网格"""
        beat_duration = 60.0 / self._cfg.bpm
        grid_duration = beat_duration * self._cfg.quantize_division
        
        nearest_grid = round(time / grid_duration) * grid_duration
        
        deviation = abs(time - nearest_grid) / grid_duration
        
        if deviation <= self._cfg.quantize_threshold:
            return nearest_grid
        
        return time
    
    def _merge_overlaps(self, notes: list[NoteEvent]) -> list[NoteEvent]:
        """合并重叠的相同音符"""
        if not notes:
            return []
        
        merged: list[NoteEvent] = []
        current: Optional[NoteEvent] = None
        
        for note in notes:
            if current is None:
                current = note
                continue
            
            if note.note != current.note:
                merged.append(current)
                current = note
                continue
            
            if note.start_s <= current.end_s + 0.01:
                current = NoteEvent(
                    note=current.note,
                    start_s=current.start_s,
                    end_s=max(current.end_s, note.end_s),
                    velocity=max(current.velocity, note.velocity),
                    confidence=max(
                        getattr(current, 'confidence', 1.0),
                        getattr(note, 'confidence', 1.0)
                    ),
                )
            else:
                merged.append(current)
                current = note
        
        if current is not None:
            merged.append(current)
        
        return merged
    
    def _create_empty_midi(self) -> 'mido.MidiFile':
        """创建空的 MIDI 文件"""
        import mido
        mid = mido.MidiFile(ticks_per_beat=self._cfg.ticks_per_beat, type=1)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        tempo = mido.bpm2tempo(self._cfg.bpm)
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
        track.append(mido.MetaMessage("end_of_track", time=0))
        
        return mid
    
    def render_to_bytes(self, notes: list[NoteEvent]) -> bytes:
        """渲染为字节流"""
        midi = self.render(notes)
        return midi.serialize()


class StemAwareRenderer(MIDIRenderer):
    """Stem 感知的渲染器 - 可以为不同 stem 分配不同 track"""
    
    def __init__(self, config: RenderingConfig | None = None):
        super().__init__(config)
    
    def render_multi_track(
        self,
        notes_by_stem: dict[str, list[NoteEvent]],
        bpm: float | None = None
    ) -> 'mido.MidiFile':
        """
        渲染为多轨道 MIDI
        
        Args:
            notes_by_stem: {stem_name: notes}
            bpm: 拍子速度
        
        Returns:
            多轨道 MIDI 文件
        """
        import mido
        
        if bpm is not None:
            self._cfg.bpm = bpm
        
        mid = mido.MidiFile(ticks_per_beat=self._cfg.ticks_per_beat, type=1)
        
        tempo = mido.bpm2tempo(self._cfg.bpm)
        
        for stem_name, notes in notes_by_stem.items():
            track = mido.MidiTrack()
            mid.tracks.append(track)
            
            track.append(mido.MetaMessage("track_name", name=stem_name, time=0))
            track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
            
            notes = self._preprocess(notes)
            
            for note in notes:
                start_tick = int(mido.second2tick(
                    note.start_s,
                    self._cfg.ticks_per_beat,
                    tempo
                ))
                end_tick = int(mido.second2tick(
                    note.end_s,
                    self._cfg.ticks_per_beat,
                    tempo
                ))
                
                velocity = note.velocity if self._cfg.velocity_scaling else self._scale_velocity(note.velocity)
                
                track.append(mido.Message(
                    "note_on",
                    note=note.note,
                    velocity=velocity,
                    time=start_tick
                ))
                track.append(mido.Message(
                    "note_off",
                    note=note.note,
                    velocity=0,
                    time=end_tick
                ))
            
            track.append(mido.MetaMessage("end_of_track", time=0))
        
        return mid
