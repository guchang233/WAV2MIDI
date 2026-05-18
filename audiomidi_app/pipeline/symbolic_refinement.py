"""
Symbolic Graph Optimization Module
符号图优化 - 音乐语言级别的后处理
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from audiomidi_app.midi import NoteEvent


@dataclass(frozen=True)
class SymbolicRefinementConfig:
    """符号优化配置"""
    enable_harmony_check: bool = True
    enable_voice_leading: bool = True
    enable_rhythm_quantization: bool = True
    enable_confidence_weighting: bool = True
    enable_tempo_constraints: bool = True
    
    tempo: float = 120.0
    beat_division: float = 4.0
    
    min_note_gap: float = 0.05
    max_octave_error: float = 0.3


class NoteGraphNode:
    """音符图节点"""
    
    def __init__(self, note: NoteEvent, confidence: float = 1.0):
        self.note = note
        self.confidence = confidence
        self.pitch = note.note
        self.start_time = note.start_s
        self.end_time = note.end_s
        self.velocity = note.velocity
        
        self.duration = note.end_s - note.start_s
        self.pitch_class = note.note % 12
        self.octave = note.note // 12
        
        self.connections: list[NoteGraphNode] = []
        self.in_degree: int = 0
        self.out_degree: int = 0


class SymbolicRefiner:
    """符号级图优化器"""
    
    def __init__(self, config: SymbolicRefinementConfig | None = None):
        self._cfg = config or SymbolicRefinementConfig()
        self._graph: list[NoteGraphNode] = []
    
    def refine(self, notes: list[NoteEvent]) -> list[NoteEvent]:
        """完整符号优化流程"""
        if not notes:
            return notes
        
        nodes = self._build_graph(notes)
        
        nodes = self._harmony_check(nodes)
        
        nodes = self._voice_leading_optimization(nodes)
        
        nodes = self._tempo_constraint(nodes)
        
        nodes = self._confidence_weighted_cleanup(nodes)
        
        refined_notes = [node.note for node in nodes]
        return refined_notes
    
    def _build_graph(self, notes: list[NoteEvent]) -> list[NoteGraphNode]:
        """构建音符图"""
        nodes = []
        
        for note in notes:
            confidence = getattr(note, 'confidence', 1.0)
            node = NoteGraphNode(note, confidence)
            nodes.append(node)
        
        self._graph = nodes
        
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i >= j:
                    continue
                
                if self._are_connected(node_i, node_j):
                    node_i.connections.append(node_j)
                    node_j.connections.append(node_i)
                    node_i.out_degree += 1
                    node_j.in_degree += 1
        
        return nodes
    
    def _are_connected(self, node1: NoteGraphNode, node2: NoteGraphNode) -> bool:
        """检查两个节点是否相连"""
        time_overlap = (
            node1.start_time < node2.end_time and
            node2.start_time < node1.end_time
        )
        
        pitch_distance = abs(node1.pitch - node2.pitch)
        
        return time_overlap and pitch_distance <= 12
    
    def _harmony_check(self, nodes: list[NoteGraphNode]) -> list[NoteGraphNode]:
        """和声一致性检查"""
        if not self._cfg.enable_harmony_check:
            return nodes
        
        clusters = self._find_vertical_chords(nodes)
        
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            
            pitch_classes = [n.pitch_class for n in cluster]
            
            if self._is_dissonant_cluster(pitch_classes):
                self._resolve_dissonance(cluster)
        
        return nodes
    
    def _find_vertical_chords(self, nodes: list[NoteGraphNode]) -> list[list[NoteGraphNode]]:
        """找同时发声的音符组"""
        clusters: list[list[NoteGraphNode]] = []
        
        sorted_nodes = sorted(nodes, key=lambda n: n.start_time)
        
        i = 0
        while i < len(sorted_nodes):
            current_time = sorted_nodes[i].start_time
            
            cluster = [
                n for n in sorted_nodes
                if n.start_time <= current_time < n.end_time
            ]
            
            if len(cluster) > 1:
                clusters.append(cluster)
            
            i += 1
        
        return clusters
    
    def _is_dissonant_cluster(self, pitch_classes: list[int]) -> bool:
        """检测不协和音程"""
        dissonant_intervals = {1, 6, 10, 11}
        
        for i in range(len(pitch_classes)):
            for j in range(i + 1, len(pitch_classes)):
                interval = (pitch_classes[j] - pitch_classes[i]) % 12
                
                if interval in dissonant_intervals:
                    return True
        
        return False
    
    def _resolve_dissonance(self, cluster: list[NoteGraphNode]):
        """解决不协和 - 降低置信度"""
        for node in cluster:
            node.confidence *= 0.7
    
    def _voice_leading_optimization(self, nodes: list[NoteGraphNode]) -> list[NoteGraphNode]:
        """声部进行优化"""
        if not self._cfg.enable_voice_leading:
            return nodes
        
        sorted_by_time = sorted(nodes, key=lambda n: n.start_time)
        
        for i in range(len(sorted_by_time) - 1):
            current = sorted_by_time[i]
            next_node = sorted_by_time[i + 1]
            
            if current.pitch == next_node.pitch:
                continue
            
            distance = abs(current.pitch - next_node.pitch)
            
            if distance > 12 and next_node.confidence < 0.5:
                next_node.confidence *= 0.8
        
        return nodes
    
    def _tempo_constraint(self, nodes: list[NoteGraphNode]) -> list[NoteGraphNode]:
        """节奏约束"""
        if not self._cfg.enable_tempo_constraints:
            return nodes
        
        beat_duration = 60.0 / self._cfg.tempo
        grid_size = beat_duration / self._cfg.beat_division
        
        for node in nodes:
            nearest_grid = round(node.start_time / grid_size) * grid_size
            
            deviation = abs(node.start_time - nearest_grid) / grid_size
            
            if deviation < 0.1 and node.confidence > 0.5:
                node.note.start_s = nearest_grid
        
        return nodes
    
    def _confidence_weighted_cleanup(self, nodes: list[NoteGraphNode]) -> list[NoteGraphNode]:
        """基于置信度清理"""
        if not self._cfg.enable_confidence_weighting:
            return nodes
        
        threshold = 0.3
        
        filtered_nodes = [
            node for node in nodes
            if node.confidence >= threshold
        ]
        
        return filtered_nodes
    
    def get_note_graph(self) -> list[NoteGraphNode]:
        """获取音符图"""
        return self._graph


class MusicLanguageCorrector:
    """音乐语言纠正器 - 类似 LM 的后处理"""
    
    def __init__(self):
        self._common_patterns = self._load_patterns()
    
    def _load_patterns(self) -> dict:
        """加载常见音乐模式"""
        return {
            "arpeggios": [
                [0, 4, 7],
                [0, 3, 7],
                [0, 5, 7],
            ],
            "scales": [
                list(range(12)),
                [0, 2, 4, 5, 7, 9, 11],
                [0, 2, 3, 5, 7, 8, 10],
            ],
        }
    
    def correct(self, notes: list[NoteEvent]) -> list[NoteEvent]:
        """音乐语言纠正"""
        if not notes:
            return notes
        
        notes = self._fix_octave_errors(notes)
        
        notes = self._fix_timing_jitter(notes)
        
        notes = self._enforce_voice_leading(notes)
        
        return notes
    
    def _fix_octave_errors(self, notes: list[NoteEvent]) -> list[NoteEvent]:
        """修正八度错误"""
        corrected = []
        
        for note in notes:
            confidence = getattr(note, 'confidence', 1.0)
            
            if confidence < 0.4:
                if 21 <= note.note <= 108:
                    pitch_class = note.note % 12
                    
                    expected_octave = 5
                    corrected_note = pitch_class + expected_octave * 12
                    
                    if abs(corrected_note - note.note) <= 12:
                        note = NoteEvent(
                            note=corrected_note,
                            start_s=note.start_s,
                            end_s=note.end_s,
                            velocity=note.velocity,
                            confidence=note.confidence * 0.9,
                        )
            
            corrected.append(note)
        
        return corrected
    
    def _fix_timing_jitter(self, notes: list[NoteEvent]) -> list[NoteEvent]:
        """修正时间抖动"""
        if len(notes) < 3:
            return notes
        
        sorted_notes = sorted(notes, key=lambda n: n.start_s)
        
        for i in range(1, len(sorted_notes) - 1):
            prev = sorted_notes[i - 1]
            curr = sorted_notes[i]
            next_node = sorted_notes[i + 1]
            
            if (
                abs(curr.start_s - prev.start_s) < 0.01 and
                abs(next_node.start_s - curr.start_s) < 0.01
            ):
                avg_time = (prev.start_s + curr.start_s + next_node.start_s) / 3.0
                
                curr.start_s = avg_time
        
        return sorted_notes
    
    def _enforce_voice_leading(self, notes: list[NoteEvent]) -> list[NoteEvent]:
        """强制声部进行原则"""
        sorted_notes = sorted(notes, key=lambda n: (n.start_s, n.note))
        
        for i in range(len(sorted_notes) - 1):
            curr = sorted_notes[i]
            next_node = sorted_notes[i + 1]
            
            if curr.end_s > next_node.start_s:
                overlap = curr.end_s - next_node.start_s
                
                if overlap > 0.05 and curr.velocity < next_node.velocity:
                    curr.end_s = next_node.start_s
        
        return sorted_notes
