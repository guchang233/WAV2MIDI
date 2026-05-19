from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
import heapq

import numpy as np

from audiomidi_app.midi import NoteEvent


@dataclass(frozen=True)
class VoiceSeparationConfig:
    max_voices: int = 6
    beam_width: int = 8
    
    # Cost weights
    pitch_cost_weight: float = 1.0
    time_cost_weight: float = 0.5
    velocity_cost_weight: float = 0.1
    motion_cost_weight: float = 1.0
    inactivity_cost_weight: float = 0.5
    direction_change_cost_weight: float = 1.5
    overlap_penalty_weight: float = 50.0
    crossing_penalty_weight: float = 30.0
    
    # Chord grouping
    chord_time_threshold: float = 0.05
    chord_pitch_threshold: float = 24.0
    
    # Voice behavior
    voice_activation_threshold: float = 40.0
    min_note_duration: float = 0.05
    voice_birth_cost_base: float = 80.0
    
    # Hand assignment
    hand_smoothing_factor: float = 0.7
    hand_max_span: float = 36.0
    
    # Prediction
    inertia_alpha: float = 0.6


@dataclass
class VoiceState:
    id: int
    notes: list[NoteEvent] = field(default_factory=list)
    pitch_history: list[int] = field(default_factory=list)
    time_history: list[float] = field(default_factory=list)
    velocity_history: list[int] = field(default_factory=list)
    pitch_velocity: float = 0.0
    last_pitch: float = 60.0
    last_end_time: float = 0.0
    last_direction: int = 0
    
    @property
    def avg_pitch(self) -> float:
        if not self.pitch_history:
            return 60.0
        return np.mean(self.pitch_history[-10:])
    
    @property
    def activity_score(self) -> float:
        if not self.notes:
            return 0.0
        recent = [n.end_s for n in self.notes[-3:]]
        return 1.0 / (1.0 + (max(recent) - self.last_end_time) * 0.1)
    
    def predict_next_pitch(self) -> float:
        if len(self.pitch_history) < 2:
            return self.last_pitch
        
        recent_pitches = np.array(self.pitch_history[-5:])
        recent_times = np.array(self.time_history[-5:])
        
        if len(recent_pitches) < 3:
            delta = recent_pitches[-1] - recent_pitches[0]
            return recent_pitches[-1] + delta * 0.3
        
        delta = recent_pitches[-1] - recent_pitches[0]
        return self.last_pitch + delta * 0.3
    
    def is_note_active_at(self, time: float) -> bool:
        if not self.notes:
            return False
        last = self.notes[-1]
        return last.start_s <= time < last.end_s
    
    def add_note(self, note: NoteEvent) -> None:
        self.notes.append(note)
        self.pitch_history.append(note.note)
        self.time_history.append(note.start_s)
        self.velocity_history.append(note.velocity)
        
        if len(self.pitch_history) >= 2:
            delta = self.pitch_history[-1] - self.pitch_history[-2]
            new_direction = np.sign(delta) if abs(delta) > 1 else 0
            self.pitch_velocity = delta / max(0.01, note.start_s - self.last_end_time)
            self.last_direction = new_direction
        
        self.last_pitch = note.note
        self.last_end_time = note.end_s


@dataclass
class HandState:
    center: float = 60.0
    pitch_min: float = 21.0
    pitch_max: float = 108.0
    
    def update(self, pitch: int, smoothing: float = 0.7) -> None:
        self.center = smoothing * self.center + (1 - smoothing) * pitch
        self.pitch_min = min(self.pitch_min, pitch)
        self.pitch_max = max(self.pitch_max, pitch)
    
    def distance_to(self, pitch: int) -> float:
        return abs(pitch - self.center)


@dataclass
class AssignmentHypothesis:
    states: list[VoiceState]
    total_cost: float = 0.0
    n_notes_assigned: int = 0
    left_hand: HandState = field(default_factory=HandState)
    right_hand: HandState = field(default_factory=HandState)
    
    def __lt__(self, other: AssignmentHypothesis) -> bool:
        return self.total_cost < other.total_cost


def compute_voice_cost(
    note: NoteEvent,
    voice: VoiceState,
    config: VoiceSeparationConfig,
    current_time: float,
) -> float:
    if not voice.notes:
        return 100.0
    
    pitch_diff = abs(note.note - voice.last_pitch)
    pitch_cost = config.pitch_cost_weight * pitch_diff
    
    time_gap = max(0.0, note.start_s - voice.last_end_time)
    time_cost = config.time_cost_weight * time_gap * 100.0
    
    last_note = voice.notes[-1]
    velocity_diff = abs(note.velocity - last_note.velocity)
    velocity_cost = config.velocity_cost_weight * velocity_diff
    
    predicted = voice.predict_next_pitch()
    motion_diff = abs(note.note - predicted)
    motion_cost = config.motion_cost_weight * motion_diff
    
    direction_cost = 0.0
    if len(voice.pitch_history) >= 2:
        delta = note.note - voice.last_pitch
        new_dir = np.sign(delta) if abs(delta) > 1 else 0
        if voice.last_direction != 0 and new_dir != 0 and new_dir != voice.last_direction:
            direction_cost = config.direction_change_cost_weight * 10.0
    
    inactivity = max(0.0, current_time - voice.last_end_time - 2.0)
    inactivity_cost = config.inactivity_cost_weight * inactivity * 20.0
    
    overlap_cost = 0.0
    if voice.is_note_active_at(note.start_s):
        overlap_cost = config.overlap_penalty_weight
    
    crossing_cost = 0.0
    avg_pitch = voice.avg_pitch
    if len(voice.notes) >= 2:
        recent_avg = np.mean([n.note for n in voice.notes[-5:]])
        if (recent_avg > note.note and note.note < voice.last_pitch - 3) or \
           (recent_avg < note.note and note.note > voice.last_pitch + 3):
            crossing_cost = config.crossing_penalty_weight
    
    activity_factor = 1.0 / (1.0 + voice.activity_score)
    activity_cost = inactivity_cost * activity_factor * 0.5
    
    return pitch_cost + time_cost + velocity_cost + motion_cost + direction_cost + inactivity_cost + overlap_cost + crossing_cost + activity_cost


def hungarian_chord_assignment(
    notes: list[NoteEvent],
    voices: list[VoiceState],
    config: VoiceSeparationConfig,
    current_time: float,
) -> list[tuple[int, int]]:
    n_notes = len(notes)
    n_voices = len(voices)
    
    if n_notes == 0 or n_voices == 0:
        return []
    
    cost_matrix = np.full((max(n_notes, n_voices), max(n_notes, n_voices)), 1000.0)
    
    for i, note in enumerate(notes):
        for j, voice in enumerate(voices):
            cost_matrix[i, j] = compute_voice_cost(note, voice, config, current_time)
    
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignments = []
        for r, c in zip(row_ind, col_ind):
            if r < n_notes and c < n_voices:
                assignments.append((r, c))
        return assignments
    except ImportError:
        assignments = []
        used_voices = set()
        for i in range(min(n_notes, n_voices)):
            best_voice = -1
            best_cost = float('inf')
            for j in range(n_voices):
                if j not in used_voices:
                    cost = compute_voice_cost(notes[i], voices[j], config, current_time)
                    if cost < best_cost:
                        best_cost = cost
                        best_voice = j
            if best_voice >= 0:
                assignments.append((i, best_voice))
                used_voices.add(best_voice)
        return assignments


def assign_voices_beam_search(
    events: list[NoteEvent],
    config: VoiceSeparationConfig | None = None,
) -> list[VoiceState]:
    if config is None:
        config = VoiceSeparationConfig()
    
    if not events:
        return []
    
    sorted_events = sorted(events, key=lambda n: (n.start_s, -n.note))
    chord_groups = chord_grouping_advanced(sorted_events, config)
    
    beam: list[AssignmentHypothesis] = []
    initial_hypothesis = AssignmentHypothesis(
        states=[VoiceState(id=0)],
        total_cost=0.0,
        n_notes_assigned=0,
        left_hand=HandState(center=45.0),
        right_hand=HandState(center=72.0),
    )
    beam.append(initial_hypothesis)
    
    for chord_idx, chord in enumerate(chord_groups):
        current_time = chord[0].start_s if chord else 0.0
        new_beam: list[AssignmentHypothesis] = []
        
        for hypothesis in beam:
            if len(chord) == 1:
                note = chord[0]
                
                for v_idx, voice in enumerate(hypothesis.states):
                    cost = compute_voice_cost(note, voice, config, current_time)
                    new_states = deepcopy(hypothesis.states)
                    new_states[v_idx].add_note(note)
                    
                    new_hypothesis = AssignmentHypothesis(
                        states=new_states,
                        total_cost=hypothesis.total_cost + cost,
                        n_notes_assigned=hypothesis.n_notes_assigned + 1,
                        left_hand=deepcopy(hypothesis.left_hand),
                        right_hand=deepcopy(hypothesis.right_hand),
                    )
                    
                    if v_idx < len(hypothesis.states) // 2:
                        new_hypothesis.left_hand.update(note.note, config.hand_smoothing_factor)
                    else:
                        new_hypothesis.right_hand.update(note.note, config.hand_smoothing_factor)
                    
                    heapq.heappush(new_beam, new_hypothesis)
                    if len(new_beam) > config.beam_width * 2:
                        # 保留 cost 最小的 beam_width 个（heappop 删最小，所以要先 nsmallest 再 heapify）
                        new_beam = heapq.nsmallest(config.beam_width, new_beam)
                        heapq.heapify(new_beam)
                
                if len(hypothesis.states) < config.max_voices:
                    n_active = sum(1 for v in hypothesis.states if v.is_note_active_at(current_time))
                    birth_cost = config.voice_birth_cost_base + n_active * 20.0
                    
                    new_states = deepcopy(hypothesis.states)
                    new_voice = VoiceState(id=len(new_states))
                    new_voice.add_note(note)
                    new_states.append(new_voice)
                    
                    new_hypothesis = AssignmentHypothesis(
                        states=new_states,
                        total_cost=hypothesis.total_cost + birth_cost,
                        n_notes_assigned=hypothesis.n_notes_assigned + 1,
                        left_hand=deepcopy(hypothesis.left_hand),
                        right_hand=deepcopy(hypothesis.right_hand),
                    )
                    new_hypothesis.left_hand.update(note.note, config.hand_smoothing_factor)
                    heapq.heappush(new_beam, new_hypothesis)
            
            else:
                assignments = hungarian_chord_assignment(
                    chord, hypothesis.states, config, current_time
                )
                
                new_states = deepcopy(hypothesis.states)
                new_hypothesis = AssignmentHypothesis(
                    states=new_states,
                    total_cost=hypothesis.total_cost,
                    n_notes_assigned=hypothesis.n_notes_assigned + len(chord),
                    left_hand=deepcopy(hypothesis.left_hand),
                    right_hand=deepcopy(hypothesis.right_hand),
                )
                
                assigned_notes = set()
                for note_idx, voice_idx in assignments:
                    note = chord[note_idx]
                    new_states[voice_idx].add_note(note)
                    new_hypothesis.total_cost += cost_matrix_element(
                        note, new_states[voice_idx], config, current_time
                    )
                    assigned_notes.add(note_idx)
                    
                    avg_pitch = new_states[voice_idx].avg_pitch
                    if avg_pitch < 60:
                        new_hypothesis.left_hand.update(note.note, config.hand_smoothing_factor)
                    else:
                        new_hypothesis.right_hand.update(note.note, config.hand_smoothing_factor)
                
                for i, note in enumerate(chord):
                    if i not in assigned_notes:
                        if len(new_states) < config.max_voices:
                            new_voice = VoiceState(id=len(new_states))
                            new_voice.add_note(note)
                            new_states.append(new_voice)
                            new_hypothesis.total_cost += config.voice_birth_cost_base
                            new_hypothesis.left_hand.update(note.note, config.hand_smoothing_factor)
                        else:
                            min_voice = min(range(len(new_states)), 
                                          key=lambda v: compute_voice_cost(note, new_states[v], config, current_time))
                            new_states[min_voice].add_note(note)
                            new_hypothesis.total_cost += compute_voice_cost(note, new_states[min_voice], config, current_time)
                
                heapq.heappush(new_beam, new_hypothesis)
        
        beam = heapq.nsmallest(config.beam_width, new_beam)
        if not beam:
            beam = [initial_hypothesis]
        
        if len(beam) > 1:
            costs = [h.total_cost for h in beam]
            cost_std = np.std(costs)
            if cost_std < 1.0:
                for i in range(len(beam)):
                    beam[i].total_cost += i * 0.001
                heapq.heapify(beam)
    
    if not beam:
        return []
    
    best = min(beam, key=lambda h: h.total_cost)
    result_voices = [s for s in best.states if s.notes]
    
    if not result_voices:
        result_voices = [VoiceState(id=0)]
        for e in sorted_events:
            result_voices[0].add_note(e)
    
    return result_voices


def cost_matrix_element(
    note: NoteEvent,
    voice: VoiceState,
    config: VoiceSeparationConfig,
    current_time: float,
) -> float:
    return compute_voice_cost(note, voice, config, current_time)


def chord_grouping_advanced(
    events: list[NoteEvent],
    config: VoiceSeparationConfig,
) -> list[list[NoteEvent]]:
    if not events:
        return []
    
    groups: list[list[NoteEvent]] = []
    current_group = [events[0]]
    
    for note in events[1:]:
        last_note = current_group[-1]
        time_diff = note.start_s - last_note.start_s
        
        in_time_range = time_diff <= config.chord_time_threshold
        
        min_pitch = min(n.note for n in current_group)
        max_pitch = max(n.note for n in current_group)
        avg_pitch = np.mean([n.note for n in current_group])
        pitch_spread = max_pitch - min_pitch
        note_pitch_diff = abs(note.note - avg_pitch)
        
        in_pitch_range = (
            note_pitch_diff <= config.chord_pitch_threshold and 
            pitch_spread <= config.chord_pitch_threshold
        )
        
        if in_time_range and in_pitch_range:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
    
    if current_group:
        groups.append(current_group)
    
    return groups


def assign_hands_using_trajectories(
    voices: list[VoiceState],
) -> list[int]:
    if not voices:
        return []
    
    if len(voices) == 1:
        return [1]
    
    hand_centers = []
    for voice in voices:
        if len(voice.notes) < 2:
            hand_centers.append(voice.avg_pitch)
        else:
            recent_pitches = [n.note for n in voice.notes[-10:]]
            recent_times = [n.start_s for n in voice.notes[-10:]]
            
            if len(recent_pitches) >= 2:
                slope = (recent_pitches[-1] - recent_pitches[0]) / max(0.01, recent_times[-1] - recent_times[0])
                trend = slope * 0.5
                center = np.mean(recent_pitches) + trend * 2.0
            else:
                center = np.mean(recent_pitches)
            hand_centers.append(center)
    
    left_center = np.percentile(hand_centers, 30)
    right_center = np.percentile(hand_centers, 70)
    
    assignments = []
    for voice in voices:
        avg = voice.avg_pitch
        dist_to_left = abs(avg - left_center)
        dist_to_right = abs(avg - right_center)
        assignments.append(0 if dist_to_left < dist_to_right else 1)
    
    left_count = sum(1 for a in assignments if a == 0)
    right_count = sum(1 for a in assignments if a == 1)
    
    if left_count == 0 and voices:
        assignments[0] = 0
        assignments[-1] = 1
    elif right_count == 0 and voices:
        assignments[0] = 0
        assignments[-1] = 1
    
    return assignments


@dataclass
class VoiceSeparationResult:
    voices: list[VoiceState]
    hand_assignments: list[int]
    
    def get_notes_for_voice(self, voice_idx: int) -> list[NoteEvent]:
        if 0 <= voice_idx < len(self.voices):
            return self.voices[voice_idx].notes
        return []
    
    def get_left_hand_notes(self) -> list[NoteEvent]:
        left_notes = []
        for v_idx, assignment in enumerate(self.hand_assignments):
            if assignment == 0:
                left_notes.extend(self.voices[v_idx].notes)
        return sorted(left_notes, key=lambda n: n.start_s)
    
    def get_right_hand_notes(self) -> list[NoteEvent]:
        right_notes = []
        for v_idx, assignment in enumerate(self.hand_assignments):
            if assignment == 1:
                right_notes.extend(self.voices[v_idx].notes)
        return sorted(right_notes, key=lambda n: n.start_s)


def separate_voices(
    events: list[NoteEvent],
    config: VoiceSeparationConfig | None = None,
) -> VoiceSeparationResult:
    if config is None:
        config = VoiceSeparationConfig()
    
    voices = assign_voices_beam_search(events, config)
    hand_assignments = assign_hands_using_trajectories(voices)
    
    return VoiceSeparationResult(voices=voices, hand_assignments=hand_assignments)
