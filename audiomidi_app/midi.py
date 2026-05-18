from __future__ import annotations

from dataclasses import dataclass

import mido


@dataclass(frozen=True)
class NoteEvent:
    note: int
    start_s: float
    end_s: float
    velocity: int


def events_to_midi(
    events: list[NoteEvent],
    *,
    bpm: float = 120.0,
    ticks_per_beat: int = 480,
) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=1)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    def s_to_ticks(t: float) -> int:
        return int(mido.second2tick(t, ticks_per_beat=ticks_per_beat, tempo=tempo))

    msgs: list[tuple[int, mido.Message]] = []
    for e in events:
        start = max(0, s_to_ticks(e.start_s))
        end = max(start + 1, s_to_ticks(e.end_s))
        vel = int(max(1, min(127, e.velocity)))
        msgs.append((start, mido.Message("note_on", note=e.note, velocity=vel, time=0)))
        msgs.append((end, mido.Message("note_off", note=e.note, velocity=0, time=0)))

    msgs.sort(key=lambda x: x[0])

    last_tick = 0
    for tick, msg in msgs:
        delta = tick - last_tick
        last_tick = tick
        msg.time = max(0, delta)
        track.append(msg)

    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid

