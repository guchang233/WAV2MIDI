import unittest

import numpy as np

from audiomidi_app.midi import events_to_midi
from audiomidi_app.transcribe import SpectralPeaksTranscriber


class TestSpectralPeaksTranscriber(unittest.TestCase):
    def test_chord(self) -> None:
        sr = 22050
        t = np.arange(0, 2 * sr) / sr
        x = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 659.25 * t)
        events = SpectralPeaksTranscriber().transcribe(x.astype(np.float32), sr)
        notes = {e.note for e in events}
        self.assertIn(69, notes)
        self.assertIn(76, notes)
        mid = events_to_midi(events, bpm=120.0)
        self.assertGreaterEqual(len(mid.tracks), 1)


if __name__ == "__main__":
    unittest.main()

