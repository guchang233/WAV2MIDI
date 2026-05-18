from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response

from audiomidi_app.audio import read_audio
from audiomidi_app.midi import events_to_midi
from audiomidi_app.transcribe import SpectralPeaksTranscriber, try_basic_pitch_transcriber

app = FastAPI()


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    engine: str = Form("Spectral Peaks (MVP)"),
    bpm: float = Form(120.0),
) -> Response:
    with tempfile.TemporaryDirectory() as td:
        audio_path = Path(td) / (file.filename or "input.audio")
        audio_path.write_bytes(await file.read())

        if engine == "Basic Pitch":
            bp = try_basic_pitch_transcriber()
            if bp is None or not hasattr(bp, "transcribe_file"):
                return Response(status_code=400, content=b"basic-pitch not available")
            midi_path = bp.transcribe_file(str(audio_path), out_dir=td)
            midi_bytes = Path(midi_path).read_bytes()
            return Response(content=midi_bytes, media_type="audio/midi")

        audio = read_audio(audio_path, target_sr=None, mono=True)
        events = SpectralPeaksTranscriber().transcribe(audio.samples, audio.sample_rate)
        mid = events_to_midi(events, bpm=bpm)
        midi_path = Path(td) / "out.mid"
        mid.save(str(midi_path))
        return Response(content=midi_path.read_bytes(), media_type="audio/midi")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
