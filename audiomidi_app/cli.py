from __future__ import annotations

import argparse
from pathlib import Path

from audiomidi_app.audio import read_audio
from audiomidi_app.cloud_client import CloudConfig, transcribe_via_cloud
from audiomidi_app.midi import events_to_midi
from audiomidi_app.transcribe import available_transcribers, try_basic_pitch_transcriber


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audiomidi")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--engine", default="Spectral Peaks (MVP)")
    p.add_argument("--bpm", type=float, default=120.0)
    p.add_argument("--basic-pitch-out-dir", default=None)
    p.add_argument("--cloud-url", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.engine == "Basic Pitch":
        bp = try_basic_pitch_transcriber()
        if bp is None or not hasattr(bp, "transcribe_file"):
            raise SystemExit("当前环境未安装 basic-pitch 或不兼容")
        out_dir = args.basic_pitch_out_dir or str(out_path.parent)
        midi_path = bp.transcribe_file(str(in_path), out_dir=out_dir)
        if Path(midi_path) != out_path:
            out_path.write_bytes(Path(midi_path).read_bytes())
        return 0

    if args.cloud_url:
        try:
            midi_bytes = transcribe_via_cloud(
                CloudConfig(base_url=args.cloud_url),
                audio_path=in_path,
                engine=args.engine,
                bpm=float(args.bpm),
            )
            out_path.write_bytes(midi_bytes)
            return 0
        except Exception:
            pass

    transcribers = {t.name: t for t in available_transcribers()}
    if args.engine not in transcribers:
        names = ", ".join(transcribers.keys())
        raise SystemExit(f"未知引擎 {args.engine}，可用：{names}")

    audio = read_audio(in_path, target_sr=None, mono=True)
    events = transcribers[args.engine].transcribe(audio.samples, audio.sample_rate)
    mid = events_to_midi(events, bpm=args.bpm)
    mid.save(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
