from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class CloudConfig:
    base_url: str
    timeout_s: float = 120.0


def transcribe_via_cloud(
    cfg: CloudConfig,
    *,
    audio_path: str | Path,
    engine: str,
    bpm: float,
) -> bytes:
    url = cfg.base_url.rstrip("/") + "/transcribe"
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f, "application/octet-stream")}
        data = {"engine": engine, "bpm": str(bpm)}
        r = requests.post(url, files=files, data=data, timeout=cfg.timeout_s)
    r.raise_for_status()
    return r.content

