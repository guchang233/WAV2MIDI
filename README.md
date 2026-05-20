# Audio → MIDI

音频转 MIDI 工具，专为钢琴独奏优化。

## 引擎

| 引擎 | 类型 | 说明 |
|---|---|---|
| Piano Transcription (Neural) | 神经网络 | 钢琴专用，支持 GPU | 推荐,目前效果最好 |
| Ensemble (PT + BP) | 神经网络融合 | 推荐，PT+BP 取并集 |
| Basic Pitch | 神经网络 | 通用乐器，可调阈值 |
| Harmonic Salience | DSP | 调试用 |
| Spectral Peaks | DSP | 调试用 |

## 安装

```bash
pip install -r requirements.txt
pip install piano-transcription-inference basic-pitch
```

## 使用

```bash
python run_desktop.py
```

## API

```python
from audiomidi_app.audio import read_audio
from audiomidi_app.transcribe import available_transcribers
from audiomidi_app.midi import events_to_midi

audio = read_audio("input.wav")
transcriber = available_transcribers()[0]
notes = transcriber.transcribe(audio.samples, audio.sample_rate)
midi = events_to_midi(notes, bpm=120.0)
midi.save("output.mid")
```

## 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `normalize` | 响度归一化 | True |
| `normalize_mode` | "rms" / "peak" | "peak" |
| `preemphasis` | 预加重滤波 | False |
| `confidence_threshold` | 音符置信度 | 0.2 |

## 项目结构

```
audiomidi_app/
├── __init__.py           # 模块入口
├── audio.py              # 音频读取与预处理
├── midi.py               # MIDI 生成
├── transcribe.py         # 转录引擎（PT/BP/Ensemble）
├── postprocess.py        # 后处理管道
├── voice_separation.py   # 声部分离算法
├── diagnostics.py        # 转谱报告统计
├── ui.py                 # 桌面界面
├── cli.py                # 命令行接口
├── cloud_client.py       # 云端客户端
├── cloud_server.py       # 云端服务器
├── rhythm/
│   ├── __init__.py
│   └── beat_tracking.py  # 节拍追踪
└── symbolic/
    └── __init__.py       # 符号音乐处理
```