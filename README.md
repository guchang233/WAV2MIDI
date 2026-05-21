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

## 模型安装

首次运行会自动下载模型文件（需要网络连接）。

### 自动安装

运行程序后，Piano Transcription 和 Basic Pitch 的模型会自动从 Zenodo 下载到：
- `C:\Users\<用户名>\piano_transcription_inference_data\` （约 165 MB）
- Python site-packages 目录

### 手动安装（可选）

如需将模型文件放置在项目目录，可手动下载并放入 `resources/models/`：

```
resources/
└── models/
    ├── piano_transcription_inference_data/
    │   └── note_F1=0.9677_pedal_F1=0.9186.pth   (约 165 MB)
    └── basic_pitch/
        └── icassp_2022/
            └── nmp/    (ONNX/TensorFlow 模型)
```

**Piano Transcription 模型下载**：
- 链接：https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth
- 文件名：`note_F1=0.9677_pedal_F1=0.9186.pth`

**Basic Pitch 模型**：
- 随 basic-pitch 包自动安装，位于 Python site-packages 目录
- 如需手动复制，复制 `site-packages/basic_pitch/saved_models/icassp_2022/` 到 `resources/models/basic_pitch/icassp_2022/`

### 模型优先级

程序按以下顺序查找模型：
1. `resources/models/` （项目目录，推荐）
2. `~/.piano_transcription_inference_data/` （Home 目录）
3. Python site-packages 目录

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
audio-to-midi/
├── run_desktop.py          # 桌面应用入口
├── requirements.txt         # Python 依赖
├── resources/               # 外部资源
│   └── models/             # 模型文件（可选）
│       ├── piano_transcription_inference_data/
│       └── basic_pitch/
└── audiomidi_app/
    ├── __init__.py
    ├── paths.py            # 资源路径管理
    ├── audio.py            # 音频读取与预处理
    ├── midi.py             # MIDI 生成
    ├── transcribe.py        # 转录引擎（PT/BP/Ensemble）
    ├── postprocess.py       # 后处理管道
    ├── voice_separation.py  # 声部分离算法
    ├── diagnostics.py       # 转谱报告统计
    ├── ui.py               # 桌面界面
    ├── cli.py               # 命令行接口
    ├── cloud_client.py      # 云端客户端
    ├── cloud_server.py      # 云端服务器
    ├── rhythm/
    │   ├── __init__.py
    │   └── beat_tracking.py # 节拍追踪
    └── symbolic/
        └── __init__.py     # 符号音乐处理
```