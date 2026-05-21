# Audio → MIDI

音频转 MIDI 工具，专为钢琴独奏优化。

## 引擎

| 引擎 | 类型 | 说明 |
|---|---|---|
| Piano Transcription (Neural) | 神经网络 | 钢琴专用，支持 GPU |
| Ensemble (PT + BP) | 神经网络融合 | PT+BP 置信度投票融合，推荐 |
| Basic Pitch | 神经网络 | 通用乐器，可调阈值 |
| Harmonic Salience | DSP | 调试用 |
| Spectral Peaks | DSP | 调试用 |

## 安装

```bash
pip install -r requirements.txt
pip install piano-transcription-inference basic-pitch
```

Web 模式额外依赖（已包含在 requirements.txt 中）：
```bash
pip install fastapi uvicorn websockets python-multipart
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
            └── nmp.onnx    (ONNX 模型)
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

### 桌面模式

```bash
python run_desktop.py
```

### Web 模式

```bash
python run_web.py --host 0.0.0.0 --port 8000
```

启动后访问 `http://localhost:8000/docs` 查看 Swagger API 文档。

Web 模式提供两种转谱方式：
- **同步接口** `POST /api/transcribe`：单文件，直接返回 MIDI 二进制
- **异步接口** `POST /api/jobs` + `WS /ws/{id}`：多文件批量，WebSocket 实时进度

前端开发指导详见 [WEB_FRONTEND_GUIDE.md](WEB_FRONTEND_GUIDE.md)。

## API

### Python API

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

### Web API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/engines` | GET | 获取可用引擎列表 |
| `/api/transcribe` | POST | 同步单文件转谱，返回 MIDI |
| `/api/jobs` | POST | 异步批量转谱，返回 job_id |
| `/api/jobs/{id}` | GET | 查询任务状态 |
| `/ws/{id}` | WebSocket | 实时进度推送 + 取消任务 |
| `/api/jobs/{id}/download/{file}` | GET | 下载转谱结果 |

## 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `normalize` | 响度归一化 | True |
| `normalize_mode` | "rms" / "peak" | "peak" |
| `preemphasis` | 预加重滤波 | False |
| `confidence_threshold` | 音符置信度（0.5 以下过滤 BP 补充音符） | 0.2 |
| `bp_onset_threshold` | Basic Pitch 起音判定 | 0.35 |
| `bp_frame_threshold` | Basic Pitch 持续断点 | 0.20 |
| `velocity_stretch` | 力度拉伸 | True |

## 项目结构

```
audio-to-midi/
├── run_desktop.py          # 桌面应用入口
├── run_web.py              # Web 服务入口
├── requirements.txt         # Python 依赖
├── WEB_FRONTEND_GUIDE.md   # Web 前端开发指导
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
    ├── ui.py               # 桌面界面（PySide6）
    ├── cli.py               # 命令行接口
    ├── cloud_client.py      # 云端客户端
    ├── cloud_server.py      # Web API 服务器（FastAPI）
    ├── rhythm/
    │   ├── __init__.py
    │   └── beat_tracking.py # 节拍追踪
    └── symbolic/
        └── __init__.py     # 符号音乐处理
```
