# Audio → MIDI

音频转 MIDI 工具，支持神经网络和 DSP 两种转录方式。

## 引擎

| 引擎 | 类型 | 说明 |
|---|---|---|
| Piano Transcription | 神经网络 | 推荐，钢琴专用，精度最高 |
| Basic Pitch | 神经网络 | 通用乐器，轻量快速 |
| Harmonic Salience | DSP | 调试用途 |
| Spectral Peaks | DSP | 调试用途 |

## 安装

```bash
pip install -r requirements.txt
```

可选安装神经网络引擎：

```bash
pip install piano-transcription-inference
pip install basic-pitch
```

## 使用

### 桌面应用

```bash
python run_desktop.py
```

支持拖拽音频文件、自动检测 BPM、左右手分离、实时日志。

### 命令行

```bash
python -m audiomidi_app.cli --in input.wav --out output.mid
```

### Python API

```python
from audiomidi_app.audio import read_audio
from audiomidi_app.transcribe import available_transcribers
from audiomidi_app.midi import events_to_midi

audio = read_audio("input.wav", target_sr=None, mono=True)
transcriber = available_transcribers()[0]  # 自动选择最佳引擎
notes = transcriber.transcribe(audio.samples, audio.sample_rate)
midi = events_to_midi(notes, bpm=120.0)
midi.save("output.mid")
```

## 项目结构

```
audiomidi_app/
├── audio.py            音频读取
├── midi.py             MIDI 生成
├── transcribe.py       转录引擎
├── postprocess.py      后处理
├── voice_separation.py 声部分离
├── ui.py               桌面界面
├── cloud_client.py     云端客户端
└── cloud_server.py     云端服务器
```
