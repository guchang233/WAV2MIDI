# Audio → MIDI (音频转MIDI)

一个功能强大的音频转MIDI工具，支持多种转录引擎，专为钢琴、人声、多音轨等场景优化。

## 特性

### 🎹 多引擎支持

- **Piano Transcription** - 专用钢琴转录引擎（基于MAESTRO数据集训练）
  - 准确率高达85%+
  - 准确的onset/offset/velocity检测
  - 处理钢琴延音踏板

- **Basic Pitch** - Spotify研究团队的神经网络模型
  - 适合人声、吉他等单声部乐器
  - 支持多音高检测

- **Harmonic Salience** - 谐波显著性算法
  - 抑制泛音鬼影
  - 处理多声部和弦

- **Spectral Peaks** - 基础频谱峰值检测
  - 轻量级、纯numpy实现
  - 适合简单场景

### 🎼 音频分离

集成Demucs分离引擎，支持：
- Bass (低音)
- Vocals (人声)
- Drums (鼓)
- Other (其他乐器)

### 🥁 鼓轨转谱

支持GM打击乐映射：
- Kick (36)
- Snare (38)
- Hi-Hat (42)

## 安装

### 1. 基础依赖

```bash
pip install -r requirements.txt
```

### 2. 可选：安装专用引擎

#### Piano Transcription (推荐用于钢琴)
```bash
pip install piano-transcription-inference
```

#### Basic Pitch (推荐用于人声/吉他)
```bash
pip install basic-pitch
```

#### Demucs (音频分离)
```bash
pip install demucs
```

## 使用方法

### 桌面UI

```bash
python run_desktop.py
```

### 命令行

```bash
# 使用默认引擎（Harmonic Salience）
python -m audiomidi_app.cli --in input.wav --out output.mid

# 选择引擎
python -m audiomidi_app.cli --in input.wav --out output.mid --engine="Piano Transcription"

# 可选引擎列表:
# - "Piano Transcription"
# - "Harmonic Salience"
# - "Spectral Peaks"
# - "Basic Pitch"
```

### 编程方式

```python
from audiomidi_app.transcribe import available_transcribers, detect_bpm
from audiomidi_app.audio import read_audio
from audiomidi_app.midi import events_to_midi

# 选择引擎
engines = available_transcribers()
engine = engines[0]  # 默认第一个是Piano Transcription

# 读取音频
audio = read_audio("input.wav", target_sr=None, mono=True)

# 转谱
events = engine.transcribe(audio.samples, audio.sample_rate)

# 自动检测BPM
bpm = detect_bpm(audio.samples, audio.sample_rate)

# 输出MIDI
mid = events_to_midi(events, bpm=bpm)
mid.save("output.mid")
```

## 引擎选择指南

| 场景 | 首选引擎 | 原因 |
|------|---------|------|
| 🎹 钢琴独奏 | Piano Transcription | 专用模型，准确率最高 |
| 🎸 吉他/贝斯 | Basic Pitch | 拨弦乐器效果好 |
| 🎤 人声 | Basic Pitch | 单音高跟踪准确 |
| 🎼 复杂多声部 | Basic Pitch + Demucs | 先分离后转谱 |
| ⚡ 快速原型 | Harmonic Salience | 纯numpy，速度快 |

## 项目结构

```
├── audiomidi_app/
│   ├── __init__.py
│   ├── audio.py              # 音频IO和处理
│   ├── midi.py               # MIDI事件和输出
│   ├── transcribe.py         # 转谱引擎核心
│   ├── cli.py                # 命令行接口
│   ├── ui.py                 # 桌面GUI
│   ├── cloud_client.py       # 云端API客户端
│   └── cloud_server.py       # 云端API服务端
├── run_desktop.py            # 启动桌面应用
├── requirements.txt          # 依赖列表
└── README.md
```

## 核心功能说明

### 1. Onset检测

使用librosa onset detect检测音符起始点，解决：
- 钢琴延音踏板造成的音符粘连
- 连奏乐段的音符分隔

### 2. Harmonic Salience算法

计算谐波显著性，避免：
- 泛音被误识别为独立音符
- 低音区基频能量低于泛音时的误识别

### 3. Velocity映射

对钢琴采用非线性三次方根映射：
```python
db = 20.0 * np.log10(amp)
normalized = (db + 70.0) / 70.0
v = int(normalized ** (1/3) * 126) + 1
```

## 云端支持

项目支持云端/本地混合模式：

### 启动服务器
```bash
python -m audiomidi_app.cloud_server --host 0.0.0.0 --port 8000
```

### 客户端使用
在UI中勾选"云端优先"，失败自动回退本地。

## 测试

```bash
pytest test_transcribe.py -v
```

## 贡献

欢迎提交Issue和PR！

## 许可证

MIT License

## 致谢

- [Basic Pitch](https://github.com/spotify/basic-pitch) - Spotify研究团队
- [Piano Transcription Inference](https://github.com/qiuqiangkong/piano_transcription_inference) - Kong et al.
- [Demucs](https://github.com/facebookresearch/demucs) - Facebook研究团队
- [Librosa](https://librosa.org/) - 音频分析库
