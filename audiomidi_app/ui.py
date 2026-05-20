from __future__ import annotations

import sys
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime

from audiomidi_app.audio import read_audio
from audiomidi_app.cloud_client import CloudConfig, transcribe_via_cloud
from audiomidi_app.midi import NoteEvent, events_to_midi
from audiomidi_app.transcribe import (
    available_transcribers,
    available_voice_separation_transcribers,
    VoiceSeparationTranscriber,
)
from audiomidi_app.voice_separation import separate_voices, VoiceSeparationResult

_qt_import_error: Exception | None = None
try:
    from PySide6.QtCore import QObject, Qt, QThread, Signal, QTimer, QSize
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
        QGroupBox,
        QScrollArea,
        QTextEdit,
    )
    from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont, QTextCursor, QColor, QPalette
except Exception as e:
    _qt_import_error = e


@dataclass(frozen=True)
class JobConfig:
    audio_path: str
    out_dir: str
    engine: str
    bpm: float
    auto_bpm: bool
    cloud_enabled: bool
    cloud_base_url: str
    use_voice_separation: bool
    split_hands: bool
    left_hand_channel: int
    right_hand_channel: int
    normalize_audio: bool = True
    preemphasis_audio: bool = False
    velocity_stretch: bool = True
    confidence_threshold: float = 0.2
    bp_onset_threshold: float = 0.35
    bp_frame_threshold: float = 0.20


def _get_modern_stylesheet() -> str:
    """Dark theme stylesheet"""
    return """
        QMainWindow, QWidget {
            background-color: #0A0A0A;
            font-family: "PingFang SC", "Microsoft YaHei", Consolas, monospace;
            font-size: 12px;
            color: #CCCCCC;
        }

        QScrollArea {
            background-color: transparent;
            border: none;
        }

        QWidget#densePanel {
            background-color: #121212;
            border: 1px solid #222222;
        }

        QTabWidget::pane {
            border: 1px solid #222222;
            background-color: #121212;
        }
        QTabBar {
            background-color: transparent;
            qproperty-drawBase: 0;
        }
        QTabBar::tab {
            padding: 6px 14px;
            background-color: #161616;
            color: #777777;
            border: 1px solid #222222;
            border-bottom: none;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            color: #FFFFFF;
            background-color: #121212;
            border-top: 2px solid #E03E3E;
        }
        QTabBar::tab:hover:!selected {
            color: #BBBBBB;
            background-color: #1C1C1C;
        }

        QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
            padding: 4px 6px;
            border: 1px solid #2A2A2A;
            border-radius: 0;
            background-color: #0A0A0A;
            color: #DDDDDD;
            height: 24px;
        }
        QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {
            border-color: #E03E3E;
        }

        QComboBox::drop-down {
            border: none;
            width: 18px;
        }

        QGroupBox {
            font-size: 11px;
            font-weight: bold;
            color: #888888;
            border: none;
            border-top: 1px dashed #222222;
            margin-top: 14px;
            padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 5px;
            padding: 0 4px;
            background-color: #121212;
        }

        QPushButton {
            padding: 2px 10px;
            border: 1px solid #2A2A2A;
            background-color: #1A1A1A;
            color: #CCCCCC;
            height: 24px;
        }
        QPushButton:hover {
            background-color: #262626;
            border-color: #444444;
            color: #FFFFFF;
        }
        QPushButton:pressed {
            background-color: #121212;
        }

        QPushButton#runBtn {
            background-color: #1A0D0D;
            color: #E03E3E;
            border: 1px solid #E03E3E;
            font-weight: bold;
        }
        QPushButton#runBtn:hover {
            background-color: #2E1212;
            color: #FF5252;
            border-color: #FF5252;
        }
        QPushButton#runBtn:pressed {
            background-color: #441212;
        }
        QPushButton#runBtn:disabled {
            background-color: #121212;
            border-color: #222222;
            color: #444444;
        }

        QPushButton#stopBtn {
            background-color: #121212;
            color: #888888;
            border: 1px solid #2A2A2A;
        }
        QPushButton#stopBtn:hover:enabled {
            background-color: #262626;
            color: #E03E3E;
            border-color: #E03E3E;
        }

        QCheckBox {
            spacing: 6px;
            color: #AAAAAA;
        }
        QCheckBox::indicator {
            width: 12px;
            height: 12px;
            border: 1px solid #2A2A2A;
            background-color: #0A0A0A;
        }
        QCheckBox::indicator:hover {
            border-color: #E03E3E;
        }
        QCheckBox::indicator:checked {
            background-color: #E03E3E;
            border-color: #E03E3E;
        }

        QProgressBar {
            border: 1px solid #222222;
            background-color: #0A0A0A;
            height: 4px;
            text-visible: false;
        }
        QProgressBar::chunk {
            background-color: #E03E3E;
        }

        QTextEdit {
            border: 1px solid #222222;
            background-color: #050505;
            color: #A0A0A0;
            font-family: Consolas, "JetBrains Mono", monospace;
            font-size: 11px;
            padding: 8px;
        }

        QLabel#statsLabel {
            color: #666666;
            font-size: 11px;
        }
        QLabel#statusLabel {
            color: #E03E3E;
            font-weight: bold;
        }

        QScrollBar:vertical {
            background: #0A0A0A;
            width: 5px;
        }
        QScrollBar::handle:vertical {
            background: #222222;
        }
        QScrollBar::handle:vertical:hover {
            background: #444444;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
    """


def _apply_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(_get_modern_stylesheet())


def run_app() -> None:
    if _qt_import_error is not None:
        raise RuntimeError(f"桌面UI依赖加载失败：{_qt_import_error}")

    class _SignalStream:
        def __init__(self, emit_fn):
            self._emit = emit_fn
            self._buf = io.StringIO()

        def write(self, text: str) -> int:
            self._buf.write(text)
            if '\n' in text:
                self.flush()
            return len(text)

        def flush(self) -> None:
            val = self._buf.getvalue()
            if not val:
                return
            lines = val.split('\n')
            for line in lines[:-1]:
                if line:
                    self._emit(line)
            self._buf = io.StringIO()
            if lines[-1]:
                self._buf.write(lines[-1])

        def isatty(self) -> bool:
            return False

    class Worker(QObject):
        progress = Signal(str)
        detail = Signal(str)
        progress_percent = Signal(int)
        done = Signal(str)
        failed = Signal(str)
        notes_found = Signal(int)
        voices_found = Signal(int)

        def __init__(self, cfg: JobConfig) -> None:
            super().__init__()
            self._cfg = cfg
            self._interrupted = False

        def interrupt(self) -> None:
            self._interrupted = True

        def _emit_stdout(self, line: str) -> None:
            self.detail.emit(f"[{self._time()}] [系统输出] {line}")

        def run(self) -> None:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stream = _SignalStream(self._emit_stdout)
            sys.stdout = stream
            sys.stderr = stream
            try:
                self.progress.emit("正在启动")
                self.progress_percent.emit(0)
                self.detail.emit(f"[{self._time()}] 任务初始化...")
                self.detail.emit(f"[{self._time()}] 推理引擎: {self._cfg.engine}")
                if self._cfg.auto_bpm:
                    self.detail.emit(f"[{self._time()}] 速度(BPM): 自动测速")
                else:
                    self.detail.emit(f"[{self._time()}] 速度(BPM): {self._cfg.bpm}")
                out = self._run_impl()
                if self._interrupted:
                    return
                self.progress_percent.emit(100)
                self.done.emit(out)
            except Exception as e:
                if not self._interrupted:
                    self.failed.emit(str(e))
            finally:
                stream.flush()
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        def _time(self) -> str:
            return datetime.now().strftime("%H:%M:%S")

        def _run_impl(self) -> str:
            audio_path = Path(self._cfg.audio_path)
            out_dir = Path(self._cfg.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / audio_path.with_suffix(".mid").name

            self.detail.emit(f"[{self._time()}] 输入文件: {audio_path}")
            self.detail.emit(f"[{self._time()}] 输出目录: {out_dir}")
            self.detail.emit(f"[{self._time()}] 导出目标: {out_path.name}")
            self.progress_percent.emit(5)

            if self._interrupted:
                return ""

            if self._cfg.cloud_enabled:
                try:
                    self.progress.emit("云端分布式算力处理中")
                    self.detail.emit(f"[{self._time()}] 连接云端节点: {self._cfg.cloud_base_url}")
                    midi_bytes = transcribe_via_cloud(
                        CloudConfig(base_url=self._cfg.cloud_base_url),
                        audio_path=audio_path,
                        engine=self._cfg.engine,
                        bpm=self._cfg.bpm,
                    )
                    out_path.write_bytes(midi_bytes)
                    self.detail.emit(f"[{self._time()}] ✅ 云端计算完成并成功下载")
                    return str(out_path)
                except Exception as e:
                    self.detail.emit(f"[{self._time()}] ⚠️ 云端集群异常: {e}")
                    self.progress.emit(f"触发无缝本地回退")
                    self.detail.emit(f"[{self._time()}] 正在调度本地硬件资源重算...")

            if self._interrupted:
                return ""

            self.progress.emit("音频特征分析中 (1/4)")
            self.progress_percent.emit(15)
            is_neural_engine = self._cfg.engine in ("Piano Transcription (Neural)", "Basic Pitch", "Ensemble (PT + BP)")
            audio = read_audio(
                audio_path, target_sr=None, mono=True,
                normalize=self._cfg.normalize_audio,
                normalize_mode="rms" if is_neural_engine else "peak",
                preemphasis=self._cfg.preemphasis_audio and not is_neural_engine,
            )
            duration = len(audio.samples) / audio.sample_rate
            self.detail.emit(f"[{self._time()}] ✅ 音频序列加载就绪 | 采样率: {audio.sample_rate}Hz | 总长: {duration:.2f}秒")

            bpm = self._cfg.bpm
            if self._cfg.auto_bpm:
                from audiomidi_app.transcribe import detect_bpm
                bpm = detect_bpm(audio.samples, audio.sample_rate)
                self.detail.emit(f"[{self._time()}] ✅ BPM检测完成: {bpm:.1f}")

            if self._interrupted:
                return ""

            self.progress.emit(f"模型特征解码中 (2/4) - {self._cfg.engine}")
            self.progress_percent.emit(35)

            transcribers = available_transcribers()
            transcriber: Any = None
            for t in transcribers:
                if t.name == self._cfg.engine:
                    transcriber = t
                    break

            if transcriber is None:
                raise RuntimeError(f"未找到指定的算法引擎: {self._cfg.engine}")

            if hasattr(transcriber, '_onset_threshold') and hasattr(transcriber, '_frame_threshold'):
                transcriber._onset_threshold = self._cfg.bp_onset_threshold
                transcriber._frame_threshold = self._cfg.bp_frame_threshold
            if hasattr(transcriber, '_bp') and hasattr(transcriber._bp, '_onset_threshold'):
                transcriber._bp._onset_threshold = self._cfg.bp_onset_threshold
                transcriber._bp._frame_threshold = self._cfg.bp_frame_threshold

            if self._interrupted:
                return ""

            events = transcriber.transcribe(audio.samples, audio.sample_rate)
            self.notes_found.emit(len(events))
            self.detail.emit(f"[{self._time()}] ✅ 原始序列捕获完毕，共测得节点点位: {len(events)}")
            self.progress_percent.emit(50)

            self.progress.emit("执行数据后处理优化 (3/5)")
            self.progress_percent.emit(55)
            from audiomidi_app.postprocess import full_postprocess, PostProcessConfig, OnsetDetector
            pp_config = PostProcessConfig(
                confidence_threshold=self._cfg.confidence_threshold,
                enable_velocity_normalize=self._cfg.velocity_stretch,
            )
            onset_detector = OnsetDetector(audio.sample_rate)
            onset_detector.detect(audio.samples)
            is_neural = self._cfg.engine in ("Piano Transcription (Neural)", "Basic Pitch", "Ensemble (PT + BP)")
            events = full_postprocess(
                events,
                samples=audio.samples,
                sample_rate=audio.sample_rate,
                bpm=bpm,
                onset_detector=onset_detector,
                config=pp_config,
                is_neural=is_neural,
            )
            self.detail.emit(f"[{self._time()}] ✅ 过滤后处理完成 | 有效音符留存: {len(events)}")
            self.progress_percent.emit(60)

            from audiomidi_app.diagnostics import print_transcription_report
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                print_transcription_report(events, duration)
            finally:
                sys.stdout = old_stdout
            for line in buf.getvalue().strip().split("\n"):
                if line:
                    self.detail.emit(f"[{self._time()}] {line}")

            voice_result: VoiceSeparationResult | None = None

            if self._cfg.use_voice_separation:
                self.progress.emit("声部分离中 (3/4)")
                self.progress_percent.emit(70)
                voice_result = separate_voices(events)
                n_voices = len(voice_result.voices) if voice_result else 0
                self.voices_found.emit(n_voices)
                self.detail.emit(f"[{self._time()}] ✅ 声部分离完成 | 音轨数: {n_voices}")
            else:
                self.progress_percent.emit(75)

            self.progress.emit("正在编译导出通用MIDI序列 (4/4)")
            self.progress_percent.emit(85)

            if voice_result and self._cfg.split_hands:
                mid = events_to_midi_with_hands(
                    voice_result,
                    bpm=bpm,
                    left_channel=self._cfg.left_hand_channel,
                    right_channel=self._cfg.right_hand_channel,
                )
            else:
                mid = events_to_midi(events, bpm=bpm)

            mid.save(str(out_path))
            self.progress_percent.emit(95)
            self.detail.emit(f"[{self._time()}] ✅ 编译成功: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
            return str(out_path)

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            # 💡 【原生窗口标题汉化】
            self.setWindowTitle("音频特征转谱工作台 (Audio → MIDI)")
            self.setAcceptDrops(True)
            self._setup_ui()

            self._thread: QThread | None = None
            self._worker: Worker | None = None

        def _setup_ui(self) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            main_layout = QVBoxLayout(root)
            main_layout.setContentsMargins(10, 10, 10, 10)  
            main_layout.setSpacing(8)

            # ── 核心文件IO面板 ───────────────────────────────────────
            file_panel = QWidget()
            file_panel.setObjectName("densePanel")
            file_layout = QFormLayout(file_panel)
            file_layout.setContentsMargins(10, 8, 10, 8)
            file_layout.setVerticalSpacing(6)
            file_layout.setHorizontalSpacing(10)
            file_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            file_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

            self._audio_path = QLineEdit()
            self._audio_path.setPlaceholderText("拖拽音频文件至此，或通过右侧定位...")
            pick_audio = QPushButton("浏览...")
            pick_audio.setFixedWidth(54)
            pick_audio.clicked.connect(self._on_pick_audio)
            row_audio = QHBoxLayout()
            row_audio.setSpacing(4)
            row_audio.addWidget(self._audio_path, 1)
            row_audio.addWidget(pick_audio)
            file_layout.addRow("音频输入", row_audio)

            self._out_path = QLineEdit()
            self._out_path.setPlaceholderText("MIDI 文件存储目录路径")
            pick_out = QPushButton("浏览...")
            pick_out.setFixedWidth(54)
            pick_out.clicked.connect(self._on_pick_out)
            row_out = QHBoxLayout()
            row_out.setSpacing(4)
            row_out.addWidget(self._out_path, 1)
            row_out.addWidget(pick_out)
            file_layout.addRow("输出路径", row_out)

            main_layout.addWidget(file_panel)

            # ── 核心参数设置选项卡 ────────────────────────────────────
            tabs = QTabWidget()
            tabs.setDocumentMode(True)

            # 页面 1: 基础与高级阈值参数
            tab_transcribe = QWidget()
            tab_transcribe.setStyleSheet("background-color: #121212;")
            transcribe_layout = QFormLayout(tab_transcribe)
            transcribe_layout.setContentsMargins(12, 10, 12, 10)
            transcribe_layout.setVerticalSpacing(6)
            transcribe_layout.setHorizontalSpacing(12)
            transcribe_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

            self._engine = QComboBox()
            for t in available_transcribers():
                self._engine.addItem(t.name)
            transcribe_layout.addRow("引擎", self._engine)

            bpm_row = QHBoxLayout()
            bpm_row.setSpacing(8)
            self._bpm = QDoubleSpinBox()
            self._bpm.setRange(30.0, 400.0)
            self._bpm.setValue(120.0)
            # 💡 移除固定死宽度，确保调节框有足够横向空间容纳内容而不会与周围重叠
            self._bpm.setMinimumWidth(75)
            bpm_row.addWidget(self._bpm)
            self._auto_bpm = QCheckBox("自动检测BPM")
            bpm_row.addWidget(self._auto_bpm)
            bpm_row.addStretch()
            transcribe_layout.addRow("速度 (BPM)", bpm_row)

            preprocess_group = QGroupBox("音频预处理控制")
            preprocess_layout = QVBoxLayout()
            preprocess_layout.setContentsMargins(4, 8, 4, 2)
            preprocess_layout.setSpacing(5)
            self._normalize = QCheckBox("激活音量动态增益归一化 (Normalize)")
            self._normalize.setChecked(True)
            preprocess_layout.addWidget(self._normalize)
            self._preemphasis = QCheckBox("开启高频数字预加重滤波器")
            preprocess_layout.addWidget(self._preemphasis)
            preprocess_group.setLayout(preprocess_layout)
            transcribe_layout.addRow(preprocess_group)

            postprocess_group = QGroupBox("后处理参数")
            postprocess_layout = QFormLayout()
            postprocess_layout.setContentsMargins(4, 8, 4, 2)
            postprocess_layout.setVerticalSpacing(5)
            postprocess_layout.setHorizontalSpacing(10)
            postprocess_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            self._velocity_stretch = QCheckBox("自适应触发力度拉伸扩展")
            self._velocity_stretch.setChecked(True)
            postprocess_layout.addRow("", self._velocity_stretch)
            
            self._confidence_threshold = QDoubleSpinBox()
            self._confidence_threshold.setRange(0.0, 1.0)
            self._confidence_threshold.setValue(0.2)
            self._confidence_threshold.setMinimumWidth(65)
            postprocess_layout.addRow("过滤基准置信度", self._confidence_threshold)
            
            self._bp_onset_threshold = QDoubleSpinBox()
            self._bp_onset_threshold.setRange(0.1, 0.9)
            self._bp_onset_threshold.setValue(0.35)
            self._bp_onset_threshold.setMinimumWidth(65)
            postprocess_layout.addRow("起音判定门槛", self._bp_onset_threshold)
            
            self._bp_frame_threshold = QDoubleSpinBox()
            self._bp_frame_threshold.setRange(0.1, 0.9)
            self._bp_frame_threshold.setValue(0.20)
            self._bp_frame_threshold.setMinimumWidth(65)
            postprocess_layout.addRow("音符持续断点", self._bp_frame_threshold)
            postprocess_group.setLayout(postprocess_layout)
            transcribe_layout.addRow(postprocess_group)

            tabs.addTab(tab_transcribe, "参数配置")

            # 页面 2: 声部分离映射
            tab_voice = QWidget()
            tab_voice.setStyleSheet("background-color: #121212;")
            voice_layout = QVBoxLayout(tab_voice)
            voice_layout.setContentsMargins(12, 10, 12, 10)
            voice_layout.setSpacing(8)

            self._use_voice_sep = QCheckBox("启用声部分离")
            self._use_voice_sep.stateChanged.connect(self._on_voice_sep_toggled)
            voice_layout.addWidget(self._use_voice_sep)

            voice_options = QGroupBox("通道矩阵映射")
            voice_options_layout = QFormLayout()
            voice_options_layout.setContentsMargins(4, 8, 4, 2)
            voice_options_layout.setVerticalSpacing(6)
            voice_options_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

            self._split_hands = QCheckBox("执行左右手琴键独立分轨")
            self._split_hands.setEnabled(False)
            self._split_hands.setChecked(True)
            voice_options_layout.addRow("", self._split_hands)

            ch_row = QHBoxLayout()
            ch_row.setSpacing(6)
            self._left_channel = QSpinBox()
            self._left_channel.setRange(1, 16)
            self._left_channel.setValue(1)
            self._left_channel.setEnabled(False)
            self._left_channel.setMinimumWidth(50)
            self._right_channel = QSpinBox()
            self._right_channel.setRange(1, 16)
            self._right_channel.setValue(2)
            self._right_channel.setEnabled(False)
            self._right_channel.setMinimumWidth(50)
            ch_row.addWidget(QLabel("左手通道:"))
            ch_row.addWidget(self._left_channel)
            ch_row.addSpacing(6)
            ch_row.addWidget(QLabel("右手通道:"))
            ch_row.addWidget(self._right_channel)
            ch_row.addStretch()
            voice_options_layout.addRow("通道分配", ch_row)

            voice_options.setLayout(voice_options_layout)
            voice_layout.addWidget(voice_options)
            voice_layout.addStretch()

            tabs.addTab(tab_voice, "声部分离")

            # 页面 3: 云端分布式计算
            tab_cloud = QWidget()
            tab_cloud.setStyleSheet("background-color: #121212;")
            cloud_layout = QFormLayout(tab_cloud)
            cloud_layout.setContentsMargins(12, 10, 12, 10)
            cloud_layout.setVerticalSpacing(6)
            cloud_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

            self._cloud = QCheckBox("优先调度云端分布式算力（失败自动回退本地推理）")
            self._cloud.stateChanged.connect(self._on_cloud_toggled)
            cloud_layout.addRow("", self._cloud)

            self._cloud_url = QLineEdit("http://127.0.0.1:8000")
            self._cloud_url.setEnabled(False)
            cloud_layout.addRow("节点网关地址", self._cloud_url)

            tabs.addTab(tab_cloud, "云端加速")

            # 页面 4: 极简终端日志
            tab_log = QWidget()
            tab_log.setStyleSheet("background-color: #050505;")
            log_layout = QVBoxLayout(tab_log)
            log_layout.setContentsMargins(0, 0, 0, 0)
            log_layout.setSpacing(0)

            log_header = QWidget()
            log_header.setStyleSheet("background-color: #121212; border-bottom: 1px solid #222222;")
            log_header_layout = QHBoxLayout(log_header)
            log_header_layout.setContentsMargins(8, 2, 8, 2)
            log_title = QLabel("实时控制台输出")
            log_title.setStyleSheet("font-size: 10px; font-weight: bold; color: #555555;")
            log_header_layout.addWidget(log_title)
            log_header_layout.addStretch()
            
            self._clear_log_btn = QPushButton("清空")
            self._clear_log_btn.setFixedSize(36, 18)
            self._clear_log_btn.setStyleSheet("font-size: 9px; padding: 0px;")
            self._clear_log_btn.clicked.connect(self._on_clear_log)
            log_header_layout.addWidget(self._clear_log_btn)
            log_layout.addWidget(log_header)

            self._log_text = QTextEdit()
            self._log_text.setReadOnly(True)
            self._log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
            log_layout.addWidget(self._log_text)

            tabs.addTab(tab_log, "实时日志")

            main_layout.addWidget(tabs, 1)

            # ── 底部状态控制面板 ─────────────────────────────────────
            footer = QWidget()
            footer_layout = QVBoxLayout(footer)
            footer_layout.setContentsMargins(0, 2, 0, 0)
            footer_layout.setSpacing(4)

            self._progress = QProgressBar()
            self._progress.setVisible(False)
            footer_layout.addWidget(self._progress)

            actions_row = QHBoxLayout()
            actions_row.setSpacing(6)

            self._run = QPushButton("执行音频转谱编译")
            self._run.setObjectName("runBtn")
            self._run.setFixedHeight(30)
            self._run.clicked.connect(self._on_run)
            actions_row.addWidget(self._run, 1) 

            self._stop = QPushButton("强行终止")
            self._stop.setObjectName("stopBtn")
            self._stop.setEnabled(False)
            self._stop.setFixedWidth(64)
            self._stop.setFixedHeight(30)
            self._stop.clicked.connect(self._on_stop)
            actions_row.addWidget(self._stop)

            self._notes_label = QLabel("捕获音符: -")
            self._notes_label.setObjectName("statsLabel")
            self._voices_label = QLabel("声部分离: -")
            self._voices_label.setObjectName("statsLabel")
            actions_row.addWidget(self._notes_label)
            actions_row.addWidget(self._voices_label)

            footer_layout.addLayout(actions_row)

            self._status = QLabel("控制台就绪")
            self._status.setObjectName("statusLabel")
            footer_layout.addWidget(self._status)

            main_layout.addWidget(footer)

        def _log(self, msg: str) -> None:
            escaped_msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if "✅" in msg:
                html_msg = f"<span style='color: #4ade80;'>{escaped_msg}</span>"
            elif "❌" in msg or "异常" in msg or "失败" in msg or "ERR" in msg:
                html_msg = f"<span style='color: #f87171;'>{escaped_msg}</span>"
            elif "⚠️" in msg:
                html_msg = f"<span style='color: #fbbf24;'>{escaped_msg}</span>"
            elif "====== " in msg:
                html_msg = f"<span style='color: #E03E3E; font-weight: bold;'>{escaped_msg}</span>"
            else:
                html_msg = f"<span style='color: #888888;'>{escaped_msg}</span>"
                
            self._log_text.append(html_msg)
            self._log_text.moveCursor(QTextCursor.MoveOperation.End)
            self._log_text.ensureCursorVisible()

        def _on_clear_log(self) -> None:
            self._log_text.clear()

        def dragEnterEvent(self, event: QDragEnterEvent) -> None:
            if event.mimeData().hasUrls():
                for url in event.mimeData().urls():
                    if url.isLocalFile():
                        path = url.toLocalFile()
                        if Path(path).suffix.lower() in ['.wav', '.flac', '.ogg', '.mp3', '.m4a']:
                            event.acceptProposedAction()
                            return
            event.ignore()

        def dropEvent(self, event: QDropEvent) -> None:
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if Path(path).suffix.lower() in ['.wav', '.flac', '.ogg', '.mp3', '.m4a']:
                        self._audio_path.setText(path)
                        audio_dir = str(Path(path).parent)
                        if not self._out_path.text().strip():
                            self._out_path.setText(audio_dir)
                        return

        def _on_voice_sep_toggled(self, state: int) -> None:
            enabled = state == Qt.Checked.value
            self._split_hands.setEnabled(enabled)
            self._left_channel.setEnabled(enabled and self._split_hands.isChecked())
            self._right_channel.setEnabled(enabled and self._split_hands.isChecked())

        def _on_cloud_toggled(self, state: int) -> None:
            self._cloud_url.setEnabled(state == Qt.Checked.value)

        def _on_pick_audio(self) -> None:
            # 💡 【文件选择对话框文本完全汉化】
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "定位音频源文件", 
                "", 
                "音频格式 (*.wav *.flac *.ogg *.mp3 *.m4a);;全部文件 (*)"
            )
            if path:
                self._audio_path.setText(path)
                if not self._out_path.text().strip():
                    self._out_path.setText(str(Path(path).parent))

        def _on_pick_out(self) -> None:
            # 💡 【文件夹选择对话框文本汉化】
            path = QFileDialog.getExistingDirectory(self, "指定导出目录")
            if path:
                self._out_path.setText(path)

        def _set_ui_running(self, running: bool) -> None:
            self._run.setEnabled(not running)
            self._stop.setEnabled(running)
            self._engine.setEnabled(not running)
            self._bpm.setEnabled(not running)
            self._cloud.setEnabled(not running)
            self._audio_path.setEnabled(not running)
            self._out_path.setEnabled(not running)
            self._use_voice_sep.setEnabled(not running)
            self._progress.setVisible(running)

        def _on_run(self) -> None:
            if self._thread is not None:
                return

            audio = self._audio_path.text().strip()
            outp = self._out_path.text().strip()
            if not audio or not outp or not Path(audio).exists():
                self._status.setText("错误: 输入输出路径配置非法或源文件不存在")
                return

            cfg = JobConfig(
                audio_path=audio,
                out_dir=outp,
                engine=self._engine.currentText(),
                bpm=self._bpm.value(),
                auto_bpm=self._auto_bpm.isChecked(),
                cloud_enabled=self._cloud.isChecked(),
                cloud_base_url=self._cloud_url.text().strip(),
                use_voice_separation=self._use_voice_sep.isChecked(),
                split_hands=self._split_hands.isChecked(),
                left_hand_channel=self._left_channel.value(),
                right_hand_channel=self._right_channel.value(),
                normalize_audio=self._normalize.isChecked(),
                preemphasis_audio=self._preemphasis.isChecked(),
                velocity_stretch=self._velocity_stretch.isChecked(),
                confidence_threshold=self._confidence_threshold.value(),
                bp_onset_threshold=self._bp_onset_threshold.value(),
                bp_frame_threshold=self._bp_frame_threshold.value(),
            )

            self._set_ui_running(True)
            self._status.setText("任务管线正在准备中...")
            self._notes_label.setText("捕获音符: -")
            self._voices_label.setText("声部分离: -")
            self._log_text.clear()
            self._log(f"====== 执行转谱任务: {cfg.engine} | 基准BPM: {cfg.bpm} ======")

            self._thread = QThread()
            self._worker = Worker(cfg)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.progress.connect(self._on_progress)
            self._worker.detail.connect(self._log)
            self._worker.progress_percent.connect(self._progress.setValue)
            self._worker.notes_found.connect(lambda n: self._notes_label.setText(f"捕获音符: {n}"))
            self._worker.voices_found.connect(lambda n: self._voices_label.setText(f"声部分离: {n}"))
            self._worker.done.connect(self._on_done)
            self._worker.failed.connect(self._on_failed)
            self._worker.done.connect(self._thread.quit)
            self._worker.failed.connect(self._thread.quit)
            self._thread.finished.connect(self._cleanup_thread)
            self._thread.start()

        def _on_progress(self, msg: str) -> None:
            self._status.setText(msg)

        def _on_stop(self) -> None:
            if self._worker is not None:
                self._worker.interrupt()
            self._status.setText("进程终止信号已同步发射")
            self._log("用户核心干预：转谱任务被手动终止。")
            if self._thread and self._thread.isRunning():
                self._thread.quit()
                if not self._thread.wait(2000):
                    self._thread.terminate()
                    self._thread.wait()

        def _on_done(self, out_path: str) -> None:
            if out_path:
                self._log("====== 编译管线成功结束 ======")
                self._status.setText("✅ 转谱编译成功")
            else:
                self._status.setText("⏹ 任务已安全中止")

        def _on_failed(self, msg: str) -> None:
            self._log(f"❌ 触发严重运行时异常: {msg}")
            self._status.setText("❌ 运行失败")

        def _cleanup_thread(self) -> None:
            self._set_ui_running(False)
            if self._worker: self._worker.deleteLater()
            if self._thread: self._thread.deleteLater()
            self._thread = None
            self._worker = None

    app = QApplication([])
    _apply_stylesheet(app)
    w = MainWindow()
    w.resize(600, 480) 
    w.show()
    app.exec()


def events_to_midi_with_hands(
    voice_result: VoiceSeparationResult,
    bpm: float = 120.0,
    left_channel: int = 1,
    right_channel: int = 2,
) -> Any:
    try:
        import mido
        from mido import MidiFile, MidiTrack
    except ImportError:
        raise RuntimeError("丢失环境依赖库: mido 未安装")

    mid = MidiFile(type=1, ticks_per_beat=480)
    left_notes = voice_result.get_left_hand_notes()
    right_notes = voice_result.get_right_hand_notes()

    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)
    us_per_beat = 60_000_000 / bpm
    tempo_track.append(mido.MetaMessage('set_tempo', tempo=int(us_per_beat)))
    tempo_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))

    if left_notes:
        left_track = MidiTrack()
        mid.tracks.append(left_track)
        _append_notes_to_track(left_track, left_notes, bpm, channel=left_channel - 1)

    if right_notes:
        right_track = MidiTrack()
        mid.tracks.append(right_track)
        _append_notes_to_track(right_track, right_notes, bpm, channel=right_channel - 1)

    return mid


def _append_notes_to_track(track, events: list[NoteEvent], bpm: float, channel: int) -> None:
    import mido
    sorted_events = sorted(events, key=lambda n: n.start_s)
    ticks_per_beat = 480

    messages = []
    for e in sorted_events:
        on_time = int(e.start_s * (ticks_per_beat * bpm / 60))
        off_time = int(e.end_s * (ticks_per_beat * bpm / 60))
        messages.append((on_time, 'note_on', e.note, e.velocity))
        messages.append((off_time, 'note_off', e.note, e.velocity))

    messages.sort(key=lambda x: x[0])
    last_time = 0
    for time, msg_type, note, velocity in messages:
        dt = time - last_time
        if dt < 0:
            dt = 0
        if msg_type == 'note_on':
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=dt, channel=channel))
        else:
            track.append(mido.Message('note_off', note=note, velocity=0, time=dt, channel=channel))
        last_time = time