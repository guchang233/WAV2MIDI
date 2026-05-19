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
    from PySide6.QtCore import QObject, Qt, QThread, Signal, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
        QGroupBox,
        QScrollArea,
        QTextEdit,
    )
    from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont, QTextCursor
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
            self.detail.emit(f"[{self._time()}] [stdout] {line}")

        def run(self) -> None:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stream = _SignalStream(self._emit_stdout)
            sys.stdout = stream
            sys.stderr = stream
            try:
                self.progress.emit("🚀 开始转谱")
                self.progress_percent.emit(0)
                self.detail.emit(f"[{self._time()}] 任务初始化...")
                self.detail.emit(f"[{self._time()}] 引擎: {self._cfg.engine}")
                if self._cfg.auto_bpm:
                    self.detail.emit(f"[{self._time()}] BPM: 自动检测")
                else:
                    self.detail.emit(f"[{self._time()}] BPM: {self._cfg.bpm}")
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
            
            self.detail.emit(f"[{self._time()}] 📁 输入文件: {audio_path}")
            self.detail.emit(f"[{self._time()}] 📁 输出目录: {out_dir}")
            self.detail.emit(f"[{self._time()}] 🎯 输出文件: {out_path.name}")
            self.progress_percent.emit(5)

            if self._interrupted:
                return ""

            if self._cfg.cloud_enabled:
                try:
                    self.progress.emit("☁️ 调用云端转谱")
                    self.detail.emit(f"[{self._time()}] 连接云端: {self._cfg.cloud_base_url}")
                    midi_bytes = transcribe_via_cloud(
                        CloudConfig(base_url=self._cfg.cloud_base_url),
                        audio_path=audio_path,
                        engine=self._cfg.engine,
                        bpm=self._cfg.bpm,
                    )
                    out_path.write_bytes(midi_bytes)
                    self.detail.emit(f"[{self._time()}] ✅ 云端完成，文件已保存")
                    return str(out_path)
                except Exception as e:
                    self.detail.emit(f"[{self._time()}] ⚠️ 云端失败: {e}")
                    self.progress.emit(f"⚠️ 云端失败，回退本地")
                    self.detail.emit(f"[{self._time()}] 回退到本地引擎...")

            if self._interrupted:
                return ""

            self.progress.emit("📊 分析音频 (1/4)")
            self.progress_percent.emit(15)
            self.detail.emit(f"[{self._time()}] 读取音频文件...")
            audio = read_audio(audio_path, target_sr=None, mono=True)
            duration = len(audio.samples) / audio.sample_rate
            self.detail.emit(f"[{self._time()}] ✅ 音频加载完成")
            self.detail.emit(f"[{self._time()}]    采样率: {audio.sample_rate} Hz")
            self.detail.emit(f"[{self._time()}]    时长: {duration:.2f} 秒")
            self.detail.emit(f"[{self._time()}]    采样数: {len(audio.samples):,}")

            if self._cfg.auto_bpm:
                self.detail.emit(f"[{self._time()}] 自动检测 BPM...")
                from audiomidi_app.transcribe import detect_bpm
                detected = detect_bpm(audio.samples, audio.sample_rate)
                self.detail.emit(f"[{self._time()}] ✅ BPM 检测完成: {detected:.1f}")

            if self._interrupted:
                return ""

            self.progress.emit(f"🎵 生成音符 (2/4) - {self._cfg.engine}")
            self.progress_percent.emit(35)
            self.detail.emit(f"[{self._time()}] 启动引擎: {self._cfg.engine}")
            
            # 使用传统引擎
            transcribers = available_transcribers()
            transcriber: Any = None
            for t in transcribers:
                if t.name == self._cfg.engine:
                    transcriber = t
                    break
            
            if transcriber is None:
                raise RuntimeError(f"找不到引擎：{self._cfg.engine}")

            if self._interrupted:
                return ""

            self.detail.emit(f"[{self._time()}] 正在分析音频特征...")
            events = transcriber.transcribe(audio.samples, audio.sample_rate)
            self.notes_found.emit(len(events))
            self.detail.emit(f"[{self._time()}] ✅ 音符检测完成")
            self.detail.emit(f"[{self._time()}]    检测到 {len(events)} 个音符")
            if events:
                avg_vel = sum(e.velocity for e in events) / len(events)
                self.detail.emit(f"[{self._time()}]    平均力度: {avg_vel:.1f}")
                duration_range = max(e.end_s for e in events) - min(e.start_s for e in events)
                self.detail.emit(f"[{self._time()}]    音符跨度: {duration_range:.2f} 秒")
            self.progress_percent.emit(50)

            # 后处理：onset精调、重复音合并、velocity归一化、平滑等
            self.progress.emit("🔧 后处理 (3/5)")
            self.progress_percent.emit(55)
            self.detail.emit(f"[{self._time()}] 后处理中...")
            from audiomidi_app.postprocess import full_postprocess, PostProcessConfig, OnsetDetector
            onset_detector = OnsetDetector(audio.sample_rate)
            onset_detector.detect(audio.samples)
            events = full_postprocess(
                events,
                samples=audio.samples,
                sample_rate=audio.sample_rate,
                bpm=self._cfg.bpm,
                onset_detector=onset_detector,
            )
            self.detail.emit(f"[{self._time()}] ✅ 后处理完成")
            self.detail.emit(f"[{self._time()}]    剩余 {len(events)} 个音符")
            self.progress_percent.emit(60)

            voice_result: VoiceSeparationResult | None = None

            if self._cfg.use_voice_separation:
                self.progress.emit("🎤 声部分离 (3/4)")
                self.progress_percent.emit(70)
                self.detail.emit(f"[{self._time()}] 开始声部分离...")
                voice_result = separate_voices(events)
                n_voices = len(voice_result.voices) if voice_result else 0
                self.voices_found.emit(n_voices)
                self.detail.emit(f"[{self._time()}] ✅ 声部分离完成")
                self.detail.emit(f"[{self._time()}]    分离出 {n_voices} 个声部")
                
                left_notes = voice_result.get_left_hand_notes() if voice_result else []
                right_notes = voice_result.get_right_hand_notes() if voice_result else []
                self.detail.emit(f"[{self._time()}]    左手: {len(left_notes)} 音符")
                self.detail.emit(f"[{self._time()}]    右手: {len(right_notes)} 音符")
            else:
                self.progress_percent.emit(75)

            self.progress.emit("💾 写入MIDI (4/4)")
            self.progress_percent.emit(85)
            self.detail.emit(f"[{self._time()}] 生成 MIDI 文件...")
            
            if voice_result and self._cfg.split_hands:
                self.detail.emit(f"[{self._time()}] 左右手分轨输出")
                self.detail.emit(f"[{self._time()}]    左手 Channel: {self._cfg.left_hand_channel}")
                self.detail.emit(f"[{self._time()}]    右手 Channel: {self._cfg.right_hand_channel}")
                mid = events_to_midi_with_hands(
                    voice_result, 
                    bpm=self._cfg.bpm,
                    left_channel=self._cfg.left_hand_channel,
                    right_channel=self._cfg.right_hand_channel,
                )
            else:
                mid = events_to_midi(events, bpm=self._cfg.bpm)
            
            mid.save(str(out_path))
            file_size = out_path.stat().st_size
            self.progress_percent.emit(95)
            self.detail.emit(f"[{self._time()}] ✅ MIDI 已保存")
            self.detail.emit(f"[{self._time()}]    文件: {out_path.name}")
            self.detail.emit(f"[{self._time()}]    大小: {file_size / 1024:.1f} KB")
            self.detail.emit(f"[{self._time()}]    路径: {out_path}")
            return str(out_path)

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Audio → MIDI (高级版)")
            self.setAcceptDrops(True)
            self._setup_ui()

            self._thread: QThread | None = None
            self._worker: Worker | None = None

        def _setup_ui(self) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            main_layout = QVBoxLayout(root)

            # 顶部：文件选择区域
            file_group = QGroupBox("文件")
            file_layout = QFormLayout()

            self._audio_path = QLineEdit()
            self._audio_path.setPlaceholderText("拖拽音频文件到此处或点击选择")
            pick_audio = QPushButton("选择音频")
            pick_audio.clicked.connect(self._on_pick_audio)
            row_audio = QHBoxLayout()
            row_audio.addWidget(self._audio_path, 1)
            row_audio.addWidget(pick_audio)
            file_layout.addRow("音频输入", row_audio)

            self._out_path = QLineEdit()
            self._out_path.setPlaceholderText("选择MIDI输出文件夹")
            pick_out = QPushButton("选择文件夹")
            pick_out.clicked.connect(self._on_pick_out)
            row_out = QHBoxLayout()
            row_out.addWidget(self._out_path, 1)
            row_out.addWidget(pick_out)
            file_layout.addRow("输出文件夹", row_out)

            file_group.setLayout(file_layout)
            main_layout.addWidget(file_group)

            # 标签页
            tabs = QTabWidget()

            # 转谱设置
            tab_transcribe = QWidget()
            transcribe_layout = QFormLayout(tab_transcribe)

            self._engine = QComboBox()
            for t in available_transcribers():
                self._engine.addItem(t.name)
            self._engine.currentIndexChanged.connect(self._on_engine_changed)
            transcribe_layout.addRow("转谱引擎", self._engine)

            self._bpm = QDoubleSpinBox()
            self._bpm.setRange(30.0, 400.0)
            self._bpm.setSingleStep(1.0)
            self._bpm.setDecimals(2)
            self._bpm.setValue(120.0)
            transcribe_layout.addRow("BPM", self._bpm)

            self._auto_bpm = QCheckBox("自动检测BPM")
            transcribe_layout.addRow("", self._auto_bpm)

            tabs.addTab(tab_transcribe, "转谱")

            # 声部分离
            tab_voice = QWidget()
            voice_layout = QVBoxLayout(tab_voice)

            self._use_voice_sep = QCheckBox("启用声部分离")
            self._use_voice_sep.setChecked(False)
            self._use_voice_sep.stateChanged.connect(self._on_voice_sep_toggled)
            voice_layout.addWidget(self._use_voice_sep)

            voice_options = QGroupBox("声部分离选项")
            voice_options_layout = QFormLayout()
            voice_options_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

            self._split_hands = QCheckBox("左右手分离")
            self._split_hands.setEnabled(False)
            self._split_hands.setChecked(True)
            voice_options_layout.addRow("", self._split_hands)

            self._left_channel = QSpinBox()
            self._left_channel.setRange(1, 16)
            self._left_channel.setValue(1)
            self._left_channel.setEnabled(False)
            voice_options_layout.addRow("左手Channel", self._left_channel)

            self._right_channel = QSpinBox()
            self._right_channel.setRange(1, 16)
            self._right_channel.setValue(2)
            self._right_channel.setEnabled(False)
            voice_options_layout.addRow("右手Channel", self._right_channel)

            voice_options.setLayout(voice_options_layout)
            voice_layout.addWidget(voice_options)
            voice_layout.addStretch()

            tabs.addTab(tab_voice, "声部分离")

            # 云端选项
            tab_cloud = QWidget()
            cloud_layout = QFormLayout(tab_cloud)

            self._cloud = QCheckBox("云端优先（失败自动回退本地）")
            self._cloud.stateChanged.connect(self._on_cloud_toggled)
            cloud_layout.addRow("", self._cloud)

            self._cloud_url = QLineEdit("http://127.0.0.1:8000")
            self._cloud_url.setEnabled(False)
            cloud_layout.addRow("云端地址", self._cloud_url)

            tabs.addTab(tab_cloud, "云端")

            # 日志页面
            tab_log = QWidget()
            log_layout = QVBoxLayout(tab_log)
            
            log_header = QHBoxLayout()
            log_header.addWidget(QLabel("运行日志"))
            self._clear_log_btn = QPushButton("清空")
            self._clear_log_btn.setMaximumWidth(60)
            self._clear_log_btn.clicked.connect(self._on_clear_log)
            log_header.addWidget(self._clear_log_btn)
            log_header.addStretch()
            log_layout.addLayout(log_header)
            
            self._log_text = QTextEdit()
            self._log_text.setReadOnly(True)
            self._log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
            log_layout.addWidget(self._log_text)
            
            tabs.addTab(tab_log, "📋 日志")

            main_layout.addWidget(tabs, 1)

            # 按钮
            btn_row = QHBoxLayout()
            self._run = QPushButton("开始转谱")
            self._run.setMinimumHeight(40)
            font = self._run.font()
            font.setPointSize(11)
            font.setBold(True)
            self._run.setFont(font)
            self._run.clicked.connect(self._on_run)
            btn_row.addWidget(self._run)

            self._stop = QPushButton("停止")
            self._stop.setEnabled(False)
            self._stop.setMinimumHeight(40)
            self._stop.clicked.connect(self._on_stop)
            btn_row.addWidget(self._stop)
            main_layout.addLayout(btn_row)

            # 进度
            self._progress = QProgressBar()
            self._progress.setVisible(False)
            self._progress.setRange(0, 100)
            self._progress.setTextVisible(True)
            self._progress.setFormat("%p% - %v/100")
            main_layout.addWidget(self._progress)

            # 统计栏
            stats_layout = QHBoxLayout()
            self._notes_label = QLabel("音符: -")
            self._voices_label = QLabel("声部: -")
            stats_layout.addWidget(self._notes_label)
            stats_layout.addWidget(self._voices_label)
            stats_layout.addStretch()
            main_layout.addLayout(stats_layout)

            # 状态栏
            status_group = QGroupBox("状态")
            status_layout = QVBoxLayout()

            self._status = QLabel("就绪")
            self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
            status_layout.addWidget(self._status)

            status_group.setLayout(status_layout)
            main_layout.addWidget(status_group)

        def _log(self, msg: str) -> None:
            self._log_text.append(msg)
            self._log_text.moveCursor(QTextCursor.MoveOperation.End)
            self._log_text.ensureCursorVisible()

        def _on_clear_log(self) -> None:
            self._log_text.clear()

        def dragEnterEvent(self, event: QDragEnterEvent) -> None:
            if event.mimeData().hasUrls():
                for url in event.mimeData().urls():
                    if url.isLocalFile():
                        path = url.toLocalFile()
                        ext = Path(path).suffix.lower()
                        if ext in ['.wav', '.flac', '.ogg', '.mp3', '.m4a']:
                            event.acceptProposedAction()
                            return
            event.ignore()

        def dropEvent(self, event: QDropEvent) -> None:
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    ext = Path(path).suffix.lower()
                    if ext in ['.wav', '.flac', '.ogg', '.mp3', '.m4a']:
                        self._audio_path.setText(path)
                        # 自动设置输出目录为音频文件所在目录
                        audio_dir = str(Path(path).parent)
                        if not self._out_path.text().strip():
                            self._out_path.setText(audio_dir)
                            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] 自动设置输出目录: {audio_dir}")
                        return

        def _on_engine_changed(self, idx: int) -> None:
            pass

        def _on_voice_sep_toggled(self, state: int) -> None:
            enabled = state == Qt.Checked.value
            self._split_hands.setEnabled(enabled)
            self._left_channel.setEnabled(enabled and self._split_hands.isChecked())
            self._right_channel.setEnabled(enabled and self._split_hands.isChecked())

        def _on_cloud_toggled(self, state: int) -> None:
            self._cloud_url.setEnabled(state == Qt.Checked.value)

        def _on_pick_audio(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "选择音频", 
                "", 
                "Audio (*.wav *.flac *.ogg *.mp3 *.m4a);;All (*)"
            )
            if not path:
                return
            self._audio_path.setText(path)
            # 自动设置输出目录为音频文件所在目录
            audio_dir = str(Path(path).parent)
            if not self._out_path.text().strip():
                self._out_path.setText(audio_dir)
                self._log(f"[{datetime.now().strftime('%H:%M:%S')}] 自动设置输出目录: {audio_dir}")

        def _on_pick_out(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
            if not path:
                return
            self._out_path.setText(path)

        def _set_ui_running(self, running: bool) -> None:
            self._run.setEnabled(not running)
            self._stop.setEnabled(running)
            self._engine.setEnabled(not running)
            self._bpm.setEnabled(not running)
            self._cloud.setEnabled(not running)
            self._cloud_url.setEnabled(not running and self._cloud.isChecked())
            self._audio_path.setEnabled(not running)
            self._out_path.setEnabled(not running)
            self._use_voice_sep.setEnabled(not running)
            self._split_hands.setEnabled(not running and self._use_voice_sep.isChecked())
            self._left_channel.setEnabled(not running and self._use_voice_sep.isChecked() and self._split_hands.isChecked())
            self._right_channel.setEnabled(not running and self._use_voice_sep.isChecked() and self._split_hands.isChecked())
            self._progress.setVisible(running)

        def _on_run(self) -> None:
            if self._thread is not None:
                return

            audio = self._audio_path.text().strip()
            outp = self._out_path.text().strip()
            if not audio or not outp:
                self._status.setText("请选择音频与输出路径")
                return

            if not Path(audio).exists():
                self._status.setText("音频文件不存在")
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
            )

            self._set_ui_running(True)
            self._status.setText("准备中...")
            self._notes_label.setText("音符: -")
            self._voices_label.setText("声部: -")
            self._log_text.clear()
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] ====== 开始转谱任务 ======")
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] 引擎: {cfg.engine}, BPM: {cfg.bpm}")

            self._thread = QThread()
            self._worker = Worker(cfg)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.progress.connect(self._on_progress)
            self._worker.detail.connect(self._log)
            self._worker.progress_percent.connect(self._progress.setValue)
            self._worker.notes_found.connect(lambda n: self._notes_label.setText(f"音符: {n}"))
            self._worker.voices_found.connect(lambda n: self._voices_label.setText(f"声部: {n}"))
            self._worker.done.connect(self._on_done)
            self._worker.failed.connect(self._on_failed)
            self._worker.done.connect(self._thread.quit)
            self._worker.failed.connect(self._thread.quit)
            self._thread.finished.connect(self._cleanup_thread)
            self._thread.start()

        def _on_progress(self, msg: str) -> None:
            self._status.setText(msg)
            self._log(msg)

        def _on_stop(self) -> None:
            if self._worker is not None:
                self._worker.interrupt()
            self._status.setText("正在停止...")
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] 用户请求停止...")

            if self._thread is not None and self._thread.isRunning():
                self._thread.quit()
                if not self._thread.wait(2000):
                    self._thread.terminate()
                    self._thread.wait()

        def _on_done(self, out_path: str) -> None:
            if out_path:
                self._log(f"[{datetime.now().strftime('%H:%M:%S')}] ====== 任务完成 ======")
                self._status.setText(f"✅ 完成：{out_path}")
            else:
                self._log(f"[{datetime.now().strftime('%H:%M:%S')}] 任务已停止")
                self._status.setText("⏹ 已停止")

        def _on_failed(self, msg: str) -> None:
            self._log(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 错误: {msg}")
            self._status.setText(f"❌ 失败：{msg}")

        def _cleanup_thread(self) -> None:
            self._set_ui_running(False)
            
            if self._worker is not None:
                self._worker.deleteLater()
            if self._thread is not None:
                self._thread.deleteLater()
            self._thread = None
            self._worker = None

    app = QApplication([])
    w = MainWindow()
    w.resize(800, 600)
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
        raise RuntimeError("mido 未安装")

    # Type 1 MIDI 支持多轨
    mid = MidiFile(type=1, ticks_per_beat=480)

    left_notes = voice_result.get_left_hand_notes()
    right_notes = voice_result.get_right_hand_notes()

    # Track 0: Tempo track（只包含 set_tempo 和 time_signature）
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
