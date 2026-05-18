from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from audiomidi_app.audio import read_audio
from audiomidi_app.cloud_client import CloudConfig, transcribe_via_cloud
from audiomidi_app.midi import events_to_midi
from audiomidi_app.transcribe import SpectralPeaksTranscriber, available_transcribers, try_basic_pitch_transcriber

_qt_import_error: Exception | None = None
try:
    from PySide6.QtCore import QObject, Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except Exception as e:
    _qt_import_error = e


@dataclass(frozen=True)
class JobConfig:
    audio_path: str
    out_path: str
    engine: str
    bpm: float
    cloud_enabled: bool
    cloud_base_url: str


def run_app() -> None:
    if _qt_import_error is not None:
        raise RuntimeError(f"桌面UI依赖加载失败：{_qt_import_error}")

    class Worker(QObject):
        progress = Signal(str)
        done = Signal(str)
        failed = Signal(str)

        def __init__(self, cfg: JobConfig) -> None:
            super().__init__()
            self._cfg = cfg

        def run(self) -> None:
            try:
                self.progress.emit("开始转谱")
                out = self._run_impl()
                self.done.emit(out)
            except Exception as e:
                self.failed.emit(str(e))

        def _run_impl(self) -> str:
            audio_path = Path(self._cfg.audio_path)
            out_path = Path(self._cfg.out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if self._cfg.cloud_enabled:
                try:
                    self.progress.emit("调用云端转谱")
                    midi_bytes = transcribe_via_cloud(
                        CloudConfig(base_url=self._cfg.cloud_base_url),
                        audio_path=audio_path,
                        engine=self._cfg.engine,
                        bpm=self._cfg.bpm,
                    )
                    out_path.write_bytes(midi_bytes)
                    return str(out_path)
                except Exception as e:
                    self.progress.emit(f"云端失败，回退本地：{e}")

            if self._cfg.engine == "Basic Pitch":
                self.progress.emit("运行 Basic Pitch")
                bp = try_basic_pitch_transcriber()
                if bp is None or not hasattr(bp, "transcribe_file"):
                    raise RuntimeError("当前环境未安装 basic-pitch 或不兼容")
                midi_path = bp.transcribe_file(str(audio_path), out_dir=str(out_path.parent))
                if Path(midi_path) != out_path:
                    out_path.write_bytes(Path(midi_path).read_bytes())
                return str(out_path)

            self.progress.emit("分析音频")
            audio = read_audio(audio_path, target_sr=None, mono=True)
            self.progress.emit("生成音符事件")
            events = SpectralPeaksTranscriber().transcribe(audio.samples, audio.sample_rate)
            self.progress.emit("写入MIDI")
            mid = events_to_midi(events, bpm=self._cfg.bpm)
            mid.save(str(out_path))
            return str(out_path)

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Audio→MIDI (MVP)")

            self._audio_path = QLineEdit()
            self._audio_path.setReadOnly(True)
            self._pick_audio = QPushButton("选择音频")
            self._pick_audio.clicked.connect(self._on_pick_audio)

            self._out_path = QLineEdit()
            self._out_path.setReadOnly(True)
            self._pick_out = QPushButton("选择输出")
            self._pick_out.clicked.connect(self._on_pick_out)

            self._engine = QComboBox()
            for t in available_transcribers():
                self._engine.addItem(t.name)

            self._bpm = QSpinBox()
            self._bpm.setRange(40, 240)
            self._bpm.setValue(120)

            self._cloud = QCheckBox("云端优先（失败自动回退本地）")
            self._cloud_url = QLineEdit("http://127.0.0.1:8000")

            self._run = QPushButton("开始")
            self._run.clicked.connect(self._on_run)
            self._status = QLabel("就绪")
            self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)

            root = QWidget()
            self.setCentralWidget(root)

            form = QFormLayout()
            row_audio = QHBoxLayout()
            row_audio.addWidget(self._audio_path, 1)
            row_audio.addWidget(self._pick_audio)
            form.addRow("音频", row_audio)

            row_out = QHBoxLayout()
            row_out.addWidget(self._out_path, 1)
            row_out.addWidget(self._pick_out)
            form.addRow("输出MIDI", row_out)

            form.addRow("引擎", self._engine)
            form.addRow("BPM", self._bpm)

            cloud_row = QHBoxLayout()
            cloud_row.addWidget(self._cloud)
            cloud_row.addWidget(self._cloud_url, 1)
            form.addRow("混合模式", cloud_row)

            layout = QVBoxLayout()
            layout.addLayout(form)
            layout.addWidget(self._run)
            layout.addWidget(self._status)
            root.setLayout(layout)

            self._thread: QThread | None = None
            self._worker: Worker | None = None

        def _on_pick_audio(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择音频", "", "Audio (*.wav *.flac *.ogg);;All (*)"
            )
            if not path:
                return
            self._audio_path.setText(path)
            if not self._out_path.text():
                self._out_path.setText(str(Path(path).with_suffix(".mid")))

        def _on_pick_out(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "选择输出MIDI", "", "MIDI (*.mid)")
            if not path:
                return
            if not path.lower().endswith(".mid"):
                path += ".mid"
            self._out_path.setText(path)

        def _on_run(self) -> None:
            if self._thread is not None:
                return

            audio = self._audio_path.text().strip()
            outp = self._out_path.text().strip()
            if not audio or not outp:
                self._status.setText("请选择音频与输出路径")
                return

            cfg = JobConfig(
                audio_path=audio,
                out_path=outp,
                engine=self._engine.currentText(),
                bpm=float(self._bpm.value()),
                cloud_enabled=self._cloud.isChecked(),
                cloud_base_url=self._cloud_url.text().strip(),
            )

            self._run.setEnabled(False)
            self._status.setText("排队中")

            self._thread = QThread()
            self._worker = Worker(cfg)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.progress.connect(self._status.setText)
            self._worker.done.connect(self._on_done)
            self._worker.failed.connect(self._on_failed)
            self._worker.done.connect(self._thread.quit)
            self._worker.failed.connect(self._thread.quit)
            self._thread.finished.connect(self._cleanup_thread)
            self._thread.start()

        def _on_done(self, out_path: str) -> None:
            self._status.setText(f"完成：{out_path}")

        def _on_failed(self, msg: str) -> None:
            self._status.setText(f"失败：{msg}")

        def _cleanup_thread(self) -> None:
            self._run.setEnabled(True)
            if self._worker is not None:
                self._worker.deleteLater()
            if self._thread is not None:
                self._thread.deleteLater()
            self._thread = None
            self._worker = None

    app = QApplication([])
    w = MainWindow()
    w.resize(760, 260)
    w.show()
    app.exec()
