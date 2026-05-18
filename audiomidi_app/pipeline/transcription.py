"""
Multi-Stage Transcription Engine
现代神经网络转录引擎，包含 MT3、Piano Transcription 等
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import tempfile
from pathlib import Path

from audiomidi_app.midi import NoteEvent


@dataclass(frozen=True)
class TranscriptionConfig:
    """转录配置"""
    use_separation: bool = True
    use_mt3: bool = True
    use_piano_transcription: bool = True
    use_basic_pitch: bool = True
    enable_refinement: bool = True
    confidence_threshold: float = 0.3


@dataclass
class TranscriptionResult:
    """转录结果"""
    notes: list[NoteEvent]
    confidence_scores: list[float]
    tempo: float
    beat_positions: list[float]
    stem_used: str


class MT3TranscriptionEngine:
    """MT3 (Music Transcription Transformer) - Google 的现代转录模型"""
    
    name = "MT3 (Music Transcription Transformer)"
    
    def __init__(self):
        self._model = None
        self._sp = None
    
    def _load_model(self):
        """懒加载 MT3 模型"""
        if self._model is not None:
            return
        
        try:
            import mt3
            import note_seq
            
            self._model = mt3
            self._sp = note_seq
        except ImportError:
            raise ImportError(
                "MT3 未安装。请运行: pip install mt3 note-seq"
            )
    
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
        """使用 MT3 转录"""
        notes_with_confidence = self.transcribe_with_confidence(samples, sample_rate)
        return [note for note, _ in notes_with_confidence]
    
    def transcribe_with_confidence(
        self, samples: np.ndarray, sample_rate: int
    ) -> list[tuple[NoteEvent, float]]:
        """MT3 转录，返回置信度"""
        if self._model is None:
            self._load_model()
        
        try:
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            sf.write(temp_path, samples, sample_rate)
            
            try:
                sequence = self._sp.audio_io_to_sequence(
                    temp_path,
                    sample_rate=sample_rate,
                    codec=self._sp.CODEC_FLAC,
                    steps_per_second=50,
                )
                
                notes = []
                for note in sequence.notes:
                    if note.program == 0:
                        note_event = NoteEvent(
                            note=note.pitch,
                            start_s=note.start_time,
                            end_s=note.end_time,
                            velocity=int(np.clip(note.velocity * 127, 1, 127)),
                            confidence=0.9,
                        )
                        notes.append((note_event, 0.9))
                
                return notes
                
            finally:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"MT3 transcription failed: {e}")
            return []


class PianoTranscriptionEngine:
    """钢琴专用转录引擎 - Piano Transcription Inference"""
    
    name = "Piano Transcription (Neural)"
    
    def __init__(self):
        self._model = None
    
    def _load_model(self):
        """懒加载 Piano Transcription 模型"""
        if self._model is not None:
            return
        
        try:
            from piano_transcription_inference import PianoTranscription
            self._model = PianoTranscription(device="cpu", duration=None)
        except ImportError:
            raise ImportError(
                "Piano Transcription 未安装。请运行: pip install piano-transcription-inference"
            )
    
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
        """使用 Piano Transcription 转录"""
        notes_with_confidence = self.transcribe_with_confidence(samples, sample_rate)
        return [note for note, _ in notes_with_confidence]
    
    def transcribe_with_confidence(
        self, samples: np.ndarray, sample_rate: int
    ) -> list[tuple[NoteEvent, float]]:
        """Piano Transcription 转录"""
        if self._model is None:
            self._load_model()
        
        try:
            import soundfile as sf
            from piano_transcription_inference import sample_rate as PT_SR
            
            if sample_rate != PT_SR:
                g = np.gcd(sample_rate, PT_SR)
                samples = np resample_poly(
                    samples,
                    PT_SR // g,
                    sample_rate // g
                ).astype(np.float32)
                sample_rate = PT_SR
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            sf.write(temp_path, samples, sample_rate)
            
            try:
                result = self._model.transcribe(temp_path, midi_path=None)
                
                notes = []
                for note_info in result["notes"]:
                    note_event = NoteEvent(
                        note=int(note_info["midi_note"]),
                        start_s=float(note_info["onset_time"]),
                        end_s=float(note_info["offset_time"]),
                        velocity=int(np.clip(note_info["velocity"] * 127, 1, 127)),
                        confidence=1.0,
                    )
                    notes.append((note_event, 1.0))
                
                return notes
                
            finally:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"Piano Transcription failed: {e}")
            return []


class BasicPitchTranscriptionEngine:
    """Basic Pitch 转录引擎 - Spotify 的模型，适合 vocals/monophonic"""
    
    name = "Basic Pitch"
    
    def __init__(self):
        self._model_path = None
    
    def _load_model(self):
        """懒加载 Basic Pitch 模型"""
        if self._model_path is not None:
            return
        
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
            self._model_path = ICASSP_2022_MODEL_PATH
            self._predict = predict
        except ImportError:
            raise ImportError(
                "Basic Pitch 未安装。请运行: pip install basic-pitch"
            )
    
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
        """使用 Basic Pitch 转录"""
        notes_with_confidence = self.transcribe_with_confidence(samples, sample_rate)
        return [note for note, _ in notes_with_confidence]
    
    def transcribe_with_confidence(
        self, samples: np.ndarray, sample_rate: int
    ) -> list[tuple[NoteEvent, float]]:
        """Basic Pitch 转录"""
        if self._model_path is None:
            self._load_model()
        
        try:
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            sf.write(temp_path, samples, sample_rate)
            
            try:
                _, _, note_events = self._predict(temp_path, self._model_path)
                
                notes = []
                for note in note_events:
                    note_event = NoteEvent(
                        note=int(note.pitch),
                        start_s=float(note.start_time),
                        end_s=float(note.end_time),
                        velocity=int(min(127, max(1, note.amplitude * 127))),
                        confidence=0.7,
                    )
                    notes.append((note_event, 0.7))
                
                return notes
                
            finally:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"Basic Pitch transcription failed: {e}")
            return []


class DrumsTranscriptionEngine:
    """鼓组专用转录引擎"""
    
    name = "Drums Transcription"
    
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
        """转录鼓组"""
        notes_with_confidence = self.transcribe_with_confidence(samples, sample_rate)
        return [note for note, _ in notes_with_confidence]
    
    def transcribe_with_confidence(
        self, samples: np.ndarray, sample_rate: int
    ) -> list[tuple[NoteEvent, float]]:
        """鼓组转录"""
        try:
            import librosa
            
            drum_notes = {
                36: "Kick",
                38: "Snare", 
                42: "Closed Hi-Hat",
                46: "Open Hi-Hat",
                49: "Crash",
                51: "Ride",
            }
            
            onsets = librosa.onset.onset_detect(
                y=samples,
                sr=sample_rate,
                hop_length=256,
                backtrack=True,
                units="time"
            )
            
            notes = []
            for t in onsets:
                gm_note = 38
                note_event = NoteEvent(
                    note=gm_note,
                    start_s=float(t),
                    end_s=float(t) + 0.1,
                    velocity=90,
                    confidence=0.8,
                )
                notes.append((note_event, 0.8))
            
            return notes
            
        except Exception as e:
            print(f"Drums transcription failed: {e}")
            return []


class EnsembleTranscriptionEngine:
    """集成多个转录引擎，提高准确性"""
    
    name = "Ensemble Transcription"
    
    def __init__(self, config: TranscriptionConfig | None = None):
        self._cfg = config or TranscriptionConfig()
        self._engines: list[TranscriptionEngine] = []
        self._initialize_engines()
    
    def _initialize_engines(self):
        """初始化所有可用引擎"""
        if self._cfg.use_mt3:
            try:
                self._engines.append(MT3TranscriptionEngine())
            except ImportError:
                pass
        
        if self._cfg.use_piano_transcription:
            try:
                self._engines.append(PianoTranscriptionEngine())
            except ImportError:
                pass
        
        if self._cfg.use_basic_pitch:
            try:
                self._engines.append(BasicPitchTranscriptionEngine())
            except ImportError:
                pass
    
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
        """集成转录"""
        notes_with_confidence = self.transcribe_with_confidence(samples, sample_rate)
        return [note for note, _ in notes_with_confidence]
    
    def transcribe_with_confidence(
        self, samples: np.ndarray, sample_rate: int
    ) -> list[tuple[NoteEvent, float]]:
        """多引擎置信度融合"""
        all_results: list[tuple[NoteEvent, float]] = []
        
        for engine in self._engines:
            try:
                results = engine.transcribe_with_confidence(samples, sample_rate)
                all_results.extend(results)
            except Exception as e:
                print(f"Engine {engine.name} failed: {e}")
                continue
        
        fused = self._fuse_notes(all_results)
        return fused
    
    def _fuse_notes(
        self, results: list[tuple[NoteEvent, float]]
    ) -> list[tuple[NoteEvent, float]]:
        """融合多个引擎的结果"""
        if not results:
            return []
        
        from scipy.cluster.hierarchy import fclusterdata
        
        notes_by_cluster: dict[int, list[tuple[NoteEvent, float]]] = {}
        
        for note, confidence in results:
            cluster_id = int(note.note * 100 + note.start_s * 10)
            notes_by_cluster.setdefault(cluster_id, []).append((note, confidence))
        
        fused: list[tuple[NoteEvent, float]] = []
        
        for cluster_notes in notes_by_cluster.values():
            if not cluster_notes:
                continue
            
            best_note, best_confidence = max(
                cluster_notes, key=lambda x: x[1]
            )
            
            fused.append((best_note, best_confidence))
        
        fused.sort(key=lambda x: (x[0].start_s, x[0].note))
        return fused
