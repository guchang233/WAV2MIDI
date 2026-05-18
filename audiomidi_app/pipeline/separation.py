"""
Stem Separation Module
使用 Demucs/HTDemucs 进行音频源分离
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class SeparationConfig:
    model_name: str = "htdemucs_ft"
    segment: Optional[float] = None
    shifts: int = 1
    overlap: float = 0.25


@dataclass(frozen=True)
class SeparatedStems:
    vocals: Optional[np.ndarray] = None
    drums: Optional[np.ndarray] = None
    bass: Optional[np.ndarray] = None
    other: Optional[np.ndarray] = None
    mixture: Optional[np.ndarray] = None
    sample_rate: int = 44100


class StemSeparator:
    """使用 Demucs 进行音频源分离"""
    
    def __init__(self, config: SeparationConfig | None = None):
        self._cfg = config or SeparationConfig()
        self._separator = None
        self._models_cache: dict[str, any] = {}
    
    def _load_separator(self):
        """懒加载 Demucs 模型"""
        if self._separator is not None:
            return
        
        try:
            from demucs.pretrained import get_model
            from demucs.separate import separator_io
            self._separator = get_model(self._cfg.model_name)
            self._separator.eval()
        except ImportError:
            raise ImportError(
                "Demucs 未安装。请运行: pip install demucs"
            )
    
    def separate(self, samples: np.ndarray, sample_rate: int = 44100) -> SeparatedStems:
        """
        分离音频到不同 stem
        
        Args:
            samples: 音频样本，形状为 (channels, samples) 或 (samples,)
            sample_rate: 采样率
        
        Returns:
            SeparatedStems: 包含各个分离音轨的对象
        """
        if self._separator is None:
            self._load_separator()
        
        if samples.ndim == 1:
            samples = np.stack([samples, samples])
        elif samples.ndim == 2 and samples.shape[0] == 1:
            samples = np.repeat(samples, 2, axis=0)
        
        try:
            from demucs.separate import separate_sources
            
            sources = separate_sources(
                self._separator,
                samples,
                shifts=self._cfg.shifts,
                segment=self._cfg.segment,
                overlap=self._cfg.overlap,
            )
            
            sources = sources[0]
            
            sources_dict = {
                "other": sources[0].numpy(),
                "drums": sources[1].numpy(),
                "bass": sources[2].numpy(),
                "vocals": sources[3].numpy() if sources.shape[0] > 3 else None,
            }
            
            if sources.shape[0] == 4:
                sources_dict["other"] = sources[0].numpy()
                sources_dict["drums"] = sources[1].numpy()
                sources_dict["bass"] = sources[2].numpy()
                sources_dict["vocals"] = sources[3].numpy()
            elif sources.shape[0] == 5:
                sources_dict["vocals"] = sources[0].numpy()
                sources_dict["drums"] = sources[1].numpy()
                sources_dict["bass"] = sources[2].numpy()
                sources_dict["other"] = sources[3].numpy() + sources[4].numpy()
            
            stems = SeparatedStems(
                vocals=sources_dict.get("vocals"),
                drums=sources_dict.get("drums"),
                bass=sources_dict.get("bass"),
                other=sources_dict.get("other"),
                mixture=samples.numpy() if hasattr(samples, 'numpy') else samples,
                sample_rate=sample_rate,
            )
            
            return stems
            
        except Exception as e:
            raise RuntimeError(f"Stem separation failed: {e}")
    
    def separate_simple(self, samples: np.ndarray, sample_rate: int = 44100) -> SeparatedStems:
        """
        简化版分离：如果 Demucs 不可用，返回原始音频
        
        Returns:
            只有 mixture 填充，其他为 None
        """
        if samples.ndim == 1:
            mixture = np.stack([samples, samples])
        else:
            mixture = samples
        
        return SeparatedStems(
            vocals=None,
            drums=None,
            bass=None,
            other=mixture,
            mixture=mixture,
            sample_rate=sample_rate,
        )


class HTDemucsSeparator(StemSeparator):
    """HTDemucs 专用分离器 - 推荐用于流行音乐"""
    
    def __init__(self):
        super().__init__(SeparationConfig(model_name="htdemucs_ft"))
    
    def separate(self, samples: np.ndarray, sample_rate: int = 44100) -> SeparatedStems:
        """HTDemucs FT 分离"""
        return super().separate(samples, sample_rate)


class HTDemucs6SSeparator(StemSeparator):
    """HTDemucs 6源 分离器 - 分离更细"""
    
    def __init__(self):
        super().__init__(SeparationConfig(model_name="htdemucs_6s"))
    
    def separate(self, samples: np.ndarray, sample_rate: int = 44100) -> SeparatedStems:
        """HTDemucs 6源分离"""
        return super().separate(samples, sample_rate)
