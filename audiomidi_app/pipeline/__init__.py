"""
Modern Audio-to-MIDI Transcription Pipeline
现代音频转MIDI扒谱系统架构
"""
from __future__ import annotations

from audiomidi_app.pipeline.separation import (
    StemSeparator,
    HTDemucsSeparator,
    HTDemucs6SSeparator,
    SeparatedStems,
    SeparationConfig,
)
from audiomidi_app.pipeline.routing import (
    StemRouter,
    StemType,
    RoutingConfig,
    TranscriptionEngine,
)
from audiomidi_app.pipeline.transcription import (
    MT3TranscriptionEngine,
    PianoTranscriptionEngine,
    BasicPitchTranscriptionEngine,
    DrumsTranscriptionEngine,
    EnsembleTranscriptionEngine,
    TranscriptionConfig,
    TranscriptionResult,
)
from audiomidi_app.pipeline.symbolic_refinement import (
    SymbolicRefiner,
    SymbolicRefinementConfig,
    MusicLanguageCorrector,
)
from audiomidi_app.pipeline.rendering import (
    MIDIRenderer,
    StemAwareRenderer,
    RenderingConfig,
)

__all__ = [
    "StemSeparator",
    "HTDemucsSeparator", 
    "HTDemucs6SSeparator",
    "SeparatedStems",
    "SeparationConfig",
    
    "StemRouter",
    "StemType",
    "RoutingConfig",
    "TranscriptionEngine",
    
    "MT3TranscriptionEngine",
    "PianoTranscriptionEngine",
    "BasicPitchTranscriptionEngine",
    "DrumsTranscriptionEngine",
    "EnsembleTranscriptionEngine",
    "TranscriptionConfig",
    "TranscriptionResult",
    
    "SymbolicRefiner",
    "SymbolicRefinementConfig",
    "MusicLanguageCorrector",
    
    "MIDIRenderer",
    "StemAwareRenderer",
    "RenderingConfig",
]
