"""Beatmap generator: audio in, timestamped rhythm-game chart out."""

from .audio import Audio, load_audio
from .dsp import Spectral, superflux_odf
from .onsets import Onset, OnsetConfig, detect_onsets
from .tempo import BeatGrid, TempoResult, estimate_tempo, track_beats

__version__ = "1.0.0"

__all__ = [
    "Audio",
    "BeatGrid",
    "Onset",
    "OnsetConfig",
    "Spectral",
    "TempoResult",
    "detect_onsets",
    "estimate_tempo",
    "load_audio",
    "superflux_odf",
    "track_beats",
]
