"""End-to-end analysis: audio file in, :class:`~.beatmap.Beatmap` out."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import DEFAULT_SR, load_audio
from .beatmap import PRESETS, Beatmap, DifficultyPreset, build_difficulty
from .dsp import DEFAULT_HOP, DEFAULT_N_FFT, superflux_odf
from .onsets import OnsetConfig, detect_onsets, onset_strengths, onset_times
from .tempo import MAX_BPM, MIN_BPM, estimate_tempo, needs_octave_doubling, track_beats


@dataclass
class GenerationConfig:
    """Options for :func:`generate_beatmap`."""

    sr: int = DEFAULT_SR
    hop: int = DEFAULT_HOP
    n_fft: int = DEFAULT_N_FFT
    sensitivity: float = 1.0
    difficulties: tuple[str, ...] = ("easy", "normal", "hard", "expert")
    n_lanes: int = 4
    quantize: bool = True
    max_shift: float = 0.050
    bpm: float | None = None          # override tempo detection entirely
    min_bpm: float = MIN_BPM
    max_bpm: float = MAX_BPM
    meter: int = 4
    fix_octave: bool = True           # allow the half-tempo correction
    refine_attacks: bool = True
    onset_config: OnsetConfig | None = None


def analyze(path: str | Path, config: GenerationConfig | None = None):
    """Run detection only, returning the raw analysis objects.

    Useful when you want onsets and a beat grid without building charts --
    for a custom editor, or to feed another mapping scheme.

    Returns:
        ``(audio, spectral, onsets, beat_grid, tempo_result)``.
    """
    cfg = config or GenerationConfig()

    audio = load_audio(path, sr=cfg.sr)
    spec = superflux_odf(audio.samples, audio.sr, hop=cfg.hop, n_fft=cfg.n_fft)

    onset_cfg = cfg.onset_config or OnsetConfig()
    if cfg.sensitivity != 1.0:
        onset_cfg = OnsetConfig(**{**onset_cfg.__dict__, "sensitivity": cfg.sensitivity})

    onsets = detect_onsets(
        spec, onset_cfg, signal=audio.samples if cfg.refine_attacks else None
    )

    tempo = estimate_tempo(spec, min_bpm=cfg.min_bpm, max_bpm=cfg.max_bpm)
    bpm = cfg.bpm if cfg.bpm else tempo.bpm
    grid = track_beats(spec, bpm, meter=cfg.meter)

    # ACF sits an octave low on backbeat-driven material; re-track if the
    # midpoints between beats carry as much weight as the beats themselves.
    if cfg.fix_octave and cfg.bpm is None and grid.beats.size:
        if needs_octave_doubling(
            grid, onset_times(onsets), onset_strengths(onsets), max_bpm=cfg.max_bpm
        ):
            grid = track_beats(spec, grid.bpm * 2.0, meter=cfg.meter)

    return audio, spec, onsets, grid, tempo


def generate_beatmap(
    path: str | Path,
    config: GenerationConfig | None = None,
    title: str | None = None,
) -> Beatmap:
    """Analyse `path` and build a chart at every requested difficulty.

    Args:
        path: Audio file.
        config: Generation options; defaults suit produced music.
        title: Chart title. Defaults to the audio file's stem.

    Returns:
        A fully populated :class:`~.beatmap.Beatmap`.
    """
    cfg = config or GenerationConfig()
    audio, spec, onsets, grid, tempo = analyze(path, cfg)

    beatmap = Beatmap(
        title=title or Path(path).stem,
        audio_path=Path(path).name,
        duration=audio.duration,
        bpm=grid.bpm,
        offset=grid.offset,
        meter=grid.meter,
        is_constant_tempo=grid.is_constant_tempo,
        residual_ms=grid.residual_ms,
        tempo_confidence=tempo.confidence,
        beats=grid.beats,
        downbeats=grid.downbeats,
        onsets=onsets,
    )

    for name in cfg.difficulties:
        if name not in PRESETS:
            raise KeyError(f"unknown difficulty {name!r}; choose from {sorted(PRESETS)}")
        base = PRESETS[name]
        preset = DifficultyPreset(**{**base.__dict__, "n_lanes": cfg.n_lanes})
        beatmap.difficulties[name] = build_difficulty(
            onsets, grid, spec, preset, quantize=cfg.quantize, max_shift=cfg.max_shift
        )

    return beatmap
