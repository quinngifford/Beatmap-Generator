"""Chart construction: quantisation, difficulty filtering, holds, lanes.

Detection says *when* something happens. This module decides which of those
events become notes at a given difficulty, snaps them to musical positions,
and lays them across the keyboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsp import Spectral
from .lanes import LaneConfig, assign_lanes, brightness_from_profile
from .onsets import Onset
from .tempo import BeatGrid


@dataclass
class Note:
    """A single playable note."""

    time: float             # seconds; the value a game should judge against
    lane: int
    duration: float = 0.0   # 0 for a tap, >0 for a hold
    kind: str = "melodic"
    strength: float = 0.0
    beat: float | None = None  # position on the beat grid, e.g. 12.5 = beat 12, off-beat

    @property
    def is_hold(self) -> bool:
        return self.duration > 0.0

    def to_dict(self) -> dict:
        out = {
            "time": round(self.time, 4),
            "lane": int(self.lane),
            "type": "hold" if self.is_hold else "tap",
            "kind": self.kind,
            "strength": round(float(self.strength), 3),
        }
        if self.is_hold:
            out["duration"] = round(self.duration, 4)
            out["end_time"] = round(self.time + self.duration, 4)
        if self.beat is not None:
            out["beat"] = round(self.beat, 4)
        return out


@dataclass
class DifficultyPreset:
    """Rules defining one difficulty tier."""

    name: str
    keep_quantile: float      # keep onsets above this strength quantile (0 = all)
    subdivisions: tuple[int, ...]  # allowed beat divisions, e.g. (1, 2) = down to 8ths
    max_nps: float            # ceiling on notes per second, measured over a 1 s window
    min_gap: float            # minimum time between consecutive notes
    allow_holds: bool = True
    n_lanes: int = 4

    @property
    def min_same_lane(self) -> float:
        """Same-lane refractory period, scaled to the tier's note spacing."""
        return max(self.min_gap * 2.0, 0.100)


PRESETS: dict[str, DifficultyPreset] = {
    "easy": DifficultyPreset(
        name="easy", keep_quantile=0.72, subdivisions=(1,),
        max_nps=2.0, min_gap=0.32, allow_holds=True,
    ),
    "normal": DifficultyPreset(
        name="normal", keep_quantile=0.45, subdivisions=(1, 2),
        max_nps=3.5, min_gap=0.20, allow_holds=True,
    ),
    "hard": DifficultyPreset(
        name="hard", keep_quantile=0.20, subdivisions=(1, 2, 4),
        max_nps=6.0, min_gap=0.11, allow_holds=True,
    ),
    "expert": DifficultyPreset(
        name="expert", keep_quantile=0.0, subdivisions=(1, 2, 3, 4),
        max_nps=12.0, min_gap=0.055, allow_holds=False,
    ),
}


@dataclass
class Difficulty:
    """A built chart for one tier."""

    name: str
    notes: list[Note]
    n_lanes: int

    @property
    def notes_per_second(self) -> float:
        if not self.notes:
            return 0.0
        span = self.notes[-1].time - self.notes[0].time
        return len(self.notes) / span if span > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lanes": self.n_lanes,
            "note_count": len(self.notes),
            "notes_per_second": round(self.notes_per_second, 3),
            "notes": [n.to_dict() for n in self.notes],
        }


@dataclass
class Beatmap:
    """Everything produced for one song."""

    title: str
    audio_path: str
    duration: float
    bpm: float
    offset: float
    meter: int
    is_constant_tempo: bool
    residual_ms: float
    tempo_confidence: float
    beats: np.ndarray
    downbeats: np.ndarray
    onsets: list[Onset]
    difficulties: dict[str, Difficulty] = field(default_factory=dict)

    def to_dict(self, include_onsets: bool = True) -> dict:
        out = {
            "title": self.title,
            "audio": self.audio_path,
            "duration": round(self.duration, 3),
            "tempo": {
                "bpm": round(self.bpm, 4),
                "offset": round(self.offset, 4),
                "meter": self.meter,
                "constant": self.is_constant_tempo,
                "grid_residual_ms": round(self.residual_ms, 2),
                "confidence": round(self.tempo_confidence, 3),
            },
            "beats": [round(float(t), 4) for t in self.beats],
            "downbeats": [round(float(t), 4) for t in self.downbeats],
            "difficulties": {k: v.to_dict() for k, v in self.difficulties.items()},
        }
        if include_onsets:
            out["onsets"] = [o.to_dict() for o in self.onsets]
        return out


def quantize_to_grid(
    time: float,
    beats: np.ndarray,
    subdivisions: tuple[int, ...],
    max_shift: float = 0.050,
) -> tuple[float, float | None]:
    """Snap `time` to the nearest allowed subdivision of the tracked beats.

    Snapping happens against the *tracked* beat array rather than a fitted
    constant-tempo line, so a track that drifts or accelerates still quantises
    correctly -- each interval is subdivided on its own terms.

    A note is only moved if the nearest grid position is within `max_shift`;
    otherwise the detected time is kept, since a deliberate flam or a swung
    note is better left where it was played than dragged onto a grid it was
    never on.

    Args:
        time: Detected onset time.
        beats: Tracked beat times, ascending.
        subdivisions: Allowed divisions of the beat. ``(1, 2, 4)`` permits
            quarters, eighths and sixteenths; include 3 for triplets.
        max_shift: Largest correction allowed, in seconds.

    Returns:
        ``(snapped_time, beat_position)``. `beat_position` is the fractional
        index onto `beats`, or ``None`` if the note was left unquantised.
    """
    if beats.size < 2 or not subdivisions:
        return time, None

    i = int(np.searchsorted(beats, time) - 1)
    i = int(np.clip(i, 0, beats.size - 2))
    start, end = float(beats[i]), float(beats[i + 1])
    span = end - start
    if span <= 1e-6:
        return time, None

    position = (time - start) / span
    best_time, best_beat, best_error = time, None, np.inf

    for d in subdivisions:
        k = round(position * d)
        candidate = start + (k / d) * span
        error = abs(candidate - time)
        if error < best_error:
            best_time, best_beat, best_error = candidate, i + k / d, error

    if best_error > max_shift:
        return time, None
    return best_time, best_beat


def build_difficulty(
    onsets: list[Onset],
    grid: BeatGrid,
    spec: Spectral,
    preset: DifficultyPreset,
    quantize: bool = True,
    max_shift: float = 0.050,
) -> Difficulty:
    """Select, quantise, lane-assign and hold-extend notes for one tier.

    Selection is strongest-first rather than in time order: when the density
    cap forces a choice, the musically prominent hit should survive and the
    filler should be the one dropped.

    Args:
        onsets: All detected onsets.
        grid: Tracked beat grid, used for quantisation.
        spec: Spectral bundle, used for sustain detection.
        preset: The tier's rules.
        quantize: Snap notes to the beat grid.
        max_shift: Largest quantisation correction, in seconds.

    Returns:
        The built :class:`Difficulty`.
    """
    if not onsets:
        return Difficulty(name=preset.name, notes=[], n_lanes=preset.n_lanes)

    strengths = np.asarray([o.strength for o in onsets])
    threshold = (
        float(np.quantile(strengths, preset.keep_quantile))
        if preset.keep_quantile > 0.0 else -np.inf
    )
    eligible = [o for o in onsets if o.strength >= threshold]
    if not eligible:
        eligible = [max(onsets, key=lambda o: o.strength)]

    selected = _select_by_density(eligible, preset)
    selected.sort(key=lambda o: o.time)

    times: list[float] = []
    beat_positions: list[float | None] = []
    for onset in selected:
        if quantize:
            snapped, beat = quantize_to_grid(
                onset.time, grid.beats, preset.subdivisions, max_shift
            )
        else:
            snapped, beat = onset.time, None
        times.append(snapped)
        beat_positions.append(beat)

    # Quantisation can push two notes onto the same position; keep them ordered
    # and distinct so downstream exporters never see a negative gap.
    times = _dedupe_ascending(times)

    time_array = np.asarray(times)
    kinds = [o.kind for o in selected]
    brightness = np.asarray([brightness_from_profile(o.band_profile) for o in selected])
    lane_config = LaneConfig(
        n_lanes=preset.n_lanes, min_same_lane=preset.min_same_lane
    )
    lanes = assign_lanes(time_array, kinds, brightness, lane_config)

    notes = [
        Note(
            time=float(t), lane=int(lane), kind=kind,
            strength=float(onset.strength), beat=beat,
        )
        for t, lane, kind, onset, beat in zip(
            time_array, lanes, kinds, selected, beat_positions
        )
    ]

    if preset.allow_holds:
        _apply_holds(notes, spec)
    return Difficulty(name=preset.name, notes=notes, n_lanes=preset.n_lanes)


def _select_by_density(onsets: list[Onset], preset: DifficultyPreset) -> list[Onset]:
    """Greedily keep the strongest onsets that fit the tier's spacing budget."""
    accepted: list[Onset] = []
    accepted_times: list[float] = []

    for onset in sorted(onsets, key=lambda o: o.strength, reverse=True):
        if accepted_times:
            times = np.asarray(accepted_times)
            if np.min(np.abs(times - onset.time)) < preset.min_gap:
                continue
            window = np.sum(np.abs(times - onset.time) <= 0.5)
            if window >= preset.max_nps:
                continue
        accepted.append(onset)
        accepted_times.append(onset.time)

    return accepted


def _dedupe_ascending(times: list[float], epsilon: float = 1e-3) -> list[float]:
    """Force strictly ascending times, nudging collisions forward."""
    out: list[float] = []
    for t in times:
        if out and t <= out[-1] + epsilon:
            t = out[-1] + epsilon
        out.append(t)
    return out


def _apply_holds(
    notes: list[Note],
    spec: Spectral,
    min_duration: float = 0.28,
    sustain_ratio: float = 0.45,
    max_duration: float = 4.0,
) -> None:
    """Extend notes into holds where the audio actually sustains.

    A note becomes a hold when frame energy stays above `sustain_ratio` of its
    level at the attack for at least `min_duration`, and no later note in the
    same chart interrupts it. Modifies `notes` in place.
    """
    if spec.rms.size == 0:
        return

    rms = spec.rms
    for i, note in enumerate(notes):
        start = int(np.clip(round(note.time * spec.fps), 0, rms.size - 1))
        level = float(rms[start])
        if level <= 1e-6:
            continue

        limit = note.time + max_duration
        if i + 1 < len(notes):
            limit = min(limit, notes[i + 1].time)

        stop_frame = int(np.clip(round(limit * spec.fps), 0, rms.size - 1))
        floor = level * sustain_ratio

        frame = start
        while frame < stop_frame and rms[frame] >= floor:
            frame += 1

        duration = frame / spec.fps - note.time
        if duration >= min_duration:
            note.duration = float(duration)
