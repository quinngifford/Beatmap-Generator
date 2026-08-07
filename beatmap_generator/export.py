"""Writers for the formats a level designer actually opens.

JSON is the canonical output and carries everything. The osu!mania writer
produces a file the real editor will load, so a generated chart can be
auditioned and hand-tuned immediately. The click track is the fastest way to
tell whether a map is right: play it and listen for flams against the music.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .beatmap import Beatmap, Difficulty

# osu! playfield is 512 units wide regardless of key count.
OSU_PLAYFIELD_WIDTH = 512
OSU_NOTE = 1        # hit-object type bit for a tap
OSU_HOLD = 128      # ... and for a mania hold


def write_json(beatmap: Beatmap, path: str | Path, include_onsets: bool = True) -> Path:
    """Write the full analysis and every difficulty as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(beatmap.to_dict(include_onsets=include_onsets), indent=2),
        encoding="utf-8",
    )
    return path


def write_csv(difficulty: Difficulty, path: str | Path) -> Path:
    """Write one difficulty as a flat CSV, one row per note."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["time_s", "time_ms", "lane", "type", "duration_s", "kind", "strength", "beat"]
        )
        for note in difficulty.notes:
            writer.writerow([
                f"{note.time:.4f}",
                int(round(note.time * 1000)),
                note.lane,
                "hold" if note.is_hold else "tap",
                f"{note.duration:.4f}",
                note.kind,
                f"{note.strength:.3f}",
                "" if note.beat is None else f"{note.beat:.4f}",
            ])
    return path


def write_osu(
    beatmap: Beatmap,
    difficulty: Difficulty,
    path: str | Path,
    artist: str = "Unknown",
    creator: str = "beatmap-generator",
) -> Path:
    """Write one difficulty as an osu!mania ``.osu`` file.

    Timing points follow the analysis: a constant-tempo track gets a single
    uninherited point, while a drifting one gets a point per beat so the
    editor's grid tracks the performance instead of sliding away from it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lanes = max(difficulty.n_lanes, 1)

    lines: list[str] = [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {beatmap.audio_path}",
        "AudioLeadIn: 0",
        "PreviewTime: -1",
        "Countdown: 0",
        "SampleSet: Soft",
        "StackLeniency: 0.7",
        "Mode: 3",
        "LetterboxInBreaks: 0",
        "SpecialStyle: 0",
        "WidescreenStoryboard: 0",
        "",
        "[Editor]",
        "DistanceSpacing: 1",
        "BeatDivisor: 4",
        "GridSize: 4",
        "TimelineZoom: 1",
        "",
        "[Metadata]",
        f"Title:{beatmap.title}",
        f"TitleUnicode:{beatmap.title}",
        f"Artist:{artist}",
        f"ArtistUnicode:{artist}",
        f"Creator:{creator}",
        f"Version:{difficulty.name}",
        "Source:",
        "Tags:generated",
        "BeatmapID:0",
        "BeatmapSetID:-1",
        "",
        "[Difficulty]",
        "HPDrainRate:7",
        f"CircleSize:{lanes}",  # mania reads key count from CircleSize
        f"OverallDifficulty:{_overall_difficulty(difficulty.name)}",
        "ApproachRate:5",
        "SliderMultiplier:1.4",
        "SliderTickRate:1",
        "",
        "[Events]",
        "//Background and Video events",
        "",
        "[TimingPoints]",
    ]
    lines.extend(_timing_points(beatmap))
    lines += ["", "[HitObjects]"]

    for note in difficulty.notes:
        x = int((note.lane + 0.5) * OSU_PLAYFIELD_WIDTH / lanes)
        start = int(round(note.time * 1000))
        if note.is_hold:
            end = int(round((note.time + note.duration) * 1000))
            lines.append(f"{x},192,{start},{OSU_HOLD},0,{end}:0:0:0:0:")
        else:
            lines.append(f"{x},192,{start},{OSU_NOTE},0,0:0:0:0:")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _timing_points(beatmap: Beatmap) -> list[str]:
    """Build the [TimingPoints] block."""
    if beatmap.is_constant_tempo or beatmap.beats.size < 3:
        beat_length = 60000.0 / max(beatmap.bpm, 1e-6)
        offset = int(round(beatmap.offset * 1000))
        return [f"{offset},{beat_length:.6f},{beatmap.meter},2,0,70,1,0"]

    points: list[str] = []
    beats = beatmap.beats
    for i in range(beats.size - 1):
        interval = (beats[i + 1] - beats[i]) * 1000.0
        if interval <= 0:
            continue
        points.append(
            f"{int(round(beats[i] * 1000))},{interval:.6f},{beatmap.meter},2,0,70,1,0"
        )
    return points


def _overall_difficulty(name: str) -> int:
    """Map a tier name to an osu! OD value (judgement window tightness)."""
    return {"easy": 5, "normal": 6, "hard": 7, "expert": 8}.get(name, 7)


def write_click_track(
    beatmap: Beatmap,
    path: str | Path,
    audio_samples: np.ndarray | None = None,
    sr: int = 44100,
    source: str = "beats",
    difficulty: str | None = None,
    mix: float = 0.6,
) -> Path:
    """Render clicks at the detected positions, optionally over the music.

    This is the verification step that matters. Charts look plausible on a
    timeline long after they have drifted; a click that flams against the
    snare is impossible to miss.

    Args:
        beatmap: The analysed map.
        path: Output ``.wav`` path.
        audio_samples: Original mono audio to mix under the clicks. Omit for
            clicks alone.
        sr: Sample rate for the render.
        source: ``"beats"`` clicks the beat grid (downbeats pitched higher),
            ``"notes"`` clicks every note of `difficulty`.
        difficulty: Which difficulty to click when ``source="notes"``.
        mix: Gain applied to the music underneath the clicks.

    Returns:
        The written path.
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    length = int(np.ceil(beatmap.duration * sr)) + sr
    out = np.zeros(length, dtype=np.float32)
    if audio_samples is not None:
        n = min(audio_samples.size, length)
        out[:n] += audio_samples[:n].astype(np.float32) * mix

    if source == "notes":
        name = difficulty or next(iter(beatmap.difficulties), None)
        if name is None or name not in beatmap.difficulties:
            raise KeyError(f"difficulty {name!r} not present in beatmap")
        events = [(n.time, 1000.0) for n in beatmap.difficulties[name].notes]
    else:
        downbeats = set(np.round(beatmap.downbeats, 4).tolist())
        events = [
            (float(t), 1500.0 if round(float(t), 4) in downbeats else 1000.0)
            for t in beatmap.beats
        ]

    for time, freq in events:
        _add_click(out, time, freq, sr)

    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        out /= peak
    sf.write(str(path), out, sr)
    return path


def _add_click(buffer: np.ndarray, time: float, freq: float, sr: int) -> None:
    """Mix a short decaying sine into `buffer` at `time`."""
    start = int(round(time * sr))
    if start < 0 or start >= buffer.size:
        return
    n = min(int(0.030 * sr), buffer.size - start)
    if n <= 0:
        return
    t = np.arange(n) / sr
    click = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.006) * 0.5
    buffer[start : start + n] += click.astype(np.float32)


def write_all(
    beatmap: Beatmap,
    out_dir: str | Path,
    stem: str | None = None,
    artist: str = "Unknown",
) -> dict[str, Path]:
    """Write JSON plus a ``.osu`` and CSV for every difficulty.

    Returns:
        A mapping of label to written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or _safe_stem(beatmap.title)

    written = {"json": write_json(beatmap, out_dir / f"{stem}.json")}
    for name, difficulty in beatmap.difficulties.items():
        written[f"osu:{name}"] = write_osu(
            beatmap, difficulty, out_dir / f"{stem} [{name}].osu", artist=artist
        )
        written[f"csv:{name}"] = write_csv(difficulty, out_dir / f"{stem}.{name}.csv")
    return written


def _safe_stem(title: str) -> str:
    """Strip characters that Windows will not accept in a filename."""
    cleaned = "".join(c for c in title if c not in '<>:"/\\|?*').strip()
    return cleaned or "beatmap"
