"""Command-line interface.

    python -m beatmap_generator song.mp3 -o charts/ --click
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .audio import load_audio
from .beatmap import PRESETS
from .export import write_all, write_click_track, write_json
from .pipeline import GenerationConfig, generate_beatmap
from .tempo import MAX_BPM, MIN_BPM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beatmap-generator",
        description="Detect beats and significant sounds in a song and write rhythm-game charts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="input audio file (wav/mp3/flac/ogg/...)")
    parser.add_argument("-o", "--out", type=Path, default=Path("charts"),
                        help="output directory")
    parser.add_argument("-t", "--title", default=None, help="chart title")
    parser.add_argument("--artist", default="Unknown", help="artist metadata")

    chart = parser.add_argument_group("chart")
    chart.add_argument("-d", "--difficulties", nargs="+", default=list(PRESETS),
                       choices=list(PRESETS), help="difficulty tiers to build")
    chart.add_argument("-k", "--lanes", type=int, default=4,
                       help="number of lanes / keys")
    chart.add_argument("--no-quantize", action="store_true",
                       help="keep raw detected times instead of snapping to the grid")
    chart.add_argument("--max-shift", type=float, default=0.050,
                       help="largest quantisation correction, seconds")

    detect = parser.add_argument_group("detection")
    detect.add_argument("-s", "--sensitivity", type=float, default=1.0,
                        help="onset sensitivity; >1 detects more, <1 fewer")
    detect.add_argument("--bpm", type=float, default=None,
                        help="force a known tempo and skip estimation")
    detect.add_argument("--min-bpm", type=float, default=MIN_BPM)
    detect.add_argument("--max-bpm", type=float, default=MAX_BPM)
    detect.add_argument("--meter", type=int, default=4, help="beats per bar")
    detect.add_argument("--no-octave-fix", action="store_true",
                        help="disable the half-tempo correction")
    detect.add_argument("--no-refine", action="store_true",
                        help="skip waveform attack refinement (faster, ~6ms less accurate)")

    output = parser.add_argument_group("output")
    output.add_argument("--json-only", action="store_true",
                        help="write only the JSON, no .osu/.csv")
    output.add_argument("--no-onsets", action="store_true",
                        help="omit the raw onset list from the JSON")
    output.add_argument("--click", action="store_true",
                        help="also render a click track over the music for verification")
    output.add_argument("--click-source", default="beats", choices=["beats", "notes"],
                        help="click the beat grid or the notes of a difficulty")
    output.add_argument("--click-difficulty", default=None,
                        help="difficulty to click when --click-source=notes")
    output.add_argument("-q", "--quiet", action="store_true", help="suppress the report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.audio.is_file():
        print(f"error: no such file: {args.audio}", file=sys.stderr)
        return 2
    if args.lanes < 1:
        print("error: --lanes must be at least 1", file=sys.stderr)
        return 2

    config = GenerationConfig(
        sensitivity=args.sensitivity,
        difficulties=tuple(args.difficulties),
        n_lanes=args.lanes,
        quantize=not args.no_quantize,
        max_shift=args.max_shift,
        bpm=args.bpm,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        meter=args.meter,
        fix_octave=not args.no_octave_fix,
        refine_attacks=not args.no_refine,
    )

    try:
        beatmap = generate_beatmap(args.audio, config, title=args.title)
    except Exception as exc:  # noqa: BLE001 - CLI boundary, report and exit
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json_only:
        written = {
            "json": write_json(
                beatmap, args.out / f"{args.audio.stem}.json",
                include_onsets=not args.no_onsets,
            )
        }
    else:
        written = write_all(beatmap, args.out, stem=args.audio.stem, artist=args.artist)

    if args.click:
        audio = load_audio(args.audio, sr=config.sr)
        written["click"] = write_click_track(
            beatmap, args.out / f"{args.audio.stem}.click.wav",
            audio_samples=audio.samples, sr=config.sr,
            source=args.click_source, difficulty=args.click_difficulty,
        )

    if not args.quiet:
        _report(beatmap, written)
    return 0


def _report(beatmap, written: dict) -> None:
    """Print a human-readable summary of the analysis."""
    print(f"\n{beatmap.title}  ({beatmap.duration:.1f}s)")
    print("-" * 62)

    tempo = f"{beatmap.bpm:.2f} BPM"
    if not beatmap.is_constant_tempo:
        tempo += f"  (variable: {beatmap.residual_ms:.0f}ms grid residual)"
    print(f"  tempo      {tempo}")
    print(f"  offset     {beatmap.offset * 1000:.0f} ms")
    print(f"  meter      {beatmap.meter}/4   confidence {beatmap.tempo_confidence:.2f}")
    print(f"  beats      {beatmap.beats.size}  ({beatmap.downbeats.size} downbeats)")

    kinds: dict[str, int] = {}
    for onset in beatmap.onsets:
        kinds[onset.kind] = kinds.get(onset.kind, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items(), key=lambda x: -x[1]))
    print(f"  onsets     {len(beatmap.onsets)}  ({summary})")

    if beatmap.difficulties:
        print("\n  difficulty     notes    n/s   holds  quantised")
        for name, difficulty in beatmap.difficulties.items():
            holds = sum(1 for n in difficulty.notes if n.is_hold)
            snapped = sum(1 for n in difficulty.notes if n.beat is not None)
            total = len(difficulty.notes) or 1
            print(f"  {name:<12s} {len(difficulty.notes):6d} {difficulty.notes_per_second:6.2f} "
                  f"{holds:7d}   {snapped / total * 100:5.1f}%")

    print("\n  written:")
    for label, path in written.items():
        print(f"    {label:<14s} {path}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
