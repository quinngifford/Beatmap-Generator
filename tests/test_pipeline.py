"""Accuracy and behaviour tests against synthetic tracks with known ground truth.

Run with ``pytest`` from the project root. Fixtures are generated on demand by
:mod:`make_test_audio`, so there is nothing binary to check in.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatmap_generator.audio import load_audio
from beatmap_generator.beatmap import PRESETS, quantize_to_grid
from beatmap_generator.dsp import log_frequency_filterbank, superflux_odf
from beatmap_generator.export import write_all, write_click_track
from beatmap_generator.lanes import LaneConfig, assign_lanes
from beatmap_generator.onsets import (
    OnsetConfig, classify_onset, detect_onsets, onset_times,
)
from beatmap_generator.pipeline import GenerationConfig, analyze, generate_beatmap

FIXTURES = Path(__file__).parent / "fixtures"
CONSTANT_TEMPO = ["drums_128", "drums_90_nobass", "drums_174"]
ALL_TRACKS = CONSTANT_TEMPO + ["sustained_100", "varispeed"]


@pytest.fixture(scope="session", autouse=True)
def fixtures_exist():
    """Generate the audio fixtures once per session if they are missing."""
    needed = [FIXTURES / f"{n}.wav" for n in ALL_TRACKS]
    if not all(p.is_file() for p in needed):
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "make_test_audio.py")],
            check=True, capture_output=True,
        )


def truth(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.truth.json").read_text())


def analysis(name: str, **kwargs):
    return analyze(FIXTURES / f"{name}.wav", GenerationConfig(**kwargs))


def match_events(
    estimated: np.ndarray, reference: np.ndarray, tolerance: float
) -> tuple[float, float, float, np.ndarray]:
    """One-to-one greedy matching.

    Returns:
        ``(f_measure, precision, recall, signed_errors_ms)``.
    """
    estimated = np.sort(np.asarray(estimated, dtype=float))
    reference = np.sort(np.asarray(reference, dtype=float))
    used = np.zeros(reference.size, dtype=bool)
    errors: list[float] = []

    for e in estimated:
        distance = np.abs(reference - e)
        distance[used] = np.inf
        if distance.size == 0:
            break
        i = int(np.argmin(distance))
        if distance[i] <= tolerance:
            used[i] = True
            errors.append((e - reference[i]) * 1000.0)

    tp = len(errors)
    precision = tp / estimated.size if estimated.size else 0.0
    recall = tp / reference.size if reference.size else 0.0
    f = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f, precision, recall, np.asarray(errors)


# --------------------------------------------------------------------------
# Onset detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_TRACKS)
def test_onsets_are_exact(name):
    """Every onset is found, with nothing spurious, at a 25 ms tolerance."""
    _, _, onsets, _, _ = analysis(name)
    f, precision, recall, _ = match_events(
        onset_times(onsets), np.asarray(truth(name)["onsets"]), 0.025
    )
    assert recall == 1.0, f"{name}: missed onsets (recall {recall:.3f})"
    assert precision == 1.0, f"{name}: spurious onsets (precision {precision:.3f})"
    assert f == 1.0


@pytest.mark.parametrize("name", CONSTANT_TEMPO + ["varispeed"])
def test_percussive_onset_timing_is_sub_millisecond(name):
    """Waveform attack refinement should leave well under 1 ms of error."""
    _, _, onsets, _, _ = analysis(name)
    _, _, _, errors = match_events(
        onset_times(onsets), np.asarray(truth(name)["onsets"]), 0.025
    )
    assert errors.size > 0
    assert abs(errors.mean()) < 1.0, f"{name}: bias {errors.mean():+.2f} ms"
    assert errors.std() < 1.0, f"{name}: spread {errors.std():.2f} ms"
    assert np.abs(errors).max() < 3.0, f"{name}: worst {np.abs(errors).max():.2f} ms"


def test_slow_attack_timing_stays_within_tolerance():
    """Soft attacks are looser than percussive ones, but must stay usable.

    ``sustained_100`` fades each chord in over 15 ms, so there is no single
    sample that *is* the onset; the detector lands partway up the rise. That
    is a property of the material rather than a defect, so the bound here is
    the perceptual one rather than the sub-millisecond one.
    """
    _, _, onsets, _, _ = analysis("sustained_100")
    _, _, _, errors = match_events(
        onset_times(onsets), np.asarray(truth("sustained_100")["onsets"]), 0.030
    )
    assert errors.size > 0
    assert np.abs(errors).max() < 20.0, f"worst {np.abs(errors).max():.2f} ms"


def test_attack_refinement_beats_raw_odf():
    """The refinement step must actually improve on the spectrogram estimate."""
    reference = np.asarray(truth("drums_128")["onsets"])
    audio = load_audio(FIXTURES / "drums_128.wav")
    spec = superflux_odf(audio.samples, audio.sr)

    raw = onset_times(detect_onsets(spec))
    refined = onset_times(detect_onsets(spec, signal=audio.samples))

    _, _, _, raw_err = match_events(raw, reference, 0.030)
    _, _, _, ref_err = match_events(refined, reference, 0.030)
    assert abs(ref_err.mean()) < abs(raw_err.mean())
    assert ref_err.std() < raw_err.std()


def test_sensitivity_is_monotonic():
    """Raising sensitivity must not reduce the number of detections."""
    counts = []
    for sensitivity in (0.5, 1.0, 2.0):
        _, _, onsets, _, _ = analysis("drums_128", sensitivity=sensitivity)
        counts.append(len(onsets))
    assert counts[0] <= counts[1] <= counts[2]


def test_silence_yields_no_onsets():
    spec = superflux_odf(np.zeros(44100 * 2, dtype=np.float32), 44100)
    assert detect_onsets(spec) == []


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def test_drum_classification_accuracy():
    """Kicks and snares should be labelled correctly; hats mostly so."""
    correct = total = 0
    for name in CONSTANT_TEMPO:
        info = truth(name)
        _, _, onsets, _, _ = analysis(name)
        times = onset_times(onsets)
        for reference, label in zip(info["onsets"], info["labels"]):
            i = int(np.argmin(np.abs(times - reference)))
            if abs(times[i] - reference) <= 0.025:
                total += 1
                correct += onsets[i].kind == label
    assert total > 300
    assert correct / total >= 0.90, f"classification accuracy {correct / total:.3f}"


def test_classify_onset_boundaries():
    assert classify_onset(np.array([0.90, 0.05, 0.03, 0.02])) == "kick"
    assert classify_onset(np.array([0.02, 0.03, 0.05, 0.90])) == "hat"
    assert classify_onset(np.array([0.10, 0.40, 0.30, 0.20])) == "snare"
    assert classify_onset(np.array([0.10, 0.15, 0.15, 0.60])) == "hat"


# --------------------------------------------------------------------------
# Tempo and beats
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", CONSTANT_TEMPO)
def test_tempo_within_half_a_bpm(name):
    _, _, _, grid, _ = analysis(name)
    expected = truth(name)["bpm"]
    assert abs(grid.bpm - expected) < 0.5, f"{name}: got {grid.bpm:.3f}, want {expected}"


@pytest.mark.parametrize("name", CONSTANT_TEMPO)
def test_beat_grid_is_tight(name):
    """A constant-tempo track must fit one BPM and one offset closely."""
    _, _, _, grid, _ = analysis(name)
    assert grid.is_constant_tempo
    assert grid.residual_ms < 8.0, f"{name}: residual {grid.residual_ms:.2f} ms"


@pytest.mark.parametrize("name", CONSTANT_TEMPO)
def test_beats_land_on_true_beats(name):
    _, _, _, grid, _ = analysis(name)
    f, _, recall, _ = match_events(grid.beats, np.asarray(truth(name)["beats"]), 0.030)
    assert recall >= 0.95, f"{name}: beat recall {recall:.3f}"


def test_octave_correction_recovers_fast_tempo():
    """174 BPM must not be reported as 87; the correction has to fire."""
    _, _, _, corrected, _ = analysis("drums_174", fix_octave=True)
    _, _, _, uncorrected, _ = analysis("drums_174", fix_octave=False)
    assert abs(corrected.bpm - 174.0) < 0.5
    assert abs(uncorrected.bpm - 87.0) < 1.0  # documents the ACF's preference


def test_octave_correction_leaves_correct_tempo_alone():
    """The correction must not double a tempo that was already right."""
    for name, expected in (("drums_128", 128.0), ("drums_90_nobass", 90.0)):
        _, _, _, grid, _ = analysis(name)
        assert abs(grid.bpm - expected) < 0.5, f"{name} was wrongly doubled"


def test_variable_tempo_is_flagged():
    """An accelerating track must not be presented as a fixed grid."""
    _, _, _, grid, _ = analysis("varispeed")
    assert not grid.is_constant_tempo
    assert grid.residual_ms > 25.0


def test_explicit_bpm_is_respected():
    _, _, _, grid, _ = analysis("drums_128", bpm=64.0, fix_octave=False)
    assert abs(grid.bpm - 64.0) < 1.0


def test_beat_grid_has_no_outlier_intervals():
    """Edge trimming should leave a uniform grid."""
    for name in CONSTANT_TEMPO:
        _, _, _, grid, _ = analysis(name)
        intervals = np.diff(grid.beats)
        median = np.median(intervals)
        assert np.all(np.abs(intervals - median) / median < 0.15), name


def test_downbeats_are_a_subset_of_beats():
    for name in CONSTANT_TEMPO:
        _, _, _, grid, _ = analysis(name)
        assert grid.downbeats.size > 0
        for t in grid.downbeats:
            assert np.min(np.abs(grid.beats - t)) < 1e-9


# --------------------------------------------------------------------------
# Chart construction
# --------------------------------------------------------------------------

def test_difficulties_increase_in_density():
    beatmap = generate_beatmap(FIXTURES / "drums_128.wav")
    counts = [len(beatmap.difficulties[n].notes) for n in ("easy", "normal", "hard", "expert")]
    assert counts == sorted(counts), f"densities not monotonic: {counts}"
    assert counts[0] > 0


@pytest.mark.parametrize("name", ["easy", "normal", "hard", "expert"])
def test_notes_respect_min_gap_and_ordering(name):
    beatmap = generate_beatmap(FIXTURES / "drums_128.wav")
    notes = beatmap.difficulties[name].notes
    times = [n.time for n in notes]
    assert times == sorted(times), "notes are not in time order"

    per_lane: dict[int, float] = {}
    for note in notes:
        previous = per_lane.get(note.lane)
        if previous is not None:
            gap = note.time - previous
            assert gap >= PRESETS[name].min_same_lane - 1e-6, (
                f"{name}: lane {note.lane} repeats after {gap * 1000:.0f} ms"
            )
        per_lane[note.lane] = note.time


def test_lanes_are_within_range_and_varied():
    beatmap = generate_beatmap(
        FIXTURES / "drums_128.wav", GenerationConfig(n_lanes=4, difficulties=("hard",))
    )
    lanes = [n.lane for n in beatmap.difficulties["hard"].notes]
    assert all(0 <= lane < 4 for lane in lanes)
    assert len(set(lanes)) >= 3, "chart uses too few lanes to be interesting"


@pytest.mark.parametrize("n_lanes", [1, 2, 4, 5, 6, 7])
def test_arbitrary_lane_counts(n_lanes):
    beatmap = generate_beatmap(
        FIXTURES / "drums_90_nobass.wav",
        GenerationConfig(n_lanes=n_lanes, difficulties=("normal",)),
    )
    lanes = [n.lane for n in beatmap.difficulties["normal"].notes]
    assert lanes and all(0 <= lane < n_lanes for lane in lanes)


def test_quantisation_snaps_to_the_grid():
    beatmap = generate_beatmap(
        FIXTURES / "drums_128.wav", GenerationConfig(difficulties=("hard",))
    )
    notes = beatmap.difficulties["hard"].notes
    snapped = [n for n in notes if n.beat is not None]
    assert len(snapped) / len(notes) > 0.95

    period = 60.0 / beatmap.bpm
    for note in snapped:
        offset_in_beat = (note.beat % 1.0) * period
        nearest = min(abs(offset_in_beat - k * period / 4) for k in range(5))
        assert nearest < 0.012, f"note at beat {note.beat} is off the 1/16 grid"


def test_quantisation_leaves_far_notes_alone():
    """A note nowhere near the grid keeps its detected time."""
    beats = np.arange(0, 10, 0.5)
    snapped, beat = quantize_to_grid(1.24, beats, (1, 2), max_shift=0.010)
    assert snapped == 1.24 and beat is None

    snapped, beat = quantize_to_grid(1.24, beats, (1, 2), max_shift=0.060)
    assert abs(snapped - 1.25) < 1e-9 and beat == pytest.approx(2.5)


def test_holds_appear_on_sustained_audio():
    beatmap = generate_beatmap(
        FIXTURES / "sustained_100.wav", GenerationConfig(difficulties=("normal",))
    )
    notes = beatmap.difficulties["normal"].notes
    holds = [n for n in notes if n.is_hold]
    assert holds, "no holds detected on deliberately sustained material"
    assert all(n.duration > 0 for n in holds)
    for note, following in zip(notes, notes[1:]):
        assert note.time + note.duration <= following.time + 1e-6, "holds overlap"


def test_percussive_audio_yields_few_holds():
    beatmap = generate_beatmap(
        FIXTURES / "drums_128.wav", GenerationConfig(difficulties=("normal",))
    )
    notes = beatmap.difficulties["normal"].notes
    holds = sum(1 for n in notes if n.is_hold)
    assert holds / len(notes) < 0.15, "short drum hits should rarely become holds"


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def test_write_all_and_osu_structure(tmp_path):
    beatmap = generate_beatmap(
        FIXTURES / "drums_128.wav", GenerationConfig(difficulties=("normal",))
    )
    written = write_all(beatmap, tmp_path, stem="song")
    assert all(p.is_file() for p in written.values())

    text = written["osu:normal"].read_text()
    assert text.startswith("osu file format v14")
    assert "Mode: 3" in text          # mania
    assert "CircleSize:4" in text     # key count

    body = text.split("[HitObjects]")[1].strip().splitlines()
    assert len(body) == len(beatmap.difficulties["normal"].notes)

    times = []
    for line in body:
        parts = line.split(",")
        assert len(parts) >= 6
        x = int(parts[0])
        assert 0 <= x < 512
        assert int(parts[3]) in (1, 128)
        times.append(int(parts[2]))
    assert times == sorted(times)


def test_osu_emits_timing_points_for_variable_tempo(tmp_path):
    steady = generate_beatmap(
        FIXTURES / "drums_128.wav", GenerationConfig(difficulties=("normal",))
    )
    drifting = generate_beatmap(
        FIXTURES / "varispeed.wav", GenerationConfig(difficulties=("normal",))
    )
    a = write_all(steady, tmp_path / "a", stem="a")["osu:normal"].read_text()
    b = write_all(drifting, tmp_path / "b", stem="b")["osu:normal"].read_text()

    def timing_lines(text: str) -> int:
        block = text.split("[TimingPoints]")[1].split("[HitObjects]")[0]
        return len([ln for ln in block.strip().splitlines() if ln.strip()])

    assert timing_lines(a) == 1
    assert timing_lines(b) > 10


def test_json_round_trips(tmp_path):
    beatmap = generate_beatmap(
        FIXTURES / "drums_128.wav", GenerationConfig(difficulties=("easy",))
    )
    path = write_all(beatmap, tmp_path, stem="song")["json"]
    data = json.loads(path.read_text())
    assert data["tempo"]["bpm"] == pytest.approx(beatmap.bpm, abs=1e-3)
    assert len(data["difficulties"]["easy"]["notes"]) == len(beatmap.difficulties["easy"].notes)
    assert data["onsets"] and "kind" in data["onsets"][0]


def test_click_track_is_written(tmp_path):
    beatmap = generate_beatmap(
        FIXTURES / "drums_128.wav", GenerationConfig(difficulties=("normal",))
    )
    audio = load_audio(FIXTURES / "drums_128.wav")
    path = write_click_track(
        beatmap, tmp_path / "click.wav", audio_samples=audio.samples, sr=audio.sr
    )
    rendered = load_audio(path, normalize=False)
    assert rendered.duration == pytest.approx(beatmap.duration, abs=1.5)


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------

def test_filterbank_rows_are_normalised():
    filters, centers = log_frequency_filterbank(44100, 2048)
    assert filters.shape[0] == centers.size
    assert np.allclose(filters.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(np.diff(centers) > 0)
    assert np.all(filters >= 0)


def test_filterbank_drops_duplicate_filters():
    """Below the FFT resolution, log spacing would otherwise repeat filters."""
    filters, _ = log_frequency_filterbank(44100, 2048, bands_per_octave=24)
    supports = {(f > 0).tobytes() for f in filters}
    assert len(supports) == filters.shape[0]


def test_assign_lanes_avoids_fast_repeats():
    times = np.arange(0, 4.0, 0.06)
    kinds = ["hat"] * times.size
    brightness = np.full(times.size, 0.5)
    lanes = assign_lanes(times, kinds, brightness, LaneConfig(n_lanes=4, min_same_lane=0.12))
    assert all(a != b for a, b in zip(lanes, lanes[1:])), "same lane twice in a row"


def test_chords_get_distinct_lanes():
    times = np.asarray([1.0, 1.0, 1.0])
    lanes = assign_lanes(
        times, ["kick", "snare", "hat"], np.asarray([0.1, 0.5, 0.9]),
        LaneConfig(n_lanes=4),
    )
    assert len(set(lanes.tolist())) == 3


def test_short_audio_does_not_crash(tmp_path):
    import soundfile as sf
    sf.write(tmp_path / "tiny.wav", np.zeros(2205, dtype=np.float32), 44100)
    beatmap = generate_beatmap(tmp_path / "tiny.wav", GenerationConfig(difficulties=("easy",)))
    assert beatmap.difficulties["easy"].notes == []


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_audio("does_not_exist_anywhere.wav")


def test_cli_runs(tmp_path):
    from beatmap_generator.cli import main
    code = main([
        str(FIXTURES / "drums_128.wav"), "-o", str(tmp_path),
        "-d", "normal", "--click", "-q",
    ])
    assert code == 0
    assert (tmp_path / "drums_128.json").is_file()
    assert (tmp_path / "drums_128.click.wav").is_file()
