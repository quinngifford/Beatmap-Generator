"""Tempo estimation, beat tracking and downbeat detection.

Beats come from Ellis's dynamic-programming tracker (JNMR 2007), which finds
the globally optimal beat sequence rather than chaining local decisions -- so a
single ambiguous bar cannot knock the grid permanently out of phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .dsp import Spectral, normalize_odf

MIN_BPM = 50.0
MAX_BPM = 220.0
PRIOR_CENTER_BPM = 120.0
PRIOR_WIDTH_OCTAVES = 1.0


@dataclass
class TempoResult:
    """Estimated tempo with runner-up candidates."""

    bpm: float
    confidence: float                      # 0-1, margin of the winner over the field
    candidates: list[tuple[float, float]]  # (bpm, score) sorted best-first


@dataclass
class BeatGrid:
    """A tracked beat sequence plus the constant-tempo grid fitted to it."""

    beats: np.ndarray          # beat times in seconds, as tracked
    downbeats: np.ndarray      # subset of `beats` starting each bar
    bpm: float                 # from the least-squares fit over beat indices
    offset: float              # time of the first grid line, seconds
    meter: int                 # beats per bar
    beat_strength: np.ndarray  # ODF value at each beat
    residual_ms: float         # RMS deviation of tracked beats from the fitted grid
    is_constant_tempo: bool    # residual small enough to trust bpm+offset alone

    @property
    def beat_period(self) -> float:
        return 60.0 / self.bpm

    def grid_time(self, index: float) -> float:
        """Time of grid line `index` (fractional indices allowed)."""
        return self.offset + index * self.beat_period

    def nearest_grid_index(self, time: float, subdivision: int = 1) -> float:
        """Nearest grid index to `time`, in units of 1/`subdivision` beats."""
        step = self.beat_period / subdivision
        return round((time - self.offset) / step) / subdivision


def estimate_tempo(
    spec: Spectral,
    min_bpm: float = MIN_BPM,
    max_bpm: float = MAX_BPM,
    n_harmonics: int = 4,
) -> TempoResult:
    """Estimate global tempo by harmonic-weighted autocorrelation.

    The onset envelope's autocorrelation peaks at the beat period and at every
    multiple of it. Scoring each candidate period by the sum of its own ACF
    value plus those of its multiples therefore rewards the true period and
    penalises half of it -- a half-period candidate would need peaks at odd
    multiples, where a real beat train has none. That resolves the usual
    double/half-tempo ambiguity without a hard tempo range.

    A log-normal prior centred at 120 BPM breaks remaining ties the way a
    listener would, preferring tempi near a comfortable tapping rate.
    """
    odf = normalize_odf(spec.odf)
    if odf.size < 4:
        return TempoResult(bpm=PRIOR_CENTER_BPM, confidence=0.0, candidates=[])

    acf = _autocorrelate(odf - odf.mean())

    lag_min = max(int(np.floor(60.0 / max_bpm * spec.fps)), 1)
    lag_max = min(int(np.ceil(60.0 / min_bpm * spec.fps)), acf.size - 1)
    if lag_max <= lag_min:
        return TempoResult(bpm=PRIOR_CENTER_BPM, confidence=0.0, candidates=[])

    lags = np.arange(lag_min, lag_max + 1)
    scores = np.zeros(lags.size, dtype=np.float64)
    for h in range(1, n_harmonics + 1):
        idx = lags * h
        valid = idx < acf.size
        scores[valid] += acf[idx[valid]] / h

    bpms = 60.0 * spec.fps / lags
    scores *= _log_normal_prior(bpms)
    scores = np.maximum(scores, 0.0)

    if not np.any(scores > 0):
        return TempoResult(bpm=PRIOR_CENTER_BPM, confidence=0.0, candidates=[])

    order = np.argsort(scores)[::-1]
    top = [(_interpolated_bpm(scores, lags, int(i), spec.fps), float(scores[i]))
           for i in order[:32]]
    candidates = _dedupe_candidates(top, tolerance_bpm=2.0)[:5]

    best_bpm = candidates[0][0]
    best_score = candidates[0][1]
    runner_up = candidates[1][1] if len(candidates) > 1 else 0.0
    confidence = 0.0 if best_score <= 0 else float(1.0 - runner_up / best_score)

    return TempoResult(bpm=best_bpm, confidence=confidence, candidates=candidates)


def track_beats(
    spec: Spectral,
    bpm: float,
    tightness: float = 100.0,
    meter: int = 4,
) -> BeatGrid:
    """Find beat times by dynamic programming, then fit a tempo grid to them.

    Every frame stores the best cumulative score of any beat sequence ending
    there, choosing its predecessor from a window around one beat period back
    and paying ``-tightness * log(gap / period)^2`` for straying from that
    period. One backtrace from the best-scoring endpoint yields the globally
    optimal sequence.

    Args:
        spec: Spectral bundle carrying the onset envelope.
        bpm: Target tempo from :func:`estimate_tempo`.
        tightness: How strongly the tracker holds tempo. Higher is stiffer;
            100 tolerates modest live drift while ignoring syncopation.
        meter: Beats per bar, used for downbeat phase selection.

    Returns:
        A :class:`BeatGrid`. Empty beat array if the track is too short.
    """
    odf = normalize_odf(spec.odf)
    period = 60.0 / bpm * spec.fps
    if odf.size < 2 * period or period < 2:
        return _empty_grid(bpm, meter)

    # Light smoothing so the tracker locks to attack clusters, not single frames.
    local = gaussian_filter1d(odf.astype(np.float64), sigma=max(period / 32.0, 1.0))
    std = local.std()
    if std > 1e-9:
        local /= std

    period_i = int(round(period))
    search = np.arange(-2 * period_i, -max(period_i // 2, 1) + 1)
    if search.size == 0:
        return _empty_grid(bpm, meter)
    tx_cost = -tightness * np.log(search / -period) ** 2

    n = local.size
    cumscore = np.zeros(n, dtype=np.float64)
    backlink = np.full(n, -1, dtype=np.int64)

    for i in range(n):
        prev = i + search
        valid = prev >= 0
        if not valid.any():
            cumscore[i] = local[i]
            continue
        cands = np.full(search.size, -np.inf)
        cands[valid] = cumscore[prev[valid]] + tx_cost[valid]
        best = int(np.argmax(cands))
        cumscore[i] = local[i] + cands[best]
        backlink[i] = prev[best] if valid[best] else -1

    beats_frames = _backtrace(cumscore, backlink)
    if beats_frames.size < 2:
        return _empty_grid(bpm, meter)

    beat_times = _refine_beat_times(beats_frames, odf, spec.fps)
    beat_times = _trim_edge_outliers(beat_times)
    beat_times = _extend_to_span(beat_times, spec.n_frames / spec.fps)
    strength = _strength_at(odf, beat_times, spec.fps)

    fit_bpm, offset, residual_ms = fit_tempo_grid(beat_times)
    meter, downbeats = detect_downbeats(beat_times, spec, meter_candidates=(meter,))

    return BeatGrid(
        beats=beat_times,
        downbeats=downbeats,
        bpm=fit_bpm,
        offset=offset,
        meter=meter,
        beat_strength=strength.astype(np.float64),
        residual_ms=residual_ms,
        is_constant_tempo=residual_ms < 25.0,
    )


def needs_octave_doubling(
    grid: BeatGrid,
    times: np.ndarray,
    strengths: np.ndarray,
    threshold: float = 0.72,
    max_bpm: float = MAX_BPM,
) -> bool:
    """Decide whether the tracked tempo is half of the musical one.

    Autocorrelation measures self-similarity, and a bar of kick-snare-kick-
    snare is most self-similar at *two* beats, not one -- the two-beat lag
    scored 0.99 against 0.60 for the beat itself on a 174 BPM test case. So
    ACF alone reliably lands an octave low on backbeat-driven material.

    A listener resolves this by asking whether the events halfway between the
    beats are as prominent as the beats. If they are, those midpoints are
    beats too. Comparing mean onset *strength* at beats against midpoints
    separates the cases cleanly: weak off-beat hi-hats score around 0.39,
    snares carrying a real backbeat around 0.93.

    Args:
        grid: The tracked grid to test.
        times: Detected onset times, seconds.
        strengths: Matching onset strengths.
        threshold: Midpoint-to-beat strength ratio above which the tempo is
            judged to be half the musical one.
        max_bpm: Never recommend doubling past this.

    Returns:
        True if the caller should re-track at twice `grid.bpm`.
    """
    if grid.beats.size < 4 or times.size == 0 or grid.bpm * 2.0 > max_bpm:
        return False

    tolerance = 0.10 * grid.beat_period
    beats = grid.beats
    midpoints = (beats[:-1] + beats[1:]) / 2.0

    def mean_strength(positions: np.ndarray) -> float:
        """Mean strength of the strongest onset near each position, 0 where none."""
        found = np.zeros(positions.size)
        for i, position in enumerate(positions):
            near = np.abs(times - position) <= tolerance
            if near.any():
                found[i] = strengths[near].max()
        return float(found.mean()) if found.size else 0.0

    on_beat = mean_strength(beats)
    if on_beat <= 1e-9:
        return False
    return (mean_strength(midpoints) / on_beat) > threshold


def fit_tempo_grid(beats: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit of ``time = offset + index * period`` to beat times.

    This is what a rhythm-game editor actually wants: one BPM and one offset
    that reproduce every beat. The residual doubles as a variable-tempo
    detector -- a click track fits to well under a millisecond, a live
    performance will not.

    Returns:
        ``(bpm, offset, residual_ms)``.
    """
    if beats.size < 2:
        return PRIOR_CENTER_BPM, 0.0, float("inf")

    idx = np.arange(beats.size, dtype=np.float64)
    period, offset = np.polyfit(idx, beats, 1)
    if period <= 1e-6:
        return PRIOR_CENTER_BPM, float(beats[0]), float("inf")

    residuals = beats - (offset + idx * period)
    residual_ms = float(np.sqrt(np.mean(residuals**2)) * 1000.0)

    # Report the offset as the first grid line at or after t=0.
    while offset < 0:
        offset += period
    return float(60.0 / period), float(offset), residual_ms


def detect_downbeats(
    beats: np.ndarray,
    spec: Spectral,
    meter_candidates: tuple[int, ...] = (4, 3),
) -> tuple[int, np.ndarray]:
    """Pick the bar length and phase that best explain the accent pattern.

    Bars usually start on a bass-heavy accent, so each (meter, phase) pair is
    scored by the low-band flux plus overall onset strength at the beats it
    would mark. Scores are averaged per candidate beat so meters with fewer
    downbeats are not penalised for it.

    Returns:
        ``(meter, downbeat_times)``.
    """
    if beats.size == 0:
        return meter_candidates[0], np.asarray([], dtype=np.float64)

    frames = np.clip(spec.time_to_frame(beats), 0, spec.n_frames - 1)
    odf = normalize_odf(spec.odf)
    low = spec.band_odf[0]
    low = low / max(float(np.percentile(low, 99.0)), 1e-9)

    accent = odf[frames] + 1.5 * low[frames]

    best = (-np.inf, meter_candidates[0], 0)
    for meter in meter_candidates:
        if meter < 2 or beats.size < meter:
            continue
        for phase in range(meter):
            marked = accent[phase::meter]
            if marked.size == 0:
                continue
            score = float(marked.mean())
            if score > best[0]:
                best = (score, meter, phase)

    _, meter, phase = best
    return meter, beats[phase::meter].copy()


def _autocorrelate(x: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation via FFT, normalised to acf[0] == 1."""
    n = x.size
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(x, size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    acf /= np.arange(n, 0, -1)  # correct for shrinking overlap at long lags
    if acf[0] > 1e-12:
        acf = acf / acf[0]
    return acf


def _interpolated_bpm(
    scores: np.ndarray, lags: np.ndarray, index: int, fps: float
) -> float:
    """Convert a score-array index to BPM, interpolating between integer lags.

    Lags are whole frames, so at 100 fps the reachable tempi thin out badly as
    they rise: lag 34 is 176.5 BPM and lag 35 is 171.4, with nothing between.
    A true 174 BPM peak straddles the two and reads as neither. Fitting a
    parabola across the neighbouring scores recovers the fractional lag.
    """
    if 0 < index < scores.size - 1:
        a, b, c = float(scores[index - 1]), float(scores[index]), float(scores[index + 1])
        denom = a - 2.0 * b + c
        shift = 0.0 if abs(denom) < 1e-12 else 0.5 * (a - c) / denom
        shift = float(np.clip(shift, -0.5, 0.5))
    else:
        shift = 0.0
    return float(60.0 * fps / (lags[index] + shift))


def _log_normal_prior(bpms: np.ndarray) -> np.ndarray:
    """Preference for tempi near a comfortable tapping rate."""
    z = np.log2(bpms / PRIOR_CENTER_BPM) / PRIOR_WIDTH_OCTAVES
    return np.exp(-0.5 * z**2)


def _dedupe_candidates(
    candidates: list[tuple[float, float]], tolerance_bpm: float
) -> list[tuple[float, float]]:
    """Collapse candidates that describe the same tempo to within `tolerance_bpm`."""
    out: list[tuple[float, float]] = []
    for bpm, score in candidates:
        if any(abs(bpm - kept) <= tolerance_bpm for kept, _ in out):
            continue
        out.append((bpm, score))
    return out


def _backtrace(cumscore: np.ndarray, backlink: np.ndarray) -> np.ndarray:
    """Walk back from the best endpoint, ignoring the score ramp-up at the tail."""
    is_local_max = np.r_[False, (cumscore[1:-1] >= cumscore[:-2]) &
                         (cumscore[1:-1] > cumscore[2:]), False]
    peaks = np.flatnonzero(is_local_max)
    if peaks.size == 0:
        end = int(np.argmax(cumscore))
    else:
        threshold = 0.5 * float(np.median(cumscore[peaks]))
        strong = peaks[cumscore[peaks] > threshold]
        end = int(strong[-1]) if strong.size else int(np.argmax(cumscore))

    beats = [end]
    while backlink[beats[-1]] >= 0:
        beats.append(int(backlink[beats[-1]]))
    return np.asarray(beats[::-1], dtype=np.int64)


def _trim_edge_outliers(
    beats: np.ndarray, tolerance: float = 0.12, max_trim: int = 3
) -> np.ndarray:
    """Drop leading or trailing beats whose interval is anomalous.

    The dynamic programme has no context beyond the ends of the track, so its
    first and last beats can land on whatever transient happens to be nearby
    rather than on the grid -- a 0.55 s opening interval against a 0.47 s
    period, say. Interior beats are held in place by neighbours on both sides
    and do not suffer from this, so only the edges are examined. A bad edge
    beat is worth removing: it stretches the first quantisation interval and
    drags the least-squares tempo fit with it -- a single 80 ms error was the
    entire 9.8 ms grid residual on one test track.

    Each edge interval is judged against its immediate neighbours rather than
    the track's overall median, so a piece that speeds up or slows down
    throughout is not mistaken for one with bad edges.
    """
    for _ in range(max_trim):
        if beats.size < 6:
            break
        intervals = np.diff(beats)
        leading = float(np.median(intervals[1:5]))
        trailing = float(np.median(intervals[-5:-1]))
        if leading <= 0 or trailing <= 0:
            break
        if abs(intervals[0] - leading) / leading > tolerance:
            beats = beats[1:]
            continue
        if abs(intervals[-1] - trailing) / trailing > tolerance:
            beats = beats[:-1]
            continue
        break
    return beats


def _extend_to_span(beats: np.ndarray, duration: float) -> np.ndarray:
    """Continue the grid at its own tempo to cover the whole track.

    Tracking reliably stops short of the first and last beats -- an intro
    before any steady pulse, a fade at the end. Extending at the local period
    means notes out there still quantise against real grid lines and carry
    non-negative beat numbers, instead of being extrapolated from an interval
    they sit outside of.
    """
    if beats.size < 2:
        return beats

    intervals = np.diff(beats)
    lead = float(np.median(intervals[: min(4, intervals.size)]))
    tail = float(np.median(intervals[-min(4, intervals.size):]))
    if lead <= 0 or tail <= 0:
        return beats

    before = []
    t = beats[0] - lead
    while t > 0.0:
        before.append(t)
        t -= lead

    after = []
    t = beats[-1] + tail
    while t < duration:
        after.append(t)
        t += tail

    return np.concatenate([np.asarray(before[::-1]), beats, np.asarray(after)])


def _strength_at(odf: np.ndarray, times: np.ndarray, fps: float) -> np.ndarray:
    """Sample the onset envelope at each beat time."""
    frames = np.clip(np.round(times * fps).astype(int), 0, odf.size - 1)
    return odf[frames].astype(np.float64)


def _refine_beat_times(frames: np.ndarray, odf: np.ndarray, fps: float) -> np.ndarray:
    """Snap each beat to the strongest ODF frame within +/-1 frame, sub-frame fitted."""
    times = np.empty(frames.size, dtype=np.float64)
    for i, frame in enumerate(frames):
        lo, hi = max(int(frame) - 1, 0), min(int(frame) + 2, odf.size)
        local = int(lo + np.argmax(odf[lo:hi]))
        if 0 < local < odf.size - 1:
            a, b, c = float(odf[local - 1]), float(odf[local]), float(odf[local + 1])
            denom = a - 2.0 * b + c
            shift = 0.0 if abs(denom) < 1e-12 else 0.5 * (a - c) / denom
            times[i] = (local + float(np.clip(shift, -0.5, 0.5))) / fps
        else:
            times[i] = local / fps
    return times


def _empty_grid(bpm: float, meter: int) -> BeatGrid:
    empty = np.asarray([], dtype=np.float64)
    return BeatGrid(
        beats=empty, downbeats=empty, bpm=bpm, offset=0.0, meter=meter,
        beat_strength=empty, residual_ms=float("inf"), is_constant_tempo=False,
    )
