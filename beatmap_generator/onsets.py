"""Onset peak-picking.

Turns the continuous detection function from :mod:`.dsp` into a list of
discrete, timestamped, classified hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from .dsp import BAND_NAMES, Spectral, normalize_odf


@dataclass
class Onset:
    """A single detected hit."""

    time: float                 # seconds, sub-frame interpolated
    strength: float             # normalised ODF value at the peak
    band_profile: np.ndarray    # (4,) share of flux per band group, sums to 1
    kind: str                   # kick | snare | hat | melodic
    frame: int                  # nearest ODF frame index

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 4),
            "strength": round(float(self.strength), 4),
            "kind": self.kind,
            "bands": {n: round(float(v), 4) for n, v in zip(BAND_NAMES, self.band_profile)},
        }


@dataclass
class OnsetConfig:
    """Peak-picking parameters, all in seconds unless noted."""

    pre_max: float = 0.03       # local-maximum window before the candidate
    post_max: float = 0.03      # ... and after
    pre_avg: float = 0.10       # adaptive-threshold window before
    post_avg: float = 0.07      # ... and after
    delta: float = 0.60         # threshold height in local standard deviations
    floor_ratio: float = 0.06   # absolute floor, in normalised ODF units
    min_gap: float = 0.030      # refractory period; hits closer than this merge
    sensitivity: float = 1.0    # >1 detects more, <1 detects less

    def scaled(self) -> "OnsetConfig":
        """Apply `sensitivity` to the two threshold terms."""
        if self.sensitivity == 1.0:
            return self
        s = max(self.sensitivity, 1e-3)
        return OnsetConfig(
            pre_max=self.pre_max, post_max=self.post_max,
            pre_avg=self.pre_avg, post_avg=self.post_avg,
            delta=self.delta / s, floor_ratio=self.floor_ratio / s,
            min_gap=self.min_gap, sensitivity=1.0,
        )


def detect_onsets(
    spec: Spectral,
    config: OnsetConfig | None = None,
    signal: np.ndarray | None = None,
) -> list[Onset]:
    """Pick onsets from a detection function.

    A frame is accepted when it is (a) the maximum of its local neighbourhood,
    (b) above a locally adaptive threshold, and (c) not inside the refractory
    period of an already-accepted onset. The adaptive threshold is
    ``local_mean + delta * local_std``, which is scale-free -- it tightens in
    dense passages and relaxes in sparse ones -- with a global floor so that
    near-silence does not produce onsets from noise alone.

    Peak times are then refined by fitting a parabola through the three frames
    around each peak, giving roughly 2-3 ms resolution off a 10 ms grid. If
    `signal` is supplied they are refined again against the waveform itself,
    which removes the STFT's systematic latency -- see
    :func:`refine_attack_time`.

    Args:
        spec: Output of :func:`.dsp.superflux_odf`.
        config: Peak-picking parameters. Defaults are tuned for produced music.
        signal: The mono waveform the spectrogram came from. Optional, but
            supplying it takes timing from about 6 ms of bias and 3.4 ms of
            spread down to under 1 ms.

    Returns:
        Onsets in ascending time order.
    """
    cfg = (config or OnsetConfig()).scaled()
    odf = normalize_odf(spec.odf)
    if odf.size == 0 or not np.any(odf > 0):
        return []

    fps = spec.fps
    to_frames = lambda s: max(int(round(s * fps)), 1)  # noqa: E731

    # (a) local maximum over an asymmetric window
    max_width = to_frames(cfg.pre_max) + to_frames(cfg.post_max) + 1
    origin = (to_frames(cfg.pre_max) - to_frames(cfg.post_max)) // 2
    local_max = maximum_filter1d(odf, size=max_width, mode="nearest", origin=origin)
    is_peak = odf >= local_max

    # (b) adaptive threshold from a wider window
    avg_width = to_frames(cfg.pre_avg) + to_frames(cfg.post_avg) + 1
    avg_origin = (to_frames(cfg.pre_avg) - to_frames(cfg.post_avg)) // 2
    local_mean = uniform_filter1d(odf, size=avg_width, mode="nearest", origin=avg_origin)
    local_sq = uniform_filter1d(odf**2, size=avg_width, mode="nearest", origin=avg_origin)
    local_std = np.sqrt(np.maximum(local_sq - local_mean**2, 0.0))

    # The adaptive term alone collapses wherever the music is steady: local
    # standard deviation goes to nearly zero under a held chord, and ordinary
    # ripple then clears the threshold. The ODF is normalised so that 1.0 is
    # about the height of a strong onset, so the floor can simply be an
    # absolute fraction of that -- scaling it by a percentile of this track's
    # own ODF fails exactly when onsets are sparse, because then the
    # percentile is itself measuring silence.
    threshold = local_mean + cfg.delta * local_std
    np.maximum(threshold, cfg.floor_ratio, out=threshold)

    candidates = np.flatnonzero(is_peak & (odf > threshold))
    if candidates.size == 0:
        return []

    # (c) refractory period -- keep the stronger of any two hits that collide
    kept = _enforce_min_gap(candidates, odf, to_frames(cfg.min_gap))

    onsets: list[Onset] = []
    previous_time = -np.inf
    for frame in kept:
        time = _parabolic_peak_time(odf, frame, fps)
        if signal is not None:
            time = refine_attack_time(signal, spec.sr, time, floor=previous_time)
        previous_time = time
        profile = band_profile(spec, frame)
        onsets.append(
            Onset(
                time=time,
                strength=float(odf[frame]),
                band_profile=profile.astype(np.float32),
                kind=classify_onset(profile),
                frame=int(frame),
            )
        )
    return onsets


def band_profile(spec: Spectral, frame: int, lookahead: int = 3, lookback: int = 3) -> np.ndarray:
    """Describe a hit by the energy it *adds* in each frequency band.

    Measured on linear magnitudes rather than the log-compressed flux, and
    against a reference frame just before the attack, so the result reflects
    the hit's own spectrum and not whatever was already sounding underneath.

    Args:
        spec: Spectral bundle.
        frame: ODF peak frame for the onset.
        lookahead: Frames after `frame` to search for the magnitude peak; the
            spectral peak of a hit can trail the flux peak slightly.
        lookback: Frames before `frame` to sample the background level.

    Returns:
        A non-negative ``(n_bands,)`` vector summing to 1.
    """
    n_bands, n_frames = spec.band_mag.shape
    peak_slice = spec.band_mag[:, frame : min(frame + lookahead + 1, n_frames)]
    if peak_slice.size == 0:
        return np.full(n_bands, 1.0 / n_bands, dtype=np.float32)

    peak = peak_slice.max(axis=1)
    background = spec.band_mag[:, max(frame - lookback, 0)]

    added = np.maximum(peak - background, 0.0)
    total = float(added.sum())
    if total <= 1e-12:
        return np.full(n_bands, 1.0 / n_bands, dtype=np.float32)
    return (added / total).astype(np.float32)


def _enforce_min_gap(candidates: np.ndarray, odf: np.ndarray, gap: int) -> list[int]:
    """Greedily drop the weaker of any two candidates closer than `gap` frames."""
    kept: list[int] = []
    for frame in candidates:
        if kept and frame - kept[-1] < gap:
            if odf[frame] > odf[kept[-1]]:
                kept[-1] = int(frame)  # the later, louder hit wins the slot
            continue
        kept.append(int(frame))
    return kept


def _parabolic_peak_time(odf: np.ndarray, frame: int, fps: float) -> float:
    """Sub-frame peak location via a 3-point parabola fit."""
    if frame <= 0 or frame >= odf.size - 1:
        return frame / fps
    a, b, c = float(odf[frame - 1]), float(odf[frame]), float(odf[frame + 1])
    denom = a - 2.0 * b + c
    shift = 0.0 if abs(denom) < 1e-12 else 0.5 * (a - c) / denom
    shift = float(np.clip(shift, -0.5, 0.5))
    return (frame + shift) / fps


def refine_attack_time(
    signal: np.ndarray,
    sr: int,
    coarse_time: float,
    back: float = 0.040,
    forward: float = 0.030,
    threshold: float = 0.20,
    floor: float = -np.inf,
) -> float:
    """Locate the true attack in the waveform near a spectrogram-derived time.

    A centred 46 ms analysis window starts responding to a transient well
    before the transient arrives, so ODF peaks land a few milliseconds early
    and the error varies with the attack's sharpness. Both go away if the
    waveform is consulted directly: take the energy peak near the coarse
    estimate and walk backwards to where energy last fell below `threshold` of
    it. That point is the attack.

    This holds to well under a millisecond on percussive hits, which is what
    charts are usually built from. Slow attacks are inherently looser -- a pad
    fading in over 15 ms has no single instant that *is* the onset, and the
    threshold crossing lands around 10 ms into the rise. Lowering `threshold`
    trades percussive precision for a better answer on those; the default
    favours percussion.

    Args:
        signal: Mono waveform.
        sr: Sample rate.
        coarse_time: Onset time from the ODF, in seconds.
        back: How far before `coarse_time` to search.
        forward: How far after `coarse_time` to search.
        threshold: Fraction of the local energy peak that marks attack start.
        floor: Never return a time at or before this (the previous onset), so
            refinement cannot reorder or collapse neighbouring hits.

    Returns:
        The refined time in seconds, or `coarse_time` if the window is unusable.
    """
    start = max(int((coarse_time - back) * sr), 0)
    if floor > -np.inf:
        start = max(start, int(floor * sr) + 1)
    stop = min(int((coarse_time + forward) * sr), signal.size)
    if stop - start < 32:
        return coarse_time

    energy = uniform_filter1d(
        signal[start:stop].astype(np.float64) ** 2,
        size=max(int(0.002 * sr), 3),
    )
    peak = int(np.argmax(energy))
    limit = energy[peak] * threshold
    if limit <= 0.0:
        return coarse_time

    i = peak
    while i > 0 and energy[i] > limit:
        i -= 1
    return (start + i) / sr


def classify_onset(profile: np.ndarray) -> str:
    """Label a hit from its band-energy profile.

    The four bands are low (<150 Hz), low-mid (150-600), high-mid (600-2500)
    and high (>2500). Kicks put most of their energy below 150 Hz, cymbals
    almost all of theirs above 2.5 kHz, and snares spread across the middle
    with a broadband tail. Order matters: the two unambiguous cases are
    checked before the broad middle one.

    Args:
        profile: Band shares from :func:`band_profile`, summing to 1.

    Returns:
        One of ``kick``, ``snare``, ``hat`` or ``melodic``.
    """
    low, low_mid, high_mid, high = (float(v) for v in profile)

    if low >= 0.55:
        return "kick"
    if high >= 0.60:
        return "hat"
    if (low_mid + high_mid) >= 0.35:
        return "snare"
    return "melodic"


def onset_times(onsets: list[Onset]) -> np.ndarray:
    """Extract just the timestamps."""
    return np.asarray([o.time for o in onsets], dtype=np.float64)


def onset_strengths(onsets: list[Onset]) -> np.ndarray:
    return np.asarray([o.strength for o in onsets], dtype=np.float64)
