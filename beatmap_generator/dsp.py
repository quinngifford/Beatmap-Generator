"""Spectral front-end: STFT, log-frequency filterbank, and the SuperFlux
onset detection function.

The default framing (44100 Hz, hop 441) puts frames at exactly 100 fps, so
frame index and centisecond line up and every timestamp stays on a clean grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

DEFAULT_HOP = 441          # 10 ms at 44.1 kHz
DEFAULT_N_FFT = 2048       # ~46 ms window
DEFAULT_BANDS_PER_OCTAVE = 24
DEFAULT_FMIN = 27.5        # A0
DEFAULT_FMAX = 16000.0

# Frequency splits used to profile each onset. These drive both drum
# classification and the default lane mapping.
BAND_EDGES = (0.0, 150.0, 600.0, 2500.0, 22050.0)
BAND_NAMES = ("low", "low_mid", "high_mid", "high")


@dataclass
class Spectral:
    """Everything the onset/tempo stages need from the spectral front-end."""

    odf: np.ndarray          # (n_frames,) SuperFlux onset detection function
    band_odf: np.ndarray     # (n_bands, n_frames) same flux split by band group
    band_mag: np.ndarray     # (n_bands, n_frames) linear magnitude per band group
    log_spec: np.ndarray     # (n_frames, n_filters) log-compressed filtered spectrogram
    rms: np.ndarray          # (n_frames,) frame energy, for sustain detection
    fps: float
    hop: int
    sr: int

    @property
    def n_frames(self) -> int:
        return int(self.odf.shape[0])

    def frame_to_time(self, frames: np.ndarray | float) -> np.ndarray | float:
        """Convert (possibly fractional) frame indices to seconds."""
        return np.asarray(frames) / self.fps if np.ndim(frames) else frames / self.fps

    def time_to_frame(self, times: np.ndarray | float) -> np.ndarray:
        return np.round(np.asarray(times) * self.fps).astype(int)


def stft_magnitude(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Magnitude STFT with a Hann window, centred frames, reflect padding.

    Returns:
        Array of shape ``(n_frames, n_fft // 2 + 1)``.
    """
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))

    window = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)
    pad = n_fft // 2
    padded = np.pad(y, (pad, pad), mode="reflect")

    n_frames = 1 + (padded.size - n_fft) // hop
    # Strided view avoids materialising the framed copy twice.
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, n_fft),
        strides=(padded.strides[0] * hop, padded.strides[0]),
        writeable=False,
    )
    return np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)


def log_frequency_filterbank(
    sr: int,
    n_fft: int,
    bands_per_octave: int = DEFAULT_BANDS_PER_OCTAVE,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Build triangular filters spaced logarithmically in frequency.

    Constant-Q spacing matches how we hear pitch and keeps low-frequency
    detail that a mel bank would smear. Filters too narrow to contain an FFT
    bin, and duplicates of an already-emitted filter, are dropped -- otherwise
    the sub-500 Hz region would be counted many times over in the flux sum.

    Returns:
        ``(filters, centers)`` where filters has shape ``(n_filters, n_bins)``
        with each row summing to 1, and `centers` holds the centre frequency
        of each surviving filter in Hz.
    """
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    nyquist = sr / 2.0
    fmax = min(fmax, nyquist)

    n_centers = int(np.ceil(np.log2(fmax / fmin) * bands_per_octave)) + 1
    centers = fmin * 2.0 ** (np.arange(n_centers) / bands_per_octave)
    centers = centers[centers <= nyquist]
    if centers.size < 3:
        raise ValueError("filterbank needs at least 3 centre frequencies")

    filters: list[np.ndarray] = []
    kept_centers: list[float] = []
    seen: set[bytes] = set()

    for i in range(1, len(centers) - 1):
        lo, mid, hi = centers[i - 1], centers[i], centers[i + 1]
        tri = np.zeros_like(fft_freqs)

        rising = (fft_freqs > lo) & (fft_freqs <= mid)
        tri[rising] = (fft_freqs[rising] - lo) / max(mid - lo, 1e-9)
        falling = (fft_freqs > mid) & (fft_freqs < hi)
        tri[falling] = (hi - fft_freqs[falling]) / max(hi - mid, 1e-9)

        total = tri.sum()
        if total <= 0.0:
            continue  # narrower than the FFT resolution
        tri /= total

        key = (tri > 0).tobytes()
        if key in seen:
            continue  # identical support to a filter we already kept
        seen.add(key)

        filters.append(tri.astype(np.float32))
        kept_centers.append(float(mid))

    if not filters:
        raise ValueError("filterbank collapsed to zero filters; raise n_fft")
    return np.stack(filters), np.asarray(kept_centers)


def superflux_odf(
    y: np.ndarray,
    sr: int,
    hop: int = DEFAULT_HOP,
    n_fft: int = DEFAULT_N_FFT,
    max_filter_width: int = 3,
    diff_lag: int = 2,
    gamma: float = 1000.0,
    bands_per_octave: int = DEFAULT_BANDS_PER_OCTAVE,
) -> Spectral:
    """Compute the SuperFlux onset detection function.

    SuperFlux (Böck & Widmer, DAFx 2013) is spectral flux with one addition: the
    reference frame is passed through a maximum filter across frequency before
    subtraction. Vibrato and portamento then slide *within* the widened
    reference instead of registering as a fresh onset, which is what makes
    plain spectral flux over-trigger on sustained melodic material.

    Args:
        y: Mono signal.
        sr: Sample rate of `y`.
        hop: Frame advance in samples.
        n_fft: FFT size.
        max_filter_width: Width in filterbank bins of the frequency-domain
            maximum filter. 3 tolerates roughly a semitone of drift.
        diff_lag: How many frames back the reference frame sits. A lag of 2 at
            100 fps (20 ms) is more robust to soft attacks than adjacent
            frames without smearing timing.
        gamma: Log-compression strength in ``log10(1 + gamma * S)``. Higher
            values lift quiet detail; 1000 suits peak-normalised input.
        bands_per_octave: Filterbank resolution.

    Returns:
        A :class:`Spectral` bundle with the overall and per-band ODFs.
    """
    mag = stft_magnitude(y, n_fft=n_fft, hop=hop)
    filters, centers = log_frequency_filterbank(
        sr, n_fft, bands_per_octave=bands_per_octave
    )

    filtered = mag @ filters.T                      # (n_frames, n_filters)
    log_spec = np.log10(1.0 + gamma * filtered).astype(np.float32)

    # Widen the reference frame across frequency, then difference against it.
    reference = maximum_filter1d(log_spec, size=max_filter_width, axis=1, mode="nearest")
    diff = np.empty_like(log_spec)
    diff[:diff_lag] = 0.0
    diff[diff_lag:] = log_spec[diff_lag:] - reference[:-diff_lag]
    np.maximum(diff, 0.0, out=diff)                 # only rises count as onsets

    odf = diff.sum(axis=1)

    # Group the same rectified flux into coarse bands for onset profiling.
    #
    # Average rather than sum. Log spacing plus duplicate removal leaves the
    # bands with very unequal filter counts (about 6 below 150 Hz against 64
    # above 2.5 kHz), so a plain sum would measure filterbank layout rather
    # than spectral shape -- and would label every kick as high-frequency.
    band_odf = np.zeros((len(BAND_NAMES), log_spec.shape[0]), dtype=np.float32)
    band_mag = np.zeros((len(BAND_NAMES), log_spec.shape[0]), dtype=np.float32)
    for b, (lo, hi) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
        mask = (centers >= lo) & (centers < hi)
        if mask.any():
            band_odf[b] = diff[:, mask].mean(axis=1)
            # Linear magnitude as well. Log flux is unusable for describing
            # timbre because log10(1 + gamma*S) is steepest at zero: a band
            # rising out of true silence yields tens of times more flux than
            # the same absolute rise over a sustained floor, so an identical
            # snare profiles differently depending on whether a bassline
            # happens to be playing under it.
            band_mag[b] = filtered[:, mask].mean(axis=1)

    return Spectral(
        odf=odf.astype(np.float32),
        band_odf=band_odf,
        band_mag=band_mag,
        log_spec=log_spec,
        rms=_frame_rms(y, hop=hop, n_fft=n_fft, n_frames=log_spec.shape[0]),
        fps=sr / hop,
        hop=hop,
        sr=sr,
    )


def _frame_rms(y: np.ndarray, hop: int, n_fft: int, n_frames: int) -> np.ndarray:
    """Per-frame RMS on the same grid as the STFT.

    Computed from a running sum of squares rather than frame by frame. A
    strided view would be the obvious vectorisation but materialises an
    ``(n_frames, n_fft)`` array -- half a gigabyte for a five-minute track --
    whereas prefix sums give the same windowed means in one pass.
    """
    pad = n_fft // 2
    padded = np.pad(y, (pad, pad), mode="reflect").astype(np.float64)

    available = 1 + max(padded.size - n_fft, 0) // hop
    count = min(n_frames, available)
    out = np.zeros(n_frames, dtype=np.float32)
    if count <= 0:
        return out

    cumulative = np.concatenate([[0.0], np.cumsum(padded**2)])
    starts = np.arange(count) * hop
    windows = cumulative[starts + n_fft] - cumulative[starts]
    out[:count] = np.sqrt(np.maximum(windows, 0.0) / n_fft)
    if count < n_frames:
        out[count:] = out[count - 1]
    return out


def normalize_odf(odf: np.ndarray) -> np.ndarray:
    """Scale an ODF to a roughly unit range using a robust high quantile.

    Peak-picking thresholds are expressed in these units, so a scale-free ODF
    keeps one set of defaults working across wildly different masters.
    """
    scale = float(np.percentile(odf, 99.0))
    if scale <= 1e-9:
        scale = float(np.max(odf))
    if scale <= 1e-9:
        return np.zeros_like(odf)
    return (odf / scale).astype(np.float32)
