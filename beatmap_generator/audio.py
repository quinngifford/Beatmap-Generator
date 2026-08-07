"""Audio loading and resampling.

The rest of the pipeline assumes a mono float32 signal at a known sample rate,
so everything funnels through :func:`load_audio`.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

DEFAULT_SR = 44100


@dataclass
class Audio:
    """A decoded mono signal."""

    samples: np.ndarray  # float32, shape (n,), nominally in [-1, 1]
    sr: int
    path: Path
    duration: float

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])


def load_audio(
    path: str | Path,
    sr: int = DEFAULT_SR,
    normalize: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
) -> Audio:
    """Decode `path` to mono float32 at `sr` Hz.

    Tries libsndfile (via ``soundfile``) first, then audioread, then an ffmpeg
    subprocess, then the stdlib ``wave`` module. The first three cover
    essentially every format a song is likely to arrive in.

    Args:
        path: Audio file. WAV/FLAC/OGG/MP3/AIFF and friends.
        sr: Target sample rate. 44100 keeps the default hop at an exact 100 fps.
        normalize: Scale so the waveform peaks at 0.95. Onset thresholds are
            adaptive, but normalizing keeps the log-compression knee in the same
            place for quiet and loud masters alike.
        offset: Skip this many seconds from the start.
        duration: Decode at most this many seconds (``None`` for all).

    Returns:
        An :class:`Audio` with samples already resampled to `sr`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        RuntimeError: If every decoder backend failed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")

    errors: list[str] = []
    samples: np.ndarray | None = None
    native_sr: int | None = None

    for backend in (_load_soundfile, _load_audioread, _load_ffmpeg, _load_wave_stdlib):
        try:
            result = backend(path)
        except Exception as exc:  # noqa: BLE001 - collect and try the next backend
            errors.append(f"{backend.__name__}: {type(exc).__name__}: {exc}")
            continue
        if result is not None:
            samples, native_sr = result
            break

    if samples is None or native_sr is None:
        detail = "\n  ".join(errors) if errors else "no backend available"
        raise RuntimeError(f"could not decode {path.name}. Backends tried:\n  {detail}")

    samples = _to_mono(samples)

    if offset > 0.0 or duration is not None:
        start = int(round(offset * native_sr))
        stop = None if duration is None else start + int(round(duration * native_sr))
        samples = samples[start:stop]

    samples = resample(samples, native_sr, sr)

    if samples.size == 0:
        raise RuntimeError(f"{path.name} decoded to zero samples")

    if normalize:
        peak = float(np.max(np.abs(samples)))
        if peak > 0.0:
            samples = samples * (0.95 / peak)

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    return Audio(samples=samples, sr=sr, path=path, duration=samples.size / sr)


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase-resample `y` from `sr_in` to `sr_out`."""
    if sr_in == sr_out:
        return y.astype(np.float32, copy=False)
    ratio = Fraction(sr_out, sr_in).limit_denominator(1000)
    return resample_poly(y, ratio.numerator, ratio.denominator).astype(np.float32)


def _to_mono(y: np.ndarray) -> np.ndarray:
    """Average channels down to a single one."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    # soundfile hands back (frames, channels).
    axis = 1 if y.shape[0] >= y.shape[1] else 0
    return y.mean(axis=axis, dtype=np.float32)


def _load_soundfile(path: Path) -> tuple[np.ndarray, int] | None:
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return y, int(sr)


def _load_audioread(path: Path) -> tuple[np.ndarray, int] | None:
    import audioread  # type: ignore[import-not-found]

    chunks: list[np.ndarray] = []
    with audioread.audio_open(str(path)) as fh:
        sr, channels = int(fh.samplerate), int(fh.channels)
        for chunk in fh:
            chunks.append(np.frombuffer(chunk, dtype="<i2"))
    if not chunks:
        return None
    y = np.concatenate(chunks).astype(np.float32) / 32768.0
    if channels > 1:
        y = y.reshape(-1, channels)
    return y, sr


def _load_ffmpeg(path: Path) -> tuple[np.ndarray, int] | None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    sr = DEFAULT_SR
    cmd = [
        exe, "-v", "quiet", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype="<f4").copy(), sr


def _load_wave_stdlib(path: Path) -> tuple[np.ndarray, int] | None:
    """Last resort: uncompressed PCM WAV only."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    dtype, scale = {
        1: (np.uint8, 128.0),
        2: (np.int16, 32768.0),
        4: (np.int32, 2147483648.0),
    }.get(width, (None, None))
    if dtype is None:
        raise ValueError(f"unsupported PCM sample width: {width} bytes")

    y = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        y -= 128.0
    y /= scale
    if channels > 1:
        y = y.reshape(-1, channels)
    return y, sr
