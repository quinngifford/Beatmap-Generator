"""Synthesise test tracks with exactly known beat and onset times.

Ground truth lets us measure detection accuracy properly instead of eyeballing
a waveform. Run directly to write the fixtures into ``tests/fixtures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SR = 44100
FIXTURES = Path(__file__).parent / "fixtures"


def _env(n: int, attack: float, decay: float, sr: int = SR) -> np.ndarray:
    """Percussive amplitude envelope: fast linear attack, exponential decay.

    The tail is forced to zero over the last 50 ms. Without a release the
    exponential is still at a few percent when the buffer ends, and that
    discontinuity is a real broadband transient -- one the onset detector
    correctly fires on, which would then score as a phantom false positive
    against ground truth. The release has to be long relative to the lowest
    partial to be gentle: 10 ms is less than one cycle of a 55 Hz bass note
    and still splatters energy across the spectrum, whereas 50 ms is roughly
    three cycles and is spectrally quiet.
    """
    t = np.arange(n) / sr
    a = int(max(attack * sr, 1))
    env = np.exp(-t / decay)
    env[:a] *= np.linspace(0.0, 1.0, a)
    fade = int(min(0.050 * sr, n // 2))
    if fade > 1:
        env[-fade:] *= np.linspace(1.0, 0.0, fade)
    return env


def kick(sr: int = SR) -> np.ndarray:
    n = int(0.30 * sr)
    t = np.arange(n) / sr
    freq = 110.0 * np.exp(-t / 0.03) + 45.0  # pitch drop
    body = np.sin(2 * np.pi * np.cumsum(freq) / sr)
    click = np.random.default_rng(0).standard_normal(n) * np.exp(-t / 0.002)
    return ((body + 0.3 * click) * _env(n, 0.001, 0.09)).astype(np.float32)


def snare(sr: int = SR) -> np.ndarray:
    n = int(0.22 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(n)
    # Crude band emphasis around 1.5-3 kHz, plus a 200 Hz shell tone.
    noise = np.convolve(noise, np.array([1.0, -0.6, 0.25]), mode="same")
    tone = 0.5 * np.sin(2 * np.pi * 200 * t)
    return ((noise + tone) * _env(n, 0.0005, 0.055)).astype(np.float32)


def hat(sr: int = SR, decay: float = 0.02) -> np.ndarray:
    n = int(0.10 * sr)
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(n)
    noise = np.diff(noise, prepend=0.0)  # first difference = crude high-pass
    noise = np.diff(noise, prepend=0.0)
    return (noise * _env(n, 0.0002, decay)).astype(np.float32)


def bass_note(freq: float, dur: float, sr: int = SR) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    wave = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    return (wave * _env(n, 0.004, dur / 2.5)).astype(np.float32)


def _add(buf: np.ndarray, sample: np.ndarray, time: float, gain: float, sr: int = SR):
    start = int(round(time * sr))
    end = min(start + sample.size, buf.size)
    if start < buf.size:
        buf[start:end] += sample[: end - start] * gain


def make_drum_track(
    bpm: float = 128.0,
    bars: int = 16,
    offset: float = 0.30,
    with_bass: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """A 4/4 pattern: kick on 1 and 3, snare on 2 and 4, hats on eighths.

    Returns the signal plus a ground-truth dict of beat times and onset times.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    total = offset + bars * 4 * beat + 1.0
    buf = np.zeros(int(total * SR), dtype=np.float32)

    k, s, h = kick(), snare(), hat()
    beats: list[float] = []
    onsets: list[float] = []
    labels: list[str] = []

    scale = [55.00, 61.74, 65.41, 73.42]  # A1 B1 C2 D2

    for bar in range(bars):
        for b in range(4):
            t = offset + (bar * 4 + b) * beat
            beats.append(t)

            if b in (0, 2):
                _add(buf, k, t, 1.0)
                onsets.append(t)
                labels.append("kick")
            if b in (1, 3):
                _add(buf, s, t, 0.7)
                onsets.append(t)
                labels.append("snare")

            for eighth in (0.0, 0.5):
                th = t + eighth * beat
                _add(buf, h, th, 0.35)
                if eighth != 0.0:  # on-beat hats coincide with kick/snare
                    onsets.append(th)
                    labels.append("hat")

            if with_bass and b % 2 == 0:
                freq = scale[int(rng.integers(0, len(scale)))]
                _add(buf, bass_note(freq, beat * 1.8), t, 0.45)

    order = np.argsort(onsets)
    truth = {
        "bpm": bpm,
        "offset": offset,
        "meter": 4,
        "beats": [round(float(t), 6) for t in beats],
        "downbeats": [round(float(beats[i]), 6) for i in range(0, len(beats), 4)],
        "onsets": [round(float(onsets[i]), 6) for i in order],
        "labels": [labels[i] for i in order],
    }
    peak = float(np.max(np.abs(buf))) or 1.0
    return (buf * (0.9 / peak)).astype(np.float32), truth


def make_sustained_track(
    bpm: float = 100.0, bars: int = 8, offset: float = 0.25
) -> tuple[np.ndarray, dict]:
    """Sustained chords over a sparse kick -- exercises hold-note detection.

    Each chord is held for most of a bar, so a correct chart should mark holds
    rather than a string of taps.
    """
    beat = 60.0 / bpm
    total = offset + bars * 4 * beat + 1.5
    buf = np.zeros(int(total * SR), dtype=np.float32)
    k = kick()

    chords = [(261.63, 329.63, 392.00), (220.00, 277.18, 329.63),
              (174.61, 220.00, 261.63), (196.00, 246.94, 293.66)]
    onsets: list[float] = []
    holds: list[tuple[float, float]] = []

    for bar in range(bars):
        t = offset + bar * 4 * beat
        dur = 3.5 * beat
        for freq in chords[bar % len(chords)]:
            n = int(dur * SR)
            tt = np.arange(n) / SR
            # Slow attack, long plateau, gentle release -> genuinely sustained.
            wave = np.sin(2 * np.pi * freq * tt) + 0.4 * np.sin(4 * np.pi * freq * tt)
            env = np.ones(n)
            a = int(0.015 * SR)
            env[:a] = np.linspace(0.0, 1.0, a)
            r = int(0.25 * SR)
            env[-r:] = np.linspace(1.0, 0.0, r)
            _add(buf, (wave * env).astype(np.float32), t, 0.28)
        onsets.append(t)
        holds.append((t, dur))

        _add(buf, k, t, 0.9)
        _add(buf, k, t + 2 * beat, 0.7)
        onsets.append(t + 2 * beat)

    peak = float(np.max(np.abs(buf))) or 1.0
    truth = {
        "bpm": bpm,
        "offset": offset,
        "meter": 4,
        "beats": [round(offset + i * beat, 6) for i in range(bars * 4)],
        "onsets": [round(float(x), 6) for x in sorted(onsets)],
        "holds": [[round(a, 6), round(b, 6)] for a, b in holds],
    }
    return (buf * (0.9 / peak)).astype(np.float32), truth


def make_varispeed_track(
    bpm_start: float = 100.0, bpm_end: float = 130.0, beats_total: int = 96
) -> tuple[np.ndarray, dict]:
    """A track that accelerates linearly -- exercises the variable-tempo path."""
    times: list[float] = []
    t = 0.5
    for i in range(beats_total):
        times.append(t)
        bpm = bpm_start + (bpm_end - bpm_start) * i / max(beats_total - 1, 1)
        t += 60.0 / bpm

    buf = np.zeros(int((times[-1] + 1.5) * SR), dtype=np.float32)
    k, s, h = kick(), snare(), hat()
    for i, bt in enumerate(times):
        if i % 4 in (0, 2):
            _add(buf, k, bt, 1.0)
        else:
            _add(buf, s, bt, 0.7)
        _add(buf, h, bt, 0.3)

    peak = float(np.max(np.abs(buf))) or 1.0
    truth = {
        "bpm": float(np.mean([bpm_start, bpm_end])),
        "beats": [round(float(x), 6) for x in times],
        "onsets": [round(float(x), 6) for x in times],
        "variable_tempo": True,
    }
    return (buf * (0.9 / peak)).astype(np.float32), truth


def write_fixtures() -> None:
    import soundfile as sf

    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, (audio, truth) in {
        "drums_128": make_drum_track(bpm=128.0, bars=16),
        "drums_90_nobass": make_drum_track(bpm=90.0, bars=12, offset=0.117, with_bass=False),
        "drums_174": make_drum_track(bpm=174.0, bars=20, offset=0.05, seed=3),
        "sustained_100": make_sustained_track(),
        "varispeed": make_varispeed_track(),
    }.items():
        sf.write(FIXTURES / f"{name}.wav", audio, SR)
        (FIXTURES / f"{name}.truth.json").write_text(json.dumps(truth, indent=2))
        print(f"wrote {name}.wav  ({audio.size / SR:.1f}s, {len(truth['onsets'])} onsets)")


if __name__ == "__main__":
    write_fixtures()
