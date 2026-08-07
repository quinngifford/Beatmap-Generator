# Beatmap Generator

Takes an audio file and produces precisely timestamped rhythm-game charts:
every significant sound located in the waveform, a tempo grid fitted to the
song, and playable note charts at four difficulties.

Pure Python on numpy/scipy. One binary dependency (`soundfile`) for decoding.
No model weights, no GPU, no network access. About **200x realtime** — a
five-minute song analyses in 1.5 seconds.

## Install

```bash
pip install -r requirements.txt
```

## Use

```bash
python -m beatmap_generator song.mp3 -o charts/ --click
```

That writes `charts/song.json`, an `.osu` and `.csv` per difficulty, and a
click track for checking the result by ear.

Real output, on the 174 BPM test fixture:

```
drums_174  (28.6s)
--------------------------------------------------------------
  tempo      174.04 BPM
  offset     44 ms
  meter      4/4   confidence 0.43
  beats      84  (21 downbeats)
  onsets     160  (66 hat, 52 kick, 40 snare, 2 melodic)

  difficulty     notes    n/s   holds  quantised
  easy              35   1.30       0   100.0%
  normal            80   2.94       0   100.0%
  hard             128   4.70       0   100.0%
  expert           160   5.84       0   100.0%
```

As a library:

```python
from beatmap_generator import generate_beatmap

beatmap = generate_beatmap("song.mp3")
print(beatmap.bpm, beatmap.offset)

for note in beatmap.difficulties["hard"].notes:
    print(note.time, note.lane, note.kind)      # 12.3841 2 'snare'
```

To get detection results without building charts:

```python
from beatmap_generator.pipeline import analyze

audio, spec, onsets, grid, tempo = analyze("song.mp3")
print([o.time for o in onsets])   # every hit, in seconds
print(grid.beats)                 # the beat grid
```

## Accuracy

Measured against synthetic tracks with exactly known event times
(`tests/make_test_audio.py`). Onsets are matched one-to-one at the standard
±25 ms tolerance.

| Track | F-measure | Timing bias | Spread | Worst | Tempo (true) |
|---|---|---|---|---|---|
| drums, 128 BPM, with bass | 1.000 | −0.30 ms | 0.21 ms | 0.8 ms | 127.999 (128) |
| drums, 90 BPM, no bass | 1.000 | −0.27 ms | 0.13 ms | 0.4 ms | 90.004 (90) |
| drums, 174 BPM | 1.000 | −0.28 ms | 0.18 ms | 0.6 ms | 174.041 (174) |
| sustained chords, 100 BPM | 1.000 | +9.86 ms | 6.41 ms | 17.4 ms | 50.015 — see below |
| accelerating, 100→130 BPM | 1.000 | −0.15 ms | 0.13 ms | 0.3 ms | flagged variable |

Every onset in every fixture is found, with nothing spurious. Drum
classification (kick / snare / hat) is 94.5% accurate; kicks and snares are
100%.

The sustained fixture reports 50 BPM rather than 100 because it only places
events every two beats, so 1.2 s genuinely *is* its event period — the octave
correction correctly declines to double a tempo whose midpoints are empty. Its
larger timing spread is the slow-attack effect described below.

Two caveats worth knowing:

- **Slow attacks are inherently looser.** A pad that fades in over 15 ms has
  no single sample that *is* the onset, and detection lands partway up the
  rise (~10 ms). Percussive hits — what charts are usually built from — are
  sub-millisecond.
- **Tempo octaves are ambiguous by nature.** 174 BPM and 87 BPM describe the
  same grid; only the labelling differs. There is a correction for this
  (below), but if you know the tempo, `--bpm` skips the guessing entirely.

Run the suite with `pytest` (57 tests; fixtures are generated on first run).

## How it works

**Onsets — SuperFlux.** A log-frequency filterbank (24 bands/octave) feeds a
spectral flux difference in which the reference frame is first widened by a
maximum filter across frequency. Vibrato then slides *within* the widened
reference instead of registering as a new note, which is the failure mode that
makes plain spectral flux over-trigger on sustained melodic material. Peaks are
picked against a locally adaptive threshold (`local_mean + δ·local_std`, with an
absolute floor).

**Timing — waveform refinement.** The STFT's centred 46 ms window responds to a
transient before it arrives, so ODF peaks land about 6 ms early with 3.4 ms of
spread. Each onset is therefore re-measured directly against the waveform: find
the local energy peak, walk back to where energy last fell below 20% of it.
That takes timing to −0.3 ms bias and 0.2 ms spread — a **20x improvement**,
and the single largest accuracy win in the pipeline.

**Tempo and beats.** Tempo comes from harmonic-weighted autocorrelation with a
log-normal prior at 120 BPM, interpolated between integer lags (at 100 fps the
reachable tempi are 5 BPM apart near 174, so a true peak otherwise straddles two
lags and reads as neither). Beats come from Ellis's dynamic-programming tracker,
which optimises the whole sequence at once so one ambiguous bar cannot knock the
grid permanently out of phase. Least squares over the tracked beats then yields
the BPM and offset an editor wants, and the fit residual doubles as a
variable-tempo detector.

**The octave correction.** Autocorrelation measures self-similarity, and a bar
of kick-snare-kick-snare is most self-similar at *two* beats rather than one —
on the 174 BPM test track the two-beat lag scores 0.99 against 0.60 for the beat
itself. ACF alone therefore lands an octave low on any backbeat-driven music. The
fix is the question a listener asks: are the events halfway between the beats as
strong as the beats? Weak off-beat hi-hats score 0.39; snares carrying a real
backbeat score 0.93. Above 0.72, the tempo is doubled.

**Charts.** Onsets are snapped to subdivisions of the *tracked* beats rather
than a fitted line, so tracks that drift still quantise correctly; notes further
than 50 ms from any grid position keep their detected time, since a deliberate
flam is better left where it was played. Difficulty tiers filter by onset
strength and cap density, selecting strongest-first so the musically prominent
hit survives. Lanes are assigned from the hit's spectral profile — kicks left,
snares right, cymbals inside, pitched material spread by brightness — rotating
within each region and refusing to repeat a lane too quickly, so charts stay
playable hand-over-hand. Sustained audio becomes hold notes.

## Options

```
-d, --difficulties   tiers to build (easy normal hard expert)
-k, --lanes          key count; any number, 4 by default
-s, --sensitivity    >1 detects more onsets, <1 fewer
    --bpm            force a known tempo, skipping estimation
    --min-bpm/--max-bpm   constrain the search (e.g. 150 200 for drum & bass)
    --meter          beats per bar, 4 by default
    --no-quantize    keep raw detected times, no grid snapping
    --max-shift      largest quantisation correction, default 0.050 s
    --no-octave-fix  disable the half-tempo correction
    --no-refine      skip waveform refinement (faster, ~6 ms less accurate)
    --click          render a click track over the music
    --click-source   click the beat grid, or the notes of a difficulty
```

Tuning notes: raise `--sensitivity` for sparse ambient material, lower it for
dense or distorted mixes. If the reported tempo is half or double what you
expect, pass `--bpm` — note placement is unaffected either way, since both
describe the same grid.

## Output

`song.json` is canonical and holds everything:

```json
{
  "tempo": { "bpm": 174.0409, "offset": 0.0442, "meter": 4,
             "constant": true, "grid_residual_ms": 5.36, "confidence": 0.425 },
  "beats": [0.0404, 0.3877, 0.73],
  "downbeats": [0.0404, 1.4196],
  "onsets": [ { "time": 0.05, "strength": 0.9844, "kind": "kick",
                "bands": { "low": 0.8725, "low_mid": 0.0939,
                           "high_mid": 0.0112, "high": 0.0224 } } ],
  "difficulties": {
    "hard": { "lanes": 4, "note_count": 128, "notes_per_second": 4.698,
              "notes": [ { "time": 0.0404, "lane": 0, "type": "tap",
                           "kind": "kick", "strength": 0.984, "beat": 0.0 } ] }
  }
}
```

`.osu` files are osu!mania and open directly in the editor — a good way to
audition and hand-tune a generated chart. Variable-tempo songs get a timing
point per beat so the editor's grid follows the performance. `.csv` is a flat
one-row-per-note table for importing into your own engine.

## Layout

```
beatmap_generator/
  audio.py      decoding and resampling, with decoder fallbacks
  dsp.py        STFT, log-frequency filterbank, SuperFlux ODF
  onsets.py     peak picking, attack refinement, drum classification
  tempo.py      tempo estimation, DP beat tracking, downbeats
  lanes.py      lane assignment and playability constraints
  beatmap.py    quantisation, difficulty tiers, hold notes
  pipeline.py   end-to-end orchestration
  export.py     JSON, osu!mania, CSV, click track
  cli.py        command line interface
tests/
  make_test_audio.py   synthesises fixtures with exact ground truth
  test_pipeline.py     accuracy and behaviour tests
```

To swap in a neural onset detector, replace `superflux_odf` with anything
returning a `Spectral` whose `odf` is a per-frame novelty curve; peak picking,
tempo, and chart construction are all downstream of that one array.

## References

- Böck & Widmer (2013), *Maximum Filter Vibrato Suppression for Onset
  Detection* — the SuperFlux ODF.
- Ellis (2007), *Beat Tracking by Dynamic Programming* — the beat tracker.
- Böck et al. (2012), *Evaluating the Online Capabilities of Onset Detection
  Methods* — the adaptive peak-picking scheme.
