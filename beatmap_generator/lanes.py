"""Lane assignment.

Turning a list of hits into a *playable* chart is a separate problem from
detecting them. Two rules matter more than any aesthetic choice: the same lane
must not be asked to fire twice in quick succession, and a run of fast notes
must alternate sides so it can be played hand-over-hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The span of the keyboard each drum voice may occupy, as (start, end)
# fractions. Voices get a region rather than a single lane: a kick drum pinned
# to column 0 for a whole song produces a chart that is monotonous to read and
# to play, whereas rotating through two adjacent columns reads as a natural
# hand alternation and costs nothing in clarity.
KIND_REGION = {
    "kick": (0.0, 0.34),      # left side
    "snare": (0.66, 1.0),     # right side
    "hat": (0.34, 0.72),      # inside
    "melodic": (0.0, 1.0),    # placed by pitch instead
}


@dataclass
class LaneConfig:
    """Constraints the assigner has to respect."""

    n_lanes: int = 4
    min_same_lane: float = 0.120   # seconds before a lane may repeat
    force_alternate: float = 0.180 # below this gap, avoid the previous lane entirely
    spread_melodic: bool = True    # map pitched hits across lanes by brightness


def assign_lanes(
    times: np.ndarray,
    kinds: list[str],
    brightness: np.ndarray,
    config: LaneConfig | None = None,
) -> np.ndarray:
    """Assign a lane to every note.

    Each note starts from the lane its voice prefers -- kicks left, snares
    right, hi-hats inside, pitched material placed by brightness -- and then
    moves to the nearest lane that satisfies the timing constraints. Notes
    sharing a timestamp are treated as a chord and given distinct lanes.

    Args:
        times: Note times in seconds, ascending.
        kinds: Voice label per note, as produced by :func:`.onsets.classify_onset`.
        brightness: Per-note value in [0, 1] used to place pitched hits.
        config: Lane count and spacing constraints.

    Returns:
        Integer lane index per note, in ``[0, n_lanes)``.
    """
    cfg = config or LaneConfig()
    n_lanes = max(int(cfg.n_lanes), 1)
    lanes = np.zeros(times.size, dtype=int)
    last_used = np.full(n_lanes, -np.inf)
    rotation: dict[str, int] = {}
    previous_lane = -1
    previous_time = -np.inf

    for i, time in enumerate(times):
        kind = kinds[i]
        preferred = _preferred_lane(
            kind, float(brightness[i]), n_lanes, cfg, rotation.get(kind, 0)
        )
        rotation[kind] = rotation.get(kind, 0) + 1
        is_chord = (time - previous_time) < 1e-3

        # Nearest-first, so a blocked note lands beside its ideal lane.
        order = sorted(range(n_lanes), key=lambda lane: (abs(lane - preferred), lane))

        chosen = -1
        for lane in order:
            if time - last_used[lane] < cfg.min_same_lane:
                continue
            if lane == previous_lane and (time - previous_time) < cfg.force_alternate:
                continue
            if is_chord and lane == previous_lane:
                continue
            chosen = lane
            break

        if chosen < 0:
            # Everything is blocked; fall back to whichever lane is coldest.
            chosen = int(np.argmin(last_used))

        lanes[i] = chosen
        last_used[chosen] = time
        previous_lane = chosen
        previous_time = time

    return lanes


def _preferred_lane(
    kind: str, brightness: float, n_lanes: int, cfg: LaneConfig, occurrence: int
) -> int:
    """Ideal lane for a note before playability constraints are applied.

    Pitched hits are placed by brightness so a rising line walks across the
    keyboard. Percussive hits cycle through their voice's region, `occurrence`
    being the running count of notes of that kind seen so far.
    """
    if kind == "melodic" and cfg.spread_melodic:
        position = float(np.clip(brightness, 0.0, 1.0))
        return int(round(position * (n_lanes - 1)))

    start, end = KIND_REGION.get(kind, (0.0, 1.0))
    first = int(np.floor(start * (n_lanes - 1) + 0.5))
    last = int(np.floor(end * (n_lanes - 1) + 0.5))
    first, last = min(first, last), max(first, last)
    width = last - first + 1
    return first + (occurrence % width)


def brightness_from_profile(profile: np.ndarray) -> float:
    """Collapse a band profile to a single 0-1 brightness value.

    Used to spread pitched material across lanes so a rising melodic line
    walks across the keyboard instead of piling into one column.
    """
    weights = np.linspace(0.0, 1.0, profile.size)
    total = float(profile.sum())
    if total <= 1e-12:
        return 0.5
    return float(np.dot(profile, weights) / total)
