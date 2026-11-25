"""The collision-specific behavioural markers, computed identically for human and simulated episodes.

Every function here reads TELEMETRY-FORMAT frames -- the dictionaries
``CarMDP.observationHuman`` produces, which is the format the human data was captured in
AND the format ``validate.RecordingEnv`` records. One implementation therefore serves both
sides of every comparison, which is the property that makes a marker usable for validation
at all (implementation.md Entry 32).

This module deliberately does NOT recompute lane-change rate or median speed. Those belong
to ``validate.behaviour_metrics``, which remains their single definition; ``validate.py``
merges the two dicts.

Public API:
    evasion_markers     -- the two collision-specific shares validate.py reports, plus the
                           two per-1,000-step rates behind them
    evasion_series      -- the lane-change and hazard-entry counts behind them
    car_threat_active   -- the per-frame car-threat flag
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drive_env.core.config import FPS, TILE_SIZE_M   # step rate and lane width


# ===========================================================================
# 1. Module-level constants
# ===========================================================================

# | Ego half-width, from ``CarMDP.__init__`` (``p_h_w``, mdp.py:325). Used for the lateral
# | corridor test below.
PLAYER_HALF_WIDTH_M = 0.47

# | Minimum frames before an episode is summarized at all, matching
# | ``validate.behaviour_metrics`` so both halves of the merged marker dict appear or vanish
# | together rather than one side silently producing numbers on a 5-frame episode.
MIN_FRAMES = 20

# | How far ahead a car counts as a threat, centre to centre. Taken from the simulator's own
# | warning test, which fires when a crashable object is 0-6 m ahead inside the ego corridor
# | (mdp.py:607). The range is the environment's, not a chosen value; only the restriction to
# | cars is this module's.
CAR_THREAT_RANGE_M = 6.0

# | How far back a lane change may look for the threat that prompted it. A lane change is
# | called EVASIVE when the simulator had flagged a crashable object close ahead within this
# | window before it. The value is a reaction-and-manoeuvre allowance, not a physical
# | constant, so it is exposed as an argument; the marker's sensitivity to it is reported in
# | implementation.md Entry 32 rather than assumed to be small.
THREAT_LOOKBACK_S = 0.5

# | Rates over quantities observable only on RECORDED frames are expressed per 1000 of them,
# | matching the existing ``lane_changes_per_1k`` convention (Rule 19). Rates over EVENTS use
# | simulator steps instead, because every event step is recorded while ordinary steps are
# | thinned -- see ``evasion_markers``.
FRAMES_PER_RATE_UNIT = 1000.0


# ===========================================================================
# 2. Per-frame series and the episode-level markers
# ===========================================================================

def is_tracked_car(name: str) -> bool:
    """Is this object kind a car that travels ALONG the road, rather than across it?

    One predicate for both places that ask -- the car-threat test and its crash-frame
    fold-in -- so they cannot disagree about what a car is. The environment has no
    road-crossing car type, so every car kind counts.

    :param name: Object kind name, or a crash event's kind with the prefix already removed.
    :type name: str
    :return: True for a road-following car.
    :rtype: bool
    """
    return name.startswith("car") and "horizontal" not in name


def _player_lane(frames: list[dict[str, Any]]) -> np.ndarray:
    """Lane index of the player in each frame.

    The player's lateral position is snapped to lane centres (mdp.py:561), so flooring by the
    tile size recovers the lane exactly. This is the same rule ``validate.behaviour_metrics``
    uses to count lane changes, restated here because these markers need to CLASSIFY each
    change rather than count it.

    :param frames: Telemetry-format frames.
    :type frames: list[dict[str, Any]]
    :return: Lane index per frame, ``(n_frames,)``.
    :rtype: np.ndarray
    """
    px = np.array([f["player"]["x"] for f in frames], dtype=float)   # (n_frames,)
    return np.floor(px / TILE_SIZE_M)                                # (n_frames,)


def car_threat_active(frames: list[dict[str, Any]]) -> np.ndarray:
    """Per frame, was a CAR close ahead in the ego's own corridor?

    This is the simulator's own threat test (mdp.py:607) with one restriction: only cars
    count. The environment's ``collidable_ahead`` event cannot be used directly, because it
    fires for ANY crashable object, and MEASURED on the human data that includes deer and
    landslides as well as the three enemy-car kinds. Mixing a moving car you must track with
    a signposted static obstacle would make every marker built on it uninterpretable, and
    would be inconsistent with ``crashes_car`` being the default crash measure.

    The geometry is the simulator's: its lateral corridor test, and a centre-to-centre distance of 0 to
    :data:`CAR_THREAT_RANGE_M` ahead. Crash frames are folded in, restricted to car crashes,
    because the single event slot lets a crash overwrite the very threat that produced it and
    a frame you hit a car on was certainly a car-threat frame.

    NOT EXACT, by about 5%. An earlier version of this docstring claimed every car-threat step
    is a recorded frame, reasoning that such a step always raises an event and every event
    step is kept. The premise fails: the simulator's corridor test runs on ``new_x``, the
    POST-snap lateral position of that step (mdp.py:588-600), while this function reads the
    frame's PRE-step x. MEASURED over the 230 human episodes, 1,971 of 75,036 car-threat
    frames carry no event at all, every one of them at a step the thinning rule happened to
    keep; 99.4% are immediately followed by a lane change, and 91.7% disappear when the next
    frame's x is substituted. Since a no-event step survives with probability 1/3, roughly
    5% of true car-threat steps are invisible here.

    The consequence is bounded and does not flip any result: `car_threat_share` and the entry
    rate undercount by about 5%, and `crash_per_car_threat` is correspondingly overstated
    because its numerator is complete. Both sides run the identical rule, so the comparison
    stays fair; the model changes lanes about 30% more often than the human, which makes its
    undercount slightly larger, a differential of roughly 1.5 percentage points on the
    denominator. Do NOT "fix" this by reverting to the ``collidable_ahead`` event, which would
    reintroduce the deer and landslides this function exists to exclude.

    :param frames: Telemetry-format frames.
    :type frames: list[dict[str, Any]]
    :return: Boolean per frame, ``(n_frames,)``.
    :rtype: np.ndarray
    """
    out = np.zeros(len(frames), dtype=bool)                          # (n_frames,)
    for i, frame_ in enumerate(frames):
        event = str(frame_.get("info", {}).get("event", ""))
        if event.startswith("crash:") and is_tracked_car(event.removeprefix("crash:")):
            out[i] = True
            continue
        player = frame_["player"]
        for obj in frame_.get("objects", ()):
            if not obj.get("crashable") or not is_tracked_car(str(obj.get("name", ""))):
                continue
            if abs(player["x"] - obj["x"]) >= 0.5 * float(obj["width"]) + PLAYER_HALF_WIDTH_M:
                continue
            if 0.0 < player["y"] - obj["y"] < CAR_THREAT_RANGE_M:
                out[i] = True
                break
    return out


def evasion_series(
    frames: list[dict[str, Any]],
    step_indices: np.ndarray,
    threat_lookback_s: float = THREAT_LOOKBACK_S,
    car_threat: np.ndarray | None = None,
) -> dict[str, int]:
    """Classify lane changes by whether they were evasive, and hazard entries by direction.

    Two questions, both motivated by the fitted model's two dominant coefficients being a
    collision penalty and a cognitive cost: if those drive the behaviour, the model should
    reproduce variability specifically AROUND collisions rather than in driving at large.

    1. **Was a lane change evasive?** A change counts as evasive when the simulator had a
       crashable object flagged close ahead within ``threat_lookback_s`` before it. This
       separates swerving away from a car from changing lane to reach a flag, which the raw
       lane-change count pools together.
    2. **How was a hazard entered?** At the moment a threat first appears, either the player
       stayed in their lane and closed on something ahead (LONGITUDINAL entry, driving into
       it), or their lane index changed and they arrived beside something already there
       (LATERAL entry, shifting into it). These are different attention failures: the first
       is a misjudged gap ahead, the second is a lane the driver did not check.

    NOTE on the frame convention: ``RecordingEnv`` stores the PRE-step state and attaches the
    event from the step taken out of it, so a lane change first appears as ``lane[i] !=
    lane[i-1]``, caused by the action taken at frame ``i-1``. The threat that prompted it is
    therefore looked for at or before frame ``i-1``, never at frame ``i``.

    :param frames: Telemetry-format frames.
    :type frames: list[dict[str, Any]]
    :param step_indices: Simulator step index of each frame, ``(n_frames,)``, which supplies
        the real elapsed time the lookback window is measured in.
    :type step_indices: np.ndarray
    :param threat_lookback_s: Seconds before a lane change in which a flagged threat makes
        it evasive. Ex: ``0.5``.
    :type threat_lookback_s: float
    :param car_threat: Per-frame car-threat flags from :func:`car_threat_active`, when the
        caller has already computed them. None recomputes them.
    :type car_threat: np.ndarray | None
    :return: ``{"lane_changes", "evasive_lane_changes", "hazard_entries",
        "entries_longitudinal", "entries_lateral"}``, all counts.
    :rtype: dict[str, int]
    """
    lane    = _player_lane(frames)                                   # (n_frames,)
    # | Recomputed only when the caller has not already done it; the scan is over every
    # | object of every frame, so doing it twice per episode is a real cost in a replay.
    threat  = car_threat_active(frames) if car_threat is None else car_threat   # (n_frames,)
    steps   = np.asarray(step_indices, dtype=float)                  # (n_frames,)
    window  = threat_lookback_s * float(FPS)                         # in simulator steps
    out = {"lane_changes": 0, "evasive_lane_changes": 0,
           "hazard_entries": 0, "entries_longitudinal": 0, "entries_lateral": 0}

    for i in range(1, len(frames)):
        changed_lane = lane[i] != lane[i - 1]

        # | 1. Lane changes, split by whether a threat was flagged in the lookback window.
        if changed_lane:
            out["lane_changes"] += 1
            # | Anchored at frame i-1, the state the change was decided in, not at frame i
            # | where it first shows. Anchoring at i shortens the effective window by one
            # | frame gap -- MEASURED 12-14 steps instead of the intended 15.
            earliest = steps[i - 1] - window
            recent   = (steps[:i] >= earliest)                       # (i,) selection
            if bool(np.any(threat[:i][recent])):
                out["evasive_lane_changes"] += 1

        # | 2. Hazard entries: the frame a threat first appears, classified by whether the
        # |    player arrived by staying put and closing, or by moving sideways into it.
        if threat[i] and not threat[i - 1]:
            out["hazard_entries"] += 1
            if changed_lane: out["entries_lateral"] += 1
            else:            out["entries_longitudinal"] += 1
    return out


def evasion_markers(
    frames: list[dict[str, Any]],
    step_indices: np.ndarray | None,
    threat_lookback_s: float = THREAT_LOOKBACK_S,
    car_threat: np.ndarray | None = None,
) -> dict[str, float]:
    """The four collision-specific markers, as episode-level numbers.

    ``estimation/validate.py`` -- the V4 fit's own behavioural validation -- reports the two
    SHARES in its regression plots and writes all four to its CSV.

    :param frames: Telemetry-format frames for one episode.
    :type frames: list[dict[str, Any]]
    :param step_indices: Simulator step index of each frame, ``(n_frames,)``. Without it all
        four are NaN, since three of them are rates over simulator steps and the fourth needs
        the real elapsed time for its lookback window.
    :type step_indices: np.ndarray | None
    :param threat_lookback_s: Seconds before a lane change in which a flagged car threat makes
        it evasive. Ex: ``0.5``.
    :type threat_lookback_s: float
    :param car_threat: Per-frame flags from :func:`car_threat_active`, when the caller has
        already computed them. None recomputes them.
    :type car_threat: np.ndarray | None
    :return: ``{"evasive_lane_change_share", "evasive_lane_changes_per_1k_steps",
        "car_entry_longitudinal_share", "car_threat_entries_per_1k_steps"}``.
    :rtype: dict[str, float]
    """
    out: dict[str, float] = dict.fromkeys(
        ("evasive_lane_change_share", "evasive_lane_changes_per_1k_steps",
         "car_entry_longitudinal_share", "car_threat_entries_per_1k_steps"), float("nan"))
    if step_indices is None or len(frames) < MIN_FRAMES:
        return out

    evasion = evasion_series(frames, step_indices, threat_lookback_s, car_threat)
    n_steps = float(step_indices[-1] - step_indices[0] + 1)
    if evasion["lane_changes"]:
        out["evasive_lane_change_share"] = (evasion["evasive_lane_changes"]
                                            / evasion["lane_changes"])
    out["evasive_lane_changes_per_1k_steps"] = (evasion["evasive_lane_changes"]
                                                / n_steps * FRAMES_PER_RATE_UNIT)
    out["car_threat_entries_per_1k_steps"] = (evasion["hazard_entries"]
                                              / n_steps * FRAMES_PER_RATE_UNIT)
    # | Reported as a SHARE rather than the longitudinal-to-lateral RATIO the question was
    # | posed as. The two carry the same ordering (share = ratio / (1 + ratio)), so nothing
    # | is lost, but the ratio is unbounded and undefined for any episode with no lateral
    # | entry, which would make its correlation hostage to a handful of divide-by-zeros.
    directed = evasion["entries_longitudinal"] + evasion["entries_lateral"]
    if directed:
        out["car_entry_longitudinal_share"] = evasion["entries_longitudinal"] / directed
    return out
