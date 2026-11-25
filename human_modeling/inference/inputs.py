"""VENDORED. Lineage: human_modeling/hlp_v2/inputs.py -> hlp_v3 -> here (2026-09-14,
implementation.md Entry 28). Both earlier packages are retired and deleted; their
design logs are in human_modeling/version_history.md. This package imports from no
other inference version and must not start.

inputs.py — human telemetry → per-trajectory-step arrays (role: data input).

Self-containment note (see CONTEXT.md): this package imports nothing from any other inference version
or human_modeling/llp. The loader below is COPIED from
``human_modeling/llp/scripts/compute_trajectory_loglik.py`` (HumanEpisode,
_episode_from_telemetry, load_pilot_episodes) and the array builder from
``human_modeling/inference_aaai/scripts/fit_hierarchical_params.py``
(build_episode_state_action_arrays), unchanged except for imports/typing style.

Terminology matches implementation.md: a "trajectory step" t is one stored telemetry
frame; ``s_t`` is the flat observation array row, ``a_t`` the human's action.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

from drive_env.core.data_converter import human_to_rl_observation

IDLE_ACTION = 4  # CarMDP.IDLE — the "previous action" before the episode starts.


@dataclass
class HumanEpisode:
    """One route section played by one participant.

    :param subject_id: Firebase document ID identifying the participant.
    :type subject_id: str
    :param map_base: Full tile map for this route, as a list of row strings.
    :type map_base: list[str]
    :param steps: Per-trajectory-step ``observationHuman()``-format dicts
        (player/objects/...), in chronological order.
    :type steps: list[dict[str, Any]]
    :param actions: Action index taken at each trajectory step, shape ``(N,)``.
    :type actions: np.ndarray
    :param step_indices: Simulator step index each recorded frame came from, shape ``(N,)``,
        or None when the source did not carry one. The telemetry keeps a frame on every
        event and otherwise every third step, so recorded frames are NOT evenly spaced in
        time and any per-second quantity needs these to be formed correctly
        (`estimation/markers.py`, implementation.md Entry 32). Optional and defaulted so
        synthetic episodes, which have no telemetry keys, construct unchanged.
    :type step_indices: np.ndarray | None
    """

    subject_id: str
    map_base: list[str]
    steps: list[dict[str, Any]]
    actions: np.ndarray
    step_indices: np.ndarray | None = None

    @property
    def num_steps(self) -> int:
        """Number of recorded trajectory steps in this episode.

        :return: Episode length ``N``.
        :rtype: int
        """
        return len(self.steps)


def _episode_from_telemetry(subject_id: str, telemetry: dict[str, Any]) -> HumanEpisode:
    """Build a :class:`HumanEpisode` from one decoded ``telemetry_uncompressed`` payload.

    :param subject_id: Firebase document ID of the participant who produced this telemetry.
    :type subject_id: str
    :param telemetry: Decoded payload with ``map``, ``config``, and ``data`` keys, matching
        the layout written by ``drive_env.core.telemetry.Telemetry``.
    :type telemetry: dict[str, Any]
    :return: Parsed, time-ordered episode.
    :rtype: HumanEpisode
    """
    raw_steps = telemetry["data"]
    ordered_keys = sorted(raw_steps.keys(), key=int)
    steps = [raw_steps[key][0] for key in ordered_keys]
    actions = np.array([raw_steps[key][1] for key in ordered_keys], dtype=np.int64)
    # | The telemetry's own dict keys ARE the simulator step numbers the frames came from
    # | (verified against exp1_filtered.json: keys run 2, 5, 8, ... 2402, 2405, 2406 for a
    # | 1,046-frame episode). Keeping them lets a per-second quantity be formed correctly;
    # | they were previously discarded here, which left every rate on a frame index whose
    # | spacing varies with the episode's own event count.
    return HumanEpisode(
        subject_id=subject_id,
        map_base=telemetry["map"]["base"],
        steps=steps,
        actions=actions,
        step_indices=np.array([int(key) for key in ordered_keys], dtype=np.int64),
    )


def load_pilot_episodes(path: Union[str, Path]) -> list[HumanEpisode]:
    """Load every route-section episode from a PsychDrive pilot/experiment JSON export.

    :param path: Path to a data JSON file (e.g. ``human_modeling/data/exp1_filtered.json``).
    :type path: str | Path
    :return: One :class:`HumanEpisode` per ``psychdrive_telemetry`` event, across all
        subjects and visits in the file.
    :rtype: list[HumanEpisode]
    """
    with open(path, "r", encoding="utf-8") as f:
        subjects = json.load(f)

    episodes: list[HumanEpisode] = []
    for subject in subjects:
        subject_id = subject["id"]
        page_data = subject["data"].get("pageData_psych_drive", {})
        for visit in page_data.values():
            for event in visit.get("data", []):
                if event.get("event_type") != "psychdrive_telemetry":
                    continue
                episodes.append(_episode_from_telemetry(subject_id, event["telemetry_uncompressed"]))
    return episodes


def build_episode_state_action_arrays(
    episode: HumanEpisode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a human episode into flat per-trajectory-step arrays.

    θ-independent — build once per episode and reuse across every optimizer evaluation.

    :param episode: Parsed human route-section episode.
    :type episode: HumanEpisode
    :return: ``(s, a, a_prev)`` — ``s`` shape ``(N, obs_dim)`` float32, the other two
        shape ``(N,)`` int64; ``a_prev`` is ``a`` shifted right by one, seeded with
        :data:`IDLE_ACTION`.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    s = np.stack([
        human_to_rl_observation(step, episode.map_base).reshape(-1) for step in episode.steps
    ]).astype(np.float32)
    a = episode.actions.astype(np.int64)
    a_prev = np.concatenate([[IDLE_ACTION], a[:-1]]).astype(np.int64)
    return s, a, a_prev


def load_episode_maps(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Load each episode's full map payload (for generative validation rollouts).

    :param path: Path to a data JSON file.
    :type path: str | Path
    :return: One ``{"subject_id", "game_index", "map"}`` dict per telemetry event, with
        ``game_index`` counting a subject's episodes from 1 in file order.
    :rtype: list[dict[str, Any]]
    """
    with open(path, "r", encoding="utf-8") as f:
        subjects = json.load(f)

    out: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for subject in subjects:
        subject_id = subject["id"]
        page_data = subject["data"].get("pageData_psych_drive", {})
        for visit in page_data.values():
            for event in visit.get("data", []):
                if event.get("event_type") != "psychdrive_telemetry":
                    continue
                counters[subject_id] = counters.get(subject_id, 0) + 1
                out.append({
                    "subject_id": subject_id,
                    "game_index": counters[subject_id],
                    "map": event["telemetry_uncompressed"]["map"],
                })
    return out
