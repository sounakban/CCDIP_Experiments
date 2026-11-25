"""VENDORED. Lineage: human_modeling/hlp_v2/model/kernel.py -> hlp_v3 -> here (2026-09-14,
implementation.md Entry 28). Both earlier packages are retired and deleted; their
design logs are in human_modeling/version_history.md. This package imports from no
other inference version and must not start.

kernel.py — Τ_t, E_t, R_t: subtask candidate sets, terminations, re-labelings.

The transition structure of the hlp_v2 HMM (implementation.md §2/§3): everything here is
θ-free and computed ONCE per episode from the data alone, then reused across every
optimizer evaluation.

- Goal axis: a subtask's dot is an absolute point frozen at entry; in step-relative
  labels this appears as a UNIFORM bin shift per trajectory step (the bin grid is
  uniform), plus a per-(lane, bin) termination flag from reach/miss geometry (§4a/§4b).
- Construal axis: attention bits follow physical cars via a position-matching identity
  tracker; the re-labeling is a per-step lookup table over all 2^K slot-masks
  (implementation.md §1, 32.5% drift measurement).

Terminology and symbols match implementation.md's notation table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

from modules.base_models.SSP.hlp_core import hlp_extract_car_rows
from drive_env.core.mdp import CarMDP

OBS_SIZE = CarMDP.OBS_SIZE            # rows per observation (ego + tiles + objects)
OBS_FEATURES = CarMDP.OBS_FEATURES    # features per row
_COL_DX, _COL_DY = 0, 1               # ego-relative offsets in object rows
_COL_KIND = 6                         # kind-id column
ID_MATCH_GATE_M = 6.0                 # max movement (m) for a car to keep its identity
                                      # across one stored step (validated tracker gate)


def load_hlp_metadata(checkpoint_dir: Union[str, Path]) -> dict[str, Any]:
    """Read a checkpoint's ``hlp_metadata.json``.

    :param checkpoint_dir: HLP run directory.
    :type checkpoint_dir: str | Path
    :return: Parsed metadata dict.
    :rtype: dict[str, Any]
    """
    with open(Path(checkpoint_dir) / "hlp_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class EpisodeKernel:
    """Per-episode, θ-free transition structure of the hlp_v2 HMM.

    Goal labels are flat indices ``g = lane * NB + bin`` (bin = forward-offset index);
    construal labels are K-bit integers with the v1 convention that INVALID slots carry
    bit 1 (they are forced visible downstream by ``apply_car_mask``).

    :ivar N: Number of trajectory steps.
    :vartype N: int
    :ivar NL: Number of lanes.
    :vartype NL: int
    :ivar NB: Number of forward-offset bins.
    :vartype NB: int
    :ivar K: Number of car slots.
    :vartype K: int
    :ivar x: Ego x per step, metres, shape ``(N,)``.
    :vartype x: np.ndarray
    :ivar y: Ego y per step, metres, shape ``(N,)``.
    :vartype y: np.ndarray
    :ivar dy_centers: Forward-offset bin centres, metres, shape ``(NB,)``.
    :vartype dy_centers: np.ndarray
    :ivar lane_x: Absolute lane-centre x positions, metres, shape ``(NL,)``.
    :vartype lane_x: np.ndarray
    :ivar n_valid: Valid car slots per step, shape ``(N,)``.
    :vartype n_valid: np.ndarray
    :ivar car_row_indices: Obs row index per slot, shape ``(N, K)``.
    :vartype car_row_indices: np.ndarray
    :ivar valid_slots: Slot validity, shape ``(N, K)`` bool.
    :vartype valid_slots: np.ndarray
    :ivar slot_ids: Persistent car identity per slot (−1 = invalid slot), shape ``(N, K)``.
    :vartype slot_ids: np.ndarray
    :ivar goal_shift: Uniform bin shift of the goal re-label per step, shape ``(N−1,)``.
    :vartype goal_shift: np.ndarray
    :ivar goal_ended: ``h_t`` on the goal axis — True where the (lane, bin) subtask ends
        during the move t → t+1, shape ``(N−1, NL·NB)``.
    :vartype goal_ended: np.ndarray
    :ivar mask_map: Construal re-label lookup ``t → t+1`` over all 2^K masks, shape
        ``(N−1, 2^K)``.
    :vartype mask_map: np.ndarray
    """

    N: int
    NL: int
    NB: int
    K: int
    x: np.ndarray
    y: np.ndarray
    dy_centers: np.ndarray
    lane_x: np.ndarray
    n_valid: np.ndarray
    car_row_indices: np.ndarray
    valid_slots: np.ndarray
    slot_ids: np.ndarray
    goal_shift: np.ndarray
    goal_ended: np.ndarray
    mask_map: np.ndarray

    # ------------------------------------------------------------------ helpers
    @property
    def G(self) -> int:
        """Number of goal labels per step.

        :return: ``NL · NB``.
        :rtype: int
        """
        return self.NL * self.NB

    def goal_xy(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Absolute dot coordinates of every goal label at step t.

        :param t: Trajectory step.
        :type t: int
        :return: ``(goal_x, goal_y)``, each shape ``(NL·NB,)`` — flat index
            ``g = lane · NB + bin``.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        gx = np.repeat(self.lane_x, self.NB)
        # clamp at the track start exactly as the HLP goal head does (hlp_policy.py:
        # clamp(y_m − dy, min=0)), so ρ and b condition on the same dot near episode end
        gy = np.maximum(self.y[t] - np.tile(self.dy_centers, self.NL), 0.0)
        return gx, gy

    def masks_at(self, t: int) -> np.ndarray:
        """Enumerate the construal labels of ``Τ_t`` (K-bit ints, invalid bits = 1).

        Valid slots form a prefix on this code path (see ``hlp_extract_car_rows``), so
        the enumeration varies the low ``n_valid`` bits and fixes the rest to 1.

        :param t: Trajectory step.
        :type t: int
        :return: Sorted mask integers, shape ``(2^n_valid,)``.
        :rtype: np.ndarray
        """
        nv = int(self.n_valid[t])
        invalid_bits = ((1 << self.K) - 1) ^ ((1 << nv) - 1)  # high bits forced to 1
        return np.arange(1 << nv, dtype=np.int64) | invalid_bits


def _ego_track(s: np.ndarray, y_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Ego absolute position per step from the flat observation rows.

    :param s: Flat observations, shape ``(N, OBS_SIZE·OBS_FEATURES)``.
    :type s: np.ndarray
    :param y_max: Map height in metres (de-normalizes obs column 1).
    :type y_max: float
    :return: ``(x, y)`` in metres, each shape ``(N,)``.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    return s[:, 0].astype(np.float64), (s[:, 1] * y_max).astype(np.float64)


def _track_car_ids(
    s3: np.ndarray, x: np.ndarray, y: np.ndarray,
    car_rows: np.ndarray, valid: np.ndarray,
) -> np.ndarray:
    """Assign persistent identities to slot cars by nearest-position matching.

    A car at step t+1 inherits the identity of the same-kind car at step t whose
    absolute position is nearest, within :data:`ID_MATCH_GATE_M`; otherwise it gets a
    fresh identity. (The validated tracker behind implementation.md §1's 32.5%
    measurement.)

    :param s3: Observations reshaped ``(N, OBS_SIZE, OBS_FEATURES)``.
    :type s3: np.ndarray
    :param x: Ego x per step, ``(N,)``.
    :type x: np.ndarray
    :param y: Ego y per step, ``(N,)``.
    :type y: np.ndarray
    :param car_rows: Obs row per slot, ``(N, K)``.
    :type car_rows: np.ndarray
    :param valid: Slot validity, ``(N, K)`` bool.
    :type valid: np.ndarray
    :return: Identity per slot (−1 invalid), ``(N, K)``.
    :rtype: np.ndarray
    """
    N, K = car_rows.shape
    ids = np.full((N, K), -1, dtype=np.int64)
    prev: list[tuple[int, float, float, float]] = []  # (id, abs_x, abs_y, kind)
    next_id = 0
    for t in range(N):
        # current cars: (slot, abs_x, abs_y, kind)
        cur_cars: list[tuple[int, float, float, float]] = []
        for j in range(K):
            if not valid[t, j]:
                continue
            row = int(car_rows[t, j])
            cur_cars.append((
                j,
                x[t] + float(s3[t, row, _COL_DX]),
                y[t] + float(s3[t, row, _COL_DY]),
                float(s3[t, row, _COL_KIND]),
            ))
        # greedy ONE-TO-ONE matching by ascending distance: each previous identity can
        # be inherited by at most one current car (prevents a newly-appearing car from
        # stealing a neighbour's identity within the gate)
        pairs = [
            (float(np.hypot(ax - px, ay - py)), ci, pi)
            for ci, (_, ax, ay, kind) in enumerate(cur_cars)
            for pi, (_, px, py, pk) in enumerate(prev)
            if pk == kind
        ]
        assigned_cur: dict[int, int] = {}
        used_prev: set[int] = set()
        for d, ci, pi in sorted(pairs):
            if d >= ID_MATCH_GATE_M or ci in assigned_cur or pi in used_prev:
                continue
            assigned_cur[ci] = prev[pi][0]
            used_prev.add(pi)
        new_prev: list[tuple[int, float, float, float]] = []
        for ci, (j, ax, ay, kind) in enumerate(cur_cars):
            cid = assigned_cur.get(ci, -1)
            if cid < 0:                                # unmatched -> fresh identity
                cid = next_id
                next_id += 1
            ids[t, j] = cid
            new_prev.append((cid, ax, ay, kind))
        prev = new_prev
    return ids


def _mask_relabel_tables(slot_ids: np.ndarray, valid: np.ndarray, K: int) -> np.ndarray:
    """Construal re-label lookup tables: for each step, old mask int → new mask int.

    A new slot's bit copies the bit of the slot that held the SAME car at the previous
    step; cars newly entering get 0 (hidden — the simulator's convention); invalid new
    slots get 1 (forced-visible convention).

    :param slot_ids: Identity per slot, ``(N, K)``.
    :type slot_ids: np.ndarray
    :param valid: Slot validity, ``(N, K)`` bool.
    :type valid: np.ndarray
    :param K: Slot count.
    :type K: int
    :return: Lookup tables, shape ``(N−1, 2^K)``.
    :rtype: np.ndarray
    """
    N = slot_ids.shape[0]
    n_masks = 1 << K
    all_masks = np.arange(n_masks, dtype=np.int64)
    tables = np.zeros((N - 1, n_masks), dtype=np.int64)
    for t in range(N - 1):
        # src[j'] = slot at step t holding the car that sits in slot j' at t+1 (−1 = new car)
        src = np.full(K, -1, dtype=np.int64)
        prev_ids = slot_ids[t]
        for j2 in range(K):
            if valid[t + 1, j2]:
                matches = np.nonzero(prev_ids == slot_ids[t + 1, j2])[0]
                if matches.size:
                    src[j2] = int(matches[0])
        out = np.zeros(n_masks, dtype=np.int64)
        for j2 in range(K):
            if not valid[t + 1, j2]:
                out |= 1 << j2                                   # invalid slots: bit 1
            elif src[j2] >= 0:
                out |= ((all_masks >> src[j2]) & 1) << j2        # bit follows the car
            # else: newly entered car -> bit stays 0 (hidden)
        tables[t] = out
    return tables


def _goal_terminations(
    x: np.ndarray, y: np.ndarray, lane_x: np.ndarray, dy_centers: np.ndarray,
    goal_radius_m: float, dy_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step uniform bin shift and goal-axis termination flags.

    A (lane, bin) subtask at step t has its dot at ``(lane_x, y_t − dy_bin)``. It ends
    during the move t → t+1 if the interpolated ego path comes within ``goal_radius_m``
    of the dot (reach), the ego passes the dot's row (miss), or the dot's new offset
    falls below the representable grid (label-range exit, implementation.md §4a).

    :param x: Ego x, ``(N,)``.
    :type x: np.ndarray
    :param y: Ego y, ``(N,)``.
    :type y: np.ndarray
    :param lane_x: Lane centres, ``(NL,)``.
    :type lane_x: np.ndarray
    :param dy_centers: Bin centres, ``(NB,)``.
    :type dy_centers: np.ndarray
    :param goal_radius_m: Reach radius (m).
    :type goal_radius_m: float
    :param dy_min: Minimum representable forward offset (m).
    :type dy_min: float
    :return: ``(goal_shift (N−1,), goal_ended (N−1, NL·NB))``.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    NL, NB = lane_x.shape[0], dy_centers.shape[0]
    N = x.shape[0]
    w = float(dy_centers[1] - dy_centers[0]) if NB > 1 else 1.0
    adv = y[:-1] - y[1:]                                   # forward advance per step (m)
    shift = np.rint(adv / w).astype(np.int64)              # uniform bin shift per step

    # dot coordinates per (t, lane, bin): x fixed per lane, y anchored at y_t and
    # clamped at the track start (mirrors the HLP goal head's clamp — finding 5)
    dot_x = lane_x[None, :, None]                          # (1, NL, 1)
    dot_y = np.maximum(
        y[:-1, None, None] - dy_centers[None, None, :], 0.0)  # (N−1, 1→NL, NB)

    # min distance from the segment (x_t,y_t)->(x_{t+1},y_{t+1}) to each dot
    p0x, p0y = x[:-1, None, None], y[:-1, None, None]
    dxs = (x[1:] - x[:-1])[:, None, None]
    dys = (y[1:] - y[:-1])[:, None, None]
    seg_len2 = dxs ** 2 + dys ** 2
    u = ((dot_x - p0x) * dxs + (dot_y - p0y) * dys) / np.where(seg_len2 > 0, seg_len2, 1.0)
    u = np.clip(u, 0.0, 1.0)
    cx, cy = p0x + u * dxs, p0y + u * dys
    reach = np.hypot(dot_x - cx, dot_y - cy) <= goal_radius_m

    miss = y[1:, None, None] < dot_y                       # ego passed the dot's row
    dy_after = dy_centers[None, None, :] - adv[:, None, None]
    dy_max = dy_min + NB * w
    # near-side exit (dot enters the sub-minimum zone) AND far-side exit: a NEGATIVE
    # advance (crash knockback throws the car backward) pushes dots beyond the far end
    # of the representable grid — those subtasks terminate rather than re-label off-grid
    range_exit = (dy_after < dy_min) | (dy_after > dy_max)
    # protect the shift map on BOTH sides: any bin whose re-label target falls off the
    # grid ends (reviewer finding 1: one-sided check crashed 16% of exp1 episodes)
    bins = np.arange(NB)[None, None, :]
    tgt_bins = bins - shift[:, None, None]
    off_grid = (tgt_bins < 0) | (tgt_bins >= NB)

    ended = (reach | miss | range_exit | off_grid)         # (N−1, NL, NB)
    return shift, ended.reshape(N - 1, NL * NB)


def build_episode_kernel(
    s: np.ndarray, meta: dict[str, Any], lane_x: np.ndarray,
    device: Union[str, torch.device] = "cpu",
) -> EpisodeKernel:
    """Build the full θ-free kernel for one episode.

    :param s: Flat observations, shape ``(N, OBS_SIZE·OBS_FEATURES)`` (from
        :func:`human_modeling.inference.inputs.build_episode_state_action_arrays`).
    :type s: np.ndarray
    :param meta: Checkpoint metadata (:func:`load_hlp_metadata`).
    :type meta: dict[str, Any]
    :param lane_x: Absolute lane-centre x positions, shape ``(NL,)``.
    :type lane_x: np.ndarray
    :param device: Device for the slot extraction. MUST match the device the choice
        model (HLP query) runs on: ``torch.topk(..., sorted=False)``'s slot ORDER is
        device-dependent, and a CPU kernel paired with a CUDA choice model silently
        misaligns the construal axis between ρ and b (reviewer finding 2).
    :type device: str | torch.device
    :return: Fully populated :class:`EpisodeKernel`.
    :rtype: EpisodeKernel
    """
    N = s.shape[0]
    K = int(meta["hlp_num_car_slots"])
    NB = int(meta["num_goal_dy_bins"])
    dy_min = float(meta["goal_forward_m_min"])
    w = float(meta["goal_dy_bin_width_eff"])
    dy_centers = dy_min + (np.arange(NB) + 0.5) * w
    y_max = float(meta["y_max"])

    x, y = _ego_track(s, y_max)
    s3 = s.reshape(N, OBS_SIZE, OBS_FEATURES)

    # slot allocation exactly as the HLP query path does, ON THE SAME DEVICE the
    # choice model will use (slot order is device-dependent; see the device param)
    st = torch.as_tensor(s, dtype=torch.float32, device=torch.device(device))
    rows_t, num_cars_t = hlp_extract_car_rows(st, K)
    car_rows = rows_t.cpu().numpy().astype(np.int64)
    n_valid = num_cars_t.cpu().numpy().astype(np.int64)
    valid = np.arange(K)[None, :] < n_valid[:, None]

    slot_ids = _track_car_ids(s3, x, y, car_rows, valid)
    mask_map = _mask_relabel_tables(slot_ids, valid, K)
    shift, ended = _goal_terminations(
        x, y, np.asarray(lane_x, dtype=np.float64), dy_centers,
        float(meta["goal_radius_m"]), dy_min,
    )

    return EpisodeKernel(
        N=N, NL=int(lane_x.shape[0]), NB=NB, K=K,
        x=x, y=y, dy_centers=dy_centers, lane_x=np.asarray(lane_x, dtype=np.float64),
        n_valid=n_valid, car_row_indices=car_rows, valid_slots=valid,
        slot_ids=slot_ids, goal_shift=shift, goal_ended=ended, mask_map=mask_map,
    )
