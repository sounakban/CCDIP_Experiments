"""VENDORED. Lineage: human_modeling/hlp_v2/model/choice.py -> hlp_v3 -> here (2026-09-14,
implementation.md Entry 28). Both earlier packages are retired and deleted; their
design logs are in human_modeling/version_history.md. This package imports from no
other inference version and must not start.

choice.py — ρ_t(τ): subtask choice probabilities. θ and w enter the model ONLY here.

For every trajectory step t and subtask τ = (g, m) ∈ Τ_t:

    log ρ_t(τ) = log P(g | s_t, θ)  +  log π_select(m | g, s_t, θ, w)

where P(g) is the HLP's goal softmax over all G candidates and π_select is the
heuristic-reweighted construal choice: softmax over Τ_t's masks of
``U_beh(m|g) + U_heur(m, g; w)`` (implementation.md §2).

Self-containment note (see CONTEXT.md): the Torch heuristic registry below —
``_contrib_lead_vehicle_inverse_ttc``, ``_contrib_cars_in_goal_path`` (+ its all-goals
variant), ``_value_vehicle_distance``, ``_apply_additive``, ``_apply_reduction`` and the
column constants — is COPIED from
``human_modeling/inference_aaai/utils/heuristic_hierarchical_likelihood_model.py`` unchanged.
The HLP network query is imported from ``modules/base_models`` (allowed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import torch
import torch.nn.functional as F

from drive_env.core.config import TILE_SIZE_M
from modules.base_models.SSP.hlp_policy import (
    hlp_all_goals_and_mask_logits,
    load_hlp_policy,
)

from .kernel import OBS_FEATURES, EpisodeKernel

_COL_DX = 0        # lateral offset from ego (m)
_COL_DY = 1        # longitudinal offset from ego (m); negative = ahead of ego
_COL_VY = 3        # relative longitudinal velocity; negative = closing when ahead
_LANE_HALF_WIDTH_M = TILE_SIZE_M / 2.0
_BIG = 1.0e6       # masked-out sentinel distance for the "min" reduction

HEURISTIC_NAMES: tuple[str, ...] = (
    "lead_vehicle_inverse_ttc", "cars_in_goal_path",
    "closest_vehicle_distance", "mean_vehicle_distance",
)

# 4 named HLP coefficients fit by hlp_v2 (implementation.md §9); the two flag terms stay 0.
THETA_NAMES: tuple[str, ...] = (
    "hlp_collision_penalty", "hlp_cognitive_cost_coef",
    "hlp_goal_proximity_cost_coef", "hlp_goal_flag_dist_penalty",
)

# The fitted vector when the heuristics are active: theta first, then the weights. A
# 4-vector everywhere means theta only, with every weight pinned to 0 (implementation.md
# Entry 34).
FIT_NAMES: tuple[str, ...] = THETA_NAMES + HEURISTIC_NAMES

# | Symmetric |w| bound per heuristic, MEASURED on checkpoint 310553 and adopted 2026-09-22
# | (implementation.md Entry 33). The rule: give the heuristics TOGETHER enough authority to
# | reach halfway from the policy's best construal to its worst, split equally between them,
# | and take the weight that buys that share in three quarters of decisions. These live in
# | code rather than in the calibrator's JSON because that output lands in the gitignored
# | `results/`, so a fresh checkout would otherwise have no bounds at all. Re-derive with
# | `scripts/calibrate_heuristic_bounds.py` if the CHECKPOINT changes -- the scale is a
# | property of its mask head, not of theta.
HEURISTIC_BOUNDS: dict[str, float] = {
    "lead_vehicle_inverse_ttc": 7.880,
    "cars_in_goal_path": 5.396,
    "closest_vehicle_distance": 0.668,
    "mean_vehicle_distance": 0.668,
}


# --------------------------------------------------------------------------------------
# Heuristic registry (copied; see module docstring for origin)
# --------------------------------------------------------------------------------------

def _contrib_lead_vehicle_inverse_ttc(
    dx_g: torch.Tensor, dy_g: torch.Tensor, vy_g: torch.Tensor,
    vs_g: torch.Tensor, goal_dx_g: torch.Tensor,
) -> torch.Tensor:
    """Inverse-TTC contribution, attributed entirely to the nearest same-lane car ahead.

    :param dx_g: ``(B, K)`` per-slot lateral offsets.
    :type dx_g: torch.Tensor
    :param dy_g: ``(B, K)`` per-slot longitudinal offsets.
    :type dy_g: torch.Tensor
    :param vy_g: ``(B, K)`` per-slot relative longitudinal velocities.
    :type vy_g: torch.Tensor
    :param vs_g: ``(B, K)`` bool slot validity.
    :type vs_g: torch.Tensor
    :param goal_dx_g: ``(B,)`` unused — uniform registry signature.
    :type goal_dx_g: torch.Tensor
    :return: ``(B, K)`` contributions.
    :rtype: torch.Tensor
    """
    candidates = (dx_g.abs() < _LANE_HALF_WIDTH_M) & (dy_g < 0.0) & (vy_g < 0.0) & vs_g
    dy_for_argmax = torch.where(candidates, dy_g, torch.full_like(dy_g, -_BIG))
    _, lead_idx = dy_for_argmax.max(dim=1)
    has_lead = candidates.any(dim=1)
    lead_dy_g = dy_g.gather(1, lead_idx.unsqueeze(1)).squeeze(1)
    lead_vy_g = vy_g.gather(1, lead_idx.unsqueeze(1)).squeeze(1)
    safe_dy = torch.where(has_lead, lead_dy_g, torch.ones_like(lead_dy_g))
    ttc_val = torch.where(has_lead, lead_vy_g / safe_dy, torch.zeros_like(safe_dy))
    contrib = torch.zeros_like(dx_g)
    contrib.scatter_(1, lead_idx.unsqueeze(1), ttc_val.unsqueeze(1))
    return contrib


def _contrib_cars_in_goal_path_all_goals(
    dx_g: torch.Tensor, vs_g: torch.Tensor, goal_dx_gn: torch.Tensor,
) -> torch.Tensor:
    """Per-(step, goal, slot) indicator: car on the goal's lateral side, no further out.

    :param dx_g: ``(B, K)`` per-slot lateral offsets.
    :type dx_g: torch.Tensor
    :param vs_g: ``(B, K)`` bool slot validity.
    :type vs_g: torch.Tensor
    :param goal_dx_gn: ``(B, G)`` per-goal lateral offsets.
    :type goal_dx_gn: torch.Tensor
    :return: ``(B, G, K)`` contributions in ``{0.0, 1.0}``.
    :rtype: torch.Tensor
    """
    same_side = torch.sign(dx_g).unsqueeze(1) == torch.sign(goal_dx_gn).unsqueeze(2)
    within_range = dx_g.abs().unsqueeze(1) <= goal_dx_gn.abs().unsqueeze(2)
    return (same_side & within_range & vs_g.unsqueeze(1)).to(dx_g.dtype)


def _value_vehicle_distance(dx_g: torch.Tensor, dy_g: torch.Tensor) -> torch.Tensor:
    """Per-slot Euclidean distance ``hypot(dx, dy)``.

    :param dx_g: ``(B, K)`` lateral offsets.
    :type dx_g: torch.Tensor
    :param dy_g: ``(B, K)`` longitudinal offsets.
    :type dy_g: torch.Tensor
    :return: ``(B, K)`` distances.
    :rtype: torch.Tensor
    """
    return torch.hypot(dx_g, dy_g)


def _apply_additive(contrib: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """``Σ_slot masks[..., slot] · contrib[..., slot]`` for every candidate mask.

    :param contrib: ``(..., K)`` per-slot contributions (broadcastable against masks).
    :type contrib: torch.Tensor
    :param masks: ``(..., M, K)`` candidate masks.
    :type masks: torch.Tensor
    :return: ``(..., M)``.
    :rtype: torch.Tensor
    """
    return (masks * contrib.unsqueeze(-2)).sum(dim=-1)


def _apply_reduction(
    value: torch.Tensor, masks: torch.Tensor, vs: torch.Tensor, kind: str,
) -> torch.Tensor:
    """Masked min/mean of ``value`` over attended-and-valid slots of every candidate.

    All-hidden candidates evaluate to 0.0 (the empty-construal convention).

    :param value: ``(..., K)`` per-slot values.
    :type value: torch.Tensor
    :param masks: ``(..., M, K)`` candidate masks.
    :type masks: torch.Tensor
    :param vs: ``(..., K)`` bool validity.
    :type vs: torch.Tensor
    :param kind: ``"min"`` or ``"mean"``.
    :type kind: str
    :return: ``(..., M)``.
    :rtype: torch.Tensor
    :raises ValueError: On unknown ``kind``.
    """
    attended_valid = masks * vs.unsqueeze(-2).to(masks.dtype)
    any_attended = attended_valid.sum(dim=-1) > 0
    if kind == "min":
        valid_value = torch.where(vs, value, torch.full_like(value, _BIG))
        filled = valid_value.unsqueeze(-2) + (1.0 - masks) * _BIG
        result = filled.min(dim=-1).values
    elif kind == "mean":
        summed = (value.unsqueeze(-2) * attended_valid).sum(dim=-1)
        count = attended_valid.sum(dim=-1)
        result = summed / count.clamp(min=1.0)
    else:
        raise ValueError(f"Unknown reduction kind {kind!r}; expected 'min' or 'mean'.")
    return torch.where(any_attended, result, torch.zeros_like(result))


# --------------------------------------------------------------------------------------
# ρ computation
# --------------------------------------------------------------------------------------

class ChoiceModel:
    """Computes ``log ρ_t(τ)`` for every step and subtask of one episode.

    :param checkpoint_dir: HLP run directory (weights + metadata).
    :type checkpoint_dir: str | Path
    :param model_file: Weights filename inside the run directory.
    :type model_file: str
    :param device: Torch device.
    :type device: str
    :ivar agent: Loaded HLP network.
    :vartype agent: Any
    :ivar meta: Checkpoint metadata dict.
    :vartype meta: dict[str, Any]
    """

    def __init__(self, checkpoint_dir: Union[str, Path],
                 model_file: str = "hlp_model_final.pt", device: str = "cpu") -> None:
        """Load the HLP agent once; reused across all evaluations."""
        self.agent, self.meta = load_hlp_policy(checkpoint_dir, device=device, model_file=model_file)
        self.device = torch.device(device)

    def lane_centers(self) -> np.ndarray:
        """Absolute lane-centre x positions used by the goal head.

        :return: Shape ``(NL,)`` array in metres.
        :rtype: np.ndarray
        """
        return self.agent.lane_centers.cpu().numpy().astype(np.float64)

    def log_rho(
        self,
        s: np.ndarray,
        kernel: EpisodeKernel,
        theta: dict[str, float],
        w: dict[str, float],
        full_visibility: bool = False,
    ) -> list[np.ndarray]:
        """``log ρ_t`` tables for every trajectory step.

        :param s: Flat observations, shape ``(N, obs_dim)``.
        :type s: np.ndarray
        :param kernel: The episode's θ-free kernel (mask enumerations, validity).
        :type kernel: EpisodeKernel
        :param theta: HLP coefficient values, keys ⊆ the checkpoint's parameter names.
        :type theta: dict[str, float]
        :param w: Heuristic weights, keys ⊆ :data:`HEURISTIC_NAMES` (missing → 0).
        :type w: dict[str, float]
        :param full_visibility: If True, the construal axis collapses to the all-ones
            mask (variant C): each table is ``(G, 1)`` and π_select ≡ 1.
        :type full_visibility: bool
        :return: Per-step tables ``log ρ_t`` of shape ``(G, M_t)``.
        :rtype: list[np.ndarray]
        """
        N = s.shape[0]
        dev = self.device
        st = torch.as_tensor(s, dtype=torch.float32, device=dev)
        q = hlp_all_goals_and_mask_logits(self.agent, self.meta, st, dict(theta))
        # Stay on the DEVICE through the bucket math below and copy back only the finished
        # (G, M) tables. Doing it on CPU cost ~35 ms/evaluation (11% of a CMA evaluation)
        # while the GPU idled — profiled 2026-08-12; the intermediates are small (the
        # (B, G, M, K) chunk cap is ~4M elements = 16 MB in float32).
        goal_logp = q["goal_logp"]                           # (N, G)
        mask_logits = q["mask_logits"]                       # (N, G, K)
        goal_dx = q["goal_dxdy"][..., 0]                     # (N, G)
        vs = torch.as_tensor(kernel.valid_slots, device=dev)  # (N, K)

        if full_visibility:
            gl = goal_logp.cpu().numpy().astype(np.float64)
            return [gl[t][:, None] for t in range(N)]

        wf = {name: float(w.get(name, 0.0)) for name in HEURISTIC_NAMES}
        s3 = torch.as_tensor(s.reshape(N, -1, OBS_FEATURES), dtype=torch.float32, device=dev)
        rows = torch.as_tensor(kernel.car_row_indices, device=dev)
        car = torch.gather(
            s3, 1, rows.unsqueeze(-1).expand(-1, -1, OBS_FEATURES))   # (N, K, F)
        dx, dy, vy = car[..., _COL_DX], car[..., _COL_DY], car[..., _COL_VY]
        vs_b = vs.bool()

        # bucketed by n_valid (all steps in a bucket share the mask enumeration), with
        # a step-chunk cap so the (B, G, M, K) intermediates stay small
        out: list[np.ndarray] = [np.empty(0)] * N
        n_valid = kernel.n_valid
        for nv in np.unique(n_valid):
            idx_all = np.nonzero(n_valid == nv)[0]
            m_t = 1 << int(nv)
            mask_ints = np.arange(m_t, dtype=np.int64) | (
                ((1 << kernel.K) - 1) ^ ((1 << int(nv)) - 1))
            masks = torch.as_tensor(
                (mask_ints[:, None] >> np.arange(kernel.K)[None, :]) & 1,
                dtype=torch.float32, device=dev,
            )                                                # (M, K), shared in-bucket
            # cap the (B, G, M, K) intermediate. Larger on CUDA: the cap exists to bound
            # transient memory, and a GPU has far more of it — bigger chunks also mean
            # fewer Python iterations, which is what actually costs time here.
            cap = 64_000_000 if dev.type == "cuda" else 4_000_000
            chunk = max(1, cap // max(1, kernel.G * m_t * kernel.K))
            for lo in range(0, idx_all.shape[0], chunk):
                idx = torch.as_tensor(idx_all[lo: lo + chunk], device=dev)
                B = idx.shape[0]
                ml = mask_logits[idx]                        # (B, G, K)
                vs_t = vs_b[idx]                             # (B, K)
                per_slot = (masks.view(1, 1, m_t, -1) * ml.unsqueeze(2)
                            - F.softplus(ml).unsqueeze(2))   # (B, G, M, K)
                per_slot = per_slot * vs_t.view(B, 1, 1, -1).float()
                u = per_slot.sum(dim=-1)                     # (B, G, M) — U_beh
                if wf["lead_vehicle_inverse_ttc"] != 0.0:
                    c = _contrib_lead_vehicle_inverse_ttc(
                        dx[idx], dy[idx], vy[idx], vs_b[idx], goal_dx[idx, 0])   # (B, K)
                    u = u + wf["lead_vehicle_inverse_ttc"] * _apply_additive(
                        c, masks.unsqueeze(0)).unsqueeze(1)  # (B, 1, M) broadcasts over G
                if wf["cars_in_goal_path"] != 0.0:
                    c = _contrib_cars_in_goal_path_all_goals(
                        dx[idx], vs_b[idx], goal_dx[idx])    # (B, G, K)
                    u = u + wf["cars_in_goal_path"] * _apply_additive(
                        c.unsqueeze(2), masks.view(1, 1, m_t, -1)).squeeze(-2)
                for name, kind in (("closest_vehicle_distance", "min"),
                                   ("mean_vehicle_distance", "mean")):
                    if wf[name] != 0.0:
                        v = _value_vehicle_distance(dx[idx], dy[idx])
                        u = u + wf[name] * _apply_reduction(
                            v, masks.unsqueeze(0), vs_b[idx], kind).unsqueeze(1)
                log_pi = u - torch.logsumexp(u, dim=2, keepdim=True)
                tabs = (goal_logp[idx].unsqueeze(2) + log_pi).cpu().numpy().astype(np.float64)
                for j, t in enumerate(idx_all[lo: lo + chunk]):
                    out[int(t)] = tabs[j]
        return out
