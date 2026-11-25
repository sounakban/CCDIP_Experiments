"""
likelihood_grad.py — the episode log-likelihood as ONE differentiable torch function of θ.

This is the port hlp_v4 Entry 5 describes: the chain theta -> policy -> log rho
-> forward recursion -> scalar, with the autograd graph intact end to end. Both ends already
existed (the policy carries gradients behind ``grad_enabled=True``; the recursion is
``hlp_v3/inference_torch.py`` with verified gradients); the one link rebuilt here is the
``log rho`` bucket assembly, whose original drops the graph at ``.cpu().numpy()``
(``hlp_v3/model/choice.py`` lines 242 and 297).

DESIGN RULES THIS FILE FOLLOWS
  1. **Mirror the numpy path operation for operation.** The correctness gate is agreement
     with ``hlp_v3.posterior.likelihood.log_likelihood`` on real episodes, and it is only a
     tight gate if both paths do the same arithmetic in the same precision: policy math in
     float32, cast to float64 at exactly the point the numpy path calls
     ``.astype(np.float64)``, recursion in float64.
  2. **Heuristics are optional, and off by default.** A 4-vector means theta only, with every
     heuristic weight pinned to 0, which is bit-identical to the version of this file that
     could not represent them at all (gated: ``tests/test_heuristic_fitting.py``). An
     8-vector appends the weights in ``HEURISTIC_NAMES`` order and makes them fitted
     parameters alongside theta (implementation.md Entry 34). Production readout up to that
     entry was variant B (``B_no_heuristics``, winning 230/230 episodes on BIC), so the
     4-vector path remains the default everywhere and nothing silently gains heuristics.
  3. **Import from no other inference version.** This package is self-contained (CONTEXT.md):
     V3 is the production package V4 builds on and must beat; V1/V2 stay untouched.

Public API: :func:`theta6_tensor`, :func:`log_rho_torch`, :class:`EpisodeGradContext`.

Typical usage::

    ctx = EpisodeGradContext(stack, inputs)              # 1. once per episode
    theta = torch.tensor(vec, dtype=torch.float64, requires_grad=True)
    ll = ctx.logL(theta)                                 # 2. differentiable scalar
    ll.backward()                                        # 3. exact d(logL)/d(theta)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from human_modeling.inference.inference_torch import TorchPlan, forward_pass_torch
from human_modeling.inference.model.kernel import OBS_FEATURES
from human_modeling.inference.model.choice import (HEURISTIC_NAMES,
                                                      THETA_NAMES, _apply_additive,
                                                      _apply_reduction, _COL_DX, _COL_DY,
                                                      _COL_VY,
                                                      _contrib_cars_in_goal_path_all_goals,
                                                      _contrib_lead_vehicle_inverse_ttc,
                                                      _value_vehicle_distance)
from modules.base_models.SSP.hlp_policy import hlp_all_goals_and_mask_logits

NEG_INF = float("-inf")


def split_fit_vector(vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Split a fitted vector into theta and, when present, the heuristic weights.

    Accepting both lengths is what keeps every existing 4-vector caller working unchanged
    while the 8-vector form turns the heuristics on. The two must stay one vector rather than
    two arguments, because the estimator optimises a single normalized coordinate array and
    its box projection has to cover the weights as well (implementation.md Entry 34).

    :param vec: ``(4,)`` theta only, or ``(8,)`` theta followed by the weights in
        :data:`HEURISTIC_NAMES` order.
    :type vec: torch.Tensor
    :return: ``(theta4, w4)``; ``w4`` is None for the 4-vector form.
    :rtype: tuple[torch.Tensor, torch.Tensor | None]
    :raises ValueError: If the vector is neither length.
    """
    n_theta, n_heur = len(THETA_NAMES), len(HEURISTIC_NAMES)
    if vec.shape[-1] == n_theta:
        return vec, None
    if vec.shape[-1] == n_theta + n_heur:
        return vec[:n_theta], vec[n_theta:]
    raise ValueError(f"fitted vector must have {n_theta} or {n_theta + n_heur} entries, "
                     f"got {vec.shape[-1]}")


def heuristic_features(stack: Any, inputs: Any) -> dict[int, torch.Tensor]:
    """
    Every heuristic's value for every candidate construal, precomputed once per episode.

    The four features read the observation and the candidate mask only -- never theta, never
    the weights -- so they are built once and reused by every evaluation, exactly as the
    emission tables are. That is what makes the extra four fitted dimensions nearly free
    (MEASURED 1.07x a theta-only gradient, implementation.md Entry 33).

    Stages: 1. query the policy once, with gradients OFF, for the goal offsets the goal-path
    heuristic needs; 2. gather the per-slot car rows; 3. per ``n_valid`` bucket, enumerate the
    candidates and stack the four feature columns.

    :param stack: Loaded models and metadata.
    :type stack: ModelStack
    :param inputs: Prepared episode.
    :type inputs: EpisodeInputs
    :return: ``{n_valid: (B, G, M, H) float32 features}``, one entry per bucket, with ``B``
        the bucket's steps in ``np.nonzero(kernel.n_valid == n_valid)`` order.
    :rtype: dict[int, torch.Tensor]
    """
    # | 1. Goal offsets. Theta cannot reach the features, so the cheapest valid theta does.
    ch, kernel = stack.choice, inputs.kernel
    dev = ch.device
    n_steps = inputs.s.shape[0]
    obs_t = torch.as_tensor(inputs.s, dtype=torch.float32, device=dev)
    with torch.no_grad():
        q = hlp_all_goals_and_mask_logits(ch.agent, ch.meta, obs_t, None)
    goal_dx_m = q["goal_dxdy"][..., 0]                                   # (N, G), metres

    # | 2. The per-slot car rows every heuristic reads.
    obs_3d = torch.as_tensor(inputs.s.reshape(n_steps, -1, OBS_FEATURES),
                             dtype=torch.float32, device=dev)
    slot_rows = torch.as_tensor(kernel.car_row_indices, device=dev)      # (N, K)
    car = torch.gather(obs_3d, 1,
                       slot_rows.unsqueeze(-1).expand(-1, -1, OBS_FEATURES))   # (N, K, F)
    dx_m, dy_m, vy_mps = car[..., _COL_DX], car[..., _COL_DY], car[..., _COL_VY]
    valid_slots = torch.as_tensor(kernel.valid_slots, device=dev).bool()  # (N, K)

    # | 3. One stacked feature tensor per bucket, in the bucket's own step order.
    out: dict[int, torch.Tensor] = {}
    for n_valid in np.unique(kernel.n_valid):
        step_idx = np.nonzero(kernel.n_valid == n_valid)[0]
        n_masks = 1 << int(n_valid)
        mask_ints = np.arange(n_masks, dtype=np.int64) | (
            ((1 << kernel.K) - 1) ^ ((1 << int(n_valid)) - 1))
        masks = torch.as_tensor((mask_ints[:, None] >> np.arange(kernel.K)[None, :]) & 1,
                                dtype=torch.float32, device=dev)          # (M, K)
        idx = torch.as_tensor(step_idx, device=dev)
        columns: list[torch.Tensor] = []
        contrib = _contrib_lead_vehicle_inverse_ttc(dx_m[idx], dy_m[idx], vy_mps[idx],
                                                    valid_slots[idx], goal_dx_m[idx, 0])
        columns.append(_apply_additive(contrib, masks.unsqueeze(0))
                       .unsqueeze(1).expand(-1, kernel.G, -1))            # (B, G, M)
        contrib = _contrib_cars_in_goal_path_all_goals(dx_m[idx], valid_slots[idx],
                                                       goal_dx_m[idx])    # (B, G, K)
        columns.append(_apply_additive(contrib.unsqueeze(2),
                                       masks.view(1, 1, n_masks, -1)).squeeze(-2))
        dist_m = _value_vehicle_distance(dx_m[idx], dy_m[idx])            # (B, K)
        for kind in ("min", "mean"):
            columns.append(_apply_reduction(dist_m, masks.unsqueeze(0), valid_slots[idx],
                                            kind).unsqueeze(1).expand(-1, kernel.G, -1))
        out[int(n_valid)] = torch.stack(columns, dim=-1).contiguous()     # (B, G, M, H)
    return out


def theta6_tensor(theta4: torch.Tensor, meta: dict) -> torch.Tensor:
    """
    Expand the 4 fitted coefficients into the 6-slot vector ``hlp_param_config`` defines.

    Two config entries (``hlp_flag_miss_penalty``, ``hlp_flag_collect_reward``) are pinned
    to ``[0, 0]`` and never fitted. Built with ``torch.stack`` rather than assignment into
    a zeros tensor: stacking keeps every fitted element in the autograd graph, while
    in-place assignment into a fresh tensor is a silent way to lose it.

    :param theta4: ``(4,)`` coefficients in ``THETA_NAMES`` order.
    :type theta4: torch.Tensor
    :param meta: Checkpoint metadata; read for the config ORDER rather than assuming it.
    :type meta: dict
    :return: ``(6,)`` vector in ``hlp_param_config`` order, graph intact.
    :rtype: torch.Tensor
    """
    order = [d["name"] for d in meta["hlp_param_config"]]
    zero = torch.zeros((), dtype=theta4.dtype, device=theta4.device)
    idx = {n: i for i, n in enumerate(THETA_NAMES)}
    return torch.stack([theta4[idx[n]] if n in idx else zero for n in order])


def log_rho_torch(stack: Any, inputs: Any, theta4: torch.Tensor,
                  features: dict[int, torch.Tensor] | None = None) -> list[torch.Tensor]:
    """
    Per-step choice tables ``log ρ_t`` as float64 tensors carrying a gradient to θ.

    A line-for-line mirror of ``ChoiceModel.log_rho``'s bucket math (construals active) and
    of its full-visibility branch (one column per step, chosen by ``inputs.full_visibility``;
    implementation.md Entry 36), with two deliberate differences: the policy runs under ``grad_enabled=True`` with a
    ``theta_tensor`` built from ``theta4``, and the final tables stay on the device as float64
    tensors instead of leaving through ``.cpu().numpy()`` — which is the single point where
    the original drops the graph.

    :param stack: Loaded models and metadata (``hlp_v3`` ModelStack).
    :type stack: Any
    :param inputs: Prepared episode (``hlp_v3`` EpisodeInputs).
    :type inputs: Any
    :param theta4: ``(4,)`` coefficients in ``THETA_NAMES`` order, or ``(8,)`` with the
        heuristic weights appended, on any device; ``requires_grad=True`` for a gradient.
    :type theta4: torch.Tensor
    :param features: Precomputed heuristic features from :func:`heuristic_features`, as
        :class:`EpisodeGradContext` caches them. Required when ``theta4`` carries weights.
    :type features: dict[int, torch.Tensor] | None
    :return: Per-step tables ``(G, M_t)``, float64, graph intact. For a
        full-visibility episode (``inputs.full_visibility``) every table is ``(G, 1)``.
    :rtype: list[torch.Tensor]
    :raises ValueError: If heuristic weights are given for a full-visibility episode, or
        without their precomputed features.
    """
    ch = stack.choice
    kernel = inputs.kernel
    s = inputs.s
    n = s.shape[0]
    dev = ch.device
    st = torch.as_tensor(s, dtype=torch.float32, device=dev)
    # | Theta drives the network; the weights (when present) only reweight the construal
    # | scores below, so they never enter the policy query.
    theta4, w4 = split_fit_vector(theta4)
    full_visibility = bool(getattr(inputs, "full_visibility", False))
    if full_visibility and w4 is not None:
        raise ValueError("heuristic weights have no effect on a full-visibility episode; "
                         "pass the 4-coefficient theta vector")
    if w4 is not None and features is None:
        raise ValueError("heuristic weights need the precomputed features; build an "
                         "EpisodeGradContext(..., with_heuristics=True) and call its logL")
    feat_by_bucket: dict[int, torch.Tensor] = features or {}
    # | One θ row per observation row, per the policy's contract. float32 to match the
    # | numpy path's precision at this stage; `expand` allocates nothing.
    tt = theta6_tensor(theta4.to(dev, torch.float32), ch.meta).unsqueeze(0).expand(n, -1)
    q = hlp_all_goals_and_mask_logits(ch.agent, ch.meta, st,
                                      theta_tensor=tt, grad_enabled=True)
    goal_logp = q["goal_logp"]                                # (N, G), grad

    # | [full-visibility] Every car slot is visible, so there is no attention choice: the
    # | table is the goal log-probability alone, one construal column per step. This mirrors
    # | the numpy branch (`ChoiceModel.log_rho`, `full_visibility=True`) operation for
    # | operation -- the float64 cast at the same point -- so the two paths
    # | stay comparable to recursion precision.
    if full_visibility:
        tabs_all = goal_logp.to(torch.float64).unsqueeze(2)   # (N, G, 1)
        return [tabs_all[t] for t in range(n)]

    mask_logits = q["mask_logits"]                            # (N, G, K), grad
    vs_b = torch.as_tensor(kernel.valid_slots, device=dev).bool()   # (N, K)

    out: list[torch.Tensor | None] = [None] * n
    n_valid = kernel.n_valid
    for nv in np.unique(n_valid):
        idx_all = np.nonzero(n_valid == nv)[0]
        m_t = 1 << int(nv)
        mask_ints = np.arange(m_t, dtype=np.int64) | (
            ((1 << kernel.K) - 1) ^ ((1 << int(nv)) - 1))
        masks = torch.as_tensor(
            (mask_ints[:, None] >> np.arange(kernel.K)[None, :]) & 1,
            dtype=torch.float32, device=dev)                  # (M, K)
        # | Same chunk cap as the original: bounds the (B, G, M, K) intermediate, which
        # | matters MORE here because retained activations multiply it for the backward.
        cap = 64_000_000 if dev.type == "cuda" else 4_000_000
        chunk = max(1, cap // max(1, kernel.G * m_t * kernel.K))
        for lo_i in range(0, idx_all.shape[0], chunk):
            idx = torch.as_tensor(idx_all[lo_i: lo_i + chunk], device=dev)
            b = idx.shape[0]
            ml = mask_logits[idx]                             # (B, G, K)
            vs_t = vs_b[idx]                                  # (B, K)
            # | Bernoulli log-prob of each attention pattern under the mask head:
            # | chosen slots contribute logit - softplus(logit), unchosen -softplus.
            per_slot = (masks.view(1, 1, m_t, -1) * ml.unsqueeze(2)
                        - F.softplus(ml).unsqueeze(2))        # (B, G, M, K)
            per_slot = per_slot * vs_t.view(b, 1, 1, -1).float()
            u = per_slot.sum(dim=-1)                          # (B, G, M)
            if w4 is not None:
                # | The heuristic term. The features are constants here, so this is the
                # | ONLY place the weights enter and d u / d w is just the feature — which
                # | is why the extra four dimensions cost almost nothing (Entry 33).
                feat = feat_by_bucket[int(nv)][lo_i: lo_i + chunk]   # (B, G, M, H)
                u = u + torch.einsum("bgmh,h->bgm", feat, w4.to(dev, torch.float32))
            log_pi = u - torch.logsumexp(u, dim=2, keepdim=True)
            # | float64 cast at EXACTLY the numpy path's .astype(np.float64) point, so the
            # | two paths agree to recursion precision rather than float32 epsilon.
            tabs = (goal_logp[idx].unsqueeze(2) + log_pi).to(torch.float64)
            for j, t in enumerate(idx_all[lo_i: lo_i + chunk]):
                out[int(t)] = tabs[j]
    return out                                                 # type: ignore[return-value]


class EpisodeGradContext:
    """
    Everything θ-free for one episode, prepared once and reused by every gradient step.

    The transition bookkeeping (:class:`TorchPlan`) and the emission tables do not depend
    on θ, so rebuilding them per evaluation would only add cost and noise to timings.

    :param stack: Loaded models and metadata.
    :type stack: Any
    :param inputs: Prepared episode (``prepare_episode`` output).
    :type inputs: Any
    :param with_heuristics: Precompute the heuristic features so 8-vectors can be evaluated.
        Left False, this context accepts theta-only vectors and costs exactly what it did
        before the heuristics existed.
    :type with_heuristics: bool
    :ivar plan: Reusable transition plan on the policy's device.
    :vartype plan: TorchPlan
    :ivar features: Precomputed heuristic features, or None when built theta-only.
    :vartype features: dict[int, torch.Tensor] | None
    """

    def __init__(self, stack: Any, inputs: Any, with_heuristics: bool = False) -> None:
        self.stack = stack
        self.inputs = inputs
        dev = stack.choice.device
            # | [full-visibility] One construal column per step when every car is visible;
            # | the numpy keep-plan already handles that width (inference._mask_target_positions).
        full_visibility = bool(getattr(inputs, "full_visibility", False))
        shapes = [(inputs.kernel.G, 1 if full_visibility else 1 << int(v))
                  for v in inputs.kernel.n_valid]
        self.plan = TorchPlan(inputs.kernel, shapes, device=dev)
        self._emis = [torch.as_tensor(a, dtype=torch.float64, device=dev)
                      for a in inputs.tables]
        # | theta-free and w-free, so it belongs here with the emission tables rather than
        # | being rebuilt per evaluation (MEASURED 59 ms once, Entry 33).
        self.features = heuristic_features(stack, inputs) if with_heuristics else None

    def logL(self, theta4: torch.Tensor) -> torch.Tensor:
        """
        Differentiable episode log-likelihood at one coefficient vector.

        :param theta4: ``(4,)`` in ``THETA_NAMES`` order, or ``(8,)`` with the heuristic
            weights appended (requires ``with_heuristics=True``); ``requires_grad=True`` to
            get a gradient from ``.backward()``.
        :type theta4: torch.Tensor
        :return: Scalar, graph intact back to ``theta4``.
        :rtype: torch.Tensor
        :raises ValueError: If weights are supplied to a context built without them, since
            silently ignoring them would make the fit optimise a parameter with no effect.
        """
        if theta4.shape[-1] > len(THETA_NAMES) and self.features is None:
            raise ValueError("this context was built theta-only; rebuild it with "
                             "EpisodeGradContext(..., with_heuristics=True) to fit weights")
        tabs = log_rho_torch(self.stack, self.inputs, theta4, features=self.features)
        return forward_pass_torch(tabs, self._emis, self.inputs.kernel, self.plan)
