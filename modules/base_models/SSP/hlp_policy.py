"""hlp_policy.py — minimal API for querying a pre-trained parameterized HLP policy.

[HLP-param] Entry point for simulation tooling and human-model fitting
(human_modeling/inference). Contract:

- **Identity slot allocation** — no random permutation. Slot i is the i-th car row in
  deterministic ``hlp_extract_car_rows`` order, so the object→slot mapping is purely
  positional and ``p(action | s, θ)`` is a well-defined deterministic function (as
  required for MCMC likelihoods). Random permutation is a training-time debiasing
  device only (Fix B); the pooled mask head is permutation-equivariant by construction.
- **θ as a plain dict** — ``{coefficient_name: value}``. Order and training ranges come
  from the checkpoint's ``hlp_param_config`` (hlp_metadata.json is the source of
  truth); unspecified coefficients default to their range midpoint; out-of-range values
  warn and clamp.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.base_models.SP.hrl_llp_core import LLP_LANE_X_CENTERS
from modules.base_models.SSP.hlp_core import (
    EXPANDED_FEATURES,
    OBS_SIZE,
    HLPAgent,
    build_hlp_input,
    hlp_extract_car_rows,
)


def load_hlp_policy(
    run_dir: str | Path,
    device: torch.device | str = "cpu",
    model_file: str = "hlp_model_final.pt",
) -> tuple[HLPAgent, dict]:
    """Load a trained parameterized HLP checkpoint and its metadata.

    :param run_dir: HLP run directory (contains hlp_metadata.json + model weights).
    :type run_dir: str | Path
    :param device: torch device for the agent.
    :type device: torch.device | str
    :param model_file: weights filename inside run_dir.
    :type model_file: str
    :return: (agent in eval mode, metadata dict).
    :rtype: tuple[HLPAgent, dict]
    :raises RuntimeError: if the checkpoint predates goal selection or θ-conditioning.
    :raises AssertionError: if the stored weights do not match the constructed
        architecture (incompatible checkpoint era).
    """
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "hlp_metadata.json").read_text(encoding="utf-8"))
    if "hlp_param_config" not in meta:
        raise RuntimeError("hlp_policy requires a parameterized checkpoint "
                           "(hlp_param_config present in hlp_metadata.json)")
    device = torch.device(device)
    input_dim = int(meta["hlp_input_dim"])
    hidden = int(meta["hlp_hidden_size"])
    agent = HLPAgent(
        input_dim=input_dim,
        K=int(meta["hlp_num_car_slots"]),
        hidden_size=hidden,
        critic_hidden_size=hidden * int(meta.get("hlp_critic_hidden_size_multiplier", 1)),
        num_lanes=LLP_LANE_X_CENTERS.shape[0],
        num_dy_bins=int(meta["num_goal_dy_bins"]),
        goal_dy_min=float(meta["goal_forward_m_min"]),
        goal_dy_bin_width=float(meta["goal_dy_bin_width_eff"]),
        lane_centers=LLP_LANE_X_CENTERS,
    ).to(device)
    sd = torch.load(run_dir / model_file, map_location=device, weights_only=False)
    # [HLP-param] one-line era guard — readable failure instead of a cryptic size mismatch:
    # the pooled head's context encoder reads the base input minus the car slots, plus the goal.
    _key = "pooled_actor.ctx_enc.0.weight"
    assert _key in sd, (f"Checkpoint has no {_key!r}: not a pooled-head checkpoint "
                        f"(found: {sorted({k.split('.')[0] for k in sd})}).")
    _w_in = sd[_key].shape[1]
    _expect = input_dim - int(meta["hlp_num_car_slots"]) * EXPANDED_FEATURES + 2
    assert _w_in == _expect, (
        f"Checkpoint mask-head input dim {_w_in} != current architecture {_expect} "
        f"— incompatible checkpoint era (see hlp_metadata.json)."
    )
    agent.load_state_dict(sd)
    agent.eval()
    return agent, meta


def hlp_params_tensor(
    meta: dict,
    params: dict[str, float] | None,
    device: torch.device,
    B: int = 1,
) -> torch.Tensor:
    """Build a (B, P) raw θ tensor from a name→value dict, in hlp_param_config order.

    Unspecified coefficients default to their training-range midpoint. Values outside
    the training range warn and clamp (the policy was never trained there — important
    for MCMC proposals wandering off-support).

    :param meta: checkpoint metadata (must contain hlp_param_config).
    :type meta: dict
    :param params: {coefficient_name: value}; None → all midpoints.
    :type params: dict[str, float] | None
    :param device: target device.
    :type device: torch.device
    :param B: batch size (θ row is repeated).
    :type B: int
    :return: (B, P) float tensor of raw coefficient values.
    :rtype: torch.Tensor
    :raises ValueError: on parameter names not in hlp_param_config.
    """
    cfg = meta["hlp_param_config"]
    params = params or {}
    unknown = set(params) - {p["name"] for p in cfg}
    if unknown:
        raise ValueError(
            f"Unknown HLP parameters {sorted(unknown)}; "
            f"known: {[p['name'] for p in cfg]}"
        )
    vals = []
    for p in cfg:
        v = float(params.get(p["name"], 0.5 * (p["min"] + p["max"])))
        if not (p["min"] <= v <= p["max"]):
            warnings.warn(
                f"{p['name']}={v} outside training range [{p['min']}, {p['max']}] — clamping."
            )
            v = min(max(v, p["min"]), p["max"])
        vals.append(v)
    return torch.tensor([vals], dtype=torch.float32, device=device).expand(B, -1).clone()


def _param_ranges(meta: dict, device: torch.device) -> torch.Tensor:
    """(P, 2) θ training ranges; build_hlp_input normalizes θ to [-1, 1] over them.

    :param meta: checkpoint metadata.
    :type meta: dict
    :param device: target device.
    :type device: torch.device
    :return: (P, 2) [min, max] tensor.
    :rtype: torch.Tensor
    """
    cfg = meta["hlp_param_config"]
    return torch.tensor([[p["min"], p["max"]] for p in cfg], dtype=torch.float32, device=device)


def _obs_args(meta: dict) -> SimpleNamespace:
    """Namespace of env geometry needed by build_hlp_input, from checkpoint metadata."""
    return SimpleNamespace(
        visible_width=float(meta["visible_width"]),
        visible_height=float(meta["visible_height"]),
        max_speed=float(meta["max_speed"]),
        y_max=float(meta["y_max"]),
    )


def _prepare(
    agent: HLPAgent,
    meta: dict,
    raw_obs_flat: Any,
    params: dict[str, float] | None,
    car_row_indices: torch.Tensor | None = None,
    valid_slots: torch.Tensor | None = None,
    theta_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared obs→input pipeline: identity slot allocation + θ append.

    :param theta_tensor: Optional pre-built ``(B, P)`` raw θ tensor, one row per
        observation row, overriding ``params``. Lets a caller score the SAME observations
        under DIFFERENT θ in one forward pass (build it with :func:`hlp_params_tensor`,
        which applies the usual range validation). ``None`` → the existing behaviour of
        broadcasting ``params`` across all rows.
    :type theta_tensor: torch.Tensor | None
    :return: (raw (B, flat), base_in, car_row_indices, valid_slots, x_m, y_m).
    :raises ValueError: If ``theta_tensor``'s row count does not match the observations.
    """
    device = next(agent.parameters()).device
    raw = torch.as_tensor(raw_obs_flat, dtype=torch.float32, device=device)
    if raw.ndim == 1:
        raw = raw.unsqueeze(0)
    B = raw.shape[0]
    K = agent.K
    if theta_tensor is None:
        theta = hlp_params_tensor(meta, params, device, B)
    else:
        theta = theta_tensor.to(device=device, dtype=torch.float32)
        if theta.shape[0] != B:
            raise ValueError(
                f"theta_tensor has {theta.shape[0]} rows but there are {B} observation "
                f"rows; it must supply one θ row per observation row."
            )
    if car_row_indices is None:
        # Identity allocation: slot i = i-th car row in deterministic extraction order
        car_row_indices, num_cars = hlp_extract_car_rows(raw, K)
        valid_slots = torch.arange(K, device=device).unsqueeze(0) < num_cars.unsqueeze(1)
    base_in, _, _ = build_hlp_input(
        raw, K, _obs_args(meta), device, theta,
        car_row_indices=car_row_indices, valid_slots=valid_slots,
        param_ranges=_param_ranges(meta, device),
    )
    y_max = float(meta["y_max"])
    x_m = raw[:, 0]
    y_m = raw[:, 1] * y_max
    return raw, base_in, car_row_indices, valid_slots, x_m, y_m


def pooled_mask_logits_all_goals(
    agent: HLPAgent, base_in: torch.Tensor, goal_dxdy_candidates: torch.Tensor,
) -> torch.Tensor:
    """Goal-conditioned mask-head logits for EVERY candidate goal in one batched call —
    the multi-goal form of :class:`PooledMaskHead`'s forward pass, for exact goal-bin
    marginalisation.

    Exploits :class:`PooledMaskHead`'s factored architecture rather than naively
    repeating ``base_in`` once per goal candidate (which would be expensive: ``base_in``
    is ``hlp_input_dim``-wide, e.g. ~6587 for the 2026-07 final checkpoint, so a literal
    per-candidate repeat scales badly). Two things make the expensive part
    goal-INDEPENDENT:

    1. ``PooledMaskHead.enc``/the mean/max pooling only ever see ``car_feats``, which is
       sliced out of ``base_in`` BEFORE ``goal_dxdy`` is ever concatenated in — computed
       once per row, reused for every candidate.
    2. ``ctx_enc``'s first layer is a plain ``nn.Linear`` applied to
       ``cat([context_static, goal_dxdy])``, and matrix multiplication distributes
       exactly over concatenation: ``Linear(cat([a, b])) = a @ W_a.T + b @ W_b.T + bias``.
       So the expensive term (``context_static``, everything except the 2-dim goal
       offset) is computed once per row; only the cheap term (``goal_dxdy``, 2-dim in)
       varies per candidate. Everything downstream of that sum operates in
       ``hidden_size``-dim space (e.g. 128), not ``input_dim``-dim space.

    This is an EXACT reformulation, not an approximation — at ``N=1`` it must reproduce
    the head's own forward pass bit-for-bit (mod float rounding); see
    ``TestPooledMaskLogitsAllGoals`` in ``tests/test_hlp_policy.py``.

    :param agent: Loaded HLPAgent.
    :type agent: HLPAgent
    :param base_in: ``(B, input_dim)`` goal-free base input.
    :type base_in: torch.Tensor
    :param goal_dxdy_candidates: ``(B, N, 2)`` goal offsets — N candidates per row.
    :type goal_dxdy_candidates: torch.Tensor
    :return: ``(B, N, K)`` mask-head logits, one set per (row, goal candidate).
    :rtype: torch.Tensor
    """
    pooled_actor = agent.pooled_actor
    c0 = OBS_SIZE * EXPANDED_FEATURES
    c1 = c0 + agent.K * EXPANDED_FEATURES
    car_feats = base_in[:, c0:c1].view(-1, agent.K, EXPANDED_FEATURES)          # (B, K, feat_dim)
    context_static = torch.cat([base_in[:, :c0], base_in[:, c1:]], dim=1)       # (B, context_dim - 2)
    N = goal_dxdy_candidates.shape[1]

    # Goal-independent per-slot encoding + pooling (identical to PooledMaskHead.forward).
    valid = (car_feats.abs().sum(dim=2) > 0).float().unsqueeze(2)               # (B, K, 1)
    e = pooled_actor.enc(car_feats) * valid                                     # (B, K, H)
    n_valid = valid.sum(dim=1).clamp(min=1.0)                                   # (B, 1)
    mean = e.sum(dim=1) / n_valid                                               # (B, H)
    mx = e.masked_fill(valid == 0, float("-inf")).max(dim=1).values
    mx = torch.nan_to_num(mx, neginf=0.0)                                       # (B, H)

    # Goal-dependent context encoding, decomposed at ctx_enc's first Linear layer.
    first_linear, second_linear = pooled_actor.ctx_enc[0], pooled_actor.ctx_enc[2]
    static_term = context_static @ first_linear.weight[:, :-2].T + first_linear.bias  # (B, H)
    goal_term = goal_dxdy_candidates @ first_linear.weight[:, -2:].T                  # (B, N, H)
    h = torch.relu(static_term.unsqueeze(1) + goal_term)                              # (B, N, H)
    c = torch.relu(second_linear(h))                                                  # (B, N, H)

    mean_exp = mean.unsqueeze(1).expand(-1, N, -1)                              # (B, N, H)
    mx_exp = mx.unsqueeze(1).expand(-1, N, -1)                                  # (B, N, H)
    pooled = torch.cat([mean_exp, mx_exp, c], dim=2)                            # (B, N, 3H)
    pooled = pooled.unsqueeze(2).expand(-1, -1, agent.K, -1)                    # (B, N, K, 3H)
    e_exp = e.unsqueeze(1).expand(-1, N, -1, -1)                                # (B, N, K, H)
    return pooled_actor.score(torch.cat([e_exp, pooled], dim=3)).squeeze(3)     # (B, N, K)


def hlp_all_goals_and_mask_logits(
    agent: HLPAgent,
    meta: dict,
    raw_obs_flat: Any,
    params: dict[str, float] | None = None,
    theta_tensor: torch.Tensor | None = None,
    grad_enabled: bool = False,
) -> dict:
    """Every (lane, dy_bin) goal candidate's log-probability, goal offset, and
    goal-conditioned mask logits, for exact marginalisation over the goal.

    Uses :func:`pooled_mask_logits_all_goals` to get mask logits for every candidate in
    one batched call (see that function's docstring for why this is cheap despite the
    candidate count: car-feature processing is goal-independent, and the goal-conditioned
    context encoder's first layer is linear, so only a small per-candidate term needs to
    scale with N — confirmed empirically ~1.3-7x the single-goal cost on the real
    checkpoint, not ~Nx).

    :param agent: loaded HLPAgent (from load_hlp_policy).
    :type agent: HLPAgent
    :param meta: checkpoint metadata.
    :type meta: dict
    :param raw_obs_flat: (obs_flat,) or (B, obs_flat) raw env observation.
    :type raw_obs_flat: array-like
    :param params: {coefficient_name: value}; None → range midpoints.
    :type params: dict[str, float] | None
    :param theta_tensor: Optional pre-built ``(B, P)`` raw θ tensor, one row per
        observation row, overriding ``params``. Scoring the same observations under
        several θ then costs ONE forward pass over the tiled batch instead of one pass
        per θ. Build it with :func:`hlp_params_tensor` so range validation still runs.
    :type theta_tensor: torch.Tensor | None
    :param grad_enabled: [v4-grad] Keep the autograd graph instead of running under
        ``no_grad``. Default False, so existing callers are unaffected. Set True together
        with a ``theta_tensor`` that has ``requires_grad=True`` to obtain an exact
        analytic gradient of the outputs with respect to θ — the path
        ``human_modeling/inference`` needs, and which ``no_grad`` otherwise discards. Costs a
        backward pass and the memory to retain activations, so leave it off for inference.
    :type grad_enabled: bool
    :return: dict with ``goal_logp`` ``(B, N)`` (``log P(lane) + log P(dy_bin)`` per
        candidate), ``goal_x``/``goal_y`` ``(B, N)`` (absolute world coords, for LLP
        scoring), ``goal_dxdy`` ``(B, N, 2)``, ``mask_logits`` ``(B, N, K)``,
        ``valid_slots`` ``(B, K)``, ``car_row_indices`` ``(B, K)``, ``num_lanes``,
        ``num_dy_bins`` — ``N = num_lanes * num_dy_bins``, candidate order is row-major
        over (lane, dy_bin): candidate ``n = lane * num_dy_bins + dy_bin``.
    :rtype: dict
    """
    _, base_in, cr, vs, x_m, y_m = _prepare(agent, meta, raw_obs_flat, params,
                                            theta_tensor=theta_tensor)
    # | [v4-grad] `grad_enabled` defaults to False, so every existing caller keeps the
    # | original `no_grad` behaviour byte-for-byte. It exists because inference needs an EXACT
    # | gradient of the likelihood with respect to theta, and `no_grad` discards the graph
    # | that carries it -- correct for inference, fatal for fitting by gradient. Nothing
    # | inside the block is non-differentiable; only the context was blocking it.
    with torch.set_grad_enabled(grad_enabled):
        gl = agent.goal_actor(base_in)
        lane_logp = torch.log_softmax(gl[:, : agent.num_lanes], dim=-1)      # (B, L)
        dybin_logp = torch.log_softmax(gl[:, agent.num_lanes :], dim=-1)     # (B, D)
        num_lanes = agent.num_lanes
        num_dy_bins = dybin_logp.shape[1]
        B = base_in.shape[0]
        N = num_lanes * num_dy_bins

        goal_logp = (lane_logp.unsqueeze(2) + dybin_logp.unsqueeze(1)).reshape(B, N)
        # Same goal derivation as HLPAgent's rollout mode, broadcast over every (lane, dy_bin).
        goal_x_2d = agent.lane_centers.view(1, num_lanes, 1).expand(B, -1, num_dy_bins)
        dy_2d = agent.dy_bin_centers.view(1, 1, num_dy_bins).expand(B, num_lanes, -1)
        goal_y_2d = torch.clamp(y_m.view(B, 1, 1) - dy_2d, min=0.0)
        goal_x = goal_x_2d.reshape(B, N)
        goal_y = goal_y_2d.reshape(B, N)
        goal_dxdy = torch.stack(
            [goal_x_2d - x_m.view(B, 1, 1), goal_y_2d - y_m.view(B, 1, 1)], dim=-1
        ).reshape(B, N, 2)

        mask_logits = pooled_mask_logits_all_goals(agent, base_in, goal_dxdy)  # (B, N, K)
    return {
        "goal_logp": goal_logp,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "goal_dxdy": goal_dxdy,
        "mask_logits": mask_logits,
        "valid_slots": vs,
        "car_row_indices": cr,
        "num_lanes": num_lanes,
        "num_dy_bins": num_dy_bins,
    }

