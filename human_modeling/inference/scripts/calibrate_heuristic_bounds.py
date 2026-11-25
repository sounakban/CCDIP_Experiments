"""
Heuristic-weight bounds, derived from the behavioural utility the weights compete against.

WHAT THIS REPLACES
    ``human_modeling/inference_aaai/scripts/calibrate_heuristic_weight_bounds.py``, whose rule
    ``bound = 3 * u_beh_spread / feature_std`` produced ranges wide enough that the fitted
    weights decided construal by themselves, overriding the trained policy -- the measured
    cause of the AAAI over-collision (implementation.md Entry 33, and hlp_v2 Entry 9). That
    rule is wrong in three compounding ways, each of which widens the bound:

      1. Its numerator pools across decisions. The construal softmax normalises WITHIN one
         (step, goal) pair, so only variation across THAT decision's candidate masks competes.
         Pooling absorbs between-decision variation, which never competes with anything.
      2. Its denominator pools the feature the same way, which for the distance heuristics
         UNDERSTATES their within-decision swing and so widens the bound again.
      3. It gives every heuristic the full target on its own, so H aligned heuristics reach H
         times the intended influence.

THE RULE USED HERE
    Per decision d, with H heuristics sharing one budget::

        target_d      = TARGET_FRACTION * (best - worst U_beh over that decision's candidates)
        budget_d      = target_d / H                      (equal say, in score points)
        w_required_kd = budget_d / feature_swing_kd        (the weight that buys that say)
        bound_k       = quantile(w_required_k, BOUND_QUANTILE)

    The ratio is formed PER DECISION and only then reduced to a quantile. Dividing one summary
    by another is not equivalent: decisions with more visible cars have both a wider U_beh
    spread and wider feature swings, and the per-decision form keeps those paired.

    "Equal say" means equal SCORE POINTS per heuristic. Dividing by each heuristic's own swing
    is what converts points into a weight, since a metre of clearance and a count of cars are
    not the same unit.

WHY A QUANTILE AND NOT A SINGLE ANSWER
    The required weight differs decision to decision -- where a feature barely changes between
    candidates, a large weight is needed to reach the budget; where it changes a lot, a small
    one. A scalar bound is therefore a choice of quantile: at ``BOUND_QUANTILE = 0.75`` the
    heuristic reaches at least its budgeted share in three quarters of decisions, and exceeds
    it in the quarter where the feature swings widest.

THE SAFETY CHECK
    A bound set is only meaningful against the failure it exists to prevent, so the run ends by
    putting every weight AT its bound with all H heuristics aligned, and reporting how much of
    each decision's own U_beh spread they can move. A ratio of 1.0 means the trained policy's
    construal preference can be overruled outright.

WHEN TO RE-RUN
    The scale of U_beh is a property of the CHECKPOINT's mask head, not of theta (MEASURED:
    across the whole trained theta box the within-decision spread moves only 37.4 -> 42.2).
    Re-run this once per checkpoint, not per fit.

Typical usage::

    uv run python -m human_modeling.inference.scripts.calibrate_heuristic_bounds \
        --hlp-checkpoint-dir human_modeling/inference/checkpoint \
        --output-json temp/heuristic_bounds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from human_modeling.inference.inputs import (build_episode_state_action_arrays,
                                                load_pilot_episodes)
from human_modeling.inference.model.choice import (HEURISTIC_NAMES, ChoiceModel,
                                                      _apply_additive, _apply_reduction,
                                                      _COL_DX, _COL_DY, _COL_VY,
                                                      _contrib_cars_in_goal_path_all_goals,
                                                      _contrib_lead_vehicle_inverse_ttc,
                                                      _value_vehicle_distance)
from human_modeling.inference.model.kernel import (OBS_FEATURES, EpisodeKernel,
                                                      build_episode_kernel)
from modules.base_models.SSP.hlp_policy import hlp_all_goals_and_mask_logits

# ===========================================================================
# 1. Module constants
# ===========================================================================

# | How far down the candidate list the heuristics should TOGETHER be able to reach, as a
# | fraction of that decision's own best-to-worst U_beh spread.
# | - 0.5: halfway to the worst candidate (the adopted target, Entry 33)
# | - a smaller value only lets them tip decisions the policy was near-indifferent about
TARGET_FRACTION = 0.5

# | Which quantile of the per-decision required weight becomes the bound. 0.75 was adopted
# | (Entry 33): the heuristic reaches its budgeted share in three quarters of decisions.
BOUND_QUANTILE = 0.75

# | A feature swing at or below this is treated as "constant across candidates" -- the
# | heuristic cannot influence that decision at any weight, so it is excluded from the
# | quantile rather than producing a divide-by-almost-zero.
SWING_EPSILON = 1e-12

# | Decisions per episode, evenly spaced. 200 was enough for the Entry 33 table to be stable
# | across episodes; the cost is linear in this number.
DEFAULT_STEPS_PER_EPISODE = 200
DEFAULT_N_EPISODES = 12


# ===========================================================================
# 2. Per-decision measurement
# ===========================================================================

def episode_decision_stats(
    choice: ChoiceModel,
    s: np.ndarray,
    kernel: EpisodeKernel,
    theta: dict[str, float],
) -> dict[str, np.ndarray]:
    """
    Per-decision U_beh spreads and heuristic feature swings for one episode.

    A "decision" is one (trajectory step, goal) pair, because that is what the construal
    softmax in ``model/choice.py`` normalises over. Steps with fewer than two candidate
    construals are skipped: there is nothing to choose, so no weight can matter.

    Stages: 1. query the policy for goal-conditioned mask logits; 2. gather the per-slot car
    rows the heuristics read; 3. per ``n_valid`` bucket, enumerate candidates, form U_beh and
    every heuristic feature, and reduce both to one row per decision.

    :param choice: Loaded V4 choice model (the HLP network).
    :type choice: ChoiceModel
    :param s: Flat observations, shape ``(N, obs_dim)``.
    :type s: np.ndarray
    :param kernel: The episode's theta-free kernel (mask enumerations, validity, slot rows).
    :type kernel: EpisodeKernel
    :param theta: HLP coefficient values the policy is queried under.
    :type theta: dict[str, float]
    :return: ``{"spread": (D,), "gap": (D,), "worst": (D,), <heuristic name>: (D,)}`` -- the
        best-to-worst U_beh range, the best-to-second-best gap, the worst candidate's U_beh,
        and each heuristic's max-minus-min across the same candidates.
    :rtype: dict[str, np.ndarray]
    """
    # | 1. Goal-conditioned mask logits for every candidate goal, under this theta.
    dev = choice.device
    obs_t = torch.as_tensor(s, dtype=torch.float32, device=dev)
    q = hlp_all_goals_and_mask_logits(choice.agent, choice.meta, obs_t, dict(theta))
    mask_logits = q["mask_logits"]                                   # (N, G, K)
    goal_dx_m = q["goal_dxdy"][..., 0]                               # (N, G), metres
    valid_slots = torch.as_tensor(kernel.valid_slots, device=dev).bool()   # (N, K)

    # | 2. The per-slot car rows every heuristic reads, gathered once for the episode.
    n_steps = s.shape[0]
    obs_3d = torch.as_tensor(s.reshape(n_steps, -1, OBS_FEATURES),
                             dtype=torch.float32, device=dev)        # (N, OBS_SIZE, F)
    slot_rows = torch.as_tensor(kernel.car_row_indices, device=dev)  # (N, K)
    car = torch.gather(obs_3d, 1,
                       slot_rows.unsqueeze(-1).expand(-1, -1, OBS_FEATURES))   # (N, K, F)
    dx_m, dy_m, vy_mps = car[..., _COL_DX], car[..., _COL_DY], car[..., _COL_VY]

    acc: dict[str, list[np.ndarray]] = {"spread": [], "gap": [], "worst": []}
    acc.update({name: [] for name in HEURISTIC_NAMES})

    # | 3. Bucket by the number of visible cars: every step in a bucket shares one mask
    # |    enumeration, so the whole bucket is scored in one batched pass.
    for n_valid in np.unique(kernel.n_valid):
        if n_valid < 1:
            continue                        # | one candidate only -- no choice to influence
        step_idx = np.nonzero(kernel.n_valid == n_valid)[0]
        n_masks = 1 << int(n_valid)
        masks = _enumerate_masks(int(n_valid), kernel.K, dev)        # (M, K)
        chunk = _chunk_size(dev, kernel.G, n_masks, kernel.K)
        for lo in range(0, step_idx.shape[0], chunk):
            idx = torch.as_tensor(step_idx[lo: lo + chunk], device=dev)
            n_chunk = idx.shape[0]
            logits = mask_logits[idx]                                # (B, G, K)
            # | U_beh = log P(construal | state, goal, theta): a product of independent
            # | Bernoullis over the valid slots, in the numerically safe logit form.
            per_slot = (masks.view(1, 1, n_masks, -1) * logits.unsqueeze(2)
                        - F.softplus(logits).unsqueeze(2))           # (B, G, M, K)
            per_slot = per_slot * valid_slots[idx].view(n_chunk, 1, 1, -1).float()
            u_beh = per_slot.sum(dim=-1).double().reshape(-1, n_masks)   # (B*G, M)

            top2 = u_beh.topk(2, dim=1).values                       # (B*G, 2)
            acc["spread"].append((u_beh.max(1).values - u_beh.min(1).values).cpu().numpy())
            acc["gap"].append((top2[:, 0] - top2[:, 1]).cpu().numpy())
            acc["worst"].append(u_beh.min(1).values.cpu().numpy())

            features = _heuristic_features(dx_m[idx], dy_m[idx], vy_mps[idx],
                                           valid_slots[idx], goal_dx_m[idx], masks, kernel.G)
            for name, feat in features.items():
                feat_flat = feat.reshape(-1, n_masks)                # (B*G, M)
                acc[name].append((feat_flat.max(1).values
                                  - feat_flat.min(1).values).cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def _enumerate_masks(n_valid: int, n_slots: int, device: torch.device) -> torch.Tensor:
    """
    Every construal over the valid slots, with invalid slots fixed to 1.

    Invalid bits are 1, not 0, because ``apply_car_mask`` forces empty slots visible
    downstream regardless -- fixing them to 1 makes each candidate identical to the mask the
    low-level policy actually experiences.

    :param n_valid: Number of slots holding a real car.
    :type n_valid: int
    :param n_slots: Total car slots ``K``.
    :type n_slots: int
    :param device: Device the masks are built on.
    :type device: torch.device
    :return: Candidate masks, shape ``(2 ** n_valid, K)``, values in {0.0, 1.0}.
    :rtype: torch.Tensor
    """
    n_masks = 1 << n_valid
    codes = np.arange(n_masks, dtype=np.int64) | (
        ((1 << n_slots) - 1) ^ ((1 << n_valid) - 1))
    bits = (codes[:, None] >> np.arange(n_slots)[None, :]) & 1       # (M, K)
    return torch.as_tensor(bits, dtype=torch.float32, device=device)


def _chunk_size(device: torch.device, n_goals: int, n_masks: int, n_slots: int) -> int:
    """
    Steps per batched pass, capping the ``(B, G, M, K)`` intermediate.

    Mirrors ``ChoiceModel.log_rho``'s own cap, including its larger CUDA budget: the cap exists
    to bound transient memory, and a GPU has far more of it.

    :param device: Device the intermediates live on.
    :type device: torch.device
    :param n_goals: Candidate goals ``G``.
    :type n_goals: int
    :param n_masks: Candidate construals ``M`` in this bucket.
    :type n_masks: int
    :param n_slots: Car slots ``K``.
    :type n_slots: int
    :return: Number of trajectory steps per pass, at least 1.
    :rtype: int
    """
    cap = 64_000_000 if device.type == "cuda" else 4_000_000
    return max(1, cap // max(1, n_goals * n_masks * n_slots))


def _heuristic_features(
    dx_m: torch.Tensor, dy_m: torch.Tensor, vy_mps: torch.Tensor,
    valid_slots: torch.Tensor, goal_dx_m: torch.Tensor, masks: torch.Tensor, n_goals: int,
) -> dict[str, torch.Tensor]:
    """
    Unweighted value of every heuristic for every candidate construal.

    Calls the same registry functions ``model/choice.py`` uses, so the numbers are the ones the
    fit would see. Only ``cars_in_goal_path`` varies with the goal; the others are broadcast
    across the goal axis so all four share one shape.

    :param dx_m: Per-slot lateral offsets, shape ``(B, K)``, metres.
    :type dx_m: torch.Tensor
    :param dy_m: Per-slot longitudinal offsets, shape ``(B, K)``, metres.
    :type dy_m: torch.Tensor
    :param vy_mps: Per-slot relative longitudinal velocities, shape ``(B, K)``.
    :type vy_mps: torch.Tensor
    :param valid_slots: Slot validity, shape ``(B, K)``.
    :type valid_slots: torch.Tensor
    :param goal_dx_m: Per-goal lateral offsets, shape ``(B, G)``, metres.
    :type goal_dx_m: torch.Tensor
    :param masks: Candidate construals, shape ``(M, K)``.
    :type masks: torch.Tensor
    :param n_goals: Candidate goals ``G``.
    :type n_goals: int
    :return: ``{heuristic name: (B, G, M) values}``.
    :rtype: dict[str, torch.Tensor]
    """
    n_masks = masks.shape[0]
    out: dict[str, torch.Tensor] = {}
    contrib = _contrib_lead_vehicle_inverse_ttc(dx_m, dy_m, vy_mps, valid_slots,
                                                goal_dx_m[:, 0])             # (B, K)
    out["lead_vehicle_inverse_ttc"] = _apply_additive(
        contrib, masks.unsqueeze(0)).unsqueeze(1).expand(-1, n_goals, -1).double()
    contrib = _contrib_cars_in_goal_path_all_goals(dx_m, valid_slots, goal_dx_m)   # (B, G, K)
    out["cars_in_goal_path"] = _apply_additive(
        contrib.unsqueeze(2), masks.view(1, 1, n_masks, -1)).squeeze(-2).double()
    dist_m = _value_vehicle_distance(dx_m, dy_m)                             # (B, K)
    for name, kind in (("closest_vehicle_distance", "min"), ("mean_vehicle_distance", "mean")):
        out[name] = _apply_reduction(dist_m, masks.unsqueeze(0), valid_slots,
                                     kind).unsqueeze(1).expand(-1, n_goals, -1).double()
    return out


# ===========================================================================
# 3. The rule, and the check on its result
# ===========================================================================

def weight_bounds(
    stats: dict[str, np.ndarray],
    target_fraction: float = TARGET_FRACTION,
    quantile: float = BOUND_QUANTILE,
) -> dict[str, dict[str, float]]:
    """
    The per-heuristic bound, and the distribution it was drawn from.

    :param stats: Per-decision statistics from :func:`episode_decision_stats`.
    :type stats: dict[str, np.ndarray]
    :param target_fraction: Fraction of each decision's U_beh spread the heuristics may
        together reach; see :data:`TARGET_FRACTION`.
    :type target_fraction: float
    :param quantile: Quantile of the per-decision required weight that becomes the bound.
    :type quantile: float
    :return: Per heuristic, ``{"bound", "p25", "median", "p75", "p90", "share_live",
        "swing_median"}``.
    :rtype: dict[str, dict[str, float]]
    """
    budget = target_fraction * stats["spread"] / len(HEURISTIC_NAMES)     # (D,) score points
    out: dict[str, dict[str, float]] = {}
    for name in HEURISTIC_NAMES:
        swing = stats[name]
        live_mask = swing > SWING_EPSILON
        required = budget[live_mask] / swing[live_mask]                   # (D_live,)
        out[name] = {
            "bound": float(np.quantile(required, quantile)),
            "p25": float(np.percentile(required, 25)),
            "median": float(np.median(required)),
            "p75": float(np.percentile(required, 75)),
            "p90": float(np.percentile(required, 90)),
            "share_live": float(live_mask.mean()),
            "swing_median": float(np.median(swing[live_mask])),
        }
    return out


def overrule_check(
    stats: dict[str, np.ndarray],
    bounds: dict[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Can the heuristics at these bounds overrule the trained policy's construal preference?

    Puts every weight at its bound with all of them pulling the same way -- the worst case a
    fit could reach -- and expresses the result as a fraction of each decision's own U_beh
    spread. A fraction of 1.0 or more means the policy's preference can be overturned
    outright, which is the AAAI failure this calibration exists to prevent.

    :param stats: Per-decision statistics from :func:`episode_decision_stats`.
    :type stats: dict[str, np.ndarray]
    :param bounds: ``{heuristic name: bound}``.
    :type bounds: dict[str, float]
    :return: ``(combined, alone)`` -- the reach of all heuristics together, and of each one on
        its own, each as ``{"median", "p95", "share_can_overrule"}``.
    :rtype: tuple[dict[str, float], dict[str, dict[str, float]]]
    """
    spread = np.maximum(stats["spread"], SWING_EPSILON)                   # (D,)
    reach = sum(bounds[name] * stats[name] for name in HEURISTIC_NAMES) / spread   # (D,)
    combined = {
        "median": float(np.median(reach)),
        "p95": float(np.percentile(reach, 95)),
        "share_can_overrule": float((reach >= 1.0).mean()),
    }
    alone: dict[str, dict[str, float]] = {}
    for name in HEURISTIC_NAMES:
        one = bounds[name] * stats[name] / spread                         # (D,)
        alone[name] = {
            "median": float(np.median(one)),
            "p95": float(np.percentile(one, 95)),
            "share_can_overrule": float((one >= 1.0).mean()),
        }
    return combined, alone


def scale_summary(stats: dict[str, np.ndarray]) -> dict[str, float]:
    """
    The behavioural-utility scale the bounds are measured against.

    :param stats: Per-decision statistics from :func:`episode_decision_stats`.
    :type stats: dict[str, np.ndarray]
    :return: Spread, gap and worst-candidate summaries, in nats.
    :rtype: dict[str, float]
    """
    return {
        "n_decisions": int(stats["spread"].shape[0]),
        "spread_mean": float(stats["spread"].mean()),
        "spread_median": float(np.median(stats["spread"])),
        "spread_p95": float(np.percentile(stats["spread"], 95)),
        "gap_mean": float(stats["gap"].mean()),
        "gap_median": float(np.median(stats["gap"])),
        "worst_candidate_median": float(np.median(stats["worst"])),
    }


# ===========================================================================
# 4. Entry point
# ===========================================================================

def main() -> None:
    """
    Measure the U_beh scale, derive the bounds, and check them against the AAAI failure.

    Stages: 1. load the checkpoint; 2. collect per-decision statistics over the
    sampled episodes; 3. report the scale; 4. derive and report the bounds; 5. run the overrule
    check on the derived bounds; 6. optionally write the JSON.
    """
    # | 1. Inputs.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hlp-checkpoint-dir", required=True,
                        help="HLP run directory, self-contained with its paired LLP")
    parser.add_argument("--data-path", default="human_modeling/data/exp1_filtered.json",
                        help="human telemetry the decisions are sampled from")
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES,
                        help="episodes sampled; decisions accumulate across all of them")
    parser.add_argument("--steps-per-episode", type=int, default=DEFAULT_STEPS_PER_EPISODE,
                        help="evenly spaced trajectory steps taken from each episode")
    parser.add_argument("--output-json", default=None,
                        help="write the full result here; parent directories are created")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    choice = ChoiceModel(args.hlp_checkpoint_dir, device=device)
    # | Theta barely moves the answer (MEASURED, Entry 33: the within-decision spread shifts
    # | only 37.4 -> 42.2 across the whole box), so the trained range midpoints serve.
    theta = {c["name"]: 0.5 * (c["min"] + c["max"]) for c in choice.meta["hlp_param_config"]}
    print("=== heuristic weight bounds ===", flush=True)
    print(f"  checkpoint {choice.meta['run_name']}", flush=True)

    # | 2. One pass over the sampled episodes; decisions pool across them.
    parts = []
    for episode in load_pilot_episodes(args.data_path)[:args.n_episodes]:
        s, _actions, _prev = build_episode_state_action_arrays(episode)
        keep = np.linspace(0, s.shape[0] - 1,
                           min(args.steps_per_episode, s.shape[0])).astype(int)
        s = s[keep]
        kernel = build_episode_kernel(s, choice.meta, choice.lane_centers(),
                                      device=choice.device)
        parts.append(episode_decision_stats(choice, s, kernel, theta))
        print(f"    {episode.subject_id[:8]}: {s.shape[0]} steps", flush=True)
    stats = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    # | 3. The scale every bound is measured against.
    scale = scale_summary(stats)
    print(f"\n  U_beh over {scale['n_decisions']} multi-candidate decisions (nats)")
    print(f"    best-to-worst spread : mean {scale['spread_mean']:6.2f}  "
          f"median {scale['spread_median']:6.2f}  p95 {scale['spread_p95']:6.2f}")
    print(f"    best-to-second gap   : mean {scale['gap_mean']:6.2f}  "
          f"median {scale['gap_median']:6.2f}")
    print(f"    worst candidate      : median {scale['worst_candidate_median']:6.2f}")

    # | 4. The rule.
    bounds_table = weight_bounds(stats, TARGET_FRACTION, BOUND_QUANTILE)
    budget = TARGET_FRACTION * scale["spread_median"] / len(HEURISTIC_NAMES)
    print(f"\n  budget per heuristic: {budget:.2f} nats "
          f"({TARGET_FRACTION:g} of the median spread, split {len(HEURISTIC_NAMES)} ways)")
    print(f"  {'heuristic':>28} {'swing':>8} {'constant':>9} {'p25':>8} {'median':>8}"
          f" {'BOUND':>9} {'p90':>8}")
    for name, row in bounds_table.items():
        print(f"  {name:>28} {row['swing_median']:>8.3f} "
              f"{1.0 - row['share_live']:>8.1%} {row['p25']:>8.3f} {row['median']:>8.3f}"
              f" {row['bound']:>9.3f} {row['p90']:>8.3f}")

    # | 5. The check: can these bounds reproduce the failure they exist to prevent?
    derived = {name: row["bound"] for name, row in bounds_table.items()}
    combined, alone = overrule_check(stats, derived)
    print("\n  overrule check: every weight at its bound, all aligned")
    print(f"    combined reach, as a fraction of the decision's own spread: "
          f"median {combined['median']:.2f}, p95 {combined['p95']:.2f}")
    print(f"    decisions they can overrule outright: "
          f"{combined['share_can_overrule']:.1%}")
    for name, row in alone.items():
        print(f"      {name:>28} alone: median {row['median']:.2f}, "
              f"overrules {row['share_can_overrule']:.1%}")

    # | 6. Persist, so a later fit reads the bounds rather than re-deriving them.
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_name": choice.meta["run_name"],
            "theta": theta,
            "target_fraction": TARGET_FRACTION,
            "bound_quantile": BOUND_QUANTILE,
            "scale": scale,
            "bounds": {name: [-row["bound"], row["bound"]]
                       for name, row in bounds_table.items()},
            "distribution": bounds_table,
            "overrule_check": {"combined": combined, "alone": alone},
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
