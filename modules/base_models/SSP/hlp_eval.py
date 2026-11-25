"""Async vectorized HLP->LLP evaluation harness.

Each of ``N`` environments runs its own independent ``simulate``-style loop (a fresh HLP
decision fires the moment that env's sub-episode ends, with an identity mask-remap when the
observed object set changes), while the HLP and LLP forward passes are BATCHED across all ``N``
envs. Per-env theta (``thetas`` is ``(N, P)``) lets a single run sweep many ``(coefficient, seed)``
pairs at once, which is what makes coefficient saturation sweeps cheap.

Two metric families are reported, matching the two behavioural axes of the policy:

* **construal size** -- ``shown`` (avg cars shown / step, the DIRECT effect of
  ``cognitive_cost`` and ``collision``) and ``crash_per_1k`` (the INDIRECT performance effect).
* **goal position** -- ``goal_dy`` and ``goal_flag_dist`` (the DIRECT effect of ``proximity``
  and ``goal_flag_dist``) and ``collect_ratio`` (the INDIRECT performance effect).

SEEDING (critical): ``CarDrivingEnv`` seeds its MAP at CONSTRUCTION (``self.seed``); ``reset(seed)``
reseeds only gym's RNG, NOT the map. So each env must be built with its own construction seed, and a
saturation sweep must PAIR maps across theta values (same construction seed -> same map for every
coefficient value) so the theta comparison is within-map. :func:`sweep` does this pairing.
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch

# [import root] this module lives at modules/base_models/SSP/ -> parents[3] is repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drive_env.core.mdp import CarMDP
from drive_env.interface_gym import CarDrivingEnv, MultiEncoder, register_envs
from modules.base_models.SSP.hlp_core import (
    HLP_PARAM_NAMES, MAX_OBJECTS, OBS_FEATURES, OBS_SIZE,
    apply_car_mask, build_hlp_input, hlp_extract_car_rows, hlp_goal_flag_distance)
from modules.base_models.SSP.hlp_policy import _param_ranges, load_hlp_policy
from modules.base_models.SP.hrl_llp_core import (
    Agent as LLPAgent, GoalConditionedExtractor, llp_augment_obs, llp_compute_reward_and_flags)

register_envs()
_OBJ = CarMDP.OBS_SIZE - CarMDP.MAX_OBJECTS  # first object-row index inside the flat obs
DEV = torch.device("cpu")                    # eval is CPU-bound on env.step; batched forwards are tiny

# [operating point] run 3098306's game-score-optimal theta, used as the FIXED background when
# sweeping a single coefficient. Every term <= 0 except the two flag terms held at 0 (both OFF).
GAME_THETA: dict[str, float] = {
    "hlp_collision_penalty": 1.43, "hlp_cognitive_cost_coef": 0.003,
    "hlp_goal_proximity_cost_coef": 0.45, "hlp_flag_miss_penalty": 0.0,
    "hlp_flag_collect_reward": 0.0, "hlp_goal_flag_dist_penalty": 0.03,
}


def build_theta(overrides: Optional[dict[str, float]] = None, P: Optional[int] = None) -> torch.Tensor:
    """Build one theta row from :data:`GAME_THETA`, applying per-coefficient overrides.

    :param overrides: coefficient values to override on top of the game-optimal baseline.
    :type overrides: Optional[dict[str, float]]
    :param P: number of theta entries (defaults to ``len(HLP_PARAM_NAMES)``).
    :type P: Optional[int]
    :return: (P,) theta vector ordered by :data:`HLP_PARAM_NAMES`.
    :rtype: torch.Tensor
    """
    vals = dict(GAME_THETA)
    if overrides:
        vals.update(overrides)
    P = P if P is not None else len(HLP_PARAM_NAMES)
    t = torch.zeros(P)
    for k, name in enumerate(HLP_PARAM_NAMES):
        t[k] = vals[name]
    return t


def load_hierarchy(run_dir: str) -> tuple[Any, Any, dict, SimpleNamespace, int, float, Any]:
    """Load the HLP policy plus its frozen LLP and the args needed to drive a rollout.

    :param run_dir: HLP run directory (contains ``hlp_metadata.json``, ``hrl_llp_metadata.json``,
        ``llp_model_final.pt``).
    :type run_dir: str
    :return: ``(hlp_agent, llp_agent, hlp_meta, args_ns, K, y_max, param_ranges)``.
    :rtype: tuple
    """
    agent, meta = load_hlp_policy(run_dir, DEV)
    llp_meta = json.load(open(os.path.join(run_dir, "hrl_llp_metadata.json")))
    tmp = CarDrivingEnv(training_mode=False)
    ep = llp_meta["encoder_params"]
    me = MultiEncoder(tile_obs_size=int(ep["tile_obs_size"]), obs_features=int(ep["obs_features"]),
        ego_hidden=int(ep["ego"]), tile_hidden=int(ep["tile"]), object_hidden=int(ep["object"]),
        device=DEV, visible_width=float(ep["visible_width"]), visible_height=float(ep["visible_height"]),
        max_speed=float(ep["max_speed"]))
    gx = GoalConditionedExtractor(multi_encoder=me, goal_x_scale=float(ep["goal_x_scale"]),
                                  goal_y_scale=float(ep["goal_y_scale"])).to(DEV)

    class SP:
        """Minimal space holder matching the LLPAgent constructor's expectations."""

        def __init__(s) -> None:
            s.single_observation_space = tmp.observation_space
            s.single_action_space = tmp.action_space

    llp = LLPAgent(SP(), hidden_size=int(llp_meta.get("hidden_size", 256)), feature_extractor=gx).to(DEV)
    ck = torch.load(os.path.join(run_dir, "llp_model_final.pt"), map_location=DEV, weights_only=False)
    llp.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck)
    llp.eval()
    y_max = float(tmp.y_max)
    a = SimpleNamespace(visible_width=float(meta["visible_width"]), visible_height=float(meta["visible_height"]),
        max_speed=float(meta["max_speed"]), y_max=y_max, goal_forward_m_min=float(meta["goal_forward_m_min"]),
        goal_forward_m_max=float(meta["goal_forward_m_max"]), goal_radius_m=float(meta["goal_radius_m"]),
        llp_reach_reward=float(meta.get("llp_reach_reward", 0.0)), llp_miss_penalty=float(meta.get("llp_miss_penalty", 0.0)),
        llp_truncation_penalty=float(meta.get("llp_truncation_penalty", 0.0)),
        llp_miss_ends_subepisode=bool(meta.get("llp_miss_ends_subepisode", False)),
        sub_episode_horizon=int(meta["sub_episode_horizon"]), llp_env_reward_coef=float(meta.get("llp_env_reward_coef", 0.1)))
    return agent, llp, meta, a, int(meta["hlp_num_car_slots"]), y_max, _param_ranges(meta, DEV)


def run(run_dir: str, thetas: torch.Tensor, episodes_target: int = 20,
        seed0: int = 0,
        map_seeds: Optional[list[int]] = None) -> dict[str, np.ndarray]:
    """Run the async batched HLP->LLP rollout and return per-env metrics.

    :param run_dir: HLP run directory to evaluate.
    :type run_dir: str
    :param thetas: (N, P) per-env theta rows (one env per row).
    :type thetas: torch.Tensor
    :param episodes_target: episodes each env accumulates before it stops contributing.
    :type episodes_target: int
    :param seed0: base construction seed when ``map_seeds`` is not given.
    :type seed0: int
    :param map_seeds: per-env construction seeds; pairs maps across a sweep when supplied.
    :type map_seeds: Optional[list[int]]
    :return: metric name -> (N,) array. Keys: ``crash_per_ep``, ``crash_per_1k``, ``shown``,
        ``ep_len``, ``collect_per_ep``, ``miss_per_ep``, ``collect_ratio``, ``goal_dy``,
        ``goal_flag_dist``.
    :rtype: dict[str, np.ndarray]
    """
    agent, llp, meta, a, K, y_max, pr = load_hierarchy(run_dir)
    N = thetas.shape[0]; hp = thetas.to(DEV)
    # [seed fix] build each env with its OWN construction seed (reset(seed) does not reseed the map);
    # map_seeds pairs maps across a sweep so theta comparisons are on identical maps.
    if map_seeds is None:
        map_seeds = [seed0 + i for i in range(N)]
    envs = [CarDrivingEnv(seed=int(map_seeds[i]), training_mode=False,
                          car_buffer_y_min=1, car_buffer_y_max=3) for i in range(N)]
    obs = []
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=int(map_seeds[i])); obs.append(o)
    raw = torch.tensor(np.stack(obs), dtype=torch.float32, device=DEV)
    erc = torch.full((N,), a.llp_env_reward_coef, device=DEV)

    def obsidx() -> torch.Tensor:
        # env.mdp._obs_idx is variable-length (num visible objects); pad/truncate to MAX_OBJECTS
        # with -1 to match training's info["obs_idx"] layout.
        rows = []
        for e in envs:
            oi_raw = getattr(e.mdp, "_obs_idx", None)
            arr = np.full(MAX_OBJECTS, -1, dtype=np.int64)
            if oi_raw is not None and len(oi_raw) > 0:
                n = min(len(oi_raw), MAX_OBJECTS)
                arr[:n] = np.asarray(oi_raw)[:n]
            rows.append(arr)
        return torch.tensor(np.stack(rows), dtype=torch.long, device=DEV)

    def decide(rw: torch.Tensor, sel: torch.Tensor):
        # fresh HLP decision for the selected envs; returns mask + goal + slot bookkeeping and
        # the two goal-position metrics (goal_dy, goal->flag distance) for accumulation.
        idx = torch.nonzero(sel).squeeze(1)
        if idx.numel() == 0:
            return None
        r = rw[idx]
        cr0, nc = hlp_extract_car_rows(r, K)
        perm = torch.stack([torch.randperm(K, device=DEV) for _ in range(idx.numel())])
        cr = cr0.gather(1, perm)
        vs = (torch.arange(K, device=DEV).unsqueeze(0) < nc.unsqueeze(1)).gather(1, perm)
        hin, _, _ = build_hlp_input(r, K, a, DEV, hp[idx], car_row_indices=cr, valid_slots=vs, param_ranges=pr)
        with torch.no_grad():
            act, _, _, _, gi = agent.get_action_and_value(hin, x_m=r[:, 0], y_m=r[:, 1] * y_max,
                                                          deterministic=True, deterministic_mask=True)
        gdy = gi["dy"]                                             # (n,) chosen forward goal distance
        gfd = hlp_goal_flag_distance(gi["goal_dxdy"], r)          # (n,) goal->nearest-flag distance
        return idx, act["mask"], gi["goal_x"], gi["goal_y"], cr, vs, perm, gdy, gfd

    car_mask = torch.zeros(N, K, device=DEV); goal_x = torch.zeros(N, device=DEV); goal_y = torch.zeros(N, device=DEV)
    car_row_indices = torch.zeros(N, K, dtype=torch.long, device=DEV); valid_slots = torch.zeros(N, K, dtype=torch.bool, device=DEV)
    car_mask_snap = torch.zeros(N, K, device=DEV); init_ids = torch.full((N, K), -1, dtype=torch.long, device=DEV)
    block_perm = torch.zeros(N, K, dtype=torch.long, device=DEV)
    # accumulators: crashes/visibility/steps/episodes (per step) + flags (per step) + goals (per decision)
    crash = torch.zeros(N); vis = torch.zeros(N); nstep = torch.zeros(N); nepi = torch.zeros(N)
    fcollect = torch.zeros(N); fmiss = torch.zeros(N)
    sum_gdy = torch.zeros(N); sum_gfd = torch.zeros(N); ndec = torch.zeros(N)

    def accum_decision(res, active_vec: torch.Tensor) -> None:
        # record the two goal-position metrics for each deciding env, gated by whether the env is
        # still inside its paired first-K episodes.
        idx, gdy, gfd = res[0], res[7], res[8]
        g = active_vec[idx]
        sum_gdy[idx] += gdy * g; sum_gfd[idx] += gfd * g; ndec[idx] += g

    oi = obsidx()
    res = decide(raw, torch.ones(N, dtype=torch.bool))
    idx, m, gx, gy, cr, vs, perm = res[:7]
    car_mask[idx] = m; goal_x[idx] = gx; goal_y[idx] = gy; car_row_indices[idx] = cr; valid_slots[idx] = vs
    car_mask_snap[idx] = m.clone(); block_perm[idx] = perm
    j = (cr - _OBJ).clamp(0, MAX_OBJECTS - 1); ii = oi[idx].gather(1, j); ii[~vs] = -1; init_ids[idx] = ii
    accum_decision(res, torch.ones(N, device=DEV))
    valid_cur = valid_slots.clone()
    kind_snap = raw.view(N, OBS_SIZE, OBS_FEATURES)[:, :, 6].clone()
    subep = torch.zeros(N, device=DEV); prev = torch.full((N,), 4, device=DEV, dtype=torch.long)
    guard = 0
    while nepi.min().item() < episodes_target and guard < episodes_target * 4000:
        guard += 1
        active = (nepi < episodes_target).float()  # only accumulate the paired first-K episodes
        masked = apply_car_mask(raw.view(N, OBS_SIZE, OBS_FEATURES), car_row_indices, car_mask, valid_cur)
        lobs = llp_augment_obs(masked.view(N, -1), goal_x, goal_y, subep, prev, erc, y_max, a.sub_episode_horizon)
        with torch.no_grad():
            enc = llp.feature_extractor(lobs)
            act = torch.argmax(llp.actor(enc), dim=-1)
        vis += (car_mask * valid_cur).sum(1) * active; nstep += active; subep += 1
        nxt = []; done = torch.zeros(N, dtype=torch.bool); isc = torch.zeros(N)
        fc = torch.zeros(N); fm = torch.zeros(N)
        for i, e in enumerate(envs):
            no, r, dn, tr, info = e.step(int(act[i].item()))
            isc[i] = float(info.get("is_collision", 0.0))
            fc[i] = float(info.get("is_flag_collect", 0.0)); fm[i] = float(info.get("is_flag_miss", 0.0))
            if dn or tr:
                done[i] = True; nepi[i] += 1; no, _ = e.reset()
            nxt.append(no)
        crash += isc * active; fcollect += fc * active; fmiss += fm * active; prev = act
        raw = torch.tensor(np.stack(nxt), dtype=torch.float32, device=DEV)
        oi = obsidx()
        xm = raw[:, 0]; ym = raw[:, 1] * y_max
        _, sd_t, _ = llp_compute_reward_and_flags(xm, ym, goal_x, goal_y, subep, done, a.goal_radius_m,
            a.llp_reach_reward, a.llp_miss_penalty, a.llp_truncation_penalty, a.sub_episode_horizon, a.llp_miss_ends_subepisode)
        llp_sd = sd_t.bool()
        new_kind = raw.view(N, OBS_SIZE, OBS_FEATURES)[:, :, 6]
        kchg = (new_kind != kind_snap).any(1)
        sd_full = llp_sd | done
        rm = kchg & ~sd_full
        if rm.any():
            ridx = torch.nonzero(rm).squeeze(1)
            cr2, nc2 = hlp_extract_car_rows(raw[ridx], K)
            cr2 = cr2.gather(1, block_perm[ridx])
            vs2 = (torch.arange(K, device=DEV).unsqueeze(0) < nc2.unsqueeze(1)).gather(1, block_perm[ridx])
            jn = (cr2 - _OBJ).clamp(0, MAX_OBJECTS - 1); nid = oi[ridx].gather(1, jn); nid[~vs2] = -2
            match = (nid.unsqueeze(2) == init_ids[ridx].unsqueeze(1)); anym = match.any(2); mi = match.float().argmax(2)
            mb = car_mask_snap[ridx].gather(1, mi)
            car_mask[ridx] = torch.where(anym, mb, torch.zeros_like(mb))
            car_row_indices[ridx] = cr2; valid_cur[ridx] = vs2; kind_snap[ridx] = new_kind[ridx]
        if sd_full.any():
            res = decide(raw, sd_full)
            idx, m, gx, gy, cr, vs, perm = res[:7]
            car_mask[idx] = m; goal_x[idx] = gx; goal_y[idx] = gy; car_row_indices[idx] = cr; valid_slots[idx] = vs
            car_mask_snap[idx] = m.clone(); block_perm[idx] = perm; valid_cur[idx] = vs
            j = (cr - _OBJ).clamp(0, MAX_OBJECTS - 1); ii = oi[idx].gather(1, j); ii[~vs] = -1; init_ids[idx] = ii
            accum_decision(res, active)
            subep[idx] = 0; kind_snap[idx] = new_kind[idx]
    T = float(episodes_target); ns = nstep.clamp(min=1); nd = ndec.clamp(min=1)
    return {
        "crash_per_ep": (crash / T).numpy(),
        "crash_per_1k": (1000.0 * crash / ns).numpy(),
        "shown": (vis / ns).numpy(),
        "ep_len": (nstep / T).numpy(),
        "collect_per_ep": (fcollect / T).numpy(),
        "miss_per_ep": (fmiss / T).numpy(),
        "collect_ratio": (fcollect / (fcollect + fmiss).clamp(min=1)).numpy(),
        "goal_dy": (sum_gdy / nd).numpy(),
        "goal_flag_dist": (sum_gfd / nd).numpy(),
    }


def sweep(run_dir: str, param_name: str, values: list[float], sims: int = 20,
          base_overrides: Optional[dict[str, float]] = None) -> dict[str, np.ndarray]:
    """Paired saturation sweep of one coefficient over ``values``, ``sims`` maps each.

    Every value sees the SAME ``sims`` maps (construction seeds ``0..sims-1``), so the value-to-value
    differences are within-map. Returns per-value mean and standard error for each metric from
    :func:`run`, aggregated over the ``sims`` maps.

    :param run_dir: HLP run directory to evaluate.
    :type run_dir: str
    :param param_name: coefficient in :data:`HLP_PARAM_NAMES` to vary.
    :type param_name: str
    :param values: coefficient values to sweep (must lie in the run's trained range; theta is clamped).
    :type values: list[float]
    :param sims: number of paired maps per value.
    :type sims: int
    :param base_overrides: fixed overrides applied to the non-swept coefficients (on top of
        :data:`GAME_THETA`).
    :type base_overrides: Optional[dict[str, float]]
    :return: ``{"values": (V,)}`` plus ``{metric: (V,)}`` means and ``{metric+"_se": (V,)}`` SEs.
    :rtype: dict[str, np.ndarray]
    """
    P = len(HLP_PARAM_NAMES)
    base = dict(base_overrides) if base_overrides else {}
    thetas = torch.stack([build_theta({**base, param_name: v}, P) for v in values for _ in range(sims)])
    map_seeds = [k for _ in values for k in range(sims)]
    r = run(run_dir, thetas, episodes_target=1, map_seeds=map_seeds)
    out: dict[str, np.ndarray] = {"values": np.asarray(values, dtype=float)}
    for key, arr in r.items():
        m = np.array([arr[g * sims:(g + 1) * sims].mean() for g in range(len(values))])
        se = np.array([arr[g * sims:(g + 1) * sims].std() / np.sqrt(sims) for g in range(len(values))])
        out[key] = m; out[key + "_se"] = se
    return out


