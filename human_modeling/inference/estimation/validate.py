"""VENDORED. Lineage: human_modeling/hlp_v2/estimation/validate.py -> hlp_v3 -> here (2026-09-14,
implementation.md Entry 28). Both earlier packages are retired and deleted; their
design logs are in human_modeling/version_history.md. This package imports from no
other inference version and must not start.

validate.py -- generative validation: simulate fitted parameters, regression plots.

Battery item 9 (implementation.md section 12): roll out each episode's fitted (theta, w) on that
episode's own recorded map, then compare simulated vs human collision counts in the
established format (scatter, y = x reference on shared axes, R2, share above the line).

Self-containment note (see CONTEXT.md): :class:`FixedMapCarDrivingEnv` is COPIED from
``human_modeling/inference_aaai/utils/map_injection.py`` and the rollout loop is COPIED (adapted)
from ``human_modeling/inference_aaai/scripts/simulate_fitted_participants.py::rollout_episode``.
The adaptation: decisions come from hlp_v2's OWN choice model (the subtask sampled from
its heuristic-reweighted choice table) -- the rollout is thereby consistent with the fitted
likelihood, replacing v1's ``heuristic_act``. Identity-based mid-block mask remapping is preserved verbatim.

Run:  uv run python -m human_modeling.inference.estimation.validate
        --hlp-checkpoint-dir <dir> --fit-csv <fit_variant_A_full.csv>
        --data-path <json> --results-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import textwrap
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from drive_env.core.config import TILE_SIZE_M
from drive_env.core.mdp import CarMDP
from drive_env.core.utils import get_obs_window, initialize_world
from drive_env.interface_gym import CarDrivingEnv
from modules.base_models.SSP.hlp_core import apply_car_mask, hlp_extract_car_rows
from modules.base_models.SP.hrl_llp_core import llp_augment_obs, llp_compute_reward_and_flags

from ..inputs import load_episode_maps, load_pilot_episodes
from ..model.kernel import OBS_FEATURES, OBS_SIZE, build_episode_kernel, load_hlp_metadata
from ..model.choice import ChoiceModel, HEURISTIC_NAMES, THETA_NAMES
from ..model.emission import LLPPolicyModel
from .markers import evasion_markers   # [evasive-plots] the two collision-specific shares

_OBJ_ROW_OFFSET = OBS_SIZE - CarMDP.MAX_OBJECTS   # first object row in the obs matrix


class FixedMapCarDrivingEnv(CarDrivingEnv):
    """``CarDrivingEnv`` whose map is a fixed, externally-provided dict, not generated.

    :ivar fixed_map: The ``{"base", "objects", "plr_x", "plr_y"}`` dict every reset uses.
    :vartype fixed_map: dict[str, Any]
    """

    def __init__(self, fixed_map: dict[str, Any], **kwargs: Any) -> None:
        """Store the fixed map and defer to the parent (evaluation config only).

        :param fixed_map: ``{"base", "objects", "plr_x", "plr_y"}`` map dict.
        :type fixed_map: dict[str, Any]
        :raises AssertionError: On training-only configurations this class does not
            reproduce.
        """
        assert kwargs.get("training_mode", False) is False
        assert kwargs.get("plr_start_y_arg") is None
        self.fixed_map = fixed_map
        super().__init__(**kwargs)

    def _reset_world(self) -> None:
        """Build the world from ``self.fixed_map`` (only the map source differs from
        the parent implementation)."""
        self.time_step = 0
        map_data = self.fixed_map
        tiles, objs, player, self.mdp, visible_window_size = initialize_world(
            map_data, plr_start_y=None, mdp_cls=self.mdp_cls, seed=self.seed,
        )
        self.mdp.sync_object_view = not self.training_mode
        self.visible_width_tiles, self.visible_height_tiles = visible_window_size
        self.tile_obs_size = self.mdp._num_rel_tiles
        self.mdp.goal_reward = self.REWARD_GOAL
        self.mdp.living_penalty_val = 0
        self.y_max = len(self.mdp.world.tiles) * TILE_SIZE_M
        self.x_max = len(self.mdp.world.tiles[0]) * TILE_SIZE_M
        self.map_duration = self.mdp.get_map_duration(0.3)
        self.reset_count += 1


class RecordingEnv(FixedMapCarDrivingEnv):
    """Fixed-map environment that records frames under the TELEMETRY's own sampling rule.

    Rolling out and recording is how simulated behaviour is made comparable to human
    behaviour: the recorded frames are in exactly the ``observationHuman()`` format the
    telemetry was captured in, so one metric function serves both sides.

    The sampling rule matters. The simulator steps at 30/s but ``core/telemetry.py`` keeps a
    frame only on an event, on termination, or every 3rd step. Recording every step would make
    a simulated episode ~3x denser than a human one, and since consecutive full-rate frames are
    near-duplicates, any per-frame statistic would be computed over a different mix than the
    human numbers it is compared against.

    :ivar frames: Recorded frames, telemetry format.
    :vartype frames: list[dict[str, Any]]
    """

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.frames: list[dict[str, Any]] = []

    def step(self, action: Any):
        """Apply one action, storing the PRE-step state when the telemetry rule keeps it.

        Two properties mirror the human collection loop exactly (telemetry.py,
        ``append_observation`` typical-usage block):
        1. The stored frame is the state the action was taken FROM — snapshot
           before stepping, store after. Recording the post-step state instead
           pairs every action with the state one step LATER than the one it was
           chosen in, and the fit scores P(action | state), so that off-by-one
           is a systematic generator/likelihood mismatch.
        2. The keep rule triggers on EVENTS, not on any non-empty info: the gym
           wrapper adds its own keys to the mdp's info dict every step, so
           ``bool(info)`` is always true here — the browser collection loop saw
           the mdp's own info, where only real events appear.
        """
        # | 1. Snapshot the current (pre-step) state before the world advances.
        _, _, x0, x1, y0, y1 = get_obs_window(self.visible_height_tiles,
                                              self.visible_width_tiles,
                                              self.x_max, self.y_max, self.mdp.player)
        pre_frame = self.mdp.observationHuman((x0, x1, y0, y1))
        # | 2. Advance, then decide from the POST-step info whether to keep it.
        out  = super().step(action)
        info = self.mdp.step_info or {}
        done = bool(out[2]) if len(out) > 2 else False
        if ("event" in info) or done or (self.mdp.timestep % 3 == 0):
            pre_frame["info"] = {"event": info["event"]} if "event" in info else {}
            # | [action-metric] the action the frame's state led to, mirroring how the human
            # | telemetry itself pairs a state with the action taken from it (inputs.py
            # | ._episode_from_telemetry: raw_steps[key] = [state, action]). Additive only —
            # | no existing field changes, so every current metric is unaffected.
            pre_frame["action"] = int(action)
            # | [marker-timebase] the simulator step this frame came from, mirroring the
            # | telemetry's own dict KEY on the human side (inputs.py: HumanEpisode
            # | .step_indices). Recorded frames are not evenly spaced in time, so without
            # | this no per-second quantity can be formed on either side. Additive only.
            pre_frame["step"] = int(self.mdp.timestep)
            self.frames.append(pre_frame)
        return out


def behaviour_metrics(frames: list[dict[str, Any]]) -> dict[str, float]:
    """Behavioural summaries computed identically for human and simulated episodes.

    Chosen because they are the behaviours that most strongly identify a person: the share of
    variation attributable to the driver rather than to the episode is 0.78 for lane changes
    and 0.67 for speed, against 0.53 for collisions. Median speed rather than mean because a
    crash forces a speed drop and ~40% of an episode falls within 3 s of one, so the mean is
    partly a crash count in disguise.

    :param frames: Telemetry-format frames (human telemetry or :class:`RecordingEnv` output).
    :type frames: list[dict[str, Any]]
    :return: ``{"lane_changes_per_1k", "median_speed"}``; empty if there are too few frames.
    :rtype: dict[str, float]
    """
    if len(frames) < 20:
        return {"lane_changes_per_1k": float("nan"), "median_speed": float("nan")}
    px = np.array([f["player"]["x"] for f in frames], dtype=float)
    vy = np.array([f["player"]["vy"] for f in frames], dtype=float)
    lane = np.floor(px / TILE_SIZE_M)          # x is snapped to lane centres, so this is exact
    return {"lane_changes_per_1k": float(np.sum(np.diff(lane) != 0)) / len(frames) * 1000.0,
            "median_speed": float(np.median(vy))}


def _decide(obs: np.ndarray, choice: ChoiceModel, meta: dict[str, Any],
            theta: dict[str, float], w: dict[str, float],
            rng: np.random.Generator) -> dict[str, Any]:
    """One subtask decision from hlp_v2's own choice model.

    SAMPLES the subtask jointly from ρ_t, the same normalized distribution over ``Τ_t``
    that the likelihood marginalizes over, so the rollout exercises the fitted distribution
    rather than collapsing it to a point estimate.

    :param obs: Single flat observation, shape ``(obs_dim,)``.
    :type obs: np.ndarray
    :param choice: Loaded choice model.
    :type choice: ChoiceModel
    :param meta: Checkpoint metadata.
    :type meta: dict[str, Any]
    :param theta: Fitted HLP coefficients.
    :type theta: dict[str, float]
    :param w: Fitted heuristic weights.
    :type w: dict[str, float]
    :param rng: Draws the subtask from ρ.
    :type rng: np.random.Generator
    :return: ``goal_x``, ``goal_y`` (absolute metres), ``mask``/``car_row_indices``/
        ``valid_slots`` (each ``(K,)``).
    :rtype: dict[str, Any]
    """
    s1 = obs[None, :].astype(np.float32)
    kernel1 = build_episode_kernel(s1, meta, choice.lane_centers(),
                                   device=choice.device)
    table = choice.log_rho(s1, kernel1, theta, w)[0]
    # joint draw over all subtasks from the normalized ρ table
    flat = table.ravel()
    p = np.exp(flat - flat.max())
    p /= p.sum()
    idx = int(rng.choice(p.size, p=p))
    g_star, m_pos = divmod(idx, table.shape[1])
    mask_int = int(kernel1.masks_at(0)[m_pos])
    gx, gy = kernel1.goal_xy(0)
    bits = np.array([(mask_int >> j) & 1 for j in range(kernel1.K)], dtype=np.float32)
    return {
        "goal_x": float(gx[g_star]), "goal_y": float(gy[g_star]), "mask": bits,
        "car_row_indices": kernel1.car_row_indices[0], "valid_slots": kernel1.valid_slots[0],
    }


def rollout_episode(
    env: FixedMapCarDrivingEnv,
    choice: ChoiceModel,
    llp: LLPPolicyModel,
    meta: dict[str, Any],
    theta: dict[str, float],
    w: dict[str, float],
    rng: np.random.Generator,
    max_steps: int = 4000,
) -> dict[str, Any]:
    """Roll one episode with fitted parameters; sampled subtasks and LLP actions;
    block-persisted subtasks.

    :param env: Reset-ready fixed-map environment.
    :type env: FixedMapCarDrivingEnv
    :param choice: Loaded choice model (subtask decisions).
    :type choice: ChoiceModel
    :param llp: Loaded LLP (action selection).
    :type llp: LLPPolicyModel
    :param meta: Checkpoint metadata.
    :type meta: dict[str, Any]
    :param theta: Fitted HLP coefficients.
    :type theta: dict[str, float]
    :param w: Fitted heuristic weights.
    :type w: dict[str, float]
    :param max_steps: Safety cap on environment steps.
    :type max_steps: int
    :param rng: Every subtask decision is SAMPLED from ρ, and every low-level action is
        DRAWN from the policy's distribution rather than taken as its most likely action.
        The likelihood scores actions by their softmax probability (model/emission.py), so a
        rollout that always took the argmax would generate data more decisive than the model
        being fitted — a mismatch present at EVERY frame, which is the profile of a bias that
        pooling amplifies rather than averages away.
    :type rng: np.random.Generator
    :return: ``score``, ``collisions``, ``flags_collected``, ``mean_vehicles_visible``,
        ``episode_length``.
    :rtype: dict[str, Any]
    """
    device = llp.device
    K = int(meta["hlp_num_car_slots"])
    y_max = float(meta["y_max"])
    sub_episode_horizon = int(meta["sub_episode_horizon"])
    goal_radius_m = float(meta["goal_radius_m"])
    llp_reach_reward = float(meta.get("llp_reach_reward", 0.0))
    llp_miss_penalty = float(meta.get("llp_miss_penalty", 0.0))
    llp_truncation_penalty = float(meta.get("llp_truncation_penalty", 0.0))
    llp_miss_ends_subepisode = bool(meta.get("llp_miss_ends_subepisode", False))
    env_reward_coef = torch.full((1,), float(meta["llp_env_reward_coef"]),
                                 dtype=torch.float32, device=device)

    def _obj_mask_map(cr: torch.Tensor, vs: torch.Tensor, mask: torch.Tensor,
                      obs_idx_arr: np.ndarray) -> dict[int, float]:
        """Snapshot {physical-object index → mask bit} at decision time."""
        m: dict[int, float] = {}
        for i in range(K):
            if vs[0, i].item():
                j = int(cr[0, i].item()) - _OBJ_ROW_OFFSET
                if 0 <= j < len(obs_idx_arr):
                    oi = int(obs_idx_arr[j])
                    if oi >= 0:
                        m[oi] = float(mask[0, i].item())
        return m

    obs, _ = env.reset()
    raw_obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    subepisode_steps = torch.zeros(1, dtype=torch.long, device=device)
    prev_action = torch.full((1,), CarMDP.IDLE, dtype=torch.long, device=device)

    d = _decide(obs, choice, meta, theta, w, rng)
    car_mask = torch.as_tensor(d["mask"], device=device).unsqueeze(0)
    car_row_indices = torch.as_tensor(d["car_row_indices"], device=device).unsqueeze(0)
    valid_slots = torch.as_tensor(d["valid_slots"], device=device).unsqueeze(0)
    goal_x = torch.tensor([d["goal_x"]], dtype=torch.float32, device=device)
    goal_y = torch.tensor([d["goal_y"]], dtype=torch.float32, device=device)
    kind_id_snap = raw_obs_t[0].view(OBS_SIZE, OBS_FEATURES)[:, 6].clone()
    obj_map = _obj_mask_map(car_row_indices, valid_slots, car_mask, env.mdp._obs_idx)

    n_collisions = n_flags = step_count = block_steps = 0
    visible_sum = 0.0
    block_visible = float(car_mask[0][valid_slots[0]].sum().item())
    done = truncated = False

    while not (done or truncated) and step_count < max_steps:
        masked_obs = apply_car_mask(
            raw_obs_t.view(1, OBS_SIZE, OBS_FEATURES), car_row_indices, car_mask, valid_slots)
        llp_obs = llp_augment_obs(
            masked_obs.view(1, -1), goal_x, goal_y, subepisode_steps, prev_action,
            env_reward_coef, y_max, sub_episode_horizon)
        with torch.no_grad():
            agent = llp.agent
            enc = agent.feature_extractor(llp_obs) if agent.feature_extractor is not None else llp_obs
            logits = agent.actor(enc)
            pr = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float64)
            pr = pr / pr.sum()
            action = int(rng.choice(pr.size, p=pr))

        next_obs, _, done, truncated, info = env.step(action)
        n_collisions += int(bool(info.get("is_collision", 0)))
        n_flags += int(bool(info.get("is_flag_collect", 0)))
        subepisode_steps += 1
        step_count += 1
        block_steps += 1
        prev_action[0] = action
        raw_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        x_m = float(raw_obs_t[0, 0].item())
        y_m = float(raw_obs_t[0, 1].item()) * y_max

        new_kind = raw_obs_t[0].view(OBS_SIZE, OBS_FEATURES)[:, 6]
        kind_changed = not torch.equal(new_kind, kind_id_snap)
        _, sub_done_t, _ = llp_compute_reward_and_flags(
            torch.tensor([x_m], device=device), torch.tensor([y_m], device=device),
            goal_x, goal_y, subepisode_steps,
            torch.tensor([done or truncated], dtype=torch.bool, device=device),
            goal_radius_m, llp_reach_reward, llp_miss_penalty, llp_truncation_penalty,
            sub_episode_horizon, llp_miss_ends_subepisode)
        sub_done = bool(sub_done_t[0].item())
        # identity-based mid-block remap: same subtask, re-pointed at reshuffled rows
        if kind_changed and not sub_done and not (done or truncated):
            kind_id_snap = new_kind.clone()
            car_row_indices, num_cars = hlp_extract_car_rows(raw_obs_t, K)
            valid_slots = torch.arange(K, device=device).unsqueeze(0) < num_cars.unsqueeze(1)
            obs_idx_now = env.mdp._obs_idx
            new_mask = torch.zeros(1, K, device=device)
            for i in range(K):
                if valid_slots[0, i].item():
                    j = int(car_row_indices[0, i].item()) - _OBJ_ROW_OFFSET
                    oi = int(obs_idx_now[j]) if 0 <= j < len(obs_idx_now) else -1
                    new_mask[0, i] = float(obj_map.get(oi, 0.0))
            car_mask = new_mask

        # sub-episode completion: fresh subtask decision
        if sub_done and not (done or truncated):
            visible_sum += block_visible * block_steps
            block_steps = 0
            subepisode_steps[0] = 0
            d = _decide(next_obs, choice, meta, theta, w, rng)
            car_mask = torch.as_tensor(d["mask"], device=device).unsqueeze(0)
            car_row_indices = torch.as_tensor(d["car_row_indices"], device=device).unsqueeze(0)
            valid_slots = torch.as_tensor(d["valid_slots"], device=device).unsqueeze(0)
            goal_x = torch.tensor([d["goal_x"]], dtype=torch.float32, device=device)
            goal_y = torch.tensor([d["goal_y"]], dtype=torch.float32, device=device)
            kind_id_snap = raw_obs_t[0].view(OBS_SIZE, OBS_FEATURES)[:, 6].clone()
            obj_map = _obj_mask_map(car_row_indices, valid_slots, car_mask, env.mdp._obs_idx)
            block_visible = float(car_mask[0][valid_slots[0]].sum().item())
        if done or truncated:
            visible_sum += block_visible * block_steps
    if not (done or truncated):
        visible_sum += block_visible * block_steps      # max_steps exit: last partial block

    return {"score": float(env.mdp.score), "collisions": n_collisions,
            "flags_collected": n_flags,
            "mean_vehicles_visible": visible_sum / max(step_count, 1),
            "episode_length": step_count}


def _variance_explained_r2_ci(
    human: np.ndarray, simulated: np.ndarray, n_boot: int = 5000, seed: int = 0,
) -> tuple[float, float, float]:
    """Variance-explained R2 of the RAW simulated value against the human's own value, with a
    percentile bootstrap 95% CI. [r2-fix]

    ``R2 = 1 - SS_res/SS_tot`` measured directly against `simulated` (Nash & Sutcliffe 1970,
    "River Flow Forecasting Through Conceptual Models Part I," Journal of Hydrology 10:282-290)
    -- NOT the squared Pearson correlation, and not a recalibrated-regression R2 either: fitting
    an optimal slope/intercept first and then measuring THAT line's residuals against ITS OWN
    variance is mathematically identical to squared correlation (an OLS identity), so it is
    blind to exactly the systematic bias `bias` (computed by the caller) is meant to report. See
    hlp_v4 CLAUDE.md *Verification & Evidence*, "R2 is not one formula."

    Why a raw, uncalibrated R2 is the right choice for a model of human driving specifically,
    not just a borrowed forecasting convention: Roberts & Pashler (2000, "How Persuasive Is a
    Good Fit? A Comment on Theory Testing," Psychological Review 107:358-367) argue a good
    CORRELATIONAL fit is a weak test of a psychological theory, since a flexible model can
    achieve one without genuinely capturing the phenomenon; Pitt & Myung (2002, "When a Good
    Fit Can Be Bad," Trends in Cognitive Sciences 6:421-425) make the companion point that a
    model fitting data after tuning does not mean it explains that data. Both argue for judging
    a model on raw/absolute predictive accuracy rather than whether it is merely
    correlationally rescalable -- the same distinction this function encodes numerically.

    :param human: The episode's own recorded value, ``(n,)``.
    :type human: np.ndarray
    :param simulated: The raw simulated value, ``(n,)``.
    :type simulated: np.ndarray
    :param n_boot: Bootstrap resample count.
    :type n_boot: int
    :param seed: Bootstrap RNG seed, fixed so a re-run reproduces the same interval.
    :type seed: int
    :return: ``(r2, ci_lo, ci_hi)``, all NaN below 3 points or when `human` has no spread.
    :rtype: tuple[float, float, float]
    """
    n = human.size
    ss_tot = float(np.sum((human - human.mean()) ** 2)) if n else 0.0
    if n < 3 or ss_tot == 0:
        return float("nan"), float("nan"), float("nan")
    r2 = 1.0 - float(np.sum((simulated - human) ** 2)) / ss_tot
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    boot = np.empty(n_boot)
    with np.errstate(invalid="ignore"):
        for b in range(n_boot):
            draw = rng.choice(idx, n)
            h_b, s_b = human[draw], simulated[draw]
            ss_tot_b = np.sum((h_b - h_b.mean()) ** 2)
            boot[b] = 1.0 - np.sum((s_b - h_b) ** 2) / ss_tot_b if ss_tot_b > 0 else np.nan
    boot_valid = boot[np.isfinite(boot)]
    if boot_valid.size < 100:
        return r2, float("nan"), float("nan")
    ci_lo, ci_hi = (float(v) for v in np.percentile(boot_valid, [2.5, 97.5]))
    return r2, ci_lo, ci_hi


# | Every text element at 1.7x matplotlib's 10 pt default (user, 2026-09-24; the model-comparison
# | figures use 2.2x): these figures are read in documents and slides, where the default text
# | was too small.
_FONT_SCALE = 1.7
_LABEL_WIDTH = 34               # characters per axis-label line at that size


def collision_regression_plot(
    human: np.ndarray, simulated: np.ndarray, out_path: Path, label: str = "collisions",
) -> dict[str, float]:
    """Human-vs-simulated scatter for one behavioural metric, in the established format.

    Shared equal-aspect axes, a ``y = x`` reference as a dashed gray line (bias is visible
    only against the identity line — least-squares residuals are symmetric around the fit by
    construction), a linear trend line in red (visual aid only, per [r2-fix]: it is NOT the
    source of the reported R2), no legend, and two numbers annotated in the bottom-right corner.

    The two numbers answer different questions, so both are shown and each is named (user,
    2026-09-24). The Pearson correlation ``r`` asks whether the simulated values RANK like the
    human ones, blind to any offset or scale; the variance-explained R2 asks how close they are
    to the human values themselves, so a model that ranks people well but is shifted scores a
    high r and a negative R2 (Entry 35 addendum 3: A_full per person, r = 0.64, R2 = -3.60).
    The bootstrap interval and the share of points above ``y = x`` are still computed and
    returned in ``stats`` — neither is derivable from the two — but not drawn.

    :param human: Human collision counts, ``(n,)``.
    :type human: np.ndarray
    :param simulated: Simulated collision counts, ``(n,)``.
    :type simulated: np.ndarray
    :param out_path: PNG destination.
    :type out_path: Path
    :param label: Metric name for the axis labels.
    :type label: str
    :return: ``{"r", "r2", "ci_lo", "ci_hi", "share_above", "bias"}``; ``r`` and ``r2`` are NaN
        when either side has no spread.
    :rtype: dict[str, float]
    """
    stats: dict[str, float] = {"bias": float(np.mean(simulated - human)),
                               "share_above": float(np.mean(simulated > human)),
                               "r": float("nan"), "r2": float("nan"),
                               "ci_lo": float("nan"), "ci_hi": float("nan")}
    with plt.rc_context({"font.size": 10 * _FONT_SCALE}):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(human, simulated, alpha=0.7, s=40)
        span = max(human.max(), simulated.max()) - min(human.min(), simulated.min())
        pad = max(span * 0.05, 1e-6)
        lo = min(human.min(), simulated.min()) - pad
        hi = max(human.max(), simulated.max()) + pad
        ax.plot([lo, hi], [lo, hi], "--", color="0.45")                  # y = x
        if human.size >= 2 and np.std(human) > 0:
            slope, intercept = np.polyfit(human, simulated, 1)
            xs = np.linspace(human.min(), human.max(), 20)
            ax.plot(xs, slope * xs + intercept, color="crimson")             # trend, visual only
            stats["r2"], stats["ci_lo"], stats["ci_hi"] = _variance_explained_r2_ci(human, simulated)
            if np.std(simulated) > 0:
                stats["r"] = float(np.corrcoef(human, simulated)[0, 1])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        # | At this size a long metric name outgrows the axis, so labels wrap onto more lines.
        ax.set_xlabel(textwrap.fill(f"Human {label}", _LABEL_WIDTH))
        ax.set_ylabel(textwrap.fill(f"Simulated {label}", _LABEL_WIDTH))
        # | No legend (user, 2026-09-24: it covered points): the dashed gray line is y = x and
        # | the red one the trend, which the caption states. The numbers stay INSIDE the axes,
        # | bottom right (user, same day); they were 18 pt (double the old 9) before the 1.7x.
        # | On a dense plot the text lands on points, so it sits in a box drawn above them
        # | (user, same day). The box is 85% opaque white: the numbers read cleanly and the
        # | points under it still show faintly, so none is hidden outright.
        ax.text(0.95, 0.05, f"r = {stats['r']:.2f}\nR² = {stats['r2']:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=18 * _FONT_SCALE,
                zorder=5, bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                                "edgecolor": "0.6", "alpha": 0.85})
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return stats


# ===========================================================================
# CLI: resumable, optionally parallel, validation of a fit file
# ===========================================================================
#
# [validate-parallel] (implementation.md Entry 35 addendum 2). Two changes to the loop that
# preceded them, neither touching a rollout: every episode's row is APPENDED to the CSV as it
# finishes and skipped on a re-run (a walltime kill used to lose all 230), and the episodes can
# be spread over CPU worker processes. Rollouts are independent, the simulator steps on the
# CPU one episode at a time whatever device the networks sit on, and Entry 32 MEASURED 8 CPU
# workers at 6.8 s per rollout per worker against 11 s on the development GPU serially -- and
# reproduced the GPU rollouts to the printed digit. The seed of rollout r of the fit file's
# i-th row is `seed + 1000*i + r` exactly as before, so the output is comparable with every
# earlier validation CSV whatever the worker count or the resume history.

_WORKER: dict[str, Any] = {}      # per-process models, filled by _init_worker


def _init_worker(checkpoint_dir: str, model_filename: str) -> None:
    """Load the two networks once per worker process, on the CPU, single-threaded.

    One torch thread per worker: with 40 workers on 50 cores the default thread count would
    oversubscribe the node many times over and slow every rollout down.
    """
    torch.set_num_threads(1)
    _WORKER["choice"] = ChoiceModel(checkpoint_dir, model_file=model_filename, device="cpu")
    _WORKER["llp"] = LLPPolicyModel(checkpoint_dir, device="cpu")
    _WORKER["meta"] = load_hlp_metadata(checkpoint_dir)


def simulate_fit_row(task: dict[str, Any], choice: Any = None, llp: Any = None,
                     meta: Any = None) -> dict[str, Any]:
    """
    All rollouts of one fit row, averaged: the simulated half of one output row.

    :param task: ``index`` (the row's position in the fit file, which seeds it), ``theta``,
        ``w``, ``map``, ``n_rollouts``, ``seed``.
    :type task: dict[str, Any]
    :param choice: Loaded choice model; the worker's own when omitted.
    :type choice: ChoiceModel | None
    :param llp: Loaded action model; the worker's own when omitted.
    :type llp: LLPPolicyModel | None
    :param meta: Checkpoint metadata; the worker's own when omitted.
    :type meta: dict[str, Any] | None
    :return: The averaged metrics plus ``collisions_sd`` and ``n_rollouts``.
    :rtype: dict[str, Any]
    """
    choice = choice if choice is not None else _WORKER["choice"]
    llp = llp if llp is not None else _WORKER["llp"]
    meta = meta if meta is not None else _WORKER["meta"]
    n_roll = task["n_rollouts"]
    draws: list[dict[str, Any]] = []
    for r_i in range(n_roll):
        env = RecordingEnv(fixed_map=task["map"], training_mode=False)
        rng = np.random.default_rng(task["seed"] + 1000 * task["index"] + r_i)
        d = rollout_episode(env, choice, llp, meta, task["theta"], task["w"], rng)
        d.update(behaviour_metrics(env.frames))     # same function as the human side
        # | [evasive-plots] RecordingEnv stamps the simulator step on every frame it keeps,
        # | so the simulated side feeds this the same two inputs the human side does.
        d.update(evasion_markers(env.frames,
                                 np.array([f["step"] for f in env.frames], dtype=np.int64)))
        draws.append(d)
    # average the metrics across draws (identical to the single draw when n_roll==1)
    m = {k_: float(np.mean([d[k_] for d in draws])) for k_ in draws[0]}
    m["collisions_sd"] = float(np.std([d["collisions"] for d in draws]))
    m["n_rollouts"] = n_roll
    return m


def fit_row_parameters(row: dict[str, str]) -> tuple[dict[str, float], dict[str, float]]:
    """
    Theta and heuristic weights from one fit-file row.

    :param row: A row of a fit CSV.
    :type row: dict[str, str]
    :return: ``(theta, w)``; ``w`` is empty for a fit without heuristics.
    :rtype: tuple[dict[str, float], dict[str, float]]
    """
    theta = {n: float(row[n]) for n in THETA_NAMES if n in row and row[n] != ""}
    # | [variants] run_human_inference.py writes the weights as `w_<name>` (Entry 35); a
    # | blank cell is a weight the variant pins or does not have.
    w: dict[str, float] = {}
    for n in HEURISTIC_NAMES:
        cell = row.get(f"w_{n}", "")
        if cell != "":
            w[n] = float(cell)
    return theta, w


def read_done_rows(out_csv: Path) -> dict[tuple[str, int], dict[str, str]]:
    """
    Rows already on disk from an earlier run, keyed like the fit file.

    :param out_csv: The output CSV, which may not exist yet.
    :type out_csv: Path
    :return: ``{(subject_id, game_index): row}``.
    :rtype: dict[tuple[str, int], dict[str, str]]
    """
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return {}
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        return {(r["subject_id"], int(r["game_index"])): r for r in csv.DictReader(f)}


METRICS = (("collisions", "human_collisions", "collisions"),
           ("lane_changes", "human_lane_changes_per_1k", "lane_changes_per_1k"),
           ("median_speed", "human_median_speed", "median_speed"),
           # | [evasive-plots] Both depend on where OTHER CARS are, which is the channel
           # | Entry 32 measured as the one that tracks a person; the three above are
           # | either the headline count or, for two of them, measured nulls.
           ("evasive_lane_change_share", "human_evasive_lane_change_share",
            "evasive_lane_change_share"),
           ("car_entry_longitudinal_share", "human_car_entry_longitudinal_share",
            "car_entry_longitudinal_share"))


def draw_regressions(out_rows: list[dict[str, Any]], plots_dir: Path, tag: str) -> None:
    """One regression plot per behavioural metric, from the finished rows.

    Collisions stay the headline for continuity, but lane-change rate and speed are the
    behaviours that actually identify a person (0.78 and 0.67 of variance is the driver,
    against 0.53 for collisions), so a model that tracks collisions while missing these has
    not captured individual differences.

    :param out_rows: Every finished output row.
    :type out_rows: list[dict[str, Any]]
    :param plots_dir: Destination; every figure goes here, not the results root.
    :type plots_dir: Path
    :param tag: Filename suffix, ``"_<tag>"`` or empty.
    :type tag: str
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    for name, hcol, scol in METRICS:
        hv = np.array([float(r.get(hcol, "nan") or "nan") for r in out_rows], dtype=float)
        sv = np.array([float(r.get(scol, "nan") or "nan") for r in out_rows], dtype=float)
        ok = ~(np.isnan(hv) | np.isnan(sv))
        if ok.sum() < 3:
            print(f"skipped {name} regression — too few usable episodes")
            continue
        png = plots_dir / f"validate_regression_{name}{tag}.png"
        st = collision_regression_plot(hv[ok], sv[ok], png, label=name.replace("_", " "))
        print(f"saved regression plot {png.name}: r={st['r']:+.3f} "
              f"R2={st['r2']:.3f} [{st['ci_lo']:+.3f}, {st['ci_hi']:+.3f}] bias={st['bias']:+.2f} "
              f"above={100 * st['share_above']:.0f}%")


def main() -> None:
    """CLI: simulate every row of a fit CSV on its own map; append per row; plot when complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlp-checkpoint-dir", required=True)
    parser.add_argument("--hlp-model-filename", default="hlp_model_final.pt")
    parser.add_argument("--fit-csv", required=True)
    parser.add_argument("--data-path", default="human_modeling/data/exp1_filtered.json")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--n-rollouts", type=int, default=1,
                        help="Rollouts per episode; >1 averages out sampling variance.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for sampling.")
    parser.add_argument("--tag", default="", help="Suffix for the output filenames.")
    parser.add_argument("--workers", type=int, default=1,
                        help="[validate-parallel] CPU worker processes, one episode each at a "
                             "time; the cluster sbatch passes 40 on a 50-core node. Rollouts "
                             "and seeds are identical whatever the count.")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    maps = {(m["subject_id"], m["game_index"]): m["map"]
            for m in load_episode_maps(args.data_path)}
    # human ground truth: collision events from the telemetry
    counters: dict[str, int] = {}
    human_col: dict[tuple[str, int], int] = {}
    human_beh: dict[tuple[str, int], dict[str, float]] = {}
    for ep in load_pilot_episodes(args.data_path):
        counters[ep.subject_id] = counters.get(ep.subject_id, 0) + 1
        k_ = (ep.subject_id, counters[ep.subject_id])
        human_col[k_] = sum(1 for st in ep.steps
                            if str(st.get("info", {}).get("event", "")).startswith("crash:"))
        # | [evasive-plots] the two collision-specific shares, from the same definition the
        # | marker read-out uses. They need the telemetry's own step numbers, which
        # | HumanEpisode carries since implementation.md Entry 32.
        human_beh[k_] = {**behaviour_metrics(ep.steps),
                         **evasion_markers(ep.steps, ep.step_indices)}

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.fit_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"fit CSV is empty: {args.fit_csv}")
    for r in rows:
        if (r["subject_id"], int(r["game_index"])) not in human_col:
            raise KeyError(f"episode {(r['subject_id'], r['game_index'])} missing from human "
                           f"telemetry — cannot regress")

    tag = f"_{args.tag}" if args.tag else ""
    out_csv = results_dir / f"validate_simulated{tag}.csv"
    done = read_done_rows(out_csv)
    if done:
        print(f"RESUMING: {len(done)} of {len(rows)} episodes already in {out_csv.name}", flush=True)
    # | Tasks keep the fit-file index, which seeds them, whether or not earlier rows are skipped.
    tasks = []
    for i, r in enumerate(rows):
        key = (r["subject_id"], int(r["game_index"]))
        if key in done:
            continue
        theta, w = fit_row_parameters(r)
        tasks.append({"index": i, "key": key, "theta": theta, "w": w, "map": maps[key],
                      "n_rollouts": args.n_rollouts, "seed": args.seed})

    def human_side(key: tuple[str, int]) -> dict[str, Any]:
        hb = human_beh.get(key, {})
        return {"subject_id": key[0], "game_index": key[1],
                "human_collisions": human_col[key],
                "human_lane_changes_per_1k": hb.get("lane_changes_per_1k", float("nan")),
                "human_median_speed": hb.get("median_speed", float("nan")),
                # | [evasive-plots]
                "human_evasive_lane_change_share":
                    hb.get("evasive_lane_change_share", float("nan")),
                "human_car_entry_longitudinal_share":
                    hb.get("car_entry_longitudinal_share", float("nan"))}

    def results() -> Any:
        """Yield ``(task, simulated metrics)`` in fit-file order, from the worker pool."""
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(args.hlp_checkpoint_dir, args.hlp_model_filename)) as pool:
            # | imap keeps fit-file order, so the CSV reads as it always did; a slow episode
            # | only delays the WRITE of the rows behind it, not their computation.
            for task, m in zip(tasks, pool.imap(simulate_fit_row, tasks, chunksize=1)):
                yield task, m

    t_start = time.time()
    n_written = 0
    fh = None
    writer = None
    try:
        for task, m in results():
            key = task["key"]
            out_row = {**human_side(key), **m}
            if writer is None:
                fh = open(out_csv, "a", newline="", encoding="utf-8")
                writer = csv.DictWriter(fh, fieldnames=list(out_row.keys()))
                if not done:
                    writer.writeheader()
            writer.writerow(out_row)
            fh.flush()
            n_written += 1
            print(f"[{task['index'] + 1}/{len(rows)}] {key[0][:8]} g{key[1]} sim_col={m['collisions']} "
                  f"hum_col={human_col.get(key)} ({(time.time() - t_start) / n_written:.0f}s/episode "
                  f"amortised over {args.workers} worker(s))", flush=True)
    finally:
        if fh is not None:
            fh.close()

    all_rows = list(read_done_rows(out_csv).values())
    print(f"simulated {n_written} episodes this run; {len(all_rows)} of {len(rows)} on disk", flush=True)
    if len(all_rows) < len(rows):
        print(f"NOT complete: re-run the same command to finish; plots are drawn only from a "
              f"complete file", flush=True)
        return
    draw_regressions(all_rows, results_dir / "plots", tag)
    print(f"saved {out_csv.name}")


if __name__ == "__main__":
    main()
