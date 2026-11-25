"""train_hlp_PPO.py — PPO training script for the High-Level Policy (HLP).

[HLP-delta] Derived from PPO_cleanRL.py. All deviations are marked [HLP-delta].
The HLP learns which enemy cars to show the LLP (K independent Bernoulli mask bits).
The LLP is loaded from a checkpoint and kept frozen throughout HLP training.
"""

import functools
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import pufferlib
    import pufferlib.emulation
    import pufferlib.vector
    PUFFER_AVAILABLE = True
except ImportError:
    PUFFER_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# [HLP-delta] new imports
from drive_env.interface_gym import CarDrivingEnv, MultiEncoder
from modules.base_models.SP.hrl_llp_core import (
    Agent as LLPAgent,
    GoalConditionedExtractor,
    LLP_LANE_X_CENTERS,
)
from modules.base_models.SSP.hlp_core import (
    OBS_SIZE, EXPANDED_FEATURES, MAX_OBJECTS, FLAG_FEAT_DIM,
    HLP_PARAM_NAMES, HLP_NUM_PARAMS,
    HLPAgent, hlp_collect_rollout, hlp_sample_params,
)

# [HLP-param] θ column indices used by the TensorBoard charts below
_P_COG  = HLP_PARAM_NAMES.index("hlp_cognitive_cost_coef")
_P_PROX = HLP_PARAM_NAMES.index("hlp_goal_proximity_cost_coef")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 158
    torch_deterministic: bool = True
    cuda: bool = True

    # Env
    env_id: str = "CarDrivingEnv-v0"
    total_timesteps: int = 2_500_000
    learning_rate: float = 2.5e-4
    num_envs: int = 15
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    out_path: str = "modules/base_models/trained_policies"
    save_frequency: int = 200
    vectorization_mode: str = "async"
    puffer_num_workers: int = 0

    # [HLP-delta] HLP-specific args
    hlp_num_car_slots: int = 6
    """Number of Bernoulli outputs K (car slots the HLP can mask)."""
    hlp_hidden_size: int = 64
    """Hidden layer width for HLPAgent actor."""
    hlp_critic_hidden_size_multiplier: int = 1
    """Critic hidden size = hlp_hidden_size * this multiplier. 1 = same as actor."""
    llp_checkpoint_dir: str = "modules/base_models/trained_policies/final_model_llp__260524_2054__1779670440"
    """Path to LLP checkpoint directory (contains model_final.pt + hrl_llp_metadata.json)."""


    # [HLP-density] Training-only traffic-density randomization (change the range here or via
    # CLI to tune the crash-signal density). Per episode each env samples the map car spacing
    # (buffer_y) uniformly in [min, max]; LOWER = DENSER (buffer_y 3->1 ~= 2x cars). Threaded
    # into every training env via env_kwargs; eval/sim omit it so the env default (3) is
    # unchanged. Default {1,3}: denser traffic strengthens BOTH crash-control signals (bad-goal
    # crashes AND hidden-car crashes) without the goal-vs-mask tradeoff that raising
    # llp_env_reward_coef would cause. Use an LLP trained over the SAME range (train_llp_PPO.py
    # default {1,3}) so the frozen LLP is in-distribution across it.
    car_buffer_y_min: int = 1
    """Denser end of the per-episode car-spacing range (lower = denser)."""
    car_buffer_y_max: int = 3
    """Sparser end of the range; 3 = the environment's default density."""

    # [HLP-delta] HLP reward coefficients
    hlp_reach_reward: float = 0.0
    """Reward added when the LLP reaches its goal within the block."""
    hlp_env_reward_coef: float = 0.0
    """Scale factor applied to the sum of raw env rewards accumulated over the block."""
    goal_dy_bin_width_m: float = 0.5
    """Target dy bin width in metres. num_goal_dy_bins is derived from the LLP
    checkpoint goal range; the effective width tiles the range exactly."""

    # [HLP-param] Parameterized reward coefficients — θ. Sampled per env-EPISODE
    # uniformly from [min, max]; values are NORMALIZED to [-1, 1] over their range
    # before being appended to the HLP obs (see Design Log 6).
    # Order: HLP_PARAM_NAMES in hlp_core.py. min == max → fixed coefficient.
    # These pairs REPLACE the former scalar args — old flags like --hlp_collision_penalty
    # error out.
    #
    # [2026-07-09] Ranges = param2_gamma's PROVEN set (Design Log 7): that 1B run
    # conditions on all four coefficients with a stable critic. Every range has a 0
    # ("indifferent") endpoint for contrast. Proximity is the dominant far-lever [0, 0.3]
    # (sole far-pull — flag_collect banned by the non-positive-reward rule). KEY RULE
    # (from the fmiss-vs-dy sweep on param2_gamma): fmiss_max ≈ 1.5-1.7·prox_max so
    # flag-miss can oppose proximity across its whole range — hence fmiss [0, 0.5] with
    # prox [0, 0.3]. The earlier [0, 0.08] was ~4x too narrow (it sat on the flat top of
    # the fmiss->dy curve and could not counter proximity, esp. at high gamma).
    hlp_collision_penalty_min: float = 0.0
    """Min penalty per collision event accumulated over the block."""
    hlp_collision_penalty_max: float = 1.5
    """Max penalty per collision event accumulated over the block."""
    hlp_cognitive_cost_coef_min: float = 0.0
    """Min cognitive cost coefficient: -coef * (valid cars shown) per block."""
    hlp_cognitive_cost_coef_max: float = 0.05
    """Max cognitive cost coefficient."""
    hlp_goal_proximity_cost_coef_min: float = 0.0
    """Min proximity cost coefficient: -coef * (1 - d_norm) per block — closer goals
    penalised more (dominant far-goal lever)."""
    hlp_goal_proximity_cost_coef_max: float = 0.3
    """Max proximity cost coefficient (dominant far-lever)."""
    # [HLP-flag] Flag-miss penalty — drives HLP goal selection (near-pull; misses rise
    # with distance). LLP is unaffected (sparse1 shaping untouched → LLP stays flag-blind).
    hlp_flag_miss_penalty_min: float = 0.0
    """Min penalty per flag missed within the block."""
    hlp_flag_miss_penalty_max: float = 0.5
    """Max penalty per flag missed (fmiss_max ≈ 1.5-1.7·prox_max so it can counter proximity)."""
    # [HLP-flag] Flag-collect reward is a POSITIVE term → BANNED by the non-positive-reward
    # rule (CLAUDE.md). Stays fixed at 0. Do not widen without explicit instruction.
    hlp_flag_collect_reward_min: float = 0.0
    """Fixed at 0 — positive rewards are disallowed (see CLAUDE.md Reward Design Conventions)."""
    hlp_flag_collect_reward_max: float = 0.0
    """Fixed at 0 — positive rewards are disallowed (see CLAUDE.md Reward Design Conventions)."""
    # [HLP-flag] Goal-flag alignment penalty — penalise the chosen goal's distance (metres)
    # to the nearest in-view flag (0 when no flag visible). Denser flag signal than the
    # outcome-based flag-miss penalty: it shapes goal PLACEMENT toward flags directly. Like
    # flag-miss, this competes with the proximity far-pull — a flag near the player pulls the
    # goal in, proximity pulls it out, so tune this against prox_max (start comparable).
    hlp_goal_flag_dist_penalty_min: float = 0.0
    """Min goal-flag distance penalty coefficient: -coef * max(0, dist - hinge) per block."""
    hlp_goal_flag_dist_penalty_max: float = 0.0
    """Max goal-flag distance penalty coefficient (per metre)."""
    # [HLP-flag] HINGE on the goal-flag penalty: it reaches 0 once the goal is within this radius
    # of a flag, instead of only at an unreachable dist=0. Fixed reward SHAPE (not a fittable θ).
    # Removes the measured asymmetry where a flag-present state could never score 0 while a
    # flag-less state got 0 for free (the critic had learned it: V(flag) -0.163 vs V(no-flag) -0.102).
    #
    # CALIBRATION (lane spacing 1.30 m, dy bin 0.492 m):
    #   * MUST be > dy quantization (max 0.246 m) or 0 stays unattainable → no fix.
    #   * MUST be < LANE SPACING (1.30 m) or an ADJACENT-lane goal scores 0 → the agent is
    #     rewarded for the WRONG LANE. Asserted in train() below.
    #   * 0.5 chosen so the Euclidean hinge effectively means "correct lane REQUIRED (any wrong
    #     lane is >=1.30 m away), dy within ~1 bin". At 1.0 it also forgave 2 dy bins (0.98 m) for
    #     free while an adjacent-lane goal cost only -0.015 — almost no lane signal.
    hlp_goal_flag_hinge_m: float = 0.5
    """Radius (m) within which the goal counts as ON the flag → zero goal-flag penalty."""

    # [HLP-delta] Rollout structure
    num_hlp_steps: int = 128
    """Number of HLP blocks per iteration (analogous to num_steps in PPO_cleanRL.py)."""

    # [HLP-delta] Fixed LLP obs parameter — HLP-side choice, NOT loaded from checkpoint.
    # The LLP was trained over a range of env_reward_coef values; HLP training fixes it
    # to a single constant passed to llp_augment_obs every step. Changing this shifts the
    # LLP's obs distribution; use the value the LLP was most commonly trained with (0.1).
    llp_env_reward_coef: float = 0.1
    """Fixed env_reward_coef passed to llp_augment_obs every step during HLP rollouts."""

    # [HLP-delta] LLP args — the defaults below are DEAD VALUES, never used in training.
    # They are unconditionally overwritten in train() right after the LLP checkpoint
    # loads: sub_episode_horizon … llp_miss_ends_subepisode come from the checkpoint's
    # hrl_llp_metadata.json (the LLP must be simulated with the exact values it was
    # trained under), and visible_width/height, max_speed, y_max are measured from a
    # live env. Passing any of these on the CLI has no effect. The ONE LLP-side value
    # that IS taken from Args (not the checkpoint) is llp_env_reward_coef above.
    sub_episode_horizon: int = 100
    goal_forward_m_min: float = 0.2
    goal_forward_m_max: float = 12.0
    goal_radius_m: float = 1.0
    llp_reach_reward: float = 0.0
    llp_miss_penalty: float = 0.0
    llp_truncation_penalty: float = 0.0
    llp_miss_ends_subepisode: bool = False
    visible_width: float = 8.0
    visible_height: float = 11.0
    max_speed: float = 30.0
    y_max: float = 0.0  # filled from env at runtime

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ---------------------------------------------------------------------------
# Env factory (unchanged from PPO_cleanRL.py)
# ---------------------------------------------------------------------------

def _largest_divisor_at_most(n: int, limit: int) -> int:
    for d in range(max(1, min(n, limit)), 0, -1):
        if n % d == 0:
            return d
    return 1


def make_env(env_id, **env_kwargs):
    def thunk():
        env = gym.make(env_id, **env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def make_env_puffer(env_id, buf=None, seed=0, **env_kwargs):
    thunk = make_env(env_id, **env_kwargs)
    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=thunk, buf=buf, seed=seed)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(args: Args) -> Tuple[str, HLPAgent]:
    # [HLP-delta] batch_size = num_hlp_steps * num_envs (one block per env per HLP step)
    args.batch_size     = int(args.num_hlp_steps * args.num_envs)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # Total env steps per iteration = num_hlp_steps * num_envs * sub_episode_horizon
    # For LR schedule we count HLP update iterations
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{os.getpid()}__{int(time.time())}"

    writer = SummaryWriter(f"{args.out_path}/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % "\n".join(
                        [f"|{k}|{v}|" for k, v in vars(args).items()]))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'}")

    # ── Env setup (identical to PPO_cleanRL.py) ─────────────────────────────────
    from drive_env.interface_gym import register_envs
    register_envs()

    # [HLP-density] Thread the per-episode traffic-density range into EVERY training env
    # (SyncVectorEnv / AsyncVectorEnv / puffer + the geometry probe below all consume env_kwargs).
    env_kwargs = {"car_buffer_y_min": args.car_buffer_y_min,
                  "car_buffer_y_max": args.car_buffer_y_max}

    if args.vectorization_mode == "puffer" and not PUFFER_AVAILABLE:
        print("PufferLib not available. Falling back to async vectorization.")
        args.vectorization_mode = "async"
    if args.vectorization_mode == "puffer":
        puffer_env_creators = [functools.partial(make_env_puffer, args.env_id) for _ in range(args.num_envs)]
        puffer_env_args   = [() for _ in range(args.num_envs)]
        puffer_env_kwargs = [env_kwargs for _ in range(args.num_envs)]
        if args.puffer_num_workers > 0:
            nw = _largest_divisor_at_most(args.num_envs, args.puffer_num_workers)
        else:
            cpu_count = os.cpu_count() or args.num_envs
            nw = _largest_divisor_at_most(args.num_envs, max(1, min(args.num_envs, cpu_count // 2)))
        envs = pufferlib.vector.make(
            puffer_env_creators, env_args=puffer_env_args, env_kwargs=puffer_env_kwargs,
            backend=pufferlib.vector.Multiprocessing, num_envs=args.num_envs,
            num_workers=nw, zero_copy=False, overwork=False,
        )
    elif args.vectorization_mode == "async":
        envs = gym.vector.AsyncVectorEnv(
            [make_env(args.env_id, **env_kwargs) for _ in range(args.num_envs)],
            shared_memory=True,
        )
    else:
        raise ValueError(f"unknown vectorization_mode {args.vectorization_mode!r}: use 'puffer' or 'async'")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    # [HLP-delta] Capture env geometry from a temporary single env
    _tmp_env = CarDrivingEnv(**env_kwargs)
    args.visible_width  = float(_tmp_env.visible_width_tiles)
    args.visible_height = float(_tmp_env.visible_height_tiles)
    args.max_speed      = float(_tmp_env.mdp.max_speed)
    args.y_max          = float(_tmp_env.y_max)
    _tmp_env.close()

    B  = args.num_envs
    K  = args.hlp_num_car_slots

    # ── LLP loading ─────────────────────────────────────────────────────────────
    # [HLP-delta] Load frozen LLP from checkpoint + override matching args
    assert args.llp_checkpoint_dir, "Must provide --llp_checkpoint_dir"
    llp_dir = Path(args.llp_checkpoint_dir)
    with open(llp_dir / "hrl_llp_metadata.json") as f:
        llp_meta = json.load(f)

    # Override args that must match LLP training exactly — this is where the Args
    # defaults (and any CLI-passed values) for these fields are discarded in favour of
    # the checkpoint's hrl_llp_metadata.json. See the note in the Args dataclass.
    args.sub_episode_horizon       = int(llp_meta["sub_episode_horizon"])
    args.goal_forward_m_min        = float(llp_meta["goal_forward_m_min"])
    args.goal_forward_m_max        = float(llp_meta["goal_forward_m_max"])
    args.goal_radius_m             = float(llp_meta["goal_radius_m"])
    args.llp_reach_reward          = float(llp_meta["llp_reach_reward"])
    args.llp_miss_penalty          = float(llp_meta["llp_miss_penalty"])
    args.llp_truncation_penalty    = float(llp_meta.get("llp_truncation_penalty", 0.0))
    args.llp_miss_ends_subepisode  = bool(llp_meta["llp_miss_ends_subepisode"])
    # llp_env_reward_coef is a fixed HLP-side constant (see Args) — intentionally NOT
    # loaded from the LLP checkpoint. Passed unchanged to llp_augment_obs every step.

    ep = llp_meta["encoder_params"]
    multi_encoder = MultiEncoder(
        tile_obs_size=int(ep["tile_obs_size"]),
        obs_features=int(ep["obs_features"]),
        ego_hidden=int(ep["ego"]),
        tile_hidden=int(ep["tile"]),
        object_hidden=int(ep["object"]),
        device=device,
        visible_width=float(ep["visible_width"]),
        visible_height=float(ep["visible_height"]),
        max_speed=float(ep["max_speed"]),
    )
    goal_extractor = GoalConditionedExtractor(
        multi_encoder=multi_encoder,
        goal_x_scale=float(ep["goal_x_scale"]),
        goal_y_scale=float(ep["goal_y_scale"]),
    )
    llp_agent = LLPAgent(
        envs=envs,
        hidden_size=int(llp_meta.get("hidden_size", 256)),
        feature_extractor=goal_extractor,
    ).to(device)
    # | The SP trainer writes model_final.pt; an SSP run directory (and the committed checkpoint)
    # | carries the same weights as llp_model_final.pt, so either directory can serve as the SP.
    llp_weights = next((p for p in (llp_dir / "model_final.pt", llp_dir / "llp_model_final.pt") if p.exists()), None)
    if llp_weights is None:
        raise FileNotFoundError(f"no model_final.pt or llp_model_final.pt in {llp_dir}")
    llp_ckpt = torch.load(llp_weights, map_location=device, weights_only=False)
    state_dict = llp_ckpt["model_state_dict"] if isinstance(llp_ckpt, dict) and "model_state_dict" in llp_ckpt else llp_ckpt
    llp_agent.load_state_dict(state_dict)
    llp_agent.requires_grad_(False).eval()
    print(f"LLP loaded from {llp_dir} — {sum(p.numel() for p in llp_agent.parameters()):,} params (frozen)")

    # ── HLP agent ───────────────────────────────────────────────────────────────
    # [HLP-goal] input_dim = BASE dim (goal removed from input — it is now an action;
    # the mask head appends the chosen goal internally)
    # [HLP-param] +HLP_NUM_PARAMS raw θ features appended at the end of the base input
    # [HLP-flag] +FLAG_FEAT_DIM: explicit nearest-flag [present, rel_x, rel_y] for the goal head
    hlp_input_dim = OBS_SIZE * EXPANDED_FEATURES + K * EXPANDED_FEATURES + FLAG_FEAT_DIM + HLP_NUM_PARAMS
    hlp_critic_hidden_size = args.hlp_hidden_size * args.hlp_critic_hidden_size_multiplier
    # [HLP-param] θ sampling ranges in canonical HLP_PARAM_NAMES order
    param_ranges = torch.tensor(
        [[getattr(args, f"{n}_min"), getattr(args, f"{n}_max")] for n in HLP_PARAM_NAMES],
        dtype=torch.float32, device=device,
    )
    # [HLP-flag] The goal-flag hinge must stay INSIDE one lane: at hinge >= lane spacing an
    # adjacent-lane goal lands within the hinge and scores ZERO — i.e. the agent gets rewarded for
    # picking the WRONG LANE. It must also exceed the dy quantization or 0 is unattainable.
    _lane_gap = float(LLP_LANE_X_CENTERS[1] - LLP_LANE_X_CENTERS[0])
    assert args.hlp_goal_flag_hinge_m < _lane_gap, (
        f"hlp_goal_flag_hinge_m={args.hlp_goal_flag_hinge_m} must be < lane spacing {_lane_gap:.2f} m, "
        "else an adjacent-lane goal scores zero goal-flag penalty (wrong lane rewarded)."
    )
    assert args.hlp_goal_flag_hinge_m > args.goal_dy_bin_width_m / 2, (
        f"hlp_goal_flag_hinge_m={args.hlp_goal_flag_hinge_m} must exceed half the dy bin "
        f"({args.goal_dy_bin_width_m/2:.3f} m), else zero penalty is unattainable and the hinge does nothing."
    )

    assert (param_ranges[:, 1] >= param_ranges[:, 0]).all(), \
        "each hlp_*_max must be >= its hlp_*_min"
    # [HLP-goal] dy bins: derive count from LLP checkpoint goal range; effective width
    # re-tiles the range exactly (target width args.goal_dy_bin_width_m)
    _dy_range = args.goal_forward_m_max - args.goal_forward_m_min
    num_goal_dy_bins = max(1, round(_dy_range / args.goal_dy_bin_width_m))
    goal_dy_bin_width_eff = _dy_range / num_goal_dy_bins
    num_lanes = LLP_LANE_X_CENTERS.shape[0]
    hlp_agent = HLPAgent(
        input_dim=hlp_input_dim,
        K=K,
        hidden_size=args.hlp_hidden_size,
        critic_hidden_size=hlp_critic_hidden_size,
        num_lanes=num_lanes,
        num_dy_bins=num_goal_dy_bins,
        goal_dy_min=args.goal_forward_m_min,
        goal_dy_bin_width=goal_dy_bin_width_eff,
        lane_centers=LLP_LANE_X_CENTERS,
    ).to(device)
    optimizer = optim.Adam(hlp_agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"HLPAgent: input_dim={hlp_input_dim}, K={K}, actor_hidden={args.hlp_hidden_size}, critic_hidden={hlp_critic_hidden_size}")
    print(f"HLPAgent goals: {num_lanes} lanes × {num_goal_dy_bins} dy bins "
          f"(width {goal_dy_bin_width_eff:.3f} m over [{args.goal_forward_m_min}, {args.goal_forward_m_max}] m)")
    print(f"HLPAgent parameters: {sum(p.numel() for p in hlp_agent.parameters()):,}")

    # ── Save run metadata + local LLP copies ────────────────────────────────────
    # [HLP-delta] mirrors LLP checkpoint convention; makes each run directory self-contained
    run_dir = Path(args.out_path) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    hlp_meta = {
        "run_name":                 run_name,
        "seed":                     args.seed,
        "num_envs":                 args.num_envs,
        "num_hlp_steps":            args.num_hlp_steps,
        "sub_episode_horizon":      args.sub_episode_horizon,
        "total_timesteps":          args.total_timesteps,
        "learning_rate":            args.learning_rate,
        "gamma":                    args.gamma,
        "gae_lambda":               args.gae_lambda,
        "num_minibatches":          args.num_minibatches,
        "update_epochs":            args.update_epochs,
        "clip_coef":                args.clip_coef,
        "ent_coef":                 args.ent_coef,
        "vf_coef":                  args.vf_coef,
        "hlp_num_car_slots":        args.hlp_num_car_slots,
        "hlp_hidden_size":          args.hlp_hidden_size,
        "hlp_critic_hidden_size_multiplier": args.hlp_critic_hidden_size_multiplier,
        "hlp_input_dim":            hlp_input_dim,
        "hlp_reach_reward":         args.hlp_reach_reward,
        # [HLP-misspen] sampled range; readers reconstruct the conditioning column from these
        "hlp_env_reward_coef":      args.hlp_env_reward_coef,
        # [HLP-goal] goal-selection config — the loaders reconstruct the goal heads from these
        "num_goal_dy_bins":              num_goal_dy_bins,
        "goal_dy_bin_width_eff":         goal_dy_bin_width_eff,
        # [HLP-param] θ config — SOURCE OF TRUTH for which coefficients are
        # parameterized, their obs-feature order, and their training ranges. Downstream
        # consumers (simulate_hlp.py, hlp_policy.py) reconstruct the layout from this.
        "hlp_param_config": [
            {"name": n,
             "min": float(getattr(args, f"{n}_min")),
             "max": float(getattr(args, f"{n}_max"))}
            for n in HLP_PARAM_NAMES
        ],
        "llp_checkpoint_dir":       str(llp_dir.resolve()),
        "goal_forward_m_min":       args.goal_forward_m_min,
        "goal_forward_m_max":       args.goal_forward_m_max,
        "goal_radius_m":            args.goal_radius_m,
        "llp_reach_reward":         args.llp_reach_reward,
        "llp_miss_penalty":         args.llp_miss_penalty,
        "llp_truncation_penalty":   args.llp_truncation_penalty,
        "llp_miss_ends_subepisode": args.llp_miss_ends_subepisode,
        "llp_env_reward_coef":      args.llp_env_reward_coef,
        # [HLP-density] per-episode car-spacing range used during this training run
        "car_buffer_y_min":         args.car_buffer_y_min,
        "car_buffer_y_max":         args.car_buffer_y_max,
        # [HLP-flag] goal-flag penalty hinge radius (reward shape; simulate/probe read it from here)
        "hlp_goal_flag_hinge_m":    args.hlp_goal_flag_hinge_m,
        "anneal_lr":                args.anneal_lr,
        "y_max":                    args.y_max,
        "visible_width":            args.visible_width,
        "visible_height":           args.visible_height,
        "max_speed":                args.max_speed,
    }
    with open(run_dir / "hlp_metadata.json", "w") as f:
        json.dump(hlp_meta, f, indent=2)

    shutil.copy2(llp_weights,                         run_dir / "llp_model_final.pt")
    shutil.copy2(llp_dir / "hrl_llp_metadata.json",   run_dir / "hrl_llp_metadata.json")
    print(f"Run dir: {run_dir}")
    print(f"Saved hlp_metadata.json + LLP copies to {run_dir}")

    # ── Buffer allocation ────────────────────────────────────────────────────────
    # [HLP-delta] block-level buffers (num_hlp_steps × B) replacing step-level buffers
    buffers = {
        "hlp_obs":      torch.zeros(args.num_hlp_steps, B, hlp_input_dim, device=device),
        "car_masks":    torch.zeros(args.num_hlp_steps, B, K, device=device),
        # [HLP-cogperstep] per-block active-step count, so mean_cognitive_cost can reflect the
        # per-step-scaled penalty (understated by ~block length otherwise). TODO(cogperstep):
        # drop with the flag if per-block mode wins.
        "active_steps": torch.zeros(args.num_hlp_steps, B, device=device),
        "hlp_logprobs": torch.zeros(args.num_hlp_steps, B, device=device),
        "hlp_values":   torch.zeros(args.num_hlp_steps, B, device=device),
        "hlp_rewards":  torch.zeros(args.num_hlp_steps, B, device=device),
        "hlp_dones":    torch.zeros(args.num_hlp_steps, B, device=device),
        # [HLP-delta] Per-env V(s) evaluated at each env's first sub_done within a block.
        # Used in GAE as the bootstrap nextvalues instead of hlp_values[t+1].
        # See hlp_collect_rollout (inner loop, newly_done block) for where it is written.
        "hlp_next_values": torch.zeros(args.num_hlp_steps, B, device=device),
        # [HLP-goal] goal action components + the goal offsets that conditioned the mask head.
        "goal_lanes":   torch.zeros(args.num_hlp_steps, B, dtype=torch.long, device=device),
        "goal_dy_bins": torch.zeros(args.num_hlp_steps, B, dtype=torch.long, device=device),
        "goal_dxdy":    torch.zeros(args.num_hlp_steps, B, 2, device=device),
        # [HLP-flag] per-block flag outcome counts (TensorBoard logging only)
        "flag_misses":   torch.zeros(args.num_hlp_steps, B, device=device),
        "flag_collects": torch.zeros(args.num_hlp_steps, B, device=device),
        "goal_flag_dists": torch.zeros(args.num_hlp_steps, B, device=device),
        # [HLP-param] per-block θ (TensorBoard logging only — PPO reads θ from hlp_obs)
        "hlp_params":    torch.zeros(args.num_hlp_steps, B, HLP_NUM_PARAMS, device=device),
    }

    # ── Initial env state ────────────────────────────────────────────────────────
    global_step = 0
    start_time  = time.time()
    obs_np, _   = envs.reset(seed=args.seed)
    next_raw_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    next_done    = torch.zeros(B, device=device)

    # [HLP-delta] Initial LLP carry-over state
    prev_actions = torch.full((B,), 4, dtype=torch.long, device=device)  # 4 = IDLE

    # [HLP-param] Initial per-env θ + lazy-resample flag (see hlp_collect_rollout)
    hlp_params = hlp_sample_params(B, param_ranges, device)
    pending_param_reset = torch.zeros(B, dtype=torch.bool, device=device)


    # [HLP-delta] Detect infos format from a dummy step (do once before main loop)
    _dummy_act = np.zeros(B, dtype=np.int64)
    _d_obs, _d_rew, _d_ter, _d_tru, _d_info = envs.step(_dummy_act)
    infos_are_dict = isinstance(_d_info, dict)
    next_raw_obs = torch.tensor(_d_obs, dtype=torch.float32, device=device)
    next_done    = torch.tensor(np.logical_or(_d_ter, _d_tru), dtype=torch.float32, device=device)
    global_step += B
    # [HLP-exp] Extract initial obs_idx. For pufferlib, guard against non-scalar stacking
    # failures — fall back to all-(-1) (treat all cars as new on first obs change) if bad.
    if infos_are_dict:
        _oi_raw = _d_info.get("obs_idx")
        if _oi_raw is not None and np.asarray(_oi_raw).shape == (B, MAX_OBJECTS):
            _oi = np.asarray(_oi_raw, dtype=np.int32)
        else:
            _oi = np.full((B, MAX_OBJECTS), -1, dtype=np.int32)
    else:
        _oi = np.stack([
            inf.get("obs_idx", np.full(MAX_OBJECTS, -1, dtype=np.int32))
            for inf in _d_info
        ])
    obs_idx_current = torch.tensor(_oi, dtype=torch.long, device=device)
    print(f"Infos format: {'dict — gym async (stacked)' if infos_are_dict else 'list-of-dicts — puffer/sync/async'}")

    # ── Training loop ─────────────────────────────────────────────────────────────
    # [HLP-delta] Block length is variable (ends on sub-episode completion), so we
    # loop until global_step >= total_timesteps rather than using a fixed iteration
    # count. This makes total_timesteps an exact env-step budget regardless of how
    # quickly sub-episodes complete.
    max_steps_per_iter = args.num_hlp_steps * B * args.sub_episode_horizon
    est_iterations = args.total_timesteps // max_steps_per_iter
    print(f"Iterations (est. upper bound): {est_iterations}, max env steps/iter: {max_steps_per_iter:,}")

    cumulative_episodes = 0
    iteration = 0
    pbar = tqdm(total=args.total_timesteps, desc="Training", unit="step")

    while global_step < args.total_timesteps:
        iteration += 1
        if args.anneal_lr:
            frac   = max(0.0, 1.0 - global_step / args.total_timesteps)
            lrnow  = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ── Rollout ──────────────────────────────────────────────────────────────
        # [HLP-delta] replaces `for step in range(args.num_steps):`
        hlp_agent.eval()
        (next_raw_obs, next_done, prev_actions, global_step, mean_sub_ep_len,
         obs_idx_current, hlp_params, pending_param_reset) = hlp_collect_rollout(
            envs=envs,
            llp_agent=llp_agent,
            hlp_agent=hlp_agent,
            buffers=buffers,
            next_raw_obs=next_raw_obs,
            next_done=next_done,
            prev_actions=prev_actions,
            obs_idx_current=obs_idx_current,
            hlp_params=hlp_params,
            pending_param_reset=pending_param_reset,
            infos_are_dict=infos_are_dict,
            B=B, K=K, y_max=args.y_max,
            global_step=global_step,
            device=device,
            args=args,
            param_ranges=param_ranges,
        )
        hlp_agent.train()

        # ── GAE ──────────────────────────────────────────────────────────────────
        # [HLP-delta] renamed variables; bootstrap uses per-env sub_done snapshots
        with torch.no_grad():
            # nextvalues source: buffers["hlp_next_values"][t] — V(s) evaluated at
            # each env's first sub_done within block t, stored by hlp_collect_rollout.
            #
            # Why not hlp_values[t+1] (value at the start of the next block)?
            # The block stays open until ALL envs complete a sub-episode. A fast env
            # that finishes at inner step 10 keeps stepping — and may complete 2–3
            # more sub-episodes — while slow envs catch up. By block-close, its state
            # is 30–80 inner steps past its actual sub_done. hlp_values[t+1] would
            # bootstrap from that displaced state, injecting noise into the critic
            # target for every fast-finishing env. hlp_next_values[t] eliminates this
            # by snapshotting V(s) at the exact moment sub_done fires.
            #
            # nextnonterminal uses hlp_dones[t] uniformly for all t.
            #
            # hlp_dones[t] = episode_done_this_block for block t — it captures any
            # env_done fired DURING block t (written at block close, after all inner
            # steps). This is semantically cleanRL's dones[t+1] (done AFTER step t),
            # so the correct gate is:  nextnonterminal = 1 - hlp_dones[t]  for all t.
            #
            # A prior version used hlp_dones[t+1] for t < T-1 by direct analogy with
            # cleanRL's `1 - dones[t+1]`, but that analogy is wrong: cleanRL's dones[t]
            # is written BEFORE env.step at step t (pre-step convention), so dones[t+1]
            # refers to the done produced BY step t. hlp_dones[t] already carries that
            # semantic — there is no +1 needed. Using [t+1] shifted the episode-boundary
            # mask one block ahead, leaving the actual terminal block unmasked.
            advantages = torch.zeros_like(buffers["hlp_rewards"])
            lastgaelam = 0
            for t in reversed(range(args.num_hlp_steps)):
                nextnonterminal = 1.0 - buffers["hlp_dones"][t]
                # [HLP-delta] sub_done-time bootstrap replaces the old hlp_values[t+1]
                nextvalues = buffers["hlp_next_values"][t]
                delta = (buffers["hlp_rewards"][t]
                         + args.gamma * nextvalues * nextnonterminal
                         - buffers["hlp_values"][t])
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + buffers["hlp_values"]

        # ── Flatten batch ─────────────────────────────────────────────────────────
        b_hlp_obs   = buffers["hlp_obs"].reshape(-1, hlp_input_dim)
        b_logprobs  = buffers["hlp_logprobs"].reshape(-1)
        b_car_masks = buffers["car_masks"].reshape(-1, K)
        b_advantages= advantages.reshape(-1)
        b_returns   = returns.reshape(-1)
        b_values    = buffers["hlp_values"].reshape(-1)
        # [HLP-goal] flattened goal action + conditioning offsets
        b_goal_lanes   = buffers["goal_lanes"].reshape(-1)
        b_goal_dy_bins = buffers["goal_dy_bins"].reshape(-1)
        b_goal_dxdy    = buffers["goal_dxdy"].reshape(-1, 2)

        # ── PPO update ────────────────────────────────────────────────────────────
        b_inds    = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end    = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # [HLP-delta] given_samples=stored actions (PPO ratio recomputation).
                # [HLP-goal] stored goal_dxdy re-conditions the mask head exactly as at
                # rollout time; goal logits depend only on the base input.
                _, newlogprob, entropy, newvalue, _ = hlp_agent.get_action_and_value(
                    b_hlp_obs[mb_inds],
                    goal_dxdy=b_goal_dxdy[mb_inds],
                    given_samples=(
                        b_goal_lanes[mb_inds],
                        b_goal_dy_bins[mb_inds],
                        b_car_masks.float()[mb_inds],
                    ),
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio    = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl     = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(hlp_agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y          = np.var(y_true)
        explained_var  = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Count episode dones across the rollout
        batch_episodes = int(buffers["hlp_dones"].sum().item())
        cumulative_episodes += batch_episodes

        # ── Logging ───────────────────────────────────────────────────────────────
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/episodes_per_batch", batch_episodes, global_step)
        writer.add_scalar("charts/cumulative_episodes", cumulative_episodes, global_step)
        writer.add_scalar("charts/mean_hlp_reward", buffers["hlp_rewards"].mean().item(), global_step)
        writer.add_scalar("charts/mean_mask_density", buffers["car_masks"].mean().item(), global_step)
        # [HLP-param] cognitive cost from per-env θ (scalar arg no longer exists)
        # [HLP-cogperstep] scaled by active_steps so the metric matches the charged penalty
        _cog_scale = buffers["active_steps"]
        writer.add_scalar("charts/mean_cognitive_cost",
                          (buffers["hlp_params"][..., _P_COG]
                           * buffers["car_masks"].sum(dim=-1) * _cog_scale).mean().item(),
                          global_step)
        writer.add_scalar("charts/mean_subepisode_length", mean_sub_ep_len, global_step)
        # [HLP-flag] mean flag outcomes per block (goal-selection driver)
        writer.add_scalar("charts/mean_flag_misses", buffers["flag_misses"].mean().item(), global_step)
        writer.add_scalar("charts/mean_flag_collects", buffers["flag_collects"].mean().item(), global_step)
        # [HLP-flag] mean goal-flag distance per block (goal-alignment monitor — should fall if
        # the HLP learns to place goals on flags)
        writer.add_scalar("charts/mean_goal_flag_dist", buffers["goal_flag_dists"].mean().item(), global_step)
        # [HLP-param] mean sampled θ per coefficient (sanity: matches configured ranges)
        for _i, _n in enumerate(HLP_PARAM_NAMES):
            writer.add_scalar(f"charts/mean_param_{_n}", buffers["hlp_params"][..., _i].mean().item(), global_step)
        # [HLP-goal] chosen goal statistics (dy from bin centres stored in the agent)
        _dy_vals = hlp_agent.dy_bin_centers[buffers["goal_dy_bins"].reshape(-1)]
        writer.add_scalar("charts/mean_goal_dy", _dy_vals.mean().item(), global_step)
        _d_norm = ((_dy_vals - args.goal_forward_m_min) / _dy_range).clamp(0.0, 1.0)
        # [HLP-param] proximity cost from per-env θ (scalar arg no longer exists)
        _prox = buffers["hlp_params"][..., _P_PROX].reshape(-1)
        writer.add_scalar("charts/mean_goal_cost",
                          (_prox * (1.0 - _d_norm)).mean().item(),
                          global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/mean_advantage", b_advantages.abs().mean().item(), global_step)
        writer.add_scalar("charts/mean_return", b_returns.mean().item(), global_step)

        sps = int(global_step / (time.time() - start_time))
        pbar.update(global_step - pbar.n)
        pbar.set_postfix(SPS=sps, iter=iteration)
        writer.add_scalar("charts/SPS", sps, global_step)

        if args.save_frequency > 0 and iteration % args.save_frequency == 0:
            os.makedirs(f"{args.out_path}/{run_name}", exist_ok=True)
            torch.save(hlp_agent.state_dict(),
                       f"{args.out_path}/{run_name}/hlp_step_{global_step}.pt")

    pbar.close()
    print(f"HLPAgent parameters: {sum(p.numel() for p in hlp_agent.parameters()):,}")
    envs.close()
    writer.close()
    print(f"Total training time: {time.time() - start_time:.2f}s")

    # Save final model
    final_path = run_dir / "hlp_model_final.pt"
    torch.save(hlp_agent.state_dict(), str(final_path))
    print(f"Training complete. Run: {run_name}")
    print(f"Saved model: {final_path}")

    import subprocess
    # [HLP-exp] simulate with this module's visualiser (goal-selection aware)
    simulate_path     = str(Path(__file__).resolve().parent / "simulate_hlp.py")

    print(f"\nGenerating rollout for run: {run_dir}")
    try:
        subprocess.run(
            [sys.executable, simulate_path, "--run-dir", str(run_dir)],
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("Simulation timed out after 300s — skipping.")

    return run_name, hlp_agent


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
