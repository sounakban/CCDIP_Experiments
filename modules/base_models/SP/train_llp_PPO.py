"""PPO_simpleHRL.py — PPO trainer with a hardcoded High-Level Policy (HLP).

Extends PPO_cleanRL.py with minimal changes to support hierarchical RL:
  - A hardcoded HLP samples random forward goals (lane x, ahead y).
  - The Low-Level Policy (LLP) is a standard PPO agent trained on those goals.
  - Sub-episodes end when the LLP reaches the goal OR a step horizon is hit.
  - llp_done = env_done OR sub_episode_done  (the PPO/GAE done mask).
  - The base environment keeps running across sub-episode boundaries.
    - Env rewards can be blended into LLP reward with a configurable coefficient.

All modifications to the original PPO loop are clearly marked:
    # <<< LLP ADDITION: ... >>>
"""

from __future__ import annotations

import os
import sys
import json
import functools
import random
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

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

# Ensure project root on sys.path when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drive_env.interface_gym import CarDrivingEnv, register_envs
from modules.base_models.SP.hrl_llp_core import (
    _ACTION_LAT,
    _ACTION_LON,
    Agent,
    GoalConditionedExtractor,
    llp_augment_obs,
    llp_compute_reward_and_flags,
    llp_extract_player_xy_m,
    llp_sample_env_reward_coef,
    llp_sample_goals,
)


# ===========================================================================
# PPO infrastructure (unchanged from PPO_cleanRL.py)
# ===========================================================================

@dataclass
class Args:
    exp_name: str = "hrl_llp_test"
    """experiment name; the run directory is '<exp_name>__<date>__<timestamp>' under out_path"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    env_id: str = "CarDrivingEnv-v0"
    """the id of the environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 15
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    hidden_size: int = 256
    """the number of neurons in the hidden layers of the actor and critic networks"""
    encoder_ego_hidden: int = 64
    """hidden size for ego branch of MultiEncoder"""
    encoder_tile_hidden: int = 256
    """hidden size for tile branch of MultiEncoder"""
    encoder_object_hidden: int = 256
    """hidden size for object branch of MultiEncoder"""
    out_path: str = "modules/base_models/trained_policies"
    """Location for storing all outputs from training"""
    save_frequency: int = 0
    """the frequency of saving the model (0 to disable)"""
    vectorization_mode: str = "async"
    """the vectorization mode to use ('puffer', or 'async' where pufferlib is unavailable)"""
    puffer_num_workers: int = 0
    """number of puffer workers (0 means auto-tune)"""

    # <<< LLP ADDITION: HRL / goal-conditioning arguments >>>
    sub_episode_horizon: int = 100
    """max LLP steps per sub-episode before forced goal reset"""
    goal_forward_m_min: float = 0.2
    """minimum forward distance (metres) for HLP goal sampling; dy ~ Uniform(min, max)"""
    goal_forward_m_max: float = 12.0
    """maximum forward distance (metres) for HLP goal sampling; dy ~ Uniform(min, max)"""
    goal_radius_m: float = 1.0
    """goal success radius in metres"""
    llp_reach_reward: float = 0.0
    """reward given to LLP on successfully reaching the goal"""
    llp_miss_penalty: float = 0.3
    """penalty given to LLP when it crosses goal_y without reaching goal. Set 0, when llp_miss_ends_subepisode == False"""
    llp_miss_ends_subepisode: bool = True
    """if True, a goal-miss event ends the current sub-episode and triggers a new goal sample; if False, the sub-episode continues after a miss"""
    llp_env_reward_coef_min: float = 0.0
    """minimum env-reward blend coefficient; sampled uniformly each sub-episode from [min, max]"""
    llp_env_reward_coef_max: float = 1.0
    """maximum env-reward blend coefficient; set equal to min for a constant coefficient"""
    llp_speed_coef: float = 0.005
    """per-timestep constant penalty applied to LLP reward: llp -= llp_speed_coef"""
    llp_truncation_penalty: float = 0.0
    """extra penalty applied when LLP sub-episode hits sub_episode_horizon without reach/miss"""
    llp_action_switch_coef: float = 0.007
    """per-unit penalty for action switching; max switch (e.g. ACCEL_LEFT→BRAKE_RIGHT) costs 4× this"""
    # [HLP-density] Per-episode traffic-density randomization: each env samples the map car
    # spacing (buffer_y) uniformly in [min, max]; LOWER = DENSER (buffer_y 3->1 ~= 2x cars).
    # Default {1,3} trains the LLP ACROSS the density range so it stays in-distribution when the
    # HLP later uses denser traffic. NOTE: buffer_y={3,3} is the KNOWN-WORKING range — the
    # validated LLP checkpoints were trained at the environment's DEFAULT density (3); {1,3} is
    # the newer, denser regime (added 2026-07-13; see SSP/implementation.md Design Log 9).
    car_buffer_y_min: int = 1
    """Denser end of the per-episode car-spacing range (lower = denser). {3,3}=known-working."""
    car_buffer_y_max: int = 3
    """Sparser end; 3 = the environment's default (validated) density."""
    # <<< END LLP ADDITION >>>

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def _build_run_metadata(args: Args, run_name: str, agent: Agent | None = None) -> dict:
    """Build metadata payload for run artifacts."""
    return {
        "run_name": run_name,
        "exp_name": args.exp_name,
        "env_id": args.env_id,
        "hidden_size": args.hidden_size,
        "encoder_params": getattr(agent, "encoder_params", None),
        "goal_forward_m_min": args.goal_forward_m_min,
        "goal_forward_m_max": args.goal_forward_m_max,
        "goal_radius_m": args.goal_radius_m,
        "llp_reach_reward": args.llp_reach_reward,
        "llp_miss_penalty": args.llp_miss_penalty,
        "llp_env_reward_coef_min": args.llp_env_reward_coef_min,
        "llp_env_reward_coef_max": args.llp_env_reward_coef_max,
        "llp_speed_coef": args.llp_speed_coef,
        "llp_truncation_penalty": args.llp_truncation_penalty,
        "llp_action_switch_coef": args.llp_action_switch_coef,
        # [HLP-density] per-episode car-spacing range this LLP was trained over ({3,3}=default density)
        "car_buffer_y_min": args.car_buffer_y_min,
        "car_buffer_y_max": args.car_buffer_y_max,
        "llp_miss_ends_subepisode": args.llp_miss_ends_subepisode,
        "sub_episode_horizon": args.sub_episode_horizon,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
    }


def make_env(env_id, **env_kwargs):
    def thunk():
        env = gym.make(env_id, **env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def make_env_puffer(env_id, buf=None, seed=0, **env_kwargs):
    thunk = make_env(env_id, **env_kwargs)
    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=thunk, buf=buf, seed=seed)


def _largest_divisor_at_most(n: int, limit: int) -> int:
    for d in range(max(1, min(n, limit)), 0, -1):
        if n % d == 0:
            return d
    return 1


# ===========================================================================
# Training loop
# ===========================================================================

def train(args: Args) -> tuple[str, Agent]:
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.exp_name}__{datetime.now().strftime('%y%m%d_%H%M')}__{int(time.time())}"
    run_dir = Path(args.out_path) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    print(f"using {'GPU' if torch.cuda.is_available() else 'CPU'}")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # <<< LLP ADDITION: action-switch penalty matrix [9×9] built once on device >>>
    _switch_penalty_matrix = (
        (_ACTION_LAT.unsqueeze(0) - _ACTION_LAT.unsqueeze(1)).abs() +
        (_ACTION_LON.unsqueeze(0) - _ACTION_LON.unsqueeze(1)).abs()
    ).to(device)
    # <<< END LLP ADDITION >>>

    # Register env id so async/puffer workers can import it.
    register_envs()

    # env setup — identical vectorization branches to PPO_cleanRL
    # [HLP-density] thread the per-episode traffic-density range into every training env
    env_kwargs = {"training_mode": True,
                  "car_buffer_y_min": args.car_buffer_y_min,
                  "car_buffer_y_max": args.car_buffer_y_max}

    # <<< LLP ADDITION: staggered first-episode start positions to prevent synchronised env resets >>>
    # Evenly-spaced plr_start_y_arg values give each env a unique first-episode start position.
    # plr_start_y_arg is consumed after one use in _reset_world(), so subsequent resets are unaffected;
    # equal episode lengths then keep the phase offsets self-maintaining throughout training.
    _per_env_kwargs = [
        {**env_kwargs, "plr_start_y_arg": i / args.num_envs}
        for i in range(args.num_envs)
    ]
    # <<< END LLP ADDITION >>>
    if args.vectorization_mode == "puffer":
        if PUFFER_AVAILABLE:
            # pufferlib 3.0.0 requires list inputs when creators are passed as a list.
            puffer_env_creators = [
                functools.partial(make_env_puffer, args.env_id)
                for i in range(args.num_envs)
            ]
            puffer_env_args = [() for _ in range(args.num_envs)]
            puffer_env_kwargs = _per_env_kwargs  # <<< LLP ADDITION: per-env staggered start >>>
            if args.puffer_num_workers > 0:
                puffer_num_workers = _largest_divisor_at_most(args.num_envs, args.puffer_num_workers)
            else:
                cpu_count = os.cpu_count() or args.num_envs
                target = max(1, min(args.num_envs, cpu_count // 2))
                puffer_num_workers = _largest_divisor_at_most(args.num_envs, target)
            print(
                f"Puffer config: num_envs={args.num_envs}, "
                f"num_workers={puffer_num_workers}"
            )
            envs = pufferlib.vector.make(
                puffer_env_creators,
                env_args=puffer_env_args,
                env_kwargs=puffer_env_kwargs,
                backend=pufferlib.vector.Multiprocessing,  # For debugging use: pufferlib.vector.Serial
                num_envs=args.num_envs,
                num_workers=puffer_num_workers,
                zero_copy=False,
                overwork=False,
            )
            # Keep PufferLib-provided single_* spaces; do not overwrite with joint spaces.
        else:
            print("PufferLib not available. Falling back to async vectorization.")
            args.vectorization_mode = "async"
            envs = gym.vector.AsyncVectorEnv(
                [make_env(args.env_id, **_per_env_kwargs[i]) for i in range(args.num_envs)],  # <<< LLP ADDITION: per-env staggered start >>>
                shared_memory=True,
            )
    elif args.vectorization_mode == "async":
        envs = gym.vector.AsyncVectorEnv(
            [make_env(args.env_id, **_per_env_kwargs[i]) for i in range(args.num_envs)],  # <<< LLP ADDITION: per-env staggered start >>>
            shared_memory=True,
        )
    else:
        raise ValueError(f"unknown vectorization_mode {args.vectorization_mode!r}: use 'puffer' or 'async'")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # <<< LLP ADDITION: get y_max once for coordinate conversion and build obs encoder >>>
    _sample_env = CarDrivingEnv(training_mode=True)
    y_max = float(_sample_env.y_max)
    encoder_factory = _sample_env.get_obs_encoder_network(
        ego_hidden_size=args.encoder_ego_hidden,
        tile_hidden_size=args.encoder_tile_hidden,
        object_hidden_size=args.encoder_object_hidden,
    )
    feature_extractor = GoalConditionedExtractor(
        encoder_factory(device=device),
        goal_x_scale=float(_sample_env.visible_width_tiles),
        goal_y_scale=float(_sample_env.visible_height_tiles),
    ).to(device)
    encoder_params = {
        "ego": args.encoder_ego_hidden,
        "tile": args.encoder_tile_hidden,
        "object": args.encoder_object_hidden,
        "tile_obs_size": _sample_env.tile_obs_size,
        "obs_features": _sample_env.mdp.OBS_FEATURES,
        "visible_width": _sample_env.visible_width_tiles,
        "visible_height": _sample_env.visible_height_tiles,
        "max_speed": float(_sample_env.mdp.max_speed),
        "goal_x_scale": float(_sample_env.visible_width_tiles),
        "goal_y_scale": float(_sample_env.visible_height_tiles),
    }
    _sample_env.close()
    # <<< END LLP ADDITION >>>

    agent = Agent(envs, hidden_size=args.hidden_size, feature_extractor=feature_extractor).to(device)
    agent.encoder_params = encoder_params
    # Save metadata before training starts so run config is available immediately.
    (run_dir / "hrl_llp_metadata.json").write_text(
        json.dumps(_build_run_metadata(args, run_name, agent), indent=2),
        encoding="utf-8",
    )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # <<< LLP ADDITION: obs storage uses augmented dim (base_obs + 2 goal features) >>>
    base_obs_dim = int(np.prod(envs.single_observation_space.shape))
    aug_obs_dim = base_obs_dim + 6
    obs        = torch.zeros((args.num_steps, args.num_envs, aug_obs_dim), device=device)
    # <<< END LLP ADDITION >>>
    # ALGO Logic: Storage setup
    actions    = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs   = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards    = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones      = torch.zeros((args.num_steps, args.num_envs), device=device)
    values     = torch.zeros((args.num_steps, args.num_envs), device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    raw_obs, _ = envs.reset(seed=args.seed)
    next_obs_base = torch.tensor(raw_obs, dtype=torch.float32, device=device)  # [N, obs_dim]
    next_done = torch.zeros(args.num_envs, device=device)
    cumulative_episodes_count = 0

    # <<< LLP ADDITION: per-env goal state and sub-episode counters (trainer-side tensors) >>>
    x_m, y_m = llp_extract_player_xy_m(next_obs_base, y_max)
    goal_x, goal_y = llp_sample_goals(x_m, y_m, args.goal_forward_m_min, args.goal_forward_m_max)      # [N]
    subepisode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=device)

    # Diagnostic accumulators
    env_reward_accum = torch.zeros(args.num_envs, device=device)
    llp_reward_accum = torch.zeros(args.num_envs, device=device)
    action_switch_accum = torch.zeros(args.num_envs, device=device)
    subepisode_count = 0
    subepisode_goals_reached = 0
    prev_actions = torch.full((args.num_envs,), 4, dtype=torch.long, device=device)  # IDLE
    env_reward_coef = llp_sample_env_reward_coef(  # <<< LLP ADDITION: per-subepisode env reward coef >>>
        args.num_envs, args.llp_env_reward_coef_min, args.llp_env_reward_coef_max, device
    )
    # <<< END LLP ADDITION >>>

    next_obs = llp_augment_obs(next_obs_base, goal_x, goal_y, subepisode_steps, prev_actions, env_reward_coef, y_max, args.sub_episode_horizon)

    pbar = tqdm(range(1, args.num_iterations + 1))
    for iteration in pbar:
        batch_episodes_tensor = torch.zeros(1, device=device)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = max(frac, 0.0) * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            raw_next_obs, env_reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_obs_base = torch.tensor(raw_next_obs, dtype=torch.float32, device=device)
            env_done = torch.tensor(
                np.logical_or(terminations, truncations), dtype=torch.float32, device=device
            )

            # <<< LLP ADDITION: compute LLP reward and sub-episode boundary >>>
            subepisode_steps += 1
            x_m, y_m = llp_extract_player_xy_m(next_obs_base, y_max)
            env_reward_t = torch.tensor(env_reward, dtype=torch.float32, device=device)
            env_reward_accum += env_reward_t

            llp_reward, sub_episode_done, goal_reached = llp_compute_reward_and_flags(
                x_m, y_m, goal_x, goal_y, subepisode_steps,
                env_done.bool(),
                args.goal_radius_m,
                args.llp_reach_reward, args.llp_miss_penalty,
                args.llp_truncation_penalty,
                args.sub_episode_horizon,
                args.llp_miss_ends_subepisode,
            )

            # Blend base environment reward into LLP training reward using per-subepisode coef.
            llp_reward = llp_reward + env_reward_coef * env_reward_t  # <<< LLP ADDITION: per-subepisode coef >>>

            # Per-timestep constant living penalty.
            if args.llp_speed_coef != 0.0:
                llp_reward = llp_reward - args.llp_speed_coef

            # <<< LLP ADDITION: action-switch penalty >>>
            switch_pen = _switch_penalty_matrix[prev_actions, action]  # [num_envs], values 0–4
            if args.llp_action_switch_coef != 0.0:
                llp_reward = llp_reward - args.llp_action_switch_coef * switch_pen
            action_switch_accum += switch_pen
            prev_actions = action.clone()
            prev_actions[env_done.bool()] = 4  # reset to IDLE on hard env reset
            # <<< END LLP ADDITION >>>

            llp_reward_accum += llp_reward

            # llp_done is the PPO/GAE done mask — fires at sub-episode OR env boundary
            llp_done = (sub_episode_done.float() + env_done).clamp(max=1.0)

            # Store LLP reward (not env reward) for PPO training
            rewards[step] = llp_reward
            next_done = llp_done

            # Reset goals for envs where sub-episode ended but env is still running
            new_sub = sub_episode_done & ~env_done.bool()
            if new_sub.any():
                new_gx, new_gy = llp_sample_goals(x_m, y_m, args.goal_forward_m_min, args.goal_forward_m_max)
                goal_x = torch.where(new_sub, new_gx, goal_x)
                goal_y = torch.where(new_sub, new_gy, goal_y)
                subepisode_steps[new_sub] = 0
                # <<< LLP ADDITION: resample env_reward_coef for new sub-episodes >>>
                new_coef = llp_sample_env_reward_coef(
                    args.num_envs, args.llp_env_reward_coef_min, args.llp_env_reward_coef_max, device
                )
                env_reward_coef = torch.where(new_sub, new_coef, env_reward_coef)
                # <<< END LLP ADDITION >>>

            # Re-sample goals for envs that env-reset (auto-reset by vectorized env)
            if env_done.any():
                reset_mask = env_done.bool()
                new_gx, new_gy = llp_sample_goals(x_m, y_m, args.goal_forward_m_min, args.goal_forward_m_max)
                goal_x = torch.where(reset_mask, new_gx, goal_x)
                goal_y = torch.where(reset_mask, new_gy, goal_y)
                subepisode_steps[reset_mask] = 0
                # <<< LLP ADDITION: resample env_reward_coef on env reset >>>
                new_coef = llp_sample_env_reward_coef(
                    args.num_envs, args.llp_env_reward_coef_min, args.llp_env_reward_coef_max, device
                )
                env_reward_coef = torch.where(reset_mask, new_coef, env_reward_coef)
                # <<< END LLP ADDITION >>>

            subepisode_count += int(sub_episode_done.sum().item())
            subepisode_goals_reached += int(goal_reached.sum().item())
            # <<< END LLP ADDITION >>>

            next_obs = llp_augment_obs(next_obs_base, goal_x, goal_y, subepisode_steps, prev_actions, env_reward_coef, y_max, args.sub_episode_horizon)

            # Count env-level episodes (for logging parity with PPO_cleanRL)
            # Count episodes that just finished (reliable across all vectorization modes)
            batch_episodes_tensor += env_done.sum()

        batch_episodes = int(batch_episodes_tensor.item())
        cumulative_episodes_count += batch_episodes

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        # flatten the batch
        b_obs       = obs.reshape((-1, aug_obs_dim))   # <<< LLP ADDITION: use aug_obs_dim >>>
        b_logprobs  = logprobs.reshape(-1)
        b_actions   = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns   = returns.reshape(-1)
        b_values    = values.reshape(-1)

        # PPO update (unchanged)
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/episodes_per_batch", batch_episodes, global_step)
        writer.add_scalar("charts/cumulative_episodes", cumulative_episodes_count, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/advantage", b_advantages.mean().abs().item(), global_step)

        # <<< LLP ADDITION: LLP-specific diagnostics >>>
        batch_steps = args.num_steps * args.num_envs
        writer.add_scalar("llp/mean_llp_reward",  (llp_reward_accum.sum() / batch_steps).item(), global_step)
        writer.add_scalar("llp/mean_env_reward",  (env_reward_accum.sum() / batch_steps).item(), global_step)
        if subepisode_count > 0:
            writer.add_scalar("llp/mean_episode_llp_reward", (llp_reward_accum.sum() / subepisode_count).item(), global_step)
        writer.add_scalar("llp/subepisode_count", subepisode_count, global_step)
        if subepisode_count > 0:
            writer.add_scalar("llp/goal_reached_rate", subepisode_goals_reached / subepisode_count, global_step)
        writer.add_scalar("llp/mean_action_switch_distance", (action_switch_accum.sum() / batch_steps).item(), global_step)
        env_reward_accum.zero_()
        llp_reward_accum.zero_()
        action_switch_accum.zero_()
        subepisode_count = 0
        subepisode_goals_reached = 0
        # <<< END LLP ADDITION >>>

        sps = int(global_step / (time.time() - start_time))
        pbar.set_postfix(SPS=sps)
        writer.add_scalar("charts/SPS", sps, global_step)

        # Checkpoint saving
        if args.save_frequency > 0 and iteration % args.save_frequency == 0:
            ckpt = f"{args.out_path}/{run_name}/hrl_llp_step_{global_step}.pt"
            Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save(agent.state_dict(), ckpt)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent parameters: {total_params:,}")
    envs.close()
    writer.close()
    print(f"Total training time: {time.time() - start_time:.2f}s")
    return run_name, agent


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    args = tyro.cli(Args)

    run_name, agent = train(args)

    run_dir = Path(args.out_path) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model_final.pt"
    torch.save(agent.state_dict(), model_path)

    metadata = _build_run_metadata(args, run_name, agent)
    (run_dir / "hrl_llp_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Training complete. Run: {run_name}")
    print(f"Saved model: {model_path}")



if __name__ == "__main__":
    main()
