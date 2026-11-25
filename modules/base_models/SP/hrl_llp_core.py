"""hrl_llp_core.py — Shared LLP model and environment-interaction helpers.

Extracted from PPO_simpleHRL.py so that the training loop, rollout script,
and any future HLP trainer can import a common set of building blocks without
creating a dependency on training-specific code.

Public API
----------
Constants:   LLP_LANE_X_CENTERS, _ACTION_LAT, _ACTION_LON
Helpers:     llp_extract_player_xy_m, llp_sample_goals,
             llp_sample_env_reward_coef, llp_compute_reward_and_flags,
             llp_augment_obs
Model:       layer_init, GoalConditionedExtractor, Agent
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drive_env.core.config import BOUNDRY_WIDTH, LANECOUNT, TILE_SIZE_M


# ===========================================================================
# Module-level constants
# ===========================================================================

LLP_LANE_X_CENTERS: torch.Tensor = torch.tensor(
    [(BOUNDRY_WIDTH + i + 0.5) * TILE_SIZE_M for i in range(LANECOUNT)],
    dtype=torch.float32,
)

# Action attribute vectors: lateral and longitudinal components per action.
# Order: ACCEL=0, LEFT=1, RIGHT=2, BRAKE=3, IDLE=4, ACCEL_LEFT=5, ACCEL_RIGHT=6, BRAKE_LEFT=7, BRAKE_RIGHT=8
_ACTION_LAT = torch.tensor([ 0, -1,  1,  0,  0, -1,  1, -1,  1], dtype=torch.float32)
_ACTION_LON = torch.tensor([ 1,  0,  0, -1,  0,  1,  1, -1, -1], dtype=torch.float32)

# ===========================================================================
# LLP environment-interaction helpers
# ===========================================================================

def llp_extract_player_xy_m(obs: torch.Tensor, y_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract player x (m) and y (m) from batched observation tensor [N, obs_dim].

    CarDrivingEnv emits obs[0]=x_m, obs[1]=y_normalized (y_norm = y_m / y_max).
    Returns x_m and y_m tensors of shape [N].
    """
    return obs[:, 0], obs[:, 1] * y_max


def llp_sample_goals(
    x_m: torch.Tensor,
    y_m: torch.Tensor,
    goal_forward_m_min: float,
    goal_forward_m_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hardcoded HLP: sample a random lane and forward distance for each env.

    goal_x — lane centre from a uniformly sampled lane index.
    goal_y — current y_m minus dy, where dy ~ Uniform(goal_forward_m_min, goal_forward_m_max).
    Returns (goal_x, goal_y) tensors of shape [N], same device as inputs.
    """
    device = x_m.device
    n = x_m.shape[0]
    lane_idx = torch.randint(0, LANECOUNT, (n,), device=device)
    goal_x = LLP_LANE_X_CENTERS.to(device)[lane_idx]
    dy = torch.rand(n, device=device) * (goal_forward_m_max - goal_forward_m_min) + goal_forward_m_min
    goal_y = torch.clamp(y_m - dy, min=0.0)
    return goal_x, goal_y


def llp_sample_env_reward_coef(
    n: int, coef_min: float, coef_max: float, device: torch.device
) -> torch.Tensor:
    """Sample per-env env-reward blend coefficient uniformly from [coef_min, coef_max].

    When coef_min >= coef_max, returns a constant tensor (no randomness).
    """
    if coef_min >= coef_max:
        return torch.full((n,), coef_min, dtype=torch.float32, device=device)
    return torch.rand(n, device=device) * (coef_max - coef_min) + coef_min


def llp_compute_reward_and_flags(
    x_m: torch.Tensor,
    y_m: torch.Tensor,
    goal_x: torch.Tensor,
    goal_y: torch.Tensor,
    subepisode_steps: torch.Tensor,
    env_done: torch.Tensor,
    args_goal_radius_m: float,
    args_reach_reward: float,
    args_miss_penalty: float,
    args_truncation_penalty: float,
    args_sub_episode_horizon: int,
    args_miss_ends_subepisode: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-env LLP reward and sub-episode done flags.

    Reward structure:
      - Miss:    -miss_penalty when y_m crosses below goal_y without reaching goal
      - Reach:   +reach_reward when goal_dist <= goal_radius_m
      - Timeout: -(miss_penalty + truncation_penalty) when sub_episode_horizon hit

    sub_episode_done fires on: goal reached, miss (if args_miss_ends_subepisode),
    horizon hit, or env_done.
    Returns (llp_reward, sub_episode_done, goal_reached) — float/bool tensors [N].
    """
    goal_dist = torch.hypot(x_m - goal_x, y_m - goal_y)
    goal_reached = goal_dist <= args_goal_radius_m
    missed = (y_m < goal_y) & ~goal_reached

    llp_reward = torch.zeros_like(goal_dist)
    llp_reward[goal_reached] = args_reach_reward
    llp_reward[missed] = -args_miss_penalty

    horizon_hit = subepisode_steps >= args_sub_episode_horizon
    timeout = horizon_hit & ~goal_reached
    llp_reward[timeout] = -(args_miss_penalty + args_truncation_penalty)
    miss_terminates = missed if args_miss_ends_subepisode else torch.zeros_like(missed)
    sub_episode_done = goal_reached | miss_terminates | horizon_hit | env_done

    return llp_reward, sub_episode_done, goal_reached


def llp_augment_obs(
    base_obs: torch.Tensor,
    goal_x: torch.Tensor,
    goal_y: torch.Tensor,
    subepisode_steps: torch.Tensor,
    prev_actions: torch.Tensor,
    env_reward_coef: torch.Tensor,
    y_max: float,
    sub_episode_horizon: int,
) -> torch.Tensor:
    """Build augmented LLP observation tensor [N, base_obs_dim + 6].

    Appends [goal_dx | goal_dy | ts_norm | lat_prev | lon_prev | env_reward_coef]
    to a masked copy of base_obs (ego absolute x/y zeroed out).

    Args:
        base_obs:         [N, obs_dim] raw environment observation
        goal_x/goal_y:    [N] current goal position in world metres
        subepisode_steps: [N] steps elapsed in current sub-episode (long)
        prev_actions:     [N] previous CHOSEN action indices (long), used for lat/lon encoding
        env_reward_coef:  [N] per-env environment reward blend coefficient
        y_max:            scalar — environment y_max for coordinate conversion
        sub_episode_horizon: scalar — horizon for ts_norm normalisation
    """
    device = base_obs.device
    player_x_m, player_y_m = llp_extract_player_xy_m(base_obs, y_max)
    goal_dx = goal_x - player_x_m
    goal_dy = goal_y - player_y_m

    masked_obs = base_obs.clone()
    masked_obs[:, 0] = 0.0
    masked_obs[:, 1] = 0.0

    ts_norm = (subepisode_steps.float() / sub_episode_horizon).unsqueeze(1)
    lat_prev = _ACTION_LAT.to(device)[prev_actions].unsqueeze(1)
    lon_prev = _ACTION_LON.to(device)[prev_actions].unsqueeze(1)
    coef_obs = env_reward_coef.unsqueeze(1)
    parts = [masked_obs, goal_dx.unsqueeze(1), goal_dy.unsqueeze(1), ts_norm, lat_prev, lon_prev, coef_obs]
    return torch.cat(parts, dim=1)


# ===========================================================================
# Model building blocks
# ===========================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GoalConditionedExtractor(nn.Module):
    """Apply env MultiEncoder to obs and append goal features in matching scaled units.
    """

    def __init__(
        self,
        multi_encoder: nn.Module,
        goal_x_scale: float,
        goal_y_scale: float,
    ) -> None:
        super().__init__()
        self.encoder = multi_encoder
        self.n_context = 6  # appended columns after the base obs
        self.output_size = multi_encoder.output_size + self.n_context  # type: ignore[operator]
        self.register_buffer(
            "goal_scale",
            torch.tensor([goal_x_scale, goal_y_scale], dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.n_context
        obs = x[..., :-c]
        goal = x[..., -c:-c + 2] / self.goal_scale
        rest = x[..., -c + 2:]   # ts_norm, lat_prev, lon_prev, env_reward_coef: unscaled
        return torch.cat([self.encoder(obs), goal, rest], dim=-1)


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        feature_extractor: nn.Module,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        input_size = feature_extractor.output_size

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        x = self.feature_extractor(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.feature_extractor(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
