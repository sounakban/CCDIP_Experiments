"""
Gymnasium wrapper for CarMDP.
"""

from typing import Dict, Any, Callable

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn

from drive_env.core.mdp import CarMDP
from drive_env.core.config import FPS, LANECOUNT, BOUNDRY_WIDTH, TILE_SIZE_M
from drive_env.core.utils import initialize_world, get_obs_window
from drive_env.core.maps.objects import objectID_dict
from drive_env.core.maps.mapgenerator import MapGenerator



class CarDrivingEnv(gym.Env):
    # | Gym properties
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    # | Environment Properties
    REWARD_GOAL = 100           # Absolute reward



    def __init__(self, seed = 42,
                 training_mode:bool = True,
                 plr_start_y_arg:float|None = None,
                 car_buffer_y_min:int|None = None,
                 car_buffer_y_max:int|None = None):
        """
        Class __init__

        :param seed: See to use for world initializations
        :param training_mode: Whether environment is being used for training
        :type training_mode: bool
        :param plr_start_y_arg: Specify the agent start location as normalized [0,1] distance from goal, for
                                testing environments and low level policy (Hierarchical RL) training desynchronization.
        :type plr_start_y_arg: float
        :param car_buffer_y_min: Training-only traffic-density randomization. If set (with
                                car_buffer_y_max), each episode samples the map car spacing
                                (buffer_y) uniformly in [min, max]; LOWER = DENSER. None/None
                                (default) disables it — the map generator keeps its default
                                spacing (3), so behaviour is unchanged for all other callers.
        :type car_buffer_y_min: int | None
        :param car_buffer_y_max: Upper bound of the sampled car spacing (see car_buffer_y_min).
        :type car_buffer_y_max: int | None
        """
        super().__init__()

        self.seed = seed
        self.mdp_cls = CarMDP
        self.plr_start_y_arg = plr_start_y_arg
        # [HLP-density] Optional per-episode traffic-density randomization (training only).
        # None/None => disabled => map generator keeps its default spacing (buffer_y=3), so
        # every existing caller and all eval/deploy are byte-for-byte unchanged.
        self.car_buffer_y_min = car_buffer_y_min
        self.car_buffer_y_max = car_buffer_y_max

        # | Gym Variables
            # | Define spaces
        self.action_space = spaces.Discrete(9)  # check MDP logic
        high = np.array([np.inf]* (CarMDP.OBS_SIZE*CarMDP.OBS_FEATURES), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.reset_count = 1    # Track number of time environment reset

        self.training_mode = training_mode  # Training mode: objects are not synced back to Obj instances
        # | Map dimensions
        self.sec_count = 5
        self.sec_len = 200
        self.car_freq = self.sec_count * (self.sec_len // 2)

        # | Seeded NumPy RNG — used for all stochastic choices in the env wrapper
        self.np_rng = np.random.default_rng(seed)

        self.init_mapgenerator(seed)
        self._reset_world()



    def init_mapgenerator(self, seed: int):
        self.mapgen_obj =  MapGenerator(section_count=self.sec_count,
                                        section_length=self.sec_len,
                                        lanecount=LANECOUNT,
                                        boundry_width=BOUNDRY_WIDTH,
                                        seed=seed,
                                        car_freq=self.car_freq)



    def _reset_world(self):
        self.time_step = 0

        # [HLP-density] Per-episode traffic-density randomization. Gated: only overrides the
        # generator's car spacing when a bound is set (else current behaviour is untouched).
        # The map regenerates every reset, so a fresh spacing here applies to this episode.
        if self.car_buffer_y_min is not None or self.car_buffer_y_max is not None:
            _lo = self.car_buffer_y_min if self.car_buffer_y_min is not None else 3
            _hi = self.car_buffer_y_max if self.car_buffer_y_max is not None else 3
            self.mapgen_obj.car_buffer_y = int(self.np_rng.integers(min(_lo, _hi), max(_lo, _hi) + 1))
        map = self.mapgen_obj.generate_map()

        plr_start_y = None
        if self.plr_start_y_arg is not None and self.reset_count >= 2:
            # | Apply the requested start position for this episode only, then clear it so
            # | subsequent resets revert to the default start (top of map).
            # | reset_count >= 2 skips the __init__ call to _reset_world() (reset_count == 1)
            # | so the stagger takes effect on the first external envs.reset() call instead.
            plr_start_y = min(map["plr_y"], 17 + (self.plr_start_y_arg * len(map["base"])))     # 17 buffer rows in map
            self.plr_start_y_arg = None  # | Consume after first use — subsequent resets use default start

        # | Initialize environment
        tiles, objs, player, self.mdp, visible_window_size = initialize_world(
                                                                                map,
                                                                                plr_start_y=plr_start_y,
                                                                                mdp_cls=self.mdp_cls,
                                                                                seed=self.seed,
                                                                            )
        self.mdp.sync_object_view = not self.training_mode  # Do not sync objects in training mode

        self.visible_width_tiles, self.visible_height_tiles = visible_window_size
        # | Tile row count must match what observationRL writes (the MDP's relative tile
        # | mesh), NOT the camera window area — the two differ (96 vs 88), and using the
        # | camera value misrouted the last tile-grid row into the object encoder.
        self.tile_obs_size = self.mdp._num_rel_tiles

        # |Change rewards
        self.mdp.goal_reward = self.REWARD_GOAL
        self.mdp.living_penalty_val = 0 # Disable sparse living penalty in MDP

        # | Useful variables
            # | Get the mixumum allowed values of x-y coordinates
        self.y_max = len(self.mdp.world.tiles) * TILE_SIZE_M
        self.x_max = len(self.mdp.world.tiles[0]) * TILE_SIZE_M

        self.map_duration = self.mdp.get_map_duration(0.3)     # Estimated framecount to reach goal (assuming 30% of max speed)
        self.reset_count += 1



    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_world()
        obs = self._obs()
        info = self._get_episode_outcome_info(info={})
        return obs, info



    def _get_episode_outcome_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        event = info.get("event", "")

        # [HLP-delta] per-step collision signal (a crash never ends the episode)
        is_collision = int(event.startswith("crash:"))
        # [HLP-flag] per-step flag outcome signals — consumed ONLY by the HLP block reward
        # (goal-selection driver). LLP-facing shaping (sparse1) is untouched, so the LLP
        # stays flag-blind. Flag-specific match: future miss-able object kinds do not join
        # this signal. Pickup event is "interact:object_flag" (flag reward=0.0, removed);
        # miss event is "miss:object_flag" (raw penalty −1.0, removed) — see mdp.py.
        is_flag_miss = int(event.startswith("miss:object_flag"))
        is_flag_collect = int(event == "interact:object_flag")
        # [HLP-exp] obs_idx: maps obs object-section position j → obj_state row index.
        # Padded to CarMDP.MAX_OBJECTS with -1 for unused slots (fixed shape for vectorised envs).
        _obs_idx_pad = np.full(CarMDP.MAX_OBJECTS, -1, dtype=np.int32)
        _obs_idx_pad[:len(self.mdp._obs_idx)] = self.mdp._obs_idx
        info.update({
            "is_collision": np.asarray(is_collision, dtype=np.float32),
            "is_flag_miss": np.asarray(is_flag_miss, dtype=np.float32),
            "is_flag_collect": np.asarray(is_flag_collect, dtype=np.float32),
            "obs_idx": _obs_idx_pad,
        })
        return info



    def _obs(self, reshape_obs=True):
        _, _, x0, x1, y0, y1 = get_obs_window(self.visible_height_tiles,
                                                self.visible_width_tiles,
                                                self.x_max, self.y_max,
                                                self.mdp.player)
        w = (x0, x1, y0, y1)
        obs = self.mdp.observationRL(w)

        if reshape_obs:
            return obs.reshape(-1)
        return obs



    def step(self, action):
        _, reward, terminated, info = self.mdp.step(int(action))

        self.time_step += 1
        obs = self._obs(reshape_obs=False)

        truncated = True if self.mdp.timestep >= self.map_duration else False

        reward = self.reward_sparse1(reward, info, terminated, truncated)
        info = self._get_episode_outcome_info(info=dict(info))

        # | Return raw flattened observation. Normalization happens on GPU inside MultiEncoder.
        return obs.reshape(-1), reward, terminated, truncated, info



    def reward_sparse1(self, reward, info, terminated, truncated):
        """
        Simple sprase reward function, which rewards goal reaching, crashing and timing out
        (Set gamma to very high values since end-of-episode reward is important, gamma = 0.9997)

        :param reward: Base reward from MDP step (only its sign is used)
        :param info: Info dict from MDP step (its "event" names a crash or interaction)
        :param terminated: Whether the MDP reached the goal this step
        :param truncated: Whether the step budget ran out this step
        :return: The computed reward
        :rtype: float
        """

        ## Episode Event Rewards ##
        if terminated and reward > 0:
            return 0.0
        elif  reward < 0 and any(e in info.get('event', '') for e in ['crash','interact']):
            # | If terminated from crashing, cause large penalty (ignore crasing reward multiplier)
            return -1.0
        elif truncated:
            # | Massive penalty for not reaching goal in time
            # | This makes timing out much worse than occasional crashes
            return -1.0

        return 0.0  # Disable MDP and event-based rewards


    def get_obs_encoder_network(self,
                                ego_hidden_size=32,
                                tile_hidden_size=64,
                                object_hidden_size=128,
                                ) -> Callable:
        """
        Returns a torch network with three single-layer encoders for ego state, tiles, and objects.

        The network takes a flattened observation and splits it into three parts:
        - Ego state: first 7 features (1 observation * 7 features)
        - Tile observations: next (tile_obs_size * 7) features
        - Object observations: remaining features

        Each part is encoded separately with a single linear layer and concatenated.

        :param ego_hidden_size: Hidden layer size for ego encoder
        :param tile_hidden_size: Hidden layer size for tile encoder
        :param object_hidden_size: Hidden layer size for object encoder
        :return: A torch Module that encodes observations
        """

        def obs_encoder(device=None) -> MultiEncoder:
            assert device is not None, "Please provide device info to initialize encoder"
            return MultiEncoder(
                tile_obs_size=self.tile_obs_size,
                obs_features=CarMDP.OBS_FEATURES,
                ego_hidden=ego_hidden_size,
                tile_hidden=tile_hidden_size,
                object_hidden=object_hidden_size,
                device=device,
                visible_width=self.visible_width_tiles,
                visible_height=self.visible_height_tiles,
                max_speed=self.mdp.max_speed
            )

        return obs_encoder


    def close(self):
        pass


def preprocess_obs(
    obs_flat: torch.Tensor,
    visible_width: float,
    visible_height: float,
    max_speed: float,
    num_classes: int,
    ego_pos_reset: bool = True,
) -> torch.Tensor:
    """Normalize and one-hot expand a raw flat observation, without encoding or pooling.

    Extracted from MultiEncoder.forward() steps 1-3 for use by the HLP, which
    needs the preprocessed observation as direct input rather than a pooled encoding.

    Args:
        obs_flat:      (..., OBS_SIZE * OBS_FEATURES) raw flat observation tensor.
        visible_width: Observation window width in tiles (for dx normalization).
        visible_height: Observation window height in tiles (for dy normalization).
        max_speed:     Maximum agent speed (for vx/vy normalization).
        num_classes:   Total number of entity kind IDs (= max(objectID_dict.values()) + 1).
        ego_pos_reset: If True, zero ego absolute x and y (col 0 and 1 of row 0).

    Returns:
        (..., OBS_SIZE * expanded_features) preprocessed flat tensor where
        expanded_features = 6 + num_classes.
    """
    original_shape = obs_flat.shape
    squeeze = obs_flat.ndim == 1
    if squeeze:
        obs_flat = obs_flat.unsqueeze(0)
    elif obs_flat.ndim > 2:
        obs_flat = obs_flat.reshape(-1, obs_flat.shape[-1])

    obs = obs_flat.view(obs_flat.shape[0], -1, 7).clone()

    obs[..., 0] /= visible_width
    obs[:, 1:, 1] /= visible_height
    obs[..., 2:4] /= max_speed

    if ego_pos_reset:
        obs[:, 0, 0] = 0.0
        obs[:, 0, 1] = 0.0

    base_feats = obs[..., :6]
    ids = obs[..., 6].long().clamp(0, num_classes - 1)
    one_hot = torch.nn.functional.one_hot(ids, num_classes=num_classes).float()
    expanded = torch.cat([base_feats, one_hot], dim=-1)
    out = expanded.reshape(obs_flat.shape[0], -1)

    if squeeze:
        out = out.squeeze(0)
    elif len(original_shape) > 2:
        out = out.reshape(*original_shape[:-1], -1)
    return out


class MultiEncoder(nn.Module):
    """Observation encoder with per-entity encoding and max pooling over tiles and objects.

    Each tile and object is encoded independently through a shared MLP, then max-pooled
    across the entity dimension to produce a permutation-invariant representation.
    """

    def __init__(self, tile_obs_size, obs_features=7, ego_hidden=32, tile_hidden=64, object_hidden=128,
                 device=None, visible_width=1.0, visible_height=1.0, max_speed=1.0):
        super().__init__()
        self.tile_obs_size = tile_obs_size
        self.obs_features = obs_features
        self.num_classes = max(objectID_dict.values()) + 1
        self.expanded_features = (obs_features - 1) + self.num_classes

        self.visible_width = visible_width
        self.visible_height = visible_height
        self.max_speed = max_speed

        ef = self.expanded_features

        # | Per-entity 2-layer MLPs with LayerNorm; applied independently to each entity
        self.ego_encoder = nn.Sequential(
            nn.Linear(ef, ego_hidden),
            nn.ReLU(),
            nn.Linear(ego_hidden, ego_hidden),
        )
        self.tile_encoder = nn.Sequential(
            nn.Linear(ef, tile_hidden),
            nn.ReLU(),
            nn.Linear(tile_hidden, tile_hidden),
        )
        self.object_encoder = nn.Sequential(
            nn.Linear(ef, object_hidden),
            nn.ReLU(),
            nn.Linear(object_hidden, object_hidden),
        )

        self.output_size = ego_hidden + tile_hidden + object_hidden

        if device is not None:
            self.to(device)

    def forward(self, obs, ego_pos_reset: bool = True):
        # | obs shape: (*batch_dims, total_features)
        original_shape = obs.shape
        squeeze_output = False

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        elif obs.ndim > 2:
            obs = obs.reshape(-1, obs.shape[-1])

        # | 1. Reshape to (batch, N_entities, 7)
        obs = obs.view(obs.shape[0], -1, 7)

        # | 2. Normalization on GPU
        obs[..., 0] /= self.visible_width
        obs[:, 1:, 1] /= self.visible_height
        obs[..., 2:4] /= self.max_speed

        if ego_pos_reset:
            obs[:, 0, 0] = 0.0
            obs[:, 0, 1] = 0.0

        # | 3. One-hot expansion: (batch, N_entities, expanded_features)
        base_feats = obs[..., :6]
        ids = obs[..., 6].long()
        one_hot = torch.nn.functional.one_hot(ids, num_classes=self.num_classes).float()
        obs = torch.cat([base_feats, one_hot], dim=-1)

        # | 4. Split along entity dimension
        ego_obs    = obs[:, 0, :]                               # (batch, ef)
        tile_obs   = obs[:, 1:1 + self.tile_obs_size, :]       # (batch, tile_obs_size, ef)
        object_obs = obs[:, 1 + self.tile_obs_size:, :]        # (batch, N_objects, ef)

        # | 5. Encode and pool
        # Ego is a single entity — no pooling needed
        ego_encoded    = self.ego_encoder(ego_obs)                        # (batch, ego_hidden)
        # Each tile/object encoded independently, then max-pool over entity dim
        tile_encoded   = self.tile_encoder(tile_obs).max(dim=1).values    # (batch, tile_hidden)
        object_encoded = self.object_encoder(object_obs).max(dim=1).values  # (batch, object_hidden)

        encoded = torch.cat([ego_encoded, tile_encoded, object_encoded], dim=-1)

        if squeeze_output:
            encoded = encoded.squeeze(0)
        elif len(original_shape) > 2:
            encoded = encoded.reshape(*original_shape[:-1], -1)

        return encoded







#############################################################
# Gymnasium registration for PsychDrive environments.       #
# Import this module (or PsychDrive) before calling GymEnv. #
#############################################################

def register_envs() -> None:
    """Register PsychDrive Gymnasium environments."""
    # | Avoid double-registration when imported in multiple places.
    if "CarDrivingEnv-v0" in gym.registry:
        return

    gym.register(
        id="CarDrivingEnv-v0",
        entry_point="drive_env.interface_gym:CarDrivingEnv",
    )