"""Core Markov Decision Process (MDP) for the 2D PsychDrive environment.

This module is intentionally self-contained and documents all core runtime
concepts used by the simulator:

1. World representation:
    The map is loaded as a list of text rows (tile symbols) and then converted
    into integer lookup tables for fast per-step access.

2. Agent and objects:
    The ego vehicle (player) is represented by :class:`PlayerState`, while
    traffic/obstacles/reward objects are represented by :class:`Obj`.

3. Physics model:
    Motion uses per-second units and integrates using ``dt = 1/FPS``.
    The player and dynamic objects are updated separately.

4. Collision and interaction:
    Player-vs-object contact is resolved with axis-aware overlap checks.
    Objects can either terminate episode (crash) or trigger interaction rewards.

5. Observations:
    Two observation APIs are provided:
    - :meth:`CarMDP.observationRL` for dense, fixed-size numeric tensors.
    - :meth:`CarMDP.observationHuman` for readable dictionaries.

The implementation favors deterministic, vectorized NumPy operations for speed
and predictable behavior across UI and RL training entry points.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import math
import numpy as np
import random

from .config import FPS, MAX_SPEED, CONTEXT_DURATION, CONTEXT_DELAY, TILE_SIZE_M, FORWARD_RATIO, VISIBILITY_HEIGHT, LANECOUNT, BOUNDRY_WIDTH
from .maps.objects import objectID_dict, dynamic_objs_list


def _overlap_check(new_x: float, new_y: float, prev_x: float, prev_y: float,
                   oi_x: float, oi_y: float, oi_vx: float, oi_vy: float,
                   half_w: float, half_h: float, dt: float) -> Tuple[float, float, bool, bool]:
    """Collision overlap detection.

    Args:
        new_x, new_y: Proposed ego position.
        prev_x, prev_y: Previous ego position.
        oi_x, oi_y: Object center position (current).
        oi_vx, oi_vy: Object velocity.
        half_w, half_h: Combined half-extents (object + ego collision radius).
        dt: Time step.
    
    Returns:
        (overlap_x, overlap_y, entered_from_side, entered_from_front_back)
    """
    overlap_x = half_w - abs(new_x - oi_x)
    overlap_y = half_h - abs(new_y - oi_y)
    
    # Calculate object's previous position
    prev_oi_x = oi_x - oi_vx * dt
    prev_oi_y = oi_y + oi_vy * dt
    prev_overlap_x = half_w - abs(prev_x - prev_oi_x)
    prev_overlap_y = half_h - abs(prev_y - prev_oi_y)
    
    entered_from_side = prev_overlap_x <= 0.0 and prev_overlap_y > 0.0
    entered_from_front_back = prev_overlap_y <= 0.0 and prev_overlap_x > 0.0
    
    return overlap_x, overlap_y, entered_from_side, entered_from_front_back


def _speed_control(v_mag: np.ndarray, target_v: np.ndarray,
                   accel_rate: float, decel_rate: np.ndarray,
                   dt: float, v_limit: np.ndarray) -> np.ndarray:
    """Asymmetric speed adjustment for obstacles.

    Args:
        v_mag: Current speed magnitudes (N,).
        target_v: Target speed magnitudes (N,).
        accel_rate: Acceleration rate scalar.
        decel_rate: Per-object deceleration rates (N,).
        dt: Time step.
        v_limit: Speed limits (N,).
    
    Returns:
        new_v_mag: Updated speed magnitudes (N,).
    """
    diff = target_v - v_mag
    dv = np.where(diff > 0,
                  np.minimum(diff, accel_rate * dt),
                  np.maximum(diff, -decel_rate * dt))
    new_v_mag = np.clip(v_mag + dv, 0, v_limit)
    return new_v_mag

@dataclass
class Obj:
    """Runtime object/entity in world coordinates.

    Attributes use map units (tiles/meters depending on map authoring).
    ``vx``/``vy`` are interpreted as per-frame values in some upstream assets,
    then normalized to internal per-second values inside :class:`CarMDP`.

    Attributes:
        kind: Semantic object label (car, deer, coin, etc.).
        x: Center x-position in world coordinates.
        y: Center y-position in world coordinates.
        w: Object width used for AABB collision.
        h: Object height/length used for AABB collision.
        vx: Lateral velocity component.
        vy: Forward velocity magnitude (toward negative map-y by convention).
        max_v: Object speed cap used by dynamic motion logic.
        crash: If true, collision ends the episode.
        reward: Reward added on collision/interaction.
        interactable: If true, collision gives reward without immediate failure.
        sprite: Optional rendering identifier.
    """
    kind: str
    x: float
    y: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    max_v: float = 0.0
    crash: bool = True
    reward: float = 0.0
    miss_penalty: float = 0.0
    interactable: bool = False
    sprite: str = ""

@dataclass
class PlayerState:
    """Mutable state for the ego vehicle controlled by the policy/user.

    Attributes:
        x: Center x-position in world coordinates.
        y: Center y-position in world coordinates.
        vx: Lateral velocity component (map +x is right).
        vy: Forward velocity magnitude (applied toward negative map-y).
    """
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0

@dataclass
class World:
    """Tile map container with fast lookup tables.

    The original map is provided as text rows in ``tiles``. During
    :meth:`__post_init__`, the map is converted into integer arrays so repeated
    queries during physics and observation building avoid string/dict overhead.

    Args:
        tiles: List of map rows where each character is a tile symbol.
        tile_size: Physical width/height of one tile in world units.
        tile_types: Mapping from tile symbol to tile properties.
        boundary_symbol: Symbol treated as out-of-bounds boundary fallback.
    """
    tiles: List[str]
    tile_size: float
    tile_types: Dict[str, Dict[str, float]]
    boundary_symbol: str = "#"

    def __post_init__(self):
        """Build cached lookup tables for tile queries and observations.

        Main outputs created here:
        - ``_sym_to_idx`` / ``_idx_to_sym``: symbol-index conversion tables.
        - ``_props_lut``: direct index->properties lookup.
        - ``_tile_idx_grid``: dense ``(H, W)`` integer map for fast indexing.
        - ``_tile_obs_id_lut``: index->observation object-id lookup.
        """
        # Normalize symbol ordering so indices are deterministic for a map.
        self._tile_symbols = sorted(self.tile_types.keys())
        self._sym_to_idx = {s: i for i, s in enumerate(self._tile_symbols)}
        self._idx_to_sym = np.array(self._tile_symbols, dtype=object)

        # Keep tile property dictionaries in a compact list indexed by tile id.
        self._props_lut = [self.tile_types[s] for s in self._tile_symbols]
        self._default_props = {"name": "boundary", "steer_mult": 0.0, "accel_mult": 0.0, "type": "boundary"}

        # Choose robust defaults for maps that omit expected symbols.
        asphalt_symbol = "A" if "A" in self._sym_to_idx else "."
        self._asphalt_idx = self._sym_to_idx.get(asphalt_symbol, 0)
        self._boundary_idx = self._sym_to_idx.get(self.boundary_symbol, self._asphalt_idx)

        # Pre-resolve tile object ids used by observationRL to avoid per-step dict lookups.
        self._tile_obs_id_lut = np.array(
            [objectID_dict.get(str(self.tile_types[s].get("name", "")), 0) for s in self._tile_symbols],
            dtype=np.int32,
        )

        # Build dense integer map once; queries later become O(1) array indexing.
        self._height = len(self.tiles)
        self._width = max((len(r) for r in self.tiles), default=0)
        self._tile_idx_grid = np.full((self._height, self._width), self._boundary_idx, dtype=np.int16)
        for y, row in enumerate(self.tiles):
            for x, ch in enumerate(row):
                if not ch.strip():
                    idx = self._asphalt_idx
                else:
                    idx = self._sym_to_idx.get(ch, self._boundary_idx)
                self._tile_idx_grid[y, x] = idx

    def tile_idx_at(self, x: float, y: float) -> int:
        """Return integer tile index at world coordinate ``(x, y)``.

        Out-of-map coordinates are intentionally mapped to boundary tile index,
        so callers do not need separate bounds checks before querying tile props.
        """
        tx = int(x / self.tile_size)
        ty = int(y / self.tile_size)
        if ty < 0 or ty >= self._height or tx < 0 or tx >= self._width:
            return self._boundary_idx
        return int(self._tile_idx_grid[ty, tx])

    def tile_props(self, x: float, y: float) -> Dict[str, Any]:
        """Return tile property dictionary at world coordinate ``(x, y)``.

        Always returns a dictionary; when index is invalid, a boundary-like
        default is returned. This keeps physics code branch-light.
        """
        idx = self.tile_idx_at(x, y)
        if 0 <= idx < len(self._props_lut):
            return self._props_lut[idx]
        return self._default_props

class CarMDP:
    """Main simulator class implementing transition dynamics and observations.

    This class stores world/player/object state, applies one-step transitions
    via :meth:`step`, and exposes both machine and human-friendly observations.

    Coordinate convention:
    - Positive x: right.
    - Positive y in map coordinates: downward in stored arrays.
    - Player forward motion reduces y (``new_y = old_y - vy * dt``).
    """
    # Action Constants
    ACCEL, LEFT, RIGHT, BRAKE, IDLE = 0, 1, 2, 3, 4
    ACCEL_LEFT, ACCEL_RIGHT, BRAKE_LEFT, BRAKE_RIGHT = 5, 6, 7, 8

    # State Indices
    OBJ_X, OBJ_Y, OBJ_VX, OBJ_VY, OBJ_MAX_V = 0, 1, 2, 3, 4
    MAX_OBJECTS: int = 40
    _NUM_REL_TILES: int = (
        int(VISIBILITY_HEIGHT * FORWARD_RATIO) + int(VISIBILITY_HEIGHT * (1 - FORWARD_RATIO)) + 2
    ) * (LANECOUNT + 2 * BOUNDRY_WIDTH)
    OBS_SIZE: int = 1 + _NUM_REL_TILES + MAX_OBJECTS
    OBS_FEATURES = 7

    def __init__(self, world: World, 
                 player: PlayerState, 
                 objects: List[Obj], 
                 visible_height: int,
                 seed: int = 283,
                 sync_object_view: bool = True,
                ) -> None:
        """Initialize a simulation episode state and all caches.

        Args:
            world: Tile world descriptor and lookup tables.
            player: Initial ego state.
            objects: Initial list of world objects.
            visible_height: Observation window height in tiles.
            seed: RNG seed for deterministic stochastic choices.
            sync_object_view: If true, keep ``Obj`` list synchronized with
                internal NumPy arrays whenever exposed.
        """
        self.world = world
        self.player = player
        self.player_start_y = player.y
        self._objects = objects
        self.rng = random.Random(seed)
        self.sync_object_view = sync_object_view

        # Simulation parameters (all physical rates are expressed per second).
        self.dt = 1.0 / FPS
        self.accel_unit = 12.0       # units/s^2
        self.brake_unit = -30.0      # units/s^2
        self.steer_unit = 5.0        # units/s (Must be < max_speed to preserve vy)
        self.max_speed = MAX_SPEED   # units/s
        self.min_speed = 15          # units/s (-0.5 if reversing is allowed)
        self.drag_factor = 0.99 ** 30.0
        
        # Collision / behavior constants.
        # p_h_w and p_h_h are half-extents of the ego collision box.
        self.p_h_w, self.p_h_h = 0.47, 0.6
        self.min_v_brake_dist = 20.0
        self.min_v_stop_dist = 5.0

        # Handle laterally moving objects
        self.map_width = self.world._width * self.world.tile_size

        # Rewards & Timing
        self.collision_timer = 0.0
        self.living_penalty_val = -1.0
        self.living_penalty_freq = 2.0 # Seconds

        # Sustained road condition effects
        self.steer_ctrl_timer = 0.0

        # Lane-snap latency: prevents rapid successive lane shifts when button is held.
        # Cooldown is reset to zero whenever the action changes (new key press).
        self.lane_shift_cooldown = 0.0
        self.lane_shift_delay = 0.3   # seconds of lockout after each snap
        self._last_action = self.IDLE

        # Performance management: only process objects near player, at each frame.
        self._active_idx = np.arange(len(self._objects), dtype=np.int32)
        self._active_buffer = 100
        self._active_update_freq = max(1, round(FPS / 2))

        # RL observation precomputations: relative tile mesh around player.
        v_h, v_w = visible_height, world._width
        rel_y0, rel_y1 = -int(v_h * FORWARD_RATIO), int(v_h * (1 - FORWARD_RATIO)) + 1
        dy = np.arange(rel_y0, rel_y1 + 1, dtype=np.int32)
        dx = np.arange(0, v_w, dtype=np.int32)
        self._rel_tile_gx, self._rel_tile_gy = np.meshgrid(dx, dy, indexing="xy")
        self._num_rel_tiles = self._rel_tile_gx.size
        self._obs_buffer = np.zeros((self.OBS_SIZE, self.OBS_FEATURES), dtype=np.float32)
        self._obs_idx    = np.empty(0, dtype=np.int32)   # internal indices of objects in current obs

        # Episode state initialization.
        self.timestep = 0
        self.done = False
        self.score = float(round(len(self.world.tiles) / 2, -2))
        self.goal_reward = 0
        self.step_reward = 0.0
        self.step_info = {}

        self._init_object_state_cache()

    def _init_object_state_cache(self) -> None:
        """Build contiguous NumPy caches from ``self._objects``.

        Internal arrays are the authoritative simulation state; object dataclass
        instances are synchronized on demand for rendering/debug convenience.
        """
        n = len(self._objects)
        # Packed state matrix: [x, y, vx, vy, max_v].
        self.obj_state = np.zeros((n, 5), dtype=np.float32)
        self.obj_w, self.obj_h = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        self.obj_reward = np.zeros(n, dtype=np.float32)
        self.obj_miss_penalty = np.zeros(n, dtype=np.float32)
        self.obj_crash, self.obj_interactable = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        self.obj_kind_id = np.zeros(n, dtype=np.int32)
        self.obj_kind_names = [""] * n
        self.obj_is_dynamic = np.zeros(n, dtype=bool)
        self.obj_dir = np.zeros((n, 2), dtype=np.float32) # [ux, uy] intended direction
        
        # Staggered personalities reduce lock-step traffic behavior.
        # Speed multipliers: 85%-100%; gap multipliers: 80%-120%.
        self.obj_speed_mult = 0.85 + (np.arange(n) % 4) * 0.05
        self.obj_gap_mult = 0.8 + (np.arange(n) % 5) * 0.1

        for i, o in enumerate(self._objects):
            ivx, ivy = o.vx * 30.0, o.vy * 30.0
            self.obj_state[i, :] = [o.x, o.y, ivx, ivy, o.max_v * 30.0]
            
            # Precompute facing direction used when instantaneous speed is near zero.
            # vy is stored as forward speed magnitude; map-y component is therefore -vy.
            v_mag = math.hypot(ivx, ivy)
            if v_mag > 0.01:
                self.obj_dir[i] = [ivx / v_mag, -ivy / v_mag]
            else:
                self.obj_dir[i] = [0.0, -1.0] # Default forward (towards -Y)

            self.obj_w[i], self.obj_h[i] = o.w, o.h
            self.obj_reward[i] = o.reward
            self.obj_miss_penalty[i] = o.miss_penalty
            self.obj_crash[i], self.obj_interactable[i] = o.crash, o.interactable
            self.obj_kind_id[i] = objectID_dict.get(o.kind, 0)
            self.obj_kind_names[i] = o.kind
            self.obj_is_dynamic[i] = o.kind in dynamic_objs_list

        # Sign pinning: signs within CONTEXT_DURATION/2 rows of the player are held
        # at a fixed world-space offset ahead, making them visually static on screen.
        self._sign_pin_half  = CONTEXT_DURATION // 2
        self._sign_pin_trail = CONTEXT_DELAY * TILE_SIZE_M  # metres past sign until context is reached

        # Build mask for all sign objects after the loop, so obj_kind_names is fully populated.
        sign_mask = np.array([n.startswith('sign_') for n in self.obj_kind_names], dtype=bool)
        self._sign_indices = np.flatnonzero(sign_mask)
        self._sign_orig_y = (self.obj_state[self._sign_indices, self.OBJ_Y].copy()
                             if len(self._sign_indices) else np.empty(0, dtype=np.float32))

    def _sync_to_objects(self, indices: Optional[np.ndarray] = None) -> None:
        """Write internal NumPy state back into ``Obj`` dataclass list.

        Args:
            indices: Optional subset of object indices to synchronize.
        """
        target_idxs = indices if indices is not None else np.arange(len(self._objects))
        for i in target_idxs:
            o = self._objects[i]
            o.x, o.y = float(self.obj_state[i, self.OBJ_X]), float(self.obj_state[i, self.OBJ_Y])
            o.vx, o.vy = float(self.obj_state[i, self.OBJ_VX]) / 30.0, float(self.obj_state[i, self.OBJ_VY]) / 30.0

    @property
    def objects(self) -> List[Obj]:
        """Return object list, optionally synced from internal state arrays."""
        if self.sync_object_view:
            self._sync_to_objects()
        return self._objects

    def step(self, action: int) -> Tuple[PlayerState, float, bool, Dict[str, Any]]:
        """Advance simulation by one frame.

        Args:
            action: Discrete player action id.

        Returns:
            Tuple ``(player_state, reward, done, info)`` in Gym-like format.

        Pipeline:
        1. Episode bookkeeping and periodic penalties.
        2. Refresh nearby-object active subset.
        3. Update player and dynamic object physics.
        4. Decay timers and evaluate goal conditions.
        """
        if self.done: return self.player, 0.0, True, {}
        self.timestep += 1
        self.step_reward = 0.0
        self.step_info = {}

        # 1. Living Penalty
        if self.timestep % max(1, round(self.living_penalty_freq / self.dt)) == 0:
            self.step_reward = float(self.step_reward + self.living_penalty_val)

        # 2. Update Active Subset
        if self.timestep % self._active_update_freq == 0 or self.timestep == 1:
            py = self.player.y
            y_states = self.obj_state[:, self.OBJ_Y]
            mask = (y_states >= py - self._active_buffer) & (y_states <= py + self._active_buffer)
            self._active_idx = np.flatnonzero(mask)

        # 3. Physics & Interactions
        self._update_player_physics(action)
        self._update_object_physics()
        self._update_sign_pins()

        # 3a. Missed flag check: flag still present and rear of car has cleared its y
        if self._active_idx.size > 0:
            act = self._active_idx
            miss_mask = self.obj_miss_penalty[act] != 0
            if miss_mask.any():
                flag_idx = act[miss_mask]
                crossed = (self.player.y + self.p_h_h) <= self.obj_state[flag_idx, self.OBJ_Y]
                for idx in sorted(flag_idx[crossed].tolist(), reverse=True):
                    self.step_reward += float(self.obj_miss_penalty[idx])
                    self.step_info["event"] = f"miss:{self.obj_kind_names[idx]}"
                    self._remove_object(idx)
        
        # 4. Global State Updates
        self.collision_timer = max(0.0, self.collision_timer - self.dt)
        
        # 5. Goal Check
        props = self.world.tile_props(self.player.x, self.player.y)
        if "goal" in props.get("name", ""):
            self.done = True
            self.step_reward = float(self.step_reward + self.goal_reward)
            self.step_info["event"] = "goal"

        self.score = float(self.score + self.step_reward)
        return self.player, float(self.step_reward), self.done, self.step_info

    def _update_player_physics(self, action: int) -> None:
        """Apply one-step ego dynamics, world-bound checks, and object contact.

        The player is moved first using tile modifiers and drag, then collisions
        against world boundaries and relevant objects are resolved.
        """
        # Tile-surface modifiers: a_mult scales acceleration/braking; steer_mult != 1 flags a slippery surface.
        props = self.world.tile_props(self.player.x, self.player.y)
        a_mult = props.get("accel_mult", 1.0)
        turn_actions = [self.LEFT, self.ACCEL_LEFT, self.BRAKE_LEFT, self.RIGHT, self.ACCEL_RIGHT, self.BRAKE_RIGHT]
        is_slipping = props.get("steer_mult", 1.0) != 1.0 and (
            action in turn_actions or                                    # slip when steering
            self.player.vy)                                              # or while moving forward
        new_slip = is_slipping and self.steer_ctrl_timer == 0            # True only on the onset frame.

        # Slip-effect state machine: onset starts a timer that marks the slip window.
        # During this window each left/right press snaps in a fresh random direction
        # (loss of traction — player input is ignored).
        if new_slip:
            self.steer_ctrl_timer = self.rng.randint(2, 4) / 2          # Hold effect for 1–2 s.
        elif self.steer_ctrl_timer > 0:
            self.steer_ctrl_timer = max(0.0, self.steer_ctrl_timer - self.dt)

        # Lane-snap decision: determine snap direction (-1 left, +1 right, None = no snap).
        # Cooldown prevents rapid multi-lane shifts while a key is held; it resets on action
        # change so switching left↔right always snaps immediately.
        is_left  = action in [self.LEFT,  self.ACCEL_LEFT,  self.BRAKE_LEFT]
        is_right = action in [self.RIGHT, self.ACCEL_RIGHT, self.BRAKE_RIGHT]
        if action != self._last_action:
            self.lane_shift_cooldown = 0.0                               # New key press: allow an immediate snap.
        self._last_action = action
        self.lane_shift_cooldown = max(0.0, self.lane_shift_cooldown - self.dt)

        snap_dir = None
        if new_slip:
            snap_dir = self.rng.choice([-1, 1])                         # Slip onset: random direction, ignores input.
        elif (is_left or is_right) and self.lane_shift_cooldown <= 0:
            if self.steer_ctrl_timer > 0:
                snap_dir = self.rng.choice([-1, 1])                     # Inside slip window: fresh random direction per press.
            else:
                snap_dir = -1 if is_left else 1                         # Normal: honour player direction.

        # Resolve snap target: centre of the tile (current_tile + snap_dir).
        # Skipped if the candidate tile is not driveable road.
        _snap_x = None
        if snap_dir is not None:
            ts = self.world.tile_size
            candidate_x = (int(self.player.x / ts) + snap_dir + 0.5) * ts  # Lane-centre formula.
            if self.world.tile_props(candidate_x, self.player.y).get("type") == "road":
                _snap_x = candidate_x
                self.lane_shift_cooldown = self.lane_shift_delay         # Lock out further snaps for one interval.

        # Forward dynamics (lateral position is snap-only; no continuous vx).
        # Integrate longitudinal acceleration then apply exponential drag.
        ay = 0.0
        if action in [self.ACCEL, self.ACCEL_LEFT, self.ACCEL_RIGHT]:   ay = self.accel_unit * a_mult
        elif action in [self.BRAKE, self.BRAKE_LEFT, self.BRAKE_RIGHT]: ay = self.brake_unit * a_mult
        self.player.vy = max(self.min_speed, min(self.max_speed,
                             (self.player.vy + ay * self.dt) * self.drag_factor ** self.dt))

        prev_x, prev_y = self.player.x, self.player.y
        new_x = prev_x                                                   # Lateral position changes only via snap.
        new_y = min(prev_y - self.player.vy * self.dt, self.player_start_y)

        # Encode snap as ±tile_size in vx so the observation carries a lateral-change signal.
        # vx = 0 every non-snap frame since there is no continuous lateral motion.
        if _snap_x is not None:
            new_x = _snap_x
            self.player.vx = math.copysign(self.world.tile_size, _snap_x - prev_x)
        else:
            self.player.vx = 0.0

        # A. Horizontal boundary snapping.
        # Query left/right edge probes around candidate x and snap to nearest legal side.
        edge = self.world.tile_size / 2
        is_l_inv = self.world.tile_props(new_x - edge, prev_y).get("type") != "road"
        is_r_inv = self.world.tile_props(new_x + edge, prev_y).get("type") != "road"
        if is_l_inv: new_x = (math.floor((new_x - edge) / self.world.tile_size) + 1) * self.world.tile_size + edge
        elif is_r_inv: new_x = math.floor((new_x + edge) / self.world.tile_size) * self.world.tile_size - edge

        # B. Object interaction and collision response.
        # Only crashable/interactable objects in active subset can affect the player.
        active_rel = self._active_idx[self.obj_crash[self._active_idx] | self.obj_interactable[self._active_idx]]
        if active_rel.size > 0:
            o_x, o_w = self.obj_state[active_rel, self.OBJ_X], self.obj_w[active_rel]
            # Coarse x-corridor gate keeps subsequent checks cheap.
            in_corridor = (np.abs(new_x - o_x) < (o_w / 2 + self.p_h_w))
            
            dy_ahead = self.obj_state[active_rel, self.OBJ_Y] - prev_y
            # Warn about crashable hazards only. Interactable objects (flags) share this
            # corridor test but are goals rather than obstacles, so they must not raise the
            # warning. The filter stays here rather than on active_rel above, which also
            # feeds the resolution loop below and would make flags uncollectable.
            if (in_corridor & self.obj_crash[active_rel] & (dy_ahead < 0) & (dy_ahead > -6.0)).any(): self.step_info["event"] = "collidable_ahead"

            if in_corridor.any():
                candidates = active_rel[in_corridor]
                o_y, o_h = self.obj_state[candidates, self.OBJ_Y], self.obj_h[candidates]
                
                # Resolve first actual overlap encountered among nearby candidates.
                for i, idx in enumerate(candidates):
                    # Fetch per-object AABB parameters.
                    oi_x = float(self.obj_state[idx, self.OBJ_X])
                    oi_y = float(self.obj_state[idx, self.OBJ_Y]) - 0.15 # Slightly offset the collision box for consistancy
                    oi_vx = float(self.obj_state[idx, self.OBJ_VX])
                    oi_vy = float(self.obj_state[idx, self.OBJ_VY])
                    oi_w = float(self.obj_w[idx])
                    oi_h = float(self.obj_h[idx])
                    
                    # Compute overlap depth along each axis using combined half-extents.
                    half_w = oi_w / 2 + self.p_h_w
                    half_h = oi_h / 2 + self.p_h_h
                    
                    overlap_x, overlap_y, entered_from_side, entered_from_front_back = _overlap_check(
                        new_x, new_y, prev_x, prev_y, oi_x, oi_y, oi_vx, oi_vy, half_w, half_h, self.dt
                    )
                    if overlap_x <= 0.0 or overlap_y <= 0.0:
                        continue

                    push_offset = 0.1  # Small constant to push the agent away from the boundary

                    if self.obj_crash[idx]:
                        if entered_from_side or (not entered_from_front_back and overlap_x > overlap_y):
                            # Side hit: keep player on approach side + push offset, then bounce lateral velocity.
                            if prev_x <= oi_x:
                                new_x = oi_x - half_w - push_offset * 2
                            else:
                                new_x = oi_x + half_w + push_offset * 2
                            # Snap to the centre of the tile the push landed in so the
                            # player is never left stranded between lane centres.
                            ts = self.world.tile_size
                            snapped = (int(new_x / ts) + 0.5) * ts
                            if self.world.tile_props(snapped, new_y).get("type") == "road":
                                new_x = snapped
                            self.player.vx = 0.0
                        else:
                            # Front/back hit: keep player on approach side + push offset, then cancel forward velocity.
                            if prev_y >= oi_y:
                                new_y = oi_y + half_h + push_offset
                            else:
                                new_y = oi_y - half_h - push_offset
                            self.player.vy = 0.0

                    self._handle_interaction(idx)
                    break # Resolve first collision found

        # Commit resolved state.
        self.player.x, self.player.y = new_x, new_y

    def _handle_interaction(self, obj_idx: int) -> None:
        """Apply reward and episode effects from a resolved player-object contact.

        Args:
            obj_idx: Absolute index of contacted object.
        """
        if self.obj_crash[obj_idx] and self.collision_timer <= 0:
            self.step_reward += float(self.obj_reward[obj_idx])
            self.collision_timer = 0.4
            self.step_info["event"] = f"crash:{self.obj_kind_names[obj_idx]}"
            self._remove_object(obj_idx)
        elif self.obj_interactable[obj_idx]:
            self.step_reward += float(self.obj_reward[obj_idx])
            self.step_info["event"] = f"interact:{self.obj_kind_names[obj_idx]}"
            self._remove_object(obj_idx)

    def _remove_object(self, obj_idx: int) -> None:
        """Remove a runtime object and rebuild caches so it disappears immediately."""
        if obj_idx < 0 or obj_idx >= len(self._objects):
            return

        # Restore pinned signs to their map-original Y before syncing to Python
        # objects; otherwise _sign_orig_y gets corrupted by the current pin offset
        # and signs appear permanently in the context window after any removal.
        if len(self._sign_indices) > 0:
            self.obj_state[self._sign_indices, self.OBJ_Y] = self._sign_orig_y

        # Preserve current vectorized positions before rebuilding from object instances.
        self._sync_to_objects()
        del self._objects[obj_idx]
        self._init_object_state_cache()

        # Keep active subset aligned after index compaction.
        if len(self._objects) == 0:
            self._active_idx = np.empty(0, dtype=np.int32)
            return

        py = self.player.y
        y_states = self.obj_state[:, self.OBJ_Y]
        mask = (y_states >= py - self._active_buffer) & (y_states <= py + self._active_buffer)
        self._active_idx = np.flatnonzero(mask)

    def _update_sign_pins(self) -> None:
        """Hold contextual signs at a fixed world-space position during their context window.

        Each sign is pinned `_sign_pin_half` rows ahead of the player for the full
        CONTEXT_DURATION window (CONTEXT_DURATION/2 rows before and after its original Y).
        This makes the sign appear static in both the RL observation and the rendered view.
        If two signs would land on the same row, the second is shifted one row further ahead.

        At pin activation the sign is already at its natural world position, so there is
        no visual discontinuity. After the window closes the sign stays at its final Y
        (behind the player) and is naturally out of view.
        """
        if len(self._sign_indices) == 0:
            return
        py = self.player.y
        half = self._sign_pin_half
        orig_y = self._sign_orig_y

        # Asymmetric window: activate CONTEXT_DURATION/2 metres before sign, deactivate
        # only once the player has reached the context object (CONTEXT_DELAY tiles ahead).
        diff = py - orig_y          # positive → player behind sign; negative → player past sign
        in_window = (diff <= half) & (diff >= -self._sign_pin_trail)
        if not in_window.any():
            return

        active_local = np.flatnonzero(in_window)
        # Sort by orig_y descending: the sign with the highest original Y (encountered first,
        # now closest to or behind the player) gets the nearest pin slot.  Using a constant
        # key prevents the two signs from swapping slots as |diff| crosses midpoint.
        active_local = active_local[np.argsort(-orig_y[active_local])]

        base_pin_y = py - 3      # pin row = 3 rows ahead of player
        used_rows: set = set()
        for li in active_local:
            target_y = base_pin_y
            while target_y in used_rows:
                target_y -= 2.0     # push further ahead to resolve overlap
            used_rows.add(target_y)
            self.obj_state[self._sign_indices[li], self.OBJ_Y] = target_y

    def _update_object_physics(self) -> None:
        """Update dynamic objects using vectorized car-following style behavior.

        Static objects are ignored. Dynamic objects detect nearest obstruction in
        their forward corridor and adjust speed with asymmetric accel/decel.
        """
        if self._active_idx.size == 0: return
        active_dyn_mask = self.obj_is_dynamic[self._active_idx]
        if not active_dyn_mask.any(): return
        dyn_idx = self._active_idx[active_dyn_mask]
        
        # 1) Read state slices for active dynamic objects.
        x = self.obj_state[dyn_idx, self.OBJ_X]
        y = self.obj_state[dyn_idx, self.OBJ_Y]
        vx = self.obj_state[dyn_idx, self.OBJ_VX]
        vy = self.obj_state[dyn_idx, self.OBJ_VY]
        v_max = self.obj_state[dyn_idx, self.OBJ_MAX_V] # Assuming MAX_V is the speed limit
        
        # 2) Build candidate obstacle set (player plus relevant objects).
        rel_obs_idx = self._active_idx[self.obj_crash[self._active_idx]]
        obs_x = np.concatenate([self.obj_state[rel_obs_idx, self.OBJ_X], [self.player.x]])
        obs_y = np.concatenate([self.obj_state[rel_obs_idx, self.OBJ_Y], [self.player.y]])
        
        dx = obs_x - x[:, np.newaxis]
        dy = obs_y - y[:, np.newaxis]
        
        # Velocity components in map coordinates.
        # vy stores forward speed magnitude, so map-y component is -vy.
        ux, uy = vx, -vy
        v_mag = np.sqrt(ux**2 + uy**2)
        
        # Use precomputed orientation when speed is near zero to keep heading stable.
        safe_v = np.where(v_mag < 0.01, 1.0, v_mag)
        face_ux = np.where(v_mag < 0.01, self.obj_dir[dyn_idx, 0], ux / safe_v)
        face_uy = np.where(v_mag < 0.01, self.obj_dir[dyn_idx, 1], uy / safe_v)
        
        # Project relative displacement onto longitudinal and lateral axes.
        dist_long = dx * face_ux[:, np.newaxis] + dy * face_uy[:, np.newaxis]
        dist_lat = np.abs(dx * (-face_uy[:, np.newaxis]) + dy * face_ux[:, np.newaxis])
        
        # Individual personality adjustments.
        indiv_min_v = self.min_v_brake_dist * self.obj_gap_mult[dyn_idx]
        v_limit = v_max * self.obj_speed_mult[dyn_idx]

        # "Ahead" = in front and close enough in same movement corridor.
        # Lateral threshold stays narrow to allow side-by-side lane passing.
        is_ahead = (dist_long > 0.1) & (dist_long < indiv_min_v[:, np.newaxis]) & (dist_lat < 0.6)
        
        # Distance to nearest obstruction (large sentinel for "none found").
        min_dist = np.min(np.where(is_ahead, dist_long, 1e6), axis=1)
        
        # 3) Speed planning: reduce target speed when obstruction is closer.
        target_v = v_limit * np.clip((min_dist - self.min_v_stop_dist) / (indiv_min_v - self.min_v_stop_dist), 
                                     0.0, 1.0)
        
        # Asymmetric dynamics: gentle acceleration, stronger proximity braking.
        accel_rate = 1.5  # tiles/s^2
        
        # Brake scale rises as free distance shrinks.
        decel_scale = np.clip((indiv_min_v - self.min_v_stop_dist) / 
                                (np.maximum(min_dist, (self.min_v_stop_dist + 0.1)) - self.min_v_stop_dist), 
                              1.0, 8.0)
        decel_rate = 12.0 * decel_scale
        
        # Apply speed delta.
        new_v_mag = _speed_control(v_mag, target_v, accel_rate, decel_rate, self.dt, v_limit)
        self.obj_state[dyn_idx, self.OBJ_VX] = new_v_mag * face_ux
        self.obj_state[dyn_idx, self.OBJ_VY] = new_v_mag * (-face_uy) # map-y to vy
        
        # 4) Integrate positions using updated velocities.
        self.obj_state[dyn_idx, self.OBJ_X] += self.obj_state[dyn_idx, self.OBJ_VX] * self.dt
        self.obj_state[dyn_idx, self.OBJ_Y] -= self.obj_state[dyn_idx, self.OBJ_VY] * self.dt

        # Loop laterally moving objects back to start position, so they keep crossing road indefinitely
        self.obj_state[dyn_idx, self.OBJ_X] = self.obj_state[dyn_idx, self.OBJ_X] % self.map_width

        # 5) Revert any object that now overlaps the player
        new_x = self.obj_state[dyn_idx, self.OBJ_X]
        new_y = self.obj_state[dyn_idx, self.OBJ_Y]
        half_w = self.obj_w[dyn_idx] / 2 + self.p_h_w
        half_h = self.obj_h[dyn_idx] / 2 + self.p_h_h
        sweep_x = np.abs(self.obj_state[dyn_idx, self.OBJ_VX]) * self.dt
        sweep_y = np.abs(self.obj_state[dyn_idx, self.OBJ_VY]) * self.dt
        overlap = (np.abs(new_x - self.player.x) < half_w + sweep_x) & \
                    (np.abs(new_y - self.player.y) < half_h - sweep_y)
        self.obj_state[dyn_idx[overlap], self.OBJ_X] -= self.obj_state[dyn_idx[overlap], self.OBJ_VX] * self.dt
        self.obj_state[dyn_idx[overlap], self.OBJ_Y] += self.obj_state[dyn_idx[overlap], self.OBJ_VY] * self.dt
        
        if self.sync_object_view: self._sync_to_objects(dyn_idx)

    def observationRL(self, window: Tuple[float,float,float,float]) -> np.ndarray:
        """Build fixed-size numeric observation tensor for learning agents.

        Layout summary per row: ``[x, y, vx, vy, crashable, interactable, kind_id]``

        Row 0 is reserved for ego state; subsequent rows first encode local tiles,
        then visible objects in the query window. Velocities of every non-ego row are
        relative to the ego's.

        Args:
            window: ``(xmin, xmax, ymin, ymax)`` visibility box in tile indices.

        Returns:
            Copy of internal ``(OBS_SIZE, OBS_FEATURES)`` float32 buffer.
        """
        ts = self.world.tile_size
        xmin, xmax, ymin, ymax = (v * ts for v in window)   # get values in meters
        self._obs_buffer.fill(0)
        # Ego state in row 0; y is normalized by total map height in metres → [0, 1] fraction.
        self._obs_buffer[0, :4] = [self.player.x, self.player.y/(len(self.world.tiles) * self.world.tile_size), self.player.vx, self.player.vy]
        
        # Tile rows around ego are sampled from precomputed relative mesh.
        py_int = int(self.player.y / self.world.tile_size)
        gx, gy = self._rel_tile_gx, py_int + self._rel_tile_gy
        in_b = (gx >= 0) & (gx < self.world._width) & (gy >= 0) & (gy < self.world._height)
        tile_idx = np.full(gx.shape, self.world._boundary_idx, dtype=np.int16)
        tile_idx[in_b] = self.world._tile_idx_grid[gy[in_b], gx[in_b]]
        num_t = self._num_rel_tiles
        self._obs_buffer[1:1+num_t, 0] = gx.ravel() * ts + ts / 2
        self._obs_buffer[1:1+num_t, 1] = gy.ravel() * ts + ts / 2
        self._obs_buffer[1:1+num_t, 6] = self.world._tile_obs_id_lut[tile_idx].ravel()
        
        # Append objects visible in current window, capped by fixed observation size.
        active_x, active_y = self.obj_state[self._active_idx, self.OBJ_X], self.obj_state[self._active_idx, self.OBJ_Y]
        mask = (xmin <= active_x) & (active_x <= xmax) & (ymin <= active_y) & (active_y <= ymax)
        obs_idx = self._active_idx[mask]
        num_o = min(len(obs_idx), self.OBS_SIZE - 1 - num_t)
        obs_idx = obs_idx[:num_o]; s_row = 1 + num_t; e_row = s_row + num_o
        self._obs_idx = obs_idx  # slot j → obj_state row obs_idx[j]
        self._obs_buffer[s_row:e_row, :4] = self.obj_state[obs_idx, :4]
        self._obs_buffer[s_row:e_row, 4] = self.obj_crash[obs_idx]
        self._obs_buffer[s_row:e_row, 5] = self.obj_interactable[obs_idx]
        self._obs_buffer[s_row:e_row, 6] = self.obj_kind_id[obs_idx]
        
        # Convert absolute positions/velocities into ego-centric representation.
        px, py = self.player.x, self.player.y
        self._obs_buffer[1:e_row, 0] -= px; self._obs_buffer[1:e_row, 1] -= py
        self._obs_buffer[1:e_row, 2] -= self.player.vx; self._obs_buffer[1:e_row, 3] -= self.player.vy
        return self._obs_buffer.copy()

    def observationHuman(self, window: Tuple[int,int,int,int]) -> Dict[str, Any]:
        """Build human-readable observation dictionary for UI/debugging.

        This method favors clarity over compactness and ensures values are cast to
        native Python numeric types for robust JSON serialization.
        """
        xmin_i, xmax_i, ymin_i, ymax_i = (int(v) for v in window)  # tile indices
        ts = self.world.tile_size
        xmin_m, xmax_m, ymin_m, ymax_m = xmin_i * ts, xmax_i * ts, ymin_i * ts, ymax_i * ts  # convert to meters
        obj_x, obj_y = self.obj_state[:, self.OBJ_X], self.obj_state[:, self.OBJ_Y]
        vis = np.flatnonzero((xmin_m <= obj_x) & (obj_x <= xmax_m) & (ymin_m <= obj_y) & (obj_y <= ymax_m))
        objs = [{'name': self.obj_kind_names[i], 'x': round(float(self.obj_state[i, 0]), 4), 'y': round(float(self.obj_state[i, 1]), 4),
                 'vx': round(float(self.obj_state[i, 2]), 6), 'vy': round(float(self.obj_state[i, 3]), 6),
                 'width': float(self.obj_w[i]), 'length': float(self.obj_h[i]),
                 'interactable': bool(self.obj_interactable[i]), 'crashable': bool(self.obj_crash[i])} for i in vis]

        # Rigorous casting to native types for JSON serialization.
        return {
            'player': {
                'x': float(round(self.player.x, 4)),
                'y': float(round(self.player.y, 4)),
                'vx': float(round(self.player.vx, 6)),
                'vy': float(round(self.player.vy, 6))
            },
            'objects': objs,
            'reward': float(self.step_reward),
            'info': self.step_info,
            'score': float(self.score)
        }

    def get_map_duration(self, speed_ratio: float = 0.3) -> float:
        """Estimate traversal duration for current map at relative speed ratio.

        Args:
            speed_ratio: Fraction of ``max_speed`` expected in practice.

        Returns:
            Estimated completion time in seconds.
        """
        min_duration = self.player.y / (self.max_speed * self.dt)
        return min_duration / speed_ratio
