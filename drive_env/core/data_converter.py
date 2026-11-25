"""
Utility functions for converting between observation formats.
"""

from typing import Dict, List, Any

import numpy as np

from .maps.objects import objectID_dict, TILE_TYPES
from .config import TILE_SIZE_M, FORWARD_RATIO, VISIBILITY_HEIGHT, LANECOUNT, BOUNDRY_WIDTH


# ---------------------------------------------------------------------------
# Tile mesh — mirrors the precomputation in CarMDP.__init__
# ---------------------------------------------------------------------------
_v_h = VISIBILITY_HEIGHT
_v_w = LANECOUNT + 2 * BOUNDRY_WIDTH
_rel_y0 = -int(_v_h * FORWARD_RATIO)
_rel_y1 = int(_v_h * (1 - FORWARD_RATIO)) + 1
_dy = np.arange(_rel_y0, _rel_y1 + 1, dtype=np.int32)
_dx = np.arange(0, _v_w, dtype=np.int32)
_rel_tile_gx, _rel_tile_gy = np.meshgrid(_dx, _dy, indexing="xy")
_gx_flat = _rel_tile_gx.ravel()   # absolute x tile indices (0..v_w-1)
_gy_flat = _rel_tile_gy.ravel()   # relative y tile offsets

_NUM_REL_TILES = _gx_flat.size    # 96
_MAX_OBJECTS   = 40
OBS_SIZE       = 1 + _NUM_REL_TILES + _MAX_OBJECTS   # 137
OBS_FEATURES   = 7

_BOUNDARY_KIND = objectID_dict.get("boundary", 3)


def _char_to_kind_id(ch: str) -> int:
    entry = TILE_TYPES.get(ch)
    if entry is None:
        return _BOUNDARY_KIND
    return objectID_dict.get(entry["name"], _BOUNDARY_KIND)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def human_to_rl_observation(
    human_obs: Dict[str, Any],
    map_base: List[str],
    relative_vel: bool = True,
) -> np.ndarray:
    """Convert observationHuman() output to observationRL() format.

    Args:
        human_obs: dict from observationHuman() {player, objects, reward, ...}
        map_base: full tile map as list of strings (tel['map']['base'])
        relative_vel: subtract ego velocity from all non-ego rows

    Returns:
        (OBS_SIZE, OBS_FEATURES) float32 array matching observationRL layout.
        Row 0: ego [abs_x, norm_y, vx, vy, 0, 0, 0]
        Rows 1..NUM_REL_TILES: tile mesh (ego-centric dx, dy, vx, vy, 0, 0, kind_id)
        Rows NUM_REL_TILES+1..: visible objects (ego-centric)
    """
    buf = np.zeros((OBS_SIZE, OBS_FEATURES), dtype=np.float32)

    player = human_obs['player']
    px, py  = player['x'], player['y']
    pvx, pvy = player['vx'], player['vy']

    map_h = len(map_base)
    map_w = len(map_base[0]) if map_base else _v_w

    # -- Row 0: ego --
    buf[0, :4] = [px, py / (map_h * TILE_SIZE_M), pvx, pvy]

    # -- Rows 1..NUM_REL_TILES: tile mesh --
    py_int = int(py / TILE_SIZE_M)
    gx = _gx_flat
    gy = py_int + _gy_flat

    buf[1:1+_NUM_REL_TILES, 0] = gx * TILE_SIZE_M + TILE_SIZE_M / 2 - px
    buf[1:1+_NUM_REL_TILES, 1] = gy * TILE_SIZE_M + TILE_SIZE_M / 2 - py
    for j in range(_NUM_REL_TILES):
        tx, ty = int(gx[j]), int(gy[j])
        ch = map_base[ty][tx] if 0 <= ty < map_h and 0 <= tx < map_w else '#'
        buf[1+j, 6] = _char_to_kind_id(ch)

    # -- Rows NUM_REL_TILES+1..: objects --
    s_row   = 1 + _NUM_REL_TILES
    objects = human_obs.get('objects', [])
    num_o   = min(len(objects), OBS_SIZE - s_row)
    for k, obj in enumerate(objects[:num_o]):
        buf[s_row+k, 0] = obj['x']  - px
        buf[s_row+k, 1] = obj['y']  - py
        buf[s_row+k, 2] = obj['vx']
        buf[s_row+k, 3] = obj['vy']
        buf[s_row+k, 4] = float(obj['crashable'])
        buf[s_row+k, 5] = float(obj['interactable'])
        buf[s_row+k, 6] = float(objectID_dict.get(obj['name'], 0))

    # -- Relative velocity: subtract ego vel from all non-ego rows --
    if relative_vel:
        e_row = s_row + num_o
        buf[1:e_row, 2] -= pvx
        buf[1:e_row, 3] -= pvy

    return buf
