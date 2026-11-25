"""
Render frames of the car game to RGB arrays, for rollout videos and figures.
Uses Pygame for hardware-accelerated rendering (much faster than PIL).
"""

from pathlib import Path
import numpy as np
import os

import pygame

from drive_env.core.config import TILE_SIZE_M
from drive_env.core.maps.objects import TILE_TYPES, ASSETS
from drive_env.core.utils import get_obs_window
from drive_env.interface_web import (world_to_screen as _wts,
                                      tile_pixel_size as _tps,
                                      object_pixel_size as _ops)

# Initialize pygame in headless mode (no display window)
os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()
pygame.font.init()

# Set a dummy display mode (required for convert_alpha)
_dummy_display = pygame.display.set_mode((1, 1))

# Action name mapping for display
ACTION_NAMES = {
    0: "ACCEL",
    1: "LEFT",
    2: "RIGHT",
    3: "BRAKE",
    4: "IDLE",
    5: "ACCEL_LEFT",
    6: "ACCEL_RIGHT",
    7: "BRAKE_LEFT",
    8: "BRAKE_RIGHT",
}



# ----------------------------- Rendering functions -----------------------------

def _load_img_cache():
    """Load all sprite images as pygame surfaces"""
    cache = {}
    for key, path in ASSETS.items():
        p = Path(path)
        if not p.exists():
            # Fallback: create a 1x1 placeholder surface
            surf = pygame.Surface((1, 1), pygame.SRCALPHA)
            surf.fill((255, 0, 255, 255))
            cache[key] = surf
        else:
            # Load image and ensure it has alpha channel
            cache[key] = pygame.image.load(str(p)).convert_alpha()
    return cache

def _tile_sprite_for(ch, cache):
    """Get the sprite for a tile character using TILE_TYPES"""
    if ch in TILE_TYPES:
        tile_name = TILE_TYPES[ch]["name"]
        return cache.get(tile_name, cache.get("road_asphalt", cache.get("placeholder")))
    # Fallback to road_asphalt for unknown tiles
    return cache.get("road_asphalt", cache.get("placeholder"))

def _world_to_screen(wx, wy, cam_x, cam_y, visible_width_tiles, visible_height_tiles, W, H):
    """Convert world coordinates to screen coordinates. Delegates to drive_env.core.rendering."""
    return _wts(wx, wy, cam_x, cam_y, visible_width_tiles, visible_height_tiles, W, H)



def render_frame(env, W=800, H=600, cache=None, sprite_cache=None, font=None, current_reward=None, action=None, construal_obj_ids=None, dim_alpha=70):
    """
    Renders the current env.mdp state (tiles + objects + player) to an RGB numpy array (H, W, 3).
    Uses Pygame for hardware-accelerated rendering.

    :param current_reward: Optional current frame reward to display in HUD
    :param action: Optional action taken in current frame to display in HUD
    :param construal_obj_ids: Optional set of internal obj indices in the current HLP construal.
        Objects NOT in this set are rendered at ~27% opacity. None = no masking (default).
    """
    if cache is None:
        cache = _load_img_cache()

    if sprite_cache is None:
        sprite_cache = {}

    if font is None:
        font = pygame.font.Font(None, 24)

    mdp = env.mdp
    world = mdp.world
    tiles = world.tiles
    player = mdp.player

    # Visible window dimensions, from the environment
    visible_width_tiles = env.visible_width_tiles
    visible_height_tiles = env.visible_height_tiles

    # Base canvas (pygame surface)
    frame = pygame.Surface((W, H), pygame.SRCALPHA)
    frame.fill((16, 20, 24, 255))  # Background color #101418

    # Tile size in pixels (TILE_SIZE_M cancels in tile_pixel_size)
    _tw, _th = _tps(visible_width_tiles, visible_height_tiles, W, H)
    tile_w, tile_h = int(_tw), int(_th)

    # Get map dimensions
    y_max, x_max = len(tiles), len(tiles[0]) if len(tiles) > 0 else 0

    # Get visible window using the same logic as interface_web.py.
    # get_obs_window expects xmax/ymax in METRES, not tile counts — multiply here.
    cam_origin_x, cam_origin_y, x0, x1, y0, y1 = get_obs_window(
        visible_height_tiles, visible_width_tiles,
        x_max * TILE_SIZE_M, y_max * TILE_SIZE_M, player
    )

    # Draw tiles
    for ty in range(y0, y1):
        for tx in range(x0, x1):
            if 0 <= ty < len(tiles) and 0 <= tx < len(tiles[ty]):
                ch = tiles[ty][tx]
            else:
                ch = "#"
            
            # Cache resized tile sprites (tiles have constant sizes)
            cache_key = (ch, tile_w, tile_h)
            if cache_key not in sprite_cache:
                sprite_cache[cache_key] = pygame.transform.smoothscale(
                    _tile_sprite_for(ch, cache), (tile_w, tile_h)
                )
            sprite = sprite_cache[cache_key]
            
            sx, sy = _world_to_screen(
                tx * TILE_SIZE_M + TILE_SIZE_M / 2,
                ty * TILE_SIZE_M + TILE_SIZE_M / 2,
                cam_origin_x, cam_origin_y,
                visible_width_tiles, visible_height_tiles,
                W, H
            )
            frame.blit(sprite, (int(sx - tile_w / 2), int(sy - tile_h / 2)))

    # Draw objects from the vectorized state, to match collision physics exactly.
    obj_indices = np.arange(mdp.obj_state.shape[0], dtype=np.int32)
    temp_objects = sorted(obj_indices, key=lambda i: mdp.obj_kind_names[int(i)], reverse=True)
    for obj_i in temp_objects:
        kind = mdp.obj_kind_names[int(obj_i)]
        sprite_key = kind if kind in cache else "object_rock"
        spr = cache.get(sprite_key, cache.get("object_rock", cache.get("placeholder")))

        # Scale from world metres to pixels
        _sw, _sh = _ops(float(mdp.obj_w[obj_i]), float(mdp.obj_h[obj_i]),
                        visible_width_tiles, visible_height_tiles, W, H)
        sw, sh = int(_sw), int(_sh)
        if sw <= 0 or sh <= 0:
            continue

        # Cache object sprites by size
        obj_cache_key = (sprite_key, sw, sh)
        if obj_cache_key not in sprite_cache:
            sprite_cache[obj_cache_key] = pygame.transform.smoothscale(spr, (sw, sh))
        spr_resized = sprite_cache[obj_cache_key]

        x = float(mdp.obj_state[obj_i, mdp.OBJ_X])
        y = float(mdp.obj_state[obj_i, mdp.OBJ_Y])
        sx, sy = _world_to_screen(x, y, cam_origin_x, cam_origin_y,
                                  visible_width_tiles, visible_height_tiles, W, H)
        if construal_obj_ids is not None and int(obj_i) not in construal_obj_ids:
            spr_to_blit = spr_resized.copy()
            # convert_alpha() gives per-pixel SRCALPHA; set_alpha() is ignored on such surfaces.
            # BLEND_RGBA_MULT multiplies each pixel's alpha by 70/255 ≈ 27%.
            spr_to_blit.fill((255, 255, 255, dim_alpha), special_flags=pygame.BLEND_RGBA_MULT)
        else:
            spr_to_blit = spr_resized
        frame.blit(spr_to_blit, (int(sx - sw / 2), int(sy - sh / 2)))

    # Draw player (using fake object pattern like in web version)
    spr_player = cache.get("car_player", cache.get("placeholder"))
    _pw, _ph = _ops(0.9, 1.4, visible_width_tiles, visible_height_tiles, W, H)
    pw, ph = int(_pw), int(_ph)
    
    player_cache_key = ("car_player", pw, ph)
    if player_cache_key not in sprite_cache:
        sprite_cache[player_cache_key] = pygame.transform.smoothscale(spr_player, (pw, ph))
    spr_player_resized = sprite_cache[player_cache_key]
    
    psx, psy = _world_to_screen(player.x, player.y, cam_origin_x, cam_origin_y,
                                visible_width_tiles, visible_height_tiles, W, H)
    frame.blit(spr_player_resized, (int(psx - pw / 2), int(psy - ph / 2)))

    # HUD
    hud = f"Pos=({player.x:.1f},{player.y:.1f})  Vel=({player.vx:.1f},{player.vy:.1f})  Score={mdp.score:.2f}"
    text_surface = font.render(hud, True, (229, 229, 229))  # #e5e5e5
    text_rect = text_surface.get_rect()
    text_x = (W - text_rect.width) // 2
    text_y = 12  # Small margin from top
    frame.blit(text_surface, (text_x, text_y))
    
    # Track vertical position for HUD elements
    next_y = text_y + text_rect.height + 8
    
    # Display current frame reward if provided
    if current_reward is not None:
        reward_hud = f"Reward: {current_reward:.4f}"
        reward_surface = font.render(reward_hud, True, (255, 215, 0))  # Gold color
        reward_rect = reward_surface.get_rect()
        reward_x = (W - reward_rect.width) // 2
        frame.blit(reward_surface, (reward_x, next_y))
        next_y += reward_rect.height + 8
    
    # Display action if provided
    if action is not None:
        # Convert action to readable name
        action_value = int(action) if hasattr(action, '__iter__') else int(action)
        action_name = ACTION_NAMES.get(action_value, f"ACTION_{action_value}")
        action_hud = f"Action: {action_name}"
        action_surface = font.render(action_hud, True, (100, 200, 255))  # Light blue color
        action_rect = action_surface.get_rect()
        action_x = (W - action_rect.width) // 2
        frame.blit(action_surface, (action_x, next_y))

    # Convert pygame surface to numpy array (H, W, 3) RGB format
    # pygame uses (W, H, 3) so we need to transpose
    arr = pygame.surfarray.array3d(frame)
    arr = np.transpose(arr, (1, 0, 2))  # Swap width and height axes
    return arr
