"""
Logic to run the driving task on the browser using Pyodide: dynamically generate the whole map from a seed
"""

# -------------------------------------------------------------------------
# Shared rendering math — importable from native Python (no Pyodide needed)
# -------------------------------------------------------------------------

from drive_env.core.config import TILE_SIZE_M


def world_to_screen(wx, wy, cam_x, cam_y,
                    visible_width_tiles, visible_height_tiles,
                    W, H):
    """Convert a world-space point (metres) to screen-space pixels.

    Args:
        wx, wy: World position in metres.
        cam_x, cam_y: Camera origin in metres (top-left of visible window).
        visible_width_tiles, visible_height_tiles: Window size in tiles.
        W, H: Canvas width and height in pixels.
    """
    sx = (wx - cam_x) * (W / (visible_width_tiles * TILE_SIZE_M))
    sy = (wy - cam_y) * (H / (visible_height_tiles * TILE_SIZE_M))
    return sx, sy


def tile_pixel_size(visible_width_tiles, visible_height_tiles, W, H):
    """Return pixel dimensions of a single tile.

    TILE_SIZE_M cancels: tile_px = W / visible_width_tiles
    """
    return W / visible_width_tiles, H / visible_height_tiles


def object_pixel_size(obj_w, obj_h,
                      visible_width_tiles, visible_height_tiles,
                      W, H):
    """Return pixel dimensions for an object of given world-space (metre) size."""
    sw = obj_w * (W / (visible_width_tiles * TILE_SIZE_M))
    sh = obj_h * (H / (visible_height_tiles * TILE_SIZE_M))
    return sw, sh


# -------------------------------------------------------------------------
# Browser-only code — skipped silently when imported from native Python
# -------------------------------------------------------------------------

try:
    from js import document, Image
    from pyodide.ffi import create_proxy
    _IN_BROWSER = True
except ImportError:
    _IN_BROWSER = False

if _IN_BROWSER:
    import asyncio

    from drive_env.core.config import (FPS,
                                       SECTION_LENGTH, SECTION_COUNT,
                                       LANECOUNT, BOUNDRY_WIDTH)
    from drive_env.core.maps.objects import TILE_TYPES, ASSETS
    from drive_env.core.maps.mapgenerator import MapGenerator
    from drive_env.core.telemetry import Telemetry

    from drive_env.core.mdp import CarMDP
    from drive_env.core.utils import initialize_world, get_obs_window

    # --- Canvas and input ---
    canvas = document.getElementById("canvas")
    ctx = canvas.getContext("2d")
    W = canvas.width
    H = canvas.height

    # Status element for score display
    status_el = document.getElementById("status")

    keys = {"w":False, "s":False, "a":False, "d":False,
            "i":False, "j":False, "k":False, "l":False, }
    def on_keydown(e):
        if e.key in keys: keys[e.key] = True
    def on_keyup(e):
        if e.key in keys: keys[e.key] = False

    document.addEventListener("keydown", create_proxy(on_keydown))
    document.addEventListener("keyup", create_proxy(on_keyup))



    # --- Load sprites as HTMLImageElement ---
    IMG = {}
    images_loaded = 0
    total_images = 0

    def load_image(name, src):
        global images_loaded
        im = Image.new()

        def on_load(e):
            global images_loaded
            images_loaded += 1

        im.addEventListener('load', create_proxy(on_load))
        im.src = src
        IMG[name] = im

    for k, src in ASSETS.items():
        total_images += 1
        load_image(k, src)

    async def wait_for_images():
        # | Wait for images to load before starting game logic
        while images_loaded < total_images:
            await asyncio.sleep(0.1)


    def _draw_road(x, y, ch, cam_x, cam_y,
                   visible_width_tiles, visible_height_tiles):
        if ch in TILE_TYPES:
            img = IMG.get(TILE_TYPES[ch]["name"], IMG["road_asphalt"])
        else:
            img = IMG["road_asphalt"]
        sx, sy = world_to_screen(x + TILE_SIZE_M/2, y + TILE_SIZE_M/2,
                                 cam_x, cam_y, visible_width_tiles,
                                 visible_height_tiles, W, H)
        sw, sh = tile_pixel_size(visible_width_tiles, visible_height_tiles, W, H)
        ctx.drawImage(img, sx - sw/2, sy - sh/2, sw, sh)


    def _draw_object(o, cam_x, cam_y,
                     visible_width_tiles, visible_height_tiles,
                     name_guess=""):
        # name_guess is the object kind (an ASSETS key); unknown kinds fall back to the rock sprite
        img = IMG.get(name_guess, IMG["object_rock"])
        sx, sy = world_to_screen(o.x, o.y, cam_x, cam_y,
                                 visible_width_tiles, visible_height_tiles, W, H)
        sw, sh = object_pixel_size(o.w, o.h,
                                   visible_width_tiles, visible_height_tiles, W, H)
        ctx.drawImage(img, sx - sw/2, sy - sh/2, sw, sh)


    async def countdown(seconds=3):
        """Display countdown timer before starting the game"""
        for i in range(seconds, 0, -1):
            ctx.fillStyle = "#101418"
            ctx.fillRect(0, 0, W, H)
            ctx.fillStyle = "#e5e5e5"
            ctx.font = "72px sans-serif"
            ctx.fillText(str(i), W/2 - 20, H/2)
            await asyncio.sleep(1)

        # Show "GO!" message
        ctx.fillStyle = "#101418"
        ctx.fillRect(0, 0, W, H)
        ctx.fillStyle = "#4CAF50"
        ctx.font = "72px sans-serif"
        ctx.fillText("GO!", W/2 - 50, H/2)
        await asyncio.sleep(0.5)


    async def game_loop(seed, on_complete=None):
        """Play one round on the map built from ``seed``; the same seed also seeds the MDP's
        stochastic choices (slip direction), so a round is fully reproducible."""
        #| Build world & mdp
        active_map = MapGenerator(section_length=SECTION_LENGTH,
                                  section_count=SECTION_COUNT,
                                  lanecount=LANECOUNT,
                                  boundry_width=BOUNDRY_WIDTH,
                                  seed=seed).generate_map(include_goal=True)
        tiles, _, player, mdp, visible_window_size = initialize_world(active_map, seed=seed)

        visible_width_tiles, visible_height_tiles = visible_window_size
        y_max = len(mdp.world.tiles) * TILE_SIZE_M
        x_max = len(mdp.world.tiles[0]) * TILE_SIZE_M

        #| Display countdown before starting
        await countdown(3)

        #| Run game
        running = True
        tel_obj = Telemetry(id="test")
        tel_obj.reset_data(active_map, seed)

        # Show initial score
        status_el.textContent = f"Score: {mdp.score:.2f}"

        while running:
            # print(mdp.timestep)

            ctx.fillStyle = "#101418"
            ctx.fillRect(0, 0, W, H)

            cam_origin_x, cam_origin_y, x0, x1, y0, y1 = get_obs_window(visible_height_tiles,
                                                                        visible_width_tiles,
                                                                        x_max, y_max,
                                                                        player)

            for ty in range(y0, y1):
                for tx in range(x0, x1):
                    if 0 <= ty < len(tiles) and 0 <= tx < len(tiles[ty]):
                        ch = tiles[ty][tx]
                    else:
                        ch = "#"
                    _draw_road(tx*TILE_SIZE_M, ty*TILE_SIZE_M, ch, cam_origin_x, cam_origin_y, *visible_window_size)

            # Handle input -> MDP actions
            action = None
            # Check for simultaneous key combinations first
            if (keys["w"] and keys["d"]) or (keys["i"] and keys["l"]): action = CarMDP.ACCEL_RIGHT
            elif (keys["w"] and keys["a"]) or (keys["i"] and keys["j"]):   action = CarMDP.ACCEL_LEFT
            elif (keys["s"] and keys["d"]) or (keys["k"] and keys["l"]):   action = CarMDP.BRAKE_RIGHT
            elif (keys["s"] and keys["a"]) or (keys["k"] and keys["j"]):   action = CarMDP.BRAKE_LEFT
            # Then check for individual keys
            elif keys["w"] or keys["i"]: action = CarMDP.ACCEL
            elif keys["s"] or keys["k"]: action = CarMDP.BRAKE
            elif keys["d"] or keys["l"]: action = CarMDP.RIGHT
            elif keys["a"] or keys["j"]: action = CarMDP.LEFT
            else: action = CarMDP.IDLE

            # Capture the pre-step observation BEFORE advancing the MDP.
            # This preserves the RL convention: record the state the agent observed
            # when choosing the action, not the state it landed in afterward.
            # We defer the storage decision until after mdp.step() so that we can
            # inspect post-step info to identify event frames.
            pre_step_ts = mdp.timestep
            pre_step_obs = mdp.observationHuman((x0, x1, y0, y1))

            _, reward, done, info = mdp.step(action)

            # An event frame is one where the action had a notable consequence
            # (non-empty info dict — e.g. collision, reward, interaction) or
            # the game ended. Event frames are always stored at full resolution.
            # Non-event frames are stored at 1-in-n to reduce telemetry size
            # while retaining enough temporal resolution for behaviour analysis.
            is_event = bool(info) or done
            if is_event or (mdp.timestep % 1 == 0):
                # Overwrite the snapshot's reward/info with this step's results.
                # observationHuman() reports mdp.step_reward/step_info as they stood
                # *before* the step — i.e. the outcome of the previous action — so the
                # snapshot as captured pairs action a_k with reward/info from k-1.
                # Writing the post-step values makes each entry a self-consistent
                # (state, action, reward, info) tuple and keeps the event label on the
                # frame whose action caused it; otherwise the label survives only when
                # frame k+1 also happens to be stored.
                pre_step_obs['reward'] = reward
                pre_step_obs['info'] = dict(info)
                tel_obj.append_observation(pre_step_obs, action, pre_step_ts)

            # Draw objects
            temp_objects = sorted(mdp.objects, key=lambda o: o.kind, reverse=True) # Sort so cars are rendered last
            for o in temp_objects:
                _draw_object(o, cam_origin_x, cam_origin_y, *visible_window_size, o.kind)

            # Draw player
            # Fake object for drawing using player's sprite
            class _P: pass
            _p = _P(); _p.x=player.x; _p.y=player.y; _p.w=0.9; _p.h=1.4
            _draw_object(_p, cam_origin_x, cam_origin_y, *visible_window_size, "car_player")

            # ========================== Speed Indicator ==========================
            # Speed bar below ego car
            _sx, _sy = world_to_screen(player.x, player.y, cam_origin_x, cam_origin_y, *visible_window_size, W, H)
            _sw, _sh = object_pixel_size(0.9, 1.4, *visible_window_size, W, H)
            bar_w = _sw * 1.8
            bar_h = 5
            bar_x = _sx - bar_w / 2
            bar_y = _sy + _sh / 2 + 4
            speed_ratio = min(max(player.vy, 0.0) / mdp.max_speed, 1.0)

            # Semi-transparent dark trough (blends into road)
            ctx.fillStyle = "rgba(0,0,0,0.25)"
            ctx.fillRect(bar_x, bar_y, bar_w, bar_h)

            # Solid white fill (high contrast)
            if speed_ratio > 0:
                ctx.fillStyle = "rgba(255,255,255,1.0)"
                ctx.fillRect(bar_x, bar_y, bar_w * speed_ratio, bar_h)

            # Draw ten tick marks at 10%, 20%, 30%... 100% for scale reference
            ctx.strokeStyle = "rgba(255,255,255,0.7)"
            ctx.lineWidth = 1
            tick_height = 2
            for i in range(1, 11):  # 10%, 20%... 100%
                tick_x = bar_x + (bar_w * i / 10)
                ctx.beginPath()
                ctx.moveTo(tick_x, bar_y - tick_height)
                ctx.lineTo(tick_x, bar_y + bar_h + tick_height)
                ctx.stroke()

            # Update score display in status element
            status_el.textContent = f"Score: {mdp.score:.2f}"

            # HUD (optional debug info - can be removed if not needed)
            # ctx.fillStyle = "#e5e5e5"
            # ctx.font = "14px sans-serif"
            # ctx.fillText(f"Pos: ({player.x:.1f}, {player.y:.1f})  Vel: ({player.vx:.1f}, {player.vy:.1f})", 12, 24)

            if done:
                print("Game Over")
                running = False

            await asyncio.sleep(1/FPS)  # ~30 FPS

        # Show game over screen with final score
        ctx.fillStyle = "#101418"
        ctx.fillRect(0, 0, W, H)
        ctx.fillStyle = "#4CAF50"
        ctx.font = "48px sans-serif"
        ctx.fillText("Game Complete!", W/2 - 180, H/2 - 50)
        ctx.fillStyle = "#e5e5e5"
        ctx.font = "32px sans-serif"
        ctx.fillText(f"Final Score: {mdp.score:.2f}", W/2 - 140, H/2 + 20)

        # Update status element
        status_el.textContent = f"Game Complete! Final Score: {mdp.score:.2f}"

        # Trigger browser download of telemetry data
        from js import Blob, URL, document as js_doc
        from datetime import datetime

        tel_data = tel_obj.get_curr_data()

        # Create blob and download link
        blob = Blob.new([tel_data], {"type": "application/json"})
        url = URL.createObjectURL(blob)

        # Create temporary download link and click it
        download_link = js_doc.createElement("a")
        download_link.href = url
        filename = f"{tel_obj.id}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}_gzip_base64.txt"
        download_link.download = filename
        download_link.click()

        URL.revokeObjectURL(url)

        ctx.font = "20px sans-serif"
        ctx.fillText(f"Data saved: {filename}", W/2 - 180, H/2 + 80)
        print(f"Game data saved with score: {mdp.score:.2f}")

        # Call completion callback if provided
        if on_complete:
            on_complete(mdp.score)

        return mdp.score
