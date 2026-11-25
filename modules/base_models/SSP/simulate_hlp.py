"""simulate_hlp.py — Roll out a trained HLP+LLP policy and save a video + JSONL trace.

Loads hlp_model_final.pt + llp_model_final.pt from an HLP run directory, runs the
combined HLP→LLP hierarchy, and renders masked-out cars as translucent.

Typical usage (every theta coefficient at the midpoint of its trained range)::

    uv run python modules/base_models/SSP/simulate_hlp.py --run-dir human_modeling/inference/checkpoint \
        --seed 1 --max-steps 3000 --output-video rollout.mp4 --output-trace rollout_steps.jsonl

Writes:
  - rollout.mp4           (rendered video; masked cars are dim)
  - rollout_steps.jsonl   (per-step events)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm
import pygame
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Editable in-script defaults
# ---------------------------------------------------------------------------
HLP_MODEL_FILE      = "hlp_model_final.pt"
LLP_MODEL_FILE      = "llp_model_final.pt"
HLP_METADATA_FILE   = "hlp_metadata.json"
LLP_METADATA_FILE   = "hrl_llp_metadata.json"
OUTPUT_VIDEO_FILE   = "rollout.mp4"
OUTPUT_TRACE_FILE   = "rollout_steps.jsonl"

# ---------------------------------------------------------------------------
# Imports that need PROJECT_ROOT on sys.path
# ---------------------------------------------------------------------------
from drive_env.video_generator import render_frame, _load_img_cache
from drive_env.core.utils import get_obs_window
from drive_env.core.config import TILE_SIZE_M
from drive_env.core.mdp import CarMDP
from drive_env.core.maps.objects import objectID_dict
from drive_env.interface_gym import CarDrivingEnv, MultiEncoder, register_envs
from drive_env.interface_web import world_to_screen as _wts

from modules.base_models.SP.hrl_llp_core import (
    Agent as LLPAgent,
    GoalConditionedExtractor,
    LLP_LANE_X_CENTERS,
    llp_augment_obs,
    llp_compute_reward_and_flags,
)
from modules.base_models.SSP.hlp_core import (
    OBS_SIZE, OBS_FEATURES, EXPANDED_FEATURES, NUM_CLASSES, MAX_OBJECTS,
    HLPAgent, hlp_extract_car_rows, apply_car_mask, build_hlp_input,
)

# Obs row at which object slots start (rows 0..OBJ_ROW_OFFSET-1 are ego + tiles)
_OBJ_ROW_OFFSET: int = CarMDP.OBS_SIZE - CarMDP.MAX_OBJECTS

# kind_id → human-readable name for HUD display
_ID_TO_KIND: dict[int, str] = {v: k for k, v in objectID_dict.items()}

# Lazily loaded PIL font (load_default(size=N) requires Pillow ≥ 10; fall back gracefully)
def _hud_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=14)
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _goal_screen_pos(env, goal_x_m: float, goal_y_m: float, W: int, H: int) -> tuple[int, int]:
    mdp = env.mdp
    player = mdp.player
    vwt, vht = env.visible_width_tiles, env.visible_height_tiles
    y_max_t = len(mdp.world.tiles)
    x_max_t = len(mdp.world.tiles[0]) if y_max_t > 0 else 0
    cam_x, cam_y, *_ = get_obs_window(vht, vwt, x_max_t * TILE_SIZE_M, y_max_t * TILE_SIZE_M, player)
    sx, sy = _wts(goal_x_m, goal_y_m, cam_x, cam_y, vwt, vht, W, H)
    return int(sx), int(sy)


def _overlay_goal(frame_rgb: np.ndarray, gsx: int, gsy: int, radius: int = 18) -> np.ndarray:
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    draw.ellipse([gsx - radius, gsy - radius, gsx + radius, gsy + radius], outline=(0, 255, 80), width=3)
    draw.line([gsx - radius, gsy, gsx + radius, gsy], fill=(0, 255, 80), width=2)
    draw.line([gsx, gsy - radius, gsx, gsy + radius], fill=(0, 255, 80), width=2)
    return np.array(img)


def _build_construal_obj_ids(
    car_row_indices: torch.Tensor,  # (1, K) long — may be permuted
    car_mask: torch.Tensor,         # (1, K) float
    valid_slots: torch.Tensor,      # (1, K) bool — explicit validity mask (permutation-aware)
    n_objects: int,
    K: int,
    obs_idx: np.ndarray,            # mdp._obs_idx: maps obs position j → obj_state index
) -> set[int]:
    """Return the set of mdp.obj_state indices the LLP can see (mask=1 → included).

    obs_idx (mdp._obs_idx) is required to correctly translate obs row positions to
    obj_state indices — the obs is a filtered/permuted view of obj_state, so a direct
    subtraction of _OBJ_ROW_OFFSET gives the obs position j, not the obj_state index.
    """
    construal = set(range(n_objects))
    for k in range(K):
        if not valid_slots[0, k].item():
            continue
        if car_mask[0, k].item() < 0.5:   # this car is hidden from LLP
            obs_row = int(car_row_indices[0, k].item())
            j = obs_row - _OBJ_ROW_OFFSET   # position in the obs object section
            if 0 <= j < len(obs_idx):
                construal.discard(int(obs_idx[j]))
    return construal


def _overlay_construal_hud(
    frame_rgb: np.ndarray,
    car_mask: torch.Tensor,          # (1, K) float  {0, 1}
    car_row_indices: torch.Tensor,   # (1, K) long — may be permuted
    valid_slots: torch.Tensor,       # (1, K) bool — explicit validity mask (permutation-aware)
    raw_obs_t: torch.Tensor,         # (1, OBS_SIZE*OBS_FEATURES)
    block_step: int,
    sub_episode_horizon: int,
    K: int,
    frame_W: int,                    # frame pixel width — used to right-align the panel
    goal_text: str | None = None,    # [HLP-goal] extra line, e.g. "Goal: lane 2, dy 7.5 m"
) -> np.ndarray:
    """Draw the HLP construal panel in the top-right corner of the frame."""
    LINE_H  = 18
    DOT_R   = 5    # dot radius
    PAD     = 7
    PANEL_W = 215
    # height: title + bar row + block-text + K slot rows + summary (+ goal line)
    PANEL_H = PAD + LINE_H + 4 + 10 + 4 + LINE_H + PAD + K * LINE_H + PAD + LINE_H + PAD
    if goal_text is not None:
        PANEL_H += LINE_H

    x0 = frame_W - PANEL_W - 10
    y0 = 10

    img  = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    font = _hud_font()

    # Dark background panel
    draw.rectangle([x0, y0, x0 + PANEL_W, y0 + PANEL_H], fill=(18, 18, 18))
    draw.rectangle([x0, y0, x0 + PANEL_W, y0 + PANEL_H], outline=(60, 60, 60), width=1)

    cy = y0 + PAD

    # ── Title ────────────────────────────────────────────────────────────────
    draw.text((x0 + PAD, cy), "HLP Construal", fill=(210, 210, 210), font=font)
    cy += LINE_H + 4

    # ── Block progress bar ───────────────────────────────────────────────────
    frac   = min(block_step / max(sub_episode_horizon, 1), 1.0)
    bar_x0 = x0 + PAD
    bar_x1 = x0 + PANEL_W - PAD
    bar_w  = bar_x1 - bar_x0
    draw.rectangle([bar_x0, cy, bar_x1, cy + 9], fill=(40, 40, 40), outline=(70, 70, 70), width=1)
    if frac > 0:
        draw.rectangle([bar_x0, cy, bar_x0 + max(2, int(bar_w * frac)), cy + 9], fill=(70, 130, 210))
    cy += 14
    draw.text((x0 + PAD, cy), f"Block: {block_step} / {sub_episode_horizon}", fill=(150, 150, 150), font=font)
    cy += LINE_H + PAD

    # ── Car slots ────────────────────────────────────────────────────────────
    obs_2d  = raw_obs_t[0].view(OBS_SIZE, OBS_FEATURES)   # (OBS_SIZE, 7)
    valid_k = int(valid_slots[0].sum().item())   # total valid cars (for summary count)
    n_shown = 0

    for k in range(K):
        dot_cx = x0 + PAD + DOT_R
        dot_cy = cy + LINE_H // 2

        if not valid_slots[0, k].item():
            # Invalid / no car in this slot
            draw.ellipse(
                [dot_cx - DOT_R, dot_cy - DOT_R, dot_cx + DOT_R, dot_cy + DOT_R],
                fill=(55, 55, 55),
            )
            draw.text((x0 + PAD + DOT_R * 2 + 4, cy), f"Slot {k}  (empty)", fill=(80, 80, 80), font=font)
        else:
            shown    = car_mask[0, k].item() > 0.5
            obs_row  = int(car_row_indices[0, k].item())
            kind_id  = int(obs_2d[obs_row, 6].item())
            # Strip "car_" prefix; fall back to raw id
            kind_name = _ID_TO_KIND.get(kind_id, f"id:{kind_id}").replace("car_", "")
            color     = (75, 210, 95) if shown else (215, 85, 65)
            tag       = "SHOW" if shown else "HIDE"
            if shown:
                n_shown += 1
            draw.ellipse(
                [dot_cx - DOT_R, dot_cy - DOT_R, dot_cx + DOT_R, dot_cy + DOT_R],
                fill=color,
            )
            draw.text(
                (x0 + PAD + DOT_R * 2 + 4, cy),
                f"Slot {k}  {kind_name}  [{tag}]",
                fill=color,
                font=font,
            )
        cy += LINE_H

    # ── Summary ──────────────────────────────────────────────────────────────
    cy += PAD - 4
    draw.line([x0 + PAD, cy, x0 + PANEL_W - PAD, cy], fill=(55, 55, 55), width=1)
    cy += 4
    draw.text(
        (x0 + PAD, cy),
        f"Visible: {n_shown} / {valid_k}  cars shown",
        fill=(170, 170, 170),
        font=font,
    )

    # [HLP-goal] chosen goal line
    if goal_text is not None:
        cy += LINE_H
        draw.text((x0 + PAD, cy), goal_text, fill=(120, 220, 140), font=font)

    return np.array(img)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Roll out a trained HLP+LLP policy and render a video with masked cars dimmed."
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="HLP run directory (contains hlp_model_final.pt, hlp_metadata.json, "
                             "llp_model_final.pt, hrl_llp_metadata.json).")
    parser.add_argument("--output-video", type=str, default=None)
    parser.add_argument("--output-trace", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--mask-alpha", type=int, default=70,
        help="Opacity (0–255) of masked-out cars in the rendered video. "
             "0 = invisible, 255 = fully opaque, default = 70 (~27%%).",
    )
    return parser


def _resolve_paths(cli: argparse.Namespace):
    run_dir = Path(cli.run_dir).resolve()

    hlp_model = run_dir / HLP_MODEL_FILE
    llp_model = run_dir / LLP_MODEL_FILE
    hlp_meta_path  = run_dir / HLP_METADATA_FILE
    llp_meta_path  = run_dir / LLP_METADATA_FILE
    output_video   = Path(cli.output_video).resolve() if cli.output_video else (run_dir / OUTPUT_VIDEO_FILE)
    output_trace   = Path(cli.output_trace).resolve() if cli.output_trace else (run_dir / OUTPUT_TRACE_FILE)

    for p, label in [(hlp_model, "HLP weights"), (hlp_meta_path, "hlp_metadata.json"),
                     (llp_model, "LLP weights"), (llp_meta_path, "hrl_llp_metadata.json")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    return run_dir, hlp_model, llp_model, hlp_meta_path, llp_meta_path, output_video, output_trace


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    cli = parser.parse_args()

    run_dir, hlp_model_path, llp_model_path, hlp_meta_path, llp_meta_path, \
        output_video, output_trace = _resolve_paths(cli)

    hlp_meta = json.loads(hlp_meta_path.read_text(encoding="utf-8"))
    llp_meta = json.loads(llp_meta_path.read_text(encoding="utf-8"))
    # | NOTE: this must stay ABOVE the miss-penalty block below, which builds a tensor on it.
    # | It used to be assigned after that block, so any checkpoint with a real sampled
    # | miss-penalty range raised UnboundLocalError before rendering a single frame.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    register_envs()
    env = CarDrivingEnv(training_mode=False)
    y_max = float(env.y_max)

    # ── Build LLP ──────────────────────────────────────────────────────────────
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
    ).to(device)

    class _SpaceProxy:
        def __init__(self):
            self.single_observation_space = env.observation_space
            self.single_action_space      = env.action_space

    llp_agent = LLPAgent(
        _SpaceProxy(),
        hidden_size=int(llp_meta.get("hidden_size", 256)),
        feature_extractor=goal_extractor,
    ).to(device)
    llp_ckpt = torch.load(llp_model_path, map_location=device, weights_only=False)
    llp_sd = llp_ckpt.get("model_state_dict", llp_ckpt) if isinstance(llp_ckpt, dict) else llp_ckpt
    llp_agent.load_state_dict(llp_sd)
    llp_agent.eval()

    # ── Build HLP ──────────────────────────────────────────────────────────────
    K               = int(hlp_meta["hlp_num_car_slots"])
    hlp_hidden_size = int(hlp_meta["hlp_hidden_size"])
    hlp_input_dim   = int(hlp_meta["hlp_input_dim"])
    sub_episode_horizon = int(hlp_meta["sub_episode_horizon"])
    hlp_critic_hidden_size = hlp_hidden_size * int(hlp_meta.get("hlp_critic_hidden_size_multiplier", 1))

    # [HLP-goal] goal-selection config
    num_goal_dy_bins = int(hlp_meta["num_goal_dy_bins"])
    goal_dy_bin_width_eff = float(hlp_meta.get("goal_dy_bin_width_eff", 0.5))

    # [HLP-param] theta for this rollout: the midpoint of every trained range, in
    # hlp_param_config order (metadata = source of truth); the ranges normalise it in the input.
    param_config = hlp_meta["hlp_param_config"]
    _param_vals = [0.5 * (p["min"] + p["max"]) for p in param_config]
    hlp_params_t = torch.tensor([_param_vals], dtype=torch.float32, device=device)  # (1, P)
    hlp_params_used = {p["name"]: v for p, v in zip(param_config, _param_vals)}
    param_ranges_t = torch.tensor([[p["min"], p["max"]] for p in param_config], dtype=torch.float32, device=device)
    # ASCII-only: Windows consoles may use cp1252 and crash on non-ASCII prints
    print(f"HLP params for this rollout: {hlp_params_used}")

    hlp_agent = HLPAgent(
        input_dim=hlp_input_dim, K=K, hidden_size=hlp_hidden_size,
        critic_hidden_size=hlp_critic_hidden_size,
        num_lanes=LLP_LANE_X_CENTERS.shape[0],
        num_dy_bins=num_goal_dy_bins,
        goal_dy_min=float(hlp_meta["goal_forward_m_min"]),
        goal_dy_bin_width=goal_dy_bin_width_eff,
        lane_centers=LLP_LANE_X_CENTERS,
    ).to(device)
    hlp_sd = torch.load(hlp_model_path, map_location=device, weights_only=False)
    # [HLP-param] one-line era guard — readable failure instead of a cryptic size mismatch.
    _key = "pooled_actor.ctx_enc.0.weight"
    assert _key in hlp_sd, f"Checkpoint has no {_key!r}: not a pooled-head checkpoint"
    _w_in, _expect = hlp_sd[_key].shape[1], hlp_input_dim - K * EXPANDED_FEATURES + 2
    assert _w_in == _expect, (
        f"Checkpoint mask-head input dim {_w_in} != current architecture {_expect} "
        f"— incompatible checkpoint era (see hlp_metadata.json)."
    )
    hlp_agent.load_state_dict(hlp_sd)
    hlp_agent.eval()

    # ── Args namespace for build_hlp_input ─────────────────────────────────────
    args_ns = SimpleNamespace(
        visible_width=float(hlp_meta["visible_width"]),
        visible_height=float(hlp_meta["visible_height"]),
        max_speed=float(hlp_meta["max_speed"]),
        y_max=y_max,
        goal_forward_m_min=float(hlp_meta["goal_forward_m_min"]),
        goal_forward_m_max=float(hlp_meta["goal_forward_m_max"]),
        goal_radius_m=float(hlp_meta["goal_radius_m"]),
        llp_reach_reward=float(hlp_meta.get("llp_reach_reward", 0.0)),
        llp_miss_penalty=float(hlp_meta.get("llp_miss_penalty", 0.0)),
        llp_truncation_penalty=float(hlp_meta.get("llp_truncation_penalty", 0.0)),
        llp_miss_ends_subepisode=bool(hlp_meta.get("llp_miss_ends_subepisode", False)),
        sub_episode_horizon=sub_episode_horizon,
        llp_env_reward_coef=float(hlp_meta.get("llp_env_reward_coef", 0.1)),
    )

    # ── Rendering setup ─────────────────────────────────────────────────────────
    cache        = _load_img_cache()
    sprite_cache: dict = {}
    font         = pygame.font.Font(None, 24)
    W, H         = cli.width, cli.height
    frame_skip   = max(1, int(cli.frame_skip))
    mask_alpha   = max(0, min(255, int(cli.mask_alpha)))

    # [HLP-goal] One full HLP decision: goal-free base input → goal (self-chosen, or
    # random in legacy mode) → goal-conditioned mask. Returns HUD/trace metadata too.
    def _hlp_decide(raw_obs_t, car_row_indices, valid_slots):
        """Run one HLP decision. Returns (car_mask, goal_x, goal_y, goal_lane, goal_dy_m)."""
        x_t = raw_obs_t[:, 0]
        y_t = raw_obs_t[:, 1] * y_max
        # [HLP-param] hlp_params_t: constant (1, P) θ for the whole rollout
        hlp_in, _, _ = build_hlp_input(raw_obs_t, K, args_ns, device, hlp_params_t,
                                        car_row_indices=car_row_indices, valid_slots=valid_slots,
                                        param_ranges=param_ranges_t,
)
        with torch.no_grad():
            # [HLP-sim] The MASK is decided deterministically (deterministic_mask=True). Why it
            # matters: the mask head is K independent Bernoullis,
            # and the trained per-car show-probabilities are nearly uniform (~0.3, threat-AUC ≈ 0.5).
            # Sampling them would yield a DIFFERENT RANDOM subset of cars every block, which
            # looks like deliberate, varied construal but is really just noise around a flat prob —
            # actively misleading for a visualisation of what the policy DECIDES. The greedy
            # threshold (show iff P>0.5) gives the policy's committed, reproducible construal per
            # car — the honest picture (and identical for a fixed seed).
            actions, _, _, _, goal_info = hlp_agent.get_action_and_value(
                hlp_in, x_m=x_t, y_m=y_t, deterministic=True, deterministic_mask=True,
            )
            goal_x, goal_y = goal_info["goal_x"], goal_info["goal_y"]
            goal_lane = int(actions["lane"][0].item())
            goal_dy_m = float(goal_info["dy"][0].item())
        return actions["mask"], goal_x, goal_y, goal_lane, goal_dy_m

    # ── Initial env state ───────────────────────────────────────────────────────
    torch.manual_seed(cli.seed)
    base_obs, _ = env.reset(seed=cli.seed)
    n_objects   = env.mdp.obj_state.shape[0]   # number of MDP object slots

    raw_obs_t = torch.tensor(base_obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, flat)

    subepisode_steps = torch.zeros(1, dtype=torch.long, device=device)
    prev_action      = torch.full((1,), 4, dtype=torch.long, device=device)  # IDLE
    env_reward_coef  = torch.full((1,), args_ns.llp_env_reward_coef, dtype=torch.float32, device=device)
    block_step       = 0


    # Initial HLP decision (goal + mask)
    car_row_indices, num_cars = hlp_extract_car_rows(raw_obs_t, K)
    perm = torch.randperm(K, device=device).unsqueeze(0)                          # (1, K)
    car_row_indices = car_row_indices.gather(1, perm)
    valid_slots = (torch.arange(K, device=device).unsqueeze(0) < num_cars.unsqueeze(1)).gather(1, perm)
    car_mask, goal_x, goal_y, goal_lane, goal_dy_m = _hlp_decide(raw_obs_t, car_row_indices, valid_slots)
    construal_obj_ids = _build_construal_obj_ids(
        car_row_indices, car_mask, valid_slots, n_objects, K, env.mdp._obs_idx
    )
    kind_id_snap = raw_obs_t[0].view(OBS_SIZE, OBS_FEATURES)[:, 6].clone()

    # [HLP-exp] Identity map: obj_state_idx → mask_bit for this sub-episode.
    def _build_obj_mask_map(cr, vs, mask, obs_idx_arr):
        """Build {obj_state_idx: mask_bit} from a given HLP decision.

        mask is passed explicitly (not captured by closure) to avoid stale references
        when car_mask is reassigned later in the rollout loop.
        """
        m = {}
        for i in range(K):
            if vs[0, i].item():
                j = int(cr[0, i].item()) - _OBJ_ROW_OFFSET
                if 0 <= j < len(obs_idx_arr):
                    oi = int(obs_idx_arr[j])
                    if oi >= 0:
                        m[oi] = mask[0, i].item()
        return m

    obj_mask_map = _build_obj_mask_map(car_row_indices, valid_slots, car_mask, env.mdp._obs_idx)
    block_perm   = perm.clone()   # [HLP-exp] preserved for obs-change remap within this sub-episode

    # ── Rollout ─────────────────────────────────────────────────────────────────
    total_env_reward = 0.0
    step_count       = 0
    block_count      = 0
    goal_count       = 0
    done = truncated = False
    pbar = tqdm(total=cli.max_steps, desc="Simulating", unit="step", dynamic_ncols=True)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_trace.parent.mkdir(parents=True, exist_ok=True)

    with output_trace.open("w", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps({
            "event":               "rollout_start",
            "run_dir":             str(run_dir),
            "hlp_model_path":      str(hlp_model_path),
            "seed":                int(cli.seed),
            "sub_episode_horizon": sub_episode_horizon,
            "K":                   K,
            "mask_alpha":          mask_alpha,
            "hlp_params":          hlp_params_used,    # [HLP-param] θ for this rollout
        }) + "\n")

        with iio.imopen(str(output_video), "w", plugin="pyav") as vid_writer:
            vid_writer.init_video_stream("h264", fps=cli.fps)

            initial = render_frame(
                env, W=W, H=H, cache=cache, sprite_cache=sprite_cache, font=font,
                construal_obj_ids=construal_obj_ids, dim_alpha=mask_alpha,
            )
            try:
                gsx, gsy = _goal_screen_pos(env, float(goal_x[0]), float(goal_y[0]), W, H)
                if 0 <= gsx < W and 0 <= gsy < H:
                    initial = _overlay_goal(initial, gsx, gsy)
            except Exception:
                pass
            # [HLP-goal] HUD goal line — evaluated at call time (goal changes per block)
            _goal_text = lambda: (
                f"Goal: lane {goal_lane}, dy {goal_dy_m:.1f} m" if goal_lane is not None else None
            )
            initial = _overlay_construal_hud(
                initial, car_mask, car_row_indices, valid_slots, raw_obs_t,
                block_step, sub_episode_horizon, K, W, goal_text=_goal_text(),
            )
            vid_writer.write_frame(initial)

            while not (done or truncated) and step_count < cli.max_steps:
                # ── LLP step ─────────────────────────────────────────────────
                masked_obs  = apply_car_mask(
                    raw_obs_t.view(1, OBS_SIZE, OBS_FEATURES),
                    car_row_indices, car_mask, valid_slots,
                )
                llp_obs = llp_augment_obs(
                    masked_obs.view(1, -1), goal_x, goal_y,
                    subepisode_steps, prev_action, env_reward_coef,
                    y_max, sub_episode_horizon,
                )
                with torch.no_grad():
                    enc = llp_agent.feature_extractor(llp_obs)
                    llp_action_t = torch.argmax(llp_agent.actor(enc), dim=-1)
                action = int(llp_action_t.item())
                next_obs, env_reward, done, truncated, info = env.step(action)

                subepisode_steps += 1
                step_count       += 1
                block_step       += 1
                total_env_reward += float(env_reward)
                prev_action[0]   = action
                pbar.update(1)

                raw_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                x_m = float(raw_obs_t[0, 0].item())
                y_m = float(raw_obs_t[0, 1].item()) * y_max

                # Kind-id change detection. When car IDs shift, the mask is remapped via
                # obj_state identity (no new HLP forward pass). Goal, subepisode_steps, and
                # block_step all continue unchanged.
                new_kind_ids = raw_obs_t[0].view(OBS_SIZE, OBS_FEATURES)[:, 6]
                kind_id_changed = not torch.equal(new_kind_ids, kind_id_snap)

                # LLP termination check (goal reached, miss, horizon, env done).
                _llp_r, sub_done_t, goal_reached_t = llp_compute_reward_and_flags(
                    torch.tensor([x_m], device=device),
                    torch.tensor([y_m], device=device),
                    goal_x, goal_y,
                    subepisode_steps,
                    torch.tensor([done or truncated], dtype=torch.bool, device=device),
                    args_ns.goal_radius_m, args_ns.llp_reach_reward, args_ns.llp_miss_penalty,
                    args_ns.llp_truncation_penalty, sub_episode_horizon, args_ns.llp_miss_ends_subepisode,
                )
                llp_sub_done = bool(sub_done_t[0].item())
                goal_reached = bool(goal_reached_t[0].item())
                goal_missed  = (y_m < float(goal_y[0].item())) and not goal_reached
                goal_count  += int(goal_reached)

                # sub_done kept for backwards compat; separate fields added for clarity.
                sub_done = llp_sub_done or kind_id_changed

                _obs2d   = raw_obs_t[0].view(OBS_SIZE, OBS_FEATURES)
                _n_valid = int(num_cars[0].item())
                _rows    = car_row_indices[0].tolist()
                _kinds   = [
                    _ID_TO_KIND.get(int(_obs2d[r, 6].item()), f"id:{int(_obs2d[r, 6].item())}").replace("car_", "")
                    if valid_slots[0, k].item() else None
                    for k, r in enumerate(_rows)
                ]
                trace_file.write(json.dumps({
                    "event":            "step",
                    "step":             step_count,
                    "action":           action,             # chosen this step
                    "env_reward":       float(env_reward),
                    "total_env_reward": total_env_reward,
                    "done":             bool(done),
                    "truncated":        bool(truncated),
                    "sub_done":         sub_done,
                    "llp_sub_done":     llp_sub_done,
                    "kind_id_changed":  kind_id_changed,
                    "goal_reached":     goal_reached,
                    "goal_missed":      goal_missed,
                    "player_x_m":       x_m,
                    "player_y_m":       y_m,
                    "goal_x_m":         float(goal_x[0].item()),
                    "goal_y_m":         float(goal_y[0].item()),
                    "car_mask":         car_mask[0].tolist(),
                    "num_cars":         _n_valid,
                    "car_row_indices":  _rows,
                    "car_kinds":        _kinds,
                    "block_step":       block_step,
                    "goal_lane":        goal_lane,    # [HLP-goal] None in legacy mode
                    "goal_dy_m":        goal_dy_m,
                }) + "\n")

                # ── Kind-id change: identity-based mask remap, no new HLP decision ────────────
                # [HLP-exp] Refresh car rows using the block perm (same perm throughout the
                # sub-episode, matching hlp_collect_rollout). Remap car_mask via obj_mask_map.
                # goal_x/goal_y, subepisode_steps, and block_step are all unchanged.
                # New cars (not in obj_mask_map) are hidden by default (mask=0).
                if kind_id_changed and not llp_sub_done and not (done or truncated):
                    kind_id_snap = raw_obs_t[0].view(OBS_SIZE, OBS_FEATURES)[:, 6].clone()
                    car_row_indices_base, num_cars = hlp_extract_car_rows(raw_obs_t, K)
                    car_row_indices = car_row_indices_base.gather(1, block_perm)   # reuse block perm
                    valid_slots = (torch.arange(K, device=device).unsqueeze(0) < num_cars.unsqueeze(1)).gather(1, block_perm)
                    obs_idx_now = env.mdp._obs_idx
                    new_mask = torch.zeros(1, K, device=device)
                    for _i in range(K):
                        if valid_slots[0, _i].item():
                            _j = int(car_row_indices[0, _i].item()) - _OBJ_ROW_OFFSET
                            _oi = int(obs_idx_now[_j]) if 0 <= _j < len(obs_idx_now) else -1
                            new_mask[0, _i] = float(obj_mask_map.get(_oi, 0.0))
                    car_mask = new_mask
                    construal_obj_ids = _build_construal_obj_ids(
                        car_row_indices, car_mask, valid_slots, n_objects, K, env.mdp._obs_idx
                    )

                # ── Sub-episode completion: new goal + new HLP decision ───────────────────────
                # Fires on LLP goal/miss/horizon. Fresh HLP forward pass; rebuilds obj_mask_map.
                if llp_sub_done and not (done or truncated):
                    kind_id_snap = raw_obs_t[0].view(OBS_SIZE, OBS_FEATURES)[:, 6].clone()
                    subepisode_steps[0] = 0

                    block_step   = 0
                    block_count += 1
                    car_row_indices_base, num_cars = hlp_extract_car_rows(raw_obs_t, K)
                    perm = torch.randperm(K, device=device).unsqueeze(0)
                    car_row_indices = car_row_indices_base.gather(1, perm)
                    valid_slots = (torch.arange(K, device=device).unsqueeze(0) < num_cars.unsqueeze(1)).gather(1, perm)
                    # [HLP-goal] fresh HLP decision — goal + mask in one joint action
                    car_mask, goal_x, goal_y, goal_lane, goal_dy_m = _hlp_decide(
                        raw_obs_t, car_row_indices, valid_slots
                    )
                    construal_obj_ids = _build_construal_obj_ids(
                        car_row_indices, car_mask, valid_slots, n_objects, K, env.mdp._obs_idx
                    )
                    # [HLP-exp] Rebuild identity map and lock in block perm for new sub-episode
                    obj_mask_map = _build_obj_mask_map(car_row_indices, valid_slots, car_mask, env.mdp._obs_idx)
                    block_perm   = perm.clone()

                if done or truncated:
                    prev_action[0] = 4   # IDLE

                pbar.set_postfix(
                    block=block_count,
                    goals=goal_count,
                    reward=f"{total_env_reward:.2f}",
                    refresh=False,
                )

                # ── Render ───────────────────────────────────────────────────
                if step_count % frame_skip == 0:
                    frame = render_frame(
                        env, W=W, H=H, cache=cache, sprite_cache=sprite_cache, font=font,
                        current_reward=float(env_reward), action=action,
                        construal_obj_ids=construal_obj_ids, dim_alpha=mask_alpha,
                    )
                    try:
                        gsx, gsy = _goal_screen_pos(env, float(goal_x[0]), float(goal_y[0]), W, H)
                        if 0 <= gsx < W and 0 <= gsy < H:
                            frame = _overlay_goal(frame, gsx, gsy)
                    except Exception:
                        pass
                    frame = _overlay_construal_hud(
                        frame, car_mask, car_row_indices, valid_slots, raw_obs_t,
                        block_step, sub_episode_horizon, K, W, goal_text=_goal_text(),
                    )
                    vid_writer.write_frame(frame)

        pbar.close()
        trace_file.write(json.dumps({
            "event":             "rollout_end",
            "episode_length":    step_count,
            "total_env_reward":  total_env_reward,
        }) + "\n")

    env.close()
    print(f"Run directory: {run_dir}")
    print(f"HLP model:     {hlp_model_path}")
    print(f"LLP model:     {llp_model_path}")
    print(f"Video:         {output_video}")
    print(f"Trace:         {output_trace}")
    print(f"Steps:         {step_count}")
    print(f"Env reward:    {total_env_reward:.4f}")
    print(f"mask_alpha:    {mask_alpha}/255")


if __name__ == "__main__":
    main()
