"""hlp_core.py — HLP agent, masking utilities, and rollout collection.

[HLP-delta] New module — no equivalent in PPO_cleanRL.py.
All public symbols are imported by train_hlp_PPO.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drive_env.core.mdp import CarMDP
from drive_env.core.maps.objects import objectID_dict
from drive_env.interface_gym import preprocess_obs
from modules.base_models.SP.hrl_llp_core import (
    llp_augment_obs,
    llp_compute_reward_and_flags,
    llp_extract_player_xy_m,
    llp_sample_goals,
)

# ---------------------------------------------------------------------------
# Module-level constants (derived — never hardcoded)
# ---------------------------------------------------------------------------

OBS_SIZE: int = CarMDP.OBS_SIZE
OBS_FEATURES: int = CarMDP.OBS_FEATURES
MAX_OBJECTS: int = CarMDP.MAX_OBJECTS
_OBJ_ROW_OFFSET: int = OBS_SIZE - MAX_OBJECTS  # obs row where object section starts
NUM_CLASSES: int = max(objectID_dict.values()) + 1
EXPANDED_FEATURES: int = (OBS_FEATURES - 1) + NUM_CLASSES  # 6 non-id + one-hot kind_id

# All enemy car kind IDs — every key that starts with "car_" except "car_player"
CAR_KIND_IDS: list[int] = sorted(
    v for k, v in objectID_dict.items() if k.startswith("car_") and k != "car_player"
)

# [HLP-flag] Flag object kind ID — the goal-flag alignment penalty measures the chosen
# goal's distance to obs rows carrying this kind (matched in obs[:, :, 6], same convention
# as hlp_extract_car_rows uses for cars).
FLAG_KIND_ID: int = objectID_dict["object_flag"]

# [HLP-flag] Explicit nearest-flag features appended to the HLP base input:
# [present, rel_x_norm, rel_y_norm]. See hlp_flag_features for why this is needed.
FLAG_FEAT_DIM: int = 3

# [HLP-param] Canonical parameterized HLP reward coefficients. The order here defines
# BOTH the θ feature layout appended to the HLP base input AND the column order of all
# hlp_params tensors/buffers. hlp_param_config in hlp_metadata.json is written in this
# order — downstream consumers (simulate_hlp.py, hlp_policy.py) reconstruct the layout
# from the metadata, never from hardcoded lists.
HLP_PARAM_NAMES: list[str] = [
    "hlp_collision_penalty",
    "hlp_cognitive_cost_coef",
    "hlp_goal_proximity_cost_coef",
    "hlp_flag_miss_penalty",
    "hlp_flag_collect_reward",
    "hlp_goal_flag_dist_penalty",
]
HLP_NUM_PARAMS: int = len(HLP_PARAM_NAMES)
# Column indices into hlp_params — readable reward composition in hlp_collect_rollout
_P_COLL, _P_COG, _P_PROX, _P_FMISS, _P_FCOLL, _P_GFLAG = range(HLP_NUM_PARAMS)


def hlp_sample_params(
    n: int,
    param_ranges: torch.Tensor,   # (P, 2) — [min, max] per coefficient, HLP_PARAM_NAMES order
    device: torch.device,
) -> torch.Tensor:                # (n, P)
    """Sample per-env HLP reward parameter vectors uniformly from their training ranges.

    [HLP-param] One θ row per env, each coefficient drawn independently from
    Uniform(min, max). Degenerate ranges (min == max) yield the constant value —
    reproducing fixed-coefficient training exactly.

    :param n: number of parameter vectors to draw (one per env).
    :param param_ranges: (P, 2) tensor of [min, max] rows in HLP_PARAM_NAMES order.
    :param device: target device.
    :return: (n, P) float tensor of sampled raw coefficient values.
    """
    lo = param_ranges[:, 0].to(device)
    hi = param_ranges[:, 1].to(device)
    return lo + torch.rand(n, param_ranges.shape[0], device=device) * (hi - lo)


# ---------------------------------------------------------------------------
# HLPAgent
# ---------------------------------------------------------------------------

class PooledMaskHead(nn.Module):
    """[HLP-maskhead] Permutation-equivariant per-slot mask head (DeepSets).

    Replaces the ``global`` head's ``Linear(hidden, K)``, which maps ONE global vector to K
    logits with no structural tie between ``logit_j`` and ``car_j`` — the routing must be
    learned, and Fix B's random slot permutation (see hlp_collect_rollout) makes the target
    permutation-EQUIVARIANT, which a plain MLP has no inductive bias for. MEASURED outcome:
    the logits collapse to one effective GLOBAL logit per decision (iid test 2026-07-15 —
    p_hat=0.301 predicts 75.3% partial masks, observed 1.8%), i.e. all-ON/all-OFF mixing
    rather than per-car selection. See implementation.md Design Log 12.

    Structure — each slot is scored RELATIVE to the field, so "is car j among the most
    important here?" is directly computable, and the per-slot cognitive price makes the
    count minimise itself:

        e_j     = enc(car_feats_j)                      # shared per-car encoder
        pooled  = [mean, max] over VALID slots only     # the comparison mechanism
        logit_j = score([e_j, pooled, ctx])             # ctx = scene, WITHOUT car_feats

    Equivariant by construction, so Fix B's permutation becomes unnecessary (harmless).

    NOTE (attention variant, deliberately NOT implemented — 2026-07-15 user decision):
    replacing mean/max pooling with self-attention across slots would add PAIRWISE
    interactions (showing A only helps if B is also shown) that mean-field pooling cannot
    express. Drop-in: run ``nn.MultiheadAttention`` over the (B, K, H) slot embeddings with
    ``key_padding_mask=~valid`` before pooling; ``score`` and everything else is unchanged.

    :param K: number of car slots.
    :param feat_dim: per-slot feature width (EXPANDED_FEATURES).
    :param context_dim: width of the goal-conditioned context (car_feats EXCLUDED).
    :param hidden_size: width of the encoder / context / scorer MLPs.
    :ivar enc: shared per-car encoder applied to every slot independently.
    :vartype enc: nn.Sequential
    :ivar ctx_enc: encoder for the car-free scene context.
    :vartype ctx_enc: nn.Sequential
    :ivar score: maps [e_j | pooled(2H) | ctx] -> scalar logit for slot j.
    :vartype score: nn.Sequential
    """

    def __init__(self, K: int, feat_dim: int, context_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.K = K
        self.feat_dim = feat_dim
        self.enc = nn.Sequential(
            nn.Linear(feat_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.ctx_enc = nn.Sequential(
            nn.Linear(context_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        # [e_j | mean | max | ctx] -> logit
        self.score = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, car_feats: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Score every slot relative to the field.

        :param car_feats: (B, K, feat_dim) per-slot features; invalid slots are all-zero
            (``build_hlp_input`` does ``car_feats[~valid_slots] = 0.0``) — validity is
            RE-DERIVED from that convention here, which is why no valid_slots argument has
            to be threaded through get_action_and_value and all its call sites.
        :type car_feats: torch.Tensor
        :param context: (B, context_dim) car-free scene context incl. the goal offset.
        :type context: torch.Tensor
        :return: (B, K) per-slot mask logits.
        :rtype: torch.Tensor
        """
        # Validity from the all-zero convention: a real car always has a kind one-hot set,
        # so its feature row can never be all-zero.
        valid = (car_feats.abs().sum(dim=2) > 0).float().unsqueeze(2)   # (B, K, 1)
        e = self.enc(car_feats) * valid                                 # (B, K, H) — zero out empties
        n = valid.sum(dim=1).clamp(min=1.0)                             # (B, 1) — avoid /0 when no cars
        mean = e.sum(dim=1) / n                                         # (B, H) — mean over VALID only
        # max over VALID only: push empty slots to -inf so they never win the max
        mx = e.masked_fill(valid == 0, float("-inf")).max(dim=1).values
        mx = torch.nan_to_num(mx, neginf=0.0)                           # all-empty -> 0
        c = self.ctx_enc(context)                                       # (B, H)
        pooled = torch.cat([mean, mx, c], dim=1).unsqueeze(1).expand(-1, self.K, -1)
        return self.score(torch.cat([e, pooled], dim=2)).squeeze(2)     # (B, K)


class HLPAgent(nn.Module):
    """HLP agent: optional goal selection (lane + dy-bin categoricals) plus K-Bernoulli car mask.

    [HLP-goal] Two-stage autoregressive factorisation:
        p(goal, mask | s) = p(lane | s) · p(dy_bin | s) · p(mask | s, goal)
    Goal heads read the goal-free base input; the mask head reads base input with the
    chosen goal offset (goal_dx, goal_dy) appended — same conditioning semantics as the
    previous design, except the goal is self-chosen instead of externally sampled.
    Critic reads the base input only (V(s) must not condition on the action).

    :param input_dim: BASE input dim = (OBS_SIZE + K) * EXPANDED_FEATURES — WITHOUT the
        +2 goal dims. The mask actor internally consumes input_dim + 2.
    :param K: number of Bernoulli mask outputs (car slots).
    :param hidden_size: actor MLP hidden width (goal and mask heads).
    :param critic_hidden_size: critic MLP hidden width; defaults to hidden_size.
    :param num_lanes: number of lane choices (LANECOUNT).
    :param num_dy_bins: number of forward-distance bins.
    :param goal_dy_min: lower edge of the dy range (goal_forward_m_min).
    :param goal_dy_bin_width: effective bin width — (max − min) / num_dy_bins.
    :param lane_centers: (num_lanes,) world-x lane centres (LLP_LANE_X_CENTERS).
    """

    def __init__(
        self,
        input_dim: int,
        K: int,
        hidden_size: int = 64,
        critic_hidden_size: int | None = None,
        num_lanes: int = 0,
        num_dy_bins: int = 0,
        goal_dy_min: float = 0.0,
        goal_dy_bin_width: float = 0.5,
        lane_centers: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.K = K
        self.num_lanes = num_lanes
        self.num_dy_bins = num_dy_bins
        _c = critic_hidden_size if critic_hidden_size is not None else hidden_size
        # [HLP-maskhead] car_feats are sliced out of base_in and reach the head ONLY via the
        # per-slot path; the context is the scene WITHOUT them.
        self.pooled_actor = PooledMaskHead(
            K=K, feat_dim=EXPANDED_FEATURES,
            context_dim=input_dim - K * EXPANDED_FEATURES + 2,   # base minus cars, plus goal
            hidden_size=hidden_size,
        )
        # [HLP-goal] critic reads the goal-free base input
        self.critic = nn.Sequential(
            nn.Linear(input_dim, _c),
            nn.Tanh(),
            nn.Linear(_c, _c),
            nn.Tanh(),
            nn.Linear(_c, _c),
            nn.Tanh(),
            nn.Linear(_c, 1),
        )
        assert num_lanes > 0 and num_dy_bins > 0 and lane_centers is not None, \
            "num_lanes, num_dy_bins and lane_centers are required"
        # [HLP-goal] shared goal trunk → concatenated (lane | dy_bin) logits
        self.goal_actor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_lanes + num_dy_bins),
        )
        # Registered buffers → move with .to(device) and save with state_dict
        self.register_buffer("lane_centers", lane_centers.float())
        self.register_buffer(
            "dy_bin_centers",
            goal_dy_min + (torch.arange(num_dy_bins).float() + 0.5) * goal_dy_bin_width,
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return V(s) from the goal-free base input. Shape (B,)."""
        return self.critic(x).squeeze(-1)  # (B,)

    def get_action_and_value(
        self,
        base_in: torch.Tensor,
        x_m: torch.Tensor | None = None,
        y_m: torch.Tensor | None = None,
        goal_dxdy: torch.Tensor | None = None,
        given_samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        deterministic: bool = False,
        deterministic_mask: bool | None = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict | None]:
        """Sample (or evaluate) the joint (goal, mask) action.

        [HLP-goal] Modes:
          1. rollout: pass ``x_m``/``y_m`` (player world position, (B,)); the goal is
             sampled internally and ``goal_dxdy`` is ignored.
          2. PPO update: pass ``given_samples=(lane, dy_bin, mask)`` and ``goal_dxdy`` =
             the stored (B, 2) offsets from rollout time.

        :param base_in: (B, input_dim) goal-free base input.
        :param x_m: (B,) player world x — required in mode 1.
        :param y_m: (B,) player world y — required in mode 1.
        :param goal_dxdy: (B, 2) goal offset (goal − player) — required in mode 2.
        :param given_samples: stored actions for logprob recomputation (mode 2).
        :param deterministic: argmax lanes/bins, threshold mask (greedy rollout).
        :param deterministic_mask: override JUST the mask's determinism, independent of the goal
            heads. None (default) → follows ``deterministic`` (training/other callers unchanged).
            True forces the greedy threshold mask even when the goal is sampled — used by the
            simulator so the shown construal is the policy's committed decision, not a random draw.
        :return: (actions, log_prob, entropy, value, goal_info) where
            actions = {"mask": (B,K), "lane": (B,)|None, "dy_bin": (B,)|None};
            log_prob/entropy = joint over all sampled heads, (B,);
            goal_info (mode 1 only) = {"goal_x", "goal_y", "dy", "goal_dxdy"} else None.
        """
        B = base_in.shape[0]
        device = base_in.device
        goal_info: dict | None = None

        # ── Goal heads: lane + dy-bin categoricals from base input ──────────
        gl = self.goal_actor(base_in)                        # (B, num_lanes + num_dy_bins)
        lane_dist = Categorical(logits=gl[:, : self.num_lanes])
        bin_dist = Categorical(logits=gl[:, self.num_lanes :])
        if given_samples is None:
            # Rollout: sample (or argmax) goal, then derive world goal + offsets
            lane = lane_dist.probs.argmax(-1) if deterministic else lane_dist.sample()
            dy_bin = bin_dist.probs.argmax(-1) if deterministic else bin_dist.sample()
            goal_x = self.lane_centers[lane]                 # (B,)
            dy = self.dy_bin_centers[dy_bin]                 # (B,)
            # Same clamp convention as llp_sample_goals (goal never behind track start)
            goal_y = torch.clamp(y_m - dy, min=0.0)
            goal_dxdy = torch.stack([goal_x - x_m, goal_y - y_m], dim=1)  # (B, 2)
            goal_info = {"goal_x": goal_x, "goal_y": goal_y, "dy": dy, "goal_dxdy": goal_dxdy}
        else:
            # PPO update: logits depend only on base_in; stored goal_dxdy conditions the mask head
            lane, dy_bin, given_samples = given_samples
        goal_logp = lane_dist.log_prob(lane) + bin_dist.log_prob(dy_bin)
        goal_ent = lane_dist.entropy() + bin_dist.entropy()

        # ── Mask head conditioned on the (chosen or external) goal ──────────────
        # Slice the per-slot block back out of the base layout
        # [proc_obs | car_feats | flag_feats | theta] — see build_hlp_input.
        _c0 = OBS_SIZE * EXPANDED_FEATURES
        _c1 = _c0 + self.K * EXPANDED_FEATURES
        car_feats = base_in[:, _c0:_c1].view(B, self.K, EXPANDED_FEATURES)
        context = torch.cat(
            [base_in[:, :_c0], base_in[:, _c1:], goal_dxdy], dim=1
        )                                                    # scene WITHOUT car_feats, + goal
        logits = self.pooled_actor(car_feats, context)       # (B, K)
        dist = Bernoulli(logits=logits)
        if given_samples is None:
            # Mask determinism can be forced independently of the goal (deterministic_mask);
            # None falls back to `deterministic`, so training / other callers are unchanged.
            _det_mask = deterministic if deterministic_mask is None else deterministic_mask
            mask = (logits > 0).float() if _det_mask else dist.sample()
        else:
            mask = given_samples
        log_prob = goal_logp + dist.log_prob(mask).sum(-1)       # (B,)  joint = goal + mask
        entropy = goal_ent + dist.entropy().sum(-1)              # (B,)
        value = self.critic(base_in).squeeze(-1)                 # (B,)

        actions = {"mask": mask, "lane": lane, "dy_bin": dy_bin}
        return actions, log_prob, entropy, value, goal_info


# ---------------------------------------------------------------------------
# hlp_extract_car_rows
# ---------------------------------------------------------------------------

def hlp_extract_car_rows(
    raw_obs_flat: torch.Tensor,  # (B, OBS_SIZE * OBS_FEATURES)
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identify the K highest-priority car rows in each env's observation.

    Returns:
        car_row_indices: (B, K) long — obs row index per slot; invalid slots clamped to 0
        num_cars:        (B,)   long — number of valid car slots (≤ K)
    """
    B = raw_obs_flat.shape[0]
    device = raw_obs_flat.device
    obs_3d = raw_obs_flat.view(B, OBS_SIZE, OBS_FEATURES)
    kind_ids = obs_3d[:, :, 6].long()  # (B, OBS_SIZE)

    # [HLP-slots] OBJECT-TYPE FILTER FIRST, then NEAREST-K. score = -distance for CARS and -inf
    # for everything else, so topk(largest) returns the K NEAREST CARS. Applying the type filter
    # (non-cars → -inf) BEFORE the distance ranking is essential: it guarantees non-car objects
    # (flags / obstacles) can never take a slot and push out a relevant car — doing it in the
    # other order (nearest-K over all objects, then filter to cars) would drop cars. When >K cars
    # are visible the far overflow is excluded here and kept INVISIBLE by apply_car_mask (not part
    # of the immediate decision). Distance = ego-centric hypot(rel_x, rel_y) from the obs.
    is_car = torch.zeros(B, OBS_SIZE, dtype=torch.bool, device=device)
    for cid in CAR_KIND_IDS:
        is_car |= (kind_ids == cid)

    dist = torch.linalg.vector_norm(obs_3d[:, :, :2], dim=2)   # (B, OBS_SIZE) ego-centric distance
    scores = torch.where(is_car, -dist, torch.full_like(dist, float("-inf")))

    k_query = min(K, OBS_SIZE)
    topk_vals, topk_idx = torch.topk(scores, k=k_query, dim=1, largest=True, sorted=False)

    if K > OBS_SIZE:  # should never happen in practice
        pad_idx = torch.zeros(B, K - OBS_SIZE, dtype=torch.long, device=device)
        pad_val = torch.full((B, K - OBS_SIZE), float("-inf"), device=device)
        topk_idx = torch.cat([topk_idx, pad_idx], dim=1)
        topk_vals = torch.cat([topk_vals, pad_val], dim=1)

    # | Move real cars to the front. `sorted=False` fixes WHICH entries topk returns, not
    # | their order, and callers mark the first `num_cars` slots valid. A stable sort is the
    # | identity where cars already led (bit-identical), and correct elsewhere -- the NYU
    # | cluster's CPU build returned filler first (SSP implementation.md, Design Log 24).
    order = torch.argsort((~topk_vals.isfinite()).to(torch.int8), dim=1, stable=True)
    topk_vals = topk_vals.gather(1, order)
    topk_idx = topk_idx.gather(1, order)

    valid = topk_vals.isfinite()          # (B, K) — slots with a real car, now a prefix
    num_cars = valid.sum(dim=1)           # (B,)  long

    car_row_indices = topk_idx.long()     # (B, K)
    car_row_indices[~valid] = 0           # clamp invalid slots to row 0 (safe — never zeroed)

    return car_row_indices, num_cars


# ---------------------------------------------------------------------------
# apply_car_mask
# ---------------------------------------------------------------------------

def apply_car_mask(
    raw_obs: torch.Tensor,          # (B, OBS_SIZE, OBS_FEATURES)
    car_row_indices: torch.Tensor,  # (B, K) long — may be permuted
    car_mask: torch.Tensor,         # (B, K) float {0.0, 1.0}
    valid_slots: torch.Tensor,      # (B, K) bool — explicit validity mask (permutation-aware)
) -> torch.Tensor:                  # (B, OBS_SIZE, OBS_FEATURES) — cloned
    """Keep visible ONLY the selected car slots the HLP shows; hide every other car row.

    [HLP-slots] A car is left visible iff it is one of the K selected (nearest) cars in a VALID
    slot AND its mask bit = 1. Two groups are therefore zeroed: (a) selected slots the HLP masked
    off, and (b) OVERFLOW cars beyond the K nearest — excluded from candidacy by the slot limit
    and kept invisible, since they are not part of the immediate decision. For ≤K cars there is no
    overflow, so this reduces exactly to the previous "hide masked-off slots" behaviour.
    NOTE: the selection (car_row_indices) is the block-init / obs-change one, so a far overflow car
    that closes distance mid-block without an obs-change stays hidden until the next re-selection
    or block end — acceptable given short sub-episodes and that overflow = the farthest cars.
    Returns a clone; input unchanged.
    """
    B, K = car_mask.shape
    device = raw_obs.device

    obs = raw_obs.clone()

    # Force mask=1 for invalid slots (they clamp to row 0 / no real car, so they mark nothing).
    effective_mask = torch.where(valid_slots, car_mask, torch.ones_like(car_mask))

    # Mark the SHOWN selected car rows, then zero every car row that is not shown
    # (masked-off selected slots + all non-selected overflow cars).
    shown = torch.zeros(B, OBS_SIZE, device=device)
    shown.scatter_(1, car_row_indices, effective_mask)         # (B, OBS_SIZE) float {0,1}
    kind_ids = raw_obs[:, :, 6].long()                         # (B, OBS_SIZE)
    is_car = torch.zeros(B, OBS_SIZE, dtype=torch.bool, device=device)
    for cid in CAR_KIND_IDS:
        is_car |= (kind_ids == cid)
    rows_to_zero = is_car & (shown < 0.5)
    obs[rows_to_zero] = 0.0                                    # broadcast over OBS_FEATURES

    return obs


# ---------------------------------------------------------------------------
# hlp_goal_proximity_cost
# ---------------------------------------------------------------------------

def hlp_goal_proximity_cost(
    dy: torch.Tensor,      # (B,) chosen forward distance in metres
    dy_min: float,
    dy_max: float,
    coef: float | torch.Tensor,
) -> torch.Tensor:         # (B,) non-negative cost
    """Linear proximity cost — closer goals carry stronger penalties.

    [HLP-goal] cost = coef × (1 − d_norm), where d_norm normalises dy to [0, 1] over
    the LLP goal range. Maximal at the closest allowed goal, zero at the farthest.

    :param dy: chosen forward distance per env, metres.
    :param dy_min: goal_forward_m_min (LLP checkpoint value).
    :param dy_max: goal_forward_m_max (LLP checkpoint value).
    :param coef: hlp_goal_proximity_cost_coef — scalar, or (B,) per-env θ column
        [HLP-param]; broadcasts elementwise either way.
    :return: (B,) cost in [0, coef].
    """
    d_norm = ((dy - dy_min) / max(dy_max - dy_min, 1e-8)).clamp(0.0, 1.0)
    return coef * (1.0 - d_norm)


# ---------------------------------------------------------------------------
# hlp_goal_flag_distance
# ---------------------------------------------------------------------------

def hlp_goal_flag_distance(
    goal_dxdy: torch.Tensor,        # (B, 2) ego-centric goal offset (goal − player), metres
    raw_obs_flat: torch.Tensor,     # (B, OBS_SIZE * OBS_FEATURES)
) -> torch.Tensor:                  # (B,) distance to nearest in-view flag; 0 if none
    """Euclidean distance from the chosen goal to the nearest flag visible in the obs.

    [HLP-flag] Drives goal-flag alignment: the block-close reward subtracts
    ``coef × distance``, so the HLP is penalised for placing goals away from flags and the
    penalty shrinks to 0 as a goal lands on a flag. Flags are obs rows with
    ``kind_id == FLAG_KIND_ID``; their ``[x, y]`` features are ego-centric (same frame as
    ``goal_dxdy`` — both are offsets from the player, in metres), so the distance is a
    direct subtraction. Envs with NO flag in view return 0 → zero penalty (as specified).

    :param goal_dxdy: (B, 2) chosen goal offset from the player, metres.
    :type goal_dxdy: torch.Tensor
    :param raw_obs_flat: (B, OBS_SIZE * OBS_FEATURES) block-init observation.
    :type raw_obs_flat: torch.Tensor
    :return: (B,) distance to the nearest visible flag (0 when none in view).
    :rtype: torch.Tensor
    """
    B = raw_obs_flat.shape[0]
    obs_3d = raw_obs_flat.view(B, OBS_SIZE, OBS_FEATURES)
    is_flag = obs_3d[:, :, 6].long() == FLAG_KIND_ID          # (B, OBS_SIZE)
    flag_xy = obs_3d[:, :, :2]                                 # (B, OBS_SIZE, 2) ego-centric
    dist = torch.linalg.vector_norm(flag_xy - goal_dxdy.unsqueeze(1), dim=2)  # (B, OBS_SIZE)
    # Mask non-flag rows to +inf so the row-min picks the nearest flag; inf ⇒ no flag in view.
    dist = torch.where(is_flag, dist, torch.full_like(dist, float("inf")))
    min_dist = dist.min(dim=1).values                         # (B,) — inf when no flag
    return torch.where(torch.isfinite(min_dist), min_dist, torch.zeros_like(min_dist))


# ---------------------------------------------------------------------------
# hlp_flag_features
# ---------------------------------------------------------------------------

def hlp_flag_features(
    raw_obs_flat: torch.Tensor,     # (B, OBS_SIZE * OBS_FEATURES) — raw, for the nearest-flag pick
    proc_obs_3d: torch.Tensor,      # (B, OBS_SIZE, EXPANDED_FEATURES) — preprocess_obs output
) -> torch.Tensor:                  # (B, FLAG_FEAT_DIM) = [present, rel_x_norm, rel_y_norm]
    """Explicit nearest-flag features for the goal head — zeros when no flag is in view.

    [HLP-flag] WHY (measured 2026-07-15): the goal-flag penalty grades the chosen goal against the
    nearest visible flag, but the flag only ever appeared in the policy input as ONE row among
    OBS_SIZE inside proc_obs, at a SHIFTING index (objects are appended in active-set order — the
    flag was observed at 13 different rows). Reading it requires a conditional gather ("find the row
    whose kind one-hot is flag, then read THAT row's x/y") — a multiplicative interaction an MLP
    cannot express. Consequence: the goal head learned flag PRESENCE (a linear sum over the flag
    one-hot) but not flag POSITION — it collapsed to ~3 dy modes, sat 2.4 m from reachable flags
    (3.5x the 0.69 m goal-grid floor), and demonstrably left reward on the table (at gflag=0.05 /
    prox=0.15, targeting a 5 m flag is worth +0.25 net and it still didn't). This surfaces the very
    offset the REWARD already computes, mirroring what car_feats does for the mask head.

    TWO OBSERVATION SOURCES, deliberately:
    - The nearest flag is SELECTED on the RAW obs, i.e. in true metres — the same metric
      :func:`hlp_goal_flag_distance` uses for the reward, so the flag described here is the flag the
      reward grades against. Selecting on proc_obs would use an ANISOTROPIC metric (x is scaled by
      1/visible_width, y by 1/visible_height, ratio ~1.375), which can pick a DIFFERENT flag: e.g. a
      flag at (4 m, 0) vs one at (0, -5 m) → metres picks the first (4.0 < 5.0), proc-space picks the
      second (0.455 < 0.500).
    - The emitted POSITION is GATHERED from proc_obs (already normalized by preprocess_obs, exactly
      as car_feats are), so the normalization convention lives in ONE place. Re-dividing here would
      duplicate it and silently drift if preprocess_obs ever changes. The obs scale matters: raw
      metres (~±12) next to O(1) neighbours get ignored by the shared trunk (theta-normalization note).

    :param raw_obs_flat: (B, OBS_SIZE * OBS_FEATURES) raw observation (metres) — nearest-flag pick.
    :type raw_obs_flat: torch.Tensor
    :param proc_obs_3d: (B, OBS_SIZE, EXPANDED_FEATURES) preprocess_obs output — normalized values.
    :type proc_obs_3d: torch.Tensor
    :return: (B, 3) [present, rel_x_norm, rel_y_norm]; all-zero when no flag is visible.
    :rtype: torch.Tensor
    """
    B = raw_obs_flat.shape[0]
    device = raw_obs_flat.device
    obs_3d = raw_obs_flat.view(B, OBS_SIZE, OBS_FEATURES)
    is_flag = obs_3d[:, :, 6].long() == FLAG_KIND_ID           # (B, OBS_SIZE)
    dist = torch.linalg.vector_norm(obs_3d[:, :, :2], dim=2)    # METRES — matches the reward's metric
    dist = torch.where(is_flag, dist, torch.full_like(dist, float("inf")))
    min_d, idx = dist.min(dim=1)                                # nearest flag (inf ⇒ none)
    present = torch.isfinite(min_d).float()                     # (B,)
    # Gather the ALREADY-normalized position from proc_obs (single source of truth for the scaling).
    # When no flag, idx falls back to row 0 = ego, whose x/y proc_obs has zeroed (ego_pos_reset).
    nearest = proc_obs_3d[torch.arange(B, device=device), idx, :2]      # (B, 2)
    feats = torch.cat([present.unsqueeze(1), nearest], dim=1)   # (B, 3)
    return feats * present.unsqueeze(1)                         # zeroed when no flag


# ---------------------------------------------------------------------------
# build_hlp_input
# ---------------------------------------------------------------------------

def normalize_theta(
    hlp_params: torch.Tensor,     # (B, P) raw θ values
    param_ranges: torch.Tensor,   # (P, 2) [min, max] per coefficient, HLP_PARAM_NAMES order
) -> torch.Tensor:                # (B, P) normalized to [-1, 1]
    """Map raw θ to the [-1, 1] obs-feature scale using the training ranges.

    [HLP-param-norm] Raw θ values (magnitude ~1e-2) are ~60× smaller than the O(1)
    normalized obs features and get effectively ignored by the shared trunk (measured
    on param1: ≤0.08 corner-to-corner probability shift). Rescaling each coefficient to
    [-1, 1] over its training range puts θ on equal footing with the obs, so the
    network can condition on it. Uniform sampling over [min, max] maps to a zero-mean
    [-1, 1] feature. Degenerate dims (min == max) map to 0 (neutral centre, no info).

    :param hlp_params: (B, P) raw coefficient values.
    :type hlp_params: torch.Tensor
    :param param_ranges: (P, 2) [min, max] rows in HLP_PARAM_NAMES order.
    :type param_ranges: torch.Tensor
    :return: (B, P) normalized θ in [-1, 1] (0 for degenerate dims).
    :rtype: torch.Tensor
    """
    lo = param_ranges[:, 0]
    span = param_ranges[:, 1] - lo
    normed = 2.0 * (hlp_params - lo) / span.clamp_min(1e-8) - 1.0
    return torch.where(span > 0, normed, torch.zeros_like(hlp_params))


def build_hlp_input(
    raw_obs_flat: torch.Tensor,                     # (B, OBS_SIZE * OBS_FEATURES)
    K: int,
    args,
    device: torch.device,
    hlp_params: torch.Tensor,                       # (B, P) raw θ values; (B, 0) for pre-param-era checkpoints
    car_row_indices: torch.Tensor | None = None,    # pre-computed (B, K); if None, extracted internally
    valid_slots: torch.Tensor | None = None,        # (B, K) bool; if None, derived from extraction
    param_ranges: torch.Tensor | None = None,       # (P, 2) θ training ranges (required); θ is normalized to [-1, 1]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the goal-free HLP base input tensor and return supporting state.

    [HLP-goal] The goal is not part of this input: it is an action. The goal heads and
    the critic consume this base input directly.
    [HLP-param] Layout: [proc_obs | car_feats | flag_feats(3) | θ(P)].
    [HLP-flag] flag_feats = nearest-flag [present, rel_x, rel_y] (obs-normalized, zeros if no flag)
    — the goal head cannot gather the flag out of proc_obs itself (see hlp_flag_features).
    θ is normalized to [-1, 1] over ``param_ranges`` via :func:`normalize_theta`.

    Returns (hlp_in_base, car_row_indices, valid_slots).
    hlp_in_base shape: (B, OBS_SIZE*EF + K*EF + FLAG_FEAT_DIM + P)
    If car_row_indices and valid_slots are provided (e.g. after random permutation),
    hlp_extract_car_rows is skipped.
    """
    B = raw_obs_flat.shape[0]

    if car_row_indices is None:
        car_row_indices, num_cars = hlp_extract_car_rows(raw_obs_flat, K)
        valid_slots = torch.arange(K, device=device).unsqueeze(0) < num_cars.unsqueeze(1)

    proc_obs = preprocess_obs(
        raw_obs_flat, args.visible_width, args.visible_height,
        args.max_speed, NUM_CLASSES, ego_pos_reset=True,
    )  # (B, OBS_SIZE * EXPANDED_FEATURES)

    # Gather processed car slot features; zero invalid slots
    proc_obs_3d = proc_obs.view(B, OBS_SIZE, EXPANDED_FEATURES)
    car_feats = proc_obs_3d[
        torch.arange(B, device=device).unsqueeze(1), car_row_indices
    ]  # (B, K, EXPANDED_FEATURES)
    car_feats[~valid_slots] = 0.0
    car_feats_flat = car_feats.view(B, K * EXPANDED_FEATURES)

    # [HLP-flag] Explicit nearest-flag features so the GOAL head can actually read where the flag
    # is (cars get car_feats; flags previously got nothing — see hlp_flag_features for the measured
    # rationale). Nearest picked on RAW (metres, matching the reward); value gathered from proc_obs
    # (already normalized — same source of truth as car_feats). Zeros when no flag is in view.
    flag_feats = hlp_flag_features(raw_obs_flat, proc_obs_3d)

    # [HLP-param-norm] normalize θ to the obs scale when ranges are provided
    assert param_ranges is not None, "param_ranges is required: theta is normalized to [-1, 1]"
    theta_feat = normalize_theta(hlp_params, param_ranges)

    parts = [proc_obs, car_feats_flat, flag_feats, theta_feat]
    hlp_in = torch.cat(parts, dim=1)
    return hlp_in, car_row_indices, valid_slots


# ---------------------------------------------------------------------------
# hlp_check_termination
# ---------------------------------------------------------------------------

def hlp_check_termination(
    next_raw_obs: torch.Tensor,      # (B, OBS_SIZE * OBS_FEATURES)
    kind_id_snap: torch.Tensor,      # (B, OBS_SIZE) float — kind_id at last soft-reinit
    subepisode_steps: torch.Tensor,  # (B,) long
    goal_x: torch.Tensor,            # (B,)
    goal_y: torch.Tensor,            # (B,)
    env_done: torch.Tensor,          # (B,) bool
    y_max: float,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check LLP subepisode termination conditions.

    Returns (sub_done, goal_reached, goal_missed, kind_id_changed) — all (B,) bool.
    goal_missed is computed directly (y_m < goal_y & ~reached) regardless of
    llp_miss_ends_subepisode so the HLP reward always tracks misses.
    [HLP-exp] kind_id_changed is NOT OR'd into sub_done. Caller handles obs changes
    via identity-based mask remap independently of sub_done.
    """
    B = next_raw_obs.shape[0]
    device = next_raw_obs.device
    obs_3d = next_raw_obs.view(B, OBS_SIZE, OBS_FEATURES)

    x_m, y_m = llp_extract_player_xy_m(next_raw_obs, y_max)

    _, sub_done_float, goal_reached_float = llp_compute_reward_and_flags(
        x_m, y_m, goal_x, goal_y,
        subepisode_steps, env_done,
        args.goal_radius_m,
        args.llp_reach_reward,
        args.llp_miss_penalty,
        args.llp_truncation_penalty,
        args.sub_episode_horizon,
        args.llp_miss_ends_subepisode,
    )

    sub_done = sub_done_float.bool()
    goal_reached = goal_reached_float.bool()

    # Direct miss detection: player passed below goal without reaching it
    goal_missed = (y_m < goal_y) & ~goal_reached & ~env_done.bool()

    # [HLP-exp] kind_id_changed is no longer OR'd into sub_done.
    # Obs changes trigger identity-based mask remapping in hlp_collect_rollout instead.
    new_kind_ids = obs_3d[:, :, 6]  # (B, OBS_SIZE) float
    kind_id_changed = (new_kind_ids != kind_id_snap).any(dim=1).bool()

    return sub_done, goal_reached, goal_missed, kind_id_changed


# ---------------------------------------------------------------------------
# hlp_collect_rollout
# ---------------------------------------------------------------------------

def hlp_collect_rollout(
    envs,
    llp_agent: nn.Module,
    hlp_agent: HLPAgent,
    # Pre-allocated output buffers — written in-place
    buffers: dict,          # keys: hlp_obs, car_masks, hlp_logprobs, hlp_values, hlp_rewards,
                            #   hlp_dones, hlp_next_values, goal_lanes, goal_dy_bins, goal_dxdy,
                            #   flag_misses, flag_collects, goal_flag_dists, hlp_params
    # Carry-over state — modified in-place via returned values
    next_raw_obs: torch.Tensor,        # (B, obs_flat)
    next_done: torch.Tensor,           # (B,) float
    prev_actions: torch.Tensor,        # (B,) long
    obs_idx_current: torch.Tensor,     # (B, MAX_OBJECTS) long — obs_idx from latest step's infos
    hlp_params: torch.Tensor,          # (B, P) per-env θ — carried across iterations [HLP-param]
    pending_param_reset: torch.Tensor, # (B,) bool — envs needing a fresh θ at next block init [HLP-param]
    # Scalar config
    infos_are_dict: bool,
    B: int,
    K: int,
    y_max: float,
    global_step: int,
    device: torch.device,
    args,
    param_ranges: torch.Tensor,        # (P, 2) θ sampling ranges, HLP_PARAM_NAMES order [HLP-param]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fill one iteration's worth of block-level HLP experience buffers.

    [HLP-delta] Replaces the `for step in range(args.num_steps)` loop from PPO_cleanRL.py.
    Runs num_hlp_steps blocks; each block = one LLP sub-episode per env (ends on sub_done).
    HLP mask is fixed for the block duration; obs changes trigger identity-based mask remapping
    rather than a new HLP decision.

    [HLP-exp] obs_idx_current is carried in/out alongside next_raw_obs. It maps obs object-
    section position j → obj_state row index. Used at block init to build init_slot_obj_ids
    and in the obs-change remap to look up new slot identities.

    [HLP-param] hlp_params holds each env's current raw θ (reward coefficient vector).
    Sampled per env-EPISODE: envs whose episode ended during a block are flagged in
    pending_param_reset (latched from episode_done_this_block at block close) and get a
    fresh θ at the NEXT block init — never mid-block, so the θ inside the stored HLP obs
    and the θ used for the block reward always agree.

    Returns (next_raw_obs, next_done, prev_actions, global_step, mean_sub_ep_len,
    obs_idx_current, hlp_params, pending_param_reset).
    """
    env_reward_coef  = torch.full((B,), args.llp_env_reward_coef, dtype=torch.float32, device=device)
    sub_ep_len_sum   = 0.0   # sum of per-env sub-episode lengths across the iteration
    sub_ep_len_count = 0     # number of completed env sub-episodes

    for s in range(args.num_hlp_steps):
        # ── BLOCK INIT ──────────────────────────────────────────────────── [HLP-delta]
        # Every block starts a fresh LLP subepisode: goals resampled, subepisode_steps=0.
        # prev_actions carries over (not reset at subepisode boundaries per LLP training).

        # 0. [HLP-param] Lazy per-episode θ resample. Envs whose episode ended during
        # the previous block get fresh parameters now — the first HLP decision of the
        # new episode IS this block init, so this is exactly per-episode sampling.
        # θ is never resampled mid-block, so the reward at block close and the θ inside
        # the stored HLP obs always agree.
        if pending_param_reset.any():
            n_reset = int(pending_param_reset.sum().item())
            hlp_params[pending_param_reset] = hlp_sample_params(n_reset, param_ranges, device)
            pending_param_reset = torch.zeros_like(pending_param_reset)

        # 1-2. Kind-id snapshot + car rows
        raw_3d = next_raw_obs.view(B, OBS_SIZE, OBS_FEATURES)
        kind_id_snap = raw_3d[:, :, 6].clone()                     # (B, OBS_SIZE)
        car_row_indices, num_cars = hlp_extract_car_rows(next_raw_obs, K)
        # [HLP-delta] Fix B: random permutation breaks correlation between slot index and
        # occupancy probability. topk always fills slot 0 first; later slots are empty more
        # often → less valid training signal per slot. After permutation every slot index
        # sees equal expected occupancy over training. perm is kept for kind-id soft-reinit.
        perm = torch.stack([torch.randperm(K, device=device) for _ in range(B)])  # (B, K)
        car_row_indices  = car_row_indices.gather(1, perm)
        valid_slots_snap = (torch.arange(K, device=device).unsqueeze(0) < num_cars.unsqueeze(1)).gather(1, perm)

        # 3. subepisode_steps reset for all envs (block = fresh subepisode)
        x_m_init, y_m_init = llp_extract_player_xy_m(next_raw_obs, y_max)
        subepisode_steps = torch.zeros(B, dtype=torch.long, device=device)

        # 4. Build goal-free HLP base input (θ normalized + appended inside — [HLP-param])
        hlp_in, _, _ = build_hlp_input(next_raw_obs, K, args, device, hlp_params,
                                        car_row_indices=car_row_indices, valid_slots=valid_slots_snap,
                                        param_ranges=param_ranges)

        # 5. HLP forward — goal + mask decision
        # [HLP-goal] the HLP samples (lane, dy_bin) and the mask in one joint action;
        # goal_x/goal_y derive from the sampled goal.
        with torch.no_grad():
            actions, hlp_logprob, _, hlp_value, goal_info = hlp_agent.get_action_and_value(
                hlp_in, x_m=x_m_init, y_m=y_m_init,
            )
            goal_x, goal_y = goal_info["goal_x"], goal_info["goal_y"]
            goal_dxdy_snap = goal_info["goal_dxdy"]
            goal_dy_chosen = goal_info["dy"]                     # (B,) — proximity cost input
            goal_lane_snap = actions["lane"]
            goal_bin_snap  = actions["dy_bin"]
            car_mask = actions["mask"]
            # [HLP-flag] Goal-flag alignment: distance from the chosen goal to the nearest
            # in-view flag (0 if none), applied as a per-block penalty at block close.
            goal_flag_dist_snap = hlp_goal_flag_distance(goal_dxdy_snap, next_raw_obs)

        # 6. Store HLP decision snaps (never overwritten during block)
        hlp_in_snap      = hlp_in.detach()
        # [HLP-exp] .clone() not .detach() — car_mask is modified in-place during the block
        # (identity remap on obs changes); snap must have separate storage for PPO correctness.
        car_mask_snap    = car_mask.clone()
        hlp_logprob_snap = hlp_logprob.detach()
        hlp_value_snap   = hlp_value.detach()

        # [HLP-exp] Identity map: for each slot, store the obj_state index of the car at
        # HLP decision time. Used to remap car_mask on obs changes without a new HLP decision.
        # NOTE: objects that leave the view and return are NOT tracked — they are found in
        # init_slot_obj_ids and restore their original bit (see implementation.md gap note).
        j_idx = (car_row_indices - _OBJ_ROW_OFFSET).clamp(0, MAX_OBJECTS - 1)  # (B, K)
        init_slot_obj_ids = obs_idx_current.gather(1, j_idx)                     # (B, K)
        init_slot_obj_ids[~valid_slots_snap] = -1                                # invalid slots → sentinel

        # [HLP-exp] Mutable validity mask for apply_car_mask; updated on obs changes.
        # valid_slots_snap stays frozen for cognitive cost and PPO buffer writes.
        valid_slots_current = valid_slots_snap.clone()

        # 7. Reset block accumulators
        accum_env_reward         = torch.zeros(B, device=device)
        collision_count          = torch.zeros(B, device=device)
        # [HLP-cogperstep] active-step count per block (same active gate as collision_count);
        # scales the cognitive cost by block duration.
        active_steps             = torch.zeros(B, device=device)
        # [HLP-flag] per-block flag outcome counts (goal-selection driver)
        flag_miss_count          = torch.zeros(B, device=device)
        flag_collect_count       = torch.zeros(B, device=device)
        goal_reached_this_block  = torch.zeros(B, dtype=torch.bool, device=device)
        episode_done_this_block  = torch.zeros(B, dtype=torch.bool, device=device)
        sub_done_ever            = torch.zeros(B, dtype=torch.bool, device=device)
        # [HLP-delta] Per-env bootstrap value, written when each env's first sub_done
        # fires (see inner loop). Stored in buffers["hlp_next_values"] at block close
        # and used in GAE in place of hlp_values[t+1].
        #
        # Problem it solves: the block stays open until ALL envs finish their
        # sub-episode (sub_done_ever.all()). A fast env that finishes at step 10
        # keeps stepping while slow envs catch up, arriving at block-close in a state
        # 30–80 inner steps past its actual sub_done. Using hlp_values[t+1] — which
        # is V(state at the START of the next block) — therefore bootstraps from a
        # displaced state for fast envs, injecting noise into the critic target.
        # Snapshotting V(s) at sub_done time eliminates this displacement entirely.
        bootstrap_values_snap = torch.zeros(B, device=device)
        # ── END BLOCK INIT ───────────────────────────────────────────────────────

        # ── INNER LOOP ──────────────────────────────────────────────────── [HLP-delta]
        for _inner in range(args.sub_episode_horizon):
            # LLP inference with masked obs
            # [HLP-exp] uses valid_slots_current (updated on obs changes) not valid_slots_snap
            masked_obs  = apply_car_mask(next_raw_obs.view(B, OBS_SIZE, OBS_FEATURES),
                                         car_row_indices, car_mask, valid_slots_current)
            masked_flat = masked_obs.view(B, -1)
            llp_obs     = llp_augment_obs(
                masked_flat, goal_x, goal_y, subepisode_steps,
                prev_actions, env_reward_coef, y_max, args.sub_episode_horizon,
            )
            with torch.no_grad():
                _feat = llp_agent.feature_extractor(llp_obs)
                llp_actions = llp_agent.actor(_feat).argmax(dim=-1)

            obs_np, rewards, terminated, truncated, infos = envs.step(llp_actions.cpu().numpy())
            next_raw_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            next_done    = torch.tensor(
                np.logical_or(terminated, truncated), dtype=torch.float32, device=device
            )
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            global_step += B

            # Extract per-step signals (format detected before training loop)
            if infos_are_dict:
                is_collision = torch.tensor(
                    infos.get("is_collision", np.zeros(B, dtype=np.float32)),
                    dtype=torch.float32, device=device,
                )
                # [HLP-flag] per-step flag outcome signals (goal-selection driver)
                is_flag_miss = torch.tensor(
                    infos.get("is_flag_miss", np.zeros(B, dtype=np.float32)),
                    dtype=torch.float32, device=device,
                )
                is_flag_collect = torch.tensor(
                    infos.get("is_flag_collect", np.zeros(B, dtype=np.float32)),
                    dtype=torch.float32, device=device,
                )
                # [HLP-exp] obs_idx: (B, MAX_OBJECTS) — maps obs position j → obj_state index.
                # Guard: pufferlib may not stack non-scalar info arrays correctly — if shape is
                # wrong, fall back to all-(-1) so cars are treated as new (hidden) on obs changes.
                _oi_raw = infos.get("obs_idx")
                if _oi_raw is not None and np.asarray(_oi_raw).shape == (B, MAX_OBJECTS):
                    _oi_np = np.asarray(_oi_raw, dtype=np.int32)
                else:
                    _oi_np = np.full((B, MAX_OBJECTS), -1, dtype=np.int32)
            else:
                is_collision = torch.tensor(
                    np.array([inf.get("is_collision", 0.0) for inf in infos],
                             dtype=np.float32),
                    device=device,
                )
                # [HLP-flag] per-step flag outcome signals (goal-selection driver)
                is_flag_miss = torch.tensor(
                    np.array([inf.get("is_flag_miss", 0.0) for inf in infos],
                             dtype=np.float32),
                    device=device,
                )
                is_flag_collect = torch.tensor(
                    np.array([inf.get("is_flag_collect", 0.0) for inf in infos],
                             dtype=np.float32),
                    device=device,
                )
                _oi_np = np.stack([
                    inf.get("obs_idx", np.full(MAX_OBJECTS, -1, dtype=np.int32))
                    for inf in infos
                ])
            obs_idx_current = torch.tensor(_oi_np, dtype=torch.long, device=device)

            # Accumulate only for envs that have not yet completed a sub-episode this block.
            # Freezes after the first sub_done so a second sub-episode within the same
            # block does not pollute the reward attribution for the HLP's decision.
            active_for_accum = ~episode_done_this_block & ~sub_done_ever
            accum_env_reward += active_for_accum.float() * rewards_t
            collision_count  += active_for_accum.float() * is_collision
            # [HLP-cogperstep] count active steps this block (same gate as collision_count)
            active_steps     += active_for_accum.float()
            # [HLP-flag] accumulate flag outcomes per block (same active gate as collisions)
            flag_miss_count    += active_for_accum.float() * is_flag_miss
            flag_collect_count += active_for_accum.float() * is_flag_collect

            prev_actions = llp_actions.clone()
            prev_actions[next_done.bool()] = 4   # IDLE on env reset

            subepisode_steps += 1

            # Subepisode termination check
            sub_done, goal_reached, goal_missed, kind_id_changed = hlp_check_termination(
                next_raw_obs, kind_id_snap, subepisode_steps,
                goal_x, goal_y, next_done.bool(), y_max, args,
            )

            goal_reached_this_block |= goal_reached & active_for_accum
            episode_done_this_block |= next_done.bool()

            # Record sub-episode lengths for newly-completed envs (before updating sub_done_ever)
            newly_done = sub_done & ~sub_done_ever
            if newly_done.any():
                n_new = int(newly_done.sum().item())
                sub_ep_len_sum   += n_new * (_inner + 1)
                sub_ep_len_count += n_new

                # [HLP-delta] Bootstrap snapshot at sub_done time.
                #
                # Build a fresh HLP base input for each newly-done env, mirroring block
                # init: new hlp_extract_car_rows + new random perm. This keeps the HLP
                # input distribution consistent with what block init produces, so the
                # critic sees the same input space during bootstrap as during training.
                # [HLP-goal] The critic input is goal-free — no goal sampling needed here.
                #
                # We do NOT reuse the current block's car_row_indices/perm because those
                # were drawn at the start of this block and may no longer reflect the obs
                # after a kind-id change. A fresh extraction gives a clean state.
                _nd_raw  = next_raw_obs[newly_done]              # (n_new, obs_flat)
                _nd_cr, _nd_nc = hlp_extract_car_rows(_nd_raw, K)
                _nd_perm = torch.stack([torch.randperm(K, device=device) for _ in range(n_new)])
                _nd_cr   = _nd_cr.gather(1, _nd_perm)
                _nd_vs   = (
                    torch.arange(K, device=device).unsqueeze(0) < _nd_nc.unsqueeze(1)
                ).gather(1, _nd_perm)
                _nd_hlp_in, _, _ = build_hlp_input(
                    _nd_raw, K, args, device, hlp_params[newly_done],
                    car_row_indices=_nd_cr, valid_slots=_nd_vs,
                    param_ranges=param_ranges,
                )
                with torch.no_grad():
                    bootstrap_values_snap[newly_done] = hlp_agent.get_value(_nd_hlp_in)

            sub_done_ever |= sub_done

            # [HLP-exp] Obs-change handler — identity-based mask remap, no new HLP decision.
            # Runs independently of sub_done; kind_id_snap updated here so future steps
            # detect the next change rather than re-firing on the same stale snapshot.
            if kind_id_changed.any():
                kc = kind_id_changed
                cr, nc = hlp_extract_car_rows(next_raw_obs[kc], K)
                cr = cr.gather(1, perm[kc])
                vs = (torch.arange(K, device=device).unsqueeze(0) < nc.unsqueeze(1)).gather(1, perm[kc])
                # Vectorised identity remap — no Python loops, no GPU synchronisations.
                j_new      = (cr - _OBJ_ROW_OFFSET).clamp(0, MAX_OBJECTS - 1)  # (n_kc, K)
                new_obj_ids = obs_idx_current[kc].gather(1, j_new)              # (n_kc, K)
                new_obj_ids[~vs] = -2   # -2: distinct from -1 (init sentinel) → no false matches
                # (n_kc, K_new, 1) == (n_kc, 1, K_init) → (n_kc, K, K) match table
                matches     = (new_obj_ids.unsqueeze(2) == init_slot_obj_ids[kc].unsqueeze(1))
                any_match   = matches.any(dim=2)                                # (n_kc, K)
                match_idx   = matches.float().argmax(dim=2)                     # (n_kc, K)
                matched_bits = car_mask_snap[kc].gather(1, match_idx)           # (n_kc, K)
                car_mask[kc] = torch.where(any_match, matched_bits, torch.zeros_like(matched_bits))
                car_row_indices[kc]     = cr
                valid_slots_current[kc] = vs
                kind_id_snap[kc]        = next_raw_obs.view(B, OBS_SIZE, OBS_FEATURES)[kc, :, 6]

            # Soft reinit for true LLP sub-episode boundaries (goal/miss/horizon/env_done).
            # HLP snap and init_slot_obj_ids are never touched here.
            # [HLP-goal] Filler goals stay RANDOM in both modes: reward accumulation is
            # already frozen for sub_done_ever envs, so these post-sub_done sub-episodes
            # carry no training signal — an extra HLP forward would buy nothing.
            if sub_done.any():
                kind_id_snap[sub_done] = next_raw_obs.view(B, OBS_SIZE, OBS_FEATURES)[sub_done, :, 6]
                x_m_s, y_m_s = llp_extract_player_xy_m(next_raw_obs[sub_done], y_max)
                gx, gy = llp_sample_goals(
                    x_m_s, y_m_s, args.goal_forward_m_min, args.goal_forward_m_max
                )
                goal_x[sub_done]           = gx
                goal_y[sub_done]           = gy
                subepisode_steps[sub_done] = 0

            # Block ends once every env has completed at least one sub-episode
            if sub_done_ever.all():
                break
        # ── END INNER LOOP ───────────────────────────────────────────────────────

        # ── BLOCK CLOSE ─────────────────────────────────────────────────── [HLP-delta]
        # valid_slots_snap (permuted bool mask) already computed at block init.
        # Cognitive cost gates to valid slots only — empty slots excluded (Fix A+B).
        # [HLP-goal] Proximity cost: closer chosen goals penalised more (linear in dy).
        # [HLP-param] The six parameterized coefficients are per-env θ columns
        # (hlp_params, constant within the block) — elementwise, not scalars.
        goal_cost = hlp_goal_proximity_cost(
            goal_dy_chosen, args.goal_forward_m_min, args.goal_forward_m_max,
            hlp_params[:, _P_PROX],
        )
        hlp_reward = (
              args.hlp_reach_reward        * goal_reached_this_block.float()
            + args.hlp_env_reward_coef     * accum_env_reward
            - hlp_params[:, _P_COLL]       * collision_count
            # [HLP-cogperstep] the toll is charged per active step of the block (Design Log 13),
            # so cog/collision does not depend on the goal-set block length.
            - hlp_params[:, _P_COG]        * (car_mask_snap * valid_slots_snap).sum(dim=1) * active_steps
            - goal_cost
            # [HLP-flag] flag outcomes drive goal selection; LLP is unaffected (flag-blind)
            - hlp_params[:, _P_FMISS]      * flag_miss_count
            + hlp_params[:, _P_FCOLL]      * flag_collect_count
            # [HLP-flag] goal-flag alignment: penalise the chosen goal's distance to the nearest
            # in-view flag (0 when no flag visible). HINGED at hlp_goal_flag_hinge_m: the penalty
            # reaches 0 once the goal is within that radius of a flag, instead of only at an
            # unreachable dist=0. WHY (measured 2026-07-15): with a plain linear penalty a
            # flag-present state could essentially NEVER score 0 (best achievable ~0.69 m goal-grid
            # floor), while a flag-ABSENT state scores 0 for free — so the critic learned flag
            # states are strictly worse (V(flag) -0.163 vs V(no-flag) -0.102), a standing
            # "flags are bad" gradient, and the reward never confirmed "I landed on it" (it only
            # ever said "you are far"). The hinge makes 0 ATTAINABLE with a flag present, removing
            # that asymmetry and turning an asymptote into a target zone. Penalty per metre; ≤ 0.
            - hlp_params[:, _P_GFLAG]      * torch.clamp(
                  goal_flag_dist_snap - args.hlp_goal_flag_hinge_m, min=0.0)
        )

        # [HLP-param] Envs whose episode ended this block get fresh θ at the NEXT
        # block init (lazy per-episode resample — see block init step 0).
        pending_param_reset = pending_param_reset | episode_done_this_block

        buffers["hlp_obs"][s]         = hlp_in_snap
        buffers["car_masks"][s]       = car_mask_snap
        buffers["hlp_logprobs"][s]    = hlp_logprob_snap
        buffers["hlp_values"][s]      = hlp_value_snap
        buffers["hlp_rewards"][s]     = hlp_reward
        buffers["hlp_dones"][s]       = episode_done_this_block.float()
        buffers["hlp_next_values"][s] = bootstrap_values_snap
        # [HLP-goal] goal action + conditioning offsets for PPO logprob recomputation
        buffers["goal_lanes"][s]      = goal_lane_snap
        buffers["goal_dy_bins"][s]    = goal_bin_snap
        buffers["goal_dxdy"][s]       = goal_dxdy_snap
        # [HLP-flag] per-block flag outcome counts (TensorBoard logging only)
        buffers["flag_misses"][s]     = flag_miss_count
        buffers["flag_collects"][s]   = flag_collect_count
        # [HLP-cogperstep] persist per-block active-step count for the scaled cog metric
        buffers["active_steps"][s]    = active_steps
        # [HLP-flag] per-block goal-flag distance (TensorBoard logging only)
        buffers["goal_flag_dists"][s] = goal_flag_dist_snap
        # [HLP-param] per-block θ (TensorBoard logging only — PPO reads θ from hlp_obs)
        buffers["hlp_params"][s]      = hlp_params
        # ── END BLOCK CLOSE ──────────────────────────────────────────────────────

    mean_sub_ep_len = sub_ep_len_sum / max(1, sub_ep_len_count)
    return (next_raw_obs, next_done, prev_actions, global_step, mean_sub_ep_len,
            obs_idx_current, hlp_params, pending_param_reset)
