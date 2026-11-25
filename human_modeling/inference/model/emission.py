"""VENDORED. Lineage: human_modeling/hlp_v2/model/emission.py -> hlp_v3 -> here (2026-09-14,
implementation.md Entry 28). Both earlier packages are retired and deleted; their
design logs are in human_modeling/version_history.md. This package imports from no
other inference version and must not start.

emission.py — b_t(τ): action probability under every subtask, plus the cached table.

``b_t(τ=(g,m))`` = probability of the human's observed action ``a_t`` if the driver's
subtask is τ: the frozen LLP scored on the m-masked observation with g's absolute dot as
the goal, ``prev_action`` from data, and clock input 0 (implementation.md §8). The full
per-episode table contains NO fitted parameter, so it is computed once per episode and
reused by every optimizer evaluation and by all inference passes (implementation.md §7).

WHAT ENTRY 30 CHANGED, AND WHY THE TABLE BUILD IS CHEAPER

  The original build pushed every (step, mask) observation through the WHOLE network once per
  goal -- ``masked.repeat_interleave(G)`` then a full forward -- so the expensive observation
  encoder ran 96 times on identical input. The encoder is expensive because it applies a small
  network to EVERY tile and object in the observation separately and then max-pools over them
  (``MultiEncoder``, drive_env/interface_gym.py), which is far more arithmetic than the few
  fully-connected layers that follow. Entry 9e measured exactly that on the reference episode:
  1,644,576 encoder invocations against 17,131 distinct inputs, and 97% of the network time.

  The encoder does not see the goal. ``GoalConditionedExtractor.forward`` (hrl_llp_core.py)
  runs ``self.encoder`` on the BASE observation and concatenates the goal columns AFTER it. So
  one encoding serves all 96 goals, and only the small actor head ("head" = the three
  fully-connected layers after the encoder) is repeated. This module therefore encodes once per
  (step, mask) and broadcasts that encoding across the goals.

Self-containment note (see CONTEXT.md): :class:`LLPPolicyModel` is COPIED from
``human_modeling/llp/utils/llp_likelihood_model.py`` (with the ``LLPBehaviorParams``
dataclass dependency replaced by a plain ``env_reward_coef`` float, the only field this
package uses). ``apply_car_mask`` and the LLP building blocks are imported from
``modules/base_models`` (allowed).

Public API: :class:`LLPPolicyModel`, :func:`build_emission_tables`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch.distributions.categorical import Categorical

from drive_env.interface_gym import MultiEncoder
from modules.base_models.SP.hrl_llp_core import (
    _ACTION_LAT,
    _ACTION_LON,
    Agent,
    GoalConditionedExtractor,
)
from modules.base_models.SSP.hlp_core import apply_car_mask

from .kernel import OBS_FEATURES, OBS_SIZE, EpisodeKernel

DEFAULT_CHUNK_ROWS = 6144   # LLP rows per forward call — sized so transient activation
                            # memory fits a 6 GB GPU (implementation.md §7, measured)

# | [encoder-cache] Lateral/longitudinal encoding of the previous action — the two columns
# | llp_augment_obs appends after ts_norm. IMPORTED, not copied: the cached path has to build
# | these columns itself instead of re-entering llp_augment_obs, and a local copy would
# | silently go stale if the action ordering ever changed.
_LAT_NP = _ACTION_LAT.numpy()
_LON_NP = _ACTION_LON.numpy()

# | The four columns llp_augment_obs appends after the goal pair: ts_norm, lat_prev, lon_prev,
# | env_reward_coef. Named because the cached path has to size its own broadcast by it.
N_CONTEXT_COLUMNS = 4


class _EnvSpec:
    """Minimal stand-in for a gym VectorEnv, exposing only what ``Agent.__init__`` reads."""

    def __init__(self, n_actions: int) -> None:
        """Build a fake env-spec carrying just an action count.

        :param n_actions: Number of discrete actions the policy's actor head outputs.
        :type n_actions: int
        """
        self.single_observation_space = Box(-float("inf"), float("inf"), shape=(1,))
        self.single_action_space = Discrete(n_actions)


# ===========================================================================
# 1. The frozen action model
# ===========================================================================

class LLPPolicyModel:
    """Frozen LLP policy loaded from a checkpoint, used to score human actions.

    [encoder-cache] The network is exposed in its two halves (:meth:`encode`,
    :meth:`head_log_probs`), because the expensive half does not depend on the goal and so can
    be run once and reused — see the module docstring.

    :param checkpoint_dir: Directory containing ``hrl_llp_metadata.json`` and weights
        (the HLP run directories bundle both networks).
    :type checkpoint_dir: str | Path
    :param model_filename: Weights file within ``checkpoint_dir``.
    :type model_filename: str
    :param device: Torch device; defaults to CUDA if available.
    :type device: str | torch.device | None
    """

    N_ACTIONS = 9

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model_filename: str = "llp_model_final.pt",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        with open(self.checkpoint_dir / "hrl_llp_metadata.json", "r", encoding="utf-8") as f:
            self.metadata: dict[str, Any] = json.load(f)

        self.device = (
            torch.device(device) if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.sub_episode_horizon: int = int(self.metadata["sub_episode_horizon"])

        encoder_params = self.metadata["encoder_params"]
        multi_encoder = MultiEncoder(
            tile_obs_size=int(encoder_params["tile_obs_size"]),
            obs_features=int(encoder_params["obs_features"]),
            ego_hidden=int(encoder_params["ego"]),
            tile_hidden=int(encoder_params["tile"]),
            object_hidden=int(encoder_params["object"]),
            device=self.device,
            visible_width=float(encoder_params["visible_width"]),
            visible_height=float(encoder_params["visible_height"]),
            max_speed=float(encoder_params["max_speed"]),
        )
        feature_extractor = GoalConditionedExtractor(
            multi_encoder=multi_encoder,
            goal_x_scale=float(encoder_params["goal_x_scale"]),
            goal_y_scale=float(encoder_params["goal_y_scale"]),
        )
        self.agent = Agent(
            envs=_EnvSpec(self.N_ACTIONS),
            hidden_size=int(self.metadata.get("hidden_size", 256)),
            feature_extractor=feature_extractor,
        ).to(self.device)

        checkpoint = torch.load(
            self.checkpoint_dir / model_filename, map_location=self.device, weights_only=False
        )
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        self.agent.load_state_dict(state_dict)
        self.agent.requires_grad_(False).eval()

        # | [encoder-cache] Handles onto the two halves the cached path drives separately. Both
        # | are public attributes of the frozen agent, so nothing in modules/base_models
        # | changes. n_context is 6 on every checkpoint this package targets: the goal pair
        # | plus the four columns N_CONTEXT_COLUMNS counts.
        self.extractor = feature_extractor
        self.encoder   = feature_extractor.encoder
        self.n_context = int(feature_extractor.n_context)

    # | ----- The two halves -----
    def encode(self, base_obs: torch.Tensor) -> torch.Tensor:
        """Run ONLY the expensive encoder, on observations whose ego x/y are already zeroed.

        [encoder-cache] This is the half that does not depend on the goal, so its result is
        cached and reused across all goals for the same (step, mask).

        :param base_obs: Flat observations, ``(B, OBS_SIZE*OBS_FEATURES)``, ego columns
            0 and 1 already set to 0 exactly as ``llp_augment_obs`` does.
        :type base_obs: torch.Tensor
        :return: Encoder output, ``(B, encoder_output_size)``.
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            return self.encoder(base_obs)

    def head_log_probs(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``log P(action)`` from an ALREADY-assembled feature row.

        ``features`` is what ``GoalConditionedExtractor.forward`` returns: the encoder output
        followed by the SCALED goal columns and the unscaled context columns. The two lines
        below are exactly ``Agent.get_action_and_value``'s actor path, with the value head and
        the entropy — neither of which the emission table uses — left out.

        MEASURED, and the reason ``Categorical`` appears here rather than the shorter
        ``log_softmax(...).gather(...)``: the two spellings are not numerically identical in
        float32. They normalize the logits differently and drift up to 7.6e-06 apart at the
        logit magnitudes this policy produces, which would show up as a disagreement between
        the cached and uncached emission tables — an artifact of the SPELLING, not of the
        caching. Matching the reference path exactly makes the two bit-identical, so the
        equality gate measures the change it is supposed to measure.

        :param features: ``(B, extractor_output_size)``.
        :type features: torch.Tensor
        :param actions: Scored action index per row, ``(B,)`` long.
        :type actions: torch.Tensor
        :return: Log-probabilities, ``(B,)``.
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            return Categorical(logits=self.agent.actor(features)).log_prob(actions)


# ===========================================================================
# 2. The emission tables
# ===========================================================================

def _masked_observations(
    s3: torch.Tensor,
    kernel: EpisodeKernel,
    rows_t: torch.Tensor,
    vs_t: torch.Tensor,
    t: int,
    mask_ints: np.ndarray,
) -> torch.Tensor:
    """The construal-masked observations at one step, one row per enumerated mask.

    :param s3: Whole episode reshaped ``(N, OBS_SIZE, OBS_FEATURES)``.
    :type s3: torch.Tensor
    :param kernel: The episode's kernel (used for the slot count).
    :type kernel: EpisodeKernel
    :param rows_t: Car row indices, ``(N, K)``.
    :type rows_t: torch.Tensor
    :param vs_t: Slot validity, ``(N, K)``.
    :type vs_t: torch.Tensor
    :param t: Trajectory step.
    :type t: int
    :param mask_ints: Enumerated construal labels, ``(M_t,)``.
    :type mask_ints: np.ndarray
    :return: ``(M_t, OBS_SIZE, OBS_FEATURES)``.
    :rtype: torch.Tensor
    """
    m_t = mask_ints.shape[0]
    bits = torch.as_tensor(
        (mask_ints[:, None] >> np.arange(kernel.K)[None, :]) & 1, dtype=torch.float32)
    return apply_car_mask(
        s3[t].unsqueeze(0).expand(m_t, -1, -1),
        rows_t[t].unsqueeze(0).expand(m_t, -1),
        bits,
        vs_t[t].unsqueeze(0).expand(m_t, -1),
    )


def _context_columns(
    a_prev_t: int, env_reward_coef: float, device: torch.device,
) -> torch.Tensor:
    """[encoder-cache] The context row at one step: every column after the goal pair.

    Layout, matching ``llp_augment_obs``: ``ts_norm`` (fixed at 0, the clock convention this
    package has used since hlp_v2, implementation.md §8), ``lat_prev``, ``lon_prev``,
    ``env_reward_coef``. It varies with neither the mask nor the goal, which is why one row per
    step suffices and the cached path can broadcast it.

    :param a_prev_t: Previous chosen action at this step.
    :type a_prev_t: int
    :param env_reward_coef: LLP conditioning coefficient.
    :type env_reward_coef: float
    :param device: Device for the result.
    :type device: torch.device
    :return: ``(N_CONTEXT_COLUMNS,)`` float32.
    :rtype: torch.Tensor
    """
    return torch.tensor(
        [0.0, float(_LAT_NP[a_prev_t]), float(_LON_NP[a_prev_t]), float(env_reward_coef)],
        dtype=torch.float32, device=device)


def _build_cached(
    s3: torch.Tensor,
    kernel: EpisodeKernel,
    rows_t: torch.Tensor,
    vs_t: torch.Tensor,
    a: np.ndarray,
    a_prev: np.ndarray,
    llp: LLPPolicyModel,
    y_max: float,
    env_reward_coef: float,
    full_visibility: bool,
    chunk_rows: int,
) -> list[np.ndarray]:
    """[encoder-cache] The production path: encode once per (step, mask), reuse across goals.

    Stages, per trajectory step:
      1. build the masked observations (``M_t`` of them) and read the ego position off them;
      2. zero the ego columns and run the ENCODER once — the only expensive call;
      3. assemble ``(M_t, G)`` head rows from the cached encoding, the scaled goal offsets and
         the step's context row;
      4. buffer the rows, and flush through the actor head in ``chunk_rows`` slices.

    Arguments mirror :func:`build_emission_tables`.

    :return: Per-step tables ``log b_t`` of shape ``(G, M_t)``.
    :rtype: list[np.ndarray]
    """
    N, G, K = kernel.N, kernel.G, kernel.K
    device = llp.device
    # | get_buffer, not plain attribute access: `goal_scale` is a registered buffer, and
    # | reading it off the module directly is typed as "tensor or submodule", which the
    # | division below cannot accept.
    goal_scale = llp.extractor.get_buffer("goal_scale").to(device)       # (2,)
    all_ones = (1 << K) - 1
    tables: list[np.ndarray] = [np.empty((G, 0))] * N

    buf_feat: list[torch.Tensor] = []
    buf_act: list[torch.Tensor] = []
    spans: list[tuple[int, int]] = []      # (step, M_t) per buffered block, in order
    buffered = 0

    def _flush() -> None:
        """Run the actor head over every buffered row and scatter into `tables`."""
        nonlocal buffered
        if not spans:
            return
        feats = torch.cat(buf_feat, dim=0)
        acts = torch.cat(buf_act, dim=0)
        out = np.empty(feats.shape[0], dtype=np.float64)
        for lo in range(0, feats.shape[0], chunk_rows):
            hi = min(lo + chunk_rows, feats.shape[0])
            out[lo:hi] = llp.head_log_probs(feats[lo:hi], acts[lo:hi]).double().cpu().numpy()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pos = 0
        for step, m_t in spans:
            block = out[pos: pos + m_t * G].reshape(m_t, G)   # rows are [m][g]-ordered
            tables[step] = block.T.copy()                     # -> (G, M_t)
            pos += m_t * G
        buf_feat.clear(); buf_act.clear(); spans.clear()
        buffered = 0

    for t in range(N):
        mask_ints = np.array([all_ones]) if full_visibility else kernel.masks_at(t)
        m_t = mask_ints.shape[0]

        # | 1. Masked observations and the ego position they carry. The position is read
        # |    BEFORE the ego columns are zeroed, exactly as llp_augment_obs reads it.
        masked = _masked_observations(s3, kernel, rows_t, vs_t, t, mask_ints)
        flat = masked.reshape(m_t, OBS_SIZE * OBS_FEATURES).to(device)   # (M_t, obs_dim)
        player_x = flat[:, 0]                                            # (M_t,) metres
        player_y = flat[:, 1] * y_max                                    # (M_t,) metres

        # | 2. Zero the ego columns exactly as llp_augment_obs does, then encode ONCE.
            # | The clone is MANDATORY: MultiEncoder.forward normalizes its input IN PLACE
            # | (drive_env/interface_gym.py), so handing it `flat` would corrupt the very
            # | tensor the ego position was just read from.
        enc_in = flat.clone()
        enc_in[:, 0] = 0.0
        enc_in[:, 1] = 0.0
        enc = llp.encode(enc_in)                                         # (M_t, E)

        # | 3. Goal offsets, scaled the way GoalConditionedExtractor.forward scales them, and
        # |    the context row, which varies with neither the mask nor the goal.
        gx, gy = kernel.goal_xy(t)                                       # (G,), (G,)
        gx_t = torch.as_tensor(gx, dtype=torch.float32, device=device)   # (G,)
        gy_t = torch.as_tensor(gy, dtype=torch.float32, device=device)   # (G,)
        goal = torch.stack([gx_t[None, :] - player_x[:, None],
                            gy_t[None, :] - player_y[:, None]], dim=-1)  # (M_t, G, 2)
        goal = goal / goal_scale
        ctx = _context_columns(int(a_prev[t]), env_reward_coef, device)  # (N_CONTEXT_COLUMNS,)

        # | NOTE: this allocation is not bounded by chunk_rows — it materializes one whole
        # | step, at worst 2^K masks x G goals rows. On the checkpoints this package targets
        # | that is 64 x 96 = 6,144 rows of 582 float32 columns, about 14 MB, which is one
        # | chunk's worth and fits the 6 GB development card with room to spare. There is
        # | therefore no out-of-memory guard here; inference_v5 needs one only because its
        # | delay axis multiplies the same allocation by 16.
        feats = torch.cat([
            enc[:, None, :].expand(m_t, G, enc.shape[1]),
            goal,
            ctx[None, None, :].expand(m_t, G, N_CONTEXT_COLUMNS),
        ], dim=-1).reshape(m_t * G, -1)                                  # (M_t*G, E+2+4)
        acts = torch.full((m_t * G,), int(a[t]), dtype=torch.long, device=device)

        # | 4. Flush BEFORE appending when this step would overrun, so the buffer never holds
        # |    more than one oversized step's worth of rows.
        if buffered and buffered + feats.shape[0] > chunk_rows:
            _flush()
        buf_feat.append(feats)
        buf_act.append(acts)
        spans.append((t, m_t))
        buffered += feats.shape[0]
        if buffered >= chunk_rows:
            _flush()
    _flush()
    return tables


def build_emission_tables(
    s: np.ndarray,
    a: np.ndarray,
    a_prev: np.ndarray,
    kernel: EpisodeKernel,
    llp: LLPPolicyModel,
    y_max: float,
    env_reward_coef: float,
    full_visibility: bool = False,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> list[np.ndarray]:
    """The per-episode emission table: ``log b_t`` for every step and subtask.

    θ-free — call once per episode per variant-visibility mode, reuse everywhere.

    :param s: Flat observations, ``(N, obs_dim)``.
    :type s: np.ndarray
    :param a: Human actions, ``(N,)``.
    :type a: np.ndarray
    :param a_prev: Previous actions, ``(N,)``.
    :type a_prev: np.ndarray
    :param kernel: The episode's kernel (mask enumeration, slot metadata, goal grid).
    :type kernel: EpisodeKernel
    :param llp: Loaded frozen LLP.
    :type llp: LLPPolicyModel
    :param y_max: Map height (metres).
    :type y_max: float
    :param env_reward_coef: LLP conditioning coefficient (fixed).
    :type env_reward_coef: float
    :param full_visibility: If True, single all-ones construal per step (variant C).
    :type full_visibility: bool
    :param chunk_rows: Max LLP rows per forward call (GPU-memory knob).
    :type chunk_rows: int
    :return: Per-step tables ``log b_t`` of shape ``(G, M_t)``.
    :rtype: list[np.ndarray]
    """
    s3 = torch.as_tensor(s.reshape(kernel.N, OBS_SIZE, OBS_FEATURES), dtype=torch.float32)
    rows_t = torch.as_tensor(kernel.car_row_indices)
    vs_t = torch.as_tensor(kernel.valid_slots)
    return _build_cached(s3, kernel, rows_t, vs_t, a, a_prev, llp, y_max, env_reward_coef,
                         full_visibility, chunk_rows)
