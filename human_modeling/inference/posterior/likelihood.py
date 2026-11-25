"""
likelihood.py — the one expensive object, and the cheap arithmetic built on it.

Everything in hlp_v3 reduces to a single quantity: the log-likelihood of one episode's
observed actions at one parameter vector. It costs ~0.15 s. Everything else — the
posterior — is arithmetic on stored copies of that number, and costs nothing by comparison.

The split that makes this work, measured in hlp_v2 (implementation.md Entry 13):

  parameter-FREE, built once per episode (~5 s)    observations, kernel, emission tables
  parameter-DEPENDENT, per evaluation (~0.15 s)    the choice tables, and the forward walk

So a method needing thousands of evaluations pays the ~5 s once and 0.15 s per vector.
The parameter-free term was ~40 s until Entry 30 made the observation encoder run once per
(step, construal) instead of once per goal; MEASURED 39.4 s -> 5.0 s on the reference
episode. Nothing about the SPLIT changed -- only what the cached side costs.

Why there is no batching over parameter vectors: hlp_v2 Entry 15 measured it. Evaluating
a population of vectors in one network call gained a factor of 1.04 at two vectors and
ran out of memory at three on a 6 GB card, because only ~27% of an evaluation is the
network — the rest is the forward walk on the processor. Sequential evaluation is
therefore the honest cost model.

Public API: ModelStack, load_models, prepare_episode, log_likelihood, EpisodeInputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from human_modeling.inference.inference import forward_pass
from human_modeling.inference.inputs import HumanEpisode, build_episode_state_action_arrays
from human_modeling.inference.model.choice import (HEURISTIC_BOUNDS, HEURISTIC_NAMES,
                                                      THETA_NAMES, ChoiceModel)
from human_modeling.inference.model.emission import LLPPolicyModel, build_emission_tables
from human_modeling.inference.model.kernel import (EpisodeKernel, build_episode_kernel,
                                                load_hlp_metadata)

# | The two flag terms are pinned at 0 throughout hlp_v2 and hlp_v3: the checkpoint gives
# | them min == max == 0, so they are not free parameters and never enter a posterior.
PINNED_THETA = {"hlp_flag_miss_penalty": 0.0, "hlp_flag_collect_reward": 0.0}


@dataclass
class ModelStack:
    """
    The loaded models and checkpoint metadata, bundled so one load serves a whole run.

    :ivar choice: Goal/construal choice model; the only place parameters enter.
    :vartype choice: ChoiceModel
    :ivar llp: Frozen action model, used to build the emission tables.
    :vartype llp: LLPPolicyModel
    :ivar meta: Checkpoint metadata (``hlp_metadata.json`` contents).
    :vartype meta: dict[str, Any]
    :ivar bounds: Trained range per fitted coefficient; also the support of the prior.
    :vartype bounds: dict[str, tuple[float, float]]
    """

    choice: ChoiceModel
    llp:    LLPPolicyModel
    meta:   dict[str, Any]
    bounds: dict[str, tuple[float, float]]

    @property
    def y_max_m(self) -> float:
        """Track length in metres, the y normalisation constant."""
        return float(self.meta["y_max"])

    @property
    def env_reward_coef(self) -> float:
        """Action-model environment-reward blend coefficient from the checkpoint."""
        return float(self.meta["llp_env_reward_coef"])

    @property
    def lo_hi(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-coefficient lower and upper bounds, in ``THETA_NAMES`` order."""
        lo = np.array([self.bounds[n][0] for n in THETA_NAMES], dtype=np.float64)
        hi = np.array([self.bounds[n][1] for n in THETA_NAMES], dtype=np.float64)
        return lo, hi

    def lo_hi_with_heuristics(self) -> tuple[np.ndarray, np.ndarray]:
        """
        The 8-coefficient box: the trained theta ranges, then the heuristic weight bounds.

        Theta's bounds come from the checkpoint, which trained inside them; the weights were
        never trained, so theirs are calibrated from the checkpoint's own behaviour and live
        in :data:`HEURISTIC_BOUNDS` (implementation.md Entry 33). Both halves are handed to
        the estimator as one box so its projection covers the weights too.

        :return: ``(lo, hi)``, each ``(8,)``, in ``FIT_NAMES`` order.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        lo, hi = self.lo_hi
        bound = np.array([HEURISTIC_BOUNDS[n] for n in HEURISTIC_NAMES], dtype=np.float64)
        return np.concatenate([lo, -bound]), np.concatenate([hi, bound])


def load_models(
    checkpoint_dir: str,
    model_filename: str = "hlp_model_final.pt",
    device: str | None = None,
) -> ModelStack:
    """
    Load the choice and action models plus checkpoint metadata.

    :param checkpoint_dir: Trained-policy directory.
    :type checkpoint_dir: str
    :param model_filename: HLP weights filename inside it.
    :type model_filename: str
    :param device: Torch device, or None to pick CUDA when available.
    :type device: str | None
    :return: The loaded bundle.
    :rtype: ModelStack
    """
    dev  = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_hlp_metadata(checkpoint_dir)
    return ModelStack(
        choice=ChoiceModel(checkpoint_dir, device=dev, model_file=model_filename),
        llp=LLPPolicyModel(checkpoint_dir, device=dev),
        meta=meta,
        bounds={c["name"]: (float(c["min"]), float(c["max"]))
                for c in meta["hlp_param_config"]},
    )


@dataclass
class EpisodeInputs:
    """
    One episode's parameter-free scaffolding, built once and reused by every evaluation.

    :ivar s: Flat observations, ``(N, obs_dim)``.
    :vartype s: np.ndarray
    :ivar kernel: Candidate sets, terminations and relabelings for this episode.
    :vartype kernel: EpisodeKernel
    :ivar tables: Per-step emission tables, each ``(G, M_t)``.
    :vartype tables: list[np.ndarray]
    :ivar n_frames: Trajectory steps in this episode.
    :vartype n_frames: int
    :ivar label: Identifier carried through to results (subject and game).
    :vartype label: str
    :ivar full_visibility: True when the tables were built with every car slot visible, so
        each step has ONE construal column instead of one per attention pattern. Both
        likelihood paths read this flag, so the choice tables always match the emission
        tables they are combined with (implementation.md Entry 36).
    :vartype full_visibility: bool
    """

    s:               np.ndarray
    kernel:          EpisodeKernel
    tables:          list[np.ndarray]
    n_frames:        int
    label:           str = ""
    full_visibility: bool = False


def prepare_episode(
    stack: ModelStack,
    episode: HumanEpisode,
    label: str = "",
    full_visibility: bool = False,
) -> EpisodeInputs:
    """
    Build everything for one episode that does not depend on the parameters.

    This is the expensive step: ~5 s per episode since Entry 30 (~40 s before it). It is
    paid once; every later evaluation reuses the result.

    :param stack: Loaded models and metadata.
    :type stack: ModelStack
    :param episode: The episode to prepare.
    :type episode: HumanEpisode
    :param label: Identifier to carry through to results.
    :type label: str
    :param full_visibility: Build the tables with every car slot visible (one construal per
        step) for the full-visibility model, instead of one column per attention pattern.
        A model of the other kind cannot use the result:
        :class:`~human_modeling.inference.variants.VariantGradContext` refuses the mismatch.
    :type full_visibility: bool
    :return: The parameter-free scaffolding.
    :rtype: EpisodeInputs
    """
    s, a, a_prev = build_episode_state_action_arrays(episode)
    kernel = build_episode_kernel(s, stack.meta, stack.choice.lane_centers(),
                                  device=stack.choice.device)
    # | [full-visibility] The emission builder already had the all-visible branch; this
    # | only passes the switch through and records it on the result.
    tables = build_emission_tables(s, a, a_prev, kernel, stack.llp, stack.y_max_m,
                                   stack.env_reward_coef, full_visibility=full_visibility)
    return EpisodeInputs(s=s, kernel=kernel, tables=tables,
                         n_frames=int(kernel.N), label=label,
                         full_visibility=full_visibility)


def theta_dict(vec: np.ndarray) -> dict[str, float]:
    """
    Turn a coefficient vector into the dict the choice model expects.

    :param vec: Values in ``THETA_NAMES`` order. A longer vector carrying the heuristic
        weights is accepted and its tail ignored here — ``zip`` stops at ``THETA_NAMES``, and
        :func:`weight_dict` is what reads the rest.
    :type vec: np.ndarray
    :return: Named coefficients, with the two pinned flag terms added.
    :rtype: dict[str, float]
    """
    out = {n: float(v) for n, v in zip(THETA_NAMES, np.asarray(vec, dtype=np.float64))}
    out.update(PINNED_THETA)
    return out


def weight_dict(vec: np.ndarray) -> dict[str, float]:
    """
    Turn a fitted vector into the heuristic weights the choice model expects.

    Prefer this over reading the tail of the vector by hand: it is the single place that
    decides what a 4-vector means, and it means "every weight zero" — which is exactly the
    behaviour this package had before the heuristics were fitted at all.

    :param vec: Values in ``THETA_NAMES`` order, optionally followed by the weights in
        ``HEURISTIC_NAMES`` order.
    :type vec: np.ndarray
    :return: ``{heuristic name: weight}``, empty when the vector carries theta only.
    :rtype: dict[str, float]
    :raises ValueError: If the vector is neither length.
    """
    values = np.asarray(vec, dtype=np.float64)
    n_theta, n_heur = len(THETA_NAMES), len(HEURISTIC_NAMES)
    if values.shape[-1] == n_theta:
        return {}
    if values.shape[-1] == n_theta + n_heur:
        return {n: float(v) for n, v in zip(HEURISTIC_NAMES, values[n_theta:])}
    raise ValueError(f"fitted vector must have {n_theta} or {n_theta + n_heur} entries, "
                     f"got {values.shape[-1]}")


def log_likelihood(stack: ModelStack, inputs: EpisodeInputs, vec: np.ndarray) -> float:
    """
    Log-probability of one episode's observed actions at one coefficient vector.

    The expensive primitive. Marginalises over every possible subtask sequence — goals
    and construals both — exactly as hlp_v2 does; that averaging is the intended model,
    not an approximation (design plan, Revision 18).

    This is not only the reference implementation: it is what ``fit_gradient.fit_posterior``
    evaluates at every lattice point, so it produces the posterior summaries of every fit.
    It therefore has to understand the heuristic weights too — a torch-only port would leave
    the gradient finding a mode the lattice then described with the heuristics switched off
    (implementation.md Entry 34).

    :param stack: Loaded models and metadata.
    :type stack: ModelStack
    :param inputs: The episode's parameter-free scaffolding.
    :type inputs: EpisodeInputs
    :param vec: Coefficient values in ``THETA_NAMES`` order, or with the heuristic weights
        appended in ``HEURISTIC_NAMES`` order. A 4-vector pins every weight to 0.
    :type vec: np.ndarray
    :return: Log-likelihood of the observed actions.
    :rtype: float
    :raises ValueError: If heuristic weights are given for a full-visibility episode, where
        there is no attention choice for them to reweight.
    """
    # | [full-visibility] With every car visible there is no construal choice, so the choice
    # | model returns the goal log-probabilities alone, one column per step. The heuristic
    # | weights only reweight construals; the numpy branch would drop them silently, so a
    # | weighted vector is refused here instead.
    weights = weight_dict(vec)
    if inputs.full_visibility and weights:
        raise ValueError("heuristic weights have no effect on a full-visibility episode; "
                         "pass the 4-coefficient theta vector")
    log_rho = stack.choice.log_rho(inputs.s, inputs.kernel, theta_dict(vec), weights,
                                   full_visibility=inputs.full_visibility)
    return forward_pass(log_rho, inputs.tables, inputs.kernel)
