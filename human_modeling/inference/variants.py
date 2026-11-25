"""
variants.py -- the model variants the V4 fit compares, and the one mapping that lets them
share every estimator this package already has.

WHY A VARIANT IS A MAPPING, NOT A NEW LIKELIHOOD
    The likelihood accepts exactly two vector lengths: 4 (theta) or 8 (theta then the four
    heuristic weights, implementation.md Entry 34). A variant that pins a coefficient -- the
    no-cognitive-cost model, or the all-heuristics model without the mean-distance heuristic
    -- fits FEWER free values but still has to hand the likelihood a full vector. So each
    variant is a list of free coefficient names plus the pinned values for the rest, and
    :meth:`FitVariant.embed` turns a free vector into the full one. The gradient ascent,
    the curvature box and the posterior lattice all operate on the free vector and never
    learn that anything was pinned; the likelihood never learns that anything was free.

THE FOUR CHAINED VARIANTS (user, 2026-09-23), in the order they are fitted -- a NESTED CHAIN,
C in B in A' in A (Entry 35 addendum 2, 2026-09-24) -- plus one sanity-check variant outside it
    C_no_cognitive_cost  theta with `hlp_cognitive_cost_coef` pinned at 0, its trained lower
                         bound (3 free), no heuristics. The policy still chooses construals
                         and the likelihood still marginalises over them; only the attention
                         cost is switched off. This is NOT the AAAI `C_no_construals`, which
                         forced every car visible; in September V4's likelihood had no
                         full-visibility path and the user chose this reading over porting
                         one. That model now exists as `C_full_visibility` below. Starts at
                         the box midpoint.
    B_no_heuristics      theta only (4 free). The shipped fit: the production read-out of
                         every V4 result so far. Started TWICE -- from the midpoint (the
                         shipped start, so the result is bit-identical to the shipped fit
                         wherever that start wins) and from C's solution -- keeping the
                         higher log-likelihood.
    A_no_mean_distance   theta + three weights, `mean_vehicle_distance` pinned at 0 (7 free),
                         warm-started from B. The two distance heuristics are collinear at
                         r = 0.987 on this checkpoint (Entry 34's joint fit), so this asks
                         whether the second one buys anything.
    A_full               theta + all four heuristic weights (8 free), warm-started from
                         A_no_mean_distance.
    C_full_visibility    the same 3 free theta and the same cognitive-cost pin as C, but with
                         every car slot the network observes (the 6 nearest cars,
                         `hlp_num_car_slots`) forced visible: no attention choice exists, so
                         nothing is marginalised (user, 2026-11-10; Entry 36). A SANITY CHECK,
                         not a finding: it tests that removing the construal choice does not
                         move the fit away from C's. It is NOT nested in B -- B cannot put all
                         its probability on the all-visible pattern -- so it sits outside the
                         chain and out of `NESTED_PAIRS`, nothing warm-starts from it, and it
                         needs its own prepared episode (`prepare_episode(...,
                         full_visibility=True)`), which `VariantGradContext` enforces. The
                         pinned cognitive cost still matters here: theta is an input to the
                         goal head, so the pin changes which goals the policy predicts.
                         Starts at the box midpoint.

WARM STARTS, AND WHY THE CHAIN
    A warm start guarantees only that a fit ends no worse than where it began; it does not
    find a model's best optimum on its own. But a model that CONTAINS another can always
    reproduce the smaller model's solution exactly (zero for a dropped weight, the pinned
    value for a pinned theta), so starting each model from the one below it makes
    "more parameters never fit worse" true in the code, not only in theory. MEASURED before
    the chain (Entry 35 addendum 2): with B started from the midpoint only, the smaller C
    beat it on 3 of 23 episodes by up to 19.9 nats -- a BIC swing of ~40, enough to flip a
    comparison. Started cold from the box midpoint, the 8-parameter ascent converged
    somewhere WORSE than B on 3 of 7 agents (Entry 34), which is why the heuristic variants
    take a single warm start rather than two. B is the one model with two sensible starts:
    its midpoint start beat C's corner solution on 20 of 23 episodes, so both are tried.
    `warm_start_from` names the single parent; `extra_starts` names further solutions to try
    beside the midpoint, keeping the best. The synthetic recovery study warm-starts its
    8-parameter fit from B, not from A', on purpose (its `--paired-arm` default).

Public API: :class:`FitVariant`, :data:`VARIANTS`, :data:`FIT_ORDER`,
:data:`ATTENTION_CHOOSING_VARIANTS`, :func:`get_variant`, :func:`ordered`,
:class:`VariantGradContext`.

Typical usage::

    variant = get_variant("A_no_mean_distance")
    lo, hi = variant.lo_hi(stack)                          # 1. the box over the FREE values
    ctx = VariantGradContext(EpisodeGradContext(stack, inp, with_heuristics=True), variant)
    mode, info = fit_mode(ctx, lo, hi, start=variant.start_from(b_mode))   # 2. free vector
    full = variant.embed(mode)                             # 3. the 8-vector the model reads
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from human_modeling.inference.model.choice import (FIT_NAMES, HEURISTIC_NAMES,
                                                      THETA_NAMES)

# | Pinned value for every heuristic weight a variant leaves out: 0 is the off switch
# | (Entry 33; alpha is not reinstated). Theta pins are named per variant instead, because
# | "off" means a different number for each theta coefficient.
HEURISTIC_OFF = 0.0


@dataclass(frozen=True)
class FitVariant:
    """
    One model variant: which coefficients are fitted, and what the rest are held at.

    :param name: Stable identifier used in file names and CSV columns.
    :type name: str
    :param display: Human-readable label for figure axes.
    :type display: str
    :param free_theta: The fitted theta coefficients, a subset of :data:`THETA_NAMES`.
    :type free_theta: tuple[str, ...]
    :param free_heuristics: The fitted heuristic weights, a subset of
        :data:`HEURISTIC_NAMES`; empty means the variant has no heuristics at all.
    :type free_heuristics: tuple[str, ...]
    :param pinned_theta: Fixed value for every theta coefficient NOT in ``free_theta``.
    :type pinned_theta: dict[str, float]
    :param warm_start_from: Name of the variant whose solution seeds this ascent, or None
        to start from the box midpoint. See the module docstring.
    :type warm_start_from: str | None
    :param extra_starts: Names of variants whose solutions are ALSO tried as starting points,
        beside the midpoint (when ``warm_start_from`` is None) or the parent's solution; the
        ascent with the highest log-likelihood is kept. Each extra start costs one more
        ascent (~23 s for B on the L40S).
    :type extra_starts: tuple[str, ...]
    :param full_visibility: True when every car slot is forced visible, so the model makes no
        attention choice and its likelihood is evaluated on a prepared episode built with
        ``prepare_episode(..., full_visibility=True)``. Ex: ``C_full_visibility``.
    :type full_visibility: bool
    """

    name: str
    display: str
    free_theta: tuple[str, ...]
    free_heuristics: tuple[str, ...] = ()
    pinned_theta: dict[str, float] = field(default_factory=dict)
    warm_start_from: str | None = None
    extra_starts: tuple[str, ...] = ()
    full_visibility: bool = False

    def __post_init__(self) -> None:
        """Refuse a definition the embedding could not honour."""
        unknown = (set(self.free_theta) - set(THETA_NAMES)) | (
            set(self.free_heuristics) - set(HEURISTIC_NAMES))
        if unknown:
            raise ValueError(f"{self.name}: unknown coefficients {sorted(unknown)}")
        # | [full-visibility] The heuristic weights only reweight the attention choice, which a
        # | full-visibility model does not make, so a weight there would be a parameter with
        # | no effect on the likelihood.
        if self.full_visibility and self.free_heuristics:
            raise ValueError(f"{self.name}: a full-visibility variant makes no attention "
                             f"choice, so it cannot fit heuristic weights")
        missing = set(THETA_NAMES) - set(self.free_theta) - set(self.pinned_theta)
        if missing:
            raise ValueError(f"{self.name}: theta {sorted(missing)} is neither free nor pinned")
        overlap = set(self.free_theta) & set(self.pinned_theta)
        if overlap:
            raise ValueError(f"{self.name}: theta {sorted(overlap)} is both free and pinned")

    # -- shape -------------------------------------------------------------------------

    @property
    def uses_heuristics(self) -> bool:
        """True when the full vector is the 8-form; False means the plain 4-vector."""
        return bool(self.free_heuristics)

    @property
    def full_names(self) -> tuple[str, ...]:
        """Names of the full vector the likelihood receives, in its order."""
        return FIT_NAMES if self.uses_heuristics else THETA_NAMES

    @property
    def free_names(self) -> tuple[str, ...]:
        """Names of the free vector, in the full vector's order (theta first)."""
        return tuple(n for n in self.full_names
                     if n in self.free_theta or n in self.free_heuristics)

    @property
    def free_theta_names(self) -> tuple[str, ...]:
        """The free theta names in :data:`THETA_NAMES` order (the posterior is over these)."""
        return tuple(n for n in THETA_NAMES if n in self.free_theta)

    @property
    def n_free(self) -> int:
        """Free-parameter count -- the ``k`` in BIC."""
        return len(self.free_names)

    @property
    def n_free_theta(self) -> int:
        """How many leading entries of the free vector are theta."""
        return len(self.free_theta_names)

    def pinned_value(self, name: str) -> float:
        """
        The fixed value of a coefficient this variant does not fit.

        :param name: A coefficient in :data:`FIT_NAMES`.
        :type name: str
        :return: Its pinned value.
        :rtype: float
        :raises KeyError: If ``name`` is free in this variant.
        """
        if name in self.free_names:
            raise KeyError(f"{name} is free in {self.name}")
        return self.pinned_theta[name] if name in THETA_NAMES else HEURISTIC_OFF

    # -- bounds ------------------------------------------------------------------------

    def lo_hi(self, stack: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        The estimator's box over the FREE vector.

        Theta bounds come from the checkpoint's trained ranges; weight bounds from
        ``HEURISTIC_BOUNDS`` (Entry 33), symmetric about 0 -- both via
        ``ModelStack.lo_hi_with_heuristics``, the one place the two kinds of bound meet.

        :param stack: Loaded models and metadata (``ModelStack``).
        :type stack: Any
        :return: ``(lo, hi)``, each ``(n_free,)`` in :attr:`free_names` order.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        lo_full, hi_full = stack.lo_hi_with_heuristics()
        index = {n: i for i, n in enumerate(FIT_NAMES)}
        pick = [index[n] for n in self.free_names]
        return lo_full[pick].copy(), hi_full[pick].copy()

    # -- the mapping -------------------------------------------------------------------

    def embed(self, free: np.ndarray) -> np.ndarray:
        """
        The full 4- or 8-vector the likelihood reads, from the free vector.

        :param free: ``(n_free,)`` in :attr:`free_names` order.
        :type free: np.ndarray
        :return: ``(4,)`` or ``(8,)`` in :attr:`full_names` order.
        :rtype: np.ndarray
        :raises ValueError: If ``free`` has the wrong length.
        """
        values = np.asarray(free, dtype=np.float64)
        if values.shape != (self.n_free,):
            raise ValueError(f"{self.name}: expected {self.n_free} free values, "
                             f"got shape {values.shape}")
        free_of = dict(zip(self.free_names, values))
        return np.array([free_of[n] if n in free_of else self.pinned_value(n)
                         for n in self.full_names], dtype=np.float64)

    def embed_torch(self, free: torch.Tensor) -> torch.Tensor:
        """
        :meth:`embed` for a tensor, with the autograd graph intact.

        Built with ``torch.stack`` rather than assignment into a zeros tensor, for the same
        reason ``likelihood_grad.theta6_tensor`` is: stacking keeps every free element in
        the graph, while in-place assignment is a silent way to lose it. When nothing is
        pinned this returns ``free`` itself, so the shipped 4- and 8-vector paths are
        untouched rather than merely equivalent.

        :param free: ``(n_free,)`` in :attr:`free_names` order.
        :type free: torch.Tensor
        :return: ``(4,)`` or ``(8,)``, graph intact back to ``free``.
        :rtype: torch.Tensor
        :raises ValueError: If ``free`` has the wrong length.
        """
        if free.shape[-1] != self.n_free:
            raise ValueError(f"{self.name}: expected {self.n_free} free values, "
                             f"got shape {tuple(free.shape)}")
        if self.free_names == self.full_names:
            return free
        position = {n: i for i, n in enumerate(self.free_names)}
        pieces = [free[position[n]] if n in position
                  else torch.tensor(self.pinned_value(n), dtype=free.dtype, device=free.device)
                  for n in self.full_names]
        return torch.stack(pieces)

    def extract(self, full: np.ndarray) -> np.ndarray:
        """
        The free vector read out of a full one (the inverse of :meth:`embed` on free entries).

        :param full: ``(4,)`` or ``(8,)``; a 4-vector is padded with zero weights first.
        :type full: np.ndarray
        :return: ``(n_free,)`` in :attr:`free_names` order.
        :rtype: np.ndarray
        """
        values = as_eight(full)
        index = {n: i for i, n in enumerate(FIT_NAMES)}
        return np.array([values[index[n]] for n in self.free_names], dtype=np.float64)

    def start_from(self, parent_full: np.ndarray) -> np.ndarray:
        """
        The free starting vector seeded from another variant's full solution.

        A weight the parent did not fit reads as 0 (the padding), so a heuristic variant
        started from B begins at exactly B's log-likelihood (module docstring). The caller's
        box projection handles any value outside this variant's box.

        :param parent_full: The parent's full vector, ``(4,)`` or ``(8,)``.
        :type parent_full: np.ndarray
        :return: ``(n_free,)`` in :attr:`free_names` order.
        :rtype: np.ndarray
        """
        return self.extract(parent_full)

    def added_over(self, smaller: "FitVariant") -> tuple[str, ...]:
        """
        The coefficients this variant frees that ``smaller`` does not.

        A fit started from ``smaller``'s solution begins with exactly these at ``smaller``'s
        pinned values (0 for a weight, the pin for a theta), so they are the only axes along
        which it can beat that start. Ex: B over C is ``("hlp_cognitive_cost_coef",)``.

        :param smaller: The variant whose solution seeded the fit.
        :type smaller: FitVariant
        :return: The names, in :attr:`free_names` order; empty if ``smaller`` frees them all.
        :rtype: tuple[str, ...]
        """
        return tuple(n for n in self.free_names if n not in smaller.free_names)


def as_eight(vec: np.ndarray) -> np.ndarray:
    """
    A 4- or 8-vector as an 8-vector, zero weights appended when absent.

    :param vec: ``(4,)`` or ``(8,)``.
    :type vec: np.ndarray
    :return: ``(8,)`` in :data:`FIT_NAMES` order.
    :rtype: np.ndarray
    :raises ValueError: If ``vec`` is neither length.
    """
    values = np.asarray(vec, dtype=np.float64)
    if values.shape == (len(THETA_NAMES),):
        return np.concatenate([values, np.zeros(len(HEURISTIC_NAMES))])
    if values.shape == (len(FIT_NAMES),):
        return values
    raise ValueError(f"expected {len(THETA_NAMES)} or {len(FIT_NAMES)} entries, "
                     f"got shape {values.shape}")


# ===========================================================================
# The variant set
# ===========================================================================

_THETA_WITHOUT_COGNITIVE_COST = tuple(n for n in THETA_NAMES if n != "hlp_cognitive_cost_coef")
_HEURISTICS_WITHOUT_MEAN = tuple(n for n in HEURISTIC_NAMES if n != "mean_vehicle_distance")

# | Cognitive cost is pinned at its trained LOWER bound, not at 0 by assumption: the
# | checkpoint (human_modeling/inference/checkpoint) trains it over [0, 0.15], so 0 is
# | both "no attention cost" and inside the range the policy has seen. On a checkpoint with
# | a different lower bound this pin must be re-read from its hlp_param_config, which
# | `check_pins_inside` below enforces at load time.
NO_COGNITIVE_COST = 0.0

# | Dict order IS the fitting order: the nested chain, smallest model first, so every start a
# | variant needs has been computed before it runs.
# | The display names are the paper's labels (user, 2026-09-25), used on every figure and in
# | the summary files. The internal names stay as they are, because file names and fitted
# | CSVs carry them. NOTE: C is labelled "no-construals" although its policy still chooses
# | construals; only the attention cost is off (module docstring). The label is the user's
# | choice for the paper, not a change to the model. The model that truly makes no attention
# | choice is `C_full_visibility`, labelled "full-visibility" (user, 2026-11-10).
    # | [full-visibility] It is listed LAST so the four chained variants keep their positions
    # | in FIT_ORDER, which also orders figure colours and the per-participant test pairs.
VARIANTS: dict[str, FitVariant] = {
    "C_no_cognitive_cost": FitVariant(
        name="C_no_cognitive_cost", display="no-construals",
        free_theta=_THETA_WITHOUT_COGNITIVE_COST,
        pinned_theta={"hlp_cognitive_cost_coef": NO_COGNITIVE_COST}),
    "B_no_heuristics": FitVariant(
        name="B_no_heuristics", display="no-heuristics",
        free_theta=THETA_NAMES,
        extra_starts=("C_no_cognitive_cost",)),
    "A_no_mean_distance": FitVariant(
        name="A_no_mean_distance", display="Heuristics without mean distance",
        free_theta=THETA_NAMES, free_heuristics=_HEURISTICS_WITHOUT_MEAN,
        warm_start_from="B_no_heuristics"),
    "A_full": FitVariant(
        name="A_full", display="fully-parameterized",
        free_theta=THETA_NAMES, free_heuristics=HEURISTIC_NAMES,
        warm_start_from="A_no_mean_distance"),
    "C_full_visibility": FitVariant(
        name="C_full_visibility", display="full-visibility",
        free_theta=_THETA_WITHOUT_COGNITIVE_COST,
        pinned_theta={"hlp_cognitive_cost_coef": NO_COGNITIVE_COST},
        full_visibility=True),
}

FIT_ORDER: tuple[str, ...] = tuple(VARIANTS)
# | [full-visibility] The variants that make an attention choice -- every one but the
# | full-visibility model. Scripts that only know how to prepare or simulate the ordinary
# | model (the synthetic recovery study) and the nested-pair tests offer only these.
ATTENTION_CHOOSING_VARIANTS: tuple[str, ...] = tuple(n for n in FIT_ORDER
                                                     if not VARIANTS[n].full_visibility)
assert all(FIT_ORDER.index(p) < FIT_ORDER.index(v.name)
           for v in VARIANTS.values()
           for p in ((v.warm_start_from,) if v.warm_start_from else ()) + v.extra_starts), \
    "FIT_ORDER must list every start a variant uses before that variant"

# | The chain as (smaller, larger) pairs: the larger model contains the smaller one, so its
# | log-likelihood must be at least the smaller's. The comparison read-out counts violations.
NESTED_PAIRS: tuple[tuple[str, str], ...] = (
    ("C_no_cognitive_cost", "B_no_heuristics"),
    ("B_no_heuristics", "A_no_mean_distance"),
    ("A_no_mean_distance", "A_full"),
)

DEFAULT_VARIANT = "B_no_heuristics"     # the shipped fit; every published V4 result


def get_variant(name: str) -> FitVariant:
    """
    Look a variant up by name.

    :param name: One of :data:`VARIANTS`.
    :type name: str
    :return: The variant.
    :rtype: FitVariant
    :raises KeyError: With the valid names listed, if ``name`` is unknown.
    """
    try:
        return VARIANTS[name]
    except KeyError:
        raise KeyError(f"unknown variant {name!r}; expected one of {list(VARIANTS)}") from None


def ordered(names: list[str] | tuple[str, ...]) -> list[FitVariant]:
    """
    The requested variants in :data:`FIT_ORDER`, so a chain is fitted parent-first.

    :param names: Variant names in any order; duplicates are dropped.
    :type names: list[str] | tuple[str, ...]
    :return: The variants, in fitting order.
    :rtype: list[FitVariant]
    """
    wanted = {get_variant(n).name for n in names}
    return [VARIANTS[n] for n in FIT_ORDER if n in wanted]


def check_pins_inside(variant: FitVariant, stack: Any) -> None:
    """
    Refuse a pinned theta value the checkpoint was never trained on.

    :param variant: The variant about to be fitted.
    :type variant: FitVariant
    :param stack: Loaded models and metadata (``ModelStack``), for the trained ranges.
    :type stack: Any
    :raises ValueError: If a pinned value lies outside its coefficient's trained range.
    """
    for name, value in variant.pinned_theta.items():
        lo, hi = stack.bounds[name]
        if not lo <= value <= hi:
            raise ValueError(f"{variant.name} pins {name} at {value}, outside this "
                             f"checkpoint's trained range [{lo}, {hi}]")


# ===========================================================================
# The estimator-facing wrapper
# ===========================================================================

class VariantGradContext:
    """
    An ``EpisodeGradContext`` that takes the variant's FREE vector.

    ``fit_gradient.fit_mode`` and ``hessian_diag`` only ever call ``ctx.logL(vec)`` on a
    vector the size of the box they were given, so wrapping the context is all it takes for
    them to fit a pinned variant unchanged.

    :param ctx: The episode's differentiable likelihood. Must have been built with
        ``with_heuristics=True`` when the variant fits any weight.
    :type ctx: Any
    :param variant: The variant whose free vector this context accepts.
    :type variant: FitVariant
    :raises ValueError: If the variant fits weights but the context cannot score them, or if
        the context's episode was prepared for the other visibility mode.
    """

    def __init__(self, ctx: Any, variant: FitVariant) -> None:
        if variant.uses_heuristics and getattr(ctx, "features", None) is None:
            raise ValueError(f"{variant.name} fits heuristic weights, so its context must be "
                             f"built with EpisodeGradContext(..., with_heuristics=True)")
        # | [full-visibility] A full-visibility variant fitted on ordinary tables would fit the
        # | attention-choosing model with cognitive cost pinned -- C, under the wrong name --
        # | and nothing downstream could tell. The reverse is as silent. Every caller passes
        # | through here, so the mismatch is refused once, here.
        ctx_full = bool(getattr(getattr(ctx, "inputs", None), "full_visibility", False))
        if ctx_full != variant.full_visibility:
            raise ValueError(f"{variant.name} needs an episode prepared with "
                             f"full_visibility={variant.full_visibility}, but this context's "
                             f"episode has full_visibility={ctx_full}")
        self.ctx = ctx
        self.variant = variant

    def logL(self, free: torch.Tensor) -> torch.Tensor:
        """
        Differentiable log-likelihood at one free vector.

        :param free: ``(n_free,)``, ``requires_grad=True`` for a gradient.
        :type free: torch.Tensor
        :return: Scalar, graph intact back to ``free``.
        :rtype: torch.Tensor
        """
        return self.ctx.logL(self.variant.embed_torch(free))
