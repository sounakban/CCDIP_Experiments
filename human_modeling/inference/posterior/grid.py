"""
grid.py — tabulate the log-likelihood on a fixed lattice over the coefficients.

The lattice is the workhorse of hlp_v3. Evaluating one episode's log-likelihood at
every lattice point costs ~11 minutes and is done ONCE; afterwards every posterior,
every forgetting value, every episode ordering and every convergence curve is arithmetic
on the stored array (see ``update.py``).

Two properties earn it that role over sampling or a Gaussian fit:

1. **No shape assumptions.** Roughly 60% of production fits press against a coefficient
   bound. A lattice represents that correctly — the mass simply piles into the edge
   cells — whereas a Gaussian centred on the best fit would report an interval reaching
   past a bound the coefficient cannot cross.
2. **Reusability.** A sampler returns draws from ONE target, so a new target (one more
   episode, a different forgetting value) means sampling again. A lattice stores the
   likelihood itself, and targets are built from it by addition.

Points sit at CELL MIDPOINTS rather than at the bounds. With ``n`` points per axis the
range is cut into ``n`` equal cells and each point represents its cell, so every point
carries equal prior mass under the uniform prior and the sum over the lattice is a plain
Riemann sum. Placing points on the bounds instead would give the two end points half a
cell each and quietly distort exactly the edge region that matters most here.

RESOLUTION IS THE HARD PART, and measurement changed the picture. A lattice spanning the
whole trained range is FAR too coarse: on real episodes the posterior fits inside ONE cell
of a 4-per-axis lattice, with a single point holding 95-99% of belief and an effective
contributing-point count of 1.02 out of 256. A thousand frames of evidence sharpen the
surface far beyond what a whole-range lattice can describe, so such a lattice reports the
spacing of its own points rather than anything about the posterior. Refining the whole
range cannot fix this at any affordable cost.

The answer is to spend the points where the belief is: the gradient ascent locates the
peak, and :func:`build_zoom_grid` lays a fine lattice over a small box around it.

Public API: ParameterGrid, build_grid, build_zoom_grid, evaluate_episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from human_modeling.inference.posterior.likelihood import (EpisodeInputs, ModelStack,
                                                        log_likelihood)


@dataclass
class ParameterGrid:
    """
    A lattice over the fitted coefficients, plus the bookkeeping to interpret it.

    :ivar axes: Per-coefficient value arrays, each ``(n_per_axis,)``, in
        ``THETA_NAMES`` order.
    :vartype axes: list[np.ndarray]
    :ivar points: Every combination, ``(n_points, n_coefficients)``, in the row order
        produced by varying the LAST coefficient fastest.
    :vartype points: np.ndarray
    :ivar n_per_axis: Points along each axis.
    :vartype n_per_axis: int
    """

    axes:        list[np.ndarray]
    points:      np.ndarray
    n_per_axis:  int

    @property
    def n_points(self) -> int:
        """Total lattice points."""
        return int(self.points.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        """Lattice shape, for reshaping a flat array back to per-axis form."""
        return tuple(len(a) for a in self.axes)

    def cell_widths(self) -> np.ndarray:
        """
        Width of one cell along each axis, in the coefficients' own units.

        :return: Widths, ``(n_coefficients,)``.
        :rtype: np.ndarray
        """
        return np.array([a[1] - a[0] if len(a) > 1 else 0.0 for a in self.axes])


def build_grid(
    lo: np.ndarray,
    hi: np.ndarray,
    n_per_axis: int,
) -> ParameterGrid:
    """
    Lay a lattice of cell midpoints across the trained coefficient ranges.

    :param lo: Lower bound per coefficient, in ``THETA_NAMES`` order.
    :type lo: np.ndarray
    :param hi: Upper bound per coefficient.
    :type hi: np.ndarray
    :param n_per_axis: Points along each axis.
    :type n_per_axis: int
    :return: The lattice.
    :rtype: ParameterGrid
    :raises ValueError: If ``n_per_axis`` is below 2 or the bounds are not increasing.
    """
    if n_per_axis < 2:
        raise ValueError(f"n_per_axis must be at least 2, got {n_per_axis}")
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if not np.all(hi > lo):
        raise ValueError(f"every upper bound must exceed its lower bound: lo={lo}, hi={hi}")
    # | Midpoint of cell i out of n: lo + (i + 0.5) * (hi - lo) / n.
    frac = (np.arange(n_per_axis) + 0.5) / n_per_axis
    axes = [lo[d] + frac * (hi[d] - lo[d]) for d in range(len(lo))]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([m.reshape(-1) for m in mesh], axis=1)
    return ParameterGrid(axes=axes, points=points, n_per_axis=n_per_axis)


def evaluate_episode(
    stack: ModelStack,
    inputs: EpisodeInputs,
    grid: ParameterGrid,
    embed: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """
    Log-likelihood of one episode at every lattice point.

    The expensive call — roughly ``n_points`` times 0.165 s. Its result is what every
    later computation reuses, so it is written to disk by the caller and never repeated.

    :param stack: Loaded models and metadata.
    :type stack: ModelStack
    :param inputs: The episode's parameter-free scaffolding.
    :type inputs: EpisodeInputs
    :param grid: The lattice to evaluate on.
    :type grid: ParameterGrid
    :param embed: A function from a lattice point to the full coefficient vector the
        likelihood reads, when the lattice spans only some coefficients and the rest are
        held at a fitted value -- the heuristic weights, or a model variant's pinned theta
        (``variants.FitVariant.embed``; implementation.md Entry 34). Omitting it scores the
        points as they are. Note what the result then means: a CONDITIONAL posterior given
        the fixed values, not a marginal one, so it does not carry the fixed coefficients'
        own uncertainty into theta's.
    :type embed: Callable[[np.ndarray], np.ndarray] | None
    :return: Log-likelihoods, ``(n_points,)``, aligned with ``grid.points``.
    :rtype: np.ndarray
    """
    out = np.empty(grid.n_points, dtype=np.float64)
    for i in range(grid.n_points):
        point = grid.points[i] if embed is None else embed(grid.points[i])
        out[i] = log_likelihood(stack, inputs, point)
    return out


def build_zoom_grid(
    lo: np.ndarray,
    hi: np.ndarray,
    centre: np.ndarray,
    half_width: np.ndarray,
    n_per_axis: int,
) -> ParameterGrid:
    """
    A lattice covering only the region the posterior actually occupies.

    MEASURED and the reason this exists: on real episodes the posterior lives inside a
    SINGLE cell of a 4-per-axis lattice. One point carries 95-99% of all belief, adjacent
    points differ by 7 to 39 in log terms, and the effective number of contributing
    points is 1.02 out of 256. A thousand frames of evidence make the surface far sharper
    than a lattice spanning the whole trained range can describe, so a coarse lattice
    reports the SPACING OF ITS OWN POINTS rather than any property of the posterior.

    Spreading the same points over a small box around the peak fixes that. The cost is
    that the answer is only valid if the box really does contain the belief, which is not
    assumed — :func:`zoom_covers_mass` checks it against the coarse pass that located the
    peak, and a box that misses mass must be widened rather than trusted.

    :param lo: Lower bound per coefficient; the zoom never extends past it.
    :type lo: np.ndarray
    :param hi: Upper bound per coefficient.
    :type hi: np.ndarray
    :param centre: Where to centre the zoom, in coefficient units.
    :type centre: np.ndarray
    :param half_width: How far to reach either side, per coefficient.
    :type half_width: np.ndarray
    :param n_per_axis: Points along each axis.
    :type n_per_axis: int
    :return: A lattice over the clipped sub-box.
    :rtype: ParameterGrid
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    centre = np.asarray(centre, dtype=np.float64)
    half = np.abs(np.asarray(half_width, dtype=np.float64))
    # | Clip to the trained range: outside it the policy was never trained and clamps,
    # | so a point there would carry a meaningless likelihood.
    z_lo = np.clip(centre - half, lo, hi)
    z_hi = np.clip(centre + half, lo, hi)
    # | A degenerate axis would collapse the lattice; keep a sliver so it stays a grid.
    thin = z_hi - z_lo < 1e-12
    if np.any(thin):
        pad = 1e-6 * (hi - lo)
        z_lo = np.where(thin, np.clip(centre - pad, lo, hi), z_lo)
        z_hi = np.where(thin, np.clip(centre + pad, lo, hi), z_hi)
    return build_grid(z_lo, z_hi, n_per_axis)
