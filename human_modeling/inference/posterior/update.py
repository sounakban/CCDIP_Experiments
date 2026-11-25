"""
update.py — turn a lattice of log-likelihoods into a posterior and summarize it.

The log-prior under a uniform prior over the trained ranges is a constant inside the box
and so only fixes the support; :func:`posterior_weights` renormalises, which absorbs every
additive constant anyway.

Public API: posterior_weights, summarize.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from human_modeling.inference.model.choice import THETA_NAMES
from human_modeling.inference.posterior.grid import ParameterGrid


def posterior_weights(log_post: np.ndarray) -> np.ndarray:
    """
    Turn an unnormalised log-posterior on the lattice into probabilities summing to 1.

    :param log_post: Unnormalised log-posterior at every lattice point.
    :type log_post: np.ndarray
    :return: Probabilities, same shape, summing to 1.
    :rtype: np.ndarray
    """
    a = np.asarray(log_post, dtype=np.float64)
    # | A NaN anywhere is a broken evaluation, not evidence. Treating it as "no
    # | information" would silently discard every good point and return a uniform
    # | posterior, whose mean is the middle of the range and whose spread is the range --
    # | a confident-looking answer built from a bug. Fail instead.
    if np.any(np.isnan(a)):
        raise ValueError(f"{int(np.sum(np.isnan(a)))} lattice points are NaN; "
                         f"a likelihood evaluation failed and the posterior is unusable")
    m = float(np.max(a))
    if not np.isfinite(m):
        # | Genuinely every point impossible (all -inf). Report flat: the caller learns
        # | nothing, which is the truthful answer here.
        return np.full(a.shape, 1.0 / a.size)
    w = np.exp(a - m)
    return w / w.sum()


EDGE_FRACTION = 0.05        # outermost 5% of a TRAINED range counts as "at the bound"


def summarize(
    grid: ParameterGrid,
    log_post: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray | None] = None,
    names: Sequence[str] = THETA_NAMES,
) -> dict[str, float]:
    """
    Posterior mean, standard deviation and credible interval for each coefficient.

    Every quantity is a weighted sum over the lattice, so nothing is assumed about the
    shape of the distribution — in particular, a posterior piled against a bound gives
    an asymmetric interval rather than one straddling the bound.

    :param grid: The lattice the log-posterior lives on.
    :type grid: ParameterGrid
    :param log_post: Unnormalised log-posterior at every lattice point.
    :type log_post: np.ndarray
    :param bounds: ``(lo, hi)``, the TRAINED range per coefficient. Required for the
        edge-mass figure to mean anything: without it, "edge" can only mean the end of
        whatever lattice was passed, which on a zoomed lattice is an arbitrary small box
        rather than a bound the coefficient cannot cross. When omitted, edge mass is
        reported as NaN rather than as a number that looks meaningful and is not.
    :type bounds: tuple[np.ndarray, np.ndarray] | None
    :param names: The coefficient behind each lattice axis, in axis order. Defaults to the
        four theta names, which is every lattice this package built before model variants
        could pin a theta (``variants.py``); a variant's lattice spans only its FREE theta,
        so it passes those names and the pinned one gets no posterior at all.
    :type names: Sequence[str]
    :return: ``{name}_mean``, ``_sd``, ``_q05``, ``_q50``, ``_q95`` per coefficient,
        plus ``{name}_edge_mass`` — the probability lying within the outermost
        :data:`EDGE_FRACTION` of the coefficient's TRAINED range, which is how railing
        shows up in a posterior. Being a fraction of the trained range rather than of the
        lattice, it is comparable across resolutions and against point-estimate railing
        figures.
    :rtype: dict[str, float]
    :raises ValueError: If ``names`` does not match the lattice's axis count.
    """
    if len(names) != len(grid.shape):
        raise ValueError(f"{len(names)} coefficient names for a {len(grid.shape)}-axis lattice")
    w = posterior_weights(log_post).reshape(grid.shape)
    out: dict[str, float] = {}
    for d, name in enumerate(names):
        axis_vals = grid.axes[d]
        # | Marginal for this coefficient: sum the joint over every other axis.
        other = tuple(i for i in range(len(grid.shape)) if i != d)
        p = w.sum(axis=other)
        p = p / p.sum()
        mean = float(np.sum(p * axis_vals))
        var = float(np.sum(p * (axis_vals - mean) ** 2))
        out[f"{name}_mean"] = mean
        out[f"{name}_sd"] = float(np.sqrt(max(var, 0.0)))
        # | Interval ends are read off the running total at CELL EDGES, then located
        # | inside the cell by a straight line. Walking the running total and printing
        # | whichever lattice value it first passes rounds every end to one of
        # | `n_per_axis` values -- six, at the production setting -- so two participants
        # | with genuinely different uncertainty print identical intervals. It is also
        # | the same defect that, in the calibration gate, forced one draw in six to a
        # | false extreme and manufactured an OVERCONFIDENT verdict (Entry 6a); this is
        # | its sibling in the human-facing output.
        # |
        # | Correct rather than merely smoother: a lattice point REPRESENTS ITS CELL --
        # | that is why points sit at cell midpoints and why the mean and spread are
        # | weighted sums. Spreading a cell's belief across its width is what the lattice
        # | already means everywhere else.
        half_d = 0.5 * float(grid.cell_widths()[d])
        edges  = np.concatenate([[axis_vals[0] - half_d], axis_vals + half_d])  # (n+1,)
        cum    = np.concatenate([[0.0], np.cumsum(p)])                          # (n+1,)
        for q, tag in ((0.05, "q05"), (0.50, "q50"), (0.95, "q95")):
            out[f"{name}_{tag}"] = float(np.interp(q, cum, edges))
        # | Mass pressed against a TRAINED bound -- the data wanting to leave the range
        # | the policy was trained on. Measured against that range, never against the
        # | lattice, so a zoomed lattice sitting mid-range correctly reports zero.
        if bounds is None:
            out[f"{name}_edge_mass"] = float("nan")
        else:
            out[f"{name}_edge_mass"] = _edge_mass(
                axis_vals, p, float(bounds[0][d]), float(bounds[1][d]))
    return out


def _edge_mass(axis_vals: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    """
    Probability lying within the outermost slice of the TRAINED range.

    Each lattice point stands for a cell, and a cell may lie only PARTLY inside the edge
    slice. Counting whole cells makes the answer jump with resolution — at 8 points per
    axis no cell centre falls within 5% of a bound at all, so a flat posterior would
    report zero edge mass, while at 16 points the same posterior reports 0.125. Weighting
    each cell by how much of it overlaps the slice removes that: a flat posterior returns
    twice the slice fraction at every resolution, which is the correct answer.

    :param axis_vals: Cell centres along one coefficient.
    :type axis_vals: np.ndarray
    :param p: Marginal probability of each cell, summing to 1.
    :type p: np.ndarray
    :param lo: Lower trained bound for this coefficient.
    :type lo: float
    :param hi: Upper trained bound.
    :type hi: float
    :return: Probability inside the two edge slices.
    :rtype: float
    """
    span = hi - lo
    if span <= 0 or len(axis_vals) < 2:
        return float("nan")
    margin = EDGE_FRACTION * span
    width = float(axis_vals[1] - axis_vals[0])
    cell_lo = axis_vals - 0.5 * width
    cell_hi = axis_vals + 0.5 * width
    # | Length of each cell falling inside either slice, as a fraction of the cell.
    low_overlap = np.clip(np.minimum(cell_hi, lo + margin) - cell_lo, 0.0, None)
    high_overlap = np.clip(cell_hi - np.maximum(cell_lo, hi - margin), 0.0, None)
    frac = np.clip((low_overlap + high_overlap) / width, 0.0, 1.0)
    return float(np.sum(p * frac))
