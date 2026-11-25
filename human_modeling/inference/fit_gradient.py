"""
fit_gradient.py — V4's estimator: exact-gradient ascent to the mode, then a small local
lattice for the posterior.

WHY THIS SHAPE (Entry 2's ranking, now with Entry 7's measurements behind it)
    The gradient does the FINDING: ascent needs tens of iterations, not the 1,552-5,440
    lattice evaluations production spends today, and each step's gradient is EXACT — the
    likelihood is deterministic (bit-identical across evaluations, measured) and autodiff
    has no probing step, so a zero gradient is a real stationary point.

    The lattice does the DESCRIBING: an n=3 box around the mode is 81 points, and its
    boundary behaviour is the only one already validated on this problem — mass piles into
    edge cells and is reported honestly, which matters because 100% of human episodes rail.
    The posterior machinery (summarize, intervals) is reused from hlp_v3 unchanged.

THE POSTERIOR BOX: FIXED OR FROM CURVATURE
    The 81-point lattice is placed in a box around the mode. A fixed box (0.10 of each range)
    truncates a posterior wider than itself and reports an interval that is too narrow. The
    curvature box sizes each axis from the second derivative of the log-likelihood at the
    mode -- the Laplace approximation (Tierney & Kadane 1986, "Accurate approximations for
    posterior moments and marginal densities", JASA 81:82-86; MacKay 1992, "Bayesian
    interpolation", Neural Computation 4:415-447): standard deviation 1/sqrt(-d2 logL/dz2),
    box = +/- BOX_N_SD of those, clipped to [BOX_HALF_MIN, BOX_HALF_MAX] of the range. A flat
    axis (curvature ~0, e.g. proximity) or a mode on a bound (curvature not a peak) gets the
    WIDEST ALLOWED box, never infinity.

    NOTE on the word "ceiling", which this project uses for the maximum achievable R2 in a
    behavioural comparison: it is NOT used here for the box clip, to keep that one meaning.
    The clips are the "narrowest allowed" and "widest allowed" box.

    BOX_HALF_MIN and BOX_HALF_MAX were chosen, then MEASURED to be nearly inert -- see
    implementation.md Entry 10c. Sizing the box does not fix coverage on this problem.

Public API: :func:`fit_mode`, :func:`fit_posterior`, :func:`hessian_diag`,
:func:`curvature_half_widths`.

Typical usage::

    ctx = EpisodeGradContext(stack, inputs)
    mode, info = fit_mode(ctx, lo, hi)                                      # 1. ~tens of gradient steps
    half = curvature_half_widths(hessian_diag(ctx, mode, lo, hi), hi - lo)  # 2. box
    post = fit_posterior(stack, inputs, mode, lo, hi, half)                 # 3. 81-point local posterior
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch

from human_modeling.inference.model.choice import THETA_NAMES
from human_modeling.inference.posterior.grid import build_zoom_grid, evaluate_episode
from human_modeling.inference.posterior.update import summarize

Embed = Callable[[np.ndarray], np.ndarray]   # lattice point -> full coefficient vector

# | Convergence: stop when the best logL has improved by less than TOL over PATIENCE
# | consecutive iterations. TOL is in nats; 0.01 is far below any decision the posterior
# | will make (episode likelihoods are ~-2,000 and coefficient differences move them by
# | tens of nats).
DEFAULT_MAX_ITERS = 60
DEFAULT_TOL       = 0.01
DEFAULT_PATIENCE  = 5
DEFAULT_LR        = 0.08     # in normalized [0,1] coordinates; Adam rescales per-axis

# | Most numpy evaluations the posterior lattice may cost for one episode. 81 (3 per axis
# | over 4 coefficients) takes ~15 s and is the sizing this stage was built for; the cap
# | leaves headroom for a finer theta lattice while refusing the 6,561 points that 3 per
# | axis over 8 coefficients would silently become (see :func:`_guard_lattice_size`).
MAX_LATTICE_POINTS = 1024

# | Settings for the JOINT (theta + heuristic weight) fit. The defaults above were tuned when
# | every fitted axis was a theta range; a weight axis spans 2 * bound, and because the step
# | is taken in NORMALIZED coordinates, DEFAULT_LR moves 1.26 units along
# | `lead_vehicle_inverse_ttc` on the first step -- far enough to overshoot, after which five
# | non-improving steps end the run at exactly w = 0. MEASURED (implementation.md Entry 34),
# | on two episodes that stalled at 0.08: at 0.04 one gained +0.50 and stopped after 8
# | iterations; at 0.02 it gained +2.09 and was still improving at 60. A smaller step needs
# | proportionally more of them, so the cap rises with it.
DEFAULT_LR_JOINT        = 0.02
DEFAULT_MAX_ITERS_JOINT = 200

# | Automatic rescue for an ascent that never moved (:func:`fit_mode_with_retry`). The step is
# | divided by STALL_LR_FACTOR and the fit re-run, up to STALL_RETRIES times. A factor rather
# | than a nudge because the stall is a CLIFF: MEASURED, 0.04 and 0.03 both fail on an episode
# | where 0.02 converges, so shaving the rate would just stall again one step lower. Two
# | retries take 0.02 -> 0.005 -> 0.00125, which spans the whole range over which any episode
# | measured here changed behaviour. The cost falls almost entirely on stalled fits, which end
# | after `patience + 1` iterations and are the cheapest the estimator ever runs.
STALL_RETRIES   = 2
STALL_LR_FACTOR = 4.0

# | Curvature box (see module docstring). FD_STEP_Z is the finite-difference step for the
# | Hessian diagonal, in normalized units: 1% of a range, far above the exact gradient's
# | noise floor (bit-identical evaluations) and far below any posterior width seen.
FD_STEP_Z    = 0.01
BOX_N_SD     = 3.0       # +/- this many Laplace standard deviations
# | MEASURED to be nearly inert (Entry 10c): across n_sd in {2,3,4} and widest-allowed clip
# | in {0.25, 0.50, 1.00}, coverage moves only 42% -> 50% while the interval grows 56%.
# | Widening does not fix the coverage failure, because that failure is bias in the CENTRE,
# | not width. Keep these values; do not tune them expecting calibration to follow.
BOX_HALF_MIN = 0.05      # narrowest allowed half-width, fraction of range
BOX_HALF_MAX = 0.50      # widest allowed: a flat axis spans the range, never infinity


def _value_and_backward(
    ctx: Any,
    z_of: Callable[[], torch.Tensor],
    lo_t: torch.Tensor,
    span: torch.Tensor,
) -> float:
    """
    The episode log-likelihood at the current parameters, with ``-grad`` ACCUMULATED into
    the leaf that ``z_of`` depends on.

    :return: The log-likelihood value.
    :rtype: float
    """
    theta = lo_t + z_of() * span
    ll = ctx.logL(theta)
    (-ll).backward()                           # Adam minimises; we ascend
    return float(ll.detach())


def fit_mode(
    ctx: Any,
    lo: np.ndarray,
    hi: np.ndarray,
    start: np.ndarray | None = None,
    max_iters: int = DEFAULT_MAX_ITERS,
    lr: float = DEFAULT_LR,
    tol: float = DEFAULT_TOL,
    patience: int = DEFAULT_PATIENCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Gradient ascent to the likelihood mode for one episode: Adam on normalized coordinates
    z in [0,1]^n, clamped each step, so it can sit exactly ON a bound.

    Runs in NORMALIZED coordinates (theta = lo + z * (hi - lo)) so all four axes share one
    learning rate; the network itself normalizes theta the same way, so this matches the
    geometry the likelihood actually lives in.

    :param ctx: Prepared differentiable likelihood for the episode. Anything with a
        ``logL(vector) -> scalar tensor`` method serves -- an ``EpisodeGradContext``, or a
        ``variants.VariantGradContext`` that maps a variant's free vector onto one.
    :type ctx: Any
    :param lo: Lower trained bound per coefficient.
    :type lo: np.ndarray
    :param hi: Upper trained bound per coefficient.
    :type hi: np.ndarray
    :param start: Starting theta; the range midpoint if omitted.
    :type start: np.ndarray | None
    :param max_iters: Iteration cap; each iteration costs one gradient (~12.7 numpy
        evaluations, MEASURED).
    :type max_iters: int
    :param lr: Adam learning rate in normalized coordinates.
    :type lr: float
    :param tol: Stop when the best logL improves less than this over ``patience`` steps.
    :type tol: float
    :param patience: Consecutive low-improvement iterations before stopping.
    :type patience: int
    :return: ``(theta_mode, info)`` — info carries ``logL``, ``iters``, ``trace`` (per-step
        logL), ``at_bound`` (per-coefficient, within 1% of a bound),
        ``improvement`` (how much the ascent beat its own starting point by),
        ``no_progress`` (it did not beat it at all — see the comment at the return, and
        check it on every joint fit) and ``hit_iteration_cap``.
    :rtype: tuple[np.ndarray, dict[str, Any]]
    """
    lo_t = torch.tensor(lo, dtype=torch.float64)
    span = torch.tensor(hi - lo, dtype=torch.float64)
    z0 = (torch.tensor((start if start is not None else 0.5 * (lo + hi)),
                       dtype=torch.float64) - lo_t) / span
    z0 = z0.clamp(1e-4, 1 - 1e-4)
    z = z0.clone().requires_grad_(True)

    def z_of() -> torch.Tensor:
        return z

    opt = torch.optim.Adam([z], lr=lr)
    best_ll, best_theta = -np.inf, None
    trace: list[float] = []
    stall = 0
    for it in range(max_iters):
        opt.zero_grad()
        with torch.no_grad():
            theta_now = (lo_t + z_of() * span).cpu().numpy().copy()   # where this step is evaluated
        v = _value_and_backward(ctx, z_of, lo_t, span)
        opt.step()
        with torch.no_grad():
            z.clamp_(0.0, 1.0)
        trace.append(v)
        if v > best_ll + tol:
            best_ll, stall = v, 0
            best_theta = theta_now
        else:
            stall += 1
            if stall >= patience:
                break
    if best_theta is None:                     # never improved: report the start
        best_theta = (lo_t + z0 * span).cpu().numpy()
        best_ll = trace[0] if trace else float("nan")
    at_bound = ((best_theta <= lo + 0.01 * (hi - lo))
                | (best_theta >= hi - 0.01 * (hi - lo)))
    # | Did the ascent get anywhere at all? The first iteration is evaluated AT the starting
    # | point, so `trace[0]` is the starting value and `best_ll` can only equal it when every
    # | later step was rejected -- the fit then returns exactly what it was handed. That is
    # | the signature of a step size too large for some axis: the first step overshoots, the
    # | next `patience` steps fail to recover, and the run ends at iteration `patience + 1`
    # | reporting a coefficient that never moved. It is indistinguishable, from inside one
    # | fit, from a start that was already optimal -- so this is REPORTED, never raised, and
    # | the caller decides. MEASURED: at lr = 0.08 this fired on 8 of 21 joint fits, each
    # | returning weights of exactly 0 on data generated with substantial ones
    # | (implementation.md Entry 34).
    improvement = float(best_ll - trace[0]) if trace else 0.0
    no_progress = bool(trace) and improvement <= tol
    return best_theta, {"logL": best_ll, "iters": len(trace), "trace": trace,
                        "at_bound": at_bound.tolist(),
                        "improvement": improvement, "no_progress": no_progress,
                        "hit_iteration_cap": len(trace) >= max_iters}


def _grad_z(ctx: Any, z: np.ndarray, lo_t: torch.Tensor, span: torch.Tensor) -> np.ndarray:
    """Gradient of the log-likelihood with respect to normalized z, at ``z``. (4,)"""
    zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    theta = lo_t + zt * span
    ctx.logL(theta).backward()
    return zt.grad.detach().cpu().numpy().copy()


def fit_mode_with_retry(
    ctx: Any,
    lo: np.ndarray,
    hi: np.ndarray,
    retries: int = STALL_RETRIES,
    lr_factor: float = STALL_LR_FACTOR,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    :func:`fit_mode`, re-run at a smaller step whenever it fails to move at all.

    A stalled ascent is one that never beat its own starting point, which the fit reports as
    ``no_progress``. The cause MEASURED in this package is a step too large for one axis: the
    first move overshoots and the next few fail to recover (implementation.md Entry 34). That
    is exactly the failure a smaller step fixes, and the retry costs almost nothing where it is
    needed -- a stalled run ends after ``patience + 1`` iterations, so it is the cheapest fit
    the estimator ever performs.

    The retry is deliberately NOT silent. An episode that needed one is marked
    ``rescued_by_retry``, because a fit that only converged at a quarter of the intended step
    is evidence about the step size, and the run's settings should be reconsidered rather than
    papered over. An episode still stalled after every retry keeps ``no_progress`` set.

    :param ctx: Prepared differentiable likelihood.
    :type ctx: Any
    :param lo: Lower bound per coefficient.
    :type lo: np.ndarray
    :param hi: Upper bound per coefficient.
    :type hi: np.ndarray
    :param retries: How many times to shrink the step before giving up. 0 makes this exactly
        :func:`fit_mode`.
    :type retries: int
    :param lr_factor: What to divide the learning rate by each retry. MEASURED, the stall is a
        cliff rather than a gradient -- 0.04 and 0.03 both fail where 0.02 succeeds -- so the
        step down is a factor, not a nudge.
    :type lr_factor: float
    :param kwargs: Passed through to :func:`fit_mode`; ``lr`` is the rate the FIRST attempt
        uses.
    :type kwargs: Any
    :return: ``(theta_mode, info)`` from the best attempt, with ``attempts`` (how many ascents
        ran), ``lr_used`` (the rate that produced this answer) and ``rescued_by_retry`` added.
    :rtype: tuple[np.ndarray, dict[str, Any]]
    """
    lr = float(kwargs.pop("lr", DEFAULT_LR))
    best_theta, best_info, best_lr = None, None, lr
    attempt = 0
    for attempt in range(max(0, retries) + 1):
        attempt_lr = lr / (lr_factor ** attempt)
        theta, info = fit_mode(ctx, lo, hi, lr=attempt_lr, **kwargs)
        # | Keep the attempt with the highest likelihood, not merely the last one: a smaller
        # | step is not guaranteed to be better, only more likely to move at all (Entry 34
        # | records one episode where the larger step reached a better optimum).
        if best_info is None or info["logL"] > best_info["logL"]:
            best_theta, best_info, best_lr = theta, info, attempt_lr
        if not info["no_progress"]:
            break
    assert best_theta is not None and best_info is not None   # the loop always runs once
    best_info = dict(best_info)
    best_info["attempts"] = attempt + 1
    best_info["lr_used"] = best_lr
    best_info["rescued_by_retry"] = bool(attempt > 0 and not best_info["no_progress"])
    return best_theta, best_info


def hessian_diag(
    ctx: Any,
    theta: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    step_z: float = FD_STEP_Z,
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Diagonal of the Hessian of the log-likelihood at ``theta``, in normalized units.

    Central differences of the EXACT autodiff gradient, one axis at a time: 8 gradient
    evaluations. Differencing the exact gradient is accurate where differencing the value
    is not, and needs no double-backward graph in memory. When ``theta`` sits on a bound
    the stencil is shifted inside [0, 1] and becomes one-sided.

    :param ctx: Prepared differentiable likelihood.
    :type ctx: Any
    :param theta: Point of evaluation, normally the mode.
    :type theta: np.ndarray
    :param lo: Lower trained bound per coefficient.
    :type lo: np.ndarray
    :param hi: Upper trained bound per coefficient.
    :type hi: np.ndarray
    :param step_z: Half-stencil in normalized units.
    :type step_z: float
    :param axes: Which coordinates to differentiate along; every one by default. A joint
        (theta, weight) fit only needs theta's curvature to size its posterior box, and
        each skipped axis saves two gradient evaluations. Skipped entries come back NaN, so
        reading one by mistake fails loudly rather than as a plausible number.
    :type axes: Sequence[int] | None
    :return: ``d2 logL / dz_i2`` per coefficient; negative at a peak. ``(len(theta),)``
    :rtype: np.ndarray
    """
    lo_t = torch.tensor(lo, dtype=torch.float64)
    span = torch.tensor(hi - lo, dtype=torch.float64)
    z = (np.asarray(theta, dtype=np.float64) - lo) / (hi - lo)
    out = np.full(len(z), np.nan, dtype=np.float64)
    for i in (range(len(z)) if axes is None else axes):
        a, b = z[i] - step_z, z[i] + step_z
        if a < 0.0: a, b = z[i], z[i] + 2 * step_z         # one-sided at the lower bound
        if b > 1.0: a, b = z[i] - 2 * step_z, z[i]         # one-sided at the upper bound
        za, zb = z.copy(), z.copy()
        za[i], zb[i] = a, b
        out[i] = (_grad_z(ctx, zb, lo_t, span)[i] - _grad_z(ctx, za, lo_t, span)[i]) / (b - a)
    return out


def curvature_half_widths(
    hess_diag: np.ndarray,
    span: np.ndarray,
    n_sd: float = BOX_N_SD,
    half_min: float = BOX_HALF_MIN,
    half_max: float = BOX_HALF_MAX,
) -> np.ndarray:
    """
    Posterior-box half-widths from the curvature at the mode (Laplace; module docstring).

    Standard deviation per axis is ``1 / sqrt(-H_ii)`` in normalized units; the box spans
    ``n_sd`` of them, clipped to ``[half_min, half_max]`` of the range. An axis whose
    curvature is not a peak (``H_ii >= 0``) gets ``half_max``, the widest allowed box --
    the honest response to "the likelihood does not fall away here", since any narrower box
    would report a confidence the data does not support.

    :param hess_diag: Output of :func:`hessian_diag`.
    :type hess_diag: np.ndarray
    :param span: ``hi - lo`` per coefficient.
    :type span: np.ndarray
    :param n_sd: Half-width in Laplace standard deviations.
    :type n_sd: float
    :param half_min: Floor, fraction of range.
    :type half_min: float
    :param half_max: Ceiling, fraction of range.
    :type half_max: float
    :return: Half-width per coefficient in theta units. (4,)
    :rtype: np.ndarray
    """
    h = np.asarray(hess_diag, dtype=np.float64)
    sd_z = np.where(h < 0.0, 1.0 / np.sqrt(np.maximum(-h, 1e-300)), np.inf)
    half_z = np.clip(n_sd * sd_z, half_min, half_max)
    return half_z * np.asarray(span, dtype=np.float64)


def fit_posterior(
    stack: Any,
    inputs: Any,
    mode: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    half: np.ndarray,
    n_per_axis: int = 3,
    embed: Embed | None = None,
    names: Sequence[str] = THETA_NAMES,
) -> dict[str, float]:
    """
    The posterior: a small lattice around the gradient-found mode, hlp_v3 machinery.

    The lattice is only expensive when it must FIND the peak as well as describe it; at the
    mode, ``3^4 = 81`` points suffice for the moments and intervals. Boundary behaviour is
    inherited from the lattice, which reports edge mass honestly — the property no gradient
    posterior method has demonstrated on this problem.

    :param stack: Loaded models and metadata.
    :type stack: Any
    :param inputs: Prepared episode.
    :type inputs: Any
    :param mode: Centre for the box, normally :func:`fit_mode`'s output.
    :type mode: np.ndarray
    :param lo: Lower trained bound per coefficient.
    :type lo: np.ndarray
    :param hi: Upper trained bound per coefficient.
    :type hi: np.ndarray
    :param half: Half-width per coefficient in theta units, normally from
        :func:`curvature_half_widths`; the box is clipped to the trained bounds by
        ``build_zoom_grid``, so a mode ON a bound gets an asymmetric box, which is correct.
    :type half: np.ndarray
    :param n_per_axis: Lattice resolution; 3 gives 81 numpy evaluations (~15 s).
    :type n_per_axis: int
    :param embed: Lattice point -> full coefficient vector, when the lattice spans only some
        coefficients: a model variant whose pinned coefficients are not a trailing block
        (``variants.FitVariant``), or theta's posterior described with the heuristic weights
        pinned at the values the mode found, which keeps the lattice 4-dimensional instead of
        8 -- the whole reason the guard below does not fire. The result is CONDITIONAL on
        the pinned values, not marginal over them.
    :type embed: Embed | None
    :param names: The coefficient behind each lattice axis, in order; theta's four by
        default. A variant with a pinned theta passes its free theta names, and the summary
        then carries no entry at all for the pinned one.
    :type names: Sequence[str]
    :return: ``summarize``'s output — mean, sd, q05/q50/q95 and edge mass per coefficient.
    :rtype: dict[str, float]
    :raises ValueError: If ``names`` does not match the box.
    """
    _guard_lattice_size(len(lo), n_per_axis)
    if len(names) != len(lo):
        raise ValueError(f"{len(names)} coefficient names for a {len(lo)}-coefficient box")
    half_w = np.asarray(half, dtype=np.float64)
    centre = np.asarray(mode, dtype=np.float64).copy()
    grid = build_zoom_grid(lo, hi, centre, half_w, n_per_axis)
    vals = evaluate_episode(stack, inputs, grid, embed=embed)
    return summarize(grid, vals, bounds=(lo, hi), names=names)


def _guard_lattice_size(n_dims: int, n_per_axis: int) -> None:
    """
    Refuse a lattice whose point count has quietly exploded with the dimension.

    The lattice is ``n_per_axis ** n_dims`` numpy evaluations, so the 81 points (~15 s) that
    describe a 4-coefficient posterior become 6,561 (~18 min PER EPISODE) once the four
    heuristic weights join the fitted vector. That is a silent 80x cost, not an error, so it
    would otherwise be discovered only by a run that appeared to hang. Fit the mode in the
    full space and describe the posterior over theta alone, or lower ``n_per_axis``.

    :param n_dims: Length of the fitted vector.
    :type n_dims: int
    :param n_per_axis: Lattice resolution per axis.
    :type n_per_axis: int
    :raises ValueError: If the lattice would exceed :data:`MAX_LATTICE_POINTS`.
    """
    n_points = n_per_axis ** n_dims
    if n_points > MAX_LATTICE_POINTS:
        raise ValueError(
            f"a {n_per_axis}-per-axis lattice over {n_dims} coefficients is {n_points} "
            f"numpy evaluations, past the {MAX_LATTICE_POINTS} this stage is sized for "
            f"(implementation.md Entry 34). Describe the posterior over theta alone, or "
            f"lower n_per_axis.")
