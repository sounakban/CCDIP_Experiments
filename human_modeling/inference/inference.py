"""VENDORED. Lineage: human_modeling/hlp_v2/inference.py -> hlp_v3 -> here (2026-09-14,
implementation.md Entry 28). Both earlier packages are retired and deleted; their
design logs are in human_modeling/version_history.md. This package imports from no
other inference version and must not start.

inference.py — the forward pass over the per-step tables.

Implements implementation.md §3 (the update).
Everything here is plain array bookkeeping in log space — no network calls; the two
probability tables (``log ρ`` from model/choice.py, ``log b`` from model/emission.py)
and the θ-free kernel (model/kernel.py) are inputs.

Table convention: per trajectory step t, tables have shape ``(G, M_t)`` — goal axis flat
``g = lane·NB + bin``, construal axis in the enumeration order of
``EpisodeKernel.masks_at(t)`` (position ``p`` ↔ mask int ``p | invalid_bits``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .model.kernel import EpisodeKernel

NEG_INF = -np.inf


def _logsumexp_all(x: np.ndarray) -> float:
    """Total a table of log-probabilities: ``log Σ exp(x)`` over every element.

    Replaces ``np.logaddexp.reduce(x, axis=None)``, which is a PAIRWISE ufunc reduction —
    it combines two elements at a time with no vectorised path, and profiling made it the
    single largest cost in :func:`forward_pass` (43% of the recursion; 173 µs vs 31 µs for
    this formulation on a 96x64 table, bit-identical output).

    :param x: Log-space values; may contain ``-inf`` (impossible cells).
    :type x: np.ndarray
    :return: ``log Σ exp(x)``; ``-inf`` if every entry is ``-inf`` or ``x`` is empty.
    :rtype: float
    """
    if x.size == 0:
        return NEG_INF
    m = float(x.max())
    if not np.isfinite(m):
        # all -inf (mass is zero) or an inf leaked in — subtracting would give NaN
        return m
    return m + float(np.log(np.exp(x - m).sum()))


class _KeepPlan:
    """Cached θ-free transition bookkeeping for one episode (implementation.md §3).

    ``_keep_maps`` and the scatter index arrays depend only on the recorded trajectory —
    which subtasks end at each step, and where a survivor's label moves to next step —
    so they are IDENTICAL across every optimizer evaluation. Recomputing them per
    evaluation cost 20% of the recursion; this caches them once per episode, the same way
    the emission table is cached.

    :param kernel: Episode kernel.
    :type kernel: EpisodeKernel
    :param shapes: Per-step table shapes ``(G, M_t)``, i.e. ``[a.shape for a in log_rho]``.
    :type shapes: list[tuple[int, int]]
    :ivar alive: Per step, bool ``(G,)`` — goals that do NOT end at this step.
    :vartype alive: list[np.ndarray]
    :ivar flat_tgt: Per step, flat target indices into the NEXT table for every surviving
        ``(g, m)`` cell, or ``None`` when nothing survives.
    :vartype flat_tgt: list[np.ndarray | None]
    :ivar injective: Per step, True when the relabel is one-to-one so the scatter can be a
        plain assignment instead of a merge (measured: true on ~86% of steps).
    :vartype injective: list[bool]
    """

    def __init__(self, kernel: EpisodeKernel, shapes: list[tuple[int, int]]) -> None:
        self.alive: list[np.ndarray] = []
        self.src_g: list[np.ndarray] = []
        self.flat_tgt: list[Optional[np.ndarray]] = []
        self.injective: list[bool] = []
        self.next_size: list[int] = []
        for t in range(kernel.N - 1):
            m_src, m_next = shapes[t][1], shapes[t + 1][1]
            alive, g_tgt, m_tgt = _keep_maps(kernel, t, m_src, m_next)
            self.alive.append(alive)
            src = np.nonzero(alive)[0]
            self.src_g.append(src)
            size = shapes[t + 1][0] * m_next
            self.next_size.append(size)
            if src.size == 0:
                self.flat_tgt.append(None)
                self.injective.append(True)
                continue
            flat = (np.repeat(g_tgt[src], m_src) * m_next + np.tile(m_tgt, src.shape[0]))
            self.flat_tgt.append(flat.astype(np.int64))
            # a merge happens only when two source masks share a target position — that
            # is the sole reason a scatter-ADD is needed rather than an assignment
            self.injective.append(np.unique(m_tgt).size == m_tgt.size)


# Single-entry cache: the optimizer calls forward_pass ~2500x per episode with identical
# geometry, so one slot is hit every time after the first, and only one episode is ever in
# flight. It holds a STRONG reference to the kernel and compares with `is` — NOT with
# id(), because CPython reuses an id after its object is collected, so a freed kernel's id
# can be handed to a different episode and silently return that episode's plan.
_KEEP_PLAN_CACHE: Optional[tuple[EpisodeKernel, tuple[int, ...], "_KeepPlan"]] = None


def _keep_plan(kernel: EpisodeKernel, shapes: list[tuple[int, int]]) -> "_KeepPlan":
    """Cached :class:`_KeepPlan` for this episode's geometry.

    :param kernel: Episode kernel; matched by object identity.
    :type kernel: EpisodeKernel
    :param shapes: Per-step table shapes — part of the key because the construal widths
        differ between the marginalized and full-visibility modes of the SAME kernel.
    :type shapes: list[tuple[int, int]]
    :return: The cached plan, building it on first use.
    :rtype: _KeepPlan
    """
    global _KEEP_PLAN_CACHE
    key = tuple(m for _, m in shapes)
    cached = _KEEP_PLAN_CACHE
    if cached is not None and cached[0] is kernel and cached[1] == key:
        return cached[2]
    plan = _KeepPlan(kernel, shapes)
    _KEEP_PLAN_CACHE = (kernel, key, plan)
    return plan


def _mask_target_positions(kernel: EpisodeKernel, t: int, m_src: int, m_next: int) -> np.ndarray:
    """Construal re-label as enumeration positions: source position → target position.

    When the tables carry a single construal column (full-visibility mode, M=1), the
    re-label is trivially 0 → 0; otherwise the identity-tracking lookup table applies.

    :param kernel: Episode kernel.
    :type kernel: EpisodeKernel
    :param t: Source trajectory step (transition t → t+1).
    :type t: int
    :param m_src: Construal width of the step-t tables.
    :type m_src: int
    :param m_next: Construal width of the step-(t+1) tables.
    :type m_next: int
    :return: Target mask positions in the step-(t+1) enumeration, shape ``(m_src,)``.
    :rtype: np.ndarray
    """
    if m_src == 1 and m_next == 1:
        return np.zeros(1, dtype=np.int64)             # full visibility: all-ones → all-ones
    src_ints = kernel.masks_at(t)
    tgt_ints = kernel.mask_map[t][src_ints]
    nv_next = int(kernel.n_valid[t + 1])
    return (tgt_ints & ((1 << nv_next) - 1)).astype(np.int64)


def _keep_maps(kernel: EpisodeKernel, t: int, m_src: int, m_next: int,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source-side keep-route maps for the transition t → t+1.

    :param kernel: Episode kernel.
    :type kernel: EpisodeKernel
    :param t: Source trajectory step.
    :type t: int
    :param m_src: Construal width of the step-t tables.
    :type m_src: int
    :param m_next: Construal width of the step-(t+1) tables.
    :type m_next: int
    :return: ``(alive_g, goal_target, mask_target)`` — bool ``(G,)`` for non-ended
        goals, their target goal indices ``(G,)`` (valid where alive), and mask target
        positions ``(m_src,)``.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    ended = kernel.goal_ended[t]                       # (G,) — mask-independent (§3)
    alive = ~ended
    bins = np.arange(kernel.G) % kernel.NB
    shift = int(kernel.goal_shift[t])
    g_tgt = np.arange(kernel.G) - shift                # same lane, re-labeled bin
    # two-sided guard matching the kernel's off-grid termination: negative advance
    # (crash knockback) shifts bins UP and can push them past the far end of the grid
    tgt_bins = bins - shift
    alive = alive & (tgt_bins >= 0) & (tgt_bins < kernel.NB)
    return alive, g_tgt, _mask_target_positions(kernel, t, m_src, m_next)


def forward_pass(
    log_rho: list[np.ndarray],
    log_b: list[np.ndarray],
    kernel: EpisodeKernel,
) -> float:
    """The forward recursion: the episode log-likelihood.

    :param log_rho: Per-step ``log ρ_t`` tables, shapes ``(G, M_t)``.
    :type log_rho: list[np.ndarray]
    :param log_b: Per-step ``log b_t`` tables, same shapes.
    :type log_b: list[np.ndarray]
    :param kernel: Episode kernel (terminations + re-labelings).
    :type kernel: EpisodeKernel
    :return: ``logL``, the sum over steps of the normalization constant absorbed at each.
    :rtype: float
    """
    N = kernel.N
    consts = np.empty(N, dtype=np.float64)
    # θ-free transition bookkeeping, computed once per episode and reused by every
    # evaluation (see :class:`_KeepPlan`)
    plan = _keep_plan(kernel, [a.shape for a in log_rho])

    alpha = log_rho[0] + log_b[0]
    c = _logsumexp_all(alpha)
    alpha = alpha - c
    consts[0] = c

    for t in range(N - 1):
        nxt_rho = log_rho[t + 1]
        alive, src_g = plan.alive[t], plan.src_g[t]
        freed = _logsumexp_all(alpha[~alive, :])
        keep = np.full_like(nxt_rho, NEG_INF)
        if src_g.size:
            # move every surviving (g, m) cell to its re-labeled position. The mask
            # relabel is one-to-one on ~86% of steps (plain assignment); it merges
            # only when a departing car makes two attention patterns identical, and
            # merged cells must be COMBINED in log space, not overwritten.
            vals = alpha[src_g, :].ravel()
            flat, kflat = plan.flat_tgt[t], keep.reshape(-1)
            if plan.injective[t]:
                kflat[flat] = vals
            else:
                vmax = float(vals.max())
                if np.isfinite(vmax):
                    lin = np.bincount(flat, weights=np.exp(vals - vmax),
                                      minlength=plan.next_size[t])
                    with np.errstate(divide="ignore"):
                        kflat[:] = np.log(lin) + vmax
                # all -inf ⇒ keep stays at NEG_INF, which is already the fill
        # combine the two routes and weight by the action likelihood, writing into
        # reused buffers rather than allocating three fresh (G, M) arrays per step
        alpha = np.add(nxt_rho, freed)
        np.logaddexp(keep, alpha, out=alpha)
        np.add(log_b[t + 1], alpha, out=alpha)
        c = _logsumexp_all(alpha)
        alpha = alpha - c
        consts[t + 1] = c

    return float(consts.sum())
