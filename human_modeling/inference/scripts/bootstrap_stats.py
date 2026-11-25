"""
Two shared statistics, both reported with a percentile-bootstrap confidence interval per
CLAUDE.md *Verification & Evidence* ("prioritize the confidence interval; add a p-value only
where one is the traditional currency"), for the two different questions this package's scripts
ask of paired (x, y) data:

    :func:`bootstrap_correlation` -- Pearson r (plus a p-value: a correlation IS one of those
        traditional-p-value contexts) for "is there a relationship between two independently-
        varying quantities" (`plot_theta_vs_collisions.py`: a fitted coefficient vs. a
        collision count).
    :func:`bootstrap_spearman` -- Spearman rank correlation (plus a p-value) for the same
        question when the relationship is monotonic but NOT linear, notably parameter recovery
        (`analyze_param_recovery.py`), where clamping at a trained bound bends an intact
        ordering into a nonlinear one.

WHY BOOTSTRAP RATHER THAN FISHER'S z (bootstrap_correlation specifically)
    Fisher's z-transform (Fisher 1921, "On the Probable Error of a Coefficient of Correlation
    Deduced from a Small Sample," Metron 1:3-32) is the classical closed-form interval for r,
    but it assumes the underlying (x, y) pairs are bivariate normal. That assumption does not
    hold for this project's own data: fitted coefficients sit railed at a trained bound for a
    large share of episodes, and collision counts are non-negative integers, not a normal
    variable. The percentile bootstrap (Efron 1979, "Bootstrap Methods: Another Look at the
    Jackknife," Annals of Statistics 7:1-26; Efron & Tibshirani 1993, "An Introduction to the
    Bootstrap") makes no distributional assumption and reuses the exact resampling pattern
    already used elsewhere in this package (`analyze_peak_vs_mean.py`'s `paired_table`).

Sample reading: ``{"r": 0.278, "ci_lo": 0.152, "ci_hi": 0.398, "p": 0.012, "n": 230}``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr   # rank correlation for bootstrap_spearman

N_BOOT = 5000   # matches analyze_peak_vs_mean.py's paired bootstrap


def group_clusters(clusters: np.ndarray) -> list[np.ndarray]:
    """Row indices per cluster, computed ONCE so a bootstrap loop need not redo it per draw.

    :param clusters: Cluster label per row, shape ``(n,)``.
    :type clusters: np.ndarray
    :return: One index array per distinct label, in sorted label order.
    :rtype: list[np.ndarray]
    """
    return [np.flatnonzero(clusters == label) for label in np.unique(clusters)]


def bootstrap_indices(
    rng: np.random.Generator, n: int,
    clusters: np.ndarray | list[np.ndarray] | None = None,
) -> np.ndarray:
    """Row indices for one bootstrap draw, resampling CLUSTERS when the rows are not independent.

    With `clusters` left as None this is the ordinary case: draw `n` row indices with
    replacement. Pass `clusters` when several rows share one underlying unit -- the parameter
    recovery study fits 6 episodes per synthetic agent and all 6 carry that agent's single true
    theta, so the rows are not 144 independent observations but 24 clusters of 6. Resampling
    rows there would treat the repeats as new information and return an interval that is too
    narrow. The cluster bootstrap (Field & Welsh 2007, "Bootstrapping clustered data," JRSS-B
    69:369-390) instead draws whole clusters with replacement.

    Sample reading: with ``clusters = [0,0,1,1,2,2]`` a draw might pick clusters ``[2,0,2]``
    and return the row indices ``[4,5,0,1,4,5]``.

    :param rng: Seeded generator, so a re-run reproduces the draw.
    :type rng: np.random.Generator
    :param n: Number of rows.
    :type n: int
    :param clusters: Cluster label per row, shape ``(n,)``, or the grouped form
        :func:`group_clusters` returns, or None for independent rows. Pass the grouped form
        inside a loop: grouping is O(n) per call and the draw itself is not, so regrouping on
        every draw dominated the cost until it was hoisted out.
    :type clusters: np.ndarray | list[np.ndarray] | None
    :return: Row indices for this draw; length `n` when unclustered, and close to `n` (exactly
        `n` when clusters are equal-sized) when clustered.
    :rtype: np.ndarray
    """
    if clusters is None:
        return rng.choice(np.arange(n), n)
    members = clusters if isinstance(clusters, list) else group_clusters(clusters)
    drawn = rng.choice(len(members), len(members))
    return np.concatenate([members[i] for i in drawn])


def bootstrap_correlation(
    x: np.ndarray, y: np.ndarray, seed: int = 0,
    clusters: np.ndarray | None = None,
) -> dict[str, float]:
    """Pearson r with a percentile bootstrap CI and a two-sided p-value against rho=0.

    :param x: First variable.
    :type x: np.ndarray
    :param y: Second variable, same length as `x`.
    :type y: np.ndarray
    :param seed: Bootstrap RNG seed, fixed by default so a re-run reproduces the same interval.
    :type seed: int
    :param clusters: Cluster label per pair when the pairs are not independent; see
        :func:`bootstrap_indices`. None treats every pair as independent.
    :type clusters: np.ndarray | None
    :return: ``{"r", "ci_lo", "ci_hi", "p", "n"}``; `r`/`ci_lo`/`ci_hi`/`p` are NaN below 3
        usable pairs or when `x` has no spread.
    :rtype: dict[str, float]
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x_ok, y_ok = x[ok], y[ok]
    clusters_ok = None if clusters is None else group_clusters(np.asarray(clusters)[ok])
    n = int(ok.sum())
    if n < 3 or np.var(x_ok) == 0:
        return {"r": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p": float("nan"), "n": n}
    r = float(np.corrcoef(x_ok, y_ok)[0, 1])
    rng = np.random.default_rng(seed)
    boot = np.empty(N_BOOT)
    with np.errstate(invalid="ignore"):   # a degenerate draw's 0/0 is handled below, not a bug
        for b in range(N_BOOT):
            draw = bootstrap_indices(rng, n, clusters_ok)
            # | A resample can repeat the same episode enough times to erase all spread in x
            # | or y -- likely at small n, rare but not impossible at n=230 (repeated
            # | collision counts). corrcoef then divides by a zero std and returns NaN; the
            # | filtering below drops those draws instead of letting one NaN poison the CI.
            boot[b] = np.corrcoef(x_ok[draw], y_ok[draw])[0, 1]
    # | boot[b] <= 0 evaluates NaN <= 0 as False, not NaN -- a degenerate draw would silently
    # | count toward "not in the tail" if left in. Filter to the finite draws FIRST, then
    # | compute both the interval and the tail proportions from that same filtered array.
    boot_valid = boot[np.isfinite(boot)]
    if boot_valid.size < 100:
        return {"r": r, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p": float("nan"), "n": n}
    ci_lo, ci_hi = (float(v) for v in np.percentile(boot_valid, [2.5, 97.5]))
    # | Two-sided percentile-bootstrap p-value against rho=0 (Davison & Hinkley 1997,
    # | "Bootstrap Methods and Their Application," Cambridge University Press, SS4.4): twice
    # | the smaller of the two tails that cross zero, over the valid (non-degenerate) draws.
    p = float(2 * min(np.mean(boot_valid <= 0), np.mean(boot_valid >= 0)))
    return {"r": r, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": min(p, 1.0), "n": n}


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, seed: int = 0,
    clusters: np.ndarray | None = None,
) -> dict[str, float]:
    """Spearman rank correlation with a percentile bootstrap CI and a two-sided p-value.

    Prefer this over :func:`bootstrap_correlation` when the relationship is expected to be
    MONOTONIC but not linear. The case it was added for is parameter recovery
    (`analyze_param_recovery.py`): an estimate is clamped to the trained bound, so every truth
    beyond that bound maps to the same estimate. Clamping is monotone but strongly nonlinear —
    it depresses Pearson r while leaving the ordering intact, and the compression it causes is
    already reported separately as RMSE and bias, so a Pearson r there would count the same
    defect twice. Prefer :func:`bootstrap_correlation` when the question is how much of one
    variable is LINEARLY predictable from the other, or for continuity with an r published
    earlier. Both return the same keys apart from the statistic's own name.

    Ranks are recomputed INSIDE each bootstrap draw rather than once over the full sample: a
    resample changes the ranking, so holding the original ranks fixed would build the interval
    for a different statistic than the point estimate.

    Sample reading: ``{"rho": 0.412, "ci_lo": 0.201, "ci_hi": 0.588, "p": 0.002, "n": 144}``.

    :param x: First variable.
    :type x: np.ndarray
    :param y: Second variable, same length as `x`.
    :type y: np.ndarray
    :param seed: Bootstrap RNG seed, fixed by default so a re-run reproduces the same interval.
    :type seed: int
    :param clusters: Cluster label per pair when the pairs are not independent; see
        :func:`bootstrap_indices`. None treats every pair as independent.
    :type clusters: np.ndarray | None
    :return: ``{"rho", "ci_lo", "ci_hi", "p", "n"}``; every key but `n` is NaN below 3 usable
        pairs or when either variable has no spread.
    :rtype: dict[str, float]
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x_ok, y_ok = x[ok], y[ok]
    clusters_ok = None if clusters is None else group_clusters(np.asarray(clusters)[ok])
    n = int(ok.sum())
    nan_result = {"rho": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                  "p": float("nan"), "n": n}
    if n < 3 or np.var(x_ok) == 0 or np.var(y_ok) == 0:
        return nan_result
    rho = float(spearmanr(x_ok, y_ok).statistic)
    rng = np.random.default_rng(seed)
    boot = np.empty(N_BOOT)
    with np.errstate(invalid="ignore"):   # a degenerate draw's 0/0 is handled below, not a bug
        for b in range(N_BOOT):
            draw = bootstrap_indices(rng, n, clusters_ok)
            # | A resample can repeat one point enough times to erase all spread on a side,
            # | which makes the rank correlation undefined (NaN). Those draws are filtered out
            # | below rather than allowed to poison the interval — same handling as
            # | bootstrap_correlation.
            boot[b] = spearmanr(x_ok[draw], y_ok[draw]).statistic
    boot_valid = boot[np.isfinite(boot)]
    if boot_valid.size < 100:
        return {**nan_result, "rho": rho}
    ci_lo, ci_hi = (float(v) for v in np.percentile(boot_valid, [2.5, 97.5]))
    # | Two-sided percentile-bootstrap p-value against rho=0, computed from the SAME draws as
    # | the interval so the two can never disagree (Davison & Hinkley 1997, "Bootstrap Methods
    # | and Their Application," Cambridge University Press, SS4.4).
    p = float(2 * min(np.mean(boot_valid <= 0), np.mean(boot_valid >= 0)))
    return {"rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": min(p, 1.0), "n": n}


def format_p(p: float) -> str:
    """``"p<0.001"`` below that floor (never claim an exact zero probability), else 3 dp.

    :param p: A p-value, or NaN.
    :type p: float
    :return: A print-ready string.
    :rtype: str
    """
    if not np.isfinite(p):
        return "p=n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"
