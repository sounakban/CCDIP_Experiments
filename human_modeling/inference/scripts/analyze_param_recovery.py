"""
Parameter recovery read-out: how close is a fitted coefficient to the truth that generated
the episode, on synthetic data where that truth is known exactly.

Three numbers per coefficient, chosen so that each answers a different question and none
double-counts another:

    correlation -- does a higher true value produce a higher estimate (ORDERING)
    RMSE        -- how far is the estimate from the truth on average (ACCURACY)
    bias        -- does the estimator run systematically high or low (OFFSET)

WHY SPEARMAN IS THE HEADLINE CORRELATION HERE
    An estimate is clamped to the trained bound, so every truth beyond that bound maps to the
    same estimate at the edge. Clamping is monotone but strongly nonlinear: it depresses a
    Pearson r while leaving the ordering intact. Since the compression it causes is already
    reported by RMSE and bias, a Pearson headline would count the same defect twice. Pearson
    is still computed and written to the CSV, because the earlier recovery study's
    `param_r2.csv` tables (not kept in this repository) published a squared Pearson (verified:
    40 of 40 stored rows reproduce as r**2, 0 of 40 as variance-explained R2), and dropping it
    would break continuity with that record. See implementation.md Entry 29.

WHAT THE CEILING COLUMN MEANS
    A correlation near zero has two explanations -- no signal, or an estimate too noisy for
    any relationship to show -- and only a ceiling separates them (CLAUDE.md
    *Quantitative Reporting*, "every R2 is reported with its CEILING"). Each synthetic person
    is fitted on several episodes that share ONE true theta, so the estimate's own reliability
    is measurable: split those episodes in half, correlate the two half-means across people,
    average over every balanced split, then Spearman-Brown (1910) to the length actually used.
    Because the truth carries no measurement error, the achievable correlation is
    sqrt(reliability) (Spearman 1904 attenuation with a perfect criterion). The split-half is
    a Pearson one, which is what Spearman-Brown assumes, so the ceiling bounds the Pearson
    column exactly and the Spearman column approximately.

    What that reliability does and does not cover: both studies simulate each agent ONCE per
    route and never repeat a route, so the two halves differ by ROUTE as well as by rollout
    noise. The number therefore means "does this agent's estimate reproduce on other routes",
    which is the property recovery needs -- but it is not pure sampling noise and cannot be
    decomposed into the two without repeated rollouts of the same route.

Typical usage::

    uv run python -m human_modeling.inference.scripts.analyze_param_recovery \\
        --recovery-csv human_modeling/inference/results/model_comparison/recovery/recovery_rows_mean.csv \\
        --hlp-checkpoint-dir human_modeling/inference/checkpoint \\
        --out-dir human_modeling/inference/results/model_comparison/recovery/metrics_A_full_mean
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from human_modeling.inference.model.choice import FIT_NAMES, HEURISTIC_BOUNDS, THETA_NAMES
from human_modeling.inference.scripts.bootstrap_stats import (
    N_BOOT, bootstrap_correlation, bootstrap_indices, bootstrap_spearman, format_p,
    group_clusters,
)

# | Single categorical slot 1 of the validated default palette.
COLOR_POINT = "#2a78d6"     # validated: passes lightness, chroma and 3:1 contrast on white
COLOR_TREND = "#eb6834"     # slot 2; with slot 1 this is the pair already used across the package
COLOR_REFERENCE = "0.45"    # gray for the y = x identity line; recessive against both

EPISODES_PER_PERSON = 6     # what the study uses; the code DERIVES it, never assumes it
POOLED = "all"              # the `stratum` label every row carries: the design has no strata


# ===========================================================================
# 1. Inputs
# ===========================================================================


def read_csv(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV as string dicts.

    :param path: File to read.
    :type path: Path
    :return: One dict per row, keyed by header.
    :rtype: list[dict[str, str]]
    :raises FileNotFoundError: If `path` does not exist.
    """
    with io.open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def coefficients_in(rows: list[dict[str, str]]) -> tuple[str, ...]:
    """Which coefficients a recovery CSV carries, in `FIT_NAMES` order.

    A theta-only study (the shipped 4-coefficient fit) has `true_`/`fit_` columns for the four
    theta; a study on a heuristic variant (Entry 35) adds the fitted weights. Reading the set
    off the file, rather than assuming four, is what lets one read-out serve both.

    :param rows: All rows of the recovery CSV.
    :type rows: list[dict[str, str]]
    :return: The coefficient names present as both a truth and a fit column.
    :rtype: tuple[str, ...]
    :raises ValueError: If the file carries no coefficient at all.
    """
    columns = set(rows[0]) if rows else set()
    names = tuple(n for n in FIT_NAMES if f"true_{n}" in columns and f"fit_{n}" in columns)
    if not names:
        raise ValueError("the recovery CSV has no true_<coef>/fit_<coef> column pair")
    return names


def coefficient_ranges(checkpoint_dir: Path, names: tuple[str, ...] = THETA_NAMES) -> dict[str, float]:
    """Width of each fitted coefficient's range: trained range for theta, the calibrated
    +-bound for a heuristic weight.

    The ranges differ by a factor of ~67 across coefficients (0.15 for the cognitive cost
    against 10.0 for the collision penalty), so a raw RMSE cannot be compared between them.
    Every accuracy number is therefore reported a second time as a fraction of this width.
    Theta widths come from the checkpoint's own metadata; a heuristic weight was never
    trained, so its width is ``2 * HEURISTIC_BOUNDS[name]`` -- the box the fit searches and
    the truths are drawn from (implementation.md Entries 33 and 35).

    :param checkpoint_dir: Directory holding `hlp_metadata.json`.
    :type checkpoint_dir: Path
    :param names: The coefficients to report on, in the order the table should list them.
    :type names: tuple[str, ...]
    :return: ``{coefficient_name: width}`` for every name, in the given order.
    :rtype: dict[str, float]
    :raises FileNotFoundError: If `hlp_metadata.json` is absent.
    :raises KeyError: If a theta name has no entry in `hlp_param_config`.
    """
    meta_path = checkpoint_dir / "hlp_metadata.json"
    with io.open(meta_path, encoding="utf-8") as fh:
        config = {entry["name"]: entry for entry in json.load(fh)["hlp_param_config"]}
    # | Sample reading: {"name": "hlp_collision_penalty", "min": 0.0, "max": 10.0} -> 10.0
    return {name: (float(config[name]["max"]) - float(config[name]["min"]) if name in THETA_NAMES
                   else 2.0 * HEURISTIC_BOUNDS[name])
            for name in names}


def theta_ranges(checkpoint_dir: Path) -> dict[str, float]:
    """:func:`coefficient_ranges` for the four theta -- the pre-Entry-35 spelling, kept for
    its callers and tests.

    :param checkpoint_dir: Directory holding `hlp_metadata.json`.
    :type checkpoint_dir: Path
    :return: ``{coefficient_name: max - min}`` for every coefficient in `THETA_NAMES`.
    :rtype: dict[str, float]
    """
    return coefficient_ranges(checkpoint_dir, THETA_NAMES)


def select(rows: list[dict[str, str]], level: str) -> list[dict[str, str]]:
    """The subset of recovery rows at one aggregation level.

    :param rows: All rows of a `person_recovery.csv`.
    :type rows: list[dict[str, str]]
    :param level: ``"episode"`` (one row per fitted episode) or ``"person"`` (one row per
        synthetic agent, that agent's episodes already averaged by the producing script).
    :type level: str
    :return: The matching rows, in file order.
    :rtype: list[dict[str, str]]
    """
    return [r for r in rows if r["level"] == level]


# ===========================================================================
# 2. Statistics
# ===========================================================================


def rmse_and_bias(
    true_values: np.ndarray, estimates: np.ndarray, seed: int = 0,
    clusters: np.ndarray | None = None,
) -> dict[str, float]:
    """RMSE and mean bias of an estimate against a known truth, each with a bootstrap CI.

    Both use the user-stated definitions: ``RMSE = sqrt(mean((est - true)**2))`` and
    ``bias = mean(est - true)``. A positive bias means the estimator runs HIGH.

    The interval is a percentile bootstrap over the pairs, matching
    :func:`bootstrap_stats.bootstrap_correlation` so that every interval in one output table is
    built from the same resampling scheme.

    Sample reading: ``{"rmse": 2.354, "rmse_ci_lo": 1.98, "rmse_ci_hi": 2.71,
    "bias": -0.195, "bias_ci_lo": -0.62, "bias_ci_hi": 0.23, "n": 144}``.

    :param true_values: The known generating value per episode or person.
    :type true_values: np.ndarray
    :param estimates: The fitted value for the same episodes or people, same length.
    :type estimates: np.ndarray
    :param seed: Bootstrap RNG seed, fixed so a re-run reproduces the same interval.
    :type seed: int
    :param clusters: Cluster label per pair when the pairs are not independent; see
        :func:`bootstrap_stats.bootstrap_indices`. None treats every pair as independent.
    :type clusters: np.ndarray | None
    :return: ``{"rmse", "rmse_ci_lo", "rmse_ci_hi", "bias", "bias_ci_lo", "bias_ci_hi", "n"}``;
        every key but `n` is NaN below 3 usable pairs.
    :rtype: dict[str, float]
    """
    ok = np.isfinite(true_values) & np.isfinite(estimates)
    residuals = estimates[ok] - true_values[ok]                # (n_usable,)
    clusters_ok = None if clusters is None else group_clusters(np.asarray(clusters)[ok])
    n = int(ok.sum())
    if n < 3:
        return {k: float("nan") for k in
                ("rmse", "rmse_ci_lo", "rmse_ci_hi", "bias", "bias_ci_lo", "bias_ci_hi")} | {"n": n}
    rng = np.random.default_rng(seed)
    boot_rmse = np.empty(N_BOOT)
    boot_bias = np.empty(N_BOOT)
    for b in range(N_BOOT):
        draw = bootstrap_indices(rng, n, clusters_ok)
        boot_rmse[b] = np.sqrt(np.mean(residuals[draw] ** 2))
        boot_bias[b] = np.mean(residuals[draw])
    rmse_lo, rmse_hi = (float(v) for v in np.percentile(boot_rmse, [2.5, 97.5]))
    bias_lo, bias_hi = (float(v) for v in np.percentile(boot_bias, [2.5, 97.5]))
    return {"rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "rmse_ci_lo": rmse_lo, "rmse_ci_hi": rmse_hi,
            "bias": float(np.mean(residuals)),
            "bias_ci_lo": bias_lo, "bias_ci_hi": bias_hi, "n": n}


def spearman_brown(reliability: float, length_ratio: float) -> float:
    """Rescale a reliability to a different number of averaged observations.

    Spearman (1910) / Brown (1910): averaging more noisy observations of the same underlying
    value raises reliability along ``m*r / (1 + (m-1)*r)``, where `m` is how many times longer
    the new aggregate is.

    :param reliability: Reliability measured at the original length.
    :type reliability: float
    :param length_ratio: New length divided by the measured length; 2.0 doubles it, 1/3 cuts it
        to a third.
    :type length_ratio: float
    :return: Reliability at the new length, or NaN if the input is NaN or the formula is
        undefined (a reliability at or below ``-1/(m-1)``).
    :rtype: float
    """
    if not np.isfinite(reliability):
        return float("nan")
    denominator = 1.0 + (length_ratio - 1.0) * reliability
    if denominator <= 0:
        return float("nan")
    return float(length_ratio * reliability / denominator)


def episodes_per_person(rows: list[dict[str, str]]) -> int:
    """How many episodes each person has, required to be the same for all of them.

    Derived rather than assumed: the two studies this script reads both use 6, but a smoke run
    uses 2, and the split-half below divides by this number. Hardcoding it made the read-out
    refuse any run that was not the full study.

    :param rows: Episode-level rows.
    :type rows: list[dict[str, str]]
    :return: The shared episode count.
    :rtype: int
    :raises ValueError: If people differ in episode count, or if there are fewer than 2 --
        a split-half needs at least one episode on each side.
    """
    counts = {}
    for r in rows:
        counts[r["person"]] = counts.get(r["person"], 0) + 1
    distinct = set(counts.values())
    if len(distinct) != 1:
        raise ValueError(f"people have differing episode counts {sorted(distinct)}; the "
                         f"split-half reliability assumes a balanced design")
    count = distinct.pop()
    if count < 2:
        raise ValueError(f"{count} episode(s) per person: a split-half needs at least 2")
    return count


def split_half_reliability(rows: list[dict[str, str]], coefficient: str) -> float:
    """Reliability of a ONE-episode estimate, from disjoint halves of each person's episodes.

    Every episode of a synthetic person was generated from that person's single true theta, so
    two disjoint halves of their episodes give two independent estimates of the same quantity.
    Correlating those halves across people measures how reproducible the estimate is,
    independently of whether it is correct.

    Numbered stages:
    1. Group the episode rows by person and derive the shared episode count.
    2. For each of the 10 balanced 3-versus-3 splits, average each half and correlate the two
       halves across people; average the 10 correlations. Averaging over every split rather
       than picking one removes the arbitrariness of which episodes land together, which
       matters here because each episode is a different map.
    3. Spearman-Brown that half-length reliability down to a single episode.

    :param rows: Episode-level rows.
    :type rows: list[dict[str, str]]
    :param coefficient: Which fitted coefficient to measure.
    :type coefficient: str
    :return: Reliability of a single episode's estimate, or NaN if it cannot be computed.
    :rtype: float
    :raises ValueError: If the design is unbalanced or has fewer than 2 episodes per person.
    """
    # | 1. Group by person. `person` is unique within the file, so it alone keys the group.
    by_person: dict[str, list[float]] = {}
    for r in rows:
        by_person.setdefault(r["person"], []).append(float(r[f"fit_{coefficient}"]))
    people = sorted(by_person)
    n_episodes = episodes_per_person(rows)
    if len(people) < 3:
        return float("nan")

    # | 2. Every balanced split, counted once: fixing episode 0 in the first half removes the
    # | mirror-image duplicate of each split, leaving C(5,2) = 10 distinct ones.
    half = n_episodes // 2
    halves = [(0,) + rest for rest in itertools.combinations(range(1, n_episodes), half - 1)]
    correlations: list[float] = []
    for first in halves:
        second = tuple(i for i in range(n_episodes) if i not in first)
        a = np.array([np.mean([by_person[p][i] for i in first]) for p in people])     # (n_people,)
        b = np.array([np.mean([by_person[p][i] for i in second]) for p in people])    # (n_people,)
        if np.var(a) == 0 or np.var(b) == 0:
            continue
        correlations.append(float(np.corrcoef(a, b)[0, 1]))
    if not correlations:
        return float("nan")

    # | 3. The measured value is the reliability of a `half`-episode mean; scale it to one.
    return spearman_brown(float(np.mean(correlations)), 1.0 / half)


def ceiling_for(reliability_one: float, level: str, n_episodes: int = EPISODES_PER_PERSON) -> float:
    """Highest correlation an estimate of this reliability could reach against an exact truth.

    :param reliability_one: Reliability of a single episode's estimate.
    :type reliability_one: float
    :param n_episodes: Episodes behind a person-level estimate; ignored at episode level.
    :type n_episodes: int
    :param level: ``"episode"`` (the estimate comes from one episode) or ``"person"``.
        NOTE: a person-level estimate is a JOINT fit over that person's `EPISODES_PER_PERSON`
        episodes (one summed log-likelihood), not the mean of their episode fits -- MEASURED
        to differ from that mean by up to 0.42 on the collision penalty's 0-10 range. Scaling
        the reliability by `EPISODES_PER_PERSON` is therefore an APPROXIMATION: it assumes the
        joint fit's sampling variance falls like that of a 6-fold average, which holds under
        the usual regularity conditions but was not verified here.
    :type level: str
    :return: ``sqrt(reliability)`` at this level, or NaN if undefined.
    :rtype: float
    """
    reliability = (reliability_one if level == "episode"
                   else spearman_brown(reliability_one, float(n_episodes)))
    if not np.isfinite(reliability) or reliability <= 0:
        return float("nan")
    return float(np.sqrt(min(reliability, 1.0)))


def recovery_row(
    rows: list[dict[str, str]], episode_rows: list[dict[str, str]], coefficient: str,
    range_width: float, level: str, seed: int,
) -> dict[str, float | str]:
    """Every statistic for one (level, coefficient) cell, as one output row.

    :param rows: The rows at this level, used for the statistics themselves.
    :type rows: list[dict[str, str]]
    :param episode_rows: The EPISODE-level rows, used only for the ceiling -- the
        reliability needs the individual episodes even when the cell is person-level.
    :type episode_rows: list[dict[str, str]]
    :param coefficient: Which fitted coefficient this row describes.
    :type coefficient: str
    :param range_width: Width of that coefficient's trained range, for the normalized columns.
    :type range_width: float
    :param level: ``"episode"`` or ``"person"``.
    :type level: str
    :param seed: Bootstrap RNG seed.
    :type seed: int
    :return: One flat record; see `FIELDNAMES` for the key order written to disk.
    :rtype: dict[str, float | str]
    """
    true_values = np.array([float(r[f"true_{coefficient}"]) for r in rows])   # (n,)
    estimates = np.array([float(r[f"fit_{coefficient}"]) for r in rows])      # (n,)
    agents = np.array([r["person"] for r in rows])                            # (n,)
    rank = bootstrap_spearman(true_values, estimates, seed=seed, clusters=agents)
    linear = bootstrap_correlation(true_values, estimates, seed=seed, clusters=agents)
    accuracy = rmse_and_bias(true_values, estimates, seed=seed, clusters=agents)
    reliability_one = split_half_reliability(episode_rows, coefficient)
    return {
        "stratum": POOLED, "level": level, "coefficient": coefficient, "n": accuracy["n"],
        "spearman_rho": rank["rho"], "spearman_ci_lo": rank["ci_lo"],
        "spearman_ci_hi": rank["ci_hi"], "spearman_p": rank["p"],
        "pearson_r": linear["r"], "pearson_ci_lo": linear["ci_lo"],
        "pearson_ci_hi": linear["ci_hi"], "pearson_p": linear["p"],
        "rmse": accuracy["rmse"], "rmse_ci_lo": accuracy["rmse_ci_lo"],
        "rmse_ci_hi": accuracy["rmse_ci_hi"], "rmse_frac_range": accuracy["rmse"] / range_width,
        "bias": accuracy["bias"], "bias_ci_lo": accuracy["bias_ci_lo"],
        "bias_ci_hi": accuracy["bias_ci_hi"], "bias_frac_range": accuracy["bias"] / range_width,
        "estimate_reliability": reliability_one,
        "correlation_ceiling": ceiling_for(reliability_one, level,
                                           episodes_per_person(episode_rows)),
        "range_width": range_width,
    }


FIELDNAMES = ["stratum", "level", "coefficient", "n",
              "spearman_rho", "spearman_ci_lo", "spearman_ci_hi", "spearman_p",
              "pearson_r", "pearson_ci_lo", "pearson_ci_hi", "pearson_p",
              "rmse", "rmse_ci_lo", "rmse_ci_hi", "rmse_frac_range",
              "bias", "bias_ci_lo", "bias_ci_hi", "bias_frac_range",
              "estimate_reliability", "correlation_ceiling", "range_width"]


# ===========================================================================
# 3. Figures
# ===========================================================================


def pretty(coefficient: str) -> str:
    """A coefficient's name with the package prefix and underscores removed, for an axis label.

    :param coefficient: A name from `THETA_NAMES`.
    :type coefficient: str
    :return: Display form, e.g. ``"collision penalty"``.
    :rtype: str
    """
    return coefficient.removeprefix("hlp_").replace("_coef", "").replace("_", " ")


def shared_limits(
    rows: list[dict[str, str]], coefficient: str, level: str, range_width: float,
) -> tuple[float, float]:
    """One pair of axis limits covering the truths AND the estimates of one coefficient at
    one level, so how much of the trained range the truths actually span stays visible. That
    matters for reading the correlations: a narrow band of truth mechanically depresses a
    correlation (restriction of range) whatever the estimator does.

    :param rows: All rows of the recovery CSV.
    :type rows: list[dict[str, str]]
    :param coefficient: Which coefficient the limits are for.
    :type coefficient: str
    :param level: ``"episode"`` or ``"person"``.
    :type level: str
    :param range_width: That coefficient's trained range, setting a floor on the padding.
    :type range_width: float
    :return: ``(low, high)``, already padded.
    :rtype: tuple[float, float]
    """
    subset = select(rows, level)
    values = np.concatenate([
        np.array([float(r[f"true_{coefficient}"]) for r in subset]),
        np.array([float(r[f"fit_{coefficient}"]) for r in subset]),
    ])
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * 0.06, range_width * 1e-3)
    return low - pad, high + pad


def draw_panel(
    axis: plt.Axes, true_values: np.ndarray, estimates: np.ndarray,
    stats: dict[str, float | str], limits: tuple[float, float], title: str,
) -> None:
    """One true-versus-estimated scatter: the identity line, a fitted trend, and the correlation.

    Two lines, doing different jobs. The dashed grey one is ``y = x``, the answer a perfect
    estimator would give, so the gap to it shows bias and compression. The solid one is a
    least-squares fit through the points, which shows the direction and slope the data actually
    take -- a slope below 1 against the identity line is the shrinkage a correlation alone
    cannot show. The fit is a VISUAL AID only: the reported statistic is a rank correlation,
    which no fitted line is the source of.

    Only the correlation is annotated. RMSE, bias, the ceiling and every interval stay in
    `param_recovery_metrics.csv`, because a figure read at a glance carries one number well
    and four badly.

    :param axis: Target axes.
    :type axis: plt.Axes
    :param true_values: Known generating values, ``(n,)``.
    :type true_values: np.ndarray
    :param estimates: Fitted values, ``(n,)``.
    :type estimates: np.ndarray
    :param stats: The record `recovery_row` produced for this same subset.
    :type stats: dict[str, float | str]
    :param limits: ``(low, high)`` applied to both axes, shared across this coefficient's
        panels by :func:`shared_limits`.
    :type limits: tuple[float, float]
    :param title: Panel title.
    :type title: str
    """
    low, high = limits
    axis.plot([low, high], [low, high], "--", color=COLOR_REFERENCE, linewidth=1.4,
              zorder=1, label="y = x (perfect)")
    # | A 2 px surface-colored ring separates points that overlap, which they do wherever
    # | several agents are fitted to nearly the same value.
    axis.scatter(true_values, estimates, s=46, c=COLOR_POINT, alpha=0.85,
                 edgecolors="white", linewidths=1.0, zorder=2)
    # | The trend is drawn only across the span of the truths it was fitted to, never
    # | extrapolated to the shared axis limits, so it cannot suggest data that is not there.
    finite = np.isfinite(true_values) & np.isfinite(estimates)
    if finite.sum() >= 2 and np.std(true_values[finite]) > 0:
        slope, intercept = np.polyfit(true_values[finite], estimates[finite], 1)
        span = np.linspace(true_values[finite].min(), true_values[finite].max(), 2)
        axis.plot(span, slope * span + intercept, color=COLOR_TREND, linewidth=1.8,
                  zorder=3, label="least-squares fit")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    # | Identical limits are NOT enough to make the panel symmetric: matplotlib runs a locator
    # | per axis, so the two can land on different steps. MEASURED on this study's own data --
    # | cognitive cost drew 0.025 steps on x against 0.02 on y, and goal flag distance 0.25
    # | against 0.2, while collision penalty happened to agree. Taking one tick set from the x
    # | axis and applying it to both is what makes every panel read as a square.
    ticks = [t for t in axis.xaxis.get_major_locator().tick_values(low, high)
             if low <= t <= high]
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.grid(alpha=0.18, linewidth=0.7)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    # | Titles are OFF by user request (2026-09-22): these figures go into documents where
    # | the caption carries the title, and a printed title duplicates it. Kept commented
    # | rather than deleted so restoring is a one-line uncomment.
    # axis.set_title(title, fontsize=10)
    axis.text(0.04, 0.96,
              f"rho = {stats['spearman_rho']:+.2f} "
              f"[{stats['spearman_ci_lo']:+.2f}, {stats['spearman_ci_hi']:+.2f}]\n"
              f"n = {stats['n']}",
              transform=axis.transAxes, ha="left", va="top", fontsize=8.5,
              linespacing=1.45, color="#2b2b2b")


def plot_coefficient(
    rows: list[dict[str, str]], table: list[dict[str, float | str]], coefficient: str,
    range_width: float, level: str, out_dir: Path,
) -> Path:
    """One figure per coefficient: the true-versus-estimated scatter.

    :param rows: All rows of the recovery CSV.
    :type rows: list[dict[str, str]]
    :param table: Every computed statistics record, searched for this figure's panels.
    :type table: list[dict[str, float | str]]
    :param coefficient: Which coefficient this figure shows.
    :type coefficient: str
    :param range_width: That coefficient's trained range.
    :type range_width: float
    :param level: ``"episode"`` or ``"person"``.
    :type level: str
    :param out_dir: Directory for the PNG and PDF.
    :type out_dir: Path
    :return: Path of the written PNG.
    :rtype: Path
    """
    fig, axes = plt.subplots(1, 1, figsize=(4.0, 4.4), squeeze=False)
    axis = axes[0][0]
    limits = shared_limits(rows, coefficient, level, range_width)
    subset = select(rows, level)
    stats = next(s for s in table if s["level"] == level and s["coefficient"] == coefficient)
    draw_panel(axis,
               np.array([float(r[f"true_{coefficient}"]) for r in subset]),
               np.array([float(r[f"fit_{coefficient}"]) for r in subset]),
               stats, limits, "")
    axis.set_xlabel(f"True {pretty(coefficient)}")
    axis.set_ylabel(f"Estimated {pretty(coefficient)}")
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    # | Titles are OFF by user request (2026-09-22): these figures go into documents where
    # | the caption carries the title, and a printed title duplicates it. Kept commented
    # | rather than deleted so restoring is a one-line uncomment.
    # fig.suptitle(f"Parameter recovery: {pretty(coefficient)} (one point per {level})",
    #              fontsize=12)
    fig.tight_layout()
    png_path = out_dir / f"param_recovery_{coefficient}_{level}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_overview(
    rows: list[dict[str, str]], table: list[dict[str, float | str]],
    ranges: dict[str, float], level: str, out_dir: Path,
) -> Path:
    """One grid holding every coefficient's scatter, two per row.

    :param rows: All rows of the recovery CSV.
    :type rows: list[dict[str, str]]
    :param table: Every computed statistics record.
    :type table: list[dict[str, float | str]]
    :param ranges: Trained range width per coefficient.
    :type ranges: dict[str, float]
    :param level: ``"episode"`` or ``"person"``.
    :type level: str
    :param out_dir: Directory for the PNG and PDF.
    :type out_dir: Path
    :return: Path of the written PNG.
    :rtype: Path
    """
    # | A single column of panels wastes the page and is hard to read, so the coefficients
    # | are laid out two per row instead.
    cells = list(ranges)
    n_cols = 2
    n_rows = -(-len(cells) // n_cols)            # ceiling division, so no cell is dropped
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.4 * n_rows),
                             squeeze=False)
    subset = select(rows, level)
    for index, coefficient in enumerate(cells):
        axis = axes[index // n_cols][index % n_cols]
        limits = shared_limits(rows, coefficient, level, ranges[coefficient])
        stats = next(s for s in table if s["level"] == level and s["coefficient"] == coefficient)
        draw_panel(axis,
                   np.array([float(r[f"true_{coefficient}"]) for r in subset]),
                   np.array([float(r[f"fit_{coefficient}"]) for r in subset]),
                   stats, limits, "")
        axis.set_ylabel(f"Estimated\n{pretty(coefficient)}")
        axis.set_xlabel(f"True {pretty(coefficient)}")
    for spare in range(len(cells), n_rows * n_cols):
        axes[spare // n_cols][spare % n_cols].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.015))
    # | Titles are OFF by user request (2026-09-22): these figures go into documents where
    # | the caption carries the title, and a printed title duplicates it. Kept commented
    # | rather than deleted so restoring is a one-line uncomment.
    # fig.suptitle(f"Parameter recovery on synthetic truths (one point per {level})", fontsize=13)
    fig.tight_layout()
    png_path = out_dir / f"param_recovery_overview_{level}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_confusion(
    rows: list[dict[str, str]], level: str, out_dir: Path, seed: int,
    names: tuple[str, ...] = THETA_NAMES,
) -> Path:
    """Every TRUE coefficient against every ESTIMATED one: recovery on the diagonal, trade-offs off it.

    Readable because the truths are drawn INDEPENDENTLY per coefficient (Entry 31's design),
    so an off-diagonal correlation is the estimator, not the sampling scheme. The
    diagonal repeats the headline recovery numbers; an off-diagonal cell says the estimator
    cannot separate those two coefficients -- pushing one truth up moves the OTHER estimate.

    Colour encodes the signed correlation on a diverging scale with a neutral midpoint at
    zero, since the sign is the meaning here (a trade-off runs one way or the other), and
    every cell is also printed as text so identity never rests on colour alone.

    :param rows: All rows of the recovery CSV.
    :type rows: list[dict[str, str]]
    :param level: ``"episode"`` or ``"person"``.
    :type level: str
    :param out_dir: Directory for the PNG and PDF.
    :type out_dir: Path
    :param seed: Bootstrap RNG seed, for the per-cell interval behind the significance mark.
    :type seed: int
    :param names: The coefficients to cross, in axis order; theta's four by default, the
        eight of a heuristic variant when the CSV carries them.
    :type names: tuple[str, ...]
    :return: Path of the written PNG.
    :rtype: Path
    """
    subset = select(rows, level)
    agents = np.array([r["person"] for r in subset])                          # (n,)
    grid = np.full((len(names), len(names)), np.nan)                          # (C, C)
    excludes_zero = np.zeros_like(grid, dtype=bool)
    for i, true_name in enumerate(names):
        true_values = np.array([float(r[f"true_{true_name}"]) for r in subset])
        for j, fit_name in enumerate(names):
            estimates = np.array([float(r[f"fit_{fit_name}"]) for r in subset])
            stats = bootstrap_spearman(true_values, estimates, seed=seed, clusters=agents)
            grid[i, j] = stats["rho"]
            excludes_zero[i, j] = np.isfinite(stats["ci_lo"]) and (
                stats["ci_lo"] > 0 or stats["ci_hi"] < 0)

    side = 7.2 if len(names) <= 4 else 1.6 * len(names)
    fig, axis = plt.subplots(figsize=(side, side - 0.8))
    # | Diverging scale, symmetric about zero so the neutral midpoint really is no relationship.
    image = axis.imshow(grid, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    # | Estimated-coefficient names go on the x axis, wrapped at the spaces so four of them
    # | fit side by side without overlapping.
    wrapped = [pretty(n).replace(" ", "\n") for n in names]
    axis.set_xticks(range(len(names)), wrapped, fontsize=9)
    axis.set_yticks(range(len(names)), [pretty(n) for n in names], fontsize=9)
    axis.set_xlabel("Estimated coefficient")
    axis.set_ylabel("True coefficient")
    # | Two short lines rather than one long one: a single line runs under the colorbar.
    # | Titles are OFF by user request (2026-09-22): these figures go into documents where
    # | the caption carries the title, and a printed title duplicates it. Kept commented
    # | rather than deleted so restoring is a one-line uncomment.
    # axis.set_title(f"Recovery on the diagonal, confusion off it\n(one point per {level})",
    #                fontsize=11)
    for i in range(len(names)):
        for j in range(len(names)):
            mark = "*" if excludes_zero[i, j] else ""
            # | White text on the saturated ends, dark in the pale middle, so every cell reads.
            colour = "white" if abs(grid[i, j]) > 0.55 else "#1b1b1b"
            axis.text(j, i, f"{grid[i, j]:+.2f}{mark}", ha="center", va="center",
                      color=colour, fontsize=11)
    fig.colorbar(image, ax=axis, label="Spearman rho", shrink=0.82)
    fig.tight_layout()
    png_path = out_dir / f"param_recovery_confusion_{level}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png_path


# ===========================================================================
# 4. Entry point
# ===========================================================================


def build_table(
    rows: list[dict[str, str]], ranges: dict[str, float], seed: int,
) -> list[dict[str, float | str]]:
    """Every (level, coefficient) cell.

    :param rows: All rows of the recovery CSV.
    :type rows: list[dict[str, str]]
    :param ranges: Trained range width per coefficient.
    :type ranges: dict[str, float]
    :param seed: Bootstrap RNG seed.
    :type seed: int
    :return: One record per cell, in write order.
    :rtype: list[dict[str, float | str]]
    """
    table: list[dict[str, float | str]] = []
    episode_rows = select(rows, "episode")
    for level in ("episode", "person"):
        level_rows = select(rows, level)
        for coefficient in ranges:
            table.append(recovery_row(level_rows, episode_rows, coefficient,
                                      ranges[coefficient], level, seed))
    return table


def print_table(table: list[dict[str, float | str]], level: str) -> None:
    """Print the three headline numbers per coefficient, ASCII only (Rule 12).

    :param table: Every computed statistics record.
    :type table: list[dict[str, float | str]]
    :param level: Which level to print.
    :type level: str
    """
    print(f"\n=== parameter recovery, one point per {level} "
          f"(rho = Spearman; RMSE and bias as a percentage of each trained range) ===")
    header = f"{'coefficient':<30}{'n':>4}{'rho [95% CI]':>26}{'p':>10}" \
             f"{'ceiling':>9}{'RMSE%':>8}{'bias%':>8}"
    print(header)
    print("-" * len(header))
    for coefficient in dict.fromkeys(str(r["coefficient"]) for r in table):
        s = next(r for r in table if r["coefficient"] == coefficient and r["level"] == level)
        ceiling = (f"{s['correlation_ceiling']:.2f}"
                   if np.isfinite(s["correlation_ceiling"]) else "n/a")
        interval = (f"{s['spearman_rho']:+.3f} "
                    f"[{s['spearman_ci_lo']:+.3f}, {s['spearman_ci_hi']:+.3f}]")
        print(f"{coefficient:<30}{s['n']:>4}{interval:>26}"
              f"{format_p(s['spearman_p']):>10}{ceiling:>9}"
              f"{100 * s['rmse_frac_range']:>7.0f}%{100 * s['bias_frac_range']:>7.0f}%")
    print()


def main() -> None:
    """CLI: compute the recovery table for one synthetic fit, write it, and draw its figures."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recovery-csv", required=True,
                        help="a recovery_rows_<how>.csv holding true_<coef>/fit_<coef> columns "
                             "plus level and person")
    parser.add_argument("--hlp-checkpoint-dir", required=True,
                        help="checkpoint whose hlp_metadata.json defines each coefficient's "
                             "trained range, for the percentage-of-range columns")
    parser.add_argument("--out-dir", required=True,
                        help="destination for param_recovery_metrics.csv and plots/")
    parser.add_argument("--seed", type=int, default=0,
                        help="bootstrap RNG seed; fixed so a re-run reproduces every interval")
    parser.add_argument("--levels", default="episode,person",
                        help="comma-separated aggregation levels to plot (default both); the "
                             "CSV always holds both. Ex: --levels person")
    args = parser.parse_args()

    rows = read_csv(Path(args.recovery_csv))
    names = coefficients_in(rows)
    ranges = coefficient_ranges(Path(args.hlp_checkpoint_dir), names)
    print(f"read {len(rows)} rows from {args.recovery_csv}; coefficients: {list(names)}")
    print(f"trained ranges: " + ", ".join(f"{k}={v:g}" for k, v in ranges.items()))

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"computing {N_BOOT} bootstrap draws per cell ...")
    table = build_table(rows, ranges, args.seed)

    csv_path = out_dir / "param_recovery_metrics.csv"
    with io.open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote {csv_path}")

    for level in args.levels.split(","):
        print_table(table, level)
        for coefficient in names:
            plot_coefficient(rows, table, coefficient, ranges[coefficient], level, plots_dir)
        overview = plot_overview(rows, table, ranges, level, plots_dir)
        confusion = plot_confusion(rows, level, plots_dir, args.seed, names)
        print(f"wrote {2 + len(names)} figures for level={level}, "
              f"including {overview.name} and {confusion.name}")


if __name__ == "__main__":
    main()
