"""
Model comparison across the fitted variants: per-episode likelihood, BIC, and which variant
each episode prefers. The V4 counterpart of the AAAI pipeline's `nll_comparison.png`,
`bic_comparison.png`, `best_fit_counts.png` and `bic_and_best_fit.png`
(`inference_aaai/scripts/plot_heuristic_fit_results.py`), read off the
`fit_variant_<name>.csv` files `run_human_inference.py` writes.

WHAT IS COMPARED, AND ON WHAT
    Every number here is computed from `logL_mode`, the log-likelihood at the gradient PEAK:
    that is the maximum-likelihood value BIC is defined on, and the posterior MEAN the
    behavioural validation rolls out is by construction at or below it. BIC is the textbook
    ``k ln(n) - 2 logL`` (Schwarz 1978, "Estimating the dimension of a model", Annals of
    Statistics 6:461-464) with ``k = n_free_params`` and ``n = n_frames``, both read from the
    fit file so this script never re-derives what a variant fitted.

    Only episodes present in EVERY variant's file enter, so each comparison is PAIRED: the
    same episodes, the same data, differing only in the model. A variant with fewer rows
    would otherwise win or lose on which episodes it happened to have.

THE PAIRED DIFFERENCE, NOT ONLY THE MEANS
    Two mean BICs with standard errors say little when the episodes are shared -- most of
    the per-episode spread is the episode (its length, its traffic) and cancels in a paired
    difference. So beside the per-variant means this reports ``BIC(variant) - BIC(reference)``
    per episode, with a bootstrap interval that resamples PARTICIPANTS rather than episodes,
    since one person's six episodes are not six independent observations of the model
    (bootstrap_stats.bootstrap_indices, Field & Welsh 2007). Negative favours the variant.

THE BEST-FIT COUNT
    Per episode, the lowest-BIC variant wins only if it beats the runner-up by
    ``BIC_DELTA_THRESHOLD`` = 10, Kass & Raftery's "very strong evidence" line (1995, "Bayes
    factors", JASA 90:773-795); anything closer is counted as ambiguous, as the AAAI figure did.

NESTING VIOLATIONS
    The variants form a chain, C in B in A' in A (`variants.NESTED_PAIRS`): each larger model
    can reproduce the smaller one's solution exactly, so its peak log-likelihood must be at
    least the smaller's. An episode where it is not is the OPTIMISER stopping short, not
    evidence about the models, and it biases BIC toward the smaller model. The summary counts
    such episodes per pair with the worst shortfall in nats. MEASURED before the fits were
    chained (Entry 35 addendum 2): 3 of 23 episodes, up to 19.9 nats, for C over B.
    When `--variants` leaves a link out (A' is fitted as A's warm start but not reported,
    Entry 35 addendum 3), each reported variant is checked against the reported one below
    it, which it still contains through the chain.

OUTPUTS in ``<results-dir>/plots/`` and ``<results-dir>/``
    `nll_comparison.png`, `bic_comparison.png`, `delta_bic_vs_reference.png`,
    `best_fit_counts.png`, `bic_and_best_fit.png`; `delta_bic_vs_<baseline>_and_best_fit.png`,
    every larger reported model minus the smallest reported one (C), above the best-fit
    counts -- raw BIC spreads over the episodes' own sizes (~2,000 to ~12,000) and hides the
    70-80 point model differences that the paired difference shows;
    `model_comparison.csv` (one row per variant) and `model_comparison_summary.txt` (the
    same, printed).

Typical usage::

    uv run python -m human_modeling.inference.scripts.plot_model_comparison \\
        --results-dir human_modeling/inference/results/model_comparison/fit
"""

from __future__ import annotations

import argparse
import csv
import io
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402 - backend must be chosen before this import
import numpy as np                # noqa: E402

from human_modeling.inference.scripts.bootstrap_stats import (N_BOOT, bootstrap_indices,
                                                                 group_clusters)
from human_modeling.inference.variants import (ATTENTION_CHOOSING_VARIANTS, FIT_ORDER,
                                                  NESTED_PAIRS, VARIANTS)

BIC_DELTA_THRESHOLD = 10.0      # Kass & Raftery 1995: "very strong evidence"
DEFAULT_REFERENCE = "B_no_heuristics"

# | Categorical slots 1-5 of the validated default palette (dataviz skill,
# | references/palette.md), assigned in FIT_ORDER and never cycled. Identity is carried by
# | the labelled axis row, so the fourth slot's weaker yellow/orange separation is backed by
# | a text channel as the palette requires. Episode dots are ink-muted gray: they are the
# | distribution, not an identity.
    # | [full-visibility] Slot 5 (magenta) is the full-visibility model's. Without it, `zip`
    # | stopped at four and any figure that included that model failed on a missing colour.
VARIANT_COLORS = dict(zip(FIT_ORDER, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")))
assert set(VARIANT_COLORS) == set(FIT_ORDER), "every variant needs a palette slot"
COLOR_EPISODE = "0.62"
COLOR_AMBIGUOUS = "0.72"
COLOR_ZERO_LINE = "0.45"

# | Every text element at 3x matplotlib's 10 pt default (user, 2026-09-24, after 1.7x and 2.2x:
# | the figures go into documents and slides). A viewer fits the image to its window, so what
# | the eye sees is text size RELATIVE to the canvas: the width stays at 16 in (growing it with
# | the text would cancel the increase) and long labels wrap instead.
FONT_SCALE = 3.0
FIG_WIDTH = 16.0
HORIZONTAL_WIDTH = 30.0         # side-by-side delta + counts figure, two labelled panels wide
HEIGHT_SCALE = 2.0
LABEL_WIDTH = 48                # characters per x-label line at the scaled size
TICK_WIDTH = 24                 # characters per row-label line

# | The mean diamond and its interval bar, 2.5x the first version's size (user, 2026-09-24),
# | shared by every panel that draws them.
MARKER_SCALE = 2.5
_MEAN_STYLE: dict[str, Any] = dict(
    fmt="D", ecolor="0.2", markeredgecolor="white", zorder=4,
    markersize=8 * MARKER_SCALE, markeredgewidth=1.5 * MARKER_SCALE,
    capsize=5 * MARKER_SCALE, elinewidth=1.5 * MARKER_SCALE, capthick=1.5 * MARKER_SCALE)
# | The per-episode dots at the same 2.5x linear size (area goes with its square), spread a
# | little wider and a little lighter so 230 of them per row still read as a distribution.
_DOT_STYLE: dict[str, Any] = dict(s=12 * MARKER_SCALE ** 2, color=COLOR_EPISODE, alpha=0.35,
                                  linewidths=0, zorder=2)
DOT_JITTER_SD = 0.09            # rows are 1 apart


def _size(height: float) -> tuple[float, float]:
    """A figure size at the scaled text size, from the height the 9-inch layout used."""
    return FIG_WIDTH, height * HEIGHT_SCALE


def _wrap(text: str, width: int = LABEL_WIDTH) -> str:
    """Wrap each line of a label to `width` characters (default :data:`LABEL_WIDTH`)."""
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def _row_labels(labels: list[str]) -> list[str]:
    """Row labels wrapped to :data:`TICK_WIDTH`, so the axes keep their width at 3x."""
    return [_wrap(label, TICK_WIDTH) for label in labels]


# ===========================================================================
# 1. Reading and the per-episode table
# ===========================================================================

def available_variants(results_dir: Path) -> list[str]:
    """
    Which attention-choosing variants have a fit file here, in :data:`FIT_ORDER`.

    The default model set of this script and of `participant_bic_tests.py`. The
    full-visibility sanity model is left out even when its file is present: it is not part of
    the reported comparison (Entry 36), and the per-participant tests would mislabel it as a
    nested "larger" model. Name it with ``--variants`` here to include it in the figures.

    :param results_dir: Directory holding `fit_variant_<name>.csv` files.
    :type results_dir: Path
    :return: Variant names with a file present.
    :rtype: list[str]
    """
    return [n for n in ATTENTION_CHOOSING_VARIANTS
            if (results_dir / f"fit_variant_{n}.csv").exists()]


def read_fit(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """
    One variant's fit file as ``{(subject_id, game_index): {logL, k, n, no_progress,
    warm_start}}``. Ex: ``{("s", 1): {"logL": -1000.0, "k": 4, "n": 900, "no_progress": 1,
    "warm_start": "C_no_cognitive_cost"}}``. A file without a `warm_start` column (written
    before the chain) reads as ``"midpoint"``.

    :param path: A `fit_variant_<name>.csv`.
    :type path: Path
    :return: Per-episode record.
    :rtype: dict[tuple[str, int], dict[str, Any]]
    :raises ValueError: If the file lacks the columns BIC needs.
    """
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with io.open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        needed = {"logL_mode", "n_free_params", "n_frames"}
        if reader.fieldnames is None or not needed <= set(reader.fieldnames):
            raise ValueError(f"{path} lacks {sorted(needed - set(reader.fieldnames or []))}; "
                             f"it predates the variant schema (Entry 35) -- refit")
        for r in reader:
            out[(r["subject_id"], int(r["game_index"]))] = {
                "logL": float(r["logL_mode"]), "k": int(r["n_free_params"]),
                "n": int(r["n_frames"]), "no_progress": int(r.get("no_progress") or 0),
                "warm_start": r.get("warm_start") or "midpoint"}
    return out


def bic(logl: np.ndarray, k: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Schwarz's criterion per episode, lower is better.

    :param logl: Maximum log-likelihood per episode.
    :type logl: np.ndarray
    :param k: Free-parameter count per episode.
    :type k: np.ndarray
    :param n: Observations (trajectory steps) per episode.
    :type n: np.ndarray
    :return: ``k ln(n) - 2 logl``.
    :rtype: np.ndarray
    """
    return k * np.log(n) - 2.0 * logl


def episode_table(results_dir: Path, variants: list[str]) -> dict[str, Any]:
    """
    The paired per-episode table over the episodes every variant fitted.

    :param results_dir: Directory holding the fit files.
    :type results_dir: Path
    :param variants: Variants to include, in reporting order.
    :type variants: list[str]
    :return: ``{"keys", "subjects", "nll", "bic", "k", "no_progress", "no_progress_warm",
        "dropped"}`` where `nll`/`bic`/`no_progress`/`no_progress_warm` map variant -> array
        over `keys`, `k` maps variant -> its free-parameter count, and `dropped` counts
        episodes missing from at least one file. `no_progress_warm` marks the no-progress fits
        that started from a smaller model's solution rather than the midpoint.
    :rtype: dict[str, Any]
    :raises ValueError: If no episode is shared by every variant.
    """
    fits = {v: read_fit(results_dir / f"fit_variant_{v}.csv") for v in variants}
    shared = sorted(set.intersection(*(set(f) for f in fits.values())))
    union = set.union(*(set(f) for f in fits.values()))
    if not shared:
        raise ValueError("no episode is present in every variant's fit file")
    out: dict[str, Any] = {"keys": shared, "subjects": np.array([k[0] for k in shared]),
                           "nll": {}, "bic": {}, "k": {}, "no_progress": {},
                           "no_progress_warm": {}, "dropped": len(union) - len(shared)}
    for v, f in fits.items():
        logl = np.array([f[k]["logL"] for k in shared])
        kk = np.array([f[k]["k"] for k in shared])
        n = np.array([f[k]["n"] for k in shared])
        out["nll"][v] = -logl
        out["bic"][v] = bic(logl, kk, n)
        out["k"][v] = int(kk[0])
        out["no_progress"][v] = np.array([f[k]["no_progress"] for k in shared])
        out["no_progress_warm"][v] = np.array(
            [int(f[k]["no_progress"] and f[k]["warm_start"] in VARIANTS) for k in shared])
    return out


# ===========================================================================
# 2. Statistics
# ===========================================================================

def mean_se(x: np.ndarray) -> tuple[float, float]:
    """Mean and its standard error (``sd / sqrt(n)``; 0 for a single value)."""
    se = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return float(np.mean(x)), se


def paired_difference(x: np.ndarray, y: np.ndarray, clusters: np.ndarray,
                      seed: int = 0) -> dict[str, float]:
    """
    Mean of ``x - y`` with a cluster-bootstrap percentile interval.

    :param x: Per-episode values of the variant.
    :type x: np.ndarray
    :param y: Per-episode values of the reference, same episodes.
    :type y: np.ndarray
    :param clusters: Participant label per episode; whole participants are resampled.
    :type clusters: np.ndarray
    :param seed: Bootstrap seed, fixed so a re-run reproduces the interval.
    :type seed: int
    :return: ``{"mean", "ci_lo", "ci_hi", "share_negative"}``; the share is how many
        episodes favour the variant (a negative difference).
    :rtype: dict[str, float]
    """
    d = x - y
    rng = np.random.default_rng(seed)
    members = group_clusters(clusters)
    draws = np.array([d[bootstrap_indices(rng, len(d), members)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "share_negative": float(np.mean(d < 0))}


def best_fit_counts(table: dict[str, Any], variants: list[str]) -> tuple[dict[str, int], int]:
    """
    Per episode, the lowest-BIC variant if it beats the runner-up by the threshold.

    :param table: :func:`episode_table`'s output.
    :type table: dict[str, Any]
    :param variants: Variants in the table.
    :type variants: list[str]
    :return: ``({variant: wins}, n_ambiguous)``.
    :rtype: tuple[dict[str, int], int]
    """
    stacked = np.stack([table["bic"][v] for v in variants], axis=1)      # (episodes, variants)
    if len(variants) < 2:
        # | One model cannot "beat the runner-up"; every episode is ambiguous by definition.
        return {v: 0 for v in variants}, int(stacked.shape[0])
    order = np.argsort(stacked, axis=1)
    best, runner = order[:, 0], order[:, 1]
    margin = stacked[np.arange(len(stacked)), runner] - stacked[np.arange(len(stacked)), best]
    clear = margin >= BIC_DELTA_THRESHOLD
    counts = {v: int(np.sum(clear & (best == i))) for i, v in enumerate(variants)}
    return counts, int(np.sum(~clear))


def _containment() -> set[tuple[str, str]]:
    """Every ``(smaller, larger)`` the chain implies: :data:`NESTED_PAIRS`, closed transitively."""
    contains = set(NESTED_PAIRS)
    for _ in FIT_ORDER:                                   # transitive closure; the chain is short
        contains |= {(a, d) for a, b in contains for c, d in contains if b == c}
    return contains


def baseline_pairs(variants: list[str]) -> list[tuple[str, str]]:
    """
    ``(baseline, larger)`` for every reported variant containing the smallest reported one.

    The baseline is the reported variant lowest in the chain (C when C, B and A are
    reported), so each difference reads as what the larger model gains over it. Largest
    model first, to line up with the best-fit counts beneath it.

    :param variants: Reported variants, any order.
    :type variants: list[str]
    :return: Pairs against the baseline, largest model first; empty below two variants.
    :rtype: list[tuple[str, str]]
    """
    chain = [v for v in FIT_ORDER if v in variants]
    contains = _containment()
    return [(chain[0], v) for v in reversed(chain[1:]) if (chain[0], v) in contains]


def reported_nested_pairs(variants: list[str]) -> list[tuple[str, str]]:
    """
    ``(smaller, larger)`` for each reported variant and the reported one just below it.

    Containment is transitive, so B still nests in A when A' is left out. A pair is kept only
    if :data:`NESTED_PAIRS` actually links it, so a future variant outside the chain is never
    checked against a model it does not contain.

    :param variants: Reported variants, any order.
    :type variants: list[str]
    :return: Adjacent reported pairs in :data:`FIT_ORDER`.
    :rtype: list[tuple[str, str]]
    """
    contains = _containment()
    chain = [v for v in FIT_ORDER if v in variants]
    return [(s, l) for s, l in zip(chain, chain[1:]) if (s, l) in contains]


def nesting_violations(table: dict[str, Any], variants: list[str],
                       tol: float = 1e-6) -> dict[str, dict[str, Any]]:
    """
    Per reported larger model, how often the reported smaller model it contains out-fit it.

    :param table: :func:`episode_table`'s output.
    :type table: dict[str, Any]
    :param variants: Variants in the table; see :func:`reported_nested_pairs` for the pairs.
    :type variants: list[str]
    :param tol: Log-likelihood excess below which a difference is rounding, not a violation.
    :type tol: float
    :return: ``{larger: {"smaller", "n_violations", "worst_shortfall_nats"}}``.
    :rtype: dict[str, dict[str, Any]]
    """
    out: dict[str, dict[str, Any]] = {}
    for smaller, larger in reported_nested_pairs(variants):
        shortfall = table["nll"][larger] - table["nll"][smaller]       # > 0: larger fits WORSE
        out[larger] = {"smaller": smaller,
                       "n_violations": int(np.sum(shortfall > tol)),
                       "worst_shortfall_nats": float(max(shortfall.max(), 0.0))}
    return out


def summary_rows(table: dict[str, Any], variants: list[str], reference: str,
                 seed: int = 0) -> list[dict[str, Any]]:
    """
    One record per variant: the numbers behind every figure.

    :param table: :func:`episode_table`'s output.
    :type table: dict[str, Any]
    :param variants: Variants in reporting order.
    :type variants: list[str]
    :param reference: The variant every paired difference is taken against.
    :type reference: str
    :param seed: Bootstrap seed.
    :type seed: int
    :return: Records with the keys of :data:`SUMMARY_FIELDS`.
    :rtype: list[dict[str, Any]]
    """
    counts, ambiguous = best_fit_counts(table, variants)
    nesting = nesting_violations(table, variants)
    rows = []
    for v in variants:
        nll_mean, nll_se = mean_se(table["nll"][v])
        bic_mean, bic_se = mean_se(table["bic"][v])
        diff = paired_difference(table["bic"][v], table["bic"][reference], table["subjects"], seed)
        rows.append({"variant": v, "display": VARIANTS[v].display, "n_episodes": len(table["keys"]),
                     "n_free_params": table["k"][v],
                     "nll_mean": nll_mean, "nll_se": nll_se,
                     "bic_mean": bic_mean, "bic_se": bic_se,
                     "delta_bic_vs_reference_mean": diff["mean"],
                     "delta_bic_ci_lo": diff["ci_lo"], "delta_bic_ci_hi": diff["ci_hi"],
                     "share_episodes_favouring_variant": diff["share_negative"],
                     "best_fit_count": counts[v], "n_ambiguous": ambiguous,
                     "n_stalled_fits": int(table["no_progress"][v].sum()),
                     "n_no_progress_from_smaller": int(table["no_progress_warm"][v].sum()),
                     "nested_in": nesting.get(v, {}).get("smaller", ""),
                     "nesting_violations": nesting.get(v, {}).get("n_violations", ""),
                     "worst_nesting_shortfall_nats": nesting.get(v, {}).get("worst_shortfall_nats", ""),
                     "reference": reference})
    return rows


SUMMARY_FIELDS = ["variant", "display", "n_episodes", "n_free_params", "nll_mean", "nll_se",
                  "bic_mean", "bic_se", "delta_bic_vs_reference_mean", "delta_bic_ci_lo",
                  "delta_bic_ci_hi", "share_episodes_favouring_variant", "best_fit_count",
                  "n_ambiguous", "n_stalled_fits", "n_no_progress_from_smaller", "nested_in",
                  "nesting_violations", "worst_nesting_shortfall_nats", "reference"]
# | `n_stalled_fits` keeps its name for existing readers but counts EVERY no-progress fit;
# | `n_no_progress_from_smaller` is the part that started from a smaller model's solution
# | (format_summary explains why those are not necessarily stalls).


def format_summary(rows: list[dict[str, Any]], dropped: int) -> str:
    """
    The summary as plain ASCII text (Rule 12), the same content as the CSV.

    :param rows: :func:`summary_rows`'s output.
    :type rows: list[dict[str, Any]]
    :param dropped: Episodes excluded for being absent from at least one variant.
    :type dropped: int
    :return: Multi-line text.
    :rtype: str
    """
    ref = rows[0]["reference"]
    lines = [f"model comparison over {rows[0]['n_episodes']} paired episodes "
             f"({dropped} dropped: absent from at least one variant)",
             f"BIC = k ln(n) - 2 logL at the peak; delta BIC is against {ref}, "
             f"negative favours the variant; interval = 95% participant bootstrap",
             "",
             f"{'variant':<22}{'k':>3}{'mean NLL':>11}{'SE':>7}{'mean BIC':>11}{'SE':>7}"
             f"{'dBIC vs ref':>13}{'95% CI':>22}{'fav%':>6}{'best':>6}{'noprog':>8}"]
    for r in rows:
        ci = f"[{r['delta_bic_ci_lo']:+.1f}, {r['delta_bic_ci_hi']:+.1f}]"
        lines.append(f"{r['variant']:<22}{r['n_free_params']:>3}{r['nll_mean']:>11.1f}"
                     f"{r['nll_se']:>7.1f}{r['bic_mean']:>11.1f}{r['bic_se']:>7.1f}"
                     f"{r['delta_bic_vs_reference_mean']:>+13.1f}{ci:>22}"
                     f"{100 * r['share_episodes_favouring_variant']:>5.0f}%"
                     f"{r['best_fit_count']:>6}{r['n_stalled_fits']:>8}")
    # | The count sits under the "best" column: 80 = the widths of k through fav% (3 + 11 +
    # | 7 + 11 + 7 + 13 + 22 + 6) that come between the 22-wide name and it.
    lines.append(f"{'ambiguous (dBIC < ' + f'{BIC_DELTA_THRESHOLD:.0f}' + ')':<22}"
                 f"{'':>80}{rows[0]['n_ambiguous']:>6}")
    nested = [r for r in rows if r["nested_in"] != ""]
    if nested:
        lines += ["", "nesting (the larger model contains the smaller, so its peak logL must be "
                      ">= the smaller's; a violation is the optimiser stopping short):"]
        for r in nested:
            lines.append(f"  {r['nested_in']:<22} in {r['variant']:<22} violated on "
                         f"{r['nesting_violations']} episodes, worst shortfall "
                         f"{r['worst_nesting_shortfall_nats']:.2f} nats")
    # | [boundary-start] Split the no-progress fits by where they started. From the midpoint,
    # | one is a stall. From a smaller model's solution, it ties that model and is either a
    # | boundary optimum or a stall -- the bounds that decide it are not in the fit files,
    # | so the fit log (boundary_coefficients in run_human_inference.py) is the authority.
    warm = sum(r["n_no_progress_from_smaller"] for r in rows)
    cold = sum(r["n_stalled_fits"] for r in rows) - warm
    if warm:
        lines += ["", f"NOTE: {warm} fits never improved on the smaller model's solution they "
                      f"started from (no_progress = 1, warm_start = that model). Each ties "
                      f"that model's likelihood, so it can understate the larger model but "
                      f"never favour it. It IS the estimate when every coefficient the larger "
                      f"model adds sits at a bound there (B started from C with cognitive "
                      f"cost at 0: Entry 35 addendum 3), and a stall otherwise. The fit log "
                      f"says which for each episode."]
    if cold:
        lines += ["", f"WARNING: {cold} fits never improved on their midpoint start "
                      f"(no_progress = 1). Those rows are NOT estimates and understate their "
                      f"variant; re-fit them at a smaller step before quoting this table "
                      f"(implementation.md Entry 34)."]
    return "\n".join(lines)


# ===========================================================================
# 3. Figures
# ===========================================================================

def _strip_panel(ax: Any, table_values: dict[str, np.ndarray], variants: list[str],
                 xlabel: str, seed: int = 1) -> None:
    """
    Horizontal strip per variant: jittered episode dots, then the mean with a +-1 SE bar.

    Variants read top to bottom in :data:`FIT_ORDER`; identity is the axis label, colour
    only echoes it. No title (user convention 2026-09-22: captions carry titles).
    """
    rng = np.random.default_rng(seed)
    for pos, v in enumerate(variants):
        x = table_values[v]
        y = rng.normal(pos, DOT_JITTER_SD, size=len(x))
        ax.scatter(x, y, **_DOT_STYLE)
        mean, se = mean_se(x)
        ax.errorbar(mean, pos, xerr=se, color=VARIANT_COLORS[v], **_MEAN_STYLE)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(_row_labels([VARIANTS[v].display for v in variants]))
    ax.invert_yaxis()
    ax.set_xlabel(_wrap(xlabel))
    ax.grid(axis="x", color="0.9", linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.99, 0.02, "diamond: mean, bar: +-1 SE", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8 * FONT_SCALE, color="0.35")


def _delta_panel(ax: Any, table: dict[str, Any], variants: list[str], reference: str,
                 seed: int = 0) -> None:
    """Paired ``BIC(variant) - BIC(reference)`` per episode, with the bootstrap interval."""
    others = [v for v in variants if v != reference]
    rng = np.random.default_rng(1)
    for pos, v in enumerate(others):
        d = table["bic"][v] - table["bic"][reference]
        ax.scatter(d, rng.normal(pos, DOT_JITTER_SD, size=len(d)), **_DOT_STYLE)
        stats = paired_difference(table["bic"][v], table["bic"][reference], table["subjects"], seed)
        ax.errorbar(stats["mean"], pos,
                    xerr=[[stats["mean"] - stats["ci_lo"]], [stats["ci_hi"] - stats["mean"]]],
                    color=VARIANT_COLORS[v], **_MEAN_STYLE)
    ax.axvline(0.0, color=COLOR_ZERO_LINE, linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels(_row_labels([VARIANTS[v].display for v in others]))
    ax.invert_yaxis()
    ax.set_xlabel(_wrap(f"BIC difference against '{VARIANTS[reference].display}' per episode "
                        f"(negative favours the variant)"))
    ax.grid(axis="x", color="0.9", linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.99, 0.02, "diamond: mean, bar: 95% participant bootstrap", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8 * FONT_SCALE, color="0.35")


def pair_counts(d: np.ndarray) -> dict[str, int]:
    """
    How one nested pair splits the episodes at the best-fit margin.

    :param d: Per-episode ``BIC(larger) - BIC(smaller)``.
    :type d: np.ndarray
    :return: ``{"favours_larger", "ambiguous", "favours_smaller"}`` episode counts.
    :rtype: dict[str, int]
    """
    return {"favours_larger": int(np.sum(d <= -BIC_DELTA_THRESHOLD)),
            "ambiguous": int(np.sum(np.abs(d) < BIC_DELTA_THRESHOLD)),
            "favours_smaller": int(np.sum(d >= BIC_DELTA_THRESHOLD))}


def _pairs_panel(ax: Any, table: dict[str, Any], pairs: list[tuple[str, str]],
                 seed: int = 0) -> None:
    """
    One row per ``(smaller, larger)`` pair: ``BIC(larger) - BIC(smaller)`` per episode, the mean
    with its participant-bootstrap interval, the ambiguous band, and the pair's episode split.

    Paired within the episode, so the episode's own size -- which spreads raw BIC from ~2,000 to
    ~12,000 -- cancels, and the 70-80 point model differences become visible.
    """
    rng = np.random.default_rng(1)
    ax.axvspan(-BIC_DELTA_THRESHOLD, BIC_DELTA_THRESHOLD, color=COLOR_AMBIGUOUS, alpha=0.35,
               linewidth=0, zorder=0)
    ax.axvline(0.0, color=COLOR_ZERO_LINE, linewidth=1, linestyle="--", zorder=1)
    for pos, (smaller, larger) in enumerate(pairs):
        d = table["bic"][larger] - table["bic"][smaller]
        ax.scatter(d, rng.normal(pos, DOT_JITTER_SD, size=len(d)), **_DOT_STYLE)
        stats = paired_difference(table["bic"][larger], table["bic"][smaller], table["subjects"], seed)
        ax.errorbar(stats["mean"], pos,
                    xerr=[[stats["mean"] - stats["ci_lo"]], [stats["ci_hi"] - stats["mean"]]],
                    color=VARIANT_COLORS[larger], **_MEAN_STYLE)
        c = pair_counts(d)
        # | Two lines, so the split stays inside the axes at 3x.
        ax.text(0.99, pos - 0.26, f"favours {VARIANTS[larger].display}: {c['favours_larger']}  |  "
                f"ambiguous: {c['ambiguous']}\nfavours {VARIANTS[smaller].display}: "
                f"{c['favours_smaller']}", transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=8 * FONT_SCALE, color="0.3")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(_row_labels([f"{VARIANTS[l].display}\nminus {VARIANTS[s].display}"
                                    for s, l in pairs]))
    ax.set_ylim(len(pairs) - 0.5, -0.95)
    ax.set_xlabel(_wrap(f"BIC difference per episode, larger minus smaller\n"
                        f"(negative favours larger; shaded: within {BIC_DELTA_THRESHOLD:.0f})"))
    ax.grid(axis="x", color="0.9", linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.99, 0.02, "diamond: mean, bar: 95% participant bootstrap", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8 * FONT_SCALE, color="0.35")


def _counts_panel(ax: Any, table: dict[str, Any], variants: list[str]) -> None:
    """Best-fitting variant per episode as horizontal bars, ambiguous episodes last."""
    counts, ambiguous = best_fit_counts(table, variants)
    order = sorted(variants, key=lambda v: -counts[v])
    labels = [VARIANTS[v].display for v in order] + [f"ambiguous\n(BIC margin < {BIC_DELTA_THRESHOLD:.0f})"]
    values = [counts[v] for v in order] + [ambiguous]
    colors = [VARIANT_COLORS[v] for v in order] + [COLOR_AMBIGUOUS]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.55)
    for yi, val in zip(y, values):
        ax.text(val + 0.5, yi, str(val), va="center", fontsize=9 * FONT_SCALE, color="0.2")
    ax.set_yticks(y)
    ax.set_yticklabels(_row_labels(labels))
    ax.invert_yaxis()
    ax.set_xlabel(_wrap(f"Episodes best fit (BIC margin >= {BIC_DELTA_THRESHOLD:.0f} over the runner-up)"))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _pairs_and_counts_figure(table: dict[str, Any], variants: list[str],
                             pairs: list[tuple[str, str]], seed: int, horizontal: bool) -> Any:
    """
    The paired BIC differences against the smallest model with the best-fit counts, in one
    figure: stacked (the counts below), or side by side (the counts to the right).

    The side-by-side layout is wider than :data:`FIG_WIDTH`, by :data:`HORIZONTAL_WIDTH`: two
    panels each carrying row labels and in-axes text at 3x do not fit in 16 in. Its text is
    therefore smaller RELATIVE to the canvas than in the stacked figure.

    :param table: :func:`episode_table`'s output.
    :type table: dict[str, Any]
    :param variants: Variants in reporting order.
    :type variants: list[str]
    :param pairs: :func:`baseline_pairs`'s output, one row each in the difference panel.
    :type pairs: list[tuple[str, str]]
    :param seed: Bootstrap seed.
    :type seed: int
    :param horizontal: True for side by side, False for stacked.
    :type horizontal: bool
    :return: The figure, not yet saved.
    :rtype: Any
    """
    pair_height = 1.4 * len(pairs)
    count_height = 0.55 * (len(variants) + 1)
    if horizontal:
        fig, (ax_d, ax_cnt) = plt.subplots(
            1, 2, figsize=(HORIZONTAL_WIDTH, (max(pair_height, count_height) + 2.2) * HEIGHT_SCALE),
            gridspec_kw={"width_ratios": [1.25, 1.0]})
    else:
        fig, (ax_d, ax_cnt) = plt.subplots(
            2, 1, figsize=_size(pair_height + count_height + 2.8),
            gridspec_kw={"height_ratios": [pair_height, count_height]})
    _pairs_panel(ax_d, table, pairs, seed)
    _counts_panel(ax_cnt, table, variants)
    return fig


def draw_figures(table: dict[str, Any], variants: list[str], reference: str,
                 plots_dir: Path, seed: int = 0) -> list[Path]:
    """
    Write every figure and return their paths.

    :param table: :func:`episode_table`'s output.
    :type table: dict[str, Any]
    :param variants: Variants in reporting order.
    :type variants: list[str]
    :param reference: Reference variant for the paired panel.
    :type reference: str
    :param plots_dir: Destination directory.
    :type plots_dir: Path
    :param seed: Bootstrap seed.
    :type seed: int
    :return: The written files.
    :rtype: list[Path]
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    n = len(variants)
    written: list[Path] = []

    def save(fig: Any, name: str) -> None:
        fig.tight_layout(); written.append(plots_dir / name)
        fig.savefig(written[-1], dpi=150, bbox_inches="tight"); plt.close(fig)

    with plt.rc_context({"font.size": 10 * FONT_SCALE}):
        fig, ax = plt.subplots(figsize=_size(1.1 * n + 1.8))
        _strip_panel(ax, table["nll"], variants, "Negative log-likelihood per episode (at the peak)")
        save(fig, "nll_comparison.png")

        fig, ax = plt.subplots(figsize=_size(1.1 * n + 1.8))
        _strip_panel(ax, table["bic"], variants, "BIC per episode (lower is better)")
        save(fig, "bic_comparison.png")

        if n > 1:
            fig, ax = plt.subplots(figsize=_size(1.1 * (n - 1) + 1.8))
            _delta_panel(ax, table, variants, reference, seed)
            save(fig, "delta_bic_vs_reference.png")

        fig, ax = plt.subplots(figsize=_size(0.9 * (n + 1) + 1.5))
        _counts_panel(ax, table, variants)
        save(fig, "best_fit_counts.png")

        fig, (ax_bic, ax_cnt) = plt.subplots(2, 1, figsize=_size(2.0 * n + 3.5),
                                             gridspec_kw={"height_ratios": [n, n + 1]})
        _strip_panel(ax_bic, table["bic"], variants, "BIC per episode (lower is better)")
        _counts_panel(ax_cnt, table, variants)
        save(fig, "bic_and_best_fit.png")

        # | Every larger model minus the smallest reported one beside the best-fit counts -- the
        # | readable counterpart of bic_and_best_fit.png (user, 2026-09-24) -- stacked, and
        # | again side by side (user, 2026-09-25).
        pairs = baseline_pairs(variants)
        if pairs:
            base = f"delta_bic_vs_{pairs[0][0]}_and_best_fit"
            save(_pairs_and_counts_figure(table, variants, pairs, seed, horizontal=False),
                 f"{base}.png")
            save(_pairs_and_counts_figure(table, variants, pairs, seed, horizontal=True),
                 f"{base}_horizontal.png")
    return written


# ===========================================================================
# 4. Entry point
# ===========================================================================

def main() -> None:
    """CLI: read the variant files present, compute the paired table, write CSV, text, figures."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="directory holding fit_variant_<name>.csv; outputs land beside them")
    ap.add_argument("--variants", nargs="+", default=None, choices=list(FIT_ORDER),
                    metavar="NAME",
                    help="variants to report, any order (shown in FIT_ORDER). Default: every "
                         "variant with a fit file. A variant fitted only as another's warm "
                         "start is left out here, not at the fit")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    present = available_variants(results_dir)
    if args.variants:
        absent = sorted(set(args.variants) - set(present))
        if absent:
            raise SystemExit(f"no fit file under {results_dir} for {absent}; found {present}")
        variants = [v for v in present if v in args.variants]
    else:
        variants = present
    if not variants:
        raise SystemExit(f"no fit_variant_*.csv under {results_dir}")
    if DEFAULT_REFERENCE not in variants:
        raise SystemExit(f"reference {DEFAULT_REFERENCE} has no fit file under {results_dir}; "
                         f"found {variants}")
    table = episode_table(results_dir, variants)
    rows = summary_rows(table, variants, DEFAULT_REFERENCE)

    with io.open(results_dir / "model_comparison.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    text = format_summary(rows, table["dropped"])
    (results_dir / "model_comparison_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    for path in draw_figures(table, variants, DEFAULT_REFERENCE, results_dir / "plots"):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
