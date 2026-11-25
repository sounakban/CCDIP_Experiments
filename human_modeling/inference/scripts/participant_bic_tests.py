"""
The per-participant BIC tests the paper reports for the model comparison (implementation.md
Entry 35 addendum 4).

WHY THIS EXISTS BESIDE plot_model_comparison.py
    `plot_model_comparison.py` reports the PER-EPISODE paired BIC difference, with an interval
    that resamples participants. The paper tests at the PARTICIPANT level instead, as the AAAI
    paper did: each participant's per-episode differences are averaged first, and those
    averages are tested against zero. The two means differ slightly because participants played
    4-6 episodes each, so the numbers quoted in the paper come from this script.

WHAT IT COMPUTES, for every pair of reported variants (smaller, larger, in FIT_ORDER)
    1. Per participant, the mean over their episodes of BIC(smaller) - BIC(larger). Positive
       favours the larger model.
    2. The mean of those participant averages, with a 95% percentile-bootstrap interval over
       participants (`bootstrap_stats.N_BOOT` draws), their median, and how many are positive.
    3. A Wilcoxon signed-rank test of the averages against zero (scipy; exact for up to 50
       participants without ties or zeros).
    4. Holm's step-down adjustment of the p-values across all pairs (Holm 1979, "A simple
       sequentially rejective multiple test procedure", Scandinavian Journal of Statistics
       6:65-70), via statsmodels.

OUTPUTS in ``<results-dir>/``: `participant_bic_tests.csv` (one row per pair) and
`participant_bic_tests.txt` (the same, printed).

Typical usage::

    python -m human_modeling.inference.scripts.participant_bic_tests \\
        --results-dir human_modeling/inference/results/model_comparison/fit \\
        --variants C_no_cognitive_cost B_no_heuristics A_full
"""

from __future__ import annotations

import argparse
import csv
import io
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests   # Holm's adjustment across the pairs

from human_modeling.inference.scripts.bootstrap_stats import N_BOOT
from human_modeling.inference.scripts.plot_model_comparison import (available_variants,
                                                                       episode_table)
from human_modeling.inference.variants import ATTENTION_CHOOSING_VARIANTS, FIT_ORDER, VARIANTS

FIELDS = ["smaller", "larger", "n_participants", "mean_delta_bic", "ci_lo", "ci_hi",
          "median_delta_bic", "n_favouring_larger", "wilcoxon_w", "p", "p_holm"]
# | Sample row: {"smaller": "B_no_heuristics", "larger": "A_full", "n_participants": 39,
# | "mean_delta_bic": 80.7, "ci_lo": 71.1, "ci_hi": 90.3, "median_delta_bic": 79.5,
# | "n_favouring_larger": 39, "wilcoxon_w": 0.0, "p": 3.6e-12, "p_holm": 1.1e-11}


def participant_means(table: dict[str, Any], smaller: str, larger: str) -> np.ndarray:
    """
    Each participant's mean per-episode ``BIC(smaller) - BIC(larger)``, in sorted participant order.

    :param table: `plot_model_comparison.episode_table`'s output.
    :type table: dict[str, Any]
    :param smaller: The nested (smaller) variant.
    :type smaller: str
    :param larger: The variant containing it.
    :type larger: str
    :return: One value per participant, ``(n_participants,)``; positive favours `larger`.
    :rtype: np.ndarray
    """
    d = table["bic"][smaller] - table["bic"][larger]                   # (n_episodes,)
    subjects = table["subjects"]
    return np.array([d[subjects == s].mean() for s in np.unique(subjects)])


def pair_row(means: np.ndarray, smaller: str, larger: str, seed: int) -> dict[str, Any]:
    """
    Every statistic for one pair except the Holm adjustment, which needs all pairs.

    :param means: :func:`participant_means`'s output for this pair.
    :type means: np.ndarray
    :param smaller: The nested variant.
    :type smaller: str
    :param larger: The variant containing it.
    :type larger: str
    :param seed: Bootstrap seed.
    :type seed: int
    :return: One row of :data:`FIELDS` with ``p_holm`` still unset.
    :rtype: dict[str, Any]
    """
    rng = np.random.default_rng(seed)
    boots = [means[rng.integers(0, len(means), len(means))].mean() for _ in range(N_BOOT)]
    ci_lo, ci_hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    test = wilcoxon(means)
    return {"smaller": smaller, "larger": larger, "n_participants": len(means),
            "mean_delta_bic": float(means.mean()), "ci_lo": ci_lo, "ci_hi": ci_hi,
            "median_delta_bic": float(np.median(means)),
            "n_favouring_larger": int(np.sum(means > 0)),
            "wilcoxon_w": float(test.statistic), "p": float(test.pvalue), "p_holm": None}


def run_tests(results_dir: Path, variants: list[str], seed: int = 0) -> list[dict[str, Any]]:
    """
    The tests for every pair of `variants`, Holm-adjusted together.

    :param results_dir: Directory holding the `fit_variant_<name>.csv` files.
    :type results_dir: Path
    :param variants: Variants to compare; ordered into :data:`FIT_ORDER` here.
    :type variants: list[str]
    :param seed: Bootstrap seed.
    :type seed: int
    :return: One :data:`FIELDS` row per pair.
    :rtype: list[dict[str, Any]]
    :raises ValueError: If a variant outside the nested chain (the full-visibility model) is
        requested.
    """
    # | [full-visibility] Every pair here is read as (smaller, larger) of the nested chain; the
    # | full-visibility model is in no chain, so it would be mislabelled. Its own read-out is
    # | scripts/full_visibility_sanity.py (Entry 36).
    outside = [v for v in variants if v not in ATTENTION_CHOOSING_VARIANTS]
    if outside:
        raise ValueError(f"{outside} are not in the nested chain; use "
                         f"scripts/full_visibility_sanity.py for the full-visibility model")
    ordered = [v for v in FIT_ORDER if v in variants]
    table = episode_table(results_dir, ordered)
    rows = [pair_row(participant_means(table, s, l), s, l, seed)
            for s, l in combinations(ordered, 2)]
    # | Holm: the smallest p times the number of pairs, the next times one fewer, each raised
    # | to at least the one before it. Ex: [0.01, 0.04, 0.03] becomes [0.03, 0.06, 0.06].
    adjusted = multipletests([r["p"] for r in rows], method="holm")[1]
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = float(p_holm)
    return rows


def format_rows(rows: list[dict[str, Any]]) -> str:
    """
    The rows as plain ASCII text (Rule 12), one line per pair.

    :param rows: :func:`run_tests`'s output.
    :type rows: list[dict[str, Any]]
    :return: Multi-line text.
    :rtype: str
    """
    lines = ["per-participant mean of BIC(smaller) - BIC(larger); positive favours the larger "
             "model; 95% participant bootstrap; Wilcoxon signed-rank, Holm across pairs", ""]
    for r in rows:
        lines.append(f"{VARIANTS[r['larger']].display} over {VARIANTS[r['smaller']].display}: "
                     f"{r['mean_delta_bic']:.1f} [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}], median "
                     f"{r['median_delta_bic']:.1f}, favouring larger {r['n_favouring_larger']}/"
                     f"{r['n_participants']}, W = {r['wilcoxon_w']:.0f}, p = {r['p']:.2e}, "
                     f"Holm p = {r['p_holm']:.2e}")
    return "\n".join(lines)


def main() -> None:
    """CLI: read the fit files, run the tests, write the CSV and text beside them."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="directory holding fit_variant_<name>.csv; outputs land beside them")
    ap.add_argument("--variants", nargs="+", default=None,
                    choices=list(ATTENTION_CHOOSING_VARIANTS),
                    metavar="NAME", help="variants to compare (default: every fit file present)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    variants = args.variants or available_variants(results_dir)
    rows = run_tests(results_dir, variants)
    with io.open(results_dir / "participant_bic_tests.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    text = format_rows(rows)
    (results_dir / "participant_bic_tests.txt").write_text(text + "\n", encoding="utf-8")
    logger.info("\n" + text)
    logger.info(f"wrote {results_dir / 'participant_bic_tests.csv'} and .txt")


if __name__ == "__main__":
    main()
