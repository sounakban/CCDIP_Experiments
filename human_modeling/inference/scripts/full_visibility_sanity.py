"""
The full-visibility SANITY CHECK (implementation.md Entry 36): does forcing every car visible
change how well the no-cognitive-cost model fits the human actions?

THE QUESTION (user, 2026-11-10)
    Both models fit the same three coefficients with cognitive cost pinned at 0. The
    no-cognitive-cost model (`C_no_cognitive_cost`) still chooses which cars to attend and
    averages over those choices; the full-visibility model (`C_full_visibility`) sees the 6
    nearest cars always. The hypothesis under test is that removing the attention choice does
    NOT change the fit meaningfully. Per episode,

        delta BIC = BIC(full-visibility) - BIC(no-cognitive-cost)

    so a POSITIVE value means full visibility fits the human actions worse. Both models have
    3 free coefficients and the same n frames, so the k ln(n) terms cancel and delta BIC is
    exactly twice the difference in negative log-likelihood.

WHAT IT DOES NOT ANSWER
    Whether the two models DRIVE alike when rolled out (collisions, lane changes): that is
    `estimation/validate.py`'s question and was not asked for. Nor parameter recovery. This
    is recorded as a sanity check, not as a finding of the paper.

THE STATISTICS (user, 2026-11-10), per EPISODE as in the model-comparison figures
    mean, SD (ddof 1), and the SE two ways: SD / sqrt(n episodes), the convention of
    `model_comparison_summary.txt`, which ignores that a participant's episodes are
    correlated; and the participant-bootstrap SE, the SD of the bootstrap means, which does
    not. The 95% interval is the percentile interval of a bootstrap that resamples whole
    PARTICIPANTS (Field & Welsh 2007, "Bootstrapping clustered data", JRSS-B 69:369-390;
    Efron & Tibshirani 1993, "An Introduction to the Bootstrap", Chapman & Hall) -- the same
    draws, at the same seed, as `plot_model_comparison.paired_difference`, so it is the
    interval the figures would show.

    The verdict is an equivalence test (Schuirmann 1987, "A comparison of the two one-sided
    tests procedure...", J. Pharmacokinetics and Biopharmaceutics 15:657-680; Lakens 2017,
    "Equivalence tests: a practical primer...", Social Psychological and Personality Science
    8:355-362) with the margin fixed BEFORE the data: +-10 BIC per episode, the comparison's
    own ambiguity band (Kass & Raftery 1995, "Bayes factors", JASA 90:773-795: above 10 is
    "very strong" evidence). Using the 95% interval rather than TOST's usual 90% makes the
    test conservative (each side at 2.5%). Three outcomes:
        "no meaningful change"  the whole interval lies inside (-10, +10)
        "meaningful change"     the whole interval lies outside it, on one side
        "inconclusive"          the interval straddles a margin

    PAIRED ARM: the no-cognitive-cost model is REFITTED in the same job, so both arms share
    the code, the device and the run. The same statistics against the STORED no-cognitive-cost
    fits (the October run) are written as a second row, so a reader can see whether the
    verdict depends on which fit it is read against.

THE REFIT CHECK (user, 2026-11-10: "report if your changes caused a change in the fit")
    Per episode, the refitted no-cognitive-cost peak against the stored one: the log-likelihood
    at the peak and the three fitted coefficients. The code path the no-cognitive-cost model
    runs is unchanged by Entry 36 -- the code before and after it fitted one episode
    identically in all 40 output columns on a local CPU -- so a difference here is noise
    between devices (0.043 nats on that episode, local CPU vs the October cluster GPU), not
    the change.

OUTPUTS in ``--results-dir`` (the directory holding both `fit_variant_*.csv`):
    `delta_bic_full_visibility.csv`           one row per reference (refit, stored)
    `delta_bic_full_visibility_episodes.csv`  the per-episode values behind it
    `refit_check_C_no_cognitive_cost.csv`     per-episode refit-vs-stored differences
    `delta_bic_full_visibility.txt`           all of the above, printed

Typical usage::

    python -m human_modeling.inference.scripts.full_visibility_sanity \\
        --results-dir human_modeling/inference/results/model_comparison/sanity_full_visibility \\
        --stored-fit human_modeling/inference/results/model_comparison/fit/fit_variant_C_no_cognitive_cost.csv
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from human_modeling.inference.scripts.bootstrap_stats import (N_BOOT, bootstrap_indices,
                                                                 group_clusters)
from human_modeling.inference.scripts.plot_model_comparison import (BIC_DELTA_THRESHOLD,
                                                                       bic, read_fit)
from human_modeling.inference.variants import get_variant

FULL = "C_full_visibility"
REFERENCE = "C_no_cognitive_cost"
EQUIVALENCE_MARGIN = BIC_DELTA_THRESHOLD   # +-10 per episode, the figures' ambiguity band
# | Refit differences above this many nats are counted in the report. Set from ONE
# | measurement (2026-11-10, episode q2Hacr0m g1): the same code fitted on a local CPU and on
# | the cluster GPU of the October run reached peaks 0.043 nats apart, while the code before
# | and after Entry 36 was bit-identical on the CPU. So device noise alone reaches a few
# | hundredths of a nat; 0.1 is about twice that and 1/50 of the delta-BIC margin in nats.
REFIT_MATCH_NATS = 0.1
SUMMARY_FIELDS = ["comparison", "reference_source", "n_episodes", "n_participants", "mean",
                  "sd", "se_naive", "se_bootstrap", "ci_lo", "ci_hi", "median",
                  "n_full_visibility_better", "n_within_margin", "margin", "verdict"]
# | Sample row (ILLUSTRATIVE values, not results):
# | {"comparison": "BIC(full-visibility) - BIC(no-cognitive-cost)",
# | "reference_source": "refit (same job)", "n_episodes": 230, "n_participants": 39,
# | "mean": 4.2, "sd": 9.8, "se_naive": 0.65, "se_bootstrap": 0.9, "ci_lo": 2.5, "ci_hi": 6.1,
# | "median": 3.0, "n_full_visibility_better": 60, "n_within_margin": 190, "margin": 10.0,
# | "verdict": "no meaningful change"}
EPISODE_FIELDS = ["subject_id", "game_index", "n_frames", "logL_full_visibility",
                  "logL_reference_refit", "delta_bic_vs_refit", "logL_reference_stored",
                  "delta_bic_vs_stored"]
REFIT_FIELDS = ["subject_id", "game_index", "logL_mode_refit", "logL_mode_stored",
                "delta_logL", "max_abs_coef_diff"]




# ===========================================================================
# 1. Reading
# ===========================================================================

def check_variant_column(path: Path, expected: str) -> None:
    """
    Refuse a fit file written under another variant than the one it is read as.

    `read_fit` keeps only logL, k and n, so B's file passed as the stored C would otherwise be
    read silently as a k = 4 reference.

    :param path: A `fit_variant_*.csv`.
    :type path: Path
    :param expected: The variant name every row must carry. Ex: ``"C_no_cognitive_cost"``.
    :type expected: str
    :raises ValueError: If any row names another variant.
    """
    with io.open(path, encoding="utf-8", newline="") as fh:
        found = {r.get("variant", "") for r in csv.DictReader(fh)}
    if found != {expected}:
        raise ValueError(f"{path} holds variant(s) {sorted(found)}, expected only {expected!r}")


def read_modes(path: Path) -> dict[tuple[str, int], np.ndarray]:
    """
    The fitted peak of every FREE coefficient of the no-cognitive-cost model, per episode.

    :param path: A `fit_variant_C_no_cognitive_cost.csv`.
    :type path: Path
    :return: ``{(subject_id, game_index): (3,) peak values}`` in the variant's free order.
        Ex: ``{("q2Hacr0m...", 1): array([1.81, 4.94, 0.0])}``.
    :rtype: dict[tuple[str, int], np.ndarray]
    """
    names = get_variant(REFERENCE).free_names
    with io.open(path, encoding="utf-8", newline="") as fh:
        return {(r["subject_id"], int(r["game_index"])):
                np.array([float(r[f"mode_{n}"]) for n in names], dtype=np.float64)
                for r in csv.DictReader(fh)}


def episode_delta_bic(full: dict[tuple[str, int], dict[str, Any]],
                      reference: dict[tuple[str, int], dict[str, Any]],
                      keys: list[tuple[str, int]]) -> np.ndarray:
    """
    ``BIC(full-visibility) - BIC(reference)`` per episode, over ``keys``.

    :param full: `plot_model_comparison.read_fit` output for the full-visibility model.
    :type full: dict[tuple[str, int], dict[str, Any]]
    :param reference: The same for the no-cognitive-cost model.
    :type reference: dict[tuple[str, int], dict[str, Any]]
    :param keys: Episodes present in both, in the order wanted.
    :type keys: list[tuple[str, int]]
    :return: ``(len(keys),)``; positive means full visibility fits worse.
    :rtype: np.ndarray
    :raises ValueError: If an episode's frame count differs between the two files, which
        would mean they were fitted to different data.
    """
    mismatched = [k for k in keys if full[k]["n"] != reference[k]["n"]]
    if mismatched:
        raise ValueError(f"{len(mismatched)} episodes have different n_frames in the two fit "
                         f"files (first: {mismatched[0]}); they were not fitted to the same data")

    def column(fits: dict[tuple[str, int], dict[str, Any]], field: str) -> np.ndarray:
        """One field of every episode in ``keys``, as a float64 array."""
        return np.array([fits[k][field] for k in keys], dtype=np.float64)    # (n_keys,)

    return (bic(column(full, "logL"), column(full, "k"), column(full, "n"))
            - bic(column(reference, "logL"), column(reference, "k"), column(reference, "n")))




# ===========================================================================
# 2. Statistics
# ===========================================================================

def equivalence_verdict(ci_lo: float, ci_hi: float, margin: float) -> str:
    """
    The three-outcome equivalence verdict on an interval against ``+-margin``.

    :param ci_lo: Lower end of the interval.
    :type ci_lo: float
    :param ci_hi: Upper end of the interval.
    :type ci_hi: float
    :param margin: The equivalence margin, positive.
    :type margin: float
    :return: ``"no meaningful change"``, ``"meaningful change"`` or ``"inconclusive"``.
    :rtype: str
    """
    if -margin < ci_lo and ci_hi < margin:
        return "no meaningful change"
    if ci_lo >= margin or ci_hi <= -margin:
        return "meaningful change"
    return "inconclusive"


def delta_summary(delta: np.ndarray, subjects: np.ndarray, seed: int = 0,
                  margin: float = EQUIVALENCE_MARGIN) -> dict[str, Any]:
    """
    Mean, SD, both SEs, the participant-bootstrap interval and the verdict, per episode.

    The bootstrap draws are those of `plot_model_comparison.paired_difference` at the same
    seed (one generator, :func:`bootstrap_indices` over the grouped participants, `N_BOOT`
    draws), so the interval equals the one the model-comparison figures would draw.

    :param delta: Per-episode delta BIC, ``(n_episodes,)``.
    :type delta: np.ndarray
    :param subjects: Participant per episode, ``(n_episodes,)``; whole participants are
        resampled.
    :type subjects: np.ndarray
    :param seed: Bootstrap seed.
    :type seed: int
    :param margin: Equivalence margin per episode.
    :type margin: float
    :return: The :data:`SUMMARY_FIELDS` statistics (without the two label columns).
    :rtype: dict[str, Any]
    :raises ValueError: If fewer than two episodes are given (no SD exists).
    """
    if len(delta) < 2:
        raise ValueError(f"need at least 2 episodes for an SD, got {len(delta)}")
    # | 1. Participant bootstrap, drawn exactly as paired_difference draws it.
    rng = np.random.default_rng(seed)
    members = group_clusters(subjects)
    draws = np.array([delta[bootstrap_indices(rng, len(delta), members)].mean()
                      for _ in range(N_BOOT)])                              # (N_BOOT,)
    ci_lo, ci_hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))

    # | 2. The descriptive numbers and the verdict.
    sd = float(np.std(delta, ddof=1))
    return {"n_episodes": int(len(delta)), "n_participants": int(len(members)),
            "mean": float(delta.mean()), "sd": sd, "se_naive": sd / float(np.sqrt(len(delta))),
            "se_bootstrap": float(np.std(draws, ddof=1)), "ci_lo": ci_lo, "ci_hi": ci_hi,
            "median": float(np.median(delta)),
            "n_full_visibility_better": int(np.sum(delta < 0)),
            "n_within_margin": int(np.sum(np.abs(delta) < margin)), "margin": float(margin),
            "verdict": equivalence_verdict(ci_lo, ci_hi, margin)}


def refit_rows(refit: dict[tuple[str, int], dict[str, Any]],
               stored: dict[tuple[str, int], dict[str, Any]],
               refit_modes: dict[tuple[str, int], np.ndarray],
               stored_modes: dict[tuple[str, int], np.ndarray],
               keys: list[tuple[str, int]]) -> list[dict[str, Any]]:
    """
    Per episode, the refitted no-cognitive-cost peak against the stored one.

    :param refit: `read_fit` output of the refit.
    :type refit: dict[tuple[str, int], dict[str, Any]]
    :param stored: `read_fit` output of the stored fit.
    :type stored: dict[tuple[str, int], dict[str, Any]]
    :param refit_modes: :func:`read_modes` of the refit.
    :type refit_modes: dict[tuple[str, int], np.ndarray]
    :param stored_modes: :func:`read_modes` of the stored fit.
    :type stored_modes: dict[tuple[str, int], np.ndarray]
    :param keys: Episodes present in both.
    :type keys: list[tuple[str, int]]
    :return: One :data:`REFIT_FIELDS` row per episode.
    :rtype: list[dict[str, Any]]
    """
    return [{"subject_id": k[0], "game_index": k[1],
             "logL_mode_refit": refit[k]["logL"], "logL_mode_stored": stored[k]["logL"],
             "delta_logL": refit[k]["logL"] - stored[k]["logL"],
             "max_abs_coef_diff": float(np.max(np.abs(refit_modes[k] - stored_modes[k])))}
            for k in keys]




# ===========================================================================
# 3. Report
# ===========================================================================

def format_report(summaries: list[dict[str, Any]], refit: list[dict[str, Any]],
                  n_refit_missing: int) -> str:
    """
    Every number as plain ASCII text (Rule 12).

    :param summaries: The :data:`SUMMARY_FIELDS` rows.
    :type summaries: list[dict[str, Any]]
    :param refit: :func:`refit_rows` output.
    :type refit: list[dict[str, Any]]
    :param n_refit_missing: Refitted episodes with no stored counterpart.
    :type n_refit_missing: int
    :return: Multi-line text.
    :rtype: str
    """
    lines = ["SANITY CHECK, not a finding (implementation.md Entry 36).",
             "delta BIC per episode = BIC(full-visibility) - BIC(no-cognitive-cost); positive "
             "means full visibility fits the human actions worse. Both models have 3 free "
             "coefficients, so delta BIC = 2 x (NLL difference).",
             "SE naive = SD/sqrt(n episodes); SE bootstrap and the 95% interval resample whole "
             f"participants ({N_BOOT} draws). Verdict: equivalence at +-{EQUIVALENCE_MARGIN:g} "
             "per episode on the 95% interval.", ""]
    for s in summaries:
        lines += [f"against the {s['reference_source']} no-cognitive-cost fit: "
                  f"episodes {s['n_episodes']}, participants {s['n_participants']}",
                  f"  mean {s['mean']:+.2f}  SD {s['sd']:.2f}  SE naive {s['se_naive']:.2f}  "
                  f"SE bootstrap {s['se_bootstrap']:.2f}  95% CI [{s['ci_lo']:+.2f}, "
                  f"{s['ci_hi']:+.2f}]  median {s['median']:+.2f}",
                  f"  full visibility better on {s['n_full_visibility_better']} episodes; "
                  f"|delta| < {s['margin']:g} on {s['n_within_margin']}",
                  f"  VERDICT: {s['verdict']}"]
        # | With one participant every bootstrap draw is the same set of episodes, so the
        # | interval collapses to the mean and the verdict carries no uncertainty at all.
        if s["n_participants"] < 2:
            lines.append("  NOTE: one participant only -- the interval is degenerate and the "
                         "verdict is not meaningful")
        lines.append("")
    if refit:
        d = np.abs(np.array([r["delta_logL"] for r in refit]))
        c = np.array([r["max_abs_coef_diff"] for r in refit])
        lines += [f"refit check: the no-cognitive-cost model refitted in this job against the "
                  f"stored fit, {len(refit)} episodes",
                  f"  |logL_mode difference| in nats: max {d.max():.3g}, median "
                  f"{np.median(d):.3g}; above {REFIT_MATCH_NATS:g} nats on "
                  f"{int(np.sum(d > REFIT_MATCH_NATS))} episodes, above 1 nat on "
                  f"{int(np.sum(d > 1.0))}",
                  f"  largest fitted-coefficient difference: max {c.max():.3g}, median "
                  f"{np.median(c):.3g}"]
    if n_refit_missing:
        lines.append(f"  {n_refit_missing} refitted episodes have no stored counterpart")
    return "\n".join(lines)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """
    Write rows under a header, UTF-8.

    :param path: Output file.
    :type path: Path
    :param fields: Column order.
    :type fields: list[str]
    :param rows: One dict per row.
    :type rows: list[dict[str, Any]]
    """
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)




# ===========================================================================
# 4. Entry point
# ===========================================================================

def main() -> None:
    """
    1. read both fit files from the results dir and the stored reference; 2. per-episode
    delta BIC against the refit and against the stored fit; 3. statistics and verdicts;
    4. the refit check; 5. write the CSVs and the text.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                    help="directory holding fit_variant_C_full_visibility.csv and "
                         "fit_variant_C_no_cognitive_cost.csv (the refit); outputs land here")
    ap.add_argument("--stored-fit", default=None,
                    help="the stored fit_variant_C_no_cognitive_cost.csv to compare against "
                         "(the October run); omitted, only the paired refit row is written")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    # | 1. Read, refusing a file that holds another model. Episodes in both arms only; every
    # | count left out is reported, not hidden.
    logger.info(f"reading fits from {results_dir}")
    full_path, refit_path = (results_dir / f"fit_variant_{FULL}.csv",
                             results_dir / f"fit_variant_{REFERENCE}.csv")
    check_variant_column(full_path, FULL)
    check_variant_column(refit_path, REFERENCE)
    full, refit = read_fit(full_path), read_fit(refit_path)
    keys = sorted(set(full) & set(refit))
    if not keys:
        raise ValueError(f"no episode is in both fit files in {results_dir}")
    n_unpaired = len(set(full) ^ set(refit))
    subjects = np.array([k[0] for k in keys])
    stored: dict[tuple[str, int], dict[str, Any]] = {}
    if args.stored_fit:
        check_variant_column(Path(args.stored_fit), REFERENCE)
        stored = read_fit(Path(args.stored_fit))

    # | 2-3. Delta BIC and statistics against the refit, then against the stored fit.
    delta_refit = episode_delta_bic(full, refit, keys)
    comparison = "BIC(full-visibility) - BIC(no-cognitive-cost)"
    summaries = [{"comparison": comparison, "reference_source": "refit (same job)",
                  **delta_summary(delta_refit, subjects)}]
    stored_keys = [k for k in keys if k in stored]
    delta_stored = {}
    if stored_keys:
        values = episode_delta_bic(full, stored, stored_keys)
        delta_stored = dict(zip(stored_keys, values))
        summaries.append({"comparison": comparison, "reference_source": "stored (October run)",
                          **delta_summary(values, np.array([k[0] for k in stored_keys]))})

    # | 4. The refit check.
    refit_check: list[dict[str, Any]] = []
    if stored_keys:
        refit_check = refit_rows(refit, stored, read_modes(results_dir / f"fit_variant_{REFERENCE}.csv"),
                                 read_modes(Path(args.stored_fit)), stored_keys)

    # | 5. Write.
    episodes = [{"subject_id": k[0], "game_index": k[1], "n_frames": full[k]["n"],
                 "logL_full_visibility": full[k]["logL"], "logL_reference_refit": refit[k]["logL"],
                 "delta_bic_vs_refit": float(d),
                 "logL_reference_stored": stored[k]["logL"] if k in stored else "",
                 "delta_bic_vs_stored": float(delta_stored[k]) if k in delta_stored else ""}
                for k, d in zip(keys, delta_refit)]
    write_csv(results_dir / "delta_bic_full_visibility.csv", SUMMARY_FIELDS, summaries)
    write_csv(results_dir / "delta_bic_full_visibility_episodes.csv", EPISODE_FIELDS, episodes)
    if refit_check:
        write_csv(results_dir / "refit_check_C_no_cognitive_cost.csv", REFIT_FIELDS, refit_check)
    text = format_report(summaries, refit_check, len(keys) - len(stored_keys) if stored else 0)
    if n_unpaired:
        text += f"\nNOTE: {n_unpaired} episodes are in only one of the two fit files and were left out"
    n_stored_only = len(set(stored) - set(keys))
    if n_stored_only:
        text += (f"\nNOTE: {n_stored_only} stored episodes have no pair in this job's fits and "
                 f"are in neither the stored row nor the refit check")
    (results_dir / "delta_bic_full_visibility.txt").write_text(text + "\n", encoding="utf-8")
    logger.info("\n" + text)
    logger.info(f"wrote delta_bic_full_visibility.{{csv,txt}}, the per-episode CSV and the "
                f"refit check to {results_dir}")


if __name__ == "__main__":
    main()
