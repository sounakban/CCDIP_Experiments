"""
Person-level predicted-vs-real plots: average each behavioural metric across a person's own
episodes before plotting, instead of plotting one point per episode.

WHY A SEPARATE VIEW FROM THE PER-EPISODE REGRESSION PLOTS
    `estimation/validate.py`'s own regression plots (`validate_regression_<metric>_<variant>
    .png`) put one point per EPISODE -- 230 of them, several per person, each fitted with its
    own theta. This script instead puts one point per PERSON: that person's own recorded value
    averaged across all their episodes, against the simulated value averaged the same way
    across those same episodes (each episode's simulated value already being the mean of its
    own 5 rollouts under that episode's own fitted theta -- see run_human_inference.py and
    estimation/validate.py's `--n-rollouts`). Averaging equal-sized per-episode groups this way
    is the same number as averaging every underlying rollout directly, since every episode
    contributes the same rollout count. This answers a different question than the per-episode
    view: not "how well does each individual drive get reproduced" but "does this person's
    overall driving style come through once episode-to-episode noise is averaged out."

WHAT IT READS
    A `validate_simulated_<variant>.csv` (`estimation/validate.py`'s output): `subject_id`,
    `human_<metric>`, and the simulated `<metric>` column, one row per episode.

WHAT IT PLOTS
    Reuses `estimation/validate.py`'s own `collision_regression_plot` UNCHANGED (same
    correlation and variance-explained R2 annotation, same y=x reference) -- only the input
    arrays differ, so the two views are directly comparable and never disagree about what
    "r" or "R2" means.

Typical usage::

    uv run python -m human_modeling.inference.scripts.plot_by_person \\
        --validate-csv <results>/variants/validate_mean/validate_simulated_mean.csv \\
        --out-dir <results>/variants/validate_mean/plots
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np

from human_modeling.inference.estimation.validate import collision_regression_plot

METRICS = (("collisions", "human_collisions", "collisions"),
           ("lane_changes", "human_lane_changes_per_1k", "lane_changes_per_1k"),
           ("median_speed", "human_median_speed", "median_speed"),
           # | [evasive-plots] the two collision-specific shares validate.py now emits; a
           # | metric absent from an older CSV is skipped rather than failing the run.
           ("evasive_lane_change_share", "human_evasive_lane_change_share",
            "evasive_lane_change_share"),
           ("car_entry_longitudinal_share", "human_car_entry_longitudinal_share",
            "car_entry_longitudinal_share"))


def read_csv(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV as string dicts.

    :param path: File to read.
    :type path: Path
    :return: One dict per row, keyed by header.
    :rtype: list[dict[str, str]]
    """
    with io.open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def average_by_person(
    rows: list[dict[str, str]], hcol: str, scol: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One (human, simulated) pair per distinct `subject_id`, each the mean of that person's
    own episode rows.

    :param rows: Episode rows from a `validate_simulated_<variant>.csv`.
    :type rows: list[dict[str, str]]
    :param hcol: The human-value column for this metric.
    :type hcol: str
    :param scol: The simulated-value column for this metric.
    :type scol: str
    :return: ``(human_per_person, simulated_per_person, subject_ids)``, all length = the
        number of distinct subjects, in the same, deterministic (sorted) order.
    :rtype: tuple[np.ndarray, np.ndarray, list[str]]
    :raises ValueError: If `rows` is empty.

    Sample reading: for a person with episodes 19/11/26/19/12/23, the returned human-side
    value is their mean, 18.33 -- one number representing that whole person, not one episode.
    """
    if not rows:
        raise ValueError("no rows to average")
    by_subject: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_subject.setdefault(r["subject_id"], []).append((float(r[hcol]), float(r[scol])))
    subjects = sorted(by_subject)
    human = np.array([np.mean([p[0] for p in by_subject[s]]) for s in subjects])
    simulated = np.array([np.mean([p[1] for p in by_subject[s]]) for s in subjects])
    return human, simulated, subjects


def main() -> None:
    """CLI: read one variant's simulated CSV, save the three person-level regression plots."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate-csv", required=True,
                    help="a validate_simulated_<variant>.csv from estimation/validate.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="",
                    help="suffix distinguishing the variant in output filenames, e.g. 'mean'")
    args = ap.parse_args()

    rows = read_csv(Path(args.validate_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    n_episodes_per_subject: dict[str, int] = {}
    for r in rows:
        n_episodes_per_subject[r["subject_id"]] = n_episodes_per_subject.get(r["subject_id"], 0) + 1

    for name, hcol, scol in METRICS:
        human, simulated, subjects = average_by_person(rows, hcol, scol)
        png = out_dir / f"validate_regression_{name}{tag}_by_person.png"
        st = collision_regression_plot(human, simulated, png, label=f"mean {name.replace('_', ' ')}")
        print(f"saved {png.name}: {len(subjects)} people (from {len(rows)} episodes, "
              f"{min(n_episodes_per_subject.values())}-{max(n_episodes_per_subject.values())} "
              f"episodes/person) r={st['r']:+.3f} R2={st['r2']:.3f} [{st['ci_lo']:+.3f}, {st['ci_hi']:+.3f}] "
              f"bias={st['bias']:+.2f}")


if __name__ == "__main__":
    main()
