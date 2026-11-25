"""
Between-participant and within-participant variance of the fitted parameters (implementation.md
Entry 35 addendum 5).

WHAT IS DECOMPOSED
    One fit file (default: the fully-parameterized model, A_full), one value per episode for
    each free coefficient: the posterior MEAN for a reward parameter and the likelihood PEAK for
    a heuristic weight, the same point estimates the recovery study scores
    (`run_synthetic_recovery.fit_columns`). Each participant contributes 4-6 episodes.

HOW
    1. :func:`variance_components`, the one-way random-effects decomposition
       (Shrout & Fleiss 1979, ICC(1,1)): within = the within-participant mean square; between =
       the spread of the participants' true means once the episode noise their averages carry
       is subtracted, so it can come out slightly negative when people differ less than chance.
       share = between / (between + within).
    2. Twice: on the raw values, and with each ROUTE's mean removed first. Every participant
       drove the same six routes, so a coefficient that depends on the route puts route
       differences into the within-participant term; the route-adjusted figures are the ones to
       read (:func:`map_demean`).
    3. A 95% percentile interval for the route-adjusted figures, resampling participants
       (`bootstrap_stats.N_BOOT` draws); a participant drawn twice counts as two participants,
       and route means are re-estimated inside every draw.

ROUTES
    Each episode's route is read from the human data itself: the label is the content hash
    of the episode's map (`run_synthetic_recovery.map_id_of`), so two episodes on the same
    road share it.

OUTPUTS in ``--out-dir``: `parameter_variance_<variant>.csv` and `.txt`.

Typical usage::

    python -m human_modeling.inference.scripts.parameter_variance \\
        --fit-csv human_modeling/inference/results/model_comparison/fit/fit_variant_A_full.csv \\
        --out-dir human_modeling/inference/results/model_comparison/fit
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from human_modeling.inference.scripts.bootstrap_stats import N_BOOT
from human_modeling.inference.scripts.run_synthetic_recovery import fit_columns, human_routes
from human_modeling.inference.variants import FIT_ORDER, VARIANTS

DEFAULT_DATA_PATH = Path("human_modeling/data/exp1_filtered.json")

FIELDS = ["coefficient", "column", "n_people", "n_episodes", "mean", "total_var",
          "between_raw", "within_raw", "share_raw",
          "between", "between_ci_lo", "between_ci_hi", "within", "within_ci_lo", "within_ci_hi",
          "share", "share_ci_lo", "share_ci_hi"]
# | Sample row (route-adjusted unless _raw): {"coefficient": "hlp_collision_penalty",
# | "column": "hlp_collision_penalty", "n_people": 39, "n_episodes": 230, "mean": 7.9,
# | "between": 0.9, "within": 3.1, "share": 0.22, "share_ci_lo": 0.10, ...}


def read_episodes(fit_csv: Path, data_path: Path = DEFAULT_DATA_PATH,
                  ) -> tuple[list[dict[str, str]], np.ndarray]:
    """
    The fit rows and each row's route, refusing any episode without a route.

    :param fit_csv: A `fit_variant_<name>.csv`.
    :type fit_csv: Path
    :param data_path: The human data JSON the routes come from.
    :type data_path: Path
    :return: ``(fit rows, route label per row (n_episodes,))``.
    :rtype: tuple[list[dict[str, str]], np.ndarray]
    :raises ValueError: If a fitted episode has no route in the data.
    """
    with io.open(fit_csv, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    logger.info(f"reading routes from {data_path} (minutes: the file is ~900 MB)")
    route_of = human_routes(data_path)
    missing = [(r["subject_id"], r["game_index"]) for r in rows
               if (r["subject_id"], int(r["game_index"])) not in route_of]
    if missing:
        raise ValueError(f"{len(missing)} fitted episodes have no route in {data_path}, "
                         f"e.g. {missing[:3]}")
    routes = np.array([route_of[(r["subject_id"], int(r["game_index"]))] for r in rows])
    return rows, routes


# ===========================================================================
# The variance decomposition
# ===========================================================================

def map_demean(values: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """Remove each route's own mean, so route differences stop counting as person noise.

    Map is a crossed, near-balanced factor in this dataset, not a nuisance nested inside a
    person. A one-way person model puts all of its variance into the within-person term,
    which deflates every person-identifiability figure built on it. MEASURED on synthetic
    data with person SD 1 and map SD 2 over 20 people by 6 maps: the one-way share reads
    0.041 where the map-controlled share reads 0.950.

    :param values: Marker value per episode, ``(n_episodes,)``.
    :type values: np.ndarray
    :param maps: Route label per episode, ``(n_episodes,)``.
    :type maps: np.ndarray
    :return: Values with their own route's mean subtracted, ``(n_episodes,)``; NaN entries
        pass through untouched.
    :rtype: np.ndarray
    """
    out = np.array(values, dtype=float, copy=True)               # (n_episodes,)
    for label in set(np.asarray(maps).tolist()):
        rows = (np.asarray(maps) == label) & np.isfinite(out)    # (n_episodes,) selection
        if rows.any():
            out[rows] -= out[rows].mean()
    return out


def variance_components(
    values: np.ndarray, people: np.ndarray, maps: np.ndarray | None = None,
) -> dict[str, float]:
    """The between-person and within-person variance of one value, and their ratio.

    The one-way random-effects decomposition: the within-person variance is the within-group
    mean square ``MSW``, and the between-person variance is ``(MSB - MSW) / n0``, the spread
    of the people's true means once the episode noise their averages still carry is
    subtracted, with ``n0`` the bias-corrected average group size for unbalanced groups
    (Shrout & Fleiss 1979, "Intraclass correlations: uses in assessing rater reliability",
    Psychological Bulletin 86(2):420-428, model ICC(1,1)). ``share`` is the ratio of the
    first to their sum, computed with the original ICC(1,1) expression
    ``(MSB - MSW) / (MSB + (n0 - 1) * MSW)``, the reliability of a SINGLE episode as a
    measurement of the person who played it.

    Sample reading: ``{"between": 0.21, "within": 0.64, "share": 0.25, "n_people": 39,
    "n_episodes": 230}``.

    :param values: Value per episode, ``(n_episodes,)``; NaN entries are skipped.
    :type values: np.ndarray
    :param people: Person label per episode, ``(n_episodes,)``.
    :type people: np.ndarray
    :param maps: Route label per episode, ``(n_episodes,)``, to remove route means first
        (:func:`map_demean`); None for the plain one-way decomposition.
    :type maps: np.ndarray | None
    :return: ``between`` (can be negative when people differ less than chance), ``within``,
        ``share``, ``n_people`` and ``n_episodes``; the three variances are NaN with fewer
        than two people or no within-person replication.
    :rtype: dict[str, float]
    """
    prepared = values if maps is None else map_demean(values, maps)
    groups: dict[Any, list[float]] = collections.defaultdict(list)
    for value, person in zip(prepared, people):
        if np.isfinite(value):
            groups[person].append(float(value))
    sizes = np.array([len(v) for v in groups.values()], dtype=float)   # (n_people,)
    n_people, n_total = len(sizes), sizes.sum()
    out = {"between": float("nan"), "within": float("nan"), "share": float("nan"),
           "n_people": n_people, "n_episodes": int(n_total)}
    if n_people < 2 or n_total <= n_people:
        return out
    grand_mean = float(np.mean(np.concatenate([np.array(v) for v in groups.values()])))
    mean_sq_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2
                          for v in groups.values()) / (n_people - 1)
    mean_sq_within  = sum(float(((np.array(v) - np.mean(v)) ** 2).sum())
                          for v in groups.values()) / (n_total - n_people)
    size_correction = (n_total - (sizes ** 2).sum() / n_total) / (n_people - 1)
    out["within"] = float(mean_sq_within)
    out["between"] = float((mean_sq_between - mean_sq_within) / size_correction)
    denominator = mean_sq_between + (size_correction - 1.0) * mean_sq_within
    if denominator != 0:
        out["share"] = float((mean_sq_between - mean_sq_within) / denominator)
    return out


def bootstrap_components(values: np.ndarray, people: np.ndarray, routes: np.ndarray,
                         seed: int) -> dict[str, tuple[float, float]]:
    """
    95% participant-bootstrap intervals for the route-adjusted components.

    :param values: Value per episode, ``(n_episodes,)``.
    :type values: np.ndarray
    :param people: Participant per episode, ``(n_episodes,)``.
    :type people: np.ndarray
    :param routes: Route per episode, ``(n_episodes,)``.
    :type routes: np.ndarray
    :param seed: Bootstrap seed.
    :type seed: int
    :return: ``{"between" | "within" | "share": (lo, hi)}``.
    :rtype: dict[str, tuple[float, float]]
    """
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(people == p) for p in np.unique(people)]
    draws: dict[str, list[float]] = {"between": [], "within": [], "share": []}
    for _ in range(N_BOOT):
        picked = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in picked])                 # (~n_episodes,)
        # | Each DRAW of a participant is its own label, so a repeat is two people, not one
        # | person with twice the episodes.
        labels = np.concatenate([np.full(len(members[i]), k) for k, i in enumerate(picked)])
        comp = variance_components(values[idx], labels, routes[idx])
        for name in draws:
            draws[name].append(comp[name])
    return {name: tuple(float(v) for v in np.nanpercentile(vals, [2.5, 97.5]))
            for name, vals in draws.items()}


def decompose(rows: list[dict[str, str]], routes: np.ndarray, variant: str,
              seed: int = 0) -> list[dict[str, Any]]:
    """
    One :data:`FIELDS` row per free coefficient of `variant`.

    :param rows: Fit rows from :func:`read_episodes`.
    :type rows: list[dict[str, str]]
    :param routes: Route per row.
    :type routes: np.ndarray
    :param variant: The fitted variant's name.
    :type variant: str
    :param seed: Bootstrap seed.
    :type seed: int
    :return: The rows, in the variant's free-coefficient order.
    :rtype: list[dict[str, Any]]
    """
    people = np.array([r["subject_id"] for r in rows])
    out = []
    for coefficient, column in fit_columns(VARIANTS[variant]).items():
        logger.info(f"decomposing {coefficient} ({column})")
        values = np.array([float(r[column]) for r in rows])               # (n_episodes,)
        raw = variance_components(values, people)
        adj = variance_components(values, people, routes)
        ci = bootstrap_components(values, people, routes, seed)
        out.append({"coefficient": coefficient, "column": column,
                    "n_people": adj["n_people"], "n_episodes": adj["n_episodes"],
                    "mean": float(values.mean()), "total_var": float(values.var(ddof=1)),
                    "between_raw": raw["between"], "within_raw": raw["within"],
                    "share_raw": raw["share"],
                    **{k: adj[k] for k in ("between", "within", "share")},
                    **{f"{k}_ci_{side}": ci[k][j] for k in ("between", "within", "share")
                       for j, side in enumerate(("lo", "hi"))}})
    return out


def format_rows(rows: list[dict[str, Any]], variant: str) -> str:
    """
    The rows as plain ASCII text (Rule 12), one line per coefficient.

    :param rows: :func:`decompose`'s output.
    :type rows: list[dict[str, Any]]
    :param variant: The fitted variant's name.
    :type variant: str
    :return: Multi-line text.
    :rtype: str
    """
    lines = [f"{VARIANTS[variant].display}: variance of the fitted coefficients across "
             f"{rows[0]['n_episodes']} episodes of {rows[0]['n_people']} participants",
             "route-adjusted between / within variance (SD in brackets), share between "
             "[95% participant bootstrap]; raw share without route adjustment last", ""]
    for r in rows:
        sd_b = np.sqrt(max(r["between"], 0.0))
        lines.append(f"{r['coefficient']:<30} between {r['between']:9.4g} ({sd_b:.3g})  "
                     f"within {r['within']:9.4g} ({np.sqrt(r['within']):.3g})  share "
                     f"{r['share']:+.2f} [{r['share_ci_lo']:+.2f}, {r['share_ci_hi']:+.2f}]  "
                     f"raw share {r['share_raw']:+.2f}")
    return "\n".join(lines)


def main() -> None:
    """CLI: read the fit and route tables, decompose every coefficient, write CSV and text."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-csv", required=True, help="a fit_variant_<name>.csv")
    ap.add_argument("--variant", default="A_full", choices=list(FIT_ORDER),
                    help="the variant the fit file holds (decides which columns are read)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # | 1. Read. 2. Decompose each coefficient. 3. Write the table and its text form.
    rows, routes = read_episodes(Path(args.fit_csv))
    logger.info(f"read {len(rows)} episodes on {len(set(routes))} routes from {args.fit_csv}")
    table = decompose(rows, routes, args.variant)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"parameter_variance_{args.variant}"
    with io.open(f"{stem}.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)
    text = format_rows(table, args.variant)
    Path(f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    logger.info("\n" + text)
    logger.info(f"wrote {stem}.csv and .txt")


if __name__ == "__main__":
    main()
