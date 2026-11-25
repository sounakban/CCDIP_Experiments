"""plot_cost_performance.py — The cost-performance trade-off figure (paper Figure 2b).

Sweeps ONE cognitive-profile coefficient of a trained subtask-selection policy (SSP, the
high-level policy) over its trained range while the other coefficients stay fixed, rolls the
SSP->SP hierarchy out on the same paired maps at every value, and plots the performance cost
of the behaviour the coefficient buys: average crashes per episode against the average number
of cars the policy attends to per step. One dot per swept value, coloured by that value, so
the reader can follow the coefficient along the curve.

The rollouts come from :func:`hlp_eval.sweep`, which pairs maps across coefficient values
(the same construction seeds at every value), so a band at one value and a band at the next
are measured on identical roads. The band is +-1 standard error of the crash count over the
``--sims`` maps, one episode per map.

The swept values are chosen so that the dots land at evenly spaced attention levels along
the x axis: a short pilot sweep (``--pilot-points`` values crowded toward the low end of the
range, ``--pilot-sims`` maps each) maps the coefficient to cars attended, and that curve is
inverted at evenly spaced targets between its two ends. The non-swept coefficients stay at
the game-score background :data:`hlp_eval.GAME_THETA`. Defaults run on a laptop in a few
minutes; the paper's figure used 40 maps per value, so pass ``--sims 40`` to reproduce it.

Typical usage (from the repo root)::

    uv run python modules/base_models/SSP/plot_cost_performance.py \\
        --run-dir human_modeling/inference/checkpoint \\
        --sims 40 --points 9 --out-dir human_modeling/inference/results/cost_performance

Writes, into ``--out-dir``: ``cost_performance_<coefficient>.csv`` (the sweep table: one row
per coefficient value with the number of maps behind it, ``n_sims``, and every metric's mean
and standard error), ``cost_performance_<coefficient>_pilot.csv`` (the pilot table) and
``cost_performance_<coefficient>.png`` / ``.pdf`` (the figure).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch   # only for seeding the policies' sampled decisions
from loguru import logger

# | [import root] this module lives at modules/base_models/SSP/ -> parents[3] is the repo
# | root; the bootstrap is only needed when the file is run as a script rather than imported.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.base_models.SSP.hlp_core import HLP_PARAM_NAMES   # noqa: E402
from modules.base_models.SSP.hlp_eval import GAME_THETA, sweep  # noqa: E402

matplotlib.use("Agg")   # headless rendering; the figure is saved, never shown
import matplotlib.pyplot as plt   # noqa: E402


# ===========================================================================
# 1. Module-level constants
# ===========================================================================

# | The coefficient the paper's Figure 2b varies: the attentional cost per attended car per step.
DEFAULT_COEFFICIENT = "hlp_cognitive_cost_coef"

# | The two read-outs drawn against each other. The metric keys are those returned by
# | hlp_eval.run: "shown" is cars attended per step (x axis, high to low so the curve climbs
# | as attention drops), "crash_per_ep" is collisions per episode (y axis, with its band).
X_METRIC, X_LABEL = "shown", "avg number of cars shown per step  (high -> low)"
Y_METRIC, Y_LABEL = "crash_per_ep", "avg crashes per episode"
COLORBAR_LABEL = "attention cost"

# | CSV column holding the number of paired maps behind each row.
N_SIMS_COLUMN = "n_sims"

# | The pilot's values are spaced as lo + (hi - lo) * u**PILOT_POWER for even u, crowding the
# | low end of the range, where attention still changes.
PILOT_POWER = 2.0

# | Styling: dots coloured by the swept value on a perceptual sequential map, joined by a
# | recessive grey line; the +-1 standard-error band in a light blue.
COLORMAP     = "viridis"
LINE_COLOR   = "#8c8c8c"
BAND_COLOR   = "#2a78d6"
BAND_ALPHA   = 0.18    # translucency of the +-1 standard-error band
DOT_SIZE     = 70      # scatter marker area in points^2
FIGURE_SIZE  = (5.6, 4.0)   # inches; one panel plus its colour bar


# ===========================================================================
# 2. Helpers
# ===========================================================================

def trained_range(run_dir: Path, coefficient: str) -> tuple[float, float]:
    """
    Read a coefficient's trained ``[min, max]`` from the checkpoint's metadata.

    The sweep must stay inside this range: the policy clamps any value outside it, so points
    beyond the range would silently repeat the boundary value.

    :param run_dir: SSP checkpoint directory holding ``hlp_metadata.json``.
    :type run_dir: Path
    :param coefficient: Name of the coefficient, one of :data:`HLP_PARAM_NAMES`.
    :type coefficient: str
    :return: ``(min, max)`` of the coefficient's training range.
    :rtype: tuple[float, float]
    :raises ValueError: If the checkpoint trained the coefficient on a zero-width range.
    """
    meta   = json.loads((run_dir / "hlp_metadata.json").read_text(encoding="utf-8"))
    ranges = {p["name"]: (float(p["min"]), float(p["max"])) for p in meta["hlp_param_config"]}
    # | Sample entry of hlp_param_config: {'name': 'hlp_cognitive_cost_coef', 'min': 0.0, 'max': 0.15}
    lo, hi = ranges[coefficient]
    if hi <= lo:
        raise ValueError(f"{coefficient} was trained on the degenerate range [{lo}, {hi}]; nothing to sweep")
    return lo, hi


def sweep_values(lo: float, hi: float, points: int, power: float) -> list[float]:
    """
    Choose ``points`` values from ``lo`` to ``hi`` whose spacing grows with the value, so more
    of them fall where the behaviour still changes.

    The values are ``lo + (hi - lo) * u**power`` for ``u`` evenly spaced on ``[0, 1]``;
    ``power = 1`` is even spacing, larger powers crowd the low end of the range.

    :param lo: Lowest value (always included).
    :type lo: float
    :param hi: Highest value (always included).
    :type hi: float
    :param points: Number of values.
    :type points: int
    :param power: Spacing exponent, at least 1.
    :type power: float
    :return: The values in ascending order.
    :rtype: list[float]
    :raises ValueError: If ``power`` is below 1 or ``points`` is below 2.
    """
    if power < 1 or points < 2:
        raise ValueError(f"need power >= 1 and points >= 2, got power={power}, points={points}")
    unit = np.linspace(0.0, 1.0, points) ** power   # (points,) in [0, 1], crowded near 0 for power > 1
    return [float(v) for v in lo + (hi - lo) * unit]


def values_for_even_attention(pilot: dict[str, np.ndarray], points: int) -> list[float]:
    """
    Choose coefficient values whose attention read-outs are evenly spaced, by inverting a pilot
    sweep's coefficient -> cars-attended curve.

    Attention never rises with the cost, so the pilot curve is first forced monotone (running
    minimum over ascending coefficient values) and each attention level keeps only the lowest
    coefficient that reaches it; the targets are ``points`` evenly spaced attention levels
    between the curve's two ends, and each target's coefficient is read off by linear
    interpolation.

    :param pilot: Output of :func:`hlp_eval.sweep` on a coarse grid of the coefficient.
    :type pilot: dict[str, np.ndarray]
    :param points: Number of values wanted.
    :type points: int
    :return: Coefficient values in ascending order; the first and last are the pilot's ends.
    :rtype: list[float]
    :raises ValueError: If the pilot has fewer than two distinct attention levels.
    """
    values = pilot["values"]                            # (P,) ascending coefficient values
    shown  = np.minimum.accumulate(pilot[X_METRIC])     # (P,) cars attended, made non-increasing
    keep   = np.r_[True, shown[1:] < shown[:-1]]        # first value reaching each level
    values, shown = values[keep], shown[keep]           # shown now strictly decreasing
    if len(shown) < 2:
        raise ValueError("the pilot sweep shows no change in cars attended; nothing to space")
    targets = np.linspace(shown[-1], shown[0], points)  # (points,) ascending attention levels
    # | np.interp needs ascending x, so both pilot arrays are read back to front.
    return sorted(float(v) for v in np.interp(targets, shown[::-1], values[::-1]))


def write_sweep_csv(table: dict[str, np.ndarray], coefficient: str, n_sims: int, out_path: Path) -> None:
    """
    Write the sweep table as one CSV row per coefficient value.

    :param table: Output of :func:`hlp_eval.sweep`: ``values`` plus each metric's mean and ``_se``.
    :type table: dict[str, np.ndarray]
    :param coefficient: Name of the swept coefficient, used as the first column's header.
    :type coefficient: str
    :param n_sims: Number of paired maps behind every row, written as the ``n_sims`` column.
    :type n_sims: int
    :param out_path: Destination CSV path.
    :type out_path: Path
    """
    # | Sample row: {'hlp_cognitive_cost_coef': 0.05, 'n_sims': 40, 'shown': 1.8, 'shown_se': 0.1, ...}
    metric_keys = [k for k in table if k != "values"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([coefficient, N_SIMS_COLUMN, *metric_keys])
        for i, value in enumerate(table["values"]):
            writer.writerow([f"{value:.6g}", n_sims, *[f"{table[k][i]:.6g}" for k in metric_keys]])


def draw_figure(table: dict[str, np.ndarray], n_sims: int, out_stem: Path) -> None:
    """
    Draw the trade-off figure (crashes against cars attended, dots coloured by the coefficient)
    and save it as PNG and PDF.

    :param table: Output of :func:`hlp_eval.sweep`.
    :type table: dict[str, np.ndarray]
    :param n_sims: Number of paired maps behind the band, stated in the corner note.
    :type n_sims: int
    :param out_stem: Output path without extension; ``.png`` and ``.pdf`` are appended.
    :type out_stem: Path
    """
    values = table["values"]              # (V,) swept coefficient values, ascending
    x      = table[X_METRIC]              # (V,) cars attended per step
    y      = table[Y_METRIC]              # (V,) crashes per episode
    y_se   = table[Y_METRIC + "_se"]      # (V,)

    # | 1. Band and line along the sweep order (x decreases as the coefficient rises).
    plt.close("all")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.fill_between(x, y - y_se, y + y_se, color=BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(x, y, color=LINE_COLOR, linewidth=1.2, zorder=2)

    # | 2. One dot per swept value, coloured by that value; the colour bar names the coefficient.
    dots = ax.scatter(x, y, c=values, cmap=COLORMAP, s=DOT_SIZE, edgecolor="black", linewidth=0.6, zorder=3)
    colorbar = fig.colorbar(dots, ax=ax, pad=0.02)
    colorbar.set_label(COLORBAR_LABEL)

    # | 3. Chrome: x runs high to low so the curve climbs as attention drops; recessive grid.
    ax.invert_xaxis()
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.98, f"shaded band: ±1 s.e. ({n_sims} maps/pt)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#555555", style="italic")

    # | 4. Save both formats (PNG for quick viewing, PDF for the manuscript).
    for suffix in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=200)
    plt.close(fig)


# ===========================================================================
# 3. Entry point
# ===========================================================================

def main(argv: list[str] | None = None) -> None:
    """
    Run the pilot and the sweep, then write the tables and the figure.

    Stages: 1. parse arguments and resolve the sweep range; 2. the pilot sweep and the
    attention-spaced values; 3. the full sweep on paired maps; 4. the tables and the figure.

    :param argv: Command-line arguments; ``None`` reads ``sys.argv``.
    :type argv: list[str] | None
    """
    # | 1. Arguments. Local defaults are a smoke-sized sweep; the paper's figure is --sims 40.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=Path("human_modeling/inference/checkpoint"),
                    help="SSP checkpoint directory (hlp_metadata.json, hlp_model_final.pt, the paired SP weights and metadata)")
    ap.add_argument("--coefficient", default=DEFAULT_COEFFICIENT, choices=HLP_PARAM_NAMES,
                    help="which cognitive-profile coefficient to sweep (the paper varies the attentional cost)")
    ap.add_argument("--points", type=int, default=9,
                    help="number of swept values, at evenly spaced attention levels (the paper's figure has 9)")
    ap.add_argument("--sims", type=int, default=8,
                    help="paired maps (one episode each) per value; the paper used 40 (8 is the laptop default)")
    ap.add_argument("--pilot-points", type=int, default=13,
                    help="coefficient values in the pilot sweep that locates the attention levels")
    ap.add_argument("--pilot-sims", type=int, default=10,
                    help="paired maps per pilot value")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the policies' sampled decisions; the maps are seeded separately by hlp_eval.sweep")
    ap.add_argument("--out-dir", type=Path, default=Path("human_modeling/inference/results/cost_performance"),
                    help="where the CSVs and the figure are written")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # | hlp_eval seeds the maps but not the policies' sampled goals, masks and actions; without
    # | this two runs on the same maps differ (MEASURED: 0.72 vs 0.68 cars attended at zero cost).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    coefficient = args.coefficient
    lo, hi = trained_range(args.run_dir, coefficient)
    base = dict(GAME_THETA)

    # | 2. Pilot: a coarse, cheap sweep that locates the attention levels.
    pilot_values = sweep_values(lo, hi, args.pilot_points, PILOT_POWER)
    logger.info(f"pilot sweep of {coefficient} at {args.pilot_points} points, {args.pilot_sims} maps each")
    pilot = sweep(str(args.run_dir), coefficient, pilot_values, sims=args.pilot_sims, base_overrides=base)
    write_sweep_csv(pilot, coefficient, args.pilot_sims, args.out_dir / f"cost_performance_{coefficient}_pilot.csv")
    values = values_for_even_attention(pilot, args.points)
    logger.info(f"sweeping {coefficient} over [{lo:.4g}, {hi:.4g}] at {args.points} points: "
                f"{[round(v, 5) for v in values]}; {args.sims} paired maps each; background theta = {base}")

    # | 3. Rollouts: every value sees the same construction seeds 0..sims-1.
    table = sweep(str(args.run_dir), coefficient, values, sims=args.sims, base_overrides=base)

    # | 4. Table and figure.
    csv_path = args.out_dir / f"cost_performance_{coefficient}.csv"
    write_sweep_csv(table, coefficient, args.sims, csv_path)
    out_stem = args.out_dir / f"cost_performance_{coefficient}"
    draw_figure(table, args.sims, out_stem)
    logger.info(f"wrote {csv_path}, {out_stem}.png and {out_stem}.pdf")


if __name__ == "__main__":
    main()
