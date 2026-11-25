"""
Checks on the calibration gate's OWN measurement, before it is trusted to judge anything.

WHY THIS EXISTS
    The gate decides whether the reported intervals mean what they claim, and it is the
    only check in the battery that can catch an overconfident method. That makes its own
    arithmetic load-bearing: if the measurement is biased, the gate fails honest work and
    passes dishonest work, and nothing downstream would reveal it.

    It was in fact biased. `rank_of_truth` compared the true value against the lattice
    POINTS, which treats each point as holding all its probability exactly at the point.
    A true value inside the outermost cell but on the far side of that cell's midpoint
    then scored exactly 0 or exactly 1. At the production setting of 6 points per
    coefficient the outer half cells are one sixth of the box, so one draw in six was
    forced to an extreme -- and ranks piled at both ends is the signature the gate reads
    as OVERCONFIDENT. The gate would have failed on its own artefact.

    MEASURED on a conjugate Gaussian whose posterior is exact, 4,000 draws x 4
    coefficients:

        measurement    at rank 0.0   at rank 1.0   mean    distance from uniform
        lattice points     8.2%          8.4%      0.502          0.1453
        cell edges         0.0%          0.0%      0.499          0.0361

    Both columns describe the SAME exactly-correct posterior. The first is the defect.

WHAT THE STAND-IN IS
    A noisy observation of the truth with known spread, so the posterior on the lattice
    is the true posterior rather than an approximation of one. That is what makes "ranks
    must be uniform" a statement with a right answer instead of a hope.

Numbered stages:
    1. The old measurement piles ranks at the ends on a correct posterior (it can fail).
    2. The new measurement does not.
    3. Its residual error is discretization -- it shrinks as the lattice refines.
    4. Interval coverage comes out at its nominal rate.
"""

import unittest

import numpy as np

from human_modeling.inference.posterior.grid import build_grid
from human_modeling.inference.posterior.update import posterior_weights

# ===========================================================================
# Module-level constants
# ===========================================================================

# | A plain unit box; nothing here depends on the real trained ranges.
LO = np.zeros(4)
HI = np.ones(4)

# | Spread of the stand-in observation, as a fraction of a coefficient's range. Chosen
# | so the posterior is narrower than a coarse cell, matching the real situation that
# | forced the zoom, without being so narrow that it falls between lattice points.
OBS_SD = 0.08

# | Draws per study. 4,000 x 4 coefficients puts the uncertainty on a percentage at
# | roughly 0.4 points, small beside the ~8-point effect being measured.
N_DRAWS = 4000


def rank_by_points(grid, log_lik: np.ndarray, truth: np.ndarray) -> list[float]:
    """
    The ORIGINAL, defective measurement, kept so the defect stays reproducible.

    Sums the probability of every lattice point lying strictly below the truth, which
    silently assumes each point carries its mass at a single location.

    :param grid: The lattice.
    :type grid: ParameterGrid
    :param log_lik: Log-likelihood at every lattice point.
    :type log_lik: np.ndarray
    :param truth: The value the data was generated from.
    :type truth: np.ndarray
    :return: Rank per coefficient.
    :rtype: list[float]
    """
    w = posterior_weights(log_lik).reshape(grid.shape)
    out = []
    for d in range(len(grid.shape)):
        other = tuple(i for i in range(len(grid.shape)) if i != d)
        p = w.sum(axis=other); p = p / p.sum()
        out.append(float(np.sum(p[grid.axes[d] < truth[d]])))
    return out


def rank_by_edges(grid, log_lik: np.ndarray, truth: np.ndarray) -> list[float]:
    """
    The corrected measurement: each point stands for its cell.

    Builds the running total at cell edges and reads it off by straight-line
    interpolation inside the cell holding the truth. Mirrors `run_calibration.rank_of_truth`.

    :param grid: The lattice.
    :type grid: ParameterGrid
    :param log_lik: Log-likelihood at every lattice point.
    :type log_lik: np.ndarray
    :param truth: The value the data was generated from.
    :type truth: np.ndarray
    :return: Rank per coefficient.
    :rtype: list[float]
    """
    w    = posterior_weights(log_lik).reshape(grid.shape)
    half = 0.5 * grid.cell_widths()
    out  = []
    for d in range(len(grid.shape)):
        other = tuple(i for i in range(len(grid.shape)) if i != d)
        p = w.sum(axis=other); p = p / p.sum()
        edges = np.concatenate([[grid.axes[d][0] - half[d]], grid.axes[d] + half[d]])
        cum   = np.concatenate([[0.0], np.cumsum(p)])
        out.append(float(np.clip(np.interp(truth[d], edges, cum), 0.0, 1.0)))
    return out


def run_study(n_per_axis: int, measure, n_draws: int = N_DRAWS,
              seed: int = 7) -> np.ndarray:
    """
    Simulation-based calibration against a posterior that is known to be exact.

    Draws a truth from the uniform prior, observes it with known noise, and forms the
    posterior on the lattice -- which for this model IS the posterior, not an estimate of
    one. Honest ranks are therefore uniform by construction (Talts et al. 2018).

    :param n_per_axis: Lattice points per coefficient.
    :type n_per_axis: int
    :param measure: Rank function under test.
    :type measure: Callable
    :param n_draws: Draws to run.
    :type n_draws: int
    :param seed: Random seed; must not be 0.
    :type seed: int
    :return: Every rank produced, flattened across coefficients.
    :rtype: np.ndarray
    """
    grid = build_grid(LO, HI, n_per_axis)
    rng  = np.random.default_rng(seed)
    out: list[float] = []
    for _ in range(n_draws):
        truth = rng.uniform(LO, HI)                          # theta ~ prior
        obs   = truth + rng.normal(0.0, OBS_SD, len(LO))     # data  ~ p(y | theta)
        d2    = ((grid.points - obs) / OBS_SD) ** 2          # (n_points, 4)
        out  += measure(grid, -0.5 * d2.sum(axis=1), truth)
    return np.asarray(out)


def distance_from_uniform(ranks: np.ndarray) -> float:
    """Largest gap between the observed rank spread and a perfectly even one."""
    return float(np.max(np.abs(np.sort(ranks) - np.linspace(0.0, 1.0, len(ranks)))))


def share_at_ends(ranks: np.ndarray) -> float:
    """Share of ranks sitting exactly at 0 or exactly at 1."""
    return float(np.mean((ranks <= 1e-9) | (ranks >= 1.0 - 1e-9)))


class CalibrationMeasurement(unittest.TestCase):
    """The gate's arithmetic, checked against a posterior with a known right answer."""

    # ----- 1. The defect is real and reproducible -----

    def test_the_old_measurement_fails_on_a_correct_posterior(self):
        """
        Establishes that this check can fail -- on input known to be good.

        Without this the corrected result below would be unfalsifiable: a measurement
        that has never reported a problem tells you nothing about the thing it measures.
        """
        ranks = run_study(6, rank_by_points)
        self.assertGreater(share_at_ends(ranks), 0.10,
                           "the known defect did not reproduce; this test proves nothing")
        self.assertGreater(distance_from_uniform(ranks), 0.10)

    # ----- 2. The correction works -----

    def test_the_corrected_measurement_is_uniform_on_a_correct_posterior(self):
        ranks = run_study(6, rank_by_edges)
        self.assertLess(share_at_ends(ranks), 0.005,
                        "ranks still pile at the extremes")
        self.assertLess(distance_from_uniform(ranks), 0.06)
        self.assertAlmostEqual(float(np.mean(ranks)), 0.5, delta=0.02)

    # ----- 3. The residual is resolution, not bias -----

    def test_residual_error_shrinks_as_the_lattice_refines(self):
        """
        Separates a coarse lattice (harmless, buy accuracy with points) from a biased
        measurement (fatal, no amount of spending helps).
        """
        coarse = distance_from_uniform(run_study(6,  rank_by_edges, n_draws=1500))
        fine   = distance_from_uniform(run_study(12, rank_by_edges, n_draws=1500))
        self.assertLess(fine, coarse,
                        f"error did not fall with resolution: {coarse:.4f} -> {fine:.4f}")

    # ----- 4. Coverage lands at its nominal rate -----

    def test_ninety_percent_intervals_contain_the_truth_about_ninety_percent(self):
        """
        The claim a credible interval actually makes, checked directly.

        Read off the same running total as the rank, so an interval end may sit anywhere
        inside a cell instead of snapping to the nearest lattice point.
        """
        grid = build_grid(LO, HI, 6)
        half = 0.5 * grid.cell_widths()
        rng  = np.random.default_rng(11)
        hits = []
        for _ in range(N_DRAWS // 2):
            truth = rng.uniform(LO, HI)
            obs   = truth + rng.normal(0.0, OBS_SD, len(LO))
            d2    = ((grid.points - obs) / OBS_SD) ** 2
            w     = posterior_weights(-0.5 * d2.sum(axis=1)).reshape(grid.shape)
            for d in range(len(LO)):
                other = tuple(i for i in range(len(LO)) if i != d)
                p = w.sum(axis=other); p = p / p.sum()
                edges = np.concatenate([[grid.axes[d][0] - half[d]], grid.axes[d] + half[d]])
                cum   = np.concatenate([[0.0], np.cumsum(p)])
                lo90, hi90 = np.interp([0.05, 0.95], cum, edges)
                hits.append(float(lo90 <= truth[d] <= hi90))
        got = float(np.mean(hits))
        self.assertGreater(got, 0.86, f"90% interval covered only {got:.1%}")
        self.assertLess(got, 0.94, f"90% interval covered {got:.1%} -- too wide to be useful")


if __name__ == "__main__":
    unittest.main(verbosity=2)
