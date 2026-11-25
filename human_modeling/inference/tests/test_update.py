"""
Correctness gates for the lattice posterior summaries, on synthetic log-likelihoods.

These run in milliseconds and need no checkpoint: the summaries are pure arithmetic over
stored arrays, so they can be tested apart from the expensive likelihood. (Ported from the
retired hlp_v3 alongside the module it covers; see version_history.md.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# | Run from the repo root (python -m pytest) or directly; both must work.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from human_modeling.inference.model.choice import THETA_NAMES
from human_modeling.inference.posterior.grid import build_grid
from human_modeling.inference.posterior.update import posterior_weights, summarize

N_AXIS = 5
N_EPISODES = 6


def _fake_logliks(seed: int = 11, n_episodes: int = N_EPISODES,
                  n_points: int = N_AXIS ** 4) -> list[np.ndarray]:
    """Stand-in per-episode log-likelihoods with realistic magnitudes."""
    rng = np.random.default_rng(seed)
    return [rng.normal(-1800.0, 40.0, size=n_points) for _ in range(n_episodes)]


class TestSummaries:
    """Weighted summaries over the lattice must be correct and shape-agnostic."""

    def _grid(self):
        lo = np.array([0.0, 0.0, 0.0, 0.0])
        hi = np.array([4.0, 0.15, 2.0, 0.3])
        return build_grid(lo, hi, n_per_axis=N_AXIS)

    def test_weights_sum_to_one(self):
        """Normalisation must produce a probability distribution."""
        w = posterior_weights(_fake_logliks(n_points=N_AXIS ** 4)[0])
        assert abs(float(w.sum()) - 1.0) < 1e-12

    def test_flat_posterior_gives_prior_mean(self):
        """With no information the posterior mean is the middle of each range."""
        grid = self._grid()
        flat = np.zeros(grid.n_points)
        s = summarize(grid, flat)
        for d, name in enumerate(THETA_NAMES):
            expect = float(np.mean(grid.axes[d]))
            assert abs(s[f"{name}_mean"] - expect) < 1e-9, f"{name} mean drifted off centre"
            assert s[f"{name}_sd"] > 0.0, f"{name} reported zero spread under a flat posterior"

    def test_spike_recovers_its_own_point(self):
        """A posterior concentrated on one point must report that point, with ~no spread."""
        grid = self._grid()
        log_post = np.full(grid.n_points, -1e6)
        idx = 7 * (grid.n_points // 13)
        log_post[idx] = 0.0
        s = summarize(grid, log_post)
        for d, name in enumerate(THETA_NAMES):
            assert abs(s[f"{name}_mean"] - grid.points[idx, d]) < 1e-6
            assert s[f"{name}_sd"] < 1e-6

    def test_edge_mass_detects_railing(self):
        """Mass pushed to the top of the TRAINED range must show up as edge mass, which
        is how a posterior reports that the data wants to leave that range."""
        lo = np.array([0.0, 0.0, 0.0, 0.0])
        hi = np.array([4.0, 0.15, 2.0, 0.3])
        # | 32 points per axis so the top cell sits inside the outer slice; at a coarse
        # | resolution the top cell straddles it and contributes only partly, which is
        # | correct behaviour but makes a threshold test ambiguous.
        grid = build_grid(lo, hi, n_per_axis=32)
        w = np.zeros(grid.shape)
        w[-1, :, :, :] = 1.0                       # all mass at the top of coefficient 0
        log_post = np.log(np.maximum(w.reshape(-1), 1e-300))
        s = summarize(grid, log_post, bounds=(lo, hi))
        assert s[f"{THETA_NAMES[0]}_edge_mass"] > 0.99, "railing was not detected"
        assert s[f"{THETA_NAMES[1]}_edge_mass"] < 0.5, "edge mass fired on the wrong axis"

    def test_all_impossible_gives_flat_not_nan(self):
        """An all-impossible posterior must degrade to flat, never to NaN."""
        grid = self._grid()
        w = posterior_weights(np.full(grid.n_points, -np.inf))
        assert np.all(np.isfinite(w)) and abs(float(w.sum()) - 1.0) < 1e-12

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
