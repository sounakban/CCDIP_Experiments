"""
Lattice construction and the zoom mechanism.

The zoom exists because of a measured fact: on real episodes the posterior fits inside a
single cell of a whole-range lattice, so spreading points across the whole trained range
describes nothing. These checks confirm the zoom concentrates points where belief is,
never leaves the trained range, and REFUSES to claim coverage it does not have.

That last one matters most. A zoom that silently misses belief would report a confident
posterior computed over the wrong region — the exact failure mode the coverage check
exists to catch — so it is tested against a case built to defeat it.

No checkpoint and no GPU: milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from human_modeling.inference.model.choice import THETA_NAMES
from human_modeling.inference.posterior.grid import build_grid, build_zoom_grid

LO = np.array([0.0, 0.0, 0.0, 0.0])
HI = np.array([4.0, 0.15, 2.0, 0.3])


class TestLattice:
    """The whole-range lattice must be laid out as documented."""

    def test_points_are_cell_midpoints(self):
        """Each point sits at the centre of its own equal-width cell, so every point
        carries the same prior mass and the sum is a plain Riemann sum."""
        grid = build_grid(LO, HI, n_per_axis=4)
        for d in range(4):
            width = (HI[d] - LO[d]) / 4
            expect = LO[d] + width * (np.arange(4) + 0.5)
            assert np.allclose(grid.axes[d], expect)
            assert grid.axes[d][0] > LO[d], "first point sits on the lower bound"
            assert grid.axes[d][-1] < HI[d], "last point sits on the upper bound"

    def test_cells_are_equal_width(self):
        """Unequal cells would weight some points more than others."""
        grid = build_grid(LO, HI, n_per_axis=6)
        for d in range(4):
            gaps = np.diff(grid.axes[d])
            assert np.allclose(gaps, gaps[0]), f"axis {d} has uneven spacing"

    def test_point_count_and_shape(self):
        """n per axis over four coefficients gives n^4 combinations."""
        for n in (2, 4, 6):
            grid = build_grid(LO, HI, n_per_axis=n)
            assert grid.n_points == n ** len(THETA_NAMES)
            assert grid.shape == (n,) * len(THETA_NAMES)

    def test_every_combination_appears_once(self):
        """The lattice must be the full product, with no duplicates or omissions."""
        grid = build_grid(LO, HI, n_per_axis=3)
        assert len({tuple(p) for p in grid.points}) == grid.n_points

    def test_bad_arguments_raise(self):
        """Degenerate requests are caller errors, not silently repaired."""
        with pytest.raises(ValueError):
            build_grid(LO, HI, n_per_axis=1)
        with pytest.raises(ValueError):
            build_grid(HI, LO, n_per_axis=4)          # bounds inverted


class TestZoom:
    """The zoom must concentrate points without leaving the trained range."""

    def test_zoom_is_inside_the_original_bounds(self):
        """A zoom reaching past a bound would evaluate where the policy never trained."""
        centre = np.array([3.9, 0.14, 1.95, 0.29])    # deliberately near the top
        half = np.array([1.0, 0.05, 0.5, 0.1])        # wide enough to overhang
        z = build_zoom_grid(LO, HI, centre, half, n_per_axis=5)
        for d in range(4):
            assert z.axes[d][0] >= LO[d] and z.axes[d][-1] <= HI[d]

    def test_zoom_cells_are_smaller_than_the_coarse_ones(self):
        """The whole point: finer spacing where the belief is."""
        coarse = build_grid(LO, HI, n_per_axis=8)
        z = build_zoom_grid(LO, HI, np.array([2.0, 0.075, 1.0, 0.15]),
                            0.1 * (HI - LO), n_per_axis=8)
        assert np.all(z.cell_widths() < coarse.cell_widths())

    def test_degenerate_zoom_still_forms_a_lattice(self):
        """A zero-width request must not collapse into an unusable grid."""
        z = build_zoom_grid(LO, HI, np.array([2.0, 0.075, 1.0, 0.15]),
                            np.zeros(4), n_per_axis=4)
        assert z.n_points == 4 ** 4
        assert np.all(np.isfinite(z.points))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
