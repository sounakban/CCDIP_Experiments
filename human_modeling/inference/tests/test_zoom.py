"""
The zoomed lattice's summary, and the specific way it was previously wrong.

Every check here targets a defect that a review found in the first version of this code.
They are written so that the ORIGINAL broken behaviour fails them, because a test that
passes against both the bug and the fix proves nothing:

  edge mass     was measured against the lattice's own ends, so a posterior sitting dead
                centre of the allowed range on a zoomed lattice read as fully railed.

No checkpoint and no GPU.
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
from human_modeling.inference.posterior.update import posterior_weights, summarize

LO = np.array([0.0, 0.0, 0.0, 0.0])
HI = np.array([4.0, 0.15, 2.0, 0.3])


def _sharp(grid, centre, width):
    """A log-likelihood sharply peaked at ``centre``."""
    z = (grid.points - centre) / width
    return -0.5 * np.sum(z ** 2, axis=1)


class TestEdgeMassMeaning:
    """Edge mass must mean 'against a TRAINED bound', never 'against this lattice'."""

    def test_centred_posterior_on_a_zoom_is_NOT_railed(self):
        """The old behaviour reported 1.00 here, because the belief filled its own small
        box. Nothing about that says the coefficient is against a bound."""
        centre = np.array([2.0, 0.075, 1.0, 0.15])          # dead centre of every range
        zoom = build_zoom_grid(LO, HI, centre, 0.02 * (HI - LO), n_per_axis=8)
        s = summarize(zoom, _sharp(zoom, centre, 0.005 * (HI - LO)), bounds=(LO, HI))
        for name in THETA_NAMES:
            assert s[f"{name}_edge_mass"] < 1e-6, \
                f"{name}: a mid-range posterior was reported as railed"

    def test_posterior_at_a_bound_IS_railed(self):
        """And genuine railing must still be caught."""
        grid = build_grid(LO, HI, n_per_axis=16)
        at_top = HI * 0.995
        s = summarize(grid, _sharp(grid, at_top, 0.01 * (HI - LO)), bounds=(LO, HI))
        for name in THETA_NAMES:
            assert s[f"{name}_edge_mass"] > 0.5, f"{name}: railing was missed"

    def test_is_comparable_across_resolutions(self):
        """Being a fraction of the TRAINED range, it must not drift with lattice size --
        the old version halved every time the resolution doubled."""
        seen = []
        for n in (8, 16, 32):
            grid = build_grid(LO, HI, n_per_axis=n)
            s = summarize(grid, np.zeros(grid.n_points), bounds=(LO, HI))
            seen.append(s[f"{THETA_NAMES[0]}_edge_mass"])
        assert max(seen) - min(seen) < 0.05, \
            f"edge mass moved with resolution: {[round(x, 3) for x in seen]}"

    def test_without_bounds_it_refuses_to_guess(self):
        """No bounds means 'edge' is undefined; report that, never a plausible number."""
        grid = build_grid(LO, HI, n_per_axis=8)
        s = summarize(grid, np.zeros(grid.n_points))
        for name in THETA_NAMES:
            assert np.isnan(s[f"{name}_edge_mass"])


class TestNaNSafety:
    """A broken evaluation must fail loudly, not become a confident flat posterior."""

    def test_nan_raises(self):
        bad = np.array([1.0, np.nan, 3.0, 4.0])
        with pytest.raises(ValueError, match="NaN"):
            posterior_weights(bad)

    def test_all_impossible_is_still_flat(self):
        """All -inf is genuinely 'no information' and must NOT raise."""
        w = posterior_weights(np.full(8, -np.inf))
        assert np.allclose(w, 1.0 / 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
