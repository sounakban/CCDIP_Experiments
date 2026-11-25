"""
Checkpoint-free properties of the V4 estimator's ascent and curvature box.

WHAT A FAKE CONTEXT IS
    `fit_mode` and `hessian_diag` only ever call ``ctx.logL(theta)``. A quadratic
    log-likelihood with a known peak and known curvature stands in for the real episode, so
    the mode it finds and the curvature it measures both have exact answers to check against.
"""

import unittest

import numpy as np
import torch

from human_modeling.inference.model.choice import THETA_NAMES
from human_modeling.inference.fit_gradient import (curvature_half_widths, fit_mode,
                                                 hessian_diag)

LO = np.zeros(4)
HI = np.array([10.0, 0.15, 2.0, 0.5])
SPAN = HI - LO


class Quadratic:
    """log L(theta) = -0.5 * sum_i k_i * ((theta_i - c_i) / span_i)^2, in normalized units."""

    def __init__(self, centre_z: np.ndarray, curvature: np.ndarray) -> None:
        self.c = torch.tensor(LO + centre_z * SPAN, dtype=torch.float64)
        self.k = torch.tensor(curvature, dtype=torch.float64)
        self.span = torch.tensor(SPAN, dtype=torch.float64)

    def logL(self, theta: torch.Tensor) -> torch.Tensor:
        """The quadratic, differentiable in theta."""
        z = (theta - self.c) / self.span
        return -0.5 * (self.k * z * z).sum()


class Ascent(unittest.TestCase):
    """The ascent must find a known peak."""

    def test_finds_the_peak(self):
        a = Quadratic(np.full(4, 0.4), np.full(4, 200.0))
        mode, info = fit_mode(a, LO, HI, max_iters=400, patience=50, tol=1e-6)
        np.testing.assert_allclose((mode - LO) / SPAN, 0.4, atol=0.02)
        self.assertGreater(info["iters"], 1)


class Curvature(unittest.TestCase):
    """The finite-difference Hessian diagonal and the box it sizes."""

    def test_hessian_diag_recovers_the_known_curvature(self):
        k = np.array([50.0, 120.0, 8.0, 300.0])
        ctx = Quadratic(np.full(4, 0.5), k)
        h = hessian_diag(ctx, LO + 0.5 * SPAN, LO, HI)
        np.testing.assert_allclose(h, -k, rtol=1e-6)

    def test_hessian_diag_works_at_a_bound(self):
        """The stencil must stay inside [0, 1]; a mode ON the bound gets a one-sided one."""
        k = np.array([50.0, 120.0, 8.0, 300.0])
        ctx = Quadratic(np.full(4, 0.0), k)
        h = hessian_diag(ctx, LO.copy(), LO, HI)
        np.testing.assert_allclose(h, -k, rtol=1e-6)

    def test_box_from_curvature(self):
        """half = n_sd / sqrt(-H) in normalized units, clipped, times the span."""
        h = np.array([-400.0, -4.0, 0.0, 5.0])          # sharp, broad, flat, NOT a peak
        half = curvature_half_widths(h, SPAN, n_sd=3.0, half_min=0.05, half_max=0.5)
        z = half / SPAN
        self.assertAlmostEqual(z[0], 3.0 / 20.0)        # 0.15, inside the clip range
        self.assertAlmostEqual(z[1], 0.5)               # 3/2 = 1.5 -> clipped to half_max
        self.assertAlmostEqual(z[2], 0.5)               # flat -> half_max
        self.assertAlmostEqual(z[3], 0.5)               # positive curvature -> half_max

    def test_sharper_curvature_shrinks_the_box(self):
        """KNOWN-BAD direction: curvature x2 -> box / sqrt(2)."""
        k = np.array([50.0, 120.0, 8.0, 300.0])
        one = curvature_half_widths(hessian_diag(Quadratic(np.full(4, 0.5), k), LO + 0.5 * SPAN, LO, HI),
                                    SPAN, n_sd=1.0, half_min=0.0, half_max=10.0)
        two = curvature_half_widths(hessian_diag(Quadratic(np.full(4, 0.5), 2 * k), LO + 0.5 * SPAN, LO, HI),
                                    SPAN, n_sd=1.0, half_min=0.0, half_max=10.0)
        np.testing.assert_allclose(two, one / np.sqrt(2), rtol=1e-6)
        self.assertFalse(np.allclose(two, one))


class PosteriorBox(unittest.TestCase):
    """`fit_posterior` evaluates exactly one box, at the half-width it is given."""

    def test_one_box_at_the_given_half_width(self):
        """Checked through fakes rather than by comparing summaries, because the summary needs
        a checkpoint; the box COUNT and its size are the behavioural claims that matter."""
        from human_modeling.inference import fit_gradient as fg
        seen = {"grids": 0}

        class FakeGrid:
            points = np.zeros((1, 4))

        def fake_zoom(lo, hi, centre, half, n):
            seen["grids"] += 1
            seen["half"] = np.asarray(half).copy()
            return FakeGrid()

        orig = (fg.build_zoom_grid, fg.evaluate_episode, fg.summarize)
        try:
            fg.build_zoom_grid = fake_zoom
            fg.evaluate_episode = lambda stack, inputs, grid, embed=None: np.zeros(1)
            fg.summarize = lambda g, v, bounds=None, names=THETA_NAMES: {f"{n}_mean": 0.0 for n in names}
            est = fg.fit_posterior(None, None, np.zeros(4), LO, HI, half=0.1 * SPAN)
            self.assertEqual(seen["grids"], 1)
            np.testing.assert_allclose(seen["half"], 0.1 * SPAN)
            self.assertEqual(set(est), {f"{n}_mean" for n in THETA_NAMES})
        finally:
            fg.build_zoom_grid, fg.evaluate_episode, fg.summarize = orig


if __name__ == "__main__":
    unittest.main(verbosity=2)
