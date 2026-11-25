"""
Gates on the full-visibility model (implementation.md Entry 36).

The full-visibility model, `C_full_visibility`, is the no-cognitive-cost model with every car
slot the network observes forced visible, so it makes no attention choice. Its likelihood
runs through the same two paths as every other model -- the numpy one the posterior lattice
uses and the differentiable torch one the gradient ascent uses -- with ONE construal column
per step instead of one per attention pattern.

What is checked, each with a known-bad arm so a blind check cannot pass:
    1. The definition: same free coefficients and pin as C, outside the nested chain, and no
       heuristic weights allowed.
    2. A context prepared for one visibility mode refuses a variant of the other.
    3. Every table has one construal column (the ordinary tables have more).
    4. The torch and numpy likelihoods agree to `EQUALITY_BAR` on a real episode; the same
       comparison fails on a table shifted by a known amount.
    5. The full-visibility likelihood differs from the ordinary one at the same coefficients,
       so the switch is actually read.
    6. The autograd gradient matches central differences on the three free axes.
    7. A whole fit runs, moves off its start, and refuses the wrong prepared episode.

The checkpoint-backed gates skip when the human data is absent, as the rest of the suite does.

Run with ``python -m pytest human_modeling/inference/tests/test_full_visibility.py``.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from human_modeling.inference.inputs import load_pilot_episodes
from human_modeling.inference.likelihood_grad import EpisodeGradContext, log_rho_torch
from human_modeling.inference.model.choice import HEURISTIC_NAMES, THETA_NAMES
from human_modeling.inference.posterior.likelihood import (load_models, log_likelihood,
                                                              prepare_episode)
from human_modeling.inference.variants import (ATTENTION_CHOOSING_VARIANTS, FIT_ORDER,
                                                  NESTED_PAIRS, VARIANTS, FitVariant,
                                                  VariantGradContext, get_variant)

PROJECT_ROOT   = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "human_modeling/inference/checkpoint"
DATA_PATH      = PROJECT_ROOT / "human_modeling/data/exp1_filtered.json"

# | Same truncation as test_heuristic_fitting.py: short enough to keep the suite quick, long
# | enough to span several attention-pattern widths in the ORDINARY tables, which the
# | one-column check contrasts against.
N_STEPS = 120

# | The three free coefficients (collision penalty, goal proximity cost, goal-flag distance
# | penalty), inside their trained ranges; the same theta the other gate files use.
FREE_THETA = np.array([6.94, 3.49, 0.30])

# | MEASURED 0.000e+00 on this fixture and on the whole first episode, local CPU, 2026-11-10
# | (implementation.md Entry 36). The bar is the one the heuristic-path equality gate uses
# | (test_heuristic_fitting.EQUALITY_BAR), loose enough not to fail on another GPU's float32
# | rounding -- this suite gates the cluster fit -- and still far below a porting mistake.
EQUALITY_BAR = 1e-6

# | A constant added to every entry of one step's emission table multiplies every path's
# | probability by exp(constant), so the log-likelihood moves by exactly that constant.
TABLE_SHIFT_NATS = 0.5

# | The full-visibility and ordinary likelihoods must differ by more than this at the same
# | coefficients, or the switch is not being read.
MIN_MODEL_GAP_NATS = 1e-3

# | Central-difference steps, as in test_heuristic_fitting.py, and a mixed tolerance set from
# | a MEASUREMENT on this fixture (2026-11-10, local CPU). Theta enters the network in
# | float32, so a 1e-7 nudge of a coefficient leaves the likelihood exactly unchanged, and the
# | best central difference misses a CORRECT gradient by 0.06-0.18% of its size -- on the
# | ordinary model too, whose gradient path is the established one. The 0.1% relative bar the
# | heuristic-weight gate uses (the weights never pass through the network) failed that
# | known-good path, so the bar here is 0.5%, and the ordinary model is checked beside the
# | full-visibility one as the known-good arm.
FD_STEPS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4)
GRAD_ATOL, GRAD_RTOL = 1e-3, 5e-3


def _value(ctx, vec: np.ndarray) -> float:
    """Log-likelihood as a plain float, with the autograd graph dropped."""
    return float(ctx.logL(torch.tensor(vec, dtype=torch.float64)).detach())




# ===========================================================================
# 1. Checkpoint-free
# ===========================================================================

class TestDefinition(unittest.TestCase):
    """The full-visibility model is C with every car visible, and sits outside the chain."""

    def test_it_is_c_with_every_car_visible(self):
        full, c = get_variant("C_full_visibility"), get_variant("C_no_cognitive_cost")
        self.assertTrue(full.full_visibility)
        self.assertFalse(c.full_visibility)
        self.assertEqual(full.free_names, c.free_names)
        self.assertEqual(full.pinned_theta, c.pinned_theta)
        self.assertEqual(full.n_free, 3)
        self.assertFalse(full.uses_heuristics)
        self.assertEqual(full.display, "full-visibility")
        # | Known-bad direction: no chained variant is marked full-visibility.
        self.assertEqual([v.name for v in VARIANTS.values() if v.full_visibility],
                         ["C_full_visibility"])

    def test_it_is_outside_the_nested_chain(self):
        full = get_variant("C_full_visibility")
        self.assertNotIn(full.name, {n for pair in NESTED_PAIRS for n in pair})
        self.assertIsNone(full.warm_start_from)
        self.assertEqual(full.extra_starts, ())
        for v in VARIANTS.values():
            self.assertNotEqual(v.warm_start_from, full.name)
            self.assertNotIn(full.name, v.extra_starts)
        self.assertEqual(FIT_ORDER[-1], full.name)

    def test_heuristic_weights_are_refused_in_the_definition(self):
        with self.assertRaises(ValueError):
            FitVariant("bad", "bad", free_theta=THETA_NAMES, free_heuristics=HEURISTIC_NAMES,
                       full_visibility=True)
        FitVariant("ok", "ok", free_theta=THETA_NAMES, full_visibility=True)   # no raise

    def test_the_attention_choosing_list(self):
        # | The list the ordinary-only scripts offer (independent review, 2026-11-10).
        self.assertEqual(ATTENTION_CHOOSING_VARIANTS, FIT_ORDER[:-1])




class TestComparisonScriptsKeepItOut(unittest.TestCase):
    """The nested-comparison scripts must not pick the sanity model up by default."""

    def test_default_discovery_skips_it_and_the_pair_tests_refuse_it(self):
        import tempfile
        from human_modeling.inference.scripts.participant_bic_tests import run_tests
        from human_modeling.inference.scripts.plot_model_comparison import available_variants
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("C_no_cognitive_cost", "C_full_visibility"):
                (Path(tmp) / f"fit_variant_{name}.csv").write_text("x\n", encoding="utf-8")
            self.assertEqual(available_variants(Path(tmp)), ["C_no_cognitive_cost"])
            with self.assertRaises(ValueError):
                run_tests(Path(tmp), ["C_no_cognitive_cost", "C_full_visibility"])




class TestContextRefusesTheOtherMode(unittest.TestCase):
    """A variant fitted on the wrong prepared episode would fit the other model silently."""

    @staticmethod
    def _ctx(full_visibility: bool) -> SimpleNamespace:
        """A stand-in context whose episode carries the given visibility flag."""
        return SimpleNamespace(features=None,
                               inputs=SimpleNamespace(full_visibility=full_visibility))

    def test_mismatches_are_refused(self):
        with self.assertRaises(ValueError):
            VariantGradContext(self._ctx(False), get_variant("C_full_visibility"))
        with self.assertRaises(ValueError):
            VariantGradContext(self._ctx(True), get_variant("C_no_cognitive_cost"))

    def test_matching_modes_are_accepted(self):
        # | Known-bad direction for the test above: a check refusing everything would pass it.
        VariantGradContext(self._ctx(True), get_variant("C_full_visibility"))
        VariantGradContext(self._ctx(False), get_variant("C_no_cognitive_cost"))
        VariantGradContext(SimpleNamespace(features=None), get_variant("B_no_heuristics"))




# ===========================================================================
# 2. Checkpoint-backed
# ===========================================================================

@unittest.skipUnless(CHECKPOINT_DIR.exists() and DATA_PATH.exists(),
                     "needs the packaged checkpoint and the human data")
class TestFullVisibilityLikelihood(unittest.TestCase):
    """Both likelihood paths on one real, truncated episode, prepared in both modes."""

    @classmethod
    def setUpClass(cls):
        cls.stack = load_models(str(CHECKPOINT_DIR))
        episode = load_pilot_episodes(DATA_PATH)[0]
        episode.steps = episode.steps[:N_STEPS]
        episode.actions = episode.actions[:N_STEPS]
        if episode.step_indices is not None:
            episode.step_indices = episode.step_indices[:N_STEPS]
        cls.inputs_ordinary = prepare_episode(cls.stack, episode, label="ordinary")
        cls.inputs_full = prepare_episode(cls.stack, episode, label="full",
                                          full_visibility=True)
        cls.variant = get_variant("C_full_visibility")
        cls.ctx = VariantGradContext(EpisodeGradContext(cls.stack, cls.inputs_full),
                                     cls.variant)
        lo, hi = cls.variant.lo_hi(cls.stack)
        cls.midpoint = 0.5 * (lo + hi)                                      # (3,)

    def test_every_table_has_one_construal_column(self):
        g = self.inputs_full.kernel.G
        self.assertTrue(all(t.shape == (g, 1) for t in self.inputs_full.tables))
        theta4 = torch.tensor(self.variant.embed(FREE_THETA), dtype=torch.float64)
        self.assertTrue(all(tuple(t.shape) == (g, 1)
                            for t in log_rho_torch(self.stack, self.inputs_full, theta4)))
        # | Known-bad direction: the ordinary tables of the same episode are wider somewhere.
        self.assertGreater(max(t.shape[1] for t in self.inputs_ordinary.tables), 1)

    def test_torch_equals_numpy(self):
        for free in (FREE_THETA, self.midpoint):
            torch_value = _value(self.ctx, free)
            numpy_value = log_likelihood(self.stack, self.inputs_full, self.variant.embed(free))
            self.assertLess(abs(torch_value - numpy_value), EQUALITY_BAR,
                            f"torch {torch_value!r} vs numpy {numpy_value!r}")

    def test_the_equality_check_can_fail(self):
        # | Known-bad: one step's emission table shifted by a known constant. The numpy value
        # | must move by exactly that constant, and so fail the bar the test above passes.
        tables = list(self.inputs_full.tables)
        tables[N_STEPS // 2] = tables[N_STEPS // 2] + TABLE_SHIFT_NATS
        shifted = replace(self.inputs_full, tables=tables)
        full4 = self.variant.embed(FREE_THETA)
        moved = log_likelihood(self.stack, shifted, full4) - log_likelihood(
            self.stack, self.inputs_full, full4)
        self.assertAlmostEqual(moved, TABLE_SHIFT_NATS, places=9)
        self.assertGreater(abs(_value(self.ctx, FREE_THETA)
                               - log_likelihood(self.stack, shifted, full4)), EQUALITY_BAR)

    def test_full_visibility_is_a_different_model(self):
        full4 = self.variant.embed(FREE_THETA)
        gap = abs(log_likelihood(self.stack, self.inputs_full, full4)
                  - log_likelihood(self.stack, self.inputs_ordinary, full4))
        self.assertGreater(gap, MIN_MODEL_GAP_NATS)

    def test_weights_are_refused(self):
        eight = np.concatenate([self.variant.embed(FREE_THETA), np.full(4, 0.5)])
        with self.assertRaises(ValueError):
            log_likelihood(self.stack, self.inputs_full, eight)
        ctx8 = EpisodeGradContext(self.stack, self.inputs_full, with_heuristics=True)
        with self.assertRaises(ValueError):
            ctx8.logL(torch.tensor(eight, dtype=torch.float64))

    def test_gradient_matches_central_differences(self):
        def central(ctx, point: np.ndarray, i: int, h: float) -> float:
            """Central difference of the log-likelihood in free coordinate ``i``."""
            up, down = point.copy(), point.copy()
            up[i] += h
            down[i] -= h
            return (_value(ctx, up) - _value(ctx, down)) / (2.0 * h)

        # | The full-visibility model, then the ordinary no-cognitive-cost model as the
        # | known-good arm: the bar must pass a gradient path already established as correct.
        ordinary = VariantGradContext(EpisodeGradContext(self.stack, self.inputs_ordinary),
                                      get_variant("C_no_cognitive_cost"))
        for label, ctx in (("full-visibility", self.ctx), ("ordinary", ordinary)):
            leaf = torch.tensor(FREE_THETA, dtype=torch.float64, requires_grad=True)
            ctx.logL(leaf).backward()
            grad = leaf.grad.detach().cpu().numpy().copy()                 # (3,)
            for i, name in enumerate(self.variant.free_names):
                best = min(abs(grad[i] - central(ctx, FREE_THETA, i, h)) for h in FD_STEPS)
                self.assertLess(best, GRAD_ATOL + GRAD_RTOL * abs(grad[i]),
                                f"{label} {name}: analytic {grad[i]}, closest difference off "
                                f"by {best}")
        # | Known-bad: the full-visibility gradient against a difference taken somewhere else
        # | (the box midpoint) must fail the same bar.
        leaf = torch.tensor(FREE_THETA, dtype=torch.float64, requires_grad=True)
        self.ctx.logL(leaf).backward()
        grad = leaf.grad.detach().cpu().numpy()                            # (3,)
        i = int(np.argmax(np.abs(grad)))
        self.assertGreater(abs(grad[i] - central(self.ctx, self.midpoint, i, 1e-2)),
                           GRAD_ATOL + GRAD_RTOL * abs(grad[i]))

    def test_a_whole_fit_runs_and_refuses_the_wrong_episode(self):
        from human_modeling.inference.scripts.run_human_inference import fit_variant
        est, info, full_mode = fit_variant(self.stack, self.inputs_full, self.variant)
        self.assertEqual(info["start"], "midpoint")
        # | The ascent must actually move: `fit_mode` returns the best point seen, which
        # | includes the start, so the likelihood check below alone would pass a stall.
        self.assertFalse(info["no_progress"])
        self.assertEqual(full_mode[THETA_NAMES.index("hlp_cognitive_cost_coef")], 0.0)
        self.assertGreaterEqual(log_likelihood(self.stack, self.inputs_full, full_mode),
                                log_likelihood(self.stack, self.inputs_full,
                                               self.variant.embed(self.midpoint)) - 1e-9)
        self.assertEqual({k.rsplit("_", 1)[0] for k in est if k.endswith("_mean")},
                         set(self.variant.free_theta_names))
        with self.assertRaises(ValueError):
            fit_variant(self.stack, self.inputs_ordinary, self.variant)




if __name__ == "__main__":
    unittest.main()
