"""
Gates on the model variants (implementation.md Entry 35).

Two kinds. The checkpoint-FREE tests pin the mapping itself: which coefficients each variant
frees, that embedding and extraction invert each other, that the torch embedding keeps the
autograd graph, and that a malformed variant is refused. The checkpoint-BACKED tests are the
NESTING gates on the real likelihood: a variant's context evaluated at a free vector must
equal the underlying context at the embedded full vector EXACTLY -- the variant machinery
is a re-indexing, so anything other than bit-identity means it changed the arithmetic.

Every gate has a known-bad arm where one is meaningful, so a test that could pass by being
blind is shown not to be.

Run with ``python -m pytest human_modeling/inference/tests/test_variants.py``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from human_modeling.inference import fit_gradient as fg
from human_modeling.inference.inputs import load_pilot_episodes
from human_modeling.inference.likelihood_grad import EpisodeGradContext
from human_modeling.inference.model.choice import (FIT_NAMES, HEURISTIC_BOUNDS,
                                                      HEURISTIC_NAMES, THETA_NAMES)
from human_modeling.inference.posterior.grid import build_grid
from human_modeling.inference.posterior.likelihood import (load_models, log_likelihood,
                                                              prepare_episode)
from human_modeling.inference.posterior.update import summarize
from human_modeling.inference.scripts.run_human_inference import (
    FIELDS, POSTERIOR_TAGS, VariantSink, boundary_coefficients, full_vector_from_row, make_row,
    rollout_vector)
from human_modeling.inference.variants import (FIT_ORDER, NESTED_PAIRS, VARIANTS, FitVariant,
                                                  VariantGradContext, as_eight,
                                                  check_pins_inside, get_variant, ordered)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "human_modeling/inference/checkpoint"
DATA_PATH = PROJECT_ROOT / "human_modeling/data/exp1_filtered.json"
N_STEPS = 120                      # same truncation as test_heuristic_fitting.py

THETA = np.array([6.94, 0.102, 3.49, 0.30])
WEIGHTS = np.array([0.8, -1.2, 0.30, -0.25])
FULL8 = np.concatenate([THETA, WEIGHTS])
COG = THETA_NAMES.index("hlp_cognitive_cost_coef")
MEAN_DIST = FIT_NAMES.index("mean_vehicle_distance")


class QuadraticCtx:
    """A stand-in likelihood: ``-sum(c_i * (x_i - m_i)^2)`` so curvature is known exactly."""

    def __init__(self, centre: np.ndarray, curvature: np.ndarray) -> None:
        self.m = torch.tensor(centre, dtype=torch.float64)
        self.c = torch.tensor(curvature, dtype=torch.float64)
        self.features = object()          # duck-types "built with heuristics"

    def logL(self, x: torch.Tensor) -> torch.Tensor:
        return -(self.c * (x - self.m) ** 2).sum()


# ===========================================================================
# Checkpoint-free
# ===========================================================================

class TestDefinitions(unittest.TestCase):
    def test_the_variants_have_the_agreed_free_counts(self):
        # | [full-visibility] The fifth, C_full_visibility, was added 2026-11-10 (Entry 36).
        counts = {v.name: v.n_free for v in VARIANTS.values()}
        self.assertEqual(counts, {"B_no_heuristics": 4, "A_full": 8,
                                  "A_no_mean_distance": 7, "C_no_cognitive_cost": 3,
                                  "C_full_visibility": 3})

    def test_fit_order_lists_every_parent_before_its_child(self):
        for v in VARIANTS.values():
            if v.warm_start_from:
                self.assertLess(FIT_ORDER.index(v.warm_start_from), FIT_ORDER.index(v.name))

    def test_ordered_sorts_into_fit_order_and_drops_duplicates(self):
        names = [v.name for v in ordered(["A_full", "C_no_cognitive_cost", "B_no_heuristics",
                                          "A_full"])]
        self.assertEqual(names, ["C_no_cognitive_cost", "B_no_heuristics", "A_full"])

    def test_the_chain_is_nested_smallest_first(self):
        # | Every start a variant uses is computed before it, and each NESTED_PAIRS' larger
        # | model really does contain the smaller: same or fewer pins, superset of free names.
        # | [full-visibility] The sanity-check variant is listed after the chain, so the four
        # | chained positions are unchanged (Entry 36).
        self.assertEqual(FIT_ORDER, ("C_no_cognitive_cost", "B_no_heuristics",
                                     "A_no_mean_distance", "A_full", "C_full_visibility"))
        for v in VARIANTS.values():
            for start in ((v.warm_start_from,) if v.warm_start_from else ()) + v.extra_starts:
                self.assertLess(FIT_ORDER.index(start), FIT_ORDER.index(v.name))
        for smaller, larger in NESTED_PAIRS:
            self.assertTrue(set(VARIANTS[smaller].free_names) <= set(VARIANTS[larger].free_names))
        self.assertEqual(VARIANTS["B_no_heuristics"].extra_starts, ("C_no_cognitive_cost",))
        self.assertEqual(VARIANTS["A_full"].warm_start_from, "A_no_mean_distance")

    def test_free_names_put_theta_first_in_fit_names_order(self):
        for v in VARIANTS.values():
            positions = [FIT_NAMES.index(n) for n in v.free_names]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(v.free_names[:v.n_free_theta], v.free_theta_names)

    def test_b_is_the_plain_four_vector(self):
        b = get_variant("B_no_heuristics")
        self.assertFalse(b.uses_heuristics)
        self.assertEqual(b.full_names, THETA_NAMES)
        np.testing.assert_array_equal(b.embed(THETA), THETA)

    def test_unknown_variant_lists_the_valid_names(self):
        with self.assertRaises(KeyError) as ctx:
            get_variant("D_static_flag_mid")
        self.assertIn("B_no_heuristics", str(ctx.exception))

    def test_known_bad_definitions_are_refused(self):
        with self.assertRaises(ValueError):               # theta neither free nor pinned
            FitVariant("bad", "bad", free_theta=THETA_NAMES[:3])
        with self.assertRaises(ValueError):               # unknown coefficient
            FitVariant("bad", "bad", free_theta=THETA_NAMES + ("hlp_made_up",))
        with self.assertRaises(ValueError):               # both free and pinned
            FitVariant("bad", "bad", free_theta=THETA_NAMES,
                       pinned_theta={"hlp_cognitive_cost_coef": 0.0})


class TestBoundaryStart(unittest.TestCase):
    """[boundary-start] A no-progress fit is a boundary optimum only from a smaller model's
    solution with every added coefficient at a bound; anything else is a stall."""

    def _info(self, variant: FitVariant, start: str, at_bound: dict[str, bool],
              no_progress: bool = True) -> dict:
        return {"no_progress": no_progress, "start": start,
                "at_bound": [at_bound.get(n, False) for n in variant.free_names]}

    def test_added_over_names_what_the_larger_model_frees_beyond_the_smaller(self):
        b, a7, a8 = (get_variant(n) for n in ("B_no_heuristics", "A_no_mean_distance", "A_full"))
        self.assertEqual(b.added_over(get_variant("C_no_cognitive_cost")),
                         ("hlp_cognitive_cost_coef",))
        self.assertEqual(a8.added_over(a7), ("mean_vehicle_distance",))
        self.assertEqual(len(a7.added_over(b)), 3)
        self.assertEqual(b.added_over(a8), ())                 # smaller frees nothing new

    def test_b_kept_at_cs_solution_with_cognitive_cost_on_its_bound_is_a_boundary_optimum(self):
        b = get_variant("B_no_heuristics")
        info = self._info(b, "C_no_cognitive_cost", {"hlp_cognitive_cost_coef": True})
        self.assertEqual(boundary_coefficients(b, info), ("hlp_cognitive_cost_coef",))

    def test_known_bad_arms_read_as_stalls_or_progress(self):
        b, a8 = get_variant("B_no_heuristics"), get_variant("A_full")
        # | Cognitive cost off its bound: the start was not on the boundary, so it stalled.
        self.assertEqual(boundary_coefficients(
            b, self._info(b, "C_no_cognitive_cost", {"hlp_cognitive_cost_coef": False})), ())
        # | A midpoint start has no smaller model to tie, even with every axis on a bound.
        self.assertEqual(boundary_coefficients(
            b, self._info(b, "midpoint", {n: True for n in b.free_names})), ())
        # | A fit that moved is not a no-progress fit at all.
        self.assertEqual(boundary_coefficients(
            b, self._info(b, "C_no_cognitive_cost", {"hlp_cognitive_cost_coef": True},
                          no_progress=False)), ())
        # | Entry 34's stall: a heuristic weight started at 0, which its symmetric box never
        # | makes a bound, while an unrelated theta happens to sit on one.
        self.assertEqual(boundary_coefficients(
            a8, self._info(a8, "A_no_mean_distance", {"hlp_collision_penalty": True})), ())

    def test_a_weights_zero_start_is_never_on_a_bound(self):
        # | The weight box is [-half, +half] (ModelStack.lo_hi_with_heuristics), so a start at
        # | 0 sits mid-box, far outside fit_mode's 1%-of-span "at a bound" margin, whenever
        # | every half-width is positive. This is what keeps Entry 34's stall a stall.
        for name, half in HEURISTIC_BOUNDS.items():
            self.assertGreater(half, 0.0, name)


class TestEmbedding(unittest.TestCase):
    def test_embed_and_extract_invert_each_other(self):
        for v in VARIANTS.values():
            free = v.extract(FULL8)
            self.assertEqual(free.shape, (v.n_free,))
            np.testing.assert_array_equal(v.extract(v.embed(free)), free)

    def test_c_inserts_zero_cognitive_cost_in_place(self):
        c = get_variant("C_no_cognitive_cost")
        full = c.embed(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(full.shape, (4,))
        self.assertEqual(full[COG], 0.0)
        np.testing.assert_array_equal(np.delete(full, COG), [1.0, 2.0, 3.0])

    def test_no_mean_distance_pins_the_last_weight_at_zero(self):
        a = get_variant("A_no_mean_distance")
        full = a.embed(np.arange(1.0, 8.0))
        self.assertEqual(full.shape, (8,))
        self.assertEqual(full[MEAN_DIST], 0.0)
        np.testing.assert_array_equal(np.delete(full, MEAN_DIST), np.arange(1.0, 8.0))

    def test_embed_refuses_the_wrong_length(self):
        with self.assertRaises(ValueError):
            get_variant("A_full").embed(THETA)
        with self.assertRaises(ValueError):
            get_variant("C_no_cognitive_cost").embed_torch(torch.zeros(4))

    def test_torch_embedding_keeps_the_graph_and_is_identity_when_nothing_is_pinned(self):
        free = torch.arange(1.0, 8.0, dtype=torch.float64, requires_grad=True)
        full = get_variant("A_no_mean_distance").embed_torch(free)
        (full * torch.arange(1.0, 9.0, dtype=torch.float64)).sum().backward()
        # | d(sum_j j*full_j)/d(free_i) is the coefficient of the slot free_i lands in.
        np.testing.assert_array_equal(free.grad.numpy(), np.delete(np.arange(1.0, 9.0), MEAN_DIST))
        eight = torch.zeros(8, dtype=torch.float64)
        self.assertIs(get_variant("A_full").embed_torch(eight), eight)
        four = torch.zeros(4, dtype=torch.float64)
        self.assertIs(get_variant("B_no_heuristics").embed_torch(four), four)

    def test_warm_start_from_b_appends_zero_weights(self):
        np.testing.assert_array_equal(get_variant("A_full").start_from(THETA),
                                      np.concatenate([THETA, np.zeros(4)]))
        np.testing.assert_array_equal(get_variant("A_no_mean_distance").start_from(THETA),
                                      np.concatenate([THETA, np.zeros(3)]))

    def test_as_eight_pads_and_refuses(self):
        np.testing.assert_array_equal(as_eight(THETA)[4:], 0.0)
        np.testing.assert_array_equal(as_eight(FULL8), FULL8)
        with self.assertRaises(ValueError):
            as_eight(np.zeros(5))

    def test_lo_hi_picks_the_free_axes_of_the_eight_box(self):
        class Stack:
            def lo_hi_with_heuristics(self):
                bound = np.array([HEURISTIC_BOUNDS[n] for n in HEURISTIC_NAMES])
                return (np.concatenate([np.zeros(4), -bound]),
                        np.concatenate([np.array([10.0, 0.15, 6.0, 1.5]), bound]))
        lo, hi = get_variant("C_no_cognitive_cost").lo_hi(Stack())
        np.testing.assert_array_equal(hi, [10.0, 6.0, 1.5])
        lo, hi = get_variant("A_no_mean_distance").lo_hi(Stack())
        self.assertEqual(len(lo), 7)
        self.assertEqual(hi[-1], HEURISTIC_BOUNDS["closest_vehicle_distance"])

    def test_pins_outside_the_trained_range_are_refused(self):
        class Stack:
            bounds = {"hlp_cognitive_cost_coef": (0.05, 0.15)}
        with self.assertRaises(ValueError):
            check_pins_inside(get_variant("C_no_cognitive_cost"), Stack())
        Stack.bounds = {"hlp_cognitive_cost_coef": (0.0, 0.15)}
        check_pins_inside(get_variant("C_no_cognitive_cost"), Stack())   # no raise


class TestVariantContextOnAQuadratic(unittest.TestCase):
    """The estimator only ever calls ``ctx.logL``; a quadratic makes the answer checkable."""

    def test_c_context_matches_the_embedded_evaluation_exactly(self):
        base = QuadraticCtx(np.array([1.0, 0.0, 2.0, 3.0]), np.array([1.0, 5.0, 2.0, 3.0]))
        ctx = VariantGradContext(base, get_variant("C_no_cognitive_cost"))
        free = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
        expected = base.logL(torch.tensor([0.5, 0.0, 1.5, 2.5], dtype=torch.float64))
        self.assertEqual(float(ctx.logL(free)), float(expected))

    def test_heuristic_variant_refuses_a_theta_only_context(self):
        class ThetaOnly:
            features = None
        with self.assertRaises(ValueError):
            VariantGradContext(ThetaOnly(), get_variant("A_full"))
        VariantGradContext(ThetaOnly(), get_variant("B_no_heuristics"))      # fine

    def test_hessian_axes_skips_the_rest_and_marks_them_nan(self):
        c = np.array([1.0, 2.0, 3.0, 4.0])
        ctx = QuadraticCtx(np.full(4, 0.5), c)
        lo, hi = np.zeros(4), np.ones(4)
        h = fg.hessian_diag(ctx, np.full(4, 0.5), lo, hi, axes=range(2))
        np.testing.assert_allclose(h[:2], -2.0 * c[:2], rtol=1e-6)
        self.assertTrue(np.isnan(h[2:]).all())
        full = fg.hessian_diag(ctx, np.full(4, 0.5), lo, hi)
        np.testing.assert_allclose(full, -2.0 * c, rtol=1e-6)


class TestSummarizeNames(unittest.TestCase):
    def test_summary_is_keyed_by_the_names_given(self):
        grid = build_grid(np.zeros(3), np.ones(3), 3)
        names = get_variant("C_no_cognitive_cost").free_theta_names
        out = summarize(grid, np.zeros(grid.n_points), bounds=(np.zeros(3), np.ones(3)),
                        names=names)
        self.assertEqual({k.rsplit("_", 1)[0] for k in out if k.endswith("_mean")}, set(names))
        self.assertNotIn("hlp_cognitive_cost_coef_mean", out)

    def test_mismatched_names_are_refused(self):
        grid = build_grid(np.zeros(3), np.ones(3), 3)
        with self.assertRaises(ValueError):
            summarize(grid, np.zeros(grid.n_points))            # default 4 names, 3 axes


class TestRowSchema(unittest.TestCase):
    def _fit_like(self, variant: FitVariant, full: np.ndarray):
        est = {f"{n}_{t}": 0.25 for n in variant.free_theta_names
               for t in ("mean",) + POSTERIOR_TAGS}
        info = {"iters": 7, "no_progress": False, "rescued_by_retry": False, "lr_used": 0.02}
        return make_row(("subj", 2), 500, variant, est, info, full, -10.0, -9.0, 3.2,
                        variant.warm_start_from or "midpoint")

    def test_pinned_theta_is_written_as_its_value_with_blank_posterior_columns(self):
        c = get_variant("C_no_cognitive_cost")
        row = self._fit_like(c, c.embed(np.array([1.0, 2.0, 3.0])))
        self.assertEqual(row["hlp_cognitive_cost_coef"], 0.0)
        self.assertEqual(row["mode_hlp_cognitive_cost_coef"], 0.0)
        for tag in POSTERIOR_TAGS:
            self.assertEqual(row[f"hlp_cognitive_cost_coef_{tag}"], "")
            self.assertEqual(row[f"hlp_collision_penalty_{tag}"], 0.25)
        self.assertEqual(row["n_free_params"], 3)
        self.assertEqual(row["w_lead_vehicle_inverse_ttc"], "")

    def test_weights_are_zero_when_pinned_and_blank_when_absent(self):
        a = get_variant("A_no_mean_distance")
        row = self._fit_like(a, a.embed(np.arange(1.0, 8.0)))
        self.assertEqual(row["w_mean_vehicle_distance"], 0.0)
        self.assertEqual(row["w_closest_vehicle_distance"], 7.0)
        b = get_variant("B_no_heuristics")
        self.assertEqual(self._fit_like(b, THETA)["w_closest_vehicle_distance"], "")
        self.assertEqual(set(row), set(FIELDS))

    def test_peak_round_trips_through_the_csv_row(self):
        a = get_variant("A_no_mean_distance")
        full = a.embed(np.arange(1.0, 8.0))
        row = {k: str(v) for k, v in self._fit_like(a, full).items()}
        np.testing.assert_array_equal(full_vector_from_row(row), full)
        b_row = {k: str(v) for k, v in self._fit_like(get_variant("B_no_heuristics"), THETA).items()}
        np.testing.assert_array_equal(full_vector_from_row(b_row), as_eight(THETA))

    def test_rollout_vector_takes_theta_means_and_keeps_the_rest_at_the_peak(self):
        a = get_variant("A_no_mean_distance")
        full = a.embed(np.arange(1.0, 8.0))
        est = {f"{n}_mean": -1.0 for n in THETA_NAMES}
        out = rollout_vector(a, est, full)
        np.testing.assert_array_equal(out[:4], -1.0)
        np.testing.assert_array_equal(out[4:], full[4:])
        c = get_variant("C_no_cognitive_cost")
        out = rollout_vector(c, {f"{n}_mean": -1.0 for n in c.free_theta_names},
                             c.embed(np.array([1.0, 2.0, 3.0])))
        self.assertEqual(out[COG], 0.0)                     # the pin survives


class TestVariantSinkResume(unittest.TestCase):
    """The two resume defects the independent review found (2026-09-23), pinned."""

    def _row(self, variant: FitVariant, game: int) -> dict:
        est = {f"{n}_{t}": 0.5 for n in variant.free_theta_names for t in ("mean",) + POSTERIOR_TAGS}
        info = {"iters": 3, "no_progress": False, "rescued_by_retry": False, "lr_used": 0.08}
        return make_row(("s", game), 400, variant, est, info, variant.embed(np.full(variant.n_free, 0.5)),
                        -1.0, -1.0, 1.0, "midpoint")

    def test_header_only_file_gets_no_second_header(self):
        import tempfile
        b = get_variant("B_no_heuristics")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fit_variant_B.csv"
            VariantSink(path).close()                      # a run killed before any fit landed
            VariantSink(path).close()                      # resumed and killed again
            sink = VariantSink(path)                       # third run must still parse
            sink.write(("s", 1), self._row(b, 1), np.full(4, 0.5))
            sink.close()
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("subject_id,game_index"), 1)
            self.assertEqual(set(VariantSink(path).done), {("s", 1)})

    def test_foreign_header_is_refused_rather_than_appended_to(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "human_fits.csv"
            path.write_text("subject_id,game_index,hlp_collision_penalty\ns,1,2.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                VariantSink(path)


# ===========================================================================
# Checkpoint-backed nesting gates
# ===========================================================================

@unittest.skipUnless(CHECKPOINT_DIR.exists() and DATA_PATH.exists(),
                     "needs the packaged checkpoint and the human data")
class TestNestingOnTheRealLikelihood(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = load_models(str(CHECKPOINT_DIR))
        episode = load_pilot_episodes(DATA_PATH)[0]
        episode.steps = episode.steps[:N_STEPS]
        episode.actions = episode.actions[:N_STEPS]
        if episode.step_indices is not None:
            episode.step_indices = episode.step_indices[:N_STEPS]
        cls.inputs = prepare_episode(cls.stack, episode, label="variants")
        cls.base4 = EpisodeGradContext(cls.stack, cls.inputs)
        cls.base8 = EpisodeGradContext(cls.stack, cls.inputs, with_heuristics=True)

    @staticmethod
    def _value(ctx, vec):
        return float(ctx.logL(torch.tensor(vec, dtype=torch.float64)).detach())

    def test_c_equals_theta_with_cognitive_cost_zeroed_exactly(self):
        c = get_variant("C_no_cognitive_cost")
        free = c.extract(THETA)
        pinned = THETA.copy()
        pinned[COG] = 0.0
        self.assertEqual(self._value(VariantGradContext(self.base4, c), free),
                         self._value(self.base4, pinned))
        # | Known-bad: the pin is not a no-op -- cognitive cost at THETA's value differs.
        self.assertNotEqual(self._value(self.base4, pinned), self._value(self.base4, THETA))

    def test_no_mean_distance_equals_full_with_that_weight_zeroed_exactly(self):
        a7 = get_variant("A_no_mean_distance")
        zeroed = FULL8.copy()
        zeroed[MEAN_DIST] = 0.0
        self.assertEqual(self._value(VariantGradContext(self.base8, a7), a7.extract(FULL8)),
                         self._value(self.base8, zeroed))
        # | Known-bad: mean distance is live on this fixture at this weight.
        self.assertNotEqual(self._value(self.base8, zeroed), self._value(self.base8, FULL8))

    def test_b_and_a_full_are_the_underlying_paths_exactly(self):
        self.assertEqual(self._value(VariantGradContext(self.base4, get_variant("B_no_heuristics")), THETA),
                         self._value(self.base4, THETA))
        self.assertEqual(self._value(VariantGradContext(self.base8, get_variant("A_full")), FULL8),
                         self._value(self.base8, FULL8))

    def test_c_gradient_is_the_four_gradient_on_the_free_axes(self):
        c = get_variant("C_no_cognitive_cost")
        free = torch.tensor(c.extract(THETA), requires_grad=True)
        VariantGradContext(self.base4, c).logL(free).backward()
        pinned = THETA.copy()
        pinned[COG] = 0.0
        full = torch.tensor(pinned, requires_grad=True)
        self.base4.logL(full).backward()
        np.testing.assert_allclose(free.grad.numpy(), np.delete(full.grad.numpy(), COG),
                                   rtol=0, atol=1e-12)

    def test_two_start_b_is_never_below_c_and_records_the_winner(self):
        # | The chain's guarantee at the one link that had none: B started from the midpoint
        # | AND from C's solution must end at least as high as C (Entry 35 addendum 2).
        from human_modeling.inference.scripts.run_human_inference import fit_variant
        c = get_variant("C_no_cognitive_cost")
        b = get_variant("B_no_heuristics")
        _est_c, info_c, mode_c = fit_variant(self.stack, self.inputs, c)
        _est_b, info_b, mode_b = fit_variant(self.stack, self.inputs, b,
                                             alternative_starts=[("C_no_cognitive_cost", mode_c)])
        self.assertIn(info_b["start"], ("midpoint", "C_no_cognitive_cost"))
        self.assertEqual(set(info_b["attempts_by_start"]), {"midpoint", "C_no_cognitive_cost"})
        self.assertGreaterEqual(log_likelihood(self.stack, self.inputs, mode_b),
                                log_likelihood(self.stack, self.inputs, mode_c) - 1e-6)
        # | Known-bad arm: the single-start call must NOT report an alternative it never ran.
        _e, info_single, _m = fit_variant(self.stack, self.inputs, c)
        self.assertEqual(info_single["start"], "midpoint")
        self.assertEqual(list(info_single["attempts_by_start"]), ["midpoint"])

    def test_posterior_lattice_over_free_theta_scores_the_embedded_vectors(self):
        c = get_variant("C_no_cognitive_cost")
        lo, hi = c.lo_hi(self.stack)
        centre = c.extract(THETA)
        est = fg.fit_posterior(self.stack, self.inputs, centre, lo, hi, half=0.02 * (hi - lo),
                               n_per_axis=3, embed=lambda p: c.embed(p),
                               names=c.free_theta_names)
        self.assertEqual({k.rsplit("_", 1)[0] for k in est if k.endswith("_mean")},
                         set(c.free_theta_names))
        # | A 3-per-axis box 2% wide around the centre: the means must sit inside it.
        for j, name in enumerate(c.free_theta_names):
            self.assertLessEqual(abs(est[f"{name}_mean"] - centre[j]), 0.02 * (hi[j] - lo[j]) + 1e-9)
        # | And the embedded scoring is the plain likelihood at the pinned vector.
        pinned = THETA.copy()
        pinned[COG] = 0.0
        self.assertEqual(log_likelihood(self.stack, self.inputs, c.embed(centre)),
                         log_likelihood(self.stack, self.inputs, pinned))


if __name__ == "__main__":
    unittest.main()
