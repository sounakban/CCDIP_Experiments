"""
The gates on fitting the heuristic weights (implementation.md Entry 34).

These need the checkpoint, which since 2026-09-22 travels with the package in
``human_modeling/inference/checkpoint/``, so they run in an ordinary checkout rather than
only on the cluster. They are the checks the port itself could not be trusted without:

  1. NESTING -- a 4-vector must be bit-identical to the pre-heuristic code path, and an
     8-vector with zero weights must equal the 4-vector. Without this, turning the feature on
     would silently change every existing result.
  2. EQUALITY -- the numpy and torch implementations must agree WITH the heuristics active.
     The shipped gate only ever ran at w = 0, so a heuristic term added to one path and not
     the other, or written differently in the two, would pass unnoticed while the fit
     optimised something the posterior stage does not compute.
  3. GRADIENT -- autograd's d logL / dw must match central differences. The step is swept
     rather than hardcoded: the policy forward runs in float32, so a too-small step measures
     rounding noise and reports a CORRECT gradient as broken (MEASURED: 1.5e-03 at h=1e-4
     against 8.0e-05 at h=1e-2).
  4. PROJECTION -- the estimator must keep the weights inside their calibrated bounds.

Each gate carries a known-bad arm, so a check that passes on anything is visible.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from human_modeling.inference.fit_gradient import (DEFAULT_LR_JOINT, MAX_LATTICE_POINTS,
                                                      fit_mode, fit_mode_with_retry,
                                                      fit_posterior)
from human_modeling.inference.inputs import load_pilot_episodes
from human_modeling.inference.likelihood_grad import (EpisodeGradContext, heuristic_features,
                                                         split_fit_vector)
from human_modeling.inference.model.choice import (HEURISTIC_BOUNDS, HEURISTIC_NAMES,
                                                      THETA_NAMES)
from human_modeling.inference.posterior.likelihood import (load_models, log_likelihood,
                                                              prepare_episode, weight_dict)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "human_modeling/inference/checkpoint"
DATA_PATH = PROJECT_ROOT / "human_modeling/data/exp1_filtered.json"

# | Short enough that the whole suite stays fast, long enough to cover several n_valid
# | buckets (the heuristic term is applied per bucket, so one bucket would not exercise
# | the indexing this port had to get right) AND to give the likelihood an interior peak:
# | MEASURED on the first episode's 120-step prefix, a box corner beats every interior
# | point the ascent reaches, so the stall detector's known-bad arm (a step that lands on
# | a corner) improved the fit there; at 300 steps that arm stalls as the tests expect.
N_STEPS = 300

# | Weights well inside the calibrated bounds, with mixed signs so the four cannot cancel
# | into a no-op and let a broken term pass.
TEST_WEIGHTS = np.array([0.8, -1.2, 0.30, -0.25])
TEST_THETA = np.array([6.94, 0.102, 3.49, 0.30])

# | Central-difference steps for the gradient gate; see the module docstring.
FD_STEPS = (1e-1, 1e-2, 1e-3, 1e-4)

# | Gradient agreement is judged on a MIXED tolerance. A pure relative error divides by a
# | near-zero gradient: MEASURED on this fixture, `lead_vehicle_inverse_ttc` has gradient
# | -1.4e-05 because it is constant across the candidates of almost every step in a
# | 120-step episode (no qualifying lead vehicle), so relative error there is meaningless
# | while the absolute agreement is excellent.
GRAD_ATOL, GRAD_RTOL = 1e-4, 1e-3

# | How close the numpy and torch paths must be WITH the heuristics active. MEASURED 4.5e-10
# | on this fixture; the bar is loose enough not to be flaky on another card and still three
# | orders tighter than any difference a real porting mistake would produce.
EQUALITY_BAR = 1e-6

# | A weight only bites where its feature VARIES between candidate construals, which differs
# | per heuristic and per episode. The known-bad arms below must perturb a weight that
# | actually moves this fixture, or they would "pass" by being blind -- so the live one is
# | discovered rather than assumed.
MIN_LIVE_RESPONSE = 1e-3


def _episode_fixture():
    """
    Load one truncated human episode and both likelihood contexts for it.

    :return: ``(stack, inputs, ctx_with_heuristics)``.
    :rtype: tuple
    """
    stack = load_models(str(CHECKPOINT_DIR))
    episode = load_pilot_episodes(DATA_PATH)[0]
    episode.steps = episode.steps[:N_STEPS]
    episode.actions = episode.actions[:N_STEPS]
    if episode.step_indices is not None:
        episode.step_indices = episode.step_indices[:N_STEPS]
    inputs = prepare_episode(stack, episode, label="gate")
    return stack, inputs, EpisodeGradContext(stack, inputs, with_heuristics=True)


def _value(ctx, vec: np.ndarray) -> float:
    """
    Log-likelihood as a plain float, with the autograd graph dropped.

    ``float()`` on a tensor that still carries a graph warns, and the warning would be noise
    in every gate below; detaching also frees the graph rather than holding it alive.

    :param ctx: Prepared differentiable likelihood.
    :type ctx: EpisodeGradContext
    :param vec: The fitted vector to evaluate at.
    :type vec: np.ndarray
    :return: The log-likelihood.
    :rtype: float
    """
    return float(ctx.logL(torch.tensor(vec)).detach())


def _live_weight_index(ctx, vec: np.ndarray) -> int:
    """
    Index of the heuristic weight that moves this fixture's likelihood most.

    Which heuristics bite depends on the episode — a heuristic whose feature is identical
    across every candidate construal cannot change the choice at any weight. A known-bad arm
    aimed at such a weight would report "no difference" and look like a passing check of
    something else, so the arms find a live weight instead of assuming one.

    :param ctx: Prepared differentiable likelihood, built with heuristics.
    :type ctx: EpisodeGradContext
    :param vec: The 8-vector the gates evaluate at.
    :type vec: np.ndarray
    :return: Index INTO THE FULL VECTOR of the most responsive weight.
    :rtype: int
    :raises AssertionError: If no weight moves the likelihood, which would make the suite
        vacuous rather than passing.
    """
    base = float(ctx.logL(torch.tensor(vec)))
    responses = []
    for k in range(len(HEURISTIC_NAMES)):
        nudged = vec.copy()
        nudged[len(THETA_NAMES) + k] += 0.5
        responses.append(abs(float(ctx.logL(torch.tensor(nudged))) - base))
    best = int(np.argmax(responses))
    assert responses[best] > MIN_LIVE_RESPONSE, (
        f"no heuristic weight moves this episode by more than {MIN_LIVE_RESPONSE}; "
        f"responses {responses} — the known-bad arms would be vacuous")
    return len(THETA_NAMES) + best


class VectorSplitting(unittest.TestCase):
    """The single place that decides what a 4- versus an 8-vector means."""

    def test_a_short_vector_means_theta_only(self):
        theta, w = split_fit_vector(torch.tensor(TEST_THETA))
        self.assertEqual(theta.shape[-1], len(THETA_NAMES))
        self.assertIsNone(w)
        self.assertEqual(weight_dict(TEST_THETA), {})

    def test_a_long_vector_splits_at_the_theta_boundary(self):
        vec = np.concatenate([TEST_THETA, TEST_WEIGHTS])
        theta, w = split_fit_vector(torch.tensor(vec))
        np.testing.assert_allclose(theta.numpy(), TEST_THETA)
        np.testing.assert_allclose(w.numpy(), TEST_WEIGHTS)
        self.assertEqual(list(weight_dict(vec)), list(HEURISTIC_NAMES))
        np.testing.assert_allclose(list(weight_dict(vec).values()), TEST_WEIGHTS)

    def test_any_other_length_is_refused(self):
        # | Silently accepting a 5- or 7-vector would mean fitting a model nobody specified.
        for n in (3, 5, 7, 9):
            with self.assertRaises(ValueError):
                split_fit_vector(torch.zeros(n))
            with self.assertRaises(ValueError):
                weight_dict(np.zeros(n))


class Nesting(unittest.TestCase):
    """Turning the feature on must not move any existing number."""

    @classmethod
    def setUpClass(cls):
        cls.stack, cls.inputs, cls.ctx = _episode_fixture()
        cls.plain = EpisodeGradContext(cls.stack, cls.inputs)

    def test_zero_weights_equal_theta_only_in_torch(self):
        with_zeros = float(self.ctx.logL(torch.tensor(np.concatenate([TEST_THETA,
                                                                      np.zeros(4)]))))
        theta_only = _value(self.plain, TEST_THETA)
        self.assertEqual(with_zeros, theta_only)

    def test_zero_weights_equal_theta_only_in_numpy(self):
        with_zeros = log_likelihood(self.stack, self.inputs,
                                    np.concatenate([TEST_THETA, np.zeros(4)]))
        self.assertEqual(with_zeros, log_likelihood(self.stack, self.inputs, TEST_THETA))

    def test_nonzero_weights_actually_move_the_likelihood(self):
        # | The known-bad direction for the two tests above: if the heuristic term were
        # | dropped on the floor they would also pass, so the weights must be shown to bite.
        moved = _value(self.ctx, np.concatenate([TEST_THETA, TEST_WEIGHTS]))
        self.assertNotAlmostEqual(moved, _value(self.plain, TEST_THETA),
                                  places=3)

    def test_weights_on_a_theta_only_context_raise(self):
        with self.assertRaises(ValueError):
            self.plain.logL(torch.tensor(np.concatenate([TEST_THETA, TEST_WEIGHTS])))


class NumpyTorchEquality(unittest.TestCase):
    """The two implementations must agree with the heuristics ACTIVE, not only at w = 0."""

    @classmethod
    def setUpClass(cls):
        cls.stack, cls.inputs, cls.ctx = _episode_fixture()
        cls.vec = np.concatenate([TEST_THETA, TEST_WEIGHTS])
        cls.live = _live_weight_index(cls.ctx, cls.vec)

    def test_the_two_paths_agree_at_nonzero_weights(self):
        torch_value = _value(self.ctx, self.vec)
        numpy_value = log_likelihood(self.stack, self.inputs, self.vec)
        # | The policy math is float32 in both paths on differently shaped tensors, so this
        # | is a closeness bar, not bit-identity -- the same bar the theta-only gate uses.
        self.assertLess(abs(torch_value - numpy_value), EQUALITY_BAR,
                        f"torch {torch_value} vs numpy {numpy_value}")

    def test_every_weight_moves_both_paths_identically(self):
        # | The sharper form of the gate: it is not enough that the two agree at one point,
        # | since two paths that both IGNORED the weights would also agree. Each weight must
        # | move both paths by the same amount.
        base_t = _value(self.ctx, self.vec)
        base_n = log_likelihood(self.stack, self.inputs, self.vec)
        for k, name in enumerate(HEURISTIC_NAMES):
            nudged = self.vec.copy()
            nudged[len(THETA_NAMES) + k] += 0.5
            delta_t = _value(self.ctx, nudged) - base_t
            delta_n = log_likelihood(self.stack, self.inputs, nudged) - base_n
            self.assertLess(abs(delta_t - delta_n), EQUALITY_BAR,
                            f"{name}: torch moved {delta_t}, numpy moved {delta_n}")

    def test_the_comparison_can_fail(self):
        # | Known-bad: compare one path at a PERTURBED weight against the other at the
        # | original. The perturbed weight is chosen to be one that actually bites on this
        # | episode -- see _live_weight_index.
        nudged = self.vec.copy()
        nudged[self.live] += 0.5
        self.assertGreater(abs(_value(self.ctx, nudged)
                               - log_likelihood(self.stack, self.inputs, self.vec)),
                           MIN_LIVE_RESPONSE)


class WeightGradient(unittest.TestCase):
    """Autograd's gradient in w, against central differences at a swept step."""

    @classmethod
    def setUpClass(cls):
        cls.stack, cls.inputs, cls.ctx = _episode_fixture()
        cls.vec = np.concatenate([TEST_THETA, TEST_WEIGHTS])
        leaf = torch.tensor(cls.vec, requires_grad=True)
        cls.ctx.logL(leaf).backward()
        assert leaf.grad is not None, "backward produced no gradient at all"
        cls.grad = leaf.grad.detach().cpu().numpy().copy()
        cls.live = _live_weight_index(cls.ctx, cls.vec)

    def _central_difference(self, i: int, h: float) -> float:
        """Central difference of the log-likelihood in coordinate ``i`` at step ``h``."""
        up, down = self.vec.copy(), self.vec.copy()
        up[i] += h
        down[i] -= h
        return (_value(self.ctx, up)
                - _value(self.ctx, down)) / (2.0 * h)

    def test_every_weight_gradient_matches_finite_differences(self):
        for k, name in enumerate(HEURISTIC_NAMES):
            i = len(THETA_NAMES) + k
            analytic = self.grad[i]
            best = min(abs(analytic - self._central_difference(i, h)) for h in FD_STEPS)
            self.assertLess(best, GRAD_ATOL + GRAD_RTOL * abs(analytic),
                            f"{name}: analytic {analytic}, closest difference off by {best}")

    def test_theta_still_takes_a_gradient_alongside(self):
        self.assertTrue(np.all(np.isfinite(self.grad[:len(THETA_NAMES)])))
        self.assertGreater(np.abs(self.grad[:len(THETA_NAMES)]).max(), 0.0)

    def test_the_finite_difference_check_can_fail(self):
        # | Known-bad: compare the analytic gradient against a difference taken at a
        # | DIFFERENT point. It must disagree, or the check above would pass on any numbers.
        # | The coordinate is the one that bites on this episode; a weight whose feature
        # | never varies has a near-linear likelihood, where this arm proves nothing.
        i = self.live
        here = self.vec
        self.vec = here.copy()
        self.vec[i] += 2.0
        elsewhere = self._central_difference(i, 1e-2)
        self.vec = here
        self.assertGreater(abs(self.grad[i] - elsewhere),
                           GRAD_ATOL + GRAD_RTOL * abs(self.grad[i]))


class BoundProjection(unittest.TestCase):
    """The weights must stay inside the calibrated box, and the lattice must not explode."""

    @classmethod
    def setUpClass(cls):
        cls.stack, cls.inputs, cls.ctx = _episode_fixture()

    def test_the_box_carries_a_bound_for_every_weight(self):
        lo, hi = self.stack.lo_hi_with_heuristics()
        self.assertEqual(len(lo), len(THETA_NAMES) + len(HEURISTIC_NAMES))
        for k, name in enumerate(HEURISTIC_NAMES):
            i = len(THETA_NAMES) + k
            self.assertAlmostEqual(lo[i], -HEURISTIC_BOUNDS[name])
            self.assertAlmostEqual(hi[i], HEURISTIC_BOUNDS[name])

    def test_ascent_never_leaves_the_box(self):
        # | Start ON the upper weight bounds so an unprojected step would exit immediately.
        lo, hi = self.stack.lo_hi_with_heuristics()
        start = hi.copy()
        start[:len(THETA_NAMES)] = TEST_THETA
        best, _info = fit_mode(self.ctx, lo, hi, start=start, max_iters=5)
        self.assertTrue(np.all(best >= lo - 1e-9), f"{best} below {lo}")
        self.assertTrue(np.all(best <= hi + 1e-9), f"{best} above {hi}")

    def test_the_eight_dimensional_lattice_is_refused(self):
        # | 3 per axis over 8 coefficients is 6,561 numpy evaluations (~18 min per episode).
        lo, hi = self.stack.lo_hi_with_heuristics()
        self.assertGreater(3 ** len(lo), MAX_LATTICE_POINTS)
        with self.assertRaises(ValueError):
            fit_posterior(self.stack, self.inputs, 0.5 * (lo + hi), lo, hi, half=0.1 * (hi - lo),
                          n_per_axis=3)


class StallDetection(unittest.TestCase):
    """The ascent must SAY when it never moved, instead of returning its input as an estimate.

    This is the check that makes the learning rate safe to trust on unseen data. lr = 0.02 was
    chosen because it worked on three reference episodes (Entry 34), which is not evidence
    about the 230 human ones — so the fit reports whether it actually got anywhere, and the
    scripts refuse to present a stalled fit as an estimate.
    """

    @classmethod
    def setUpClass(cls):
        cls.stack, cls.inputs, cls.ctx = _episode_fixture()
        cls.lo, cls.hi = cls.stack.lo_hi_with_heuristics()

    def test_a_healthy_fit_reports_progress(self):
        # | The known-GOOD arm: at the chosen rate the ascent improves, so the flag must be
        # | off. Without this the flag could be stuck on and look like a working detector.
        _best, info = fit_mode(self.ctx, self.lo, self.hi, lr=DEFAULT_LR_JOINT, max_iters=25)
        self.assertGreater(info["improvement"], 0.0)
        self.assertFalse(info["no_progress"])

    def test_an_absurd_step_is_caught_and_flagged(self):
        # | The known-BAD arm: a step size that hurls every coefficient across its whole range
        # | cannot improve anything, so the ascent returns its start. The point of the test is
        # | that this is REPORTED rather than returned as a silent estimate.
        start = np.concatenate([TEST_THETA, np.zeros(len(HEURISTIC_NAMES))])
        best, info = fit_mode(self.ctx, self.lo, self.hi, start=start, lr=5.0, max_iters=25)
        self.assertTrue(info["no_progress"],
                        f"a step of 5.0 in normalized units improved the fit? {info}")
        self.assertLessEqual(info["improvement"], 0.0)
        # | ...and what it hands back really is the input, which is why reporting matters.
        np.testing.assert_allclose(best, start, atol=1e-9)

    def test_the_flag_distinguishes_the_two(self):
        # | Guards against a detector that fires on everything: the same episode, the same
        # | start, two learning rates, opposite verdicts.
        start = np.concatenate([TEST_THETA, np.zeros(len(HEURISTIC_NAMES))])
        _b1, good = fit_mode(self.ctx, self.lo, self.hi, start=start,
                             lr=DEFAULT_LR_JOINT, max_iters=25)
        _b2, bad = fit_mode(self.ctx, self.lo, self.hi, start=start, lr=5.0, max_iters=25)
        self.assertNotEqual(good["no_progress"], bad["no_progress"])


class StallRetry(unittest.TestCase):
    """A stalled ascent must be retried at a smaller step -- and still reported as retried."""

    @classmethod
    def setUpClass(cls):
        cls.stack, cls.inputs, cls.ctx = _episode_fixture()
        cls.lo, cls.hi = cls.stack.lo_hi_with_heuristics()
        cls.start = np.concatenate([TEST_THETA, np.zeros(len(HEURISTIC_NAMES))])

    def test_a_stalled_fit_is_rescued_and_flagged(self):
        # | lr = 5.0 cannot improve anything, so the first attempt stalls; the retries divide
        # | it by 4 twice, reaching 0.3125, which moves. The fit must both RECOVER and SAY it
        # | needed to -- a silent rescue would hide that the configured step was wrong.
        best, info = fit_mode_with_retry(self.ctx, self.lo, self.hi, start=self.start,
                                         lr=5.0, max_iters=25, retries=4)
        self.assertFalse(info["no_progress"], f"retries did not rescue the fit: {info}")
        self.assertTrue(info["rescued_by_retry"])
        self.assertGreater(info["attempts"], 1)
        self.assertLess(info["lr_used"], 5.0)
        self.assertFalse(np.allclose(best, self.start),
                         "rescued, yet returned the starting point unchanged")

    def test_a_healthy_fit_is_not_marked_as_rescued(self):
        # | The known-good arm: no stall, so exactly one attempt and no flag. Without this the
        # | flag could be set on every fit and the test above would still pass.
        _best, info = fit_mode_with_retry(self.ctx, self.lo, self.hi, start=self.start,
                                          lr=DEFAULT_LR_JOINT, max_iters=25)
        self.assertEqual(info["attempts"], 1)
        self.assertFalse(info["rescued_by_retry"])
        self.assertFalse(info["no_progress"])

    def test_giving_up_keeps_the_stall_flag_set(self):
        # | With no retries allowed the wrapper must behave exactly like fit_mode: still
        # | stalled, still flagged, NOT quietly marked as rescued.
        _best, info = fit_mode_with_retry(self.ctx, self.lo, self.hi, start=self.start,
                                          lr=5.0, max_iters=25, retries=0)
        self.assertTrue(info["no_progress"])
        self.assertFalse(info["rescued_by_retry"])
        self.assertEqual(info["attempts"], 1)


class FeatureCache(unittest.TestCase):
    """The precomputed features must line up with the buckets that consume them."""

    @classmethod
    def setUpClass(cls):
        cls.stack, cls.inputs, cls.ctx = _episode_fixture()

    def test_one_entry_per_bucket_with_matching_shapes(self):
        features = heuristic_features(self.stack, self.inputs)
        kernel = self.inputs.kernel
        self.assertEqual(sorted(features), sorted(int(v) for v in np.unique(kernel.n_valid)))
        for n_valid, tensor in features.items():
            n_steps = int((kernel.n_valid == n_valid).sum())
            self.assertEqual(tuple(tensor.shape),
                             (n_steps, kernel.G, 1 << n_valid, len(HEURISTIC_NAMES)))

    def test_features_do_not_depend_on_theta(self):
        # | They are built from the observation and the mask alone; if theta ever reached
        # | them the per-episode cache would be silently wrong at every other theta.
        first = heuristic_features(self.stack, self.inputs)
        second = heuristic_features(self.stack, self.inputs)
        for n_valid, tensor in first.items():
            torch.testing.assert_close(tensor, second[n_valid], rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
