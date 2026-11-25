"""
Checkpoint-free gates on the full-visibility sanity read-out (implementation.md Entry 36).

Every statistic is checked on data whose answer is known, and each verdict outcome is
produced on purpose so a read-out that always says the same thing cannot pass:
    1. The verdict's three outcomes, including both margins and both sides.
    2. The interval equals the model-comparison figures' (`paired_difference`) at the seed.
    3. SD and both SEs against direct computation; the participant SE exceeds the naive one
       when episodes within a participant are correlated (the reason both are reported).
    4. Delta BIC is twice the NLL difference for equal k and n, and mismatched n is refused.

Run with ``python -m pytest human_modeling/inference/tests/test_full_visibility_sanity.py``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from human_modeling.inference.scripts.full_visibility_sanity import (check_variant_column,
                                                                        delta_summary,
                                                                        episode_delta_bic,
                                                                        equivalence_verdict)
from human_modeling.inference.scripts.plot_model_comparison import paired_difference

N_PARTICIPANTS = 39
EPISODES_EACH = 6
SEED = 0


def _clustered(shift: float, person_sd: float, episode_sd: float,
               seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-episode values with a per-participant offset, so episodes within a person correlate.

    :param shift: True mean.
    :type shift: float
    :param person_sd: SD of the per-participant offset.
    :type person_sd: float
    :param episode_sd: SD of the per-episode noise.
    :type episode_sd: float
    :param seed: Generator seed.
    :type seed: int
    :return: ``(values, subjects)``, each ``(N_PARTICIPANTS * EPISODES_EACH,)``.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, person_sd, N_PARTICIPANTS)                    # (P,)
    subjects = np.repeat([f"s{i:02d}" for i in range(N_PARTICIPANTS)], EPISODES_EACH)
    values = (shift + np.repeat(offsets, EPISODES_EACH)
              + rng.normal(0.0, episode_sd, len(subjects)))                 # (P * E,)
    return values, subjects




class TestVerdict(unittest.TestCase):
    def test_the_three_outcomes(self):
        self.assertEqual(equivalence_verdict(-3.0, 4.0, 10.0), "no meaningful change")
        self.assertEqual(equivalence_verdict(12.0, 30.0, 10.0), "meaningful change")
        self.assertEqual(equivalence_verdict(-40.0, -11.0, 10.0), "meaningful change")
        self.assertEqual(equivalence_verdict(5.0, 15.0, 10.0), "inconclusive")
        self.assertEqual(equivalence_verdict(-15.0, -5.0, 10.0), "inconclusive")
        self.assertEqual(equivalence_verdict(-12.0, 12.0, 10.0), "inconclusive")

    def test_an_interval_touching_the_margin_is_not_inside_it(self):
        self.assertEqual(equivalence_verdict(0.0, 10.0, 10.0), "inconclusive")
        self.assertEqual(equivalence_verdict(10.0, 20.0, 10.0), "meaningful change")




class TestSummary(unittest.TestCase):
    def test_known_shifts_give_each_verdict(self):
        cases = {"no meaningful change": 2.0, "meaningful change": 60.0, "inconclusive": 10.0}
        for expected, shift in cases.items():
            values, subjects = _clustered(shift, person_sd=3.0, episode_sd=4.0)
            self.assertEqual(delta_summary(values, subjects, SEED)["verdict"], expected,
                             f"shift {shift}")

    def test_interval_is_the_figures_interval(self):
        # | paired_difference takes x - y; passing y = 0 makes its difference these values.
        values, subjects = _clustered(5.0, person_sd=3.0, episode_sd=4.0)
        ours = delta_summary(values, subjects, SEED)
        figures = paired_difference(values, np.zeros_like(values), subjects, seed=SEED)
        self.assertEqual((ours["ci_lo"], ours["ci_hi"]), (figures["ci_lo"], figures["ci_hi"]))
        self.assertEqual(ours["mean"], figures["mean"])
        # | Known-bad: a different seed gives a different interval, so the match is not luck.
        other = delta_summary(values, subjects, SEED + 1)
        self.assertNotEqual((ours["ci_lo"], ours["ci_hi"]), (other["ci_lo"], other["ci_hi"]))

    def test_sd_and_both_standard_errors(self):
        values, subjects = _clustered(5.0, person_sd=6.0, episode_sd=2.0)
        out = delta_summary(values, subjects, SEED)
        self.assertAlmostEqual(out["sd"], float(np.std(values, ddof=1)), places=12)
        self.assertAlmostEqual(out["se_naive"], out["sd"] / np.sqrt(len(values)), places=12)
        # | Strong within-person correlation: the participant SE must exceed the naive one.
        self.assertGreater(out["se_bootstrap"], 1.5 * out["se_naive"])
        # | Known-bad direction: with no person effect the two agree to within 25%.
        flat, subjects_flat = _clustered(5.0, person_sd=0.0, episode_sd=2.0)
        out_flat = delta_summary(flat, subjects_flat, SEED)
        self.assertLess(abs(out_flat["se_bootstrap"] / out_flat["se_naive"] - 1.0), 0.25)
        self.assertEqual((out["n_episodes"], out["n_participants"]),
                         (N_PARTICIPANTS * EPISODES_EACH, N_PARTICIPANTS))

    def test_counts(self):
        values = np.array([-1.0, -20.0, 3.0, 15.0])
        out = delta_summary(values, np.array(["a", "a", "b", "b"]), SEED)
        self.assertEqual(out["n_full_visibility_better"], 2)
        self.assertEqual(out["n_within_margin"], 2)




class TestDeltaBic(unittest.TestCase):
    def test_equal_k_and_n_make_it_twice_the_nll_difference(self):
        full = {("s", 1): {"logL": -1000.0, "k": 3, "n": 900},
                ("s", 2): {"logL": -500.0, "k": 3, "n": 400}}
        ref = {("s", 1): {"logL": -990.0, "k": 3, "n": 900},
               ("s", 2): {"logL": -505.0, "k": 3, "n": 400}}
        out = episode_delta_bic(full, ref, [("s", 1), ("s", 2)])
        np.testing.assert_allclose(out, [20.0, -10.0], rtol=0, atol=1e-9)

    def test_different_frame_counts_are_refused(self):
        full = {("s", 1): {"logL": -1000.0, "k": 3, "n": 900}}
        ref = {("s", 1): {"logL": -990.0, "k": 3, "n": 899}}
        with self.assertRaises(ValueError):
            episode_delta_bic(full, ref, [("s", 1)])

    def test_a_file_holding_another_model_is_refused(self):
        # | Independent review, 2026-11-10: B's file passed as the stored C would otherwise be
        # | read silently as a k = 4 reference.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fit.csv"
            path.write_text("subject_id,game_index,variant\ns,1,B_no_heuristics\n",
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                check_variant_column(path, "C_no_cognitive_cost")
            check_variant_column(path, "B_no_heuristics")                   # no raise




if __name__ == "__main__":
    unittest.main()
