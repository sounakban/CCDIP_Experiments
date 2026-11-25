"""
Checkpoint-free suite for `scripts/analyze_param_recovery.py` and the `bootstrap_spearman`
it added to `scripts/bootstrap_stats.py` (implementation.md Entry 29).

Every statistic is checked in BOTH directions -- on an input whose answer is known to be good
and on one known to be bad -- because a recovery read-out that silently reports a healthy
number on broken input is worse than no read-out at all (CLAUDE.md *Verification & Evidence*).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from human_modeling.inference.model.choice import THETA_NAMES
from human_modeling.inference.scripts.analyze_param_recovery import (
    EPISODES_PER_PERSON, POOLED, ceiling_for, rmse_and_bias, select, shared_limits,
    spearman_brown, split_half_reliability, theta_ranges,
)
from human_modeling.inference.scripts.bootstrap_stats import (
    bootstrap_correlation, bootstrap_indices, bootstrap_spearman,
)

COEFFICIENT = "hlp_collision_penalty"


def make_rows(estimates_by_person: dict[str, list[float]], stratum: str = "interior",
              coefficient: str = COEFFICIENT, truth: float = 5.0) -> list[dict[str, str]]:
    """Episode-level recovery rows in the schema `person_recovery.csv` uses.

    :param estimates_by_person: Fitted value per episode, keyed by person.
    :type estimates_by_person: dict[str, list[float]]
    :param stratum: Stratum label written on every row.
    :type stratum: str
    :param coefficient: Which coefficient the values belong to.
    :type coefficient: str
    :param truth: The single true value shared by every episode of every person.
    :type truth: float
    :return: Rows ready for `select` / `split_half_reliability`.
    :rtype: list[dict[str, str]]
    """
    return [{"level": "episode", "stratum": stratum, "person": person, "episode": str(i),
             f"true_{coefficient}": str(truth), f"fit_{coefficient}": str(value)}
            for person, values in estimates_by_person.items()
            for i, value in enumerate(values)]


# ===========================================================================
# 1. RMSE and bias
# ===========================================================================


def test_rmse_and_bias_on_a_perfect_match():
    """A perfect estimate scores zero on both, and its intervals collapse to zero."""
    values = np.array([1.0, 4.0, 9.0, 2.5, 7.0])
    out = rmse_and_bias(values, values.copy())
    assert out["rmse"] == pytest.approx(0.0)
    assert out["bias"] == pytest.approx(0.0)
    assert out["rmse_ci_hi"] == pytest.approx(0.0)
    assert out["n"] == 5


def test_rmse_and_bias_on_a_known_constant_offset():
    """A constant +2 offset must give bias exactly +2 and RMSE exactly 2 -- the known-bad arm."""
    truth = np.array([1.0, 4.0, 9.0, 2.5, 7.0])
    out = rmse_and_bias(truth, truth + 2.0)
    assert out["bias"] == pytest.approx(2.0)
    assert out["rmse"] == pytest.approx(2.0)
    # | Sign matters: a estimator running LOW must report a negative bias, not just a magnitude.
    assert rmse_and_bias(truth, truth - 2.0)["bias"] == pytest.approx(-2.0)


def test_rmse_is_not_fooled_by_cancelling_errors():
    """Symmetric errors cancel in the bias but must NOT cancel in the RMSE."""
    truth = np.array([0.0, 0.0, 0.0, 0.0])
    out = rmse_and_bias(truth, np.array([3.0, -3.0, 3.0, -3.0]))
    assert out["bias"] == pytest.approx(0.0)
    assert out["rmse"] == pytest.approx(3.0)


def test_rmse_and_bias_refuses_too_few_points():
    """Below 3 usable pairs every statistic is NaN rather than a confident small-sample number."""
    out = rmse_and_bias(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(out["rmse"]) and np.isnan(out["bias"])


# ===========================================================================
# 2. Spearman-Brown and the ceiling
# ===========================================================================


def test_spearman_brown_matches_the_published_formula():
    """Doubling a length takes r=0.5 to 0.667; halving it takes r=0.5 back to 1/3."""
    assert spearman_brown(0.5, 2.0) == pytest.approx(2 * 0.5 / (1 + 0.5))
    assert spearman_brown(0.5, 0.5) == pytest.approx(0.5 * 0.5 / (1 - 0.5 * 0.5))
    # | A perfectly reliable measure stays perfect at any length.
    assert spearman_brown(1.0, 3.0) == pytest.approx(1.0)


def test_spearman_brown_rejects_an_undefined_case():
    """A reliability that drives the denominator to zero yields NaN, not an infinity."""
    assert np.isnan(spearman_brown(float("nan"), 2.0))
    assert np.isnan(spearman_brown(-1.0, 2.0))


def test_ceiling_is_higher_for_the_person_level():
    """Averaging a person's episodes must raise the achievable correlation, never lower it."""
    episode = ceiling_for(0.4, "episode")
    person = ceiling_for(0.4, "person")
    assert episode == pytest.approx(np.sqrt(0.4))
    assert person > episode
    # | A useless estimate cannot support any correlation.
    assert np.isnan(ceiling_for(0.0, "episode"))


# ===========================================================================
# 3. Split-half reliability
# ===========================================================================


def test_reliability_is_one_when_every_episode_agrees():
    """Identical estimates within a person, differing between people: perfectly reliable."""
    rows = make_rows({"a": [1.0] * EPISODES_PER_PERSON,
                      "b": [5.0] * EPISODES_PER_PERSON,
                      "c": [9.0] * EPISODES_PER_PERSON,
                      "d": [3.0] * EPISODES_PER_PERSON})
    assert split_half_reliability(rows, COEFFICIENT) == pytest.approx(1.0, abs=1e-9)


def test_reliability_is_low_when_estimates_are_noise():
    """The known-bad direction: estimates unrelated to the person must not look reliable."""
    rng = np.random.default_rng(0)
    rows = make_rows({f"p{i}": list(rng.normal(size=EPISODES_PER_PERSON)) for i in range(12)})
    assert split_half_reliability(rows, COEFFICIENT) < 0.35


def test_reliability_rejects_an_unbalanced_design():
    """People with differing episode counts break the split-half, so it refuses rather than guesses."""
    rows = make_rows({"a": [1.0] * EPISODES_PER_PERSON, "b": [2.0] * (EPISODES_PER_PERSON - 1)})
    with pytest.raises(ValueError, match="differing episode counts"):
        split_half_reliability(rows, COEFFICIENT)


def test_reliability_works_at_a_smoke_sized_episode_count():
    """The count is DERIVED, so a 2-episode smoke reads rather than being refused.

    Hardcoding 6 meant the read-out could not be exercised on anything but the full study,
    which is exactly the run where a defect is most expensive to discover.
    """
    rows = make_rows({"a": [1.0, 1.0], "b": [5.0, 5.0], "c": [9.0, 9.0], "d": [3.0, 3.0]})
    assert split_half_reliability(rows, COEFFICIENT) == pytest.approx(1.0, abs=1e-9)


def test_ceiling_scales_with_the_episode_count_it_is_given():
    """More episodes behind an aggregate means a higher achievable correlation."""
    assert ceiling_for(0.4, "person", 6) > ceiling_for(0.4, "person", 2)
    # | Episode level ignores the count entirely -- the estimate is one episode either way.
    assert ceiling_for(0.4, "episode", 6) == pytest.approx(ceiling_for(0.4, "episode", 2))


# ===========================================================================
# 4. Spearman versus Pearson, the reason the headline statistic changed
# ===========================================================================


def test_spearman_survives_a_monotone_nonlinearity_that_pearson_does_not():
    """The case the headline choice rests on: clamping-style bending keeps the ordering."""
    x = np.linspace(-2.0, 2.0, 60)
    y = np.exp(3.0 * x)                            # monotone, strongly nonlinear
    assert bootstrap_spearman(x, y)["rho"] == pytest.approx(1.0)
    assert bootstrap_correlation(x, y)["r"] < 0.75


def test_spearman_reports_no_relationship_on_noise():
    """The known-bad arm: an interval straddling zero on unrelated variables."""
    rng = np.random.default_rng(1)
    out = bootstrap_spearman(rng.normal(size=80), rng.normal(size=80))
    assert out["ci_lo"] < 0 < out["ci_hi"]
    assert out["p"] > 0.05


def test_spearman_is_nan_without_spread():
    """A constant input has no ranking, so the statistic is NaN rather than silently 0."""
    out = bootstrap_spearman(np.ones(10), np.arange(10.0))
    assert np.isnan(out["rho"]) and out["n"] == 10


# ===========================================================================
# 5. Inputs and selection
# ===========================================================================


def test_select_filters_by_level():
    """Every row of the level, whatever its `stratum` label says."""
    rows = make_rows({"a": [1.0] * 2}, stratum=POOLED) + \
           make_rows({"b": [2.0] * 2}, stratum="other")
    assert len(select(rows, "episode")) == 4
    # | Known-bad direction: a level that appears nowhere must yield nothing, not everything.
    assert select(rows, "person") == []


def test_theta_ranges_reads_the_checkpoint_metadata(tmp_path: Path):
    """Ranges come from `hlp_param_config`, as max - min, for every fitted coefficient."""
    config = [{"name": name, "min": 0.0, "max": float(i + 2)}
              for i, name in enumerate(THETA_NAMES)]
    with io.open(tmp_path / "hlp_metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"hlp_param_config": config}, fh)
    ranges = theta_ranges(tmp_path)
    assert ranges[THETA_NAMES[0]] == pytest.approx(2.0)
    assert set(ranges) == set(THETA_NAMES)


def test_theta_ranges_fails_loudly_on_a_missing_coefficient(tmp_path: Path):
    """A metadata file that omits a fitted coefficient must raise, not default to a range."""
    with io.open(tmp_path / "hlp_metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"hlp_param_config": [{"name": "something_else", "min": 0.0, "max": 1.0}]}, fh)
    with pytest.raises(KeyError):
        theta_ranges(tmp_path)


def test_shared_limits_span_every_row():
    """Limits must cover every truth and estimate, so panels of one coefficient stay comparable."""
    rows = make_rows({"a": [1.0] * 2}, truth=1.0) + make_rows({"b": [9.0] * 2}, truth=9.0)
    low, high = shared_limits(rows, COEFFICIENT, "episode", 10.0)
    assert low < 1.0 and high > 9.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ===========================================================================
# 6. Cluster bootstrap -- the resampling unit is the agent, not the episode
# ===========================================================================


def test_cluster_bootstrap_widens_the_interval():
    """Repeats of the same truth carry no new information, so the clustered CI must be WIDER.

    This is the defect the clustered path exists to fix: with 6 episodes per agent sharing one
    true theta, resampling episodes treats 144 rows as 144 observations of the truth when there
    are only 24. The unclustered interval is therefore too narrow, and an unclustered run on
    this same input is the known-bad arm.
    """
    rng = np.random.default_rng(3)
    n_agents, per_agent = 12, 6
    truths = rng.normal(size=n_agents)
    x = np.repeat(truths, per_agent)
    y = x + rng.normal(scale=0.6, size=x.size)
    agents = np.repeat(np.arange(n_agents), per_agent)
    naive = bootstrap_spearman(x, y)
    clustered = bootstrap_spearman(x, y, clusters=agents)
    assert clustered["rho"] == pytest.approx(naive["rho"])        # point estimate unchanged
    naive_width = naive["ci_hi"] - naive["ci_lo"]
    clustered_width = clustered["ci_hi"] - clustered["ci_lo"]
    assert clustered_width > naive_width


def test_bootstrap_indices_draws_whole_clusters():
    """Every drawn index belongs to a cluster that was itself drawn, and only to such a one."""
    clusters = np.array([0, 0, 1, 1, 2, 2])
    rng = np.random.default_rng(0)
    drawn = bootstrap_indices(rng, clusters.size, clusters)
    assert drawn.size == clusters.size                            # equal-sized clusters
    # | A cluster is present either fully or not at all -- never a lone member of a pair.
    counts = {label: int(np.sum(clusters[drawn] == label)) for label in (0, 1, 2)}
    assert all(c % 2 == 0 for c in counts.values())


def test_bootstrap_indices_without_clusters_is_the_plain_bootstrap():
    """Passing None must reproduce the original row-resampling exactly, seed for seed."""
    expected = np.random.default_rng(7).choice(np.arange(20), 20)
    assert np.array_equal(bootstrap_indices(np.random.default_rng(7), 20, None), expected)


def test_rmse_and_bias_accepts_clusters_without_moving_the_point_estimate():
    """Clustering changes only the interval, never RMSE or bias themselves."""
    rng = np.random.default_rng(5)
    truth = np.repeat(rng.normal(size=8), 4)
    estimate = truth + 1.5
    agents = np.repeat(np.arange(8), 4)
    plain = rmse_and_bias(truth, estimate)
    clustered = rmse_and_bias(truth, estimate, clusters=agents)
    assert clustered["bias"] == pytest.approx(plain["bias"]) == pytest.approx(1.5)
    assert clustered["rmse"] == pytest.approx(plain["rmse"]) == pytest.approx(1.5)
