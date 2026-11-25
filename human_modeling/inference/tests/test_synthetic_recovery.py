"""
Checkpoint-free suite for `scripts/run_synthetic_recovery.py` (implementation.md Entry 31).

Nothing here loads a model or simulates: the parts that need a GPU are exercised by the
local smoke the sbatch runs before it scales up. What IS covered is everything that decides
whether the study is reproducible and whether its output can be read -- seeding, the truth
draw, aggregation, and the hand-off into `analyze_param_recovery.py`'s input schema. A silent
defect in any of those wastes a cluster window, which is exactly the case for testing them
without one.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from human_modeling.inference.model.choice import THETA_NAMES
from human_modeling.inference.scripts.analyze_param_recovery import POOLED, select
from human_modeling.inference.scripts.run_synthetic_recovery import (
    aggregate_rows, agent_truth, load_or_extract_maps, stable_seed, write_recovery_rows,
)

LO = np.array([0.0, 0.0, 0.0, 0.0])
HI = np.array([10.0, 0.15, 6.0, 1.5])
STUDY_SEED = 20260921


# ===========================================================================
# 1. Seeding and the truth draw
# ===========================================================================


def test_stable_seed_is_reproducible_and_distinguishing():
    """The same inputs give the same seed; different inputs do not collide."""
    assert stable_seed(1, 2, 3) == stable_seed(1, 2, 3)
    assert stable_seed(1, 2, 3) != stable_seed(1, 2, 4)
    # | Argument ORDER must matter, or (agent 3, map 7) and (agent 7, map 3) share a rollout.
    assert stable_seed(3, 7) != stable_seed(7, 3)
    assert 0 <= stable_seed(99, 99) < 2 ** 32


def test_agent_truth_is_reproducible_per_agent():
    """Agent i's truth must not depend on how many agents ran before it.

    This is what makes a resumed run safe: after a walltime kill, agent 57 is re-derived from
    its own index rather than from a generator that has advanced a different number of times.
    """
    first = agent_truth(STUDY_SEED, 57, LO, HI)
    assert np.array_equal(first, agent_truth(STUDY_SEED, 57, LO, HI))
    assert not np.array_equal(first, agent_truth(STUDY_SEED, 58, LO, HI))
    assert not np.array_equal(first, agent_truth(STUDY_SEED + 1, 57, LO, HI))


def test_agent_truths_lie_in_range_and_spread_across_it():
    """Every draw is inside its bound, and 200 draws cover the range rather than clustering.

    The coverage half is the point of Entry 31's design -- the study it replaces drew truths
    bimodally, which made its correlations measure a two-group separation.
    """
    draws = np.array([agent_truth(STUDY_SEED, i, LO, HI) for i in range(200)])   # (200, 4)
    assert np.all(draws >= LO) and np.all(draws <= HI)
    fractions = (draws - LO) / (HI - LO)
    # | A uniform draw puts roughly a quarter of its mass in each quarter of the range; a
    # | bimodal one leaves the middle two empty, which is the failure being excluded.
    for column in range(len(THETA_NAMES)):
        counts, _ = np.histogram(fractions[:, column], bins=4, range=(0.0, 1.0))
        assert counts.min() >= 25, f"coefficient {column} left a quarter of its range sparse"


def test_agent_truths_are_independent_across_coefficients():
    """Independence is what makes the confusion matrix readable, so it is checked, not assumed."""
    draws = np.array([agent_truth(STUDY_SEED, i, LO, HI) for i in range(400)])
    correlations = np.corrcoef(draws, rowvar=False)
    off_diagonal = correlations[~np.eye(len(THETA_NAMES), dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.2


# ===========================================================================
# 2. Aggregation
# ===========================================================================


def make_episode_rows(values_by_agent: dict[int, list[float]],
                      coefficient: str = THETA_NAMES[0]) -> list[dict[str, object]]:
    """Episode-fit rows carrying one varying coefficient, the rest held at zero."""
    rows: list[dict[str, object]] = []
    for agent_index, values in values_by_agent.items():
        for map_index, value in enumerate(values):
            row: dict[str, object] = {"agent_index": agent_index, "map_index": map_index,
                                      "map_id": f"m{map_index}"}
            for name in THETA_NAMES:
                row[name] = value if name == coefficient else 0.0
                row[f"true_{name}"] = 1.0 if name == coefficient else 0.0
            rows.append(row)
    return rows


def make_truth_rows(agent_indices: list[int],
                    coefficient: str = THETA_NAMES[0]) -> list[dict[str, object]]:
    """Truth rows matching :func:`make_episode_rows`."""
    return [{"agent_index": i,
             **{f"true_{n}": (1.0 if n == coefficient else 0.0) for n in THETA_NAMES}}
            for i in agent_indices]


def test_aggregate_computes_mean_median_and_signed_error():
    """Mean and median differ on a skewed set, and the error keeps its sign."""
    name = THETA_NAMES[0]
    rows = make_episode_rows({0: [1.0, 1.0, 1.0, 1.0, 1.0, 7.0]})
    out = aggregate_rows(rows, make_truth_rows([0]))[0]
    assert out["n_successful_episodes"] == 6
    assert out[f"mean_{name}"] == pytest.approx(2.0)
    assert out[f"median_{name}"] == pytest.approx(1.0)
    # | Truth is 1.0, so the mean runs high and the median is exact.
    assert out[f"mean_{name}_error"] == pytest.approx(1.0)
    assert out[f"median_{name}_error"] == pytest.approx(0.0)


def test_aggregate_skips_agents_with_no_successful_fits():
    """An agent whose every episode failed must be absent, not present with a blank estimate."""
    rows = make_episode_rows({0: [1.0, 1.0]})
    out = aggregate_rows(rows, make_truth_rows([0, 1]))
    assert [r["agent_index"] for r in out] == [0]


def test_aggregate_counts_partial_agents_honestly():
    """A partially completed agent reports how many episodes it actually has."""
    out = aggregate_rows(make_episode_rows({0: [1.0, 2.0, 3.0]}), make_truth_rows([0]))
    assert out[0]["n_successful_episodes"] == 3


# ===========================================================================
# 3. The hand-off into the analysis schema
# ===========================================================================


def test_recovery_rows_round_trip_into_the_analysis_schema(tmp_path: Path):
    """What the study writes is what `analyze_param_recovery.py` reads -- checked by reading it.

    The two scripts are coupled only through this file, so a column rename in either would
    otherwise surface as an empty read-out at the end of a cluster run.
    """
    name = THETA_NAMES[0]
    episode_rows = make_episode_rows({0: [1.0, 3.0], 1: [5.0, 7.0]})
    aggregate = aggregate_rows(episode_rows, make_truth_rows([0, 1]))
    path = tmp_path / "recovery_rows_mean.csv"
    write_recovery_rows(path, episode_rows, aggregate, "mean")

    with io.open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(select(rows, "episode")) == 4
    assert len(select(rows, "person")) == 2
    # | No strata in this design: every row carries the pooled label.
    assert {r["stratum"] for r in rows} == {POOLED}
    person_rows = select(rows, "person")
    assert float(person_rows[0][f"fit_{name}"]) == pytest.approx(2.0)
    assert float(person_rows[0][f"true_{name}"]) == pytest.approx(1.0)


def test_recovery_rows_median_variant_differs_from_the_mean_variant(tmp_path: Path):
    """The known-bad direction: if `how` were ignored, both files would be identical."""
    name = THETA_NAMES[0]
    episode_rows = make_episode_rows({0: [1.0, 1.0, 1.0, 1.0, 1.0, 7.0]})
    aggregate = aggregate_rows(episode_rows, make_truth_rows([0]))
    fits = {}
    for how in ("mean", "median"):
        path = tmp_path / f"recovery_rows_{how}.csv"
        write_recovery_rows(path, episode_rows, aggregate, how)
        with io.open(path, encoding="utf-8") as fh:
            person = [r for r in csv.DictReader(fh) if r["level"] == "person"][0]
        fits[how] = float(person[f"fit_{name}"])
    assert fits["mean"] == pytest.approx(2.0)
    assert fits["median"] == pytest.approx(1.0)


# ===========================================================================
# 4. Map caching
# ===========================================================================


def test_cached_maps_are_reused_without_touching_the_source(tmp_path: Path):
    """With `maps.json` present, the 900 MB source is never opened -- a nonexistent path proves it."""
    cached = [{"map_id": "aaaa1111", "map": {"base": ["..."]}, "n_episodes": 39,
               "first_subject": "s1", "first_game_index": 1},
              {"map_id": "bbbb2222", "map": {"base": ["..."]}, "n_episodes": 38,
               "first_subject": "s2", "first_game_index": 1}]
    with io.open(tmp_path / "maps.json", "w", encoding="utf-8") as fh:
        json.dump(cached, fh)
    loaded = load_or_extract_maps(tmp_path, Path("no/such/file.json"), 2)
    assert [m["map_id"] for m in loaded] == ["aaaa1111", "bbbb2222"]


def test_cached_maps_of_the_wrong_size_are_refused(tmp_path: Path):
    """A cache built for a 2-map smoke must not silently serve a 6-map study."""
    with io.open(tmp_path / "maps.json", "w", encoding="utf-8") as fh:
        json.dump([{"map_id": "aaaa1111", "map": {}, "n_episodes": 1}], fh)
    with pytest.raises(ValueError, match="delete it to re-extract"):
        load_or_extract_maps(tmp_path, Path("no/such/file.json"), 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
