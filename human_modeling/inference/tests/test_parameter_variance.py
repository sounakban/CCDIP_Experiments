"""
Gates on the between/within variance decomposition of the fitted parameters (implementation.md
Entry 35 addendum 5): the components against a hand-computed case, the share being the ICC(1,1)
ratio of those components, the route adjustment, and the refusal of an episode with no route. Each has a known-bad arm where one is meaningful.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest

from human_modeling.inference.scripts import parameter_variance as pv
from human_modeling.inference.scripts.parameter_variance import variance_components


def test_components_match_a_hand_computed_balanced_case():
    # | Three people, two episodes each: means 0, 2, 4 and every episode +-1 from its mean.
    # | MSW = 6 * 1 / (6 - 3) = 2; MSB = 2 * (4 + 0 + 4) / 2 = 8; between = (8 - 2) / 2 = 3.
    values = np.array([-1.0, 1.0, 1.0, 3.0, 3.0, 5.0])
    people = np.array(["a", "a", "b", "b", "c", "c"])
    comp = variance_components(values, people)
    assert comp["within"] == pytest.approx(2.0)
    assert comp["between"] == pytest.approx(3.0)
    assert comp["share"] == pytest.approx(3.0 / 5.0)
    assert comp["between"] != pytest.approx(np.var([0.0, 2.0, 4.0], ddof=1))   # known-bad: naive


def test_share_is_the_icc_ratio_of_the_components():
    # | share = (MSB - MSW) / (MSB + (n0 - 1) MSW) and between / (between + within) are the
    # | same expression rearranged, on an unbalanced design too.
    rng = np.random.default_rng(3)
    people = np.repeat(np.arange(12), rng.integers(4, 7, 12))
    values = rng.normal(size=len(people)) + people * 0.2
    maps = rng.integers(0, 6, len(people))
    for m in (None, maps):
        comp = variance_components(values, people, m)
        assert comp["share"] == pytest.approx(comp["between"] / (comp["between"] + comp["within"]))


def test_route_adjustment_removes_a_pure_route_effect():
    # | Everyone identical, the route alone moves the value: raw within variance is all route,
    # | and removing route means leaves nothing. The raw figure is the known-bad arm.
    people = np.repeat(np.arange(8), 6)
    routes = np.tile(np.arange(6), 8)
    values = routes * 1.5
    assert variance_components(values, people)["within"] > 1.0
    assert variance_components(values, people, routes)["within"] == pytest.approx(0.0, abs=1e-12)


def _write(path: Path, header: list[str], rows: list[dict]) -> Path:
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return path


def test_an_episode_without_a_route_is_refused(tmp_path: Path, monkeypatch):
    fit = _write(tmp_path / "fit.csv", ["subject_id", "game_index"],
                 [{"subject_id": "s", "game_index": 1}, {"subject_id": "s", "game_index": 2}])
    monkeypatch.setattr(pv, "human_routes", lambda data_path: {("s", 1): "m1"})   # game 2 has none
    with pytest.raises(ValueError, match="no route"):
        pv.read_episodes(fit)


def test_decompose_reads_every_free_coefficient_of_the_variant(monkeypatch):
    monkeypatch.setattr(pv, "N_BOOT", 40)                     # the gate is the wiring, not the CI
    rng = np.random.default_rng(0)
    columns = pv.fit_columns(pv.VARIANTS["A_full"])
    rows = [{"subject_id": f"p{i // 6}", **{c: str(rng.normal()) for c in columns.values()}}
            for i in range(60)]
    routes = np.tile(np.arange(6), 10)
    out = pv.decompose(rows, routes, "A_full")
    assert [r["coefficient"] for r in out] == list(columns)
    assert all(r["n_people"] == 10 and r["n_episodes"] == 60 for r in out)
    assert all(r["share_ci_lo"] <= r["share_ci_hi"] for r in out)
