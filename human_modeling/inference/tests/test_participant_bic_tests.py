"""
Gates on the per-participant BIC tests the paper reports (implementation.md Entry 35
addendum 4): averaging within participants before testing, the sign convention, and Holm's
adjustment across pairs. Each has a known-bad arm, so a test that could pass by construction
is shown not to.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from statsmodels.stats.multitest import multipletests

from human_modeling.inference.scripts import participant_bic_tests as pbt


def _fit_file(path: Path, rows: list[tuple[str, int, float]]) -> None:
    """(subject, game, logL_mode) rows; k and n held equal across variants so BIC = -2 logL + const."""
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject_id", "game_index", "logL_mode",
                                           "n_free_params", "n_frames", "no_progress"])
        w.writeheader()
        for s, g, ll in rows:
            w.writerow({"subject_id": s, "game_index": g, "logL_mode": ll, "n_free_params": 4,
                        "n_frames": 1000, "no_progress": 0})


def test_differences_are_averaged_within_participants_first(tmp_path: Path):
    # | BIC(B) - BIC(A) = 2 (logL_A - logL_B): participant s has 10 and 30 (mean 20), t has 50.
    # | The participant mean is 35; the per-episode mean, the known-bad arm, would be 30.
    _fit_file(tmp_path / "fit_variant_B_no_heuristics.csv",
              [("s", 1, -100.0), ("s", 2, -100.0), ("t", 1, -100.0)])
    _fit_file(tmp_path / "fit_variant_A_full.csv",
              [("s", 1, -95.0), ("s", 2, -85.0), ("t", 1, -75.0)])
    (row,) = pbt.run_tests(tmp_path, ["B_no_heuristics", "A_full"])
    assert (row["smaller"], row["larger"]) == ("B_no_heuristics", "A_full")
    assert row["mean_delta_bic"] == pytest.approx(35.0)
    assert row["mean_delta_bic"] != pytest.approx(30.0)
    assert row["n_participants"] == 2 and row["n_favouring_larger"] == 2   # positive favours A


def test_holm_adjusts_across_every_pair(tmp_path: Path):
    people = [f"p{i}" for i in range(12)]
    _fit_file(tmp_path / "fit_variant_C_no_cognitive_cost.csv", [(p, 1, -100.0) for p in people])
    _fit_file(tmp_path / "fit_variant_B_no_heuristics.csv",
              [(p, 1, -100.0 + (1.0 if i % 3 else -0.5)) for i, p in enumerate(people)])
    _fit_file(tmp_path / "fit_variant_A_full.csv",
              [(p, 1, -90.0 + 0.1 * i) for i, p in enumerate(people)])
    rows = pbt.run_tests(tmp_path, ["A_full", "C_no_cognitive_cost", "B_no_heuristics"])
    assert [(r["smaller"], r["larger"]) for r in rows] == [
        ("C_no_cognitive_cost", "B_no_heuristics"), ("C_no_cognitive_cost", "A_full"),
        ("B_no_heuristics", "A_full")]                                    # FIT_ORDER pairs
    raw = [r["p"] for r in rows]
    assert [r["p_holm"] for r in rows] == pytest.approx(list(multipletests(raw, method="holm")[1]))
    smallest = min(range(3), key=lambda i: raw[i])
    assert rows[smallest]["p_holm"] == pytest.approx(min(1.0, 3 * raw[smallest]))
    assert rows[smallest]["p_holm"] > raw[smallest]                      # known-bad: unadjusted
