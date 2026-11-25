"""
Checkpoint-free gates on the model-comparison pipeline pieces (implementation.md Entry 35):
the chunk merge, the paired BIC table, the best-fit rule, the recovery study's variant-aware
column schema, and the 8-coefficient range lookup for the recovery read-out.

Every numeric check has a known-bad or boundary arm, so a test that could pass by
construction is shown not to.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from human_modeling.inference.model.choice import FIT_NAMES, HEURISTIC_BOUNDS, THETA_NAMES
from human_modeling.inference.scripts import plot_model_comparison as pmc
from human_modeling.inference.scripts.analyze_param_recovery import (coefficient_ranges,
                                                                        coefficients_in)
from human_modeling.inference.scripts.merge_chunk_csvs import merge_rows
from human_modeling.inference.scripts.run_synthetic_recovery import (aggregate_rows,
                                                                        episode_fields,
                                                                        fit_columns,
                                                                        parse_agent_range,
                                                                        truth_rows_for,
                                                                        write_outputs,
                                                                        write_recovery_rows)
from human_modeling.inference.variants import get_variant


# ===========================================================================
# 1. Merge
# ===========================================================================

def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_merge_concatenates_in_order_and_drops_repeated_keys(tmp_path: Path):
    a = _write(tmp_path / "a.csv", ["subject_id", "game_index", "x"], [["s", "1", "1.0"], ["s", "2", "2.0"]])
    b = _write(tmp_path / "b.csv", ["subject_id", "game_index", "x"], [["s", "2", "9.0"], ["t", "1", "3.0"]])
    header, rows = merge_rows([a, b], ["subject_id", "game_index"])
    assert header == ["subject_id", "game_index", "x"]
    assert [(r["subject_id"], r["game_index"], r["x"]) for r in rows] == [
        ("s", "1", "1.0"), ("s", "2", "2.0"), ("t", "1", "3.0")]      # first occurrence wins


def test_merge_refuses_a_header_mismatch(tmp_path: Path):
    a = _write(tmp_path / "a.csv", ["subject_id", "game_index", "x"], [["s", "1", "1.0"]])
    b = _write(tmp_path / "b.csv", ["subject_id", "game_index", "y"], [["t", "1", "1.0"]])
    with pytest.raises(ValueError):
        merge_rows([a, b], ["subject_id", "game_index"])


def test_merge_tolerates_an_empty_chunk(tmp_path: Path):
    a = _write(tmp_path / "a.csv", ["subject_id", "game_index"], [["s", "1"]])
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    header, rows = merge_rows([empty, a], ["subject_id", "game_index"])
    assert header == ["subject_id", "game_index"] and len(rows) == 1


# ===========================================================================
# 2. BIC table and the best-fit rule
# ===========================================================================

def _fit_file(path: Path, rows: list[tuple[str, int, float, int, int, int]]) -> None:
    """(subject, game, logL_mode, k, n_frames, no_progress) rows in the fit schema."""
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject_id", "game_index", "logL_mode",
                                           "n_free_params", "n_frames", "no_progress"])
        w.writeheader()
        for s, g, ll, k, n, np_ in rows:
            w.writerow({"subject_id": s, "game_index": g, "logL_mode": ll, "n_free_params": k,
                        "n_frames": n, "no_progress": np_})


def test_bic_is_schwarz():
    out = pmc.bic(np.array([-100.0]), np.array([4]), np.array([1000]))
    assert out[0] == pytest.approx(4 * np.log(1000) + 200.0)


def test_table_is_paired_over_shared_episodes_and_reports_the_dropped(tmp_path: Path):
    _fit_file(tmp_path / "fit_variant_B_no_heuristics.csv",
              [("s", 1, -100.0, 4, 1000, 0), ("s", 2, -120.0, 4, 1000, 0), ("t", 1, -80.0, 4, 500, 0)])
    _fit_file(tmp_path / "fit_variant_A_full.csv",
              [("s", 1, -90.0, 8, 1000, 0), ("t", 1, -79.0, 8, 500, 1)])
    table = pmc.episode_table(tmp_path, ["B_no_heuristics", "A_full"])
    assert table["keys"] == [("s", 1), ("t", 1)]
    assert table["dropped"] == 1
    assert table["k"] == {"B_no_heuristics": 4, "A_full": 8}
    np.testing.assert_allclose(table["nll"]["A_full"], [90.0, 79.0])
    assert table["no_progress"]["A_full"].sum() == 1
    # | Episode (s,1): A gains 10 nats for 4 more parameters at n=1000: dBIC = 4 ln 1000 - 20
    d = table["bic"]["A_full"][0] - table["bic"]["B_no_heuristics"][0]
    assert d == pytest.approx(4 * np.log(1000) - 20.0)


def test_table_refuses_a_pre_variant_schema(tmp_path: Path):
    with io.open(tmp_path / "fit_variant_B_no_heuristics.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject_id", "game_index", "logL_mean"])
        w.writeheader()
        w.writerow({"subject_id": "s", "game_index": 1, "logL_mean": -1.0})
    with pytest.raises(ValueError):
        pmc.episode_table(tmp_path, ["B_no_heuristics"])


def test_best_fit_counts_with_one_variant_is_all_ambiguous():
    table = {"bic": {"B": np.array([100.0, 90.0])}}
    assert pmc.best_fit_counts(table, ["B"]) == ({"B": 0}, 2)


def test_best_fit_counts_apply_the_ten_point_margin():
    table = {"bic": {"B": np.array([100.0, 100.0, 100.0]),
                     "A": np.array([80.0, 95.0, 120.0])}}
    counts, ambiguous = pmc.best_fit_counts(table, ["B", "A"])
    assert counts == {"B": 1, "A": 1} and ambiguous == 1     # 20 clear, 5 ambiguous, 20 clear


def test_paired_difference_recovers_a_known_shift_with_a_tight_interval():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 50, size=120)                          # large shared episode effect
    x = y - 3.0 + rng.normal(0, 0.5, size=120)               # variant better by 3
    clusters = np.repeat(np.arange(20), 6)
    out = pmc.paired_difference(x, y, clusters, seed=0)
    assert out["mean"] == pytest.approx(-3.0, abs=0.2)
    assert out["ci_lo"] < -3.0 < out["ci_hi"]
    assert out["ci_hi"] - out["ci_lo"] < 1.0                 # the episode effect cancelled
    assert out["share_negative"] > 0.99
    null = pmc.paired_difference(y, y.copy(), clusters, seed=0)      # known-bad: no difference
    assert null["mean"] == 0.0 and null["ci_lo"] == 0.0 == null["ci_hi"]


def test_nesting_violations_count_the_smaller_model_out_fitting_the_larger(tmp_path: Path):
    # | C in B: on (s,2) the smaller C beats B by 5 nats; on the others B is at least as good.
    _fit_file(tmp_path / "fit_variant_C_no_cognitive_cost.csv",
              [("s", 1, -110.0, 3, 1000, 0), ("s", 2, -100.0, 3, 1000, 0), ("t", 1, -90.0, 3, 500, 0)])
    _fit_file(tmp_path / "fit_variant_B_no_heuristics.csv",
              [("s", 1, -100.0, 4, 1000, 0), ("s", 2, -105.0, 4, 1000, 0), ("t", 1, -90.0, 4, 500, 0)])
    _fit_file(tmp_path / "fit_variant_A_full.csv",
              [("s", 1, -99.0, 8, 1000, 0), ("s", 2, -104.0, 8, 1000, 0), ("t", 1, -89.0, 8, 500, 0)])
    variants = ["C_no_cognitive_cost", "B_no_heuristics", "A_full"]
    table = pmc.episode_table(tmp_path, variants)
    out = pmc.nesting_violations(table, variants)
    assert out["B_no_heuristics"] == {"smaller": "C_no_cognitive_cost", "n_violations": 1,
                                      "worst_shortfall_nats": pytest.approx(5.0)}
    # | A' is not reported, so A is checked against B, which it contains through A'.
    assert out["A_full"] == {"smaller": "B_no_heuristics", "n_violations": 0,
                             "worst_shortfall_nats": 0.0}
    rows = pmc.summary_rows(table, variants, "B_no_heuristics", seed=0)
    b = next(r for r in rows if r["variant"] == "B_no_heuristics")
    assert b["nesting_violations"] == 1 and b["nested_in"] == "C_no_cognitive_cost"
    assert "violated on 1 episodes, worst shortfall 5.00 nats" in pmc.format_summary(rows, 0)
    # | Known-good: an exact tie is not a violation.
    assert pmc.nesting_violations({"nll": {"C_no_cognitive_cost": np.array([1.0]),
                                           "B_no_heuristics": np.array([1.0])}},
                                  ["C_no_cognitive_cost", "B_no_heuristics"])["B_no_heuristics"]["n_violations"] == 0


def test_reported_pairs_follow_the_chain_through_an_unreported_link():
    from human_modeling.inference.variants import NESTED_PAIRS
    full = ["A_full", "C_no_cognitive_cost", "A_no_mean_distance", "B_no_heuristics"]
    assert pmc.reported_nested_pairs(full) == list(NESTED_PAIRS)       # order-independent
    assert pmc.reported_nested_pairs(["C_no_cognitive_cost", "B_no_heuristics", "A_full"]) == [
        ("C_no_cognitive_cost", "B_no_heuristics"), ("B_no_heuristics", "A_full")]
    assert pmc.reported_nested_pairs(["C_no_cognitive_cost", "A_full"]) == [
        ("C_no_cognitive_cost", "A_full")]
    assert pmc.reported_nested_pairs(["B_no_heuristics"]) == []         # nothing to pair


def test_pair_counts_split_at_the_ten_point_margin():
    d = np.array([-20.0, -10.0, -9.9, 0.0, 9.9, 10.0, 30.0])     # both edges of the band
    assert pmc.pair_counts(d) == {"favours_larger": 2, "ambiguous": 3, "favours_smaller": 2}


def test_baseline_pairs_subtract_the_smallest_reported_model():
    assert pmc.baseline_pairs(["A_full", "B_no_heuristics", "C_no_cognitive_cost"]) == [
        ("C_no_cognitive_cost", "A_full"), ("C_no_cognitive_cost", "B_no_heuristics")]
    assert pmc.baseline_pairs(["B_no_heuristics", "A_full"]) == [("B_no_heuristics", "A_full")]
    assert pmc.baseline_pairs(["B_no_heuristics"]) == []                  # nothing to subtract


def test_figures_include_the_delta_against_c_above_the_counts(tmp_path: Path):
    for name, k, shift in (("C_no_cognitive_cost", 3, 0.0), ("B_no_heuristics", 4, 20.0),
                           ("A_full", 8, 60.0)):
        _fit_file(tmp_path / f"fit_variant_{name}.csv",
                  [(s, g, -1000.0 + shift + g, k, 1000, 0) for s in "stuv" for g in (1, 2)])
    variants = ["C_no_cognitive_cost", "B_no_heuristics", "A_full"]
    table = pmc.episode_table(tmp_path, variants)
    names = {p.name for p in pmc.draw_figures(table, variants, "B_no_heuristics", tmp_path / "plots")}
    assert "delta_bic_vs_C_no_cognitive_cost_and_best_fit.png" in names
    assert "delta_bic_vs_C_no_cognitive_cost_and_best_fit_horizontal.png" in names  # side by side
    assert not any("_vs_" in n and n.endswith("_vs_C_no_cognitive_cost.png") for n in names)
    assert all((tmp_path / "plots" / n).stat().st_size > 0 for n in names)


def test_summary_rows_and_text_carry_the_stall_count(tmp_path: Path):
    _fit_file(tmp_path / "fit_variant_B_no_heuristics.csv",
              [(s, g, -100.0 - g, 4, 1000, 0) for s in "stuv" for g in (1, 2, 3)])
    _fit_file(tmp_path / "fit_variant_A_full.csv",
              [(s, g, -95.0 - g, 8, 1000, int(s == "s")) for s in "stuv" for g in (1, 2, 3)])
    table = pmc.episode_table(tmp_path, ["B_no_heuristics", "A_full"])
    rows = pmc.summary_rows(table, ["B_no_heuristics", "A_full"], "B_no_heuristics", seed=0)
    a = next(r for r in rows if r["variant"] == "A_full")
    assert a["n_stalled_fits"] == 3 and a["n_free_params"] == 8
    assert a["delta_bic_vs_reference_mean"] == pytest.approx(4 * np.log(1000) - 10.0)
    text = pmc.format_summary(rows, table["dropped"])
    assert "WARNING: 3 fits" in text and "A_full" in text
    assert "NOTE:" not in text                     # a file with no warm_start reads as midpoint


def test_no_progress_from_a_smaller_model_is_a_note_not_a_warning(tmp_path: Path):
    # | [boundary-start] B kept C's solution on two episodes (the Entry 35 addendum 3 case)
    # | and stalled from the midpoint on one: only that one is called NOT an estimate.
    _fit_file(tmp_path / "fit_variant_C_no_cognitive_cost.csv",
              [(s, 1, -100.0, 3, 1000, 0) for s in "stuv"])
    with io.open(tmp_path / "fit_variant_B_no_heuristics.csv", "w", encoding="utf-8",
                 newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject_id", "game_index", "logL_mode",
                                           "n_free_params", "n_frames", "no_progress",
                                           "warm_start"])
        w.writeheader()
        for s, np_, start in (("s", 1, "C_no_cognitive_cost"), ("t", 1, "C_no_cognitive_cost"),
                              ("u", 1, "midpoint"), ("v", 0, "midpoint")):
            w.writerow({"subject_id": s, "game_index": 1, "logL_mode": -100.0, "n_free_params": 4,
                        "n_frames": 1000, "no_progress": np_, "warm_start": start})
    variants = ["C_no_cognitive_cost", "B_no_heuristics"]
    table = pmc.episode_table(tmp_path, variants)
    rows = pmc.summary_rows(table, variants, "C_no_cognitive_cost", seed=0)
    b = next(r for r in rows if r["variant"] == "B_no_heuristics")
    assert b["n_stalled_fits"] == 3 and b["n_no_progress_from_smaller"] == 2
    text = pmc.format_summary(rows, table["dropped"])
    assert "NOTE: 2 fits never improved on the smaller model's solution" in text
    assert "WARNING: 1 fits never improved on their midpoint start" in text


# ===========================================================================
# 3. Recovery schema for a heuristic variant
# ===========================================================================

def test_fit_columns_follow_the_human_fit_schema():
    a = get_variant("A_no_mean_distance")
    cols = fit_columns(a)
    assert cols["hlp_collision_penalty"] == "hlp_collision_penalty"
    assert cols["closest_vehicle_distance"] == "w_closest_vehicle_distance"
    assert "mean_vehicle_distance" not in cols
    paired = fit_columns(get_variant("B_no_heuristics"), "B_no_heuristics_")
    assert paired["hlp_collision_penalty"] == "B_no_heuristics_hlp_collision_penalty"


def test_truths_for_a_full_cover_all_eight_and_respect_the_weight_bounds():
    a = get_variant("A_full")
    bound = np.array([HEURISTIC_BOUNDS[n] for n in FIT_NAMES[4:]])
    lo = np.concatenate([np.zeros(4), -bound])
    hi = np.concatenate([np.array([10.0, 0.15, 6.0, 1.5]), bound])
    rows = truth_rows_for(a, lo, hi, 50, study_seed=1)
    assert len(rows) == 50 and set(rows[0]) == {"agent_index", *(f"true_{n}" for n in FIT_NAMES)}
    w = np.array([[r[f"true_{n}"] for n in FIT_NAMES[4:]] for r in rows])
    assert np.all(np.abs(w) <= bound) and np.any(w < 0) and np.any(w > 0)


def test_episode_fields_hold_both_arms_without_collisions():
    fields = episode_fields(get_variant("A_full"), get_variant("B_no_heuristics"))
    assert len(fields) == len(set(fields))
    assert "w_mean_vehicle_distance" in fields and "B_no_heuristics_hlp_collision_penalty" in fields
    assert "B_no_heuristics_w_lead_vehicle_inverse_ttc" not in fields    # B has no weights
    assert "true_lead_vehicle_inverse_ttc" in fields


def test_an_extra_arm_adds_its_own_columns_and_leaves_the_rest_in_order():
    # | [extra-arms] Entry 35 addendum 6: C beside A and B, with a seconds column of its own.
    a, b, c = (get_variant(n) for n in ("A_full", "B_no_heuristics", "C_no_cognitive_cost"))
    base = episode_fields(a, b)
    fields = episode_fields(a, b, [c])
    assert fields[:len(base)] == base                                  # A and B unchanged
    assert len(fields) == len(set(fields))
    assert {"C_no_cognitive_cost_logL_mode", "C_no_cognitive_cost_hlp_collision_penalty",
            "C_no_cognitive_cost_seconds"} <= set(fields)
    assert "C_no_cognitive_cost_hlp_cognitive_cost_coef" not in fields   # C pins it


def test_outputs_hold_back_done_until_every_episode_has_the_extra_arm(tmp_path: Path):
    a, b, c = (get_variant(n) for n in ("A_full", "B_no_heuristics", "C_no_cognitive_cost"))
    truth = {f"true_{n}": 0.1 * (i + 1) for i, n in enumerate(FIT_NAMES)}
    rows = []
    for m in range(2):
        row = {"agent_index": 0, "map_index": m, "map_id": f"m{m}", "no_progress": 0, **truth}
        for arm, prefix in ((a, ""), (b, "B_no_heuristics_"), (c, "C_no_cognitive_cost_")):
            row.update({col: 0.5 for col in fit_columns(arm, prefix).values()})
            row[f"{prefix}logL_mode"] = -100.0
        rows.append(row)
    rows[1]["C_no_cognitive_cost_logL_mode"] = ""                     # C not yet added here
    truth_rows = [{"agent_index": 0, **truth}]
    assert not write_outputs(tmp_path, rows, truth_rows, a, b, 1, 2, 0, [c])
    assert not (tmp_path / "DONE").exists()
    with io.open(tmp_path / "recovery_rows_C_no_cognitive_cost_mean.csv", encoding="utf-8") as fh:
        c_rows = list(csv.DictReader(fh))
    assert [r["level"] for r in c_rows] == ["episode", "person"]      # only the episode with C
    with io.open(tmp_path / "recovery_rows_B_no_heuristics_mean.csv", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 3                      # B untouched: both episodes
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["episodes_missing_extra_arm"] == {"C_no_cognitive_cost": 1}
    rows[1]["C_no_cognitive_cost_logL_mode"] = -101.0
    assert write_outputs(tmp_path, rows, truth_rows, a, b, 1, 2, 0, [c])
    assert (tmp_path / "DONE").exists()


class _FakeFits:
    """Stand-in for `fit_variant` / `log_likelihood`: each (arm, start) ends at a set logL,
    carried in the peak vector's first entry, and every call is recorded in order."""

    def __init__(self, logl: dict[tuple[str, str], float]):
        self.logl, self.calls = logl, []

    def fit_variant(self, stack, inp, arm, start_full=None, start_label="midpoint",
                    alternative_starts=()):
        from human_modeling.inference.scripts.run_human_inference import POSTERIOR_TAGS
        tried = [start_label] + [label for label, _ in alternative_starts]
        self.calls.append((arm.name, tuple(tried),
                           None if start_full is None else float(start_full[0])))
        start = max(tried, key=lambda s: self.logl[(arm.name, s)])
        mode = np.full(len(arm.full_names), 0.1)
        mode[0] = self.logl[(arm.name, start)]
        est = {f"{n}_{t}": 0.5 for n in THETA_NAMES for t in ("mean",) + POSTERIOR_TAGS}
        return est, {"iters": 3, "no_progress": False, "start": start,
                     "at_bound": [False] * arm.n_free}, mode

    @staticmethod
    def log_likelihood(stack, inp, vec):
        return float(vec[0])


def _patch(monkeypatch, fake: _FakeFits) -> None:
    from human_modeling.inference.scripts import run_synthetic_recovery as rsr
    monkeypatch.setattr(rsr, "fit_variant", fake.fit_variant)
    monkeypatch.setattr(rsr, "log_likelihood", fake.log_likelihood)
    monkeypatch.setattr(rsr, "log", lambda msg: None)


def test_a_new_episode_is_fitted_in_the_human_chain_order(monkeypatch):
    from human_modeling.inference.scripts.run_synthetic_recovery import fit_new_episode
    a, b, c = (get_variant(n) for n in ("A_full", "B_no_heuristics", "C_no_cognitive_cost"))
    fake = _FakeFits({("C_no_cognitive_cost", "midpoint"): -120.0,
                      ("B_no_heuristics", "midpoint"): -115.0,
                      ("B_no_heuristics", "C_no_cognitive_cost"): -110.0,
                      ("A_full", "B_no_heuristics"): -90.0})
    _patch(monkeypatch, fake)
    row: dict = {}
    fit_new_episode(row, None, None, a, b, [c])
    # | C first; B tries the midpoint AND C; A starts from B's winning solution (logL -110).
    assert fake.calls == [("C_no_cognitive_cost", ("midpoint",), None),
                          ("B_no_heuristics", ("midpoint", "C_no_cognitive_cost"), None),
                          ("A_full", ("B_no_heuristics",), -110.0)]
    assert row["B_no_heuristics_start"] == "C_no_cognitive_cost"
    assert row["B_no_heuristics_logL_mode"] == -110.0 and row["logL_mode"] == -90.0


def test_recovery_rows_for_a_heuristic_variant_feed_the_read_out(tmp_path: Path):
    a = get_variant("A_full")
    cols = fit_columns(a)
    truth = {f"true_{n}": float(i) for i, n in enumerate(FIT_NAMES)}
    episode_rows = [{"agent_index": 0, "map_index": m, "map_id": "abc",
                     **truth, **{c: float(i) + 0.5 for i, c in enumerate(cols.values())}}
                    for m in range(2)]
    aggregate = aggregate_rows(episode_rows, [{"agent_index": 0, **truth}], cols)
    assert aggregate[0]["mean_lead_vehicle_inverse_ttc"] == pytest.approx(4.5)
    path = tmp_path / "recovery_rows_mean.csv"
    write_recovery_rows(path, episode_rows, aggregate, "mean", cols)
    with io.open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert coefficients_in(rows) == FIT_NAMES
    assert [r["level"] for r in rows] == ["episode", "episode", "person"]


def test_coefficient_ranges_add_the_weight_box(tmp_path: Path):
    config = [{"name": name, "min": 0.0, "max": float(i + 2)} for i, name in enumerate(THETA_NAMES)]
    with io.open(tmp_path / "hlp_metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"hlp_param_config": config}, fh)
    ranges = coefficient_ranges(tmp_path, FIT_NAMES)
    assert list(ranges) == list(FIT_NAMES)
    assert ranges["hlp_collision_penalty"] == pytest.approx(2.0)
    assert ranges["closest_vehicle_distance"] == pytest.approx(2 * HEURISTIC_BOUNDS["closest_vehicle_distance"])


def test_parse_agent_range():
    assert parse_agent_range(None, 100) == (0, 100)
    assert parse_agent_range("10:20", 100) == (10, 20)
    assert parse_agent_range("90:200", 100) == (90, 100)
    with pytest.raises(ValueError):
        parse_agent_range("5:5", 100)
