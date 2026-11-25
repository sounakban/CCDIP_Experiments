"""
Checkpoint-free gates on `estimation/validate.py`'s resume bookkeeping and row parsing
(Entry 35 addendum 2): rows already on disk are recognised by the fit file's identity, the
seed a task carries follows the fit-file INDEX rather than the position in the pending list,
and the weight columns are read under their `w_<name>` spelling only.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from human_modeling.inference.estimation.validate import fit_row_parameters, read_done_rows
from human_modeling.inference.model.choice import HEURISTIC_NAMES, THETA_NAMES


def test_done_rows_are_keyed_like_the_fit_file(tmp_path: Path):
    out = tmp_path / "validate_simulated_x.csv"
    assert read_done_rows(out) == {}                       # absent
    out.write_text("", encoding="utf-8")
    assert read_done_rows(out) == {}                       # header-only-or-empty after a kill
    with io.open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject_id", "game_index", "collisions"])
        w.writeheader()
        w.writerow({"subject_id": "s", "game_index": 2, "collisions": 3.0})
    done = read_done_rows(out)
    assert set(done) == {("s", 2)} and done[("s", 2)]["collisions"] == "3.0"


def test_fit_row_parameters_read_the_w_columns_and_skip_blanks():
    row = {n: str(i + 1.0) for i, n in enumerate(THETA_NAMES)}
    theta, w = fit_row_parameters(row)
    assert theta == {n: i + 1.0 for i, n in enumerate(THETA_NAMES)} and w == {}   # B-style row
    row[f"w_{HEURISTIC_NAMES[0]}"] = "0.5"
    row[HEURISTIC_NAMES[1]] = "-0.25"                     # a bare name is not a weight column
    row[f"w_{HEURISTIC_NAMES[2]}"] = ""                   # blank: pinned or absent
    _theta, w = fit_row_parameters(row)
    assert w == {HEURISTIC_NAMES[0]: 0.5}
