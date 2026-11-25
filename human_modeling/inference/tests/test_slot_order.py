"""
Checkpoint-free gates on the slot-order audit (`scripts/check_slot_order.py`).

The audit is itself a gate, so it gets both directions: it must flag a step whose leading
slots hold the ego row (the filler-first defect) and must not flag a well-formed one; the
forced-filler-first wrapper must actually reorder and must leave no trace once it exits; and
the unfixed stand-in must reproduce the defect that the real function now absorbs.
"""

from __future__ import annotations

import torch

from human_modeling.inference.scripts.check_slot_order import (FillerFirstTopk,
                                                                  contract_violations,
                                                                  unfixed_extract_car_rows)
from modules.base_models.SSP import hlp_core

INF = float("-inf")


def _rows(*slots):
    return torch.tensor([list(slots)])


def test_well_formed_steps_are_not_flagged():
    assert not contract_violations(_rows(30, 45, 0, 0), torch.tensor([2])).any()
    assert not contract_violations(_rows(30, 45, 12, 7), torch.tensor([4])).any()   # all real
    assert not contract_violations(_rows(0, 0, 0, 0), torch.tensor([0])).any()      # no cars


def test_filler_ahead_of_a_car_is_flagged():
    assert contract_violations(_rows(0, 0, 30, 45), torch.tensor([2])).tolist() == [True]
    assert contract_violations(_rows(30, 0, 45, 0), torch.tensor([2])).tolist() == [True]
    # | Known-bad the other way: a real row sitting in a slot the count says is filler.
    assert contract_violations(_rows(30, 45, 12, 0), torch.tensor([2])).tolist() == [True]


def _obs_with_cars(rows: list[int]) -> torch.Tensor:
    obs = torch.zeros(1, hlp_core.OBS_SIZE, hlp_core.OBS_FEATURES)
    for i, r in enumerate(rows):
        obs[0, r, 0] = 1.0 + i            # distinct distances so the ranking is well defined
        obs[0, r, 6] = float(hlp_core.CAR_KIND_IDS[0])
    return obs.view(1, -1)


def test_forced_filler_first_breaks_the_unfixed_code_and_not_the_fixed_one():
    obs = _obs_with_cars([30, 45])
    with FillerFirstTopk():
        bad_rows, bad_n = unfixed_extract_car_rows(obs, 6)
        good_rows, good_n = hlp_core.hlp_extract_car_rows(obs, 6)
    assert contract_violations(bad_rows, bad_n).tolist() == [True]
    assert contract_violations(good_rows, good_n).tolist() == [False]
    assert int(good_n[0]) == 2 and set(good_rows[0, :2].tolist()) == {30, 45}


def test_wrapper_reorders_only_unsorted_2d_calls_and_is_removed_on_exit():
    original = torch.topk
    scores = torch.tensor([[-1.0, INF, -2.0, INF, INF]])
    with FillerFirstTopk():
        unsorted_vals, _ = torch.topk(scores, k=4, dim=1, sorted=False)
        sorted_vals, _ = torch.topk(scores, k=2, dim=1, sorted=True)
    assert torch.topk is original                         # known-bad: a leaked patch
    assert torch.isinf(unsorted_vals[0, 0]) and torch.isfinite(unsorted_vals[0, -1])
    assert sorted_vals.tolist() == [[-1.0, -2.0]]         # sorted=True passes through untouched
