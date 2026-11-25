"""Tests for the async HLP->LLP eval harness (hlp_eval.py) — run with:
python -m pytest modules/base_models/SSP/tests/test_hlp_eval.py -v

Two guards:

* :func:`test_map_seed_pairing` -- checkpoint-FREE. Locks the seeding invariant the whole harness
  rests on: the map is seeded at CONSTRUCTION, so same construction seed => identical map and
  different seed => different map. This is the exact bug that once made every "independent" sim run
  the same map; the paired sweep is only valid while this holds.
* :func:`test_run_deterministic` / :func:`test_metrics_present` -- need a trained run under
  ``trained_policies/`` and SKIP if none is present (e.g. CI without checkpoints). They assert the
  batched rollout is reproducible under a fixed RNG seed and that every metric is finite and sane.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # tests/ is one level below the package
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drive_env.interface_gym import CarDrivingEnv, register_envs
from modules.base_models.SSP.hlp_core import HLP_PARAM_NAMES
from modules.base_models.SSP.hlp_eval import build_theta, run

register_envs()

# discover any trained HLP run to exercise the harness against; None => checkpoint tests skip.
_RUNS = sorted(glob.glob(str(PROJECT_ROOT / "modules/base_models/trained_policies/**/hlp_metadata.json"),
                         recursive=True))
RUN_DIR = str(Path(_RUNS[0]).parent) if _RUNS else None
needs_ckpt = pytest.mark.skipif(RUN_DIR is None, reason="no trained HLP checkpoint under trained_policies/")


def test_map_seed_pairing() -> None:
    """Same construction seed => identical map; different seed => different map (checkpoint-free)."""
    o_a, _ = CarDrivingEnv(seed=7, training_mode=False).reset(seed=7)
    o_b, _ = CarDrivingEnv(seed=7, training_mode=False).reset(seed=7)
    o_c, _ = CarDrivingEnv(seed=8, training_mode=False).reset(seed=8)
    assert np.allclose(o_a, o_b), "same construction seed must reproduce the same map"
    assert not np.allclose(o_a, o_c), "different construction seed must give a different map"


@needs_ckpt
def test_run_deterministic() -> None:
    """The batched rollout is reproducible: fixed RNG seed + identical args => identical metrics."""
    thetas = torch.stack([build_theta(None) for _ in range(3)])
    torch.manual_seed(0)
    r1 = run(RUN_DIR, thetas, episodes_target=1, map_seeds=[0, 1, 2])
    torch.manual_seed(0)
    r2 = run(RUN_DIR, thetas, episodes_target=1, map_seeds=[0, 1, 2])
    for key in r1:
        assert np.allclose(r1[key], r2[key], equal_nan=True), f"metric {key} not reproducible"


@needs_ckpt
def test_metrics_present() -> None:
    """run() returns all documented metrics, finite, in sane ranges."""
    thetas = torch.stack([build_theta(None) for _ in range(4)])
    r = run(RUN_DIR, thetas, episodes_target=1, map_seeds=[0, 1, 2, 3])
    expected = {"crash_per_ep", "crash_per_1k", "shown", "ep_len",
                "collect_per_ep", "miss_per_ep", "collect_ratio", "goal_dy", "goal_flag_dist"}
    assert expected <= set(r), f"missing metrics: {expected - set(r)}"
    for key, arr in r.items():
        assert np.isfinite(arr).all(), f"metric {key} has non-finite values"
    assert (r["shown"] >= 0).all() and (r["shown"] <= 8).all()      # cars shown per step, K=6 range
    assert (0.0 <= r["collect_ratio"]).all() and (r["collect_ratio"] <= 1.0).all()
    assert (r["goal_dy"] >= 0).all()                                # forward goal distance, metres
