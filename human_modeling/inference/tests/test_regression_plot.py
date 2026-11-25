"""
The behavioural regression plot reports two different numbers (Entry 35 addendum 3): the
Pearson correlation, blind to offset and scale, and the variance-explained R2 against
``y = x``. A shifted but perfectly ranked prediction is the case that tells them apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from human_modeling.inference.estimation.validate import collision_regression_plot


def test_a_shifted_prediction_correlates_perfectly_but_explains_no_variance(tmp_path: Path):
    human = np.array([8.0, 11.0, 15.0, 18.0, 22.0, 27.0])
    shifted = human + 10.0                                   # right order, wrong level
    st = collision_regression_plot(human, shifted, tmp_path / "shifted.png")
    assert st["r"] == pytest.approx(1.0)
    ss_tot = np.sum((human - human.mean()) ** 2)
    assert st["r2"] == pytest.approx(1.0 - len(human) * 100.0 / ss_tot)
    assert st["r2"] < 0 and st["bias"] == pytest.approx(10.0)
    assert (tmp_path / "shifted.png").stat().st_size > 0


def test_the_correlation_is_pearson_and_undefined_without_spread(tmp_path: Path):
    rng = np.random.default_rng(0)
    human = rng.normal(16.0, 5.0, size=40)
    simulated = 0.5 * human + rng.normal(0.0, 3.0, size=40)
    st = collision_regression_plot(human, simulated, tmp_path / "noisy.png")
    assert st["r"] == pytest.approx(np.corrcoef(human, simulated)[0, 1])
    flat = collision_regression_plot(human, np.full(40, 16.0), tmp_path / "flat.png")
    assert np.isnan(flat["r"])                               # no spread to correlate with
