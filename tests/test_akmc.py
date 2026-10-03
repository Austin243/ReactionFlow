import math

import numpy as np
import pytest

from reactionflow.search.akmc import AKMCConfig, confidence, in_window, kmc_step, rate


def test_confidence_window_and_rates_follow_eon():
    assert confidence(0) == 0 and confidence(20) == pytest.approx(0.95)
    config = AKMCConfig(temperature_K=300.0)
    assert in_window(0.5 + 20 * config.kT, 0.5, config)
    assert not in_window(0.5 + 20.01 * config.kT, 0.5, config)
    assert rate(1e13, 0.5, config.kT) == pytest.approx(1e13 * math.exp(-0.5 / config.kT))


def test_kmc_steps_pick_by_rate_and_wait_one_over_the_total_rate():
    picks, waits = zip(*(kmc_step([1.0, 3.0], [7, 1, step]) for step in range(20000)), strict=True)
    assert np.mean(picks) == pytest.approx(0.75, abs=0.01)
    assert np.mean(waits) == pytest.approx(0.25, abs=0.01)
