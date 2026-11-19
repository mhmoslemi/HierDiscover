import sys

import pytest

from problems.base import ParentContext, SeedState
from problems.gpu_mode import GpuMode


def test_trimul_reward_anchors_and_growth():
    problem = GpuMode({"kernel_python": sys.executable})
    reward = problem._runtime_reward
    assert [reward(6500), reward(2800), reward(550)] == pytest.approx([2, 2.4, 5])
    assert 0 < reward(12000) < reward(6500) < reward(3500) < reward(2800)
    assert reward(1800) - reward(2800) > reward(2800) - reward(3800)
    assert reward(250) > reward(550)
    assert problem.initial_spo_rs_rewards([SeedState()]) == [2.0]
    assert problem.compute_reward("No kernel", ParentContext(), 1).reward == 0


def test_trimul_legacy_reward_scales():
    anchors = GpuMode({"kernel_python": sys.executable, "erdos_reward_anchors": [7000, 550]})
    inverse = GpuMode({"kernel_python": sys.executable, "erdos_reward_anchors": None,
                       "score_scale": 1500})
    assert anchors._runtime_reward(7000) == 2.0
    assert anchors._runtime_reward(550) == pytest.approx(1 / 0.38086)
    assert inverse._runtime_reward(1000) == 1.5
