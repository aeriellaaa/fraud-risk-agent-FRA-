"""
Tests for app.capacity -- the capacity-aware threshold calculator.
"""
import os
import subprocess
import sys
import pytest
import numpy as np
from app.capacity import CapacityAssumptions, find_capacity_constrained_threshold


def test_capacity_assumptions_math_is_correct():
    a = CapacityAssumptions(team_size=5, minutes_per_review=14.0, shift_hours=8.0,
                             daily_transaction_volume=5000)
    assert a.reviews_per_analyst_per_day == pytest.approx(480 / 14, rel=1e-6)
    assert a.max_daily_escalations == pytest.approx(5 * (480 / 14), rel=1e-6)
    assert a.max_sustainable_escalate_rate == pytest.approx((5 * (480 / 14)) / 5000, rel=1e-6)


def test_larger_team_finds_lower_or_equal_threshold_than_smaller_team():
    rng = np.random.default_rng(42)
    probs = rng.beta(2, 20, size=2000)
    y = (rng.random(2000) < 0.02).astype(int)
    small_team = find_capacity_constrained_threshold(
        probs, y, CapacityAssumptions(team_size=1, daily_transaction_volume=5000)
    )
    large_team = find_capacity_constrained_threshold(
        probs, y, CapacityAssumptions(team_size=20, daily_transaction_volume=5000)
    )
    assert large_team.auto_approve_threshold <= small_team.auto_approve_threshold


def test_no_sustainable_threshold_raises_clear_error():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.001, 0.149, size=1000)
    y = (rng.random(1000) < 0.5).astype(int)
    with pytest.raises(ValueError, match="No threshold"):
        find_capacity_constrained_threshold(
            probs, y, CapacityAssumptions(team_size=1, daily_transaction_volume=1_000_000),
            search_max=0.05,
        )


def test_decision_module_defaults_to_cost_optimal_when_team_size_unset():
    env = os.environ.copy()
    env.pop("TEAM_SIZE", None)
    out = subprocess.run(
        [sys.executable, "-c", "from app.decision import AUTO_APPROVE_THRESHOLD; print(AUTO_APPROVE_THRESHOLD)"],
        cwd=os.getcwd(), env=env, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0.01"


def test_decision_module_switches_threshold_when_team_size_set():
    env = os.environ.copy()
    env["TEAM_SIZE"] = "5"
    out = subprocess.run(
        [sys.executable, "-c", "from app.decision import AUTO_APPROVE_THRESHOLD; print(AUTO_APPROVE_THRESHOLD)"],
        cwd=os.getcwd(), env=env, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert float(out.stdout.strip()) != 0.01


def test_decision_module_fails_loudly_for_unsupported_team_size():
    env = os.environ.copy()
    env["TEAM_SIZE"] = "999"
    out = subprocess.run(
        [sys.executable, "-c", "from app.decision import AUTO_APPROVE_THRESHOLD"],
        cwd=os.getcwd(), env=env, capture_output=True, text=True,
    )
    assert out.returncode != 0
    assert "not found in cached results" in out.stderr or "not sustainable" in out.stderr


def test_decision_module_safety_fix_still_present_after_capacity_merge():
    import app.decision as decision_module
    source = open(decision_module.__file__).read()
    assert "score_result.score >= AUTO_APPROVE_THRESHOLD" in source
    assert "score_result.score <= AUTO_APPROVE_THRESHOLD" in source
