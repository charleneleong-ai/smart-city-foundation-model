from sctwin.reason.reward import (
    accuracy_reward,
    conservation_reward,
    coverage_reward,
    interventional_reward,
)


def test_accuracy_is_1_at_exact_and_decays_with_error():
    assert accuracy_reward(5.0, 5.0, scale=2.0) == 1.0
    near = accuracy_reward(5.5, 5.0, scale=2.0)
    far = accuracy_reward(9.0, 5.0, scale=2.0)
    assert 0.0 < far < near < 1.0


def test_coverage_in_and_out_of_interval():
    assert coverage_reward(5.0, 4.0, 6.0) == 1.0
    assert coverage_reward(7.0, 4.0, 6.0) == 0.0


def test_interventional_rewards_direction_and_magnitude():
    assert interventional_reward(-3.0, -3.0, scale=2.0) == 1.0  # right way + exact
    right_dir = interventional_reward(-1.0, -5.0, scale=2.0)  # right way, off magnitude
    wrong_dir = interventional_reward(2.0, -3.0, scale=2.0)  # wrong way
    assert 0.5 <= right_dir < 1.0
    assert wrong_dir < 0.5  # missing the sign loses the direction half


def test_conservation_rewards_balance():
    assert conservation_reward([3.0, 7.0], 10.0) == 1.0  # parts sum to whole
    assert conservation_reward([3.0, 3.0], 10.0) < 1.0  # 6 != 10 -> penalised
