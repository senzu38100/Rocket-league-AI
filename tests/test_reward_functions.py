"""Reward functions exercised against a real RocketSim GameState (no mocked physics)."""

import numpy as np
import pytest
from rlgym.rocket_league import common_values

import genericBot
import kickoffBot

# genericBot and kickoffBot each define their own copies of these reward
# classes with identical logic; both are tested to get true per-file coverage.
MODULES = [genericBot, kickoffBot]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
class TestSpeedTowardBallReward:
    def test_returns_all_agents(self, module, kickoff_game_state, blue_and_orange_agents):
        reward_fn = module.SpeedTowardBallReward()
        agents = list(blue_and_orange_agents)
        rewards = reward_fn.get_rewards(agents, kickoff_game_state, {}, {}, {})
        assert set(rewards.keys()) == set(agents)

    def test_stationary_kickoff_car_gets_zero(self, module, kickoff_game_state, blue_and_orange_agents):
        # At kickoff both cars are stationary, so speed-toward-ball is 0 for both.
        reward_fn = module.SpeedTowardBallReward()
        agents = list(blue_and_orange_agents)
        rewards = reward_fn.get_rewards(agents, kickoff_game_state, {}, {}, {})
        for reward in rewards.values():
            assert reward == pytest.approx(0.0)

    def test_reward_never_negative(self, module, kickoff_game_state, blue_and_orange_agents):
        reward_fn = module.SpeedTowardBallReward()
        agents = list(blue_and_orange_agents)
        rewards = reward_fn.get_rewards(agents, kickoff_game_state, {}, {}, {})
        for reward in rewards.values():
            assert reward >= 0.0

    def test_reset_is_a_noop(self, module, kickoff_game_state):
        reward_fn = module.SpeedTowardBallReward()
        assert reward_fn.reset([], kickoff_game_state, {}) is None


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
class TestInAirReward:
    def test_on_ground_car_gets_zero(self, module, kickoff_game_state, blue_and_orange_agents):
        # Kickoff cars start on the ground.
        reward_fn = module.InAirReward()
        agents = list(blue_and_orange_agents)
        rewards = reward_fn.get_rewards(agents, kickoff_game_state, {}, {}, {})
        assert rewards == {agent: 0.0 for agent in agents}

    def test_reset_is_a_noop(self, module, kickoff_game_state):
        reward_fn = module.InAirReward()
        assert reward_fn.reset([], kickoff_game_state, {}) is None


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
class TestVelocityBallToGoalReward:
    def test_stationary_ball_gets_zero(self, module, kickoff_game_state, blue_and_orange_agents):
        reward_fn = module.VelocityBallToGoalReward()
        agents = list(blue_and_orange_agents)
        rewards = reward_fn.get_rewards(agents, kickoff_game_state, {}, {}, {})
        for reward in rewards.values():
            assert reward == pytest.approx(0.0)

    def test_reward_never_negative(self, module, kickoff_game_state, blue_and_orange_agents):
        reward_fn = module.VelocityBallToGoalReward()
        agents = list(blue_and_orange_agents)
        rewards = reward_fn.get_rewards(agents, kickoff_game_state, {}, {}, {})
        for reward in rewards.values():
            assert reward >= 0.0

    def test_reset_is_a_noop(self, module, kickoff_game_state):
        reward_fn = module.VelocityBallToGoalReward()
        assert reward_fn.reset([], kickoff_game_state, {}) is None


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
class TestTouchReward:
    def test_no_touch_gives_zero(self, module, kickoff_game_state, blue_and_orange_agents):
        reward_fn = module.TouchReward()
        agents = list(blue_and_orange_agents)
        rewards = reward_fn.get_rewards(agents, kickoff_game_state, {}, {}, {})
        assert rewards == {agent: 0.0 for agent in agents}

    def test_touch_gives_one(self, module, kickoff_game_state, blue_and_orange_agents):
        reward_fn = module.TouchReward()
        blue, orange = blue_and_orange_agents
        kickoff_game_state.cars[blue].ball_touches = 1
        try:
            rewards = reward_fn.get_rewards([blue, orange], kickoff_game_state, {}, {}, {})
            assert rewards[blue] == 1.0
            assert rewards[orange] == 0.0
        finally:
            kickoff_game_state.cars[blue].ball_touches = 0


class TestStepReward:
    """StepReward only exists in kickoffBot.py."""

    def test_punishes_every_agent_equally(self, blue_and_orange_agents):
        reward_fn = kickoffBot.StepReward()
        agents = list(blue_and_orange_agents)
        rewards = reward_fn.get_rewards(agents, None, {}, {}, {})
        for reward in rewards.values():
            assert reward == pytest.approx(-1.0 / 30.0)

    def test_reset_is_a_noop(self):
        reward_fn = kickoffBot.StepReward()
        assert reward_fn.reset([], None, {}) is None


class TestTouchOnce:
    """TouchOnce (an early-termination condition) only exists in kickoffBot.py."""

    def test_no_one_touched_ball(self, kickoff_game_state):
        done_cond = kickoffBot.TouchOnce()
        result = done_cond.is_done(list(kickoff_game_state.cars.keys()), kickoff_game_state, {})
        assert all(not done for done in result.values())

    def test_any_touch_ends_episode_for_everyone(self, kickoff_game_state, blue_and_orange_agents):
        blue, orange = blue_and_orange_agents
        kickoff_game_state.cars[blue].ball_touches = 1
        try:
            done_cond = kickoffBot.TouchOnce()
            result = done_cond.is_done([blue, orange], kickoff_game_state, {})
            assert result == {blue: True, orange: True}
        finally:
            kickoff_game_state.cars[blue].ball_touches = 0


class TestFaceBallReward:
    """FaceBallReward only exists in genericBot.py."""

    def test_returns_all_agents(self, kickoff_game_state, blue_and_orange_agents):
        reward_fn = genericBot.FaceBallReward()
        agents = list(blue_and_orange_agents)
        rewards = reward_fn.get_rewards(agents, kickoff_game_state, {}, {}, {})
        assert set(rewards.keys()) == set(agents)

    def test_kickoff_cars_face_the_ball(self, kickoff_game_state, blue_and_orange_agents):
        # Kickoff spawns both cars pointed at the ball, so both should score near +1.
        reward_fn = genericBot.FaceBallReward()
        agents = list(blue_and_orange_agents)
        rewards = reward_fn.get_rewards(agents, kickoff_game_state, {}, {}, {})
        for reward in rewards.values():
            assert reward > 0.9

    def test_reset_is_a_noop(self, kickoff_game_state):
        reward_fn = genericBot.FaceBallReward()
        reward_fn.reset([], kickoff_game_state, {})  # must not raise


class TestZeroSumVelocityBallToGoalReward:
    """ZeroSumVelocityBallToGoalReward only exists in genericBot.py."""

    def test_stationary_ball_is_zero_for_both(self, kickoff_game_state, blue_and_orange_agents):
        reward_fn = genericBot.ZeroSumVelocityBallToGoalReward()
        agents = list(blue_and_orange_agents)
        rewards = reward_fn.get_rewards(agents, kickoff_game_state, {}, {}, {})
        for reward in rewards.values():
            assert reward == pytest.approx(0.0)

    def test_ball_moving_toward_orange_net_is_a_real_penalty_for_orange(
        self, kickoff_game_state, blue_and_orange_agents
    ):
        # Unlike VelocityBallToGoalReward, the defending side must come out
        # genuinely negative here, not just 0 -- the exact opposite of the attacker.
        blue, orange = blue_and_orange_agents
        reward_fn = genericBot.ZeroSumVelocityBallToGoalReward()
        original_vel = kickoff_game_state.ball.linear_velocity.copy()
        kickoff_game_state.ball.linear_velocity = np.array([0.0, common_values.BALL_MAX_SPEED, 0.0])
        try:
            rewards = reward_fn.get_rewards([blue, orange], kickoff_game_state, {}, {}, {})
            assert rewards[blue] > 0.0
            assert rewards[orange] == pytest.approx(-rewards[blue])
        finally:
            kickoff_game_state.ball.linear_velocity = original_vel


class TestTouchStrengthReward:
    """TouchStrengthReward only exists in genericBot.py."""

    def test_no_touch_gives_zero_even_if_ball_velocity_changed(
        self, kickoff_game_state, blue_and_orange_agents
    ):
        blue, orange = blue_and_orange_agents
        reward_fn = genericBot.TouchStrengthReward()
        reward_fn.reset([blue, orange], kickoff_game_state, {})
        original_vel = kickoff_game_state.ball.linear_velocity.copy()
        kickoff_game_state.ball.linear_velocity = np.array([5000.0, 0.0, 0.0])
        try:
            rewards = reward_fn.get_rewards([blue, orange], kickoff_game_state, {}, {}, {})
            assert rewards[blue] == 0.0
            assert rewards[orange] == 0.0
        finally:
            kickoff_game_state.ball.linear_velocity = original_vel

    def test_harder_touch_scores_higher_than_a_weak_one(self, kickoff_game_state, blue_and_orange_agents):
        blue, orange = blue_and_orange_agents
        original_vel = kickoff_game_state.ball.linear_velocity.copy()
        kickoff_game_state.cars[blue].ball_touches = 1
        try:
            weak_reward_fn = genericBot.TouchStrengthReward()
            weak_reward_fn.reset([blue, orange], kickoff_game_state, {})
            kickoff_game_state.ball.linear_velocity = np.array([300.0, 0.0, 0.0])
            weak = weak_reward_fn.get_rewards([blue, orange], kickoff_game_state, {}, {}, {})

            kickoff_game_state.ball.linear_velocity = original_vel
            strong_reward_fn = genericBot.TouchStrengthReward()
            strong_reward_fn.reset([blue, orange], kickoff_game_state, {})
            kickoff_game_state.ball.linear_velocity = np.array([5000.0, 0.0, 0.0])
            strong = strong_reward_fn.get_rewards([blue, orange], kickoff_game_state, {}, {}, {})

            assert strong[blue] > weak[blue] > 0.0
            assert strong[blue] <= 1.0
        finally:
            kickoff_game_state.ball.linear_velocity = original_vel
            kickoff_game_state.cars[blue].ball_touches = 0


class TestBallToGoalDirectionSign:
    """Regression check for the is_orange -> goal_y sign flip shared by both reward classes."""

    @pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
    def test_blue_and_orange_aim_at_opposite_goals(self, module, kickoff_game_state, blue_and_orange_agents):
        blue, orange = blue_and_orange_agents
        reward_fn = module.VelocityBallToGoalReward()

        # Give the ball a large +Y velocity: this should count as "toward goal"
        # for the blue car (whose goal_y is +BACK_NET_Y) and as moving away for orange.
        original_vel = kickoff_game_state.ball.linear_velocity.copy()
        kickoff_game_state.ball.linear_velocity = np.array([0.0, common_values.BALL_MAX_SPEED, 0.0])
        try:
            rewards = reward_fn.get_rewards([blue, orange], kickoff_game_state, {}, {}, {})
            assert rewards[blue] > 0.0
            assert rewards[orange] == pytest.approx(0.0)
        finally:
            kickoff_game_state.ball.linear_velocity = original_vel
