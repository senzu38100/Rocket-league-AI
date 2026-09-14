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
