"""Shared fixtures backed by the real RocketSim engine (no mocked physics)."""

import numpy as np
import pytest
from rlgym.api import RLGym
from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence


def _build_two_car_env():
    obs_builder = DefaultObs(
        zero_padding=3,
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator(),
        ),
        obs_builder=obs_builder,
        action_parser=LookupTableAction(),
        reward_fn=GoalReward(),
        transition_engine=RocketSimEngine(),
    )


@pytest.fixture(scope="session")
def kickoff_game_state():
    """A real post-kickoff-mutator GameState with one blue car and one orange car."""
    env = _build_two_car_env()
    env.reset()
    state = env.state
    try:
        yield state
    finally:
        env.close()


@pytest.fixture(scope="session")
def blue_and_orange_agents(kickoff_game_state):
    agents = list(kickoff_game_state.cars.keys())
    blue = next(a for a in agents if not kickoff_game_state.cars[a].is_orange)
    orange = next(a for a in agents if kickoff_game_state.cars[a].is_orange)
    return blue, orange
