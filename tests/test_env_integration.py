"""End-to-end checks that each bot's build_rlgym_v2_env() produces a working env.

These exercise the real RocketSim physics engine (no mocking) the same way the
actual training entry points do, just for a handful of steps instead of a full run.
"""

import numpy as np
import pytest

import genericBot
import kickoffBot


@pytest.mark.parametrize("module", [genericBot, kickoffBot], ids=lambda m: m.__name__)
def test_env_resets_and_steps(module):
    env = module.build_rlgym_v2_env()
    try:
        obs = env.reset()
        n_agents = obs.shape[0]
        assert n_agents >= 1

        for _ in range(5):
            actions = np.array([[env.action_space.sample()] for _ in range(n_agents)])
            obs, rewards, done, truncated, info = env.step(actions)
            assert obs.shape[0] == n_agents
            assert len(rewards) == n_agents
            assert isinstance(done, bool | np.bool_)
            assert isinstance(truncated, bool | np.bool_)
    finally:
        env.close()


def test_generic_bot_spawns_both_teams():
    env = genericBot.build_rlgym_v2_env()
    try:
        obs = env.reset()
        assert obs.shape[0] == 2
    finally:
        env.close()


def test_kickoff_bot_spawns_only_blue():
    env = kickoffBot.build_rlgym_v2_env()
    try:
        obs = env.reset()
        assert obs.shape[0] == 1
    finally:
        env.close()
