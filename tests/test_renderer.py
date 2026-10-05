"""Tests for RocketSimVisRenderer, using real Car/PhysicsObject data from RocketSim."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

import genericBot
import kickoffBot
import renderer

# genericBot.py and kickoffBot.py each duplicate renderer.py's RocketSimVisRenderer verbatim.
MODULES = [renderer, genericBot, kickoffBot]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
class TestWritePhysobj:
    def test_returns_expected_keys(self, module, kickoff_game_state):
        j = module.RocketSimVisRenderer.write_physobj(kickoff_game_state.ball)
        assert set(j.keys()) == {"pos", "forward", "up", "vel", "ang_vel"}

    def test_values_are_plain_lists(self, module, kickoff_game_state):
        j = module.RocketSimVisRenderer.write_physobj(kickoff_game_state.ball)
        for value in j.values():
            assert isinstance(value, list)


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
class TestWriteCar:
    def test_returns_expected_keys(self, module, kickoff_game_state, blue_and_orange_agents):
        blue, _ = blue_and_orange_agents
        car = kickoff_game_state.cars[blue]
        j = module.RocketSimVisRenderer.write_car(car)
        assert set(j.keys()) == {
            "team_num",
            "phys",
            "boost_amount",
            "on_ground",
            "has_flipped_or_double_jumped",
            "is_demoed",
            "has_flip",
        }
        assert "controls" not in j

    def test_no_controls_by_default(self, module, kickoff_game_state, blue_and_orange_agents):
        blue, _ = blue_and_orange_agents
        car = kickoff_game_state.cars[blue]
        j = module.RocketSimVisRenderer.write_car(car)
        assert "controls" not in j

    def test_array_controls_are_mapped_to_button_names(
        self, module, kickoff_game_state, blue_and_orange_agents
    ):
        blue, _ = blue_and_orange_agents
        car = kickoff_game_state.cars[blue]
        controls = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        j = module.RocketSimVisRenderer.write_car(car, controls=controls)
        assert j["controls"] == dict(zip(module.BUTTON_NAMES, controls.tolist(), strict=True))

    def test_dict_controls_are_passed_through(self, module, kickoff_game_state, blue_and_orange_agents):
        blue, _ = blue_and_orange_agents
        car = kickoff_game_state.cars[blue]
        controls = {"throttle": 1.0}
        j = module.RocketSimVisRenderer.write_car(car, controls=controls)
        assert j["controls"] == controls


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
class TestRender:
    def test_sends_one_udp_packet_with_expected_shape(self, module, kickoff_game_state):
        r = module.RocketSimVisRenderer()
        r.sock = MagicMock()

        r.render(kickoff_game_state, shared_info={})

        r.sock.sendto.assert_called_once()
        payload, addr = r.sock.sendto.call_args[0]
        assert addr == (module.DEFAULT_UDP_IP, module.DEFAULT_UDP_PORT)

        j = json.loads(payload.decode("utf-8"))
        assert set(j.keys()) == {"ball_phys", "cars", "boost_pad_states"}
        assert len(j["cars"]) == len(kickoff_game_state.cars)

    def test_close_does_not_raise(self, module):
        module.RocketSimVisRenderer().close()
