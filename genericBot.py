import json
import os
import socket
from typing import Any

import numpy as np
from rlgym.api import AgentID, Renderer, RewardFunction
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import Car, GameState

DEFAULT_UDP_IP = "127.0.0.1"
DEFAULT_UDP_PORT = 9273

# True: watch the agent live via RocketSimVis (must be run separately, see
# RocketSimVisRenderer below). False: full-speed bulk training. rlgym_ppo only
# ever renders proc 0 regardless of this flag, so this isn't about throttling
# all n_proc workers -- it's about not paying render_delay on that one process,
# and not requiring RocketSimVis to be running, when nobody's watching.
VISUALIZE_MODE = True

if VISUALIZE_MODE:
    RENDER_MODE = True
    RENDER_DELAY = 0.047
else:
    RENDER_MODE = False
    RENDER_DELAY = 0.0


BUTTON_NAMES = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")


class RocketSimVisRenderer(Renderer[GameState]):
    """
    A renderer that sends game state information to RocketSimVis.

    This is just the client side, you need to run RocketSimVis to see the visualization.
    Code is here: https://github.com/ZealanL/RocketSimVis
    """

    def __init__(self, udp_ip=DEFAULT_UDP_IP, udp_port=DEFAULT_UDP_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self.udp_ip = udp_ip
        self.udp_port = udp_port

    @staticmethod
    def write_physobj(physobj):
        j = {
            "pos": physobj.position.tolist(),
            "forward": physobj.forward.tolist(),
            "up": physobj.up.tolist(),
            "vel": physobj.linear_velocity.tolist(),
            "ang_vel": physobj.angular_velocity.tolist(),
        }

        return j

    @staticmethod
    def write_car(car: Car, controls=None):
        j = {
            "team_num": int(car.team_num),
            "phys": RocketSimVisRenderer.write_physobj(car.physics),
            "boost_amount": car.boost_amount,
            "on_ground": bool(car.on_ground),
            "has_flipped_or_double_jumped": bool(car.has_flipped or car.has_double_jumped),
            "is_demoed": bool(car.is_demoed),
            "has_flip": bool(car.can_flip),
        }

        if controls is not None:
            if isinstance(controls, np.ndarray):
                controls = {k: float(v) for k, v in zip(BUTTON_NAMES, controls, strict=True)}
            j["controls"] = controls

        return j

    def render(self, state: GameState, shared_info: dict[str, Any]) -> Any:
        controls = shared_info.get("controls", {})
        j = {
            "ball_phys": self.write_physobj(state.ball),
            "cars": [self.write_car(car, controls.get(agent_id)) for agent_id, car in state.cars.items()],
            "boost_pad_states": (state.boost_pad_timers <= 0).tolist(),
        }

        self.sock.sendto(json.dumps(j).encode("utf-8"), (self.udp_ip, self.udp_port))

    def close(self):
        pass


# kickoffBot.py also hardcodes this exact name, so running both without
# changing one first will have them silently read/write the same
# data/checkpoints/ folder. kickoffBot.py is deprioritized (not this bot's
# task) rather than renamed, so just don't run both against a shared checkpoint
# folder without checking which one you mean to resume.
project_name = "GenericBot"

# One dict per curriculum stage: these weights are what CombinedReward actually
# uses below, AND what gets logged to wandb config in __main__ so a run's
# metadata records what it actually trained under -- rlgym_ppo's own
# auto-logged config (n_proc, ent_coef, lr, ...) says nothing about custom
# reward weights, since they're just Python objects invisible to Learner.
STAGE_CONFIGS: dict[int, dict[str, Any]] = {
    1: {
        "description": "Can't reliably touch the ball yet -- no goal-directed reward at all.",
        "weights": {"FaceBallReward": 1, "SpeedTowardBallReward": 5, "InAirReward": 0.15, "TouchReward": 50},
    },
    2: {
        "description": (
            "Touches reliably -- strength-scaled touch, zero-sum goal-direction, moderated goal reward."
        ),
        "weights": {
            "SpeedTowardBallReward": 2,
            "InAirReward": 0.1,
            "TouchStrengthReward": 10,
            "ZeroSumVelocityBallToGoalReward": 8,
            "GoalReward": 80.0,
        },
    },
}

# 1 = can't reliably touch the ball yet (touch + speed + air, no goal-directed
#     reward -- a goal signal this early is mostly noise before touching is
#     solved).
# 2 = touches reliably, learning to direct those touches toward scoring.
# 3 = not designed yet -- build it once stage 2's actual behavior shows what's
#     missing, not before; build_rlgym_v2_env() raises NotImplementedError
#     past 2 on purpose.
# Bump this by hand once watching the bot / the wandb graphs shows it's hit
# the next stage -- no auto-schedule, on purpose. Changing this and resuming
# from the same checkpoint is safe: reward logic is environment-side, it
# isn't part of what a checkpoint saves.
TRAINING_STAGE = 1


class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for moving quickly toward the ball"""

    def reset(self, agents: list[AgentID], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            player_vel = car_physics.linear_velocity
            pos_diff = ball_physics.position - car_physics.position
            dist_to_ball = np.linalg.norm(pos_diff)
            dir_to_ball = pos_diff / dist_to_ball

            speed_toward_ball = np.dot(player_vel, dir_to_ball)

            rewards[agent] = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)
        return rewards


class InAirReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being in the air"""

    def reset(self, agents: list[AgentID], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for hitting the ball toward the opponent's goal.

    Superseded by ZeroSumVelocityBallToGoalReward below: this version clips at
    0 independently per side, so pushing the ball toward their net pays the
    attacker while the defender gets exactly 0 -- never an actual penalty for
    failing to defend. Kept for reference, not used by either stage.
    """

    def reset(self, agents: list[AgentID], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            goal_y = -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y

            ball_vel = ball.linear_velocity
            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)
            dir_to_goal = pos_diff / dist

            vel_toward_goal = np.dot(ball_vel, dir_to_goal)
            rewards[agent] = max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0)
        return rewards


class ZeroSumVelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Stage 2+ replacement for VelocityBallToGoalReward. One shared signed
    quantity -- ball velocity toward orange's net -- and each side gets +it or
    -it depending on which way benefits them, so a push toward one net is a
    real, equal penalty for whoever's defending it, not just a missed reward.
    """

    def reset(self, agents: list[AgentID], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        ball = state.ball
        pos_diff = np.array([0, common_values.BACK_NET_Y, 0]) - ball.position
        dist = np.linalg.norm(pos_diff)
        dir_to_orange_net = pos_diff / dist
        vel_toward_orange_net = np.dot(ball.linear_velocity, dir_to_orange_net) / common_values.BALL_MAX_SPEED

        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # Blue wants the ball moving toward orange's net; orange wants the opposite.
            rewards[agent] = vel_toward_orange_net if not car.is_orange else -vel_toward_orange_net
        return rewards


class FaceBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Stage 1 addition. Rewards facing the ball, since SpeedTowardBallReward
    alone only cares about velocity direction and doesn't penalize reaching
    the ball by driving in reverse.
    """

    def reset(self, agents: list[AgentID], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            pos_diff = ball_physics.position - car_physics.position
            dist = np.linalg.norm(pos_diff)
            dir_to_ball = pos_diff / dist
            rewards[agent] = float(np.dot(car_physics.forward, dir_to_ball))
        return rewards


class TouchReward(RewardFunction[AgentID, GameState, float]):
    """
    A RewardFunction that gives a reward of 1 if the agent touches the ball, 0 otherwise.

    Stage 1 only: flat reward-per-touch is right while touching at all is the
    hard part, but it becomes farmable (weak repeated bumps instead of a real
    hit) once touching stops being hard -- see TouchStrengthReward for stage 2.
    """

    def reset(self, agents: list[AgentID], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        return 1.0 if state.cars[agent].ball_touches > 0 else 0.0


class TouchStrengthReward(RewardFunction[AgentID, GameState, float]):
    """
    Stage 2 replacement for TouchReward. Scales with how much a touch actually
    changed the ball's velocity instead of flat +1 for any contact, so a weak
    farmed bump no longer pays the same as a committed hit. Needs the previous
    step's ball velocity, so unlike every other reward here it's stateful
    across steps.
    """

    def reset(self, agents: list[AgentID], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        self.last_ball_vel: np.ndarray = initial_state.ball.linear_velocity.copy()

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        vel_change = np.linalg.norm(state.ball.linear_velocity - self.last_ball_vel)
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches > 0:
                rewards[agent] = min(vel_change / common_values.BALL_MAX_SPEED, 1.0)
            else:
                rewards[agent] = 0.0
        self.last_ball_vel = state.ball.linear_velocity.copy()
        return rewards


def build_rlgym_v2_env():
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        AnyCondition,
        GoalCondition,
        NoTouchTimeoutCondition,
        TimeoutCondition,
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
    from rlgym_ppo.util import RLGymV2GymWrapper

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
    )

    if TRAINING_STAGE == 1:
        w = STAGE_CONFIGS[1]["weights"]
        reward_fn = CombinedReward(
            (FaceBallReward(), w["FaceBallReward"]),
            (SpeedTowardBallReward(), w["SpeedTowardBallReward"]),
            (InAirReward(), w["InAirReward"]),
            (TouchReward(), w["TouchReward"]),
        )
    elif TRAINING_STAGE == 2:
        w = STAGE_CONFIGS[2]["weights"]
        reward_fn = CombinedReward(
            (SpeedTowardBallReward(), w["SpeedTowardBallReward"]),
            (InAirReward(), w["InAirReward"]),
            (TouchStrengthReward(), w["TouchStrengthReward"]),
            (ZeroSumVelocityBallToGoalReward(), w["ZeroSumVelocityBallToGoalReward"]),
            (GoalReward(), w["GoalReward"]),
        )
    else:
        raise NotImplementedError(
            f"TRAINING_STAGE={TRAINING_STAGE} has no reward design yet -- "
            "figure out stage 3 once stage 2's actual behavior shows what's missing."
        )

    obs_builder = DefaultObs(
        zero_padding=3,
        pos_coef=np.asarray(
            [1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size), KickoffMutator()
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer() if VISUALIZE_MODE else None,
    )

    return RLGymV2GymWrapper(rlgym_env)


def _checkpoint_is_complete(folder_path: str) -> bool:
    """
    rlgym_ppo's ppo_learner.load_from() does torch.load() on these 4 files with
    no existence check. *.pt files are gitignored, so a folder that only has
    BOOK_KEEPING_VARS.json (e.g. straight out of a git checkout) is NOT
    loadable -- pointing checkpoint_load_folder at one crashes with
    FileNotFoundError before training starts.
    """
    required = ("PPO_POLICY.pt", "PPO_VALUE_NET.pt", "PPO_POLICY_OPTIMIZER.pt", "PPO_VALUE_NET_OPTIMIZER.pt")
    return all(os.path.exists(os.path.join(folder_path, f)) for f in required)


if __name__ == "__main__":
    import wandb
    from rlgym_ppo import Learner

    n_proc = 32  # untuned for this machine -- raise/lower until CPU usage maxes out

    min_inference_size = max(1, int(round(n_proc * 0.9)))
    checkpoint_folder = f"data/checkpoints/{project_name}"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_folder)
        if f.isdigit()
        and os.path.isdir(os.path.join(checkpoint_folder, f))
        and _checkpoint_is_complete(os.path.join(checkpoint_folder, f))
    ]
    # key=int matters: folder names are timestep counts as strings, and plain
    # max() on strings compares lexicographically -- "999999999" would sort
    # ABOVE "1000000005" since '9' > '1' at position 0. timestep_limit below is
    # 1_000_000_000, so crossing that digit boundary isn't hypothetical.
    checkpoint_load_folder = (
        os.path.join(checkpoint_folder, max(checkpoint_files, key=int)) if checkpoint_files else None
    )

    should_log_to_wandb = True

    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=None,
        ppo_batch_size=100_000,
        policy_layer_sizes=[512, 512, 512],  # policy network
        critic_layer_sizes=[512, 512, 512],  # critic network
        ts_per_iteration=100_000,  # timesteps per training iteration, set equal to batch size
        exp_buffer_size=300_000,  # experience buffer size, keep 2-3x the batch size
        ppo_minibatch_size=50_000,  # keep this small; scale compute via layer sizes instead
        ppo_ent_coef=0.05,
        render=RENDER_MODE,
        render_delay=RENDER_DELAY,
        add_unix_timestamp=False,
        checkpoint_load_folder=checkpoint_load_folder,
        checkpoints_save_folder=checkpoint_folder,
        policy_lr=1e-4,
        device="auto",  # CUDA when available, falls back to cpu otherwise
        critic_lr=1e-4,  # critic learning rate
        ppo_epochs=2,  # number of PPO epochs
        standardize_returns=True,  # Don't touch these.
        standardize_obs=False,  # Don't touch these.
        save_every_ts=1_000_000,  # save every 1M steps
        timestep_limit=1_000_000_000,  # Train for 1B steps
        random_seed=123,  # explicit rather than silently inherited (this IS rlgym_ppo's own default)
        log_to_wandb=should_log_to_wandb,
        wandb_project_name="RocketLeague1v1",  # separate from kickoffBot.py's "RocketLeagueKickoff"
        wandb_group_name="training",
        wandb_run_name="Essai1",  # rename freely
    )

    # rlgym_ppo's own auto-logged wandb config (n_proc, ent_coef, lr, ...)
    # says nothing about which TRAINING_STAGE or reward weights this run used
    # -- log it ourselves so the run's metadata is self-describing later.
    if should_log_to_wandb and wandb.run is not None:
        wandb.config.update(
            {
                "training_stage": TRAINING_STAGE,
                "stage_description": STAGE_CONFIGS[TRAINING_STAGE]["description"],
                "reward_weights": STAGE_CONFIGS[TRAINING_STAGE]["weights"],
            }
        )

    learner.learn()
