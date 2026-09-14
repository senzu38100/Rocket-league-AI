# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Rocket League agent trained with rlgym + RocketSim (headless physics engine) using rlgym-ppo. There is no
model-serving/inference code here — the repo is training scripts only.

## Commands

Dependencies are managed with `uv`; everything below runs via `uv run` (no manual venv activation needed).

```bash
uv sync                      # install/update the environment (downloads Python 3.12 if needed)

uv run python genericBot.py  # train the full-match agent (long-running, up to 1B timesteps)
uv run python kickoffBot.py  # train the kickoff-only agent (long-running)

uv run pytest                                          # run tests (with coverage, fails under 95%)
uv run pytest tests/test_reward_functions.py -k touch  # run a subset by keyword
uv run pytest path/to/test_file.py::TestClass::test_name  # run a single test

uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy .                # type check
```

## Architecture

- **`genericBot.py`** and **`kickoffBot.py`** are standalone, self-contained training entry points — neither
  imports the other, and each embeds its own copy of `RocketSimVisRenderer` and the shared reward classes
  (`SpeedTowardBallReward`, `InAirReward`, `VelocityBallToGoalReward`, `TouchReward`). **`renderer.py`** holds a
  third, independent copy of just the renderer class. This duplication is real, not an import you're missing —
  a change to one file's reward/renderer logic does not propagate to the others.
- Each bot script has the same shape: reward-function classes → `build_rlgym_v2_env()` (constructs the
  `rlgym.api.RLGym` env: state mutator, obs builder, action parser, reward fn, `RocketSimEngine` transition
  engine) → an `if __name__ == "__main__":` block that wraps the env in `rlgym_ppo`'s `Learner` and calls
  `.learn()`. The `__main__` blocks launch real, long-running training runs — never execute them as part of
  testing or exploration.
- `genericBot.py` trains a full 1v1 match (`spawn_opponents=True`, episode ends on `GoalCondition`).
  `kickoffBot.py` trains kickoffs only (`spawn_opponents=False`, single blue car, episode ends on first ball
  touch via the custom `TouchOnce` done-condition or a 3s timeout). `kickoffBot.py` also defines `InAirReward`
  but never wires it into its `CombinedReward` — that's vestigial, not active during kickoff training.
- `RocketSimVisRenderer` only sends game state over UDP (`127.0.0.1:9273`); it is the client half of
  [RocketSimVis](https://github.com/ZealanL/RocketSimVis), which must be run separately to see anything —
  nothing in this repo renders visuals itself.
- Checkpoints land in `data/checkpoints/<project_name>/<timestep>/`; both scripts auto-resume from the
  highest-numbered checkpoint folder under their project name if one exists.
- `device="auto"` is used in both `Learner(...)` calls (picks CUDA if available, else CPU) — deliberate so the
  same code trains on either a CPU-only dev machine or a CUDA box without editing.
- Python is pinned to 3.12 via `.python-version`: `rlgym-rocket-league` requires `numpy<2`, and numpy 1.26.x has
  no prebuilt wheel for Python 3.13, so the newer interpreter can't be used here.
- `torch` is pulled from the PyTorch CPU wheel index (`[tool.uv.sources]` in `pyproject.toml`) rather than the
  default CUDA build, since the CUDA build's ~10 `nvidia-*-cu12` dependencies are dead weight on a machine
  without an NVIDIA GPU. `rlgym-ppo` is installed from a pinned git commit (not PyPI).
- Tests (`tests/conftest.py`) build a real 2-car `GameState` through the actual `RocketSimEngine` rather than
  mocking game physics — reward-function and renderer tests operate on genuine simulated car/ball data.
