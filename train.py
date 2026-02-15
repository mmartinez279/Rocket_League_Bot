"""
RLGym PPO training script (from https://rlgym.org/Rocket%20League/training_an_agent).

Run from project root: python train.py
Run with --quick-test to verify the pipeline (short run + one checkpoint).
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path for src.training imports
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
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
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)
from rlgym.rocket_league import common_values
from rlgym_ppo.util import RLGymV2GymWrapper

from src.training.rewards import (
    InAirReward,
    SpeedTowardBallReward,
    VelocityBallToGoalReward,
)


def build_rlgym_v2_env():
    """Build the RLGym environment wrapped for Gym (PPO)."""
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

    reward_fn = CombinedReward(
        (InAirReward(), 0.002),
        (SpeedTowardBallReward(), 0.01),
        (VelocityBallToGoalReward(), 0.1),
        (GoalReward(), 10.0),
    )

    obs_builder = DefaultObs(
        zero_padding=None,
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

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from rlgym_ppo import Learner

    parser = argparse.ArgumentParser(description="RLGym PPO training")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Short run (~1–2 min) that saves one checkpoint to verify the pipeline",
    )
    args = parser.parse_args()

    if args.quick_test:
        n_proc = 4
        timestep_limit = 50_000
        ts_per_iteration = 10_000
        save_every_ts = 25_000
        ppo_batch_size = 10_000
        exp_buffer_size = 25_000
        ppo_minibatch_size = 5_000
    else:
        n_proc = 32
        timestep_limit = 1_000_000_000
        ts_per_iteration = 100_000
        save_every_ts = 1_000_000
        ppo_batch_size = 100_000
        exp_buffer_size = 300_000
        ppo_minibatch_size = 50_000

    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=None,
        ppo_batch_size=ppo_batch_size,
        policy_layer_sizes=[2048, 2048, 1024, 1024],
        critic_layer_sizes=[2048, 2048, 1024, 1024],
        ts_per_iteration=ts_per_iteration,
        exp_buffer_size=exp_buffer_size,
        ppo_minibatch_size=ppo_minibatch_size,
        ppo_ent_coef=0.01,
        policy_lr=1e-4,
        critic_lr=1e-4,
        ppo_epochs=2,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=save_every_ts,
        timestep_limit=timestep_limit,
        log_to_wandb=False,
    )
    learner.learn()
