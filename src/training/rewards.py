"""Custom reward functions for RLGym training (from https://rlgym.org/Rocket%20League/training_an_agent)."""

from typing import List, Dict, Any

import numpy as np

from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values


class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for moving quickly toward the ball."""

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
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

            rewards[agent] = max(
                speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0
            )
        return rewards


class InAirReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being in the air."""

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        return {
            agent: float(not state.cars[agent].on_ground) for agent in agents
        }


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for hitting the ball toward the opponent's goal."""

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            if car.is_orange:
                goal_y = -common_values.BACK_NET_Y
            else:
                goal_y = common_values.BACK_NET_Y

            ball_vel = ball.linear_velocity
            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)
            dir_to_goal = pos_diff / dist

            vel_toward_goal = np.dot(ball_vel, dir_to_goal)
            rewards[agent] = max(
                vel_toward_goal / common_values.BALL_MAX_SPEED, 0
            )
        return rewards


# Horizontal distance threshold for "ball on roof" (uu); car ~120, ball radius ~91
HORIZ_ROOF_THRESHOLD = 150.0


def _ball_on_roof(car_physics, ball_physics) -> tuple[bool, float]:
    """Return (is_above, proximity_reward). Ball above car and within horizontal range."""
    pos_diff = ball_physics.position - car_physics.position
    dist_xy = np.linalg.norm(pos_diff[0:2])
    ball_above = ball_physics.position[2] > car_physics.position[2]
    proximity = max(0.0, 1.0 - dist_xy / HORIZ_ROOF_THRESHOLD) if ball_above else 0.0
    return ball_above, proximity


class BallOnRoofReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent when the ball is above the car and within horizontal range (roof zone)."""

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            _, proximity = _ball_on_roof(car_physics, ball_physics)
            rewards[agent] = proximity
        return rewards


class BallCarryStabilityReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent when the ball's velocity matches the car's (ball carried, not sliding off). Only when ball is on roof."""

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            ball_above, _ = _ball_on_roof(car_physics, ball_physics)
            if not ball_above:
                rewards[agent] = 0.0
                continue
            vel_diff = np.linalg.norm(
                ball_physics.linear_velocity - car_physics.linear_velocity
            )
            scale = common_values.CAR_MAX_SPEED
            rewards[agent] = max(0.0, 1.0 - vel_diff / scale)
        return rewards
