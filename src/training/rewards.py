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
            car_physics = car.inverted_physics if car.is_orange else car.physics
            ball_physics = state.inverted_ball if car.is_orange else state.ball
            player_vel = car_physics.linear_velocity
            pos_diff = ball_physics.position - car_physics.position
            dist_to_ball = max(np.linalg.norm(pos_diff), 1e-6)
            dir_to_ball = pos_diff / dist_to_ball

            speed_toward_ball = np.dot(player_vel, dir_to_ball)

            rewards[agent] = max(
                speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0
            )
        return rewards


class BallTouchReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for making contact with the ball."""

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self._prev_touches: Dict[AgentID, int] = {
            agent: initial_state.cars[agent].ball_touches for agent in agents
        }

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
            current_touches = state.cars[agent].ball_touches
            prev_touches = self._prev_touches.get(agent, 0)
            rewards[agent] = float(current_touches > prev_touches)
            self._prev_touches[agent] = current_touches
        return rewards


class ProximityToBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being close to the ball.

    Returns a value in [0, 1] that increases as the car gets closer,
    scaled linearly by the field diagonal.
    """

    def __init__(self) -> None:
        super().__init__()
        self._max_dist = np.sqrt(
            common_values.SIDE_WALL_X ** 2
            + common_values.BACK_NET_Y ** 2
            + common_values.CEILING_Z ** 2
        )

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
            car_pos = state.cars[agent].physics.position
            ball_pos = state.ball.position
            dist = np.linalg.norm(ball_pos - car_pos)
            rewards[agent] = 1.0 - min(dist / self._max_dist, 1.0)
        return rewards


class DefensivePenaltyReward(RewardFunction[AgentID, GameState, float]):
    """Penalizes the agent when the ball moves toward their own goal."""

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
            ball = state.inverted_ball if car.is_orange else state.ball

            # In blue's perspective the agent's own goal is at -BACK_NET_Y
            own_goal = np.array([0, -common_values.BACK_NET_Y, 0])
            pos_diff = own_goal - ball.position
            dist = max(np.linalg.norm(pos_diff), 1e-6)
            dir_to_own_goal = pos_diff / dist

            vel_toward_own_goal = np.dot(ball.linear_velocity, dir_to_own_goal)
            # Negative reward when ball moves toward own goal
            rewards[agent] = -max(
                vel_toward_own_goal / common_values.BALL_MAX_SPEED, 0.0
            )
        return rewards


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
            ball = state.inverted_ball if car.is_orange else state.ball

            # In blue's perspective the opponent's goal is always at +BACK_NET_Y
            goal_pos = np.array([0, common_values.BACK_NET_Y, 0])
            pos_diff = goal_pos - ball.position
            dist = max(np.linalg.norm(pos_diff), 1e-6)
            dir_to_goal = pos_diff / dist

            vel_toward_goal = np.dot(ball.linear_velocity, dir_to_goal)
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


class BoostAccumulationReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for having boost and for picking up boost pads.

    Combines a small continuous reward for current boost amount with a
    larger one-time reward each time boost increases (pad pickup).

    WARNING: This reward can teach the bot to sit still and hoard boost
    instead of playing. Use with very low weight or only in later training
    stages when the bot already knows how to drive and hit the ball.
    """

    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        self._prev_boost: Dict[AgentID, float] = {
            agent: initial_state.cars[agent].boost_amount for agent in agents
        }

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
            boost = state.cars[agent].boost_amount  # 0–100
            prev = self._prev_boost.get(agent, 0.0)

            # Continuous reward for having boost (normalized to 0–1)
            hold_reward = boost / 100.0

            # Bonus when boost increases (picked up a pad)
            gain = max(boost - prev, 0.0)
            pickup_reward = gain / 100.0

            rewards[agent] = 0.5 * hold_reward + 0.5 * pickup_reward
            self._prev_boost[agent] = boost
        return rewards


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
            car_physics = car.inverted_physics if car.is_orange else car.physics
            ball_physics = state.inverted_ball if car.is_orange else state.ball
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
            car_physics = car.inverted_physics if car.is_orange else car.physics
            ball_physics = state.inverted_ball if car.is_orange else state.ball
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


class FaceBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for pointing its nose toward the ball.

    Uses the dot product of the car's forward vector with the direction
    to the ball.  Returns a value in [-1, 1] (1 = perfectly facing ball,
    -1 = facing directly away).  Clamped to [0, 1] so only positive
    alignment is rewarded.

    This is a critical early-training reward: a bot that faces the ball
    can learn to drive toward it; a bot facing away cannot.
    """

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
            car_physics = car.inverted_physics if car.is_orange else car.physics
            ball_physics = state.inverted_ball if car.is_orange else state.ball

            pos_diff = ball_physics.position - car_physics.position
            dist = max(np.linalg.norm(pos_diff), 1e-6)
            dir_to_ball = pos_diff / dist

            # Car forward vector from rotation matrix (first column)
            forward = car_physics.forward
            alignment = float(np.dot(forward, dir_to_ball))

            rewards[agent] = max(alignment, 0.0)
        return rewards
