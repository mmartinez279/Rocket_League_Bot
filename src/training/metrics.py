"""Custom MetricsLogger that prints per-iteration diagnostics to the console.

Tracks car behavior (speed, distance to ball, ball touches) so you can tell
at a glance whether the bot is learning to move and engage.
"""

import numpy as np

from rlgym_ppo.util import MetricsLogger
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values


class BehaviorLogger(MetricsLogger):
    """Log observable bot behavior every iteration.

    Metrics collected per timestep (7 floats):
      [0] car_speed          - np.linalg.norm(linear_velocity)
      [1] dist_to_ball       - distance from car to ball
      [2] speed_toward_ball  - velocity component toward ball (>0 = approaching)
      [3] ball_speed         - np.linalg.norm(ball.linear_velocity)
      [4] facing_ball        - dot(car_forward, dir_to_ball), 1=facing, -1=away
      [5] blue_score         - goals scored
      [6] orange_score       - goals scored
    """

    NUM_METRICS = 7

    def _collect_metrics(self, game_state: GameState) -> list:
        # Pick the first car we find (works for 1v1)
        metrics = np.zeros(self.NUM_METRICS, dtype=np.float32)

        if not game_state.cars:
            return [metrics]

        agent_id = next(iter(game_state.cars))
        car = game_state.cars[agent_id]
        car_phys = car.physics
        ball_phys = game_state.ball

        car_vel = car_phys.linear_velocity
        ball_vel = ball_phys.linear_velocity
        car_speed = float(np.linalg.norm(car_vel))

        pos_diff = ball_phys.position - car_phys.position
        dist_to_ball = float(np.linalg.norm(pos_diff))
        dir_to_ball = pos_diff / max(dist_to_ball, 1e-6)

        speed_toward_ball = float(np.dot(car_vel, dir_to_ball))
        ball_speed = float(np.linalg.norm(ball_vel))

        # Car forward vector from physics
        try:
            forward = car_phys.forward
        except Exception:
            forward = np.array([1, 0, 0], dtype=np.float32)
        facing_ball = float(np.dot(forward, dir_to_ball))

        metrics[0] = car_speed
        metrics[1] = dist_to_ball
        metrics[2] = speed_toward_ball
        metrics[3] = ball_speed
        metrics[4] = facing_ball
        metrics[5] = float(getattr(game_state, "blue_score", 0) or 0)
        metrics[6] = float(getattr(game_state, "orange_score", 0) or 0)

        return [metrics]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        if not collected_metrics:
            return

        # collected_metrics is a list of arrays, each from _collect_metrics
        all_m = np.array([m[0] for m in collected_metrics if len(m) > 0])
        if len(all_m) == 0:
            return

        avg_speed = np.mean(all_m[:, 0])
        avg_dist = np.mean(all_m[:, 1])
        avg_toward = np.mean(all_m[:, 2])
        avg_ball_speed = np.mean(all_m[:, 3])
        avg_facing = np.mean(all_m[:, 4])
        max_speed = np.max(all_m[:, 0])
        pct_moving = np.mean(all_m[:, 0] > 100) * 100  # % of time speed > 100 uu/s

        report = {
            "behavior/avg_car_speed": float(avg_speed),
            "behavior/max_car_speed": float(max_speed),
            "behavior/pct_time_moving": float(pct_moving),
            "behavior/avg_dist_to_ball": float(avg_dist),
            "behavior/avg_speed_toward_ball": float(avg_toward),
            "behavior/avg_ball_speed": float(avg_ball_speed),
            "behavior/avg_facing_ball": float(avg_facing),
        }

        # Always print to console (even without wandb)
        print("\n--- Behavior Diagnostics ---")
        print(f"  Avg car speed:         {avg_speed:>8.1f} / {common_values.CAR_MAX_SPEED} uu/s")
        print(f"  Max car speed:         {max_speed:>8.1f} uu/s")
        print(f"  % time moving (>100):  {pct_moving:>8.1f}%")
        print(f"  Avg dist to ball:      {avg_dist:>8.1f} uu")
        print(f"  Avg speed toward ball: {avg_toward:>8.1f} uu/s")
        print(f"  Avg ball speed:        {avg_ball_speed:>8.1f} uu/s")
        print(f"  Avg facing ball:       {avg_facing:>8.3f}  (1=facing, 0=perpendicular, -1=away)")
        print("----------------------------\n")

        # Log to wandb if available
        if wandb_run is not None:
            report["Cumulative Timesteps"] = cumulative_timesteps
            wandb_run.log(report)
