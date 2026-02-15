"""
RLBot 2.0 bot that runs a trained rlgym-ppo policy so you can play against it.

Set PPO_CHECKPOINT to the folder containing PPO_POLICY.pt (e.g. data/checkpoints/rlgym-ppo-run-<timestamp>/50000).
Run a match with this bot (e.g. Orange) vs Human (Blue) from rlbot.toml.
"""

import os
import sys
from pathlib import Path

# Project root so we can import train and training modules
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Any, Dict

import numpy as np
import torch
from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot

from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym_ppo.ppo import DiscreteFF

from src.training.packet_to_state import game_state_from_packet


# Must match train.py: action repeat 8
TICK_SKIP = 8


def _get_obs_builder():
    """Same DefaultObs config as train.py."""
    return DefaultObs(
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


def _action_to_controller(action_8: np.ndarray) -> ControllerState:
    """Map 8-float action (throttle, steer, pitch, yaw, roll, jump, boost, handbrake) to ControllerState."""
    c = ControllerState()
    c.throttle = float(np.clip(action_8[0], -1, 1))
    c.steer = float(np.clip(action_8[1], -1, 1))
    c.pitch = float(np.clip(action_8[2], -1, 1))
    c.yaw = float(np.clip(action_8[3], -1, 1))
    c.roll = float(np.clip(action_8[4], -1, 1))
    c.jump = bool(action_8[5] > 0.5)
    c.boost = bool(action_8[6] > 0.5)
    c.handbrake = bool(action_8[7] > 0.5)
    return c


class PPOBot(Bot):
    """Bot that runs a trained rlgym-ppo policy (1v1, same obs/action as training)."""

    def initialize(self) -> None:
        checkpoint_path = os.environ.get("PPO_CHECKPOINT", "").strip()
        if not checkpoint_path or not Path(checkpoint_path).is_dir():
            # Try default: latest run under data/checkpoints
            base = _project_root / "data" / "checkpoints"
            if base.is_dir():
                runs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                for run_dir in runs:
                    if not run_dir.is_dir():
                        continue
                    # Subdirs are timestep numbers
                    subs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                    if subs:
                        checkpoint_path = str(max(subs, key=lambda d: int(d.name)))
                        break
        if not checkpoint_path or not (Path(checkpoint_path) / "PPO_POLICY.pt").exists():
            raise FileNotFoundError(
                "PPO_CHECKPOINT must point to a folder containing PPO_POLICY.pt (e.g. data/checkpoints/rlgym-ppo-run-<ts>/50000). "
                "Run a quick test first: python train.py --quick-test"
            )

        # Get obs/action size from the same env as training
        from train import build_rlgym_v2_env
        env = build_rlgym_v2_env()
        obs_size = int(env.observation_space.shape[0])
        act_size = int(env.action_space.n)
        env.close()

        self.obs_builder = _get_obs_builder()
        self.action_parser = LookupTableAction()
        self.obs_size = obs_size
        self.act_size = act_size

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.policy = DiscreteFF(
            obs_size,
            act_size,
            (2048, 2048, 1024, 1024),
            device,
        ).to(device)
        self.policy.load_state_dict(
            torch.load(Path(checkpoint_path) / "PPO_POLICY.pt", map_location=device)
        )
        self.policy.eval()
        self.device = device

        self._tick = 0
        self._last_action: np.ndarray = np.zeros(8, dtype=np.float32)
        self._shared_info: Dict[str, Any] = {}

    def get_output(self, packet: GamePacket) -> ControllerState:
        if len(packet.balls) == 0:
            return ControllerState()

        state = game_state_from_packet(
            packet,
            self.field_info,
            tick_count=0,
            goal_scored=False,
        )
        team = int(packet.players[self.index].team)
        agent_id = (team, 0)

        if agent_id not in state.cars:
            return ControllerState()

        # Update action every TICK_SKIP ticks (match RepeatAction(8) from training)
        if self._tick % TICK_SKIP == 0:
            obs_dict = self.obs_builder.build_obs(
                [agent_id],
                state,
                self._shared_info,
            )
            obs = obs_dict.get(agent_id)
            if obs is None:
                return _action_to_controller(self._last_action)
            obs = np.asarray(obs, dtype=np.float32)
            if obs.size != self.obs_size:
                obs = np.reshape(obs, -1)[: self.obs_size]
            if obs.size < self.obs_size:
                obs = np.pad(obs, (0, self.obs_size - obs.size), mode="constant", constant_values=0)

            with torch.no_grad():
                x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                act = self.policy.get_action(x, deterministic=True)
                # get_action may return (action,) or (action, log_prob); take first element
                if isinstance(act, tuple):
                    act = act[0]
                action_index = int(act.item())

            # Parse action index to 8-float controller (parser expects Dict[AgentID, ndarray])
            actions_dict = {agent_id: np.array([action_index], dtype=np.int64)}
            parsed_dict = self.action_parser.parse_actions(
                actions_dict,
                state,
                self._shared_info,
            )
            if agent_id in parsed_dict:
                arr = np.asarray(parsed_dict[agent_id], dtype=np.float32).flatten()
                self._last_action = arr[:8] if len(arr) >= 8 else np.pad(arr, (0, 8 - len(arr)))

        self._tick += 1
        return _action_to_controller(self._last_action)


if __name__ == "__main__":
    PPOBot("developer/ppo_bot").run()
