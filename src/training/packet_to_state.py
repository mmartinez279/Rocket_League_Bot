"""
Convert RLBot 2.0 GamePacket + FieldInfo to rlgym GameState for PPO inference.
Keeps observation/action parity with training (DefaultObs, LookupTableAction).
"""

from typing import Dict, Any

import numpy as np

from rlgym.api import AgentID
from rlgym.rocket_league.api import Car, GameConfig, GameState, PhysicsObject
from rlgym.rocket_league import common_values


def _vec3_to_np(loc) -> np.ndarray:
    """Convert flatbuffer Vector3 or (x,y,z) to float32 array."""
    return np.array([float(loc.x), float(loc.y), float(loc.z)], dtype=np.float32)


def _rotation_to_euler(rot) -> np.ndarray:
    """Convert RLBot Rotator (pitch, yaw, roll in radians) to rlgym euler order."""
    return np.array(
        [float(rot.pitch), float(rot.yaw), float(rot.roll)],
        dtype=np.float32,
    )


def _make_physics(
    position: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    euler_angles: np.ndarray,
) -> PhysicsObject:
    """Build rlgym PhysicsObject from position, velocity, angular velocity, euler."""
    from rlgym.rocket_league.api.physics_object import PhysicsObject as P
    from rlgym.rocket_league.math import euler_to_rotation

    obj = P()
    obj.position = position
    obj.linear_velocity = linear_velocity
    obj.angular_velocity = angular_velocity
    obj._euler_angles = euler_angles
    obj._rotation_mtx = None
    obj._quaternion = None
    return obj


def _make_car_from_player(player, agent_id: AgentID) -> Car:
    """Build rlgym Car from RLBot 2.0 player (flatbuffer)."""
    phys = player.physics
    position = _vec3_to_np(phys.location)
    linear_velocity = _vec3_to_np(phys.velocity)
    angular_velocity = _vec3_to_np(phys.angular_velocity)
    euler = _rotation_to_euler(phys.rotation)

    physics = _make_physics(position, linear_velocity, angular_velocity, euler)

    # Wheels: RLBot often exposes single has_wheel_contact or per-wheel
    try:
        wc = getattr(player, "wheels_with_contact", None)
        if wc is not None and len(wc) >= 4:
            wheels_with_contact = (bool(wc[0]), bool(wc[1]), bool(wc[2]), bool(wc[3]))
        else:
            has_contact = getattr(player, "has_wheel_contact", True)
            wheels_with_contact = (has_contact,) * 4
    except Exception:
        wheels_with_contact = (True, True, True, True)

    team_num = int(getattr(player, "team", 0))
    boost_amount = float(getattr(player, "boost", 0.0))
    demo_respawn_timer = float(getattr(player, "demo_respawn_timer", 0.0))

    car = Car()
    car.team_num = team_num
    car.hitbox_type = int(getattr(player, "hitbox_type", common_values.OCTANE))
    car.ball_touches = int(getattr(player, "ball_touches", 0))
    car.bump_victim_id = None
    car.demo_respawn_timer = demo_respawn_timer
    car.wheels_with_contact = wheels_with_contact
    car.supersonic_time = float(getattr(player, "supersonic_time", 0.0))
    car.boost_amount = boost_amount
    car.boost_active_time = 0.0
    car.handbrake = 0.0
    car.is_jumping = bool(getattr(player, "jumped", False))
    car.has_jumped = car.is_jumping
    car.is_holding_jump = False
    car.jump_time = 0.0
    car.has_flipped = False
    car.has_double_jumped = bool(getattr(player, "double_jumped", False))
    car.air_time_since_jump = 0.0
    car.flip_time = 1.0
    car.flip_torque = np.zeros(3, dtype=np.float32)
    car.is_autoflipping = False
    car.autoflip_timer = 0.0
    car.autoflip_direction = 1.0
    car.physics = physics
    car._inverted_physics = None
    return car


def _default_game_config() -> GameConfig:
    """Default GameConfig matching standard soccar."""
    cfg = GameConfig()
    cfg.gravity = common_values.GRAVITY
    cfg.boost_consumption = common_values.BOOST_CONSUMPTION_RATE
    cfg.dodge_deadzone = 0.5
    return cfg


def game_state_from_packet(
    packet,
    field_info,
    tick_count: int = 0,
    goal_scored: bool = False,
) -> GameState:
    """
    Build rlgym GameState from RLBot 2.0 GamePacket and FieldInfo.

    Use the same field_info and packet your bot receives in get_output().
    """
    state = GameState()
    state.tick_count = tick_count
    state.goal_scored = goal_scored
    state.config = _default_game_config()
    state.cars = {}
    state._inverted_ball = None
    state._inverted_boost_pad_timers = None

    # Ball
    if len(packet.balls) > 0:
        b = packet.balls[0].physics
        state.ball = _make_physics(
            _vec3_to_np(b.location),
            _vec3_to_np(b.velocity),
            _vec3_to_np(b.angular_velocity),
            _rotation_to_euler(b.rotation),
        )
    else:
        state.ball = _make_physics(
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )

    # Cars: use (team_num, index_in_team) as AgentID for compatibility
    blue_idx = 0
    orange_idx = 0
    for i, player in enumerate(packet.players):
        team = int(getattr(player, "team", 0))
        if team == common_values.BLUE_TEAM:
            agent_id: AgentID = (common_values.BLUE_TEAM, blue_idx)
            blue_idx += 1
        else:
            agent_id = (common_values.ORANGE_TEAM, orange_idx)
            orange_idx += 1
        state.cars[agent_id] = _make_car_from_player(player, agent_id)

    # Boost pad timers: time until pad is available (0 = available)
    n_pads = len(field_info.boost_pads) if field_info else 34
    state.boost_pad_timers = np.zeros(n_pads, dtype=np.float32)
    if hasattr(packet, "boost_pads") and packet.boost_pads is not None:
        for i, pad in enumerate(packet.boost_pads):
            if i >= n_pads:
                break
            # RLBot: is_active means available; timer often = seconds inactive
            is_active = getattr(pad, "is_active", True)
            timer = float(getattr(pad, "timer", 0.0))
            state.boost_pad_timers[i] = 0.0 if is_active else max(0.0, timer)

    return state
