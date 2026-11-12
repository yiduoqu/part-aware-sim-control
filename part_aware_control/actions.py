"""AIRBOT action contract. Metres, world-frame deltas, radians and xyzw rotations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

MAX_APERTURE_M = 0.080  # Two 40 mm slides in the shipped AIRBOT MJCF.


class InvalidPlan(ValueError):
    """A malformed or unreachable plan, distinct from a simulator/API failure."""


@dataclass(frozen=True)
class Action:
    delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    width: float | None = None  # Total pad-face separation, never half width.
    duration: float = 1.0

    def to_dict(self):
        return asdict(self)


def normalize_plan(plan) -> list[Action]:
    """Accept upstream ActionProposal or its JSON, rejecting impossible commands."""
    try:
        return _normalize_plan(plan)
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidPlan(str(error)) from error


def _normalize_plan(plan) -> list[Action]:
    if hasattr(plan, "to_dict"):
        plan = plan.to_dict()
    if isinstance(plan, dict) and "action_proposals" in plan:
        if len(plan["action_proposals"]) != 1:
            raise ValueError("replay requires exactly one selected plan")
        plan = plan["action_proposals"][0]
    sequence = plan["action_sequence"] if isinstance(plan, dict) else plan
    if not isinstance(sequence, list) or not 1 <= len(sequence) <= 60:
        raise ValueError("plan must contain 1..60 actions")
    result = []
    for raw in sequence:
        kind = raw["type"]
        delta, rotation, width = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None
        if kind in {"PUSH", "FLICK", "move"}:
            keys = ("delta_x", "delta_y", "delta_z") if kind != "PUSH" else ("delta_x", "delta_y")
            values = tuple(float(raw[k]) for k in keys)
            delta = values if len(values) == 3 else (*values, 0.0)
            if kind == "move":
                rotation = tuple(float(raw[k]) for k in ("delta_roll", "delta_pitch", "delta_yaw"))
        elif kind in {"LIFT", "DESCEND"}:
            height = float(raw["delta_z"])
            if height < 0:
                raise ValueError(f"{kind}.delta_z is a positive magnitude")
            delta = (0.0, 0.0, height if kind == "LIFT" else -height)
        elif kind in {"ROTATE", "ROLL"}:
            rotation = (0.0, 0.0, float(raw["delta_yaw"])) if kind == "ROTATE" else (float(raw["delta_roll"]), 0.0, 0.0)
        elif kind in {"GRASP", "gripper_control"}:
            width = float(raw["grasp_width"] if "grasp_width" in raw else raw["width"])
        elif kind == "RELEASE":
            width = MAX_APERTURE_M
        else:
            raise ValueError(f"unsupported action type {kind!r}")
        # Conservative fixed execution policy, shared by scripted and VLM plans.
        distance = math.sqrt(sum(v * v for v in delta))
        duration = float(raw.get("duration", max(0.8, distance / 0.06, max(map(abs, rotation)) / 0.5)))
        values = (*delta, *rotation, duration, *((width,) if width is not None else ()))
        if not all(math.isfinite(v) for v in values):
            raise ValueError("action parameters must be finite")
        if not 0.1 <= duration <= 30 or distance > 0.6 or max(map(abs, rotation)) > math.pi:
            raise ValueError("action exceeds duration, displacement or rotation limits")
        if width is not None and not 0 <= width <= MAX_APERTURE_M:
            raise ValueError(f"AIRBOT total jaw opening must be in [0, {MAX_APERTURE_M}] m")
        result.append(Action(delta=delta, rotation=rotation, width=width, duration=duration))
    if sum(a.duration for a in result) > 180:
        raise ValueError("plan exceeds 180 seconds of simulation")
    return result


def rotate_world(quat, angles):
    """Apply extrinsic world X, Y, then Z rotations to an xyzw quaternion."""

    def multiply(a, b):
        x, y, z, w = a
        i, j, k, r = b
        return (
            w * i + x * r + y * k - z * j,
            w * j - x * k + y * r + z * i,
            w * k + x * j - y * i + z * r,
            w * r - x * i - y * j - z * k,
        )

    for axis, angle in enumerate(angles):
        part = [0.0, 0.0, 0.0, math.cos(angle / 2)]
        part[axis] = math.sin(angle / 2)
        quat = multiply(part, quat)
    norm = math.sqrt(sum(v * v for v in quat))
    return tuple(v / norm for v in quat)


def width_to_command(width):
    """0=open, 1=closed. Reject instead of silently clipping an invalid aperture."""
    if not math.isfinite(width) or not 0 <= width <= MAX_APERTURE_M:
        raise ValueError("invalid total AIRBOT aperture")
    return 1.0 - width / MAX_APERTURE_M
