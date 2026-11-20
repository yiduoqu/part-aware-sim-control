# SPDX-License-Identifier: Apache-2.0
"""Plunge a drain: the AIRBOT grasps a toilet plunger by its shaft, centres the cup over a drain and pumps.

The plunger starts with its cup's edge on a flat slab and its shaft leaning on a post beside it: an
upright one tips over (Checkpoint A). The slab has a round drain hole narrower than the cup. The hand
pinches the shaft, lifts the plunger upright with the cup over the drain, lowers it, presses
``--press-mm`` past first touch so the rubber cup flattens and seals, pumps twice and holds it pressed.
``result.json`` reports the cup's compression, how much of its rim seals on the slab round the hole,
and the shaft's slip in the grip.

Why heterogeneous: the wooden shaft stays rigid and drives the cup, while the rubber cup deforms
against the slab and seals round the drain.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path

import arm_gripper
import hetero_scene
import numpy as np
import tool_scene
from loop_geometry import smooth
from scenery import revolved_obj
from scipy.spatial.transform import Rotation
from tool_asset_check import kabsch

import robosimstudio as rss
from robosimstudio.specs import AssetSpec, StaticSpec

DURATION = 8.0
SETTLE = 0.45
DRAIN_AT = (0.0, -0.12)
SLAB, FLOOR, DRAIN = 0.10, 0.012, 0.02
"""The slab's radius and thickness, the drain hole's radius [m]."""
RADIUS, RIM = 0.075, 0.14
"""The lean post's near face (x from the drain) and its height over the bench [m]."""
CUP_RADIUS, SHAFT_RADIUS, PIVOT_X = 0.055, 0.0125, -0.058
"""The cup's rim radius, the shaft's, and where the cup's edge rests on the slab (x from the drain) [m]."""
GRIP_FRAC = 0.7
PHASES = {"descend": (0.55, 1.0), "close": (1.1, 1.7), "lift": (2.0, 3.4), "lower": (3.6, 4.4)}
RELEASE = 0.4
PUMPS = ((4.6, 0.0), (5.2, 1.0), (5.6, RELEASE), (6.1, 1.0), (6.5, RELEASE), (7.0, 1.0))
"""Pump keys (time [s], depth as a fraction of ``--press-mm``): pressed, eased to 40% twice, then held."""
HOLD = (7.2, DURATION)
CLEAR = 0.035
"""The cup's rim stays this far over the slab while it moves over the drain [m]."""
SEAL_GAP = 0.005
"""A rim node seals when it is within this of the slab's top [m]; contact holds soft nodes ~2.5 mm off it."""
SUCCESS_COMPRESSION, SUCCESS_COVERAGE = 0.3, 0.9


def lean_angle() -> float:
    """The shaft's tilt [rad, about y] at which it touches the post's top edge with the cup's edge at ``PIVOT_X``."""

    def gap(theta: float) -> float:
        centre = np.array((PIVOT_X + CUP_RADIUS * np.cos(theta), FLOOR + CUP_RADIUS * np.sin(theta)))
        normal = np.array((np.cos(theta), -np.sin(theta)))
        return float((np.array((RADIUS, RIM)) - centre) @ normal) - SHAFT_RADIUS

    lo, hi = 0.1, 1.2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def scenery(scene) -> list:
    # A revolved mesh's faces point inward and do not render: one copy collides, a reversed one is drawn.
    profile = [(DRAIN, 0.0), (DRAIN, FLOOR), (SLAB, FLOOR), (SLAB, 0.0), (DRAIN, 0.0)]
    xform = (DRAIN_AT[0], DRAIN_AT[1], 0.0, 0.0, 0.0, 0.0, 1.0)
    slab = StaticSpec(
        name="drain_slab",
        asset=AssetSpec(path=str(revolved_obj("drain_slab", profile, segments=48))),
        xform=xform,
        color=(0.70, 0.71, 0.73),
    )
    shown = dataclasses.replace(
        slab,
        name="drain_slab_shown",
        asset=AssetSpec(path=str(revolved_obj("drain_slab_out", profile[::-1], segments=48))),
        collide=False,
    )
    post = rss.slab(
        name="lean_post",
        half=(0.0075, 0.025, 0.5 * RIM),
        at=(DRAIN_AT[0] + RADIUS + 0.0075, DRAIN_AT[1], 0.5 * RIM),
        color=(0.45, 0.47, 0.50),
    )
    return [slab, shown, post]


def kinematics(grasp: dict, theta: float, rim_centre: np.ndarray, floor_z: float, press: float, touch: float):
    """The hand's ``tcp_at``, ``quat_at`` for a plunger held rigidly from its resting lean.

    The cup's rim centre ``C(t)`` goes from where it rests to over the drain, the lean ``phi`` from
    ``theta`` to upright; the hand stays at the grip point ``C + R_y(phi) g`` (``g`` the grip's offset
    in the plunger frame). ``touch`` is ``C.z`` over the slab with the upright cup's rim on it.
    """
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    point = np.asarray(grasp["point_m"])
    ry = lambda a: Rotation.from_rotvec((0.0, a, 0.0))  # noqa: E731
    offset = ry(-theta).apply(point - rim_centre)

    def pumped(t: float) -> float:
        """How far the cup is pressed past first touch [m]: pump up and down between the keys."""
        for (t0, d0), (t1, d1) in itertools.pairwise(PUMPS):
            if t < t1:
                return (d0 + (d1 - d0) * smooth(t, t0, t1)) * press if t >= t0 else 0.0
        return PUMPS[-1][1] * press

    def centre(t: float) -> np.ndarray:
        up, down = smooth(t, *PHASES["lift"]), smooth(t, *PHASES["lower"])
        high = np.array((DRAIN_AT[0], DRAIN_AT[1], floor_z + touch + CLEAR))
        c = rim_centre + (high - rim_centre) * up
        return c - np.array((0.0, 0.0, CLEAR * down * up + pumped(t)))

    def phi(t: float) -> float:
        return theta * (1.0 - smooth(t, *PHASES["lift"]))

    def tcp_at(t: float) -> np.ndarray:
        hover = np.array((0.0, 0.0, 0.05 * (1.0 - smooth(t, *PHASES["descend"]))))
        return surface + centre(t) + ry(phi(t)).apply(offset) + hover

    return tcp_at, tool_scene.tilt_quat(grasp, lambda t: phi(t) - theta)


def run(out: Path, asset, args) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    theta = lean_angle()
    asset = dataclasses.replace(
        asset, quat=tuple(arm_gripper.axis_angle_quat((0.0, 1.0, 0.0), theta)), grip_frac=GRIP_FRAC
    )
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    floor_z = FLOOR
    vertices, tets, labels = hetero_scene.load(asset)
    base = tool_scene.rest_pose(
        vertices, asset.quat, at=(DRAIN_AT[0] + PIVOT_X + CUP_RADIUS * np.cos(theta), DRAIN_AT[1])
    )
    pose = (base[0], base[1], base[2] + floor_z)
    sim, obj, camera, vertices, tets, labels, settings = tool_scene.build_scene(
        asset,
        name="tool_plunge_drain",
        pose=pose,
        extras=scenery,
        # Low (~15 deg) from the side opposite the post, the arm's base off to the right of the frame.
        camera=((-0.32, DRAIN_AT[1] + 0.10, 0.12), (DRAIN_AT[0], DRAIN_AT[1], 0.04)),
        **{k: getattr(args, k) for k in ("solver", "iterations", "mu", "grip_mu", "force_limit", "self_contact")},
    )
    entity = obj.entity
    src_tets = entity.source_tets
    soft = hetero_scene.soft_labels(asset, labels)
    handle = np.unique(src_tets[labels == asset.handle])
    cup = np.unique(src_tets[np.isin(labels, soft)])
    rim_nodes = np.unique(src_tets[labels == 2])
    seal_nodes = rim_nodes[vertices[rim_nodes, 2] < vertices[cup, 2].min() + 0.002]
    arm = arm_gripper.ArmControl(sim)
    capacity = max(args.steps, round(DURATION * tool_scene.CONTROL_HZ))
    half_open = 0.04
    half_close = [None]

    def half_gap_at(t: float) -> float:
        return half_open + (half_close[0] - half_open) * smooth(t, *PHASES["close"])

    def plan(grasp, rim_centre, theta_now, touch):
        tcp_at, quat_at = kinematics(grasp, theta_now, rim_centre, floor_z, args.press_mm * 1e-3, touch)
        route = arm.plan(
            quat=quat_at, capacity=capacity, dt=1.0 / tool_scene.CONTROL_HZ, tcp_at=tcp_at, half_gap_at=half_gap_at
        )
        if route.worst_position_error_m > 0.025:
            raise RuntimeError(f"the arm misses the path by {route.worst_position_error_m:.3f} m")
        return route

    pose_local = np.asarray(pose)
    q_start = arm_gripper._rotate(np.asarray(asset.quat), vertices) + pose_local
    grasp = tool_scene.grasp_handle(asset, vertices, tets, labels, pose=pose_local)
    half_close[0] = grasp["half_width_m"] - 0.0015
    rim_centre = q_start[rim_nodes].mean(axis=0)
    # Upright, the rim centre's height over the cup's lowest point.
    touch = float(vertices[rim_nodes, 2].mean() - vertices[cup, 2].min())
    route = plan(grasp, rim_centre, theta, touch)
    arm.start(route)
    settle = round(SETTLE * tool_scene.CONTROL_HZ)
    state = {"ref": None, "route": route}
    handle_rest = q_start[handle]
    grip_frame = round(PHASES["lift"][0] * tool_scene.CONTROL_HZ)

    def follow(frame: int):
        if frame == settle:
            # Re-grasp where the plunger came to rest: its shaft's axis and its cup's rim centre from the nodes.
            now = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
            shaft = now[handle] - now[handle].mean(axis=0)
            axis = np.linalg.svd(shaft, full_matrices=False)[2][0]
            axis = axis if axis[2] > 0 else -axis
            lean = float(np.arctan2(axis[0], axis[2]))
            grasp_now = tool_scene.grasp_handle(asset, now, src_tets, labels, axis=axis)
            half_close[0] = grasp_now["half_width_m"] - 0.0015
            centre_now = now[rim_nodes].mean(axis=0)
            state["route"] = plan(grasp_now, centre_now, lean, touch)
            print(json.dumps({"settled": {"lean_deg": float(np.degrees(lean)), "axis": axis.tolist()}}), flush=True)
        arm.follow(state["route"], frame)
        return arm

    def task_row(frame: int, local: np.ndarray, body_q: np.ndarray) -> dict:
        hand_p, hand_q = arm.arm.ee_pose(sim.pipeline.state_0)
        hand = Rotation.from_quat(np.asarray(hand_q, dtype=np.float64))
        tool = kabsch(handle_rest, local[handle])
        offset = tool.apply(-handle_rest.mean(axis=0)) + local[handle].mean(axis=0)
        in_hand = (hand.inv().apply(offset - (np.asarray(hand_p) - surface)), hand.inv() * tool)
        if frame == grip_frame:
            state["ref"] = in_hand
        rim = local[seal_nodes]
        r = np.hypot(rim[:, 0] - DRAIN_AT[0], rim[:, 1] - DRAIN_AT[1])
        row = {
            "cup_height_m": float(local[cup, 2].max() - local[cup, 2].min()),
            "cup_bottom_m": float(local[cup, 2].min()),
            "coverage": float(np.mean((r > DRAIN) & (rim[:, 2] - FLOOR < SEAL_GAP))),
        }
        if state["ref"] is not None:
            row["slip_mm"] = float(1e3 * np.linalg.norm(in_hand[0] - state["ref"][0]))
            row["slip_deg"] = float(np.degrees((state["ref"][1].inv() * in_hand[1]).magnitude()))
        return row

    rows, elapsed, video_error = hetero_scene.episode(
        sim,
        entity,
        camera,
        out,
        soft=soft,
        steps=args.steps,
        render_every=args.render_every,
        control_hz=tool_scene.CONTROL_HZ,
        follow=follow,
        task_row=task_row,
    )
    hz = tool_scene.CONTROL_HZ
    rest_height = float(vertices[cup, 2].max() - vertices[cup, 2].min())
    held = [r for r in rows if HOLD[0] * hz <= r["frame"]]
    pressing = [r for r in rows if PUMPS[0][0] * hz <= r["frame"]]
    compression = rest_height - float(np.mean([r["cup_height_m"] for r in held])) if held else None
    coverage = float(np.mean([r["coverage"] for r in held])) if held else None
    slips = [r for r in rows if "slip_mm" in r]
    fraction = None if compression is None else compression / rest_height
    result = {
        "lean_deg": float(np.degrees(theta)),
        "grasp": grasp,
        "cup_rest_height_mm": 1e3 * rest_height,
        "cup_compression_mm": None if compression is None else 1e3 * compression,
        "cup_compression_frac": fraction,
        "cup_compression_max_mm": 1e3 * (rest_height - min((r["cup_height_m"] for r in pressing), default=rest_height)),
        "drain_coverage": coverage,
        "slip_max_mm": max((r["slip_mm"] for r in slips), default=None),
        "slip_max_deg": max((r["slip_deg"] for r in slips), default=None),
        "slip_final_mm": slips[-1]["slip_mm"] if slips else None,
        "finger_contacts_max": max(r["finger_contacts"] for r in rows),
        "max_inverted_tetrahedra": max(r["inverted_tetrahedra"] for r in rows),
        "max_soft_inverted": max(r["cable_inverted"] for r in rows),
        "min_z_m": min(r["minimum_z_m"] for r in rows),
        "arm_path_error_m": state["route"].worst_position_error_m,
        "steps": args.steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "config": tool_scene.config(args, asset, settings),
    }
    result["score"] = None if held == [] else min(1.0, fraction / SUCCESS_COMPRESSION) * coverage
    result["success"] = bool(held and fraction >= SUCCESS_COMPRESSION and coverage >= SUCCESS_COVERAGE)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("grasp", "config")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=tool_scene.OUTPUT / "tool_plunge_drain" / "run")
    parser.add_argument("--steps", type=int, default=round(DURATION * tool_scene.CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--press-mm", type=float, default=25.0, help="how far the cup is pressed past first touch [mm]")
    tool_scene.add_arguments(parser)
    parser.set_defaults(force_limit=100.0)  # q5: at 50 N the shaft slid 32 mm up through the pads
    args = parser.parse_args()
    asset = hetero_scene.with_levers(tool_scene.TOOLS["toilet_plunger"], args)
    run(args.out, asset, args)


if __name__ == "__main__":
    main()
