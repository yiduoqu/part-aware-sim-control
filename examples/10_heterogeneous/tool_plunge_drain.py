# SPDX-License-Identifier: Apache-2.0
"""Plunge a drain: the AIRBOT grasps a toilet plunger by its shaft, centres the cup over a clogged drain and pumps.

The plunger starts in a raised basin (a concave static mesh with a through drain), the cup's edge on
the floor and its shaft leaning on the rim: an upright one tips over (Checkpoint A). A soft ball
(the clog) sits in the drain's mouth, 8 mm proud of the floor. The hand pinches the shaft, lifts the
plunger upright with the cup over the drain, lowers it, presses ``--press-mm`` past first touch so the
rubber cup flattens, and pumps ``PUMPS`` times. There is no fluid, so any push of the clog is
mechanical: the cup's collapsing top and the ball squeezed against the drain.

Why heterogeneous: the wooden shaft stays rigid and drives the cup, while the rubber cup deforms
against the floor and seals round the drain.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import tempfile
from pathlib import Path

import arm_gripper
import cable_scene
import numpy as np
import tool_scene
from loop_geometry import smooth
from scenery import GLASS, revolved_obj
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

import robosimstudio as rss
from robosimstudio.specs import AssetSpec

DURATION = 9.2
SETTLE = 0.45
BASIN_AT = (0.0, -0.12)
RADIUS, RIM, DRAIN, FLOOR, LIFT_Z = 0.075, 0.14, 0.02, 0.012, 0.05
"""Basin: inner radius, rim height over its foot, drain radius, floor thickness; the foot's height over the bench [m]."""
CUP_RADIUS, SHAFT_RADIUS, PIVOT_X = 0.055, 0.0125, -0.058
"""The cup's rim radius, the shaft's, and where the cup's edge rests in the basin (x from its centre) [m]."""
CLOG_RADIUS, CLOG_YOUNG = 0.030, 2.0e4
GRIP_FRAC = 0.7
PHASES = {"descend": (0.55, 1.0), "close": (1.1, 1.7), "lift": (2.0, 3.4), "lower": (3.6, 4.4)}
RELEASE = 0.4
PUMPS = ((4.6, 0.0), (5.3, 1.0), (5.7, RELEASE), (6.3, 1.0), (6.7, RELEASE), (7.3, 1.0), (7.7, RELEASE))
"""Pump keys (time [s], depth as a fraction of ``--press-mm``): three presses, each released to 40%."""
EXIT = (8.0, 8.8)
CLEAR = 0.035
"""The cup's rim stays this far over the floor while it moves over the drain [m]."""
PAD_HALF = 0.02


def lean_angle() -> float:
    """The shaft's tilt [rad, about y] at which it touches the basin rim with the cup's edge at ``PIVOT_X``."""

    def gap(theta: float) -> float:
        centre = np.array((PIVOT_X + CUP_RADIUS * np.cos(theta), FLOOR + CUP_RADIUS * np.sin(theta)))
        normal = np.array((np.cos(theta), -np.sin(theta)))
        return float((np.array((RADIUS, RIM)) - centre) @ normal) - SHAFT_RADIUS

    lo, hi = 0.1, 1.2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def ball_mesh(radius: float, spacing: float = 0.0075) -> Path:
    """A tet ball (Delaunay of Fibonacci shells, hull slivers dropped) as an npz of vertices and tets."""
    points = [np.zeros(3)]
    for r in np.arange(radius, 0.5 * spacing, -spacing):
        n = max(6, round(4.0 * np.pi * r * r / (0.87 * spacing**2)))
        i = np.arange(n) + 0.5
        z, phi = 1.0 - 2.0 * i / n, np.pi * (1.0 + 5.0**0.5) * i
        s = np.sqrt(1.0 - z * z)
        points.extend(r * np.column_stack((s * np.cos(phi), s * np.sin(phi), z)))
    vertices = np.asarray(points)
    tets = Delaunay(vertices).simplices
    edges = vertices[tets[:, 1:]] - vertices[tets[:, :1]]
    volume = np.linalg.det(edges) / 6.0
    flip = volume < 0.0
    tets[flip, 0], tets[flip, 1] = tets[flip, 1].copy(), tets[flip, 0].copy()
    volume = np.abs(volume)
    keep = volume > 0.05 * (spacing**3 / 6.0)
    path = Path(tempfile.gettempdir()) / f"plunge_clog_{radius:.4f}.npz"
    np.savez(path, vertices=vertices, tets=tets[keep])
    return path


def clog_centre() -> np.ndarray:
    """The clog's centre [m, bench frame]: a ball wider than the drain sits in its mouth, a millimetre up."""
    depth = float(np.sqrt(CLOG_RADIUS**2 - DRAIN**2))
    return np.array((BASIN_AT[0], BASIN_AT[1], LIFT_Z + FLOOR - depth + 0.001))


def scenery(scene) -> list:
    # tool_scene.basin's faces point inward, so it renders as nothing: it keeps colliding, and a
    # reversed copy that touches nothing is drawn, glassy so the cup shows through.
    outer = RADIUS + 0.006
    profile = [(DRAIN, 0.0), (DRAIN, FLOOR), (RADIUS, FLOOR), (RADIUS, RIM), (outer, RIM), (outer, 0.0), (DRAIN, 0.0)]
    collider = tool_scene.basin(at=BASIN_AT, radius=RADIUS, rim=RIM, drain=DRAIN, floor=FLOOR)
    xform = (BASIN_AT[0], BASIN_AT[1], LIFT_Z, 0.0, 0.0, 0.0, 1.0)
    basin = dataclasses.replace(collider, xform=xform)
    shown = dataclasses.replace(
        basin,
        name="basin_shown",
        asset=AssetSpec(path=str(revolved_obj("basin_out", profile[::-1]))),
        collide=False,
        finish=GLASS,
    )
    posts = [
        rss.slab(
            name=f"basin_leg{i}",
            half=(0.012, 0.012, 0.5 * LIFT_Z),
            at=(BASIN_AT[0] + sx * 0.085, BASIN_AT[1] + sy * 0.085, 0.5 * LIFT_Z),
            color=(0.45, 0.47, 0.50),
        )
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))
    ]
    clog = rss.SoftSpec(
        name="clog",
        asset=AssetSpec(path=str(ball_mesh(CLOG_RADIUS))),
        pose=tuple(float(c) for c in clog_centre()),
        material=rss.SoftMaterialSpec(young=CLOG_YOUNG, poisson=0.35, density=1000.0, friction=0.5),
        settle_frames=0,
        color=(0.30, 0.22, 0.12),
    )
    return [basin, shown, *posts, clog]


def kinematics(grasp: dict, theta: float, rim_centre: np.ndarray, floor_z: float, press: float, touch: float):
    """The hand's ``tcp_at``, ``quat_at`` for a plunger held rigidly from its resting lean.

    The cup's rim centre ``C(t)`` goes from where it rests to over the drain, the lean ``phi`` from
    ``theta`` to upright; the hand stays at the grip point ``C + R_y(phi) g`` (``g`` the grip's offset
    in the plunger frame). ``touch`` is ``C.z`` with the cup's edge on the floor.
    """
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
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
        high = np.array((BASIN_AT[0], BASIN_AT[1], floor_z + touch + CLEAR))
        c = rim_centre + (high - rim_centre) * up
        return c - np.array((0.0, 0.0, CLEAR * down * up + pumped(t) - 0.07 * smooth(t, *EXIT)))

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
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    floor_z = LIFT_Z + FLOOR
    vertices, tets, labels = cable_scene.load(asset)
    base = tool_scene.rest_pose(
        vertices, asset.quat, at=(BASIN_AT[0] + PIVOT_X + CUP_RADIUS * np.cos(theta), BASIN_AT[1])
    )
    pose = (base[0], base[1], base[2] + floor_z)
    sim, obj, camera, vertices, tets, labels, settings = tool_scene.build_scene(
        asset,
        name="tool_plunge_drain",
        pose=pose,
        extras=scenery,
        camera=((0.50, -0.62, 0.42), (BASIN_AT[0], BASIN_AT[1] + 0.02, 0.12)),
        **{k: getattr(args, k) for k in ("solver", "iterations", "mu", "grip_mu", "force_limit", "self_contact")},
    )
    entity = obj.entity
    src_tets = entity.source_tets
    soft = cable_scene.soft_labels(asset, labels)
    handle = np.unique(src_tets[labels == asset.handle])
    cup = np.unique(src_tets[np.isin(labels, soft)])
    rim_nodes = np.unique(src_tets[labels == 2])
    clog = sim.handles.deformables["clog"].particles
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
    touch = float(rim_centre[2] - q_start[cup, 2].min())
    route = plan(grasp, rim_centre, theta, touch)
    arm.start(route)
    settle = round(SETTLE * tool_scene.CONTROL_HZ)
    state = {"base": None, "route": route}
    clog_rest = clog_centre()

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
            state["route"] = plan(grasp_now, centre_now, lean, float(centre_now[2] - now[cup, 2].min()))
            print(json.dumps({"settled": {"lean_deg": float(np.degrees(lean)), "axis": axis.tolist()}}), flush=True)
        arm.follow(state["route"], frame)
        return arm

    def task_row(frame: int, local: np.ndarray, body_q: np.ndarray) -> dict:
        row = {
            "cup_height_m": float(local[cup, 2].max() - local[cup, 2].min()),
            "cup_bottom_m": float(local[cup, 2].min()),
        }
        c = sim.pipeline.state_0.particle_q.numpy()[clog] - surface
        row["clog_centre_m"] = c.mean(axis=0).tolist()
        row["clog_top_m"] = float(c[:, 2].max())
        return row

    rows, elapsed, video_error = cable_scene.episode(
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
    before = [r for r in rows if PHASES["lower"][1] * hz - 12 <= r["frame"] <= PHASES["lower"][1] * hz]
    rest_height = float(np.mean([r["cup_height_m"] for r in before])) if before else None
    pumping = [r for r in rows if PUMPS[0][0] * hz <= r["frame"] <= PUMPS[-1][0] * hz]
    compression = None if rest_height is None or not pumping else rest_height - min(r["cup_height_m"] for r in pumping)
    clog_rows = [r for r in rows if "clog_top_m" in r]
    result = {
        "lean_deg": float(np.degrees(theta)),
        "grasp": grasp,
        "cup_rest_height_mm": None if rest_height is None else 1e3 * rest_height,
        "cup_compression_mm": None if compression is None else 1e3 * compression,
        "clog_start_centre_m": clog_rest.tolist(),
        "clog_final_centre_m": clog_rows[-1]["clog_centre_m"] if clog_rows else None,
        "clog_final_top_m": clog_rows[-1]["clog_top_m"] if clog_rows else None,
        "drain_plane_m": LIFT_Z,
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
    # Pushed through: the clog's top has dropped below the basin floor's underside.
    result["score"] = None if not clog_rows else float(clog_rest[2] - clog_rows[-1]["clog_centre_m"][2])
    result["success"] = bool(
        clog_rows
        and clog_rows[-1]["clog_top_m"] < LIFT_Z
        and (compression or 0.0) > 0.005
        and result["max_soft_inverted"] == 0
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("grasp", "config")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=tool_scene.OUTPUT / "tool_plunge_drain" / "run")
    parser.add_argument("--steps", type=int, default=round(DURATION * tool_scene.CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--press-mm", type=float, default=30.0, help="how far the cup is pressed past first touch [mm]")
    tool_scene.add_arguments(parser)
    args = parser.parse_args()
    asset = cable_scene.with_levers(tool_scene.TOOLS["toilet_plunger"], args)
    run(args.out, asset, args)


if __name__ == "__main__":
    main()
