# SPDX-License-Identifier: Apache-2.0
"""TT2: the AIRBOT erases hand-drawn marks from a whiteboard with a felt eraser.

The eraser lies on a whiteboard on the bench. The hand grips its palm grip, lifts it over the marks,
presses the felt pad (``--press-mm`` past first touch) and wipes in overlapping strokes. A mark is
rubbed off where the pad presses and moves over it (:class:`tool_scene.StainMap`). Score: the fraction
of marks erased; success at 80 percent.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import arm_gripper
import cable_scene
import numpy as np
import tool_scene
from clawhauser_scene import REFERENCE
from loop_geometry import smooth

CENTRE = np.array((0.0, -0.10))
"""The board's and the marks' centre [m, bench frame]: clear of the arm's riser."""
BOARD, BOARD_TOP = (0.30, 0.20), 0.008
START = np.array((0.0, -0.17))
"""Where the eraser lies, in front of the marks."""
STROKE = 0.03
"""The wipe's half stroke along x [m]; with the pad's 110 mm it covers the marks' 150 mm."""
ROWS = (-0.02, 0.0, 0.02, 0.0)
"""The strokes' y offsets [m]: the pad is 48 mm wide, so neighbouring strokes overlap."""
WIPE = [(x, row) for i, row in enumerate(ROWS) for x in ((-STROKE, STROKE) if i % 2 == 0 else (STROKE, -STROKE))]
"""Stroke ends [m]: each row swept along x, alternating direction."""
SEGMENT = 1.3
"""Seconds per stroke."""
GRIP_ABOVE = 0.037
"""Pad-bottom-to-hand-centre [m]: the hand's 40 mm pads sit just over the backing, on the grip."""
HOVER, LIFT = 0.04, 0.012
T_DESCEND, T_CLOSE, T_LIFT, T_MOVE, T_DOWN = (0.55, 1.0), (1.1, 1.7), (1.9, 2.5), (2.5, 3.4), (3.4, 4.0)
T_WIPE = 4.3
T_END = T_WIPE + SEGMENT * (2 * len(ROWS) - 1)
T_UP = (T_END + 0.2, T_END + 0.8)
DURATION = T_UP[1] + 0.4
SETTLE = 0.4
SQUEEZE = 0.016
TILE, SPACING = 0.005, 0.0035


def marks() -> np.ndarray:
    """Hand-drawn-looking marker strokes (n, 2) [m] about the origin: letters, a circle, an arrow, a squiggle, a zigzag."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, 2 * np.pi, 40)
    lines = [
        [(-70, -5), (-69, 25)],
        [(-53, -4), (-54, 25)],
        [(-70, 10), (-53, 11)],  # H
        [(-42, -5), (-42, 12)],
        [(-42, 19), (-42, 20)],  # i
        np.c_[-5 + 15 * np.cos(t) * (1 + 0.1 * np.sin(2 * t)), 10 + 13 * np.sin(t)],  # circle
        [(18, 10), (48, 12)],
        [(40, 19), (48, 12), (40, 4)],  # arrow
        np.c_[np.linspace(-70, 62, 60), -25 + 4 * np.sin(np.linspace(0, 7 * np.pi, 60))],  # underline
        [(58, -8), (62, 24), (67, -6), (72, 25)],  # zigzag
    ]
    points = []
    for stroke in lines:
        line = np.asarray(stroke, dtype=np.float64) * 1e-3
        for a, b in pairwise(line):
            n = max(2, int(np.linalg.norm(b - a) / SPACING))
            points += list(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
        points.append(line[-1])
    points = np.asarray(points) + rng.normal(0, 0.0003, (len(points), 2))
    keep = []
    for p in points:
        if all(np.linalg.norm(p - q) > 0.8 * SPACING for q in keep):
            keep.append(p)
    return np.asarray(keep)


def route(centre0, grasp, offset, press):
    """``tcp_at(t)`` over the grasp: lift, carry to the first stroke, press, wipe, lift off."""
    point = np.asarray(grasp["point_m"], dtype=np.float64)
    path = [centre0] + [CENTRE + p for p in WIPE]

    def tcp_at(t: float) -> np.ndarray:
        xy = centre0 + (path[1] - centre0) * smooth(t, *T_MOVE)
        k = int((t - T_WIPE) // SEGMENT)
        if t >= T_WIPE:
            k = min(k, len(path) - 3)
            xy = path[1 + k] + (path[2 + k] - path[1 + k]) * smooth(t, T_WIPE + k * SEGMENT, T_WIPE + (k + 1) * SEGMENT)
        up = HOVER * (1 - smooth(t, *T_DESCEND)) + LIFT * smooth(t, *T_LIFT) * (1 - smooth(t, *T_DOWN))
        up += -press * smooth(t, *T_DOWN) + 0.025 * smooth(t, *T_UP)
        return np.array((*(xy + offset), point[2] + up))

    return tcp_at


def run(out: Path, asset, args) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    vertices, _, labels = cable_scene.load(asset)
    pose = tool_scene.rest_pose(vertices, asset.quat, at=START)
    pose = (pose[0], pose[1], pose[2] + BOARD_TOP)
    centres = CENTRE + marks()

    def extras(scene):
        scene.add(tool_scene.whiteboard(at=CENTRE, size=BOARD, thickness=BOARD_TOP))
        return stain.specs()

    stain = tool_scene.StainMap(
        points=centres,
        z=BOARD_TOP,
        radius=REFERENCE.particle_radius,
        contact_ke=REFERENCE.contact_ke,
        tile=TILE,
        force_min=args.force_min,
        reach=args.reach_mm * 1e-3,
        half_z=0.00005,
    )
    sim, obj, camera, vertices, _, labels, settings = tool_scene.build_scene(
        asset,
        name="tool_whiteboard_erase",
        pose=pose,
        extras=extras,
        camera=((0.45, -0.55, 0.40), (0.0, -0.10, 0.03)),
        **{k: getattr(args, k) for k in ("solver", "iterations", "mu", "grip_mu", "force_limit", "self_contact")},
    )
    stain.make_live(sim)
    entity = obj.entity
    src_tets = entity.source_tets
    soft = cable_scene.soft_labels(asset, labels)
    soft_nodes = np.unique(src_tets[np.isin(labels, soft)])
    arm = arm_gripper.ArmControl(sim)
    capacity = max(args.steps, round(DURATION * tool_scene.CONTROL_HZ))

    def plan(grasp, centre0, offset):
        tcp_at = route(centre0, grasp, offset, args.press_mm * 1e-3)
        half_open = min(0.04, grasp["half_width_m"] + 0.014)
        half_close = grasp["half_width_m"] - SQUEEZE
        half_gap_at = lambda t: half_open + (half_close - half_open) * smooth(t, *T_CLOSE)  # noqa: E731
        plan_ = arm.plan(
            quat=np.asarray(grasp["quat"]),
            capacity=capacity,
            dt=1.0 / tool_scene.CONTROL_HZ,
            tcp_at=lambda t: tcp_at(t) + surface,
            half_gap_at=half_gap_at,
        )
        if plan_.worst_position_error_m > 0.025:
            raise RuntimeError(f"the arm misses the path by {plan_.worst_position_error_m:.3f} m")
        return plan_

    def grasp_of(local):
        grasp = tool_scene.grasp_handle(asset, local, src_tets, labels, axis=np.array((1.0, 0.0, 0.0)))
        centre0 = local[soft_nodes, :2].mean(axis=0)
        grasp["point_m"][2] = float(local[soft_nodes, 2].min()) + GRIP_ABOVE
        return grasp, centre0, np.asarray(grasp["point_m"][:2]) - centre0

    first = np.asarray(vertices, dtype=np.float64) + pose
    state = {"grasp": grasp_of(first)}
    route_ = plan(*state["grasp"])
    arm.start(route_)
    settle = round(SETTLE * tool_scene.CONTROL_HZ)

    def follow(frame: int):
        nonlocal route_
        if frame == settle:
            now = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
            state["grasp"] = grasp_of(now)
            route_ = plan(*state["grasp"])
        arm.follow(route_, frame)
        return arm

    def task_row(frame, local, body_q):
        pressed = local[soft_nodes]
        if frame >= settle:
            new = stain.update(pressed)
            if len(new):
                stain.hide(sim, new)
        return {"erased_fraction": stain.fraction_erased()}

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
    score = stain.fraction_erased()
    result = {
        "asset": "whiteboard_eraser",
        "marks": len(stain.points),
        "erased": int(stain.erased.sum()),
        "score": score,
        "success": bool(score >= 0.8),
        "max_inverted_tetrahedra": max(r["inverted_tetrahedra"] for r in rows),
        "max_soft_inverted": max(r["cable_inverted"] for r in rows),
        "min_finger_contacts_while_wiping": min(
            (r["finger_contacts"] for r in rows if T_WIPE * tool_scene.CONTROL_HZ <= r["frame"]), default=None
        ),
        "arm_path_error_m": route_.worst_position_error_m,
        "steps": args.steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "config": tool_scene.config(args, asset, settings),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "config"}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=tool_scene.OUTPUT / "tool_whiteboard_erase" / "run")
    parser.add_argument("--steps", type=int, default=round(DURATION * tool_scene.CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--press-mm", type=float, default=1.5, help="how far the hand is lowered past first touch [mm]")
    parser.add_argument("--force-min", type=float, default=0.2, help="per-particle contact force that rubs a mark [N]")
    parser.add_argument("--reach-mm", type=float, default=8.0, help="how far from a pressed node a mark is rubbed [mm]")
    tool_scene.add_arguments(parser)
    args = parser.parse_args()
    run(args.out, cable_scene.with_levers(tool_scene.TOOLS["whiteboard_eraser"], args), args)


if __name__ == "__main__":
    main()
