# SPDX-License-Identifier: Apache-2.0
"""Checkpoint A: the AIRBOT lifts a tool by its handle, presses its soft part on the bench and lifts again.

The tool lies on the bench as its asset is built. The hand pinches the handle (``--force-limit`` N, pad
mu ``--grip-mu``), lifts it, lowers it ``--press-mm`` past where its soft part touched the bench,
holds and lifts. ``result.json`` reports what a tool task needs from the asset: how far the soft part
compresses, how far the handle slips in the grip (mm and degrees), inverted tets (soft and rigid) and
the soft part's stretch.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arm_gripper
import hetero_scene
import numpy as np
import tool_scene
from loop_geometry import smooth
from scipy.spatial.transform import Rotation

DURATION = 8.0
SETTLE = 0.4
"""The tool settles on the bench this long [s] (the pads hover) before the grasp is measured: as built a handle can stand off the bench."""
HOVER, LIFT, SQUEEZE = 0.04, 0.06, 0.016
PHASES = {"descend": (0.55, 1.0), "close": (1.1, 1.7), "lift": (2.0, 3.2), "press": (3.7, 4.7), "release": (6.2, 7.2)}
"""Phase windows [s]: hold from 4.7 s to 6.2 s under the press, then lift off."""
PAD_HALF = 0.02
"""The hand's pad half-height [m]; the pinch stays this far over the bench."""


def kabsch(rest: np.ndarray, now: np.ndarray) -> Rotation:
    a, b = rest.mean(axis=0), now.mean(axis=0)
    u, _, vt = np.linalg.svd((rest - a).T @ (now - b))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return Rotation.from_matrix(vt.T @ np.diag((1.0, 1.0, d)) @ u.T)


def level_about(soft: np.ndarray, pivot: np.ndarray, across: np.ndarray):
    """The height the pivot must rise [m] for the soft nodes' lowest point to stay put after turning ``angle`` about ``across``."""
    relative, floor = soft - pivot, float(soft[:, 2].min())

    def rise(angle: float) -> float:
        return floor - float((Rotation.from_rotvec(across * angle).apply(relative)[:, 2] + pivot[2]).min())

    return rise


def motion(grasp: dict, press: float, surface: np.ndarray, soft: np.ndarray, level: float):
    """``tcp_at``, ``quat_at`` and ``half_gap_at``; the hand turns ``level`` [rad] about the closing axis as it lifts.

    The turn levels a tool that rests tilted (a wand with its handle on the bench), so its soft face lands flat.
    """
    point = surface + np.array(grasp["point_m"])
    point[2] = max(point[2], surface[2] + PAD_HALF + 0.005)
    rise = level_about(soft + surface, point, np.asarray(grasp["across"]))
    angle_at = lambda t: level * smooth(t, *PHASES["lift"])  # noqa: E731
    open_half = min(0.04, grasp["half_width_m"] + 0.014)
    half_close = grasp["half_width_m"] - SQUEEZE

    def tcp_at(t: float) -> np.ndarray:
        up = LIFT * smooth(t, *PHASES["lift"]) * (1.0 - smooth(t, *PHASES["press"])) - press * smooth(
            t, *PHASES["press"]
        )
        up = up * (1.0 - smooth(t, *PHASES["release"])) + LIFT * smooth(t, *PHASES["release"])
        return point + np.array((0.0, 0.0, HOVER * (1.0 - smooth(t, *PHASES["descend"])) + up + rise(angle_at(t))))

    def half_gap_at(t: float) -> float:
        return open_half + (half_close - open_half) * smooth(t, *PHASES["close"])

    return tcp_at, tool_scene.tilt_quat(grasp, angle_at), half_gap_at


def run(out: Path, asset, args) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    vertices, tets, labels = hetero_scene.load(asset)
    grip = tool_scene.grasp_handle(asset, vertices, tets, labels)
    # The grip lands at x = 0, y = -0.10: clear of the arm's riser.
    pose = tool_scene.rest_pose(vertices, asset.quat, at=(-grip["point_m"][0], -0.10 - grip["point_m"][1]))
    sim, obj, camera, vertices, tets, labels, settings = tool_scene.build_scene(
        asset,
        name="tool_asset_check",
        pose=pose,
        camera=((0.45, -0.55, 0.40), (0.0, -0.10, 0.04)),
        **{k: getattr(args, k) for k in ("solver", "iterations", "mu", "grip_mu", "force_limit", "self_contact")},
    )
    grasp = tool_scene.grasp_handle(asset, vertices, tets, labels, pose=pose)
    entity = obj.entity
    src_tets = entity.source_tets
    soft = hetero_scene.soft_labels(asset, labels)
    rigid = np.unique(src_tets[~np.isin(labels, soft)])
    handle = np.unique(src_tets[labels == asset.handle])
    pad = np.unique(src_tets[labels == asset.pad]) if asset.pad >= 0 else handle
    soft_nodes = np.unique(src_tets[np.isin(labels, soft)])
    arm = arm_gripper.ArmControl(sim)
    q_soft = arm_gripper._rotate(np.asarray(asset.quat), vertices)[soft_nodes] + pose
    capacity = max(args.steps, round(DURATION * tool_scene.CONTROL_HZ))

    def plan(grasp: dict, soft: np.ndarray, level: float):
        tcp_at, quat_at, half_gap_at = motion(grasp, args.press_mm * 1e-3, surface, soft, level)
        route = arm.plan(
            quat=quat_at,
            capacity=capacity,
            dt=1.0 / tool_scene.CONTROL_HZ,
            tcp_at=tcp_at,
            half_gap_at=half_gap_at,
        )
        if route.worst_position_error_m > 0.025:
            raise RuntimeError(f"the arm misses the path by {route.worst_position_error_m:.3f} m")
        return route

    route = plan(grasp, q_soft, 0.0)
    arm.start(route)
    settle = round(SETTLE * tool_scene.CONTROL_HZ)
    q0 = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
    handle_rest = q0[handle]
    grip_frame = round(PHASES["lift"][0] * tool_scene.CONTROL_HZ)
    state = {"ref": None, "base": None}

    def follow(frame: int):
        nonlocal route, grasp
        if frame == settle:
            # Grasp the tool where it came to rest, its handle's axis taken from the nodes.
            now = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
            axis = now[soft_nodes].mean(axis=0) - now[handle].mean(axis=0)
            grasp = tool_scene.grasp_handle(asset, now, src_tets, labels, axis=axis / np.linalg.norm(axis))
            # Level the tool: undo its pitch (against the asset as built) about the closing axis.
            pitch = kabsch(vertices[rigid], now[rigid]).as_rotvec() @ np.asarray(grasp["across"])
            route = plan(grasp, now[soft_nodes], -float(pitch) if args.level else 0.0)
            print(json.dumps({"settled_grasp": grasp, "pitch_deg": float(np.degrees(pitch))}), flush=True)
        arm.follow(route, frame)
        return arm

    def task_row(frame: int, local: np.ndarray, body_q: np.ndarray) -> dict:
        hand_p, hand_q = arm.arm.ee_pose(sim.pipeline.state_0)
        hand = Rotation.from_quat(np.asarray(hand_q, dtype=np.float64))
        tool = kabsch(handle_rest, local[handle])
        offset = tool.apply(-handle_rest.mean(axis=0)) + local[handle].mean(axis=0)
        in_hand = (hand.inv().apply(offset - (np.asarray(hand_p) - surface)), hand.inv() * tool)
        height = float(local[pad, 2].min())
        if frame == grip_frame:
            state["ref"] = in_hand
        if frame == round(PHASES["close"][0] * tool_scene.CONTROL_HZ) - 1:
            state["base"] = height
        row = {"pad_bottom_m": height, "handle_centre_m": local[handle].mean(axis=0).tolist()}
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
    base = state["base"]
    pressing = [
        r
        for r in rows
        if PHASES["press"][1] * tool_scene.CONTROL_HZ <= r["frame"] <= PHASES["release"][0] * tool_scene.CONTROL_HZ
    ]
    slips = [r for r in rows if "slip_mm" in r]
    result = {
        "asset": args.asset,
        "grasp": grasp,
        "pad_rest_height_mm": None if base is None else 1e3 * base,
        "compression_mm": None
        if base is None or not pressing
        else 1e3 * (base - min(r["pad_bottom_m"] for r in pressing)),
        "slip_max_mm": max((r["slip_mm"] for r in slips), default=None),
        "slip_max_deg": max((r["slip_deg"] for r in slips), default=None),
        "slip_final_mm": slips[-1]["slip_mm"] if slips else None,
        "slip_final_deg": slips[-1]["slip_deg"] if slips else None,
        "max_inverted_tetrahedra": max(r["inverted_tetrahedra"] for r in rows),
        "max_soft_inverted": max(r["cable_inverted"] for r in rows),
        "max_rigid_inverted": max(r["inverted_tetrahedra"] - r["cable_inverted"] for r in rows),
        "soft_stretch_p99_max": max(r["cable_stretch_p99"] for r in rows),
        "min_z_m": min(r["minimum_z_m"] for r in rows),
        "arm_path_error_m": route.worst_position_error_m,
        "rigid_nodes": len(rigid),
        "steps": args.steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "config": tool_scene.config(args, asset, settings),
    }
    result["success"] = bool(
        result["compression_mm"] is not None
        and result["compression_mm"] > 1.0
        and (result["slip_max_mm"] or 0.0) < 5.0
        and (result["slip_max_deg"] or 0.0) < 10.0
        and result["max_soft_inverted"] == 0
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("grasp", "config")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=tool_scene.OUTPUT / "tool_asset_check" / "run")
    parser.add_argument("--asset", choices=tuple(tool_scene.TOOLS), default="sponge_wand")
    parser.add_argument("--steps", type=int, default=round(DURATION * tool_scene.CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--no-level", dest="level", action="store_false", help="keep the tool's resting tilt")
    parser.add_argument("--press-mm", type=float, default=7.0, help="how far the tool is lowered past first touch [mm]")
    tool_scene.add_arguments(parser)
    args = parser.parse_args()
    asset = hetero_scene.with_levers(tool_scene.TOOLS[args.asset], args)
    run(args.out, asset, args)


if __name__ == "__main__":
    main()
