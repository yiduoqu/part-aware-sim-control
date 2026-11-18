# SPDX-License-Identifier: Apache-2.0
"""TT4b: flip a soft pancake with the silicone spatula.

The spatula (picked up as in :mod:`tool_spatula_cook`) slides its head under a soft disc lying in the
pan, lifts it, and snaps the wrist so the head tips up and the pancake turns over. Success is the
pancake's normal turned more than 120 degrees from where it started, with the pancake still in the pan.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import arm_gripper
import cable_scene
import numpy as np
import tool_scene
from loop_geometry import smooth
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.transform import Rotation
from tool_asset_check import kabsch
from tool_spatula_cook import (
    BOWL_R,
    CAMERA,
    ENTRY_TILT,
    HOLDER,
    LOW,
    PAN_AT,
    PAN_FLOOR,
    PAN_TOP,
    SETTINGS,
    SPATULA_AT,
    SPAWN_GAP,
    run,
    soft_solid,
    steel_pan,
)

DURATION = 12.5
RADIUS, THICK, EDGE = 0.045, 0.008, 0.0025
DISC_AT = (0.04, 0.06)
"""The pancake sits at the far side of the flat floor, the rising fillet its stop."""
START_X, UNDER_X = -0.045, 0.075
SLIDE_TILT, LIFT_TILT, LIFT_H = np.radians(30.0), np.radians(-4.0), 0.07
"""The slide's held tilt (steep enough for the handle to pass over the near rim), the level tilt and head height after the lift."""
FLICKS = np.radians((-60.0, -45.0, -30.0, -20.0))
"""Wrist flick angles, tried in turn: the first the arm can follow is used."""
PRESS = PAN_FLOOR - 0.030
"""The slide's commanded depth, deep enough to bend the head flat on the floor (the rim allows it once the tip is in)."""
SLIDE, LIFT, FLICK, BACK = (5.8, 8.0), (8.0, 9.6), (9.6, 10.05), (10.05, 11.4)
HEAD_LEN, HEAD_HALF, HEAD_THICK = 0.104, 0.034, 0.014
"""The spatula head's length, half width and thickness [m]."""
RIMS = ((0.100, 0.0176), (0.105, 0.0255), (0.110, 0.035), (0.115, 0.038))
"""The pan's fillet and rim top as rings: radius from the bowl axis and height [m], from its collision mesh."""
MIN_CLEAR = 0.005


def pancake_mesh(rings: int = 10, layers: int = 2):
    """A round tet disc ``RADIUS`` across, ``THICK`` thick at the centre and rounding off to ``EDGE`` at the rim.

    Concentric rings of points are triangulated and extruded into prisms, each split into three tets by its vertex
    order (neighbours agree on the shared diagonals). The mid-plane is flat, so the thin edge sits a little off the floor.
    """
    xy = [(0.0, 0.0)]
    for k in range(1, rings + 1):
        a = np.linspace(0.0, 2 * np.pi, 6 * k, endpoint=False) + 0.5 * k * np.pi / (6 * k)
        xy += list(RADIUS * k / rings * np.c_[np.cos(a), np.sin(a)])
    xy = np.asarray(xy)
    tris = Delaunay(xy).simplices
    t = EDGE + (THICK - EDGE) * np.sqrt(np.clip(1.0 - (np.hypot(*xy.T) / RADIUS) ** 2, 0.0, 1.0))
    n = len(xy)
    vertices = np.vstack([np.c_[xy, 0.5 * (THICK - t) + t * j / layers] for j in range(layers + 1)])
    tets = []
    for j in range(layers):
        for a, b, c in np.sort(tris, axis=1) + j * n:
            tets += [(a, b, c, a + n), (b, c, a + n, b + n), (c, a + n, b + n, c + n)]
    tets = np.asarray(tets)
    e = vertices[tets[:, 1:]] - vertices[tets[:, :1]]
    flip = np.linalg.det(e) < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    return vertices, tets


def rim_low(x: float, tilt: float, clear: float) -> float:
    """The lowest the head may go [m] with its tip at ``x`` and ``tilt`` [rad] for the tool to pass ``clear`` over the near rim."""
    # The tip node may be a corner up to a thickness off the underside's lowest point.
    lowest = x - HEAD_THICK * np.sin(tilt)
    if tilt >= 0.0:
        rim = PAN_AT[0] - np.sqrt(BOWL_R**2 - HEAD_HALF**2)
    else:
        lowest, rim = lowest - HEAD_LEN * np.cos(tilt), PAN_AT[0] - BOWL_R
    return PAN_TOP + clear - (lowest - rim) * np.tan(tilt)


def keyframes(tip0: np.ndarray, low0: float, level: float, variant: int = 0):
    """Lift, carry over the pan, lower steeply, ease to the slide tilt and slide under the disc, lift and level,
    snap the tip up, then settle the spatula flat over the pan; the head keeps the tool over the near rim throughout."""
    lift, over = (2.0, 3.2), (3.4, 5.0)
    flick_tilt = FLICKS[variant]
    start = np.array((START_X, PAN_AT[1]))
    under = np.array((UNDER_X, PAN_AT[1]))

    def target(t: float):
        if t < over[0]:
            s = smooth(t, *lift)
            return tip0, low0 + 0.12 * s, level * s
        if t < SLIDE[0]:
            s, d = smooth(t, *over), smooth(t, 5.0, SLIDE[0])
            xy = (1 - s) * tip0 + s * start
            low = low0 + 0.12 + s * (0.10 - low0 - 0.12) + d * (LOW - 0.10)
            return xy, low, level + s * (ENTRY_TILT - level)
        if t < SLIDE[1]:
            x = START_X + (UNDER_X - START_X) * smooth(t, *SLIDE)
            s = smooth(t, SLIDE[0], SLIDE[0] + 0.8)
            tilt = ENTRY_TILT + (SLIDE_TILT - ENTRY_TILT) * s
            return np.array((x, PAN_AT[1])), max(LOW + (PRESS - LOW) * s, rim_low(x, tilt, 0.010)), tilt
        if t < LIFT[1]:
            s = smooth(t, *LIFT)
            tilt = SLIDE_TILT + s * (LIFT_TILT - SLIDE_TILT)
            # Level the head under the friction angle on the floor first, so the pancake rides up on it.
            return under, max(PRESS + (LIFT_H - PRESS) * s**2, rim_low(UNDER_X, tilt, 0.017)), tilt
        if t < FLICK[1]:
            s = smooth(t, *FLICK)
            x, tilt = UNDER_X - 0.03 * s, LIFT_TILT + s * (flick_tilt - LIFT_TILT)
            return np.array((x, PAN_AT[1])), max(LIFT_H + 0.02 * s, rim_low(x, tilt, 0.008)), tilt
        s = smooth(t, *BACK)
        x, tilt = UNDER_X - 0.03 * (1 - s), flick_tilt * (1 - s)
        low = (LIFT_H + 0.02) * (1 - s) + (LOW + 0.012) * s
        return np.array((x, PAN_AT[1])), max(low, rim_low(x, tilt, 0.008)), tilt

    return target


def rim_clearance(asset) -> float:
    """The least gap [m] between the tool's underside and the pan's fillet and rim over every flick variant's plan, every 50 ms.

    The tool is placed rigidly as ``hand_motion`` places it (a pressed head only bends up, so this is conservative).
    """
    vertices, tets, labels = cable_scene.load(asset)
    rest = arm_gripper._rotate(np.asarray(asset.quat), vertices)
    grasp = tool_scene.grasp_handle(asset, vertices, tets, labels)
    at = np.subtract(SPATULA_AT, grasp["point_m"][:2])
    rest = rest + np.array(tool_scene.rest_pose(vertices, asset.quat, at=at))
    head = np.unique(tets[np.isin(labels, cable_scene.soft_labels(asset, labels))])
    tip = head[np.argmax(rest[head, 0])]
    across = np.asarray(grasp["across"])
    footprint = cKDTree(rest[:, :2])
    angle = np.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    rim = np.vstack(
        [np.c_[PAN_AT[0] + r * np.cos(angle), PAN_AT[1] + r * np.sin(angle), np.full(360, z)] for r, z in RIMS]
    )
    y_off = rest[tip, 1] - 0.5 * (rest[head, 1].min() + rest[head, 1].max())
    least = np.inf
    for variant in range(len(FLICKS)):
        target = keyframes(rest[tip, :2], float(rest[head, 2].min()), 0.0, variant)
        for t in np.arange(2.0, DURATION, 0.05):
            xy, low, tilt = target(t)
            turn = Rotation.from_rotvec(across * tilt)
            placed = turn.apply(rest)
            shift = np.array((*(np.add(xy, (0.0, y_off)) - placed[tip, :2]), low - placed[head, 2].min()))
            local = turn.inv().apply(rim - shift)
            near = footprint.query_ball_point(local[:, :2], 0.003)
            # A ring point under or over the tool's outline: its gap to the slab, negative inside it.
            gaps = [
                max(rest[i, 2].min() - p[2], p[2] - rest[i, 2].max()) for i, p in zip(near, local, strict=True) if i
            ]
            least = min([least, *gaps])
    return float(least)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=tool_scene.OUTPUT / "tool_spatula_flip" / "run")
    parser.add_argument("--steps", type=int, default=round(DURATION * tool_scene.CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--pancake-young", type=float, default=1.5e4, help="Young's modulus of the pancake [Pa]")
    parser.add_argument(
        "--grip-frac", type=float, default=HOLDER.grip_frac, help="grip position along the handle from its butt"
    )
    tool_scene.add_arguments(parser)
    args = parser.parse_args()
    asset = cable_scene.with_levers(dataclasses.replace(HOLDER, grip_frac=args.grip_frac), args)
    clearance = rim_clearance(asset)
    print(json.dumps({"rim_clearance_min_m": clearance}), flush=True)
    if clearance < MIN_CLEAR:
        raise RuntimeError(
            f"planned spatula passes {1e3 * clearance:.1f} mm from the pan rim (< {1e3 * MIN_CLEAR:.0f} mm)"
        )
    vertices, tets = pancake_mesh()
    pancake, extra = [], {"rim_clearance_min_m": clearance}

    def extras(scene, args):
        specs = steel_pan()
        spec = soft_solid(
            "pancake",
            vertices,
            tets,
            at=(*DISC_AT, PAN_FLOOR + SPAWN_GAP),
            young=args.pancake_young,
            settings=SETTINGS,
            mu=args.mu,
            density=700.0,
            color=(0.86, 0.62, 0.30),
        )
        pancake.append(scene.add(spec))
        return specs

    def measure(sim) -> dict:
        surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
        q = sim.pipeline.state_0.particle_q.numpy()[pancake[0].entity.particles] - surface
        rest = vertices - vertices.mean(axis=0)
        normal_z = float(kabsch(rest, q - q.mean(axis=0)).apply((0.0, 0.0, 1.0))[2])
        c = q.mean(axis=0)
        inside = bool(np.hypot(c[0] - PAN_AT[0], c[1] - PAN_AT[1]) < BOWL_R and c[2] < 0.04)
        return {
            "flip_deg": float(np.degrees(np.arccos(np.clip(normal_z, -1.0, 1.0)))),
            "pancake_centre_m": c.tolist(),
            "in_pan": inside,
        }

    def verdict(final, rows):
        # The pancake's flip is its final turn; the best over the run says whether it turned at all.
        best = max(final["flip_deg"], *(r["flip_deg"] for r in rows))
        final["flip_max_deg"] = best
        extra["flip_max_deg"] = best
        return (min(1.0, final["flip_deg"] / 180.0) if final["in_pan"] else 0.0), bool(
            final["flip_deg"] > 120.0 and final["in_pan"]
        )

    task = {
        "name": "tool_spatula_flip",
        "extras": extras,
        "keyframes": keyframes,
        "variants": len(FLICKS),
        "duration": DURATION,
        "camera": CAMERA,
        "measure": measure,
        "verdict": verdict,
    }
    result = {**run(args.out, asset, args, task), **extra}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
