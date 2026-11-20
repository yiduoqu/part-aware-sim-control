# SPDX-License-Identifier: Apache-2.0
"""TT4b: flip a soft pancake with the silicone spatula.

The spatula (picked up as in :mod:`tool_spatula_cook`) slides its head under a soft disc lying in the
pan, then turns the head about the pancake's far edge (held by the pan's fillet) so the pancake tips over. Success is the
pancake's normal turned more than 120 degrees from where it started, with the pancake still in the pan.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
from pathlib import Path

import arm_gripper
import hetero_scene
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
SLIDE_TILT = np.radians(30.0)
"""The slide's held tilt, steep enough for the handle to pass over the near rim."""
PRESS = PAN_FLOOR - 0.030
"""The slide's commanded depth, deep enough to bend the head flat on the floor (the rim allows it once the tip is in)."""
PIVOT = np.array((DISC_AT[0] + RADIUS + 0.015, PAN_FLOOR + 0.004))
"""The head turns about this line (x, z) [m] along y, 15 mm past the pancake's far edge against the fillet, so it
also pushes the standing pancake forward over that edge."""
ARCS = np.radians((34.0, 32.0, 30.0))
"""How far the head turns about the pivot, tried in turn: the first the arm can follow is used (it follows up to
about 35 degrees, a 65 degree tool tilt)."""
PUSH = 0.020
"""After a hold at the arc's top (the pancake settles on the head) the head shoves this far forward [m] (and 5 mm up)
in 0.3 s, flinging the standing pancake over its far edge
(past about 21 mm the head comes within 5 mm of the far fillet)."""
SLIDE, ARC, SHOVE = (5.8, 8.0), (8.0, 9.2), (9.6, 9.9)
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
    """Lift, carry over the pan, lower steeply, ease to the slide tilt and slide under the disc, turn the head with the
    pancake on it about the pancake's far edge until it stands, push it over that edge and hold."""
    lift, over = (2.0, 3.2), (3.4, 5.0)
    start = np.array((START_X, PAN_AT[1]))
    profile = head_profile()
    # The slide's end pose with the head resting on the floor, its tip node and the arc from there.
    rest = turn_xz(profile, SLIDE_TILT)
    tip_arc = np.array((UNDER_X, PAN_FLOOR + 0.003 - rest[:, 1].min()))

    def arc(phi: float):
        """Tip node (x, z), head's lowest z and tilt with the slide's end pose turned ``phi`` about the pivot."""
        tip = PIVOT + turn_xz(tip_arc - PIVOT, phi)
        return tip, tip[1] + turn_xz(profile, SLIDE_TILT + phi)[:, 1].min(), SLIDE_TILT + phi

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
        if t < ARC[1]:
            tip, low, tilt = arc(ARCS[variant] * smooth(t, *ARC))
            # Ease the slide's press off over the arc's start.
            low += (PRESS - (PAN_FLOOR + 0.003)) * (1.0 - smooth(t, ARC[0], ARC[0] + 0.3))
            return np.array((tip[0], PAN_AT[1])), low, tilt
        tip, low, tilt = arc(ARCS[variant])
        s = smooth(t, *SHOVE)
        # Then hold: pulling back lets the pancake fall back, and the arm cannot lift the head at this tilt.
        return np.array((tip[0] + PUSH * s, PAN_AT[1])), low + 0.005 * s, tilt

    return target


def turn_xz(p: np.ndarray, angle: float) -> np.ndarray:
    """Points (x, z) turned ``angle`` [rad] about +y, the tool's tilt axis (positive turns +x down)."""
    c, s = np.cos(angle), np.sin(angle)
    p = np.asarray(p, dtype=np.float64)
    return np.stack((c * p[..., 0] + s * p[..., 1], c * p[..., 1] - s * p[..., 0]), axis=-1)


@functools.cache
def head_profile() -> np.ndarray:
    """The spatula head's nodes in the x-z plane at rest, relative to its tip node (the one ``hand_motion`` aims) [m]."""
    vertices, tets, labels = hetero_scene.load(HOLDER)
    rest = arm_gripper._rotate(np.asarray(HOLDER.quat), vertices)
    head = rest[np.unique(tets[np.isin(labels, hetero_scene.soft_labels(HOLDER, labels))])]
    return (head - head[np.argmax(head[:, 0])])[:, [0, 2]]


def rim_clearance(asset) -> float:
    """The least gap [m] between the tool's underside and the pan's fillet and rim over every arc variant's plan, every 50 ms.

    The tool is placed rigidly as ``hand_motion`` places it (a pressed head only bends up, so this is conservative).
    """
    vertices, tets, labels = hetero_scene.load(asset)
    rest = arm_gripper._rotate(np.asarray(asset.quat), vertices)
    grasp = tool_scene.grasp_handle(asset, vertices, tets, labels)
    at = np.subtract(SPATULA_AT, grasp["point_m"][:2])
    rest = rest + np.array(tool_scene.rest_pose(vertices, asset.quat, at=at))
    head = np.unique(tets[np.isin(labels, hetero_scene.soft_labels(asset, labels))])
    tip = head[np.argmax(rest[head, 0])]
    across = np.asarray(grasp["across"])
    footprint = cKDTree(rest[:, :2])
    angle = np.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    rim = np.vstack(
        [np.c_[PAN_AT[0] + r * np.cos(angle), PAN_AT[1] + r * np.sin(angle), np.full(360, z)] for r, z in RIMS]
    )
    y_off = rest[tip, 1] - 0.5 * (rest[head, 1].min() + rest[head, 1].max())
    least = np.inf
    for variant in range(len(ARCS)):
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
    asset = hetero_scene.with_levers(dataclasses.replace(HOLDER, grip_frac=args.grip_frac), args)
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
        surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
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
        "variants": len(ARCS),
        "duration": DURATION,
        "camera": CAMERA,
        "measure": measure,
        "verdict": verdict,
        "objects": lambda: {"pancake": pancake[0].entity},
    }
    result = {**run(args.out, asset, args, task), **extra}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
