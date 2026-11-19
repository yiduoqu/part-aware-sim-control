# SPDX-License-Identifier: Apache-2.0
"""Sponge-wand wipe: the AIRBOT drags the sponge, rolled onto its edge, along the grooves of a ridged plate.

The wand lies on the bench beside a plate of ridges (13 mm wide, 10 mm tall) and 24 mm grooves that run along x,
open at both ends. The hand pinches the handle (as :mod:`tool_asset_check` does), lifts and levels it, rolls it
``--roll-deg`` about its long axis so the pad's edge angles down into a groove, tilts the tip down ``--tilt-deg``
and, for each groove in turn, aligns the edge with it, lowers it ``--press-mm`` past the ridge tops and drags it in
a straight line along the groove and out past the plate's end. Score: the fraction of crumbs out of the grooves
(out of an end, on a ridge or off the plate) before the last raise. ``result.json`` also reports crumb displacement, handle slip in the pads and how low the
sponge reached against the groove floor.

:func:`run` is shared with :mod:`tool_gap_channel`: a :class:`Sweep` says where the tool goes and how the
crumbs are scored.

Why heterogeneous: the soft sponge conforms to the ridges and presses into the grooves, keeping contact along the whole edge,
while the rigid handle transmits the push and the rigid backing spreads the load.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import arm_gripper
import hetero_scene
import numpy as np
import tool_asset_check as check
import tool_scene
from loop_geometry import smooth
from scipy.spatial.transform import Rotation

import robosimstudio as rss

CONTROL_HZ = tool_scene.CONTROL_HZ
Y = -0.20
"""The plate's centreline [m, bench frame]: the grip lands at x = 0, clear of the arm's riser."""
THICK = 0.03
X0, X1 = 0.12, 0.40
"""The scene's extent along x."""
HOVER_Z = 0.07
"""The sponge travels this far over the ridge tops [m]."""
WINDOWS = {
    "lift": (2.0, 3.2),
    "tilt": (3.2, 3.9),
    "travel": (3.4, 4.4),
    "descend": (4.6, 5.4),
    "press": (5.4, 5.9),
    "recenter": (5.2, 5.4),
    "sweep": (6.0, 12.5),
    "raise": (12.8, 13.6),
}
PASS = 10.5
"""A later pass repeats the descend, press, sweep and raise windows this much [s] later."""
DURATION = 14.1
TOUCH = 0.003
"""The tip comes down to this far over the surface, then ramps into the press [m]."""


@dataclasses.dataclass
class Sweep:
    """One sweep: the scene's specs and where the sponge goes, in the bench frame."""

    name: str
    extras: callable
    """``extras(scene, names)``: adds specs (crumb names appended to ``names``) and returns the rest."""
    start_front: float
    """Where the sponge's leading edge starts [m, x]."""
    end_front: float
    contact_z: float
    """The surface the sponge sweeps on [m, z]."""
    hover_z: float
    """The height the sponge's bottom travels at [m, z]."""
    clear: callable
    """``clear(positions)``: bool per crumb, positions in the bench frame."""
    camera: tuple
    soft_radius: float | None = None
    """A thinner contact radius for the tool [m], so it fits a narrow channel."""
    y: float = Y
    ys: tuple = ()
    """The passes' centrelines [m, y], one stroke each; empty is one pass along ``y``."""
    lay_front: float = 0.065
    """Where the lying tool's sponge front starts [m, x]; the arm reaches the grip only so far back."""
    surface: callable = None
    """``surface(x)``: the height the sponge's tip sweeps along [m, bench frame]; ``contact_z`` when None."""
    recenter: bool = False
    """Before each press, measure the hovering tip's y and shift the pass onto its groove."""
    scored: tuple = ()
    """Name prefixes of the crumbs that are scored; all when empty."""


BASE, RIDGE_H, RIDGE_W, GROOVE_W, GROOVES = 0.020, 0.010, 0.013, 0.024, 3
"""The ridged plate's base and ridge height [m] (tops at ``THICK``), widths and groove count."""
SPAN = GROOVES * GROOVE_W + (GROOVES + 1) * RIDGE_W
PLATE_X1 = 0.32


def plate(*, across: str, at, length: float, n: int, tilt: float = 0.0) -> list:
    """The ridged plate; ``tilt`` [rad] slopes it down toward +x about its far end's top, at ``PLATE_X1``."""
    return tool_scene.ridged_plate(
        at=at,
        length=length,
        n=n,
        ridge_w=RIDGE_W,
        groove_w=GROOVE_W,
        ridge_h=RIDGE_H,
        base=BASE,
        across=across,
        tilt=tilt,
        pivot=(PLATE_X1, THICK),
    )


def top_z(x, tilt: float = 0.0):
    """The ridge tops' height at ``x`` [m, bench frame] on the plate tilted ``tilt`` about its far end."""
    return THICK - np.tan(tilt) * (np.asarray(x) - PLATE_X1)


def floor_z(x, tilt: float = 0.0):
    """The groove floor's height at ``x``: the plane ``BASE - THICK`` below the ridge tops, normal to the plate."""
    drop = BASE - THICK
    return THICK + drop * np.cos(tilt) - np.tan(tilt) * (np.asarray(x) - PLATE_X1 - drop * np.sin(tilt))


def in_groove(
    p: np.ndarray, *, x_max: float, half_y: float, tilt: float = 0.0, y: float = Y, x_min: float = X0
) -> np.ndarray:
    """Crumbs (bench frame, (n, 3)) still down in a groove of the plate centred on ``y``."""
    down = p[:, 2] < top_z(p[:, 0], tilt) - 0.002
    return (p[:, 0] > x_min) & (p[:, 0] < x_max) & (np.abs(p[:, 1] - y) < half_y) & down


def plate_specs(*, y: float, tilt: float = 0.0, x0: float = X0) -> list:
    """The ridged plate on the bench, grooves along x and open at both ends; legs hold up a tilted plate's raised end."""
    specs = plate(across="y", at=(0.5 * (x0 + PLATE_X1), y), length=PLATE_X1 - x0, n=GROOVES, tilt=tilt)
    for k, x in enumerate((0.15, 0.24) if tilt else ()):
        under = THICK - THICK * np.cos(tilt) - np.tan(tilt) * (x - PLATE_X1 + THICK * np.sin(tilt))
        specs.append(
            rss.slab(
                name=f"leg{k}",
                half=(0.01, 0.5 * SPAN - 0.01, 0.5 * (under + 0.004)),
                at=(x, y, 0.5 * (under + 0.004)),
                color=tool_scene.RIDGE_COLOR,
            )
        )
    return specs


def wipe(args) -> Sweep:
    y = args.plate_y
    ys = tuple(y + (k - 0.5 * (GROOVES - 1)) * (RIDGE_W + GROOVE_W) for k in range(GROOVES))
    tilt = np.radians(args.plate_tilt_deg)

    def extras(scene, names):
        # Crumbs start in every groove, clear of its ends.
        for k, yg in enumerate(ys):
            names.extend(
                tool_scene.crumbs(
                    scene,
                    region=(X0 + 0.05, PLATE_X1 - 0.05, yg - 0.008, yg + 0.008),
                    n=args.crumbs // GROOVES,
                    spacing=0.012,
                    size=(0.005, 0.008),
                    seed=args.seed + k,
                    z0=(lambda x: floor_z(x, tilt) + 0.002) if tilt else BASE,
                    prefix=f"crumb{k}_",
                    spheres=not tilt,
                )
            )
        return plate_specs(y=y, tilt=tilt, x0=args.plate_start)

    # Crumbs count as cleared once they are out of the grooves: past an end, on a ridge or off the plate.
    sweep = Sweep(
        "wipe",
        extras,
        start_front=X0 + 0.035,
        end_front=PLATE_X1 + 0.025,
        contact_z=THICK,
        hover_z=float(top_z(X0 + 0.035, tilt)) + (0.04 if tilt else HOVER_Z),
        clear=lambda p: ~in_groove(p, x_max=PLATE_X1, half_y=0.5 * SPAN, tilt=tilt, y=y, x_min=args.plate_start),
        camera=((0.52, y - 0.06, 0.12), (0.22, y, 0.04)),
        ys=ys,
        y=y,
        surface=lambda x: np.maximum(THICK, top_z(x, tilt)),
        recenter=True,
        # The lying tool stays clear of the plate's near end.
        lay_front=min(0.065, args.plate_start - 0.015),
        scored=tuple(f"crumb{k}_" for k in range(args.grooves)),
    )
    sweep.ys = ys[: args.grooves]
    return sweep


def run(out: Path, asset, args, sweep: Sweep) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    args.steps = args.steps or round((DURATION + PASS * (len(sweep.ys or (0,)) - 1)) * CONTROL_HZ)
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    vertices, tets, labels = hetero_scene.load(asset)
    grip = tool_scene.grasp_handle(asset, vertices, tets, labels)
    # The tool lies on the bench with its sponge's front ``lay_front`` from the origin, clear of the plate.
    lying = arm_gripper._rotate(np.asarray(asset.quat), vertices)
    pose = tool_scene.rest_pose(
        vertices, asset.quat, at=(sweep.lay_front - lying[:, 0].max(), sweep.y - grip["point_m"][1])
    )
    names: list[str] = []
    sim, obj, camera, vertices, tets, labels, settings = tool_scene.build_scene(
        asset,
        name=sweep.name,
        pose=pose,
        extras=lambda scene: sweep.extras(scene, names),
        camera=sweep.camera,
        **{k: getattr(args, k) for k in ("solver", "iterations", "mu", "grip_mu", "force_limit", "self_contact")},
    )
    plates = [n for n in sim.handles.statics if n.startswith("plate_")]
    if args.plate_mu is not None:
        # Particle-shape friction mixes as sqrt(mu * shape_mu): lower it on the plate only, not the handle or pads.
        solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
        for name in plates:
            hetero_scene.set_shape_mu(sim, solver, name, args.plate_mu, args.mu)
    if sweep.soft_radius:
        hetero_scene.set_part_radius(sim, obj.entity, np.unique(labels), sweep.soft_radius)
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
    ys = sweep.ys or (sweep.y,)
    capacity = max(args.steps, round((DURATION + PASS * (len(ys) - 1)) * CONTROL_HZ))
    bodies = [int(sim.handles.objects[n].body) for n in names if not sweep.scored or n.startswith(sweep.scored)]
    tilt, roll = np.radians(args.tilt_deg), np.radians(args.roll_deg)

    surf = sweep.surface or (lambda x: sweep.contact_z)
    corr, tip_err = [0.0] * len(ys), [None] * len(ys)
    """Each pass's y correction and the tip's y error measured before it [m]."""

    def plan(grasp: dict, soft_now: np.ndarray, level: float, tilt: float = tilt, start: int = 0):
        point = surface + np.array(grasp["point_m"])
        point[2] = max(point[2], surface[2] + check.PAD_HALF + 0.005)
        across = np.asarray(grasp["across"])
        world = soft_now + surface
        # The levelled sponge's long axis: the hand rolls the tool about it to stand the pad on its edge.
        along = np.cross(across, (0.0, 0.0, 1.0))

        def rotated(angle: float) -> np.ndarray:
            return Rotation.from_rotvec(across * angle).apply(world - point)

        # Tilt the leading edge (largest x) down: the sign that lowers the front of the levelled sponge.
        sign = 1.0
        if np.polyfit(*rotated(level + 0.1)[:, [0, 2]].T, 1)[0] > np.polyfit(*rotated(level - 0.1)[:, [0, 2]].T, 1)[0]:
            sign = -1.0

        def s(key: str, t: float, i: int = 0) -> float:
            late = PASS * i if key in ("travel", "descend", "press", "recenter", "sweep", "raise") else 0.0
            return smooth(t, WINDOWS[key][0] + late, WINDOWS[key][1] + late)

        def turn(t: float) -> Rotation:
            lift, tip = level * s("lift", t), s("tilt", t)
            return (
                Rotation.from_rotvec(across * sign * tilt * tip)
                * Rotation.from_rotvec(along * roll * tip)
                * Rotation.from_rotvec(across * lift)
            )

        def rise(t: float) -> float:
            """How far the pivot rises for the sponge's lowest point to stay at its resting height [m]."""
            return float(world[:, 2].min() - (turn(t).apply(world - point)[:, 2] + point[2]).min())

        final = turn(DURATION).apply(world - point)
        front, y_off = float(final[:, 0].max()), float(final[:, 1].mean())
        xs, xe = surface[0] + sweep.start_front - front, surface[0] + sweep.end_front - front
        y0 = surface[1] + ys[0] - y_off

        high = sweep.hover_z
        open_half = min(0.04, grasp["half_width_m"] + 0.014)
        half_close = grasp["half_width_m"] - check.SQUEEZE

        def tcp_at(t: float) -> np.ndarray:
            x = point[0] + (xs - point[0]) * s("travel", t)
            y = point[1] + (y0 - point[1]) * s("travel", t)
            for i in range(len(ys)):
                x += (xe - xs) * s("sweep", t, i)
                if i:
                    x -= (xe - xs) * s("travel", t, i)
                    y += (ys[i] - ys[i - 1]) * s("travel", t, i)
                y += (corr[i] - (corr[i - 1] if i else 0.0)) * s("recenter", t, i)
            top = float(surf(x - surface[0] + front))
            z = high * s("lift", t)
            for i in range(len(ys)):
                z = z * (1.0 - s("descend", t, i)) + (top + TOUCH) * s("descend", t, i)
                z = z * (1.0 - s("press", t, i)) + (top - args.press_mm * 1e-3) * s("press", t, i)
                z = z * (1.0 - s("raise", t, i)) + high * s("raise", t, i)
            hover = check.HOVER * (1.0 - smooth(t, *check.PHASES["descend"]))
            return np.array((x, y, point[2] + hover + z + rise(t)))

        def half_gap_at(t: float) -> float:
            return open_half + (half_close - open_half) * smooth(t, *check.PHASES["close"])

        t0, base = start / CONTROL_HZ, Rotation.from_quat(grasp["quat"])
        route = arm.plan(
            quat=lambda t: (turn(t + t0) * base).as_quat(),
            capacity=capacity - start,
            dt=1.0 / CONTROL_HZ,
            tcp_at=lambda t: tcp_at(t + t0),
            half_gap_at=lambda t: half_gap_at(t + t0),
        )
        if args.plan_only:
            print(
                json.dumps(
                    {
                        "plan_error_m": route.worst_position_error_m,
                        "tilt_deg": float(np.degrees(tilt)),
                        "roll_deg": args.roll_deg,
                        "press_mm": args.press_mm,
                    }
                ),
                flush=True,
            )
        elif route.worst_position_error_m > 0.025:
            raise RuntimeError(f"the arm misses the path by {route.worst_position_error_m:.3f} m")
        return route

    if args.plan_only:
        # The settled plan at each tilt: the tool as it rests, levelled, tilted and carried along the passes.
        now = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
        axis = now[soft_nodes].mean(axis=0) - now[handle].mean(axis=0)
        grasp = tool_scene.grasp_handle(asset, now, src_tets, labels, axis=axis / np.linalg.norm(axis))
        pitch = check.kabsch(vertices[rigid], now[rigid]).as_rotvec() @ np.asarray(grasp["across"])
        for deg in args.scan_tilts:
            plan(grasp, now[soft_nodes], -float(pitch), np.radians(deg))
        return {}
    route = plan(grasp, q_soft, 0.0)
    arm.start(route)
    settle = round(args.settle * CONTROL_HZ)
    start = {"crumbs": None, "ref": None}
    q0 = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
    handle_rest = q0[handle]
    grip_frame = round(check.PHASES["lift"][0] * CONTROL_HZ)

    state = {"off": 0, "args": None}
    measure = {round((WINDOWS["recenter"][0] + PASS * i) * CONTROL_HZ): i for i in range(len(ys))}

    def tip_of(local: np.ndarray) -> np.ndarray:
        """The centroid of the lowest 5% of the sponge's nodes."""
        sponge = local[soft_nodes]
        return sponge[sponge[:, 2] <= np.quantile(sponge[:, 2], 0.05)].mean(axis=0)

    def follow(frame: int):
        nonlocal route, grasp
        if sweep.recenter and frame in measure and state["args"] is not None:
            # Closed loop on y: the hovering tip against its groove, then re-plan from here.
            i = measure[frame]
            tip = tip_of(sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface)
            tip_err[i] = float(tip[1] - ys[i])
            corr[i] = (corr[i - 1] if i else 0.0) - tip_err[i]
            print(json.dumps({"pass": i, "tip_y_err_mm": 1e3 * tip_err[i], "tip_z_mm": 1e3 * tip[2]}), flush=True)
            route, state["off"] = plan(*state["args"], start=frame), frame
        if frame == settle:
            now = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
            axis = now[soft_nodes].mean(axis=0) - now[handle].mean(axis=0)
            grasp = tool_scene.grasp_handle(asset, now, src_tets, labels, axis=axis / np.linalg.norm(axis))
            pitch = check.kabsch(vertices[rigid], now[rigid]).as_rotvec() @ np.asarray(grasp["across"])
            state["args"] = (grasp, now[soft_nodes], -float(pitch) if args.level else 0.0)
            route = plan(*state["args"])
            print(json.dumps({"settled_grasp": grasp, "pitch_deg": float(np.degrees(pitch))}), flush=True)
        arm.follow(route, frame - state["off"])
        return arm

    def crumbs_now(body_q: np.ndarray) -> np.ndarray:
        return body_q[bodies, :3].astype(np.float64) - surface

    def task_row(frame: int, local: np.ndarray, body_q: np.ndarray) -> dict:
        p = crumbs_now(body_q)
        if start["crumbs"] is None:
            start["crumbs"] = p
        hand_p, hand_q = arm.arm.ee_pose(sim.pipeline.state_0)
        hand = Rotation.from_quat(np.asarray(hand_q, dtype=np.float64))
        tool = check.kabsch(handle_rest, local[handle])
        offset = tool.apply(-handle_rest.mean(axis=0)) + local[handle].mean(axis=0)
        in_hand = (hand.inv().apply(offset - (np.asarray(hand_p) - surface)), hand.inv() * tool)
        if frame == grip_frame:
            start["ref"] = in_hand
        row = {
            "cleared": int(sweep.clear(p).sum()),
            "crumb_dx_mean_mm": float(1e3 * (p[:, 0] - start["crumbs"][:, 0]).mean()),
            "pad_bottom_m": float(local[pad, 2].min()),
            "sponge_min_z_m": float(local[soft_nodes, 2].min()),
            "sponge_y_m": float(local[soft_nodes, 1].mean()),
            "tip_y_m": float(tip_of(local)[1]),
        }
        if start["ref"] is not None:
            row["slip_mm"] = float(1e3 * np.linalg.norm(in_hand[0] - start["ref"][0]))
            row["slip_deg"] = float(np.degrees((start["ref"][1].inv() * in_hand[1]).magnitude()))
        return row

    rows, elapsed, video_error = hetero_scene.episode(
        sim,
        entity,
        camera,
        out,
        soft=soft,
        steps=args.steps,
        render_every=args.render_every,
        control_hz=CONTROL_HZ,
        follow=follow,
        task_row=task_row,
    )
    final = crumbs_now(sim.pipeline.state_0.body_q.numpy())
    moved = np.linalg.norm(final[:, :2] - start["crumbs"][:, :2], axis=1)
    cleared = sweep.clear(final)
    # Only crumbs out of the groove before the last raise count: a tool dropped afterwards scores nothing.
    raise_at = (WINDOWS["raise"][0] + PASS * (len(ys) - 1)) * CONTROL_HZ
    before = [r["cleared"] for r in rows if r["frame"] <= raise_at]
    score = before[-1] / len(final) if before and len(final) else 0.0
    sweeping = [
        r
        for i in range(len(ys))
        for r in rows
        if (WINDOWS["sweep"][0] + PASS * i) * CONTROL_HZ <= r["frame"] <= (WINDOWS["sweep"][1] + PASS * i) * CONTROL_HZ
    ]
    slips = [r for r in rows if "slip_mm" in r]
    per_pass = []
    for i in range(len(ys)):
        lo, hi = (WINDOWS["press"][0] + PASS * i) * CONTROL_HZ, (WINDOWS["sweep"][1] + PASS * i) * CONTROL_HZ
        mine = [r for r in rows if lo <= r["frame"] <= hi]
        err = [1e3 * (r["sponge_y_m"] - ys[i]) for r in mine]
        lo, hi = (WINDOWS["sweep"][0] + PASS * i) * CONTROL_HZ, (WINDOWS["sweep"][1] + PASS * i) * CONTROL_HZ
        drag = [1e3 * (r["tip_y_m"] - ys[i]) for r in rows if lo <= r["frame"] <= hi]
        per_pass.append(
            {
                "groove_y_m": ys[i],
                "sponge_min_z_mm": 1e3 * min((r["sponge_min_z_m"] for r in mine), default=float("nan")),
                "y_err_mean_mm": float(np.mean(err)) if err else None,
                "y_err_max_abs_mm": float(np.abs(err).max()) if err else None,
                "tip_y_err_before_mm": None if tip_err[i] is None else 1e3 * tip_err[i],
                "y_correction_mm": 1e3 * corr[i],
                "tip_y_err_drag_mean_mm": float(np.mean(drag)) if drag else None,
                "tip_y_err_drag_max_abs_mm": float(np.abs(drag).max()) if drag else None,
            }
        )
    result = {
        "asset": args.asset,
        "sponge_min_z_pass0_mm": 1e3
        * min(
            (
                r["sponge_min_z_m"]
                for r in rows
                if WINDOWS["press"][0] * CONTROL_HZ <= r["frame"] <= WINDOWS["raise"][0] * CONTROL_HZ
            ),
            default=float("nan"),
        ),
        "per_pass": per_pass,
        "friction": {
            "object_mu": args.mu,
            "pad_grip_mu": args.grip_mu,
            "sponge_vs_plate_mixed_mu": args.plate_mu,
            "plate_shape_mu": None if args.plate_mu is None else args.plate_mu**2 / args.mu,
            "plate_shapes": len(plates),
        },
        "sponge_min_z_sweep_mm": 1e3 * min((r["sponge_min_z_m"] for r in sweeping), default=float("nan")),
        "ridge_top_mm": 1e3 * THICK,
        "groove_floor_mm": 1e3 * BASE,
        "slip_max_mm": max((r["slip_mm"] for r in slips), default=None),
        "slip_max_deg": max((r["slip_deg"] for r in slips), default=None),
        "grasp": grasp,
        "crumbs": len(final),
        "cleared": before[-1] if before else 0,
        "cleared_fraction": score,
        "cleared_final": int(cleared.sum()),
        "crumbs_moved_10mm_fraction": float((moved > 0.01).mean()),
        "crumb_displacement_mean_mm": float(1e3 * moved.mean()),
        "crumb_dx_mean_mm": float(1e3 * (final[:, 0] - start["crumbs"][:, 0]).mean()),
        "crumb_final_x_m": final[:, 0].tolist(),
        "finger_contacts_max": max(r["finger_contacts"] for r in rows),
        "max_soft_inverted": max(r["cable_inverted"] for r in rows),
        "max_rigid_inverted": max(r["inverted_tetrahedra"] - r["cable_inverted"] for r in rows),
        "soft_stretch_p99_max": max(r["cable_stretch_p99"] for r in rows),
        "arm_path_error_m": route.worst_position_error_m,
        "steps": args.steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "config": tool_scene.config(args, asset, settings),
        "score": score,
        "success": bool(score >= 0.7),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("grasp", "config", "crumb_final_x_m")}), flush=True)
    return result


def soft_sponge(asset, args):
    """The asset with kitchen-foam sponge (``--sponge-young``; its scouring face is 6e4)."""
    soft = dataclasses.replace(
        asset,
        cable_young=args.sponge_young,
        grip_frac=args.grip_frac,
        soft_young=tuple((label, 6.0e4) for label, _ in asset.soft_young),
    )
    return hetero_scene.with_levers(soft, args)


def add_arguments(parser) -> None:
    parser.add_argument("--steps", type=int, default=None, help="default: the whole route")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--no-level", dest="level", action="store_false", help="keep the tool's resting tilt")
    parser.add_argument(
        "--press-mm", type=float, default=5.0, help="how far the tool is lowered past the ridge tops [mm]"
    )
    parser.add_argument("--tilt-deg", type=float, default=24.0, help="leading edge down, from a flat sponge [deg]")
    parser.add_argument("--roll-deg", type=float, default=0.0, help="the hand rolls the tool about its long axis [deg]")
    parser.add_argument("--crumbs", type=int, default=12)
    parser.add_argument(
        "--settle", type=float, default=0.0, help="the tool rests this long before the grasp is planned [s]"
    )
    parser.add_argument("--sponge-young", type=float, default=2.0e4, help="the sponge's Young's modulus [Pa]")
    parser.add_argument(
        "--grip-frac", type=float, default=0.15, help="where the pads close along the handle, from its far end"
    )
    parser.add_argument("--scan-tilts", type=float, nargs="*", default=(10, 15, 20, 25, 30, 35))
    parser.add_argument("--plate-tilt-deg", type=float, default=0.0, help="the plate slopes down toward +x [deg]")
    parser.add_argument("--plate-start", type=float, default=X0, help="where the plate's grooves begin [m, x]")
    parser.add_argument("--plate-y", type=float, default=Y, help="the plate's centreline [m, bench frame]")
    parser.add_argument("--plate-mu", type=float, default=0.3, help="sponge-vs-plate mixed friction (None: object mu)")
    parser.add_argument("--plan-only", action="store_true", help="print the arm path error and stop")
    parser.add_argument("--seed", type=int, default=0)
    tool_scene.add_arguments(parser)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=tool_scene.OUTPUT / "tool_sponge_wipe" / "run")
    parser.add_argument("--asset", default="sponge_wand")
    parser.add_argument(
        "--grooves", type=int, default=GROOVES, help="wipe only the first N grooves, score their crumbs"
    )
    add_arguments(parser)
    parser.set_defaults(
        force_limit=120.0,
        grip_mu=2.5,
        roll_deg=30.0,
        tilt_deg=10.0,
        press_mm=9.0,
        plate_y=0.04,
        plate_start=0.06,
        grip_frac=0.5,
    )
    args = parser.parse_args()
    asset = soft_sponge(tool_scene.TOOLS[args.asset], args)
    run(args.out, asset, args, wipe(args))


if __name__ == "__main__":
    main()
