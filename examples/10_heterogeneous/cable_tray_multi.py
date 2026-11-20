# SPDX-License-Identifier: Apache-2.0
"""Put three cabled objects into one open box, one after another, so they pile on each other.

The generated USB hub, the charger and a second charger turned end for end lie apart on the
bench. Each is a rigid housing and a rigid connector+plug joined by an FEM cable
(:mod:`rigid_part_body`); all three share one VBD solve, so self-contact acts between them too.
The AIRBOT takes each in turn with :mod:`usb_hub_box`'s route -- pinch the housing near its cable
end, lift, carry over the box measured from what hangs, lower, open, withdraw -- then moves over
the next one and comes down open. Each lowering stops over what already lies in the box, so later
objects land on the earlier ones: cable on cable, cable on housing, housing on housing.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import arm_gripper
import cable_scene
import hetero_scene
import numpy as np
import rigid_part_body
import usb_hub_box as hub
from clawhauser_scene import REFERENCE, Trajectory, arrangement, build_sim
from loop_geometry import smooth
from scipy.spatial import cKDTree
from usb_hub_box import BOX_FLOOR, BOX_RIM, BOX_WALL, CABLE, GLASS, HOUSING, LIFT, PHASES, SUBSTEPS

import robosimstudio as rss

OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "cable_tray_multi"
CELL = arm_gripper.CELL
CONTROL_HZ = REFERENCE.control_hz
OBJECTS = (
    ("hub", "usb_hub_gen", 0.0, (0.0, -0.12)),
    ("charger", "charger", 0.0, (-0.10, -0.12)),
    ("charger_flipped", "charger", np.pi, (-0.22, -0.05)),
)
"""Name, asset, yaw about z [rad] and grasp point [m, bench frame], in picking order. They lie side
by side along y, cables away from the box, nearest the box first, so no carry passes over an object
still waiting. The hub's grasp is usb_hub_box's; the flipped charger's cable points at the arm, which
keeps its grasp 0.41 m from the base (the box is 0.43 m) where its cable along +y would not. A
parallel hand closes the same across a housing turned end for end, so it needs no wrist turn."""
BOX_AT = hub.BOX_AT
BOX_INNER = (0.18, 0.28)
"""usb_hub_box's box, 20 mm wider in each direction for three objects; its -x wall still clears the
hub's open pads by about 15 mm."""
BLOCK = 13.4
"""Seconds per object: usb_hub_box's phases, then the move to the next object [s]."""
RETURN = (10.9, 12.4)
DESCEND = (12.6, 13.4)
"""Local windows [s]: from over the box to ``APPROACH_Z`` over the next grasp, then down onto it, open."""
APPROACH_Z = 0.12
SETTLE = 1.4
"""Seconds after the last withdraw."""
DURATION = BLOCK * (len(OBJECTS) - 1) + PHASES["withdraw"][1] + SETTLE
TOUCH = 0.004
"""Two objects' particles this close count as a contact [m]: twice the 2 mm self-contact radius."""


def layout(vertices, tets, labels, quat, grasp_at) -> dict:
    """usb_hub_box's layout, moved so the pads close at ``grasp_at``."""
    geometry = hub.layout(vertices, tets, labels, quat)
    shift = np.subtract(grasp_at, hub.GRASP_AT)
    x, y, z = geometry.pop("pose")
    geometry["pose"] = (float(x + shift[0]), float(y + shift[1]), z)
    geometry["grip_centre_m"][:2] = [float(c) for c in grasp_at]
    geometry["cable_root_m"] = [float(c) for c in np.add(geometry["cable_root_m"], shift)]
    geometry.pop("box_centre_m")
    turned = arm_gripper._rotate(np.asarray(quat), vertices)[:, :2] + geometry["pose"][:2]
    geometry["extent_m"] = [turned.min(axis=0).tolist(), turned.max(axis=0).tolist()]
    geometry["grasp_from_base_m"] = float(np.linalg.norm(np.subtract(grasp_at, arm_gripper.MOUNT[:2])))
    return geometry


def check_layout(objects: dict) -> None:
    """The objects lie apart and none under the box."""
    outer = np.asarray(BOX_INNER) / 2.0 + BOX_WALL
    boxes = {name: np.asarray(g["extent_m"]) for name, g in objects.items()}
    boxes["box"] = np.array((np.subtract(BOX_AT, outer), np.add(BOX_AT, outer)))
    names = list(boxes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if np.all(boxes[a][0] < boxes[b][1]) and np.all(boxes[b][0] < boxes[a][1]):
                raise ValueError(f"{a} overlaps {b}")


def build_scene(
    *, solver: str, iterations: int, mu: float, grip_mu: float, force_limit: float, self_contact: bool, levers
):
    settings = dataclasses.replace(REFERENCE, substeps=SUBSTEPS)
    physics = arrangement(
        settings,
        claw="arm",
        coarse=False,
        solver_name=solver,
        iterations=iterations,
        self_contact=self_contact,
        feedback_smoothing=0.15,
        feedback_clamp=1.5,
        rest_exclusion=hetero_scene.REST_EXCLUSION,
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="cable_tray_multi")
    bx, by = BOX_AT
    half_x, half_y = BOX_INNER[0] / 2.0, BOX_INNER[1] / 2.0
    wall_z = (BOX_RIM + BOX_FLOOR) / 2.0
    wall_half = (BOX_RIM - BOX_FLOOR) / 2.0
    for name, half, at, finish in (
        ("box_floor", (half_x + BOX_WALL, half_y + BOX_WALL, BOX_FLOOR / 2.0), (bx, by, BOX_FLOOR / 2.0), None),
        (
            "box_wall_xn",
            (BOX_WALL / 2.0, half_y + BOX_WALL, wall_half),
            (bx - half_x - BOX_WALL / 2.0, by, wall_z),
            GLASS,
        ),
        (
            "box_wall_xp",
            (BOX_WALL / 2.0, half_y + BOX_WALL, wall_half),
            (bx + half_x + BOX_WALL / 2.0, by, wall_z),
            GLASS,
        ),
        ("box_wall_yn", (half_x, BOX_WALL / 2.0, wall_half), (bx, by - half_y - BOX_WALL / 2.0, wall_z), GLASS),
        ("box_wall_yp", (half_x, BOX_WALL / 2.0, wall_half), (bx, by + half_y + BOX_WALL / 2.0, wall_z), GLASS),
    ):
        extra = {"finish": finish} if finish is not None else {}
        scene.add(rss.slab(name=name, half=half, at=at, color=(0.55, 0.60, 0.65), **extra))
    objects, handles, assets = {}, {}, {}
    for name, key, yaw, grasp_at in OBJECTS:
        asset = hetero_scene.with_levers(cable_scene.ASSETS[key], levers)
        quat = arm_gripper.quat_multiply(arm_gripper.axis_angle_quat((0.0, 0.0, 1.0), yaw), asset.quat)
        asset = dataclasses.replace(asset, quat=tuple(float(c) for c in quat))
        vertices, tets, labels = hetero_scene.load(asset)
        objects[name] = layout(vertices, tets, labels, asset.quat, grasp_at)
        spec = hetero_scene.spec(
            asset, vertices, tets, labels, name=name, pose=objects[name]["pose"], settings=settings, mu=mu
        )
        handles[name], assets[name] = scene.add(spec), asset
    check_layout(objects)
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    camera = scene.add(
        rss.CameraSpec(
            name="overview", width=960, height=540, eye=(1.25, -0.70, 1.35), look_at=(0.66, -0.05, 0.95), fov_y=50.0
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    geometry = {
        "objects": objects,
        "box_centre_m": list(BOX_AT),
        "box_inner_m": list(BOX_INNER),
        "arm": arm_gripper.check_arm(sim, limit=force_limit),
        "friction": hetero_scene.set_friction(sim, solver, settings, mu=mu, grip_mu=grip_mu),
    }
    for name, handle in handles.items():
        objects[name]["rigid_parts"] = {
            f"part{rp.part}": {"nodes": int(len(rp.nodes)), "mass_kg": rp.part_mass} for rp in handle.entity.rigid_parts
        }
    print(json.dumps({"geometry": geometry}), flush=True)
    return sim, {n: h.entity for n, h in handles.items()}, assets, camera, geometry


def arm_route(control, geometry: dict, plans: list, start: int, capacity: int, dt: float):
    """Every object's close, lift, carry, lower, open and withdraw, then over and down onto the next.

    ``plans`` holds each object's carry (x, y) and lowering [m]; the route covers frames ``start`` on.
    """
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    grips = [np.asarray(geometry["objects"][name]["grip_centre_m"]) for name, *_ in OBJECTS]
    centres = [arm_gripper.tcp_for_pads(surface + g) for g in grips]
    reaches = [float(geometry["objects"][name]["reach_m"]) for name, *_ in OBJECTS]

    def own(k: int, local: float) -> np.ndarray:
        carry, lower = plans[k]
        return centres[k] + np.array(
            (
                carry[0] * smooth(local, *PHASES["carry"]),
                carry[1] * smooth(local, *PHASES["carry"]),
                LIFT * smooth(local, *PHASES["lift"])
                - lower * smooth(local, *PHASES["lower"])
                + hub.WITHDRAW_Z * smooth(local, *PHASES["withdraw"]),
            )
        )

    def split(time: float) -> tuple[int, float]:
        k = min(int(time // BLOCK), len(OBJECTS) - 1)
        return k, time - k * BLOCK

    def tcp_at(time: float) -> np.ndarray:
        k, local = split(time + start * dt)
        at = own(k, local)
        if k + 1 < len(OBJECTS):
            over = centres[k + 1] + np.array((0.0, 0.0, APPROACH_Z))
            at = at + (over - own(k, BLOCK)) * smooth(local, *RETURN)
            at[2] -= APPROACH_Z * smooth(local, *DESCEND)
        return at

    def half_gap_at(time: float) -> float:
        k, local = split(time + start * dt)
        opened = hub.OPENED_HALF_GAP
        face = opened + (reaches[k] - hub.GRIP_SQUEEZE - opened) * smooth(local, *PHASES["close"])
        return face + (opened - face) * smooth(local, *PHASES["open"])

    return control.plan(quat=arm_gripper.GRASP_QUAT, capacity=capacity, dt=dt, tcp_at=tcp_at, half_gap_at=half_gap_at)


def lowering(geometry: dict, name: str, stack: float) -> float:
    """How far the hand comes down: the hanging housing's bottom ``OVER_FLOOR`` over ``stack``."""
    g = geometry["objects"][name]
    return LIFT + g["grip_centre_m"][2] - (BOX_FLOOR + hub.OVER_FLOOR + g["hang_m"] + stack)


def inside_box(points: np.ndarray, top: float = BOX_RIM) -> np.ndarray:
    """Which bench-frame points are inside the box, below ``top`` over the bench [mask]."""
    return (
        (np.abs(points[:, 0] - BOX_AT[0]) <= BOX_INNER[0] / 2.0)
        & (np.abs(points[:, 1] - BOX_AT[1]) <= BOX_INNER[1] / 2.0)
        & (points[:, 2] <= top)
    )


def run(
    out: Path,
    *,
    steps: int,
    render_every: int,
    solver: str,
    iterations: int,
    mu: float,
    grip_mu: float,
    force_limit: float,
    self_contact: bool,
    levers=None,
    config: dict | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, entities, assets, camera, geometry = build_scene(
        solver=solver,
        iterations=iterations,
        mu=mu,
        grip_mu=grip_mu,
        force_limit=force_limit,
        self_contact=self_contact,
        levers=levers,
    )
    names = [name for name, *_ in OBJECTS]
    dt = 1.0 / CONTROL_HZ
    total = max(steps, round(DURATION * CONTROL_HZ))
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    # Planned before anything is measured: straight carries to the box's centre, onto its floor.
    plans = [
        (np.subtract(BOX_AT, geometry["objects"][n]["grip_centre_m"][:2]), lowering(geometry, n, 0.0)) for n in names
    ]
    control = arm_gripper.ArmControl(sim)
    route, origin = arm_route(control, geometry, plans, 0, total, dt), 0
    errors = {"planned": route.worst_position_error_m}
    print(json.dumps({"arm_path_error_m": errors["planned"], "duration_s": DURATION}), flush=True)
    if route.worst_position_error_m > 0.025:
        raise RuntimeError(f"the arm misses the path by {route.worst_position_error_m:.3f} m")
    control.start(route)
    replan_at = {round((k * BLOCK + PHASES["carry"][0]) * CONTROL_HZ) - 1: k for k in range(len(OBJECTS))}

    parts = {}
    for name, entity in entities.items():
        tets, labels = entity.source_tets, entity.tet_part_labels
        soft = np.isin(labels, hetero_scene.soft_labels(assets[name], labels))
        q0 = sim.pipeline.state_0.particle_q.numpy()[entity.particles]
        parts[name] = {
            "cable": np.unique(tets[labels == CABLE]),
            "dangling": np.setdiff1d(np.unique(tets[labels != HOUSING]), np.unique(tets[labels == HOUSING])),
            "soft": soft,
            "rest_inv": np.linalg.inv((q0[tets[:, 1:]] - q0[tets[:, :1]])[soft]),
            "trajectory": Trajectory(sim, entity),
        }
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    contacts = sim.pipeline.contacts
    finger_shapes = arm_gripper.finger_shapes(sim)
    rows, videos, video_error = [], None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet

        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // CONTROL_HZ))
        videos.__enter__()
    started = time.perf_counter()
    try:
        for frame in range(steps):
            control.follow(route, frame - origin)
            sim.run(1).sync()
            state = sim.pipeline.state_0
            q, body_q = state.particle_q.numpy(), state.body_q.numpy()
            local = {n: q[e.particles] - surface for n, e in entities.items()}
            if frame in replan_at:
                # Carry what hangs over the box's centre, and stop over what already lies in it.
                k = replan_at[frame]
                name = names[k]
                hanging = local[name][parts[name]["dangling"], :2].mean(axis=0)
                below = np.concatenate([local[n] for n in names[:k]] or [np.empty((0, 3))])
                below = below[inside_box(below, BOX_RIM + 0.05)]
                stack = max(0.0, float(np.percentile(below[:, 2], 95.0)) - BOX_FLOOR) if len(below) else 0.0
                plans[k] = (np.asarray(BOX_AT) - hanging, lowering(geometry, name, stack))
                route, origin = arm_route(control, geometry, plans, frame + 1, total - frame - 1, dt), frame + 1
                errors[name] = route.worst_position_error_m
                print(
                    json.dumps(
                        {
                            "object": name,
                            "carry_m": np.round(plans[k][0], 4).tolist(),
                            "stack_m": round(stack, 4),
                            "arm_path_error_m": route.worst_position_error_m,
                        }
                    ),
                    flush=True,
                )
            trees = {n: cKDTree(p) for n, p in local.items()}
            count = int(contacts.soft_contact_count.numpy()[0])
            shapes = contacts.soft_contact_shape.numpy()[: min(count, contacts.soft_contact_max)]
            row = {
                "frame": frame + 1,
                "time_s": (frame + 1) * dt,
                "finger_contacts": int(np.count_nonzero(np.isin(shapes, finger_shapes))),
                "aperture_m": control.aperture(),
                "gap_m": {f"{a}-{b}": float(trees[b].query(local[a])[0].min()) for a, b in pairs},
                "touching_pairs": {f"{a}-{b}": int(trees[a].count_neighbors(trees[b], TOUCH)) for a, b in pairs},
            }
            for name, entity in entities.items():
                p, tets = local[name], entity.source_tets
                edges = p[tets[:, 1:]] - p[tets[:, :1]]
                # The cable's strain: each tet's largest principal stretch; its 99th percentile skips slivers.
                stretch = np.linalg.svd(parts[name]["rest_inv"] @ edges[parts[name]["soft"]], compute_uv=False)
                row[name] = {
                    "minimum_z_m": float(p[:, 2].min()),
                    "cable_inside": float(inside_box(p[parts[name]["cable"]]).mean()),
                    "housing_m": (body_q[entity.rigid_parts[0].body, :3] - surface).tolist(),
                    "inverted_tetrahedra": int(np.count_nonzero(np.linalg.det(edges) <= 0)),
                    "cable_inverted": int(np.count_nonzero(np.linalg.det(edges[parts[name]["soft"]]) <= 0)),
                    "cable_stretch_p99": float(np.percentile(stretch.max(axis=1), 99.0)),
                    **rigid_part_body.frame_record(entity, p + surface, body_q),
                }
                parts[name]["trajectory"].add(row["time_s"], p + surface, body_q)
            rows.append(row)
            if videos is not None and frame % render_every == 0:
                videos.append_frames({camera.name: sim.render(camera)})
            if frame % 30 == 0 or frame == steps - 1:
                brief = {n: {k: v for k, v in row[n].items() if not k.startswith("part")} for n in names}
                print(json.dumps({**{k: v for k, v in row.items() if k not in names}, **brief}), flush=True)
    finally:
        if videos is not None:
            try:
                videos.__exit__(None, None, None)
            except Exception as exc:
                video_error = repr(exc)
    elapsed = time.perf_counter() - started
    for name in names:
        parts[name]["trajectory"].save(out / f"trajectory_{name}.npz")
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")

    final, tail = rows[-1], rows[-round(0.5 * CONTROL_HZ) :]
    per_object = {}
    for k, name in enumerate(names):
        lift = [r for r in rows if k * BLOCK + PHASES["lift"][1] <= r["time_s"] <= k * BLOCK + PHASES["carry"][1]]
        housing = np.asarray(final[name]["housing_m"])
        per_object[name] = {
            "cable_inside_final": final[name]["cable_inside"],
            # The housing's centre, which may lie on the others: within a housing height of the rim.
            "housing_inside_final": bool(inside_box(housing[None], BOX_RIM + 0.02)[0]),
            "housing_final_m": housing.tolist(),
            "lifted_clearance_m": min((r[name]["minimum_z_m"] for r in lift), default=None),
            "cable_stretch_p99_max": max(r[name]["cable_stretch_p99"] for r in rows),
            "max_inverted_tetrahedra": max(r[name]["inverted_tetrahedra"] for r in rows),
            "max_cable_inverted": max(r[name]["cable_inverted"] for r in rows),
            "rigid_skin_offset_max_m": max(
                v["skin_stretch_max_m"] for r in rows for key, v in r[name].items() if key.endswith("_body")
            ),
        }
    first_touch = {}
    for a, b in pairs:
        touched = [r["time_s"] for r in rows if r["touching_pairs"][f"{a}-{b}"]]
        first_touch[f"{a}-{b}"] = touched[0] if touched else None
    result = {
        "geometry": geometry,
        "objects": per_object,
        "released": max(r["finger_contacts"] for r in tail) == 0,
        "max_inverted_tetrahedra": max(o["max_inverted_tetrahedra"] for o in per_object.values()),
        "max_cable_inverted": max(o["max_cable_inverted"] for o in per_object.values()),
        "gap_final_m": final["gap_m"],
        "gap_min_m": {p: min(r["gap_m"][p] for r in rows) for p in final["gap_m"]},
        "touching_pairs_final": final["touching_pairs"],
        "first_touch_s": first_touch,
        "arm_path_error_m": errors,
        "steps": steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "mean_step_ms": 1000.0 * elapsed / max(1, len(rows) * SUBSTEPS),
        "elapsed_s": elapsed,
        "config": config,
    }
    result["success"] = bool(
        all(o["housing_inside_final"] for o in per_object.values())
        # Only the soft cable's tets count: a rigid part's own tets have almost no stiffness and ride on its body.
        and result["max_cable_inverted"] == 0
        and result["released"]
    )
    result["score"] = float(np.mean([o["cable_inside_final"] for o in per_object.values()]))
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "config")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--steps", type=int, default=round(DURATION * CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default="chebyshev")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--mu", type=float, default=hetero_scene.OBJECT_MU)
    parser.add_argument("--grip-mu", type=float, default=2.0)
    parser.add_argument("--force-limit", type=float, default=30.0)
    parser.add_argument("--no-self-contact", dest="self_contact", action="store_false")
    hetero_scene.add_lever_arguments(parser)
    args = parser.parse_args()
    settings = dataclasses.replace(REFERENCE, substeps=SUBSTEPS)
    run(
        args.out,
        steps=args.steps,
        render_every=args.render_every,
        solver=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        grip_mu=args.grip_mu,
        force_limit=args.force_limit,
        self_contact=args.self_contact,
        levers=args,
        config={
            **hetero_scene.config(args, cable_scene.ASSETS["charger"], settings),
            "asset_levers": {
                key: dataclasses.asdict(hetero_scene.with_levers(cable_scene.ASSETS[key], args))
                for _, key, *_ in OBJECTS
            },
            "objects": [[n, key, yaw, list(at)] for n, key, yaw, at in OBJECTS],
        },
    )


if __name__ == "__main__":
    main()
