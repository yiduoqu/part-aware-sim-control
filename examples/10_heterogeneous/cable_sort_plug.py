# SPDX-License-Identifier: Apache-2.0
"""Pick two cabled objects out of a pile by their rigid plugs and plug them into sockets.

The generated USB hub, the charger and a Y splitter cable lie in a pile with their cables bent and
crossed (:func:`cable_untangle_sort.lay_pile`); the splitter stays as a distractor. The arm takes the
charger's USB-C plug and then the hub's USB-A plug, each by the collar, lifts it to a socket, measures
how it sits in the pads, turns and trims the hand onto the socket's axis and pushes it in, as
:mod:`cable_plug_place` does for one. The socket is farther from each plug than its cable reaches
slack, so the cable trails and the housing is drawn along the bench. The second plug is aimed again
once the first has moved the pile.

Why heterogeneous: the plugs must keep their shape and orientation in the pads while their cables
bend, cross and drag, and the cables touch each other (and the distractor's) through self-contact,
which a stiff homogeneous object could not do.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import arm_gripper
import cable_plug_place as plug
import cable_scene
import cable_untangle_sort as pile
import numpy as np
import rigid_part_body
from clawhauser_scene import REFERENCE, Trajectory, arrangement, build_sim
from loop_geometry import smooth
from parallel_gripper import ParallelGripperSpec
from scipy.spatial.transform import Rotation, Slerp

import robosimstudio as rss
from robosimstudio.specs import AssetSpec, StaticSpec

OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "cable_sort_plug"
CONTROL_HZ = REFERENCE.control_hz
SUBSTEPS = plug.SUBSTEPS
SOCKET_FIXTURES = "socket_usb_{}_{}"
PICKS = {"charger": ("c", (0.27, 0.04)), "hub": ("a", (0.14, 0.04))}
"""The plugs in picking order: the socket's connector letter and where the plug's shoulder is
``PRE_INSERT`` short of the seat [m, bench frame]. The charger's, the bottom cable's, goes to the far
socket so the hub's carry does not cross it; both lie 0.2-0.4 m from where each plug starts."""
OBJECTS = {n: pile.OBJECTS[n] for n in ("splitter", "charger", "hub")}
PHASES = {**plug.PHASES, "open": (10.1, 11.9), "withdraw": (12.7, 13.7)}
"""The plug-in's windows with a slow release. The command closes 16 mm past the plug, so the pads let go only late in
the open window, and at 0.7 s that was fast: q5-sortplug's seated charger popped out within 0.2 s of the pads opening
(605 to 417 contacts, 6 to 14 degrees, then 36 mm down to the bench). Now 1.8 s, about half the rate."""
BLOCKS = (0.5, 15.3)
"""When each pick's phases (:data:`PHASES`) begin [s]: the laid pile settles and the hand hovers
before the first; the second begins once the hand has withdrawn from the first and travelled."""
WITHDRAWN = PHASES["withdraw"][1]
TRAVEL_UNTIL = 0.5
"""The hand travels from one pick's withdrawal to the next's hover, arriving this long into its block [s]."""
TRANSIT_Z = 0.06
"""How much higher the hand rides between picks [m]."""
TAIL = 1.1
"""Seconds after the last withdrawal."""
REACH_OK = pile.REACH_OK
PUSH_FROM = 7.7
"""From this trim on, the hand also pushes on if the plug has slipped short of its seat [s]: q4-sortplug's charger
stalled 5.6 mm short (the pads slipped under the cable's pull) and popped out when the pads opened."""
SOCKET_CLEAR = 0.01
"""The least a socket's footprint may lie from the pile's particles [m]."""


def duration() -> float:
    return BLOCKS[-1] + WITHDRAWN + TAIL


def read_plug(asset, vertices, tets, labels, item, pose, quat, kind: str, plug_at, socket: str, radius: float) -> tuple:
    """One pick's plug, socket and seated pose: where its shoulder and axes lie before it is laid, and its socket."""
    meta = json.loads((asset.path / "connectors.json").read_text())["male_ends"][0]
    mesh, connectors = cable_scene.fixture(SOCKET_FIXTURES.format(kind, socket))
    t_male = np.asarray(meta["T_asset_from_mating_frame"])
    # Flat, plug along +y, as the single plug-in lays it: the collar's reach across and its face ahead of the shoulder.
    base = arm_gripper._rotate(np.asarray(asset.quat), vertices)
    shoulder0 = arm_gripper._rotate(np.asarray(asset.quat), t_male[None, :3, 3])[0]
    collar = base[np.unique(tets[labels == plug.COLLAR])]
    band = collar[np.abs(collar[:, 1] - (shoulder0[1] - plug.GRASP_BEHIND)) < 0.004]
    tip = vertices[np.unique(tets[labels == plug.TIP])]
    geometry = plug.socket_pose(
        connectors["female_ports"][0], meta, float(collar[:, 1].max() - shoulder0[1]), radius, plug_at
    )
    geometry.update(
        {
            "reach_m": float(0.5 * np.ptp(band[:, 0])),
            "tip_profile_m": [float(np.ptp(tip[:, 1])), float(np.ptp(tip[:, 2]))],
            "grip_centre_m": [0.0, 0.0, float(ParallelGripperSpec.pad_half_height + 0.001)],
            "shoulder_settled_m": [0.0, 0.0, 0.0],
            "socket": SOCKET_FIXTURES.format(kind, socket),
        }
    )
    rest = {
        "nodes": np.unique(tets[np.isin(labels, plug.PLUG)]),
        "shoulder": arm_gripper._rotate(np.asarray(quat), t_male[None, :3, 3])[0] + pose,
        "axes": Rotation.from_quat(quat).as_matrix() @ t_male[:3, :3],
    }
    return geometry, rest, mesh


def build_scene(
    *,
    socket: str,
    plug_radius: float,
    solver: str,
    iterations: int,
    mu: float,
    grip_mu: float,
    force_limit: float,
    self_contact: bool,
    levers,
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
        rest_exclusion=cable_scene.REST_EXCLUSION,
    )
    scene = rss.Scene(cell=arm_gripper.CELL, physics=physics, name="cable_sort_plug")
    handles, geoms, rests, footprints, objects = {}, {}, {}, {}, {}
    for name, item in OBJECTS.items():
        asset = cable_scene.with_levers(cable_scene.ASSETS[item.asset], levers)
        vertices, tets, labels = cable_scene.load(asset)
        objects[name] = pile.layout(vertices, tets, labels, asset.quat, item)
        quat = arm_gripper.quat_multiply(arm_gripper.axis_angle_quat((0.0, 0.0, 1.0), item.yaw), asset.quat)
        pose = objects[name]["pose"]
        if name in PICKS:
            kind, plug_at = PICKS[name]
            geoms[name], rests[name], mesh = read_plug(
                asset, vertices, tets, labels, item, np.asarray(pose), quat, kind, plug_at, socket, plug_radius
            )
            # The block on a slab of its footprint, as a static triangle mesh that keeps its cavity.
            block = np.load(mesh.with_suffix(".npz"))["vertices_m"]
            depth, width = float(np.ptp(block[:, 0])), float(np.ptp(block[:, 1]))
            mx, my, _ = geoms[name]["socket_mouth_m"]
            footprints[name] = ((mx, my + depth / 2.0), (width / 2.0, depth / 2.0))
            scene.add(
                rss.slab(
                    name=f"socket_pedestal_{name}",
                    half=(width / 2.0, depth / 2.0, plug.PEDESTAL / 2.0),
                    at=(mx, my + depth / 2.0, plug.PEDESTAL / 2.0),
                    color=(0.45, 0.47, 0.50),
                )
            )
            scene.add(
                StaticSpec(
                    name=f"socket_{name}",
                    asset=AssetSpec(path=str(mesh)),
                    xform=tuple(geoms[name]["socket_xform"]),
                    color=(0.15, 0.16, 0.18),
                )
            )
        asset = dataclasses.replace(asset, quat=tuple(float(c) for c in quat))
        spec = cable_scene.spec(asset, vertices, tets, labels, name=name, pose=pose, settings=settings, mu=mu)
        handles[name] = scene.add(spec)
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    camera = scene.add(
        rss.CameraSpec(
            name="overview",
            width=960,
            height=540,
            eye=tuple(surface + np.array((0.30, -0.40, 0.45))),
            look_at=tuple(surface + np.array((0.0, 0.0, 0.03))),
            fov_y=45.0,
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    entities = {n: h.entity for n, h in handles.items()}
    for name, rest in rests.items():
        rest["plug_rest"] = sim.pipeline.state_0.particle_q.numpy()[entities[name].particles][rest["nodes"]] - surface
    geometry = {
        "objects": objects,
        "plugs": geoms,
        "arm": arm_gripper.check_arm(sim, limit=force_limit),
        "friction": cable_scene.set_friction(sim, solver, settings, mu=mu, grip_mu=grip_mu),
        "socket_mesh_shapes": plug.socket_shapes(sim),
        "plug_thin_particles": {
            n: cable_scene.set_part_radius(sim, entities[n], plug.PLUG, plug_radius) for n in PICKS
        },
        "laid": pile.lay_pile(sim, entities, objects),
    }
    # Sockets clear of the laid pile.
    q = sim.pipeline.state_0.particle_q.numpy() - surface
    for name, (centre, half) in footprints.items():
        out = np.maximum(np.abs(np.concatenate([q[e.particles, :2] for e in entities.values()]) - centre) - half, 0.0)
        geometry["plugs"][name]["pile_to_socket_m"] = float(np.hypot(out[:, 0], out[:, 1]).min())
        if geometry["plugs"][name]["pile_to_socket_m"] < SOCKET_CLEAR:
            raise ValueError(f"the pile lies {geometry['plugs'][name]['pile_to_socket_m']:.3f} m from {name}'s socket")
    print(json.dumps({"geometry": geometry}), flush=True)
    return sim, entities, rests, camera, geometry


def route_of(control, geoms: dict, corrections: dict, trims: dict, capacity: int, dt: float):
    """The whole route: each pick's :func:`cable_plug_place.motion` from its block, blended hand to hand between them."""
    moves = [plug.motion(geoms[n], corrections[n], trims[n], PHASES) for n in PICKS]
    done = [b + WITHDRAWN for b in BLOCKS]
    arrive = [b + TRAVEL_UNTIL for b in BLOCKS[1:]]

    def at(time: float, field: int, mix):
        for i, move in enumerate(moves):
            if time < done[i] or i == len(moves) - 1:
                return move[field](time - BLOCKS[i])
            if time < arrive[i]:
                w = smooth(time, done[i], arrive[i])
                return mix(move[field](time - BLOCKS[i]), moves[i + 1][field](time - BLOCKS[i + 1]), w)

    def lerp(a, b, w):
        return a + (b - a) * w

    def tcp_at(time):
        out = at(time, 0, lerp)
        for i, t0 in enumerate(done[:-1]):
            if t0 <= time < arrive[i]:
                out = out + np.array((0.0, 0.0, TRANSIT_Z * np.sin(np.pi * smooth(time, t0, arrive[i]))))
        return out

    def slerp(a, b, w):
        return Slerp((0.0, 1.0), Rotation.from_quat([a, b]))(min(max(w, 0.0), 1.0)).as_quat()

    return control.plan(
        quat=lambda t: at(t, 1, slerp),
        capacity=capacity,
        dt=dt,
        tcp_at=tcp_at,
        half_gap_at=lambda t: at(t, 2, lerp),
    )


def run(out: Path, *, steps: int, render_every: int, config: dict, **physics) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, entities, rests, camera, geometry = build_scene(**physics)
    geoms, names = geometry["plugs"], list(entities)
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    dt, total = 1.0 / CONTROL_HZ, round(duration() * CONTROL_HZ)
    capacity = max(steps, total)
    corrections, trims = dict.fromkeys(PICKS), {n: [] for n in PICKS}

    def plug_frame(name: str, local: np.ndarray) -> tuple:
        rest = rests[name]
        rotation, translation = plug.rigid_fit(rest["plug_rest"], local[rest["nodes"]])
        return rotation @ rest["shoulder"] + translation, rotation @ rest["axes"]

    def aim(name: str, local: np.ndarray, flip: bool) -> None:
        """Pinch the plug where it lies now; ``flip`` turns the hand half way about (the pads close the same)."""
        origin, axes = plug_frame(name, local)
        yaw = float(np.arctan2(-axes[0, 0], axes[1, 0]))
        g = geoms[name]
        g["grip_centre_m"][:2] = (origin[:2] - plug.GRASP_BEHIND * np.array((-np.sin(yaw), np.cos(yaw)))).tolist()
        g["shoulder_settled_m"] = origin.tolist()
        hand = Rotation.from_rotvec((0.0, 0.0, yaw + (np.pi if flip else 0.0))) * Rotation.from_quat(
            arm_gripper.GRASP_QUAT
        )
        g["grasp_quat"] = hand.as_quat().tolist()
        corrections[name], trims[name] = None, []

    def local_now() -> dict:
        q = sim.pipeline.state_0.particle_q.numpy()
        return {n: q[e.particles] - surface for n, e in entities.items()}

    local = local_now()
    for name in PICKS:
        aim(name, local[name], False)
    control = arm_gripper.ArmControl(sim)
    route = route_of(control, geoms, corrections, trims, capacity, dt)
    errors = {"planned": route.worst_position_error_m}
    print(
        json.dumps({"arm_path_error_m": errors["planned"], "duration_s": duration(), "order": list(PICKS)}), flush=True
    )
    if route.worst_position_error_m > REACH_OK:
        raise RuntimeError(f"the arm misses the path by {route.worst_position_error_m:.3f} m")
    control.start(route)
    # Events by frame: aim a pick at its plug as it lies (the second again once the first has moved the pile),
    # measure it in the pads, then trim it across the axis.
    events = {}

    def at(t: float, *event) -> None:
        events.setdefault(round(t * CONTROL_HZ), []).append(event)

    for k, name in enumerate(PICKS):
        if k == 0:
            at(BLOCKS[0] + 0.5, "aim", name)
        else:
            at(BLOCKS[k - 1] + WITHDRAWN, "aim", name)
            at(BLOCKS[k] + TRAVEL_UNTIL - 0.4, "aim", name)
        at(BLOCKS[k] + plug.MEASURE_AT, "measure", name)
        for t in plug.TRIMS_AT:
            at(BLOCKS[k] + t, "trim", name, t)

    parts, housing_before, trajectories = {}, {}, {}
    for name, entity in entities.items():
        tets, labels, item = entity.source_tets, entity.tet_part_labels, OBJECTS[name]
        soft = np.isin(labels, cable_scene.soft_labels(cable_scene.ASSETS[item.asset], labels))
        q0 = sim.pipeline.state_0.particle_q.numpy()[entity.particles]
        parts[name] = {
            "held": np.unique(tets[np.isin(labels, item.grasp)]),
            "soft": soft,
            "rest_inv": np.linalg.inv((q0[tets[:, 1:]] - q0[tets[:, :1]])[soft]),
        }
        trajectories[name] = Trajectory(sim, entity)
        housing_before[name] = q0[parts[name]["held"]].mean(axis=0) - surface
    housing_rest = dict(housing_before)
    contacts = sim.pipeline.contacts
    finger_shapes = arm_gripper.finger_shapes(sim)
    rows, videos, video_error, aborted, records = [], None, None, None, []
    if render_every:
        from robosimstudio.core.render.video import VideoSet

        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // CONTROL_HZ))
        videos.__enter__()
    started = time.perf_counter()
    try:
        for frame in range(steps):
            for kind, name, *rest_ in events.get(frame, ()):
                local = local_now()
                g = geoms[name]
                k = list(PICKS).index(name)
                if kind == "aim":
                    for flip in (False, True):
                        aim(name, local[name], flip)
                        route = route_of(control, geoms, corrections, trims, capacity, dt)
                        errors[f"{name}_grasp"] = route.worst_position_error_m
                        if route.worst_position_error_m <= REACH_OK:
                            break
                    else:
                        aborted = {"object": name, "reason": f"unreachable by {route.worst_position_error_m:.3f} m"}
                    record = {"aim": name, "time_s": frame * dt, "flip": flip, "shoulder_m": g["shoulder_settled_m"]}
                else:
                    seen = plug_frame(name, local[name])
                    if kind == "measure":
                        hand = plug.motion(g, phases=PHASES)[0](frame * dt - BLOCKS[k]) - surface
                        shift, quat = plug.correct(g, seen, hand, g["grasp_quat"], plug.MAX_TURN)
                        corrections[name] = (shift, quat)
                    else:
                        shift = plug.lateral(g, seen)
                        if rest_[0] >= PUSH_FROM:
                            shift = shift + plug.forward(g, seen)
                        trims[name] = [*trims[name], (shift, rest_[0])]
                    route = route_of(control, geoms, corrections, trims, capacity, dt)
                    record = {
                        kind: name,
                        "time_s": frame * dt,
                        "shift_m": shift.tolist(),
                        "before": plug.seat_error(g, seen),
                    }
                record["arm_path_error_m"] = route.worst_position_error_m
                records.append(record)
                print(json.dumps(record), flush=True)
            if aborted:
                print(json.dumps({"aborted": aborted}), flush=True)
                break
            control.follow(route, frame)
            sim.run(1).sync()
            state = sim.pipeline.state_0
            q, body_q = state.particle_q.numpy(), state.body_q.numpy()
            local = {n: q[e.particles] - surface for n, e in entities.items()}
            count = int(contacts.soft_contact_count.numpy()[0])
            shapes = contacts.soft_contact_shape.numpy()[: min(count, contacts.soft_contact_max)]
            touched = contacts.soft_contact_particle.numpy()[: len(shapes)][np.isin(shapes, finger_shapes)]
            row = {
                "frame": frame + 1,
                "time_s": (frame + 1) * dt,
                "finger_contacts": int(np.count_nonzero(np.isin(shapes, finger_shapes))),
                "aperture_m": control.aperture(),
                # What the fingers touch: each object's plug nodes (the picks') and the rest of it.
                "finger_contacts_on": {
                    n: {
                        "plug": int(np.isin(touched, entities[n].particles[rests[n]["nodes"]]).sum())
                        if n in rests
                        else 0,
                        "all": int(np.isin(touched, entities[n].particles).sum()),
                    }
                    for n in names
                },
            }
            for name, entity in entities.items():
                p, tets = local[name], entity.source_tets
                edges = p[tets[:, 1:]] - p[tets[:, :1]]
                # The cable's strain: each tet's largest principal stretch; its 99th percentile skips slivers.
                stretch = np.linalg.svd(parts[name]["rest_inv"] @ edges[parts[name]["soft"]], compute_uv=False)
                held = p[parts[name]["held"]].mean(axis=0)
                row[name] = {
                    "minimum_z_m": float(p[:, 2].min()),
                    "housing_m": held.tolist(),
                    "housing_displacement_m": float(np.linalg.norm(held[:2] - housing_rest[name][:2])),
                    "inverted_tetrahedra": int(np.count_nonzero(np.linalg.det(edges) <= 0)),
                    "cable_inverted": int(np.count_nonzero(np.linalg.det(edges[parts[name]["soft"]]) <= 0)),
                    "cable_stretch_p99": float(np.percentile(stretch.max(axis=1), 99.0)),
                    **(plug.seat_error(geoms[name], plug_frame(name, p)) if name in PICKS else {}),
                    **rigid_part_body.frame_record(entity, p + surface, body_q),
                }
                trajectories[name].add(row["time_s"], p + surface, body_q)
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
        trajectories[name].save(out / f"trajectory_{name}.npz")
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")

    final, tail = rows[-1], rows[-round(0.5 * CONTROL_HZ) :]
    settled = rows[: min(round(BLOCKS[0] * CONTROL_HZ), len(rows))]
    per_object = {}
    for name in names:
        per_object[name] = {
            "housing_displacement_max_m": max(r[name]["housing_displacement_m"] for r in rows),
            "housing_final_m": final[name]["housing_m"],
            "cable_stretch_p99_max": max(r[name]["cable_stretch_p99"] for r in rows),
            "max_inverted_tetrahedra": max(r[name]["inverted_tetrahedra"] for r in rows),
            "max_cable_inverted": max(r[name]["cable_inverted"] for r in rows),
            "cable_inverted_in_settle": max(r[name]["cable_inverted"] for r in settled),
        }
        if name in PICKS:
            seat = {k: v for k, v in final[name].items() if k.startswith("seat_") or k == "tip_inside"}
            tol = geoms[name]["tolerance"]
            per_object[name].update(
                seat,
                seated=bool(
                    abs(seat["seat_axial_error_m"]) < tol["axial_m"]
                    and seat["seat_lateral_error_m"] < tol["lateral_m"]
                    and seat["seat_angle_error_deg"] < tol["angle_deg"]
                    and seat["tip_inside"]
                ),
            )
    result = {
        "geometry": geometry,
        "order": list(PICKS),
        "objects": per_object,
        "released": max(r["finger_contacts"] for r in tail) == 0,
        "aborted": aborted,
        "records": records,
        "max_inverted_tetrahedra": max(o["max_inverted_tetrahedra"] for o in per_object.values()),
        "max_cable_inverted": max(o["max_cable_inverted"] for o in per_object.values()),
        "arm_path_error_m": errors,
        "steps": steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "elapsed_s": elapsed,
        "config": config,
    }
    result["success"] = bool(
        aborted is None
        and all(per_object[n]["seated"] for n in PICKS)
        and result["max_cable_inverted"] == 0
        and result["released"]
    )
    # Score: the two plugs' mating frames' distances from their seated poses [m].
    result["score"] = float(sum(per_object[n]["seat_position_error_m"] for n in PICKS))
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "config", "records")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default="chebyshev")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--mu", type=float, default=cable_scene.OBJECT_MU)
    parser.add_argument("--grip-mu", type=float, default=2.0)
    parser.add_argument("--force-limit", type=float, default=50.0)
    parser.add_argument("--no-self-contact", dest="self_contact", action="store_false")
    parser.add_argument("--socket", choices=("loose", "snug"), default="loose")
    parser.add_argument("--plug-radius", type=float, default=plug.PLUG_RADIUS, help="the plugs' contact radius [m]")
    cable_scene.add_lever_arguments(parser)
    args = parser.parse_args()
    settings = dataclasses.replace(REFERENCE, substeps=SUBSTEPS)
    run(
        args.out,
        steps=args.steps or round(duration() * CONTROL_HZ),
        render_every=args.render_every,
        config={
            **cable_scene.config(args, cable_scene.ASSETS["usb_hub_gen"], settings),
            "asset_levers": {
                n: dataclasses.asdict(cable_scene.with_levers(cable_scene.ASSETS[o.asset], args))
                for n, o in OBJECTS.items()
            },
            "objects": {n: [*o[:4], list(o.through), *o[5:]] for n, o in OBJECTS.items()},
            "picks": {n: [k, list(at)] for n, (k, at) in PICKS.items()},
        },
        socket=args.socket,
        plug_radius=args.plug_radius,
        solver=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        grip_mu=args.grip_mu,
        force_limit=args.force_limit,
        self_contact=args.self_contact,
        levers=args,
    )


if __name__ == "__main__":
    main()
