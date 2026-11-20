# SPDX-License-Identifier: Apache-2.0
"""Dock the soldering iron: the AIRBOT seats the iron's station plug in the station socket while the iron trails.

The iron lies on the bench with its cord laid in a hook (:func:`cable_scene.lay`), its station
plug pointing -x. Grasping the iron by its grip and hoping its plug, 0.3 m away at the end of a 5 mm cord, lands in a 1 mm
socket is not controllable, so the hand takes the rigid *plug* end (the cord and the iron's rigid grip
trail behind it); that is the insertion of ``cable_plug_place.py``, whose route is reused as it is:
measure the plug in the pads, turn and shift the hand onto the socket's axis, trim as it approaches,
push in, let go and withdraw. Only the geometry changes: a round plug, the insertion along -x, and
the socket a station block (``--socket loose`` first, ``snug`` after).

Why heterogeneous: the plug must stay rigid and accurate in the pads and the cavity, while the cord
deforms between the plug and the iron, whose rigid grip stays on the bench.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import arm_gripper
import cable_plug_place as plug_route
import cable_scene
import hetero_scene
import numpy as np
import tool_scene
from clawhauser_scene import REFERENCE, arrangement, build_sim
from parallel_gripper import ParallelGripperSpec
from scipy.spatial.transform import Rotation

import robosimstudio as rss
from robosimstudio.specs import AssetSpec, StaticSpec

CORD, GRIP, BARREL, HOUSING, METAL = 0, 1, 2, 3, 4
IRON, PLUG = (GRIP, BARREL), (HOUSING, METAL)
SOCKETS = {"loose": "socket_station_loose", "snug": "socket_station_snug"}
INSERT = np.array((-1.0, 0.0, 0.0))
SOCKET_QUAT = (0.0, 0.0, 1.0, 0.0)
"""The socket's +X (into the block) along -x: the plug's own heading as built."""
plug_route.INSERT_AXIS = INSERT
"""The reused route reads its insertion axis here."""
PLUG_AT = (-0.15, 0.0)
"""The plug's shoulder once the cord is laid [m, bench frame]."""
"""Left then right bends (``--hook-deg``, ``--hook-radius``) that leave the plug pointing -x; none lays the cord as built."""
TIP_PROFILE = (0.0124, 0.0124)
CONTROL_HZ = REFERENCE.control_hz
SUBSTEPS = plug_route.SUBSTEPS
OUTPUT = tool_scene.OUTPUT / "tool_dock_iron"


def layout(vertices, tets, labels, male: dict, female: dict, plug_radius: float, bends=()) -> dict:
    """Where the iron lies, where the pads close, and where the socket and its seated pose go."""
    t_male = np.asarray(male["T_asset_from_mating_frame"])
    shoulder, axes = t_male[:3, 3], t_male[:3, :3]
    direction = axes[:, 0]
    housing = vertices[np.unique(tets[labels == HOUSING])]
    along = (housing - shoulder) @ direction
    across = np.cross((0.0, 0.0, 1.0), direction)
    band = housing[np.abs(along + plug_route.GRASP_BEHIND) < 0.004]
    reach = float(0.5 * np.ptp(band @ across))
    # The cord leaves the grip's seam; it lies along the chord to the plug so the hook barely shears it.
    seam = np.intersect1d(np.unique(tets[np.isin(labels, IRON)]), np.unique(tets[labels == CORD]))
    root = vertices[seam].mean(axis=0)
    chord = shoulder[:2] - root[:2]
    chord /= np.linalg.norm(chord)
    near = float(((vertices[np.unique(tets[np.isin(labels, PLUG)])] - root) @ np.array((*chord, 0.0))).min())
    point, _ = cable_scene.bend_path([near], bends, float(np.arctan2(chord[1], chord[0])))
    laid = point[0] - near * chord
    shift = np.array(
        (
            PLUG_AT[0] - shoulder[0] - laid[0],
            PLUG_AT[1] - shoulder[1] - laid[1],
            -vertices[:, 2].min() + hetero_scene.PLACE_GAP,
        )
    )
    settled_z = plug_radius + shoulder[2] - housing[:, 2].min()
    overhang = float(along.max())
    seat_x = float(female["seated_shoulder_x_in_socket_m"])
    mouth = np.array(
        (*(np.asarray(PLUG_AT) + INSERT[:2] * (plug_route.PRE_INSERT - seat_x)), plug_route.PEDESTAL)
    ) + np.array((0.0, 0.0, np.asarray(female["T_asset_from_mating_frame"])[2, 3]))
    socket_axes = Rotation.from_quat(SOCKET_QUAT).as_matrix() @ np.asarray(female["T_asset_from_mating_frame"])[:3, :3]
    contact_seat_x = -(plug_radius + overhang)
    clearance = float(female["clearance_per_side_m"])
    return {
        "pose": shift.tolist(),
        "lay": {"root_m": (root + shift).tolist(), "direction": chord.tolist(), "bends": bends},
        "grip_centre_m": [
            *(np.asarray(PLUG_AT) - plug_route.GRASP_BEHIND * INSERT[:2]),
            float(ParallelGripperSpec.pad_half_height + 0.001),
        ],
        "reach_m": reach,
        "shoulder_rest_m": [PLUG_AT[0], PLUG_AT[1], float(shoulder[2] + shift[2])],
        "shoulder_settled_m": [PLUG_AT[0], PLUG_AT[1], float(settled_z)],
        "plug_axes": axes.tolist(),
        "tip_profile_m": TIP_PROFILE,
        "tip_length_m": float(male["tip_length_m"]),
        "socket_xform": [float(mouth[0]), float(mouth[1]), plug_route.PEDESTAL, *SOCKET_QUAT],
        "socket_mouth_m": mouth.tolist(),
        "socket_axes": socket_axes.tolist(),
        "seat_m": (mouth + seat_x * socket_axes[:, 0]).tolist(),
        "seat_x_m": seat_x,
        "contact_seat_x_m": contact_seat_x,
        "cavity_depth_m": float(female["cavity_depth_m"]),
        "clearance_per_side_m": clearance,
        "plug_radius_m": plug_radius,
        "tolerance": plug_route.tolerances(clearance, float(male["tip_length_m"]), contact_seat_x - seat_x),
    }


def build_scene(
    asset,
    *,
    socket: str,
    bends: tuple,
    plug_radius: float,
    solver,
    iterations,
    mu,
    grip_mu,
    force_limit,
    self_contact,
    with_socket: bool = True,
):
    vertices, tets, labels = hetero_scene.load(asset)
    mesh, connectors = hetero_scene.fixture(SOCKETS[socket])
    iron = json.loads((asset.path / "connectors.json").read_text())
    geometry = layout(vertices, tets, labels, iron["male_ends"][0], connectors["female_ports"][0], plug_radius, bends)
    geometry["socket"] = SOCKETS[socket]
    settings = dataclasses.replace(tool_scene.SETTINGS, substeps=SUBSTEPS)
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
    scene = rss.Scene(cell=arm_gripper.CELL, physics=physics, name="tool_dock_iron")
    # The block on a slab of its footprint (a static triangle mesh keeps the concave cavity).
    block = np.load(mesh.with_suffix(".npz"))["vertices_m"]
    depth, width = float(np.ptp(block[:, 0])), float(np.ptp(block[:, 1]))
    mx, my, _ = geometry["socket_mouth_m"]
    if with_socket:
        scene.add(
            rss.slab(
                name="socket_pedestal",
                half=(depth / 2.0, width / 2.0, plug_route.PEDESTAL / 2.0),
                at=(mx + INSERT[0] * depth / 2.0, my, plug_route.PEDESTAL / 2.0),
                color=(0.45, 0.47, 0.50),
            )
        )
        scene.add(
            StaticSpec(
                name="socket",
                asset=AssetSpec(path=str(mesh)),
                xform=tuple(geometry["socket_xform"]),
                color=(0.15, 0.16, 0.18),
            )
        )
    obj = scene.add(
        tool_scene.spec(asset, vertices, tets, labels, name="iron", pose=geometry["pose"], settings=settings, mu=mu)
    )
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    mouth = np.asarray(geometry["socket_mouth_m"])
    camera = scene.add(
        rss.CameraSpec(
            name="overview",
            width=960,
            height=540,
            eye=tuple(surface + mouth + (0.42, -0.42, 0.40)),
            look_at=tuple(surface + mouth + (0.14, 0.04, -0.02)),
            fov_y=35.0,
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    solver_ = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    geometry["arm"] = arm_gripper.check_arm(sim, limit=force_limit)
    geometry["friction"] = hetero_scene.set_friction(sim, solver_, settings, mu=mu, grip_mu=grip_mu)
    geometry["socket_mesh_shapes"] = plug_route.socket_shapes(sim) if with_socket else []
    geometry["plug_thin_particles"] = hetero_scene.set_part_radius(sim, obj.entity, PLUG, plug_radius)
    print(json.dumps({"geometry": geometry}), flush=True)
    return sim, obj, camera, geometry, labels


def run(out: Path, asset, *, steps: int, render_every: int, config: dict, idle_arm: bool = False, **physics) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, camera, geometry, labels = build_scene(asset, **physics)
    entity = obj.entity
    tets = entity.source_tets
    soft = hetero_scene.soft_labels(asset, labels)
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    plug = np.unique(tets[np.isin(labels, PLUG)])
    iron = np.unique(tets[np.isin(labels, IRON)])
    lay = geometry["lay"]
    geometry["laid"] = cable_scene.lay(
        sim,
        entity,
        root=surface + np.asarray(lay["root_m"]),
        direction=lay["direction"],
        bends=lay["bends"],
        keep=iron,
    )
    q0 = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
    plug_rest, iron_rest = q0[plug], q0[iron].mean(axis=0)
    shoulder_rest, axes_rest = np.asarray(geometry["shoulder_rest_m"]), np.asarray(geometry["plug_axes"])

    def plug_frame(local: np.ndarray) -> tuple:
        rotation, translation = plug_route.rigid_fit(plug_rest, local[plug])
        return rotation @ shoulder_rest + translation, rotation @ axes_rest

    capacity = max(steps, round(plug_route.DURATION * CONTROL_HZ))
    arm, route, tcp_at = plug_route.arm_route(sim, geometry, capacity, 1.0 / CONTROL_HZ)
    if route.worst_position_error_m > 0.025:
        raise RuntimeError(f"the arm misses the path by {route.worst_position_error_m:.3f} m")
    geometry["arm_path_error_m"] = route.worst_position_error_m
    if not idle_arm:
        arm.start(route)
    settle, measure = round(plug_route.SETTLE * CONTROL_HZ), round(plug_route.MEASURE_AT * CONTROL_HZ)
    trim_frames = {round(t * CONTROL_HZ): t for t in plug_route.TRIMS_AT}
    turned, trims = None, []

    def follow(frame: int):
        nonlocal arm, route, tcp_at, turned, trims
        if idle_arm:
            return arm
        if frame == settle:
            # Grasp the plug where the laid cord left it: the pads across its housing, behind the shoulder.
            local = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
            origin, axes = plug_frame(local)
            yaw = float(np.arctan2(-axes[0, 0], axes[1, 0]))
            along = np.array((-np.sin(yaw), np.cos(yaw)))
            geometry["grip_centre_m"][:2] = (origin[:2] - plug_route.GRASP_BEHIND * along).tolist()
            geometry["shoulder_settled_m"] = origin.tolist()
            geometry["grasp_quat"] = (
                (Rotation.from_rotvec((0.0, 0.0, yaw)) * Rotation.from_quat(arm_gripper.GRASP_QUAT)).as_quat().tolist()
            )
            arm, route, tcp_at = plug_route.arm_route(sim, geometry, capacity, 1.0 / CONTROL_HZ)
            geometry["settled"] = {
                "plug_shoulder_m": origin.tolist(),
                "plug_yaw_deg": float(np.degrees(yaw)),
                "arm_path_error_m": route.worst_position_error_m,
            }
            print(json.dumps({"settled": geometry["settled"]}), flush=True)
        if frame == measure or (frame in trim_frames and turned is not None):
            local = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
            frame_ = plug_frame(local)
            if frame == measure:
                shift, quat = plug_route.correct(
                    geometry, frame_, tcp_at(frame / CONTROL_HZ) - surface, geometry["grasp_quat"], plug_route.MAX_TURN
                )
                turned = (shift, quat)
            else:
                shift, quat = plug_route.lateral(geometry, frame_), turned[1]
                trims = [*trims, (shift, trim_frames[frame])]
            arm, route, tcp_at = plug_route.arm_route(sim, geometry, capacity, 1.0 / CONTROL_HZ, turned, trims)
            record = {
                "time_s": frame / CONTROL_HZ,
                "shift_m": shift.tolist(),
                "plug_error_before": plug_route.seat_error(geometry, frame_),
                "arm_path_error_m": route.worst_position_error_m,
            }
            if frame == measure:
                geometry["correction"] = record
            else:
                geometry.setdefault("trims", []).append(record)
            print(json.dumps({"correction" if frame == measure else "trim": record}), flush=True)
        arm.follow(route, frame)
        return arm

    def task_row(frame: int, local: np.ndarray, body_q: np.ndarray) -> dict:
        centre = local[iron].mean(axis=0)
        frame_ = plug_frame(local)
        return {
            "plug_shoulder_m": frame_[0].tolist(),
            **plug_route.seat_error(geometry, frame_),
            "iron_bottom_m": float(local[iron, 2].min()),
            "iron_displacement_m": float(np.linalg.norm(centre[:2] - iron_rest[:2])),
        }

    rows, elapsed, video_error = hetero_scene.episode(
        sim,
        entity,
        camera,
        out,
        soft=soft,
        steps=steps,
        render_every=render_every,
        control_hz=CONTROL_HZ,
        follow=follow,
        task_row=task_row,
    )
    final, tail = rows[-1], rows[-round(0.5 * CONTROL_HZ) :]
    settled = rows[: max(1, round(plug_route.SETTLE * CONTROL_HZ))]
    settle = {
        "cord_inverted_max": max(r["cable_inverted"] for r in settled),
        "plug_spring_back_m": float(np.linalg.norm(np.subtract(settled[-1]["plug_shoulder_m"], shoulder_rest))),
        "iron_displacement_m": settled[-1]["iron_displacement_m"],
    }
    print(json.dumps({"settle": settle}), flush=True)
    result = {
        "settle": settle,
        "geometry": geometry,
        "weight_n": float(entity.mass.sum() * 9.81),
        "released": max(r["finger_contacts"] for r in tail) == 0,
        **{k: final[k] for k in final if k.startswith("seat_") or k == "tip_inside"},
        "iron_bottom_final_m": final["iron_bottom_m"],
        "iron_displacement_max_m": max(r["iron_displacement_m"] for r in rows),
        "cord_stretch_p99_max": max(r["cable_stretch_p99"] for r in rows),
        "max_inverted_tetrahedra": max(r["inverted_tetrahedra"] for r in rows),
        "max_cord_inverted": max(r["cable_inverted"] for r in rows),
        "arm_path_error_m": max(route.worst_position_error_m, geometry["arm_path_error_m"]),
        "steps": steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "elapsed_s": elapsed,
        "config": config,
    }
    tol = geometry["tolerance"]
    result["success"] = bool(
        result["released"]
        and abs(final["seat_axial_error_m"]) < tol["axial_m"]
        and final["seat_lateral_error_m"] < tol["lateral_m"]
        and final["seat_angle_error_deg"] < tol["angle_deg"]
        and final["tip_inside"]
        and result["iron_bottom_final_m"] < 0.02
        and result["max_cord_inverted"] == 0
    )
    result["score"] = final["seat_position_error_m"]
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "config")}), flush=True)
    return result


def hook(degrees: float, radius: float) -> tuple:
    """The cord's laid path: straight, or a left then a right bend of ``degrees`` each."""
    arc = np.radians(degrees) * radius
    return ((0.02, 0.0), (arc, radius), (arc, -radius)) if degrees else ()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--steps", type=int, default=round(plug_route.DURATION * CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--hook-deg", type=float, default=0.0, help="each of the two bends laying the cord [deg]")
    parser.add_argument("--hook-radius", type=float, default=0.1, help="their radius [m]")
    parser.add_argument(
        "--end-mass", type=float, default=None, help="each rigid end's mass [kg] (default: the source's)"
    )
    parser.add_argument("--no-socket", dest="with_socket", action="store_false", help="bisect: leave out the socket")
    parser.add_argument("--idle-arm", action="store_true", help="bisect: the arm never moves")
    parser.add_argument("--socket", choices=tuple(SOCKETS), default="loose")
    parser.add_argument(
        "--plug-radius", type=float, default=plug_route.PLUG_RADIUS, help="the plug's contact radius [m]"
    )
    tool_scene.add_arguments(parser)
    args = parser.parse_args()
    asset = hetero_scene.with_levers(tool_scene.TOOLS["corded_soldering_iron"], args)
    asset = dataclasses.replace(asset, plug_end_mass=args.end_mass)
    run(
        args.out,
        asset,
        steps=args.steps,
        render_every=args.render_every,
        config=tool_scene.config(args, asset, dataclasses.replace(tool_scene.SETTINGS, substeps=SUBSTEPS)),
        solver=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        grip_mu=args.grip_mu,
        force_limit=args.force_limit,
        self_contact=args.self_contact,
        socket=args.socket,
        bends=hook(args.hook_deg, args.hook_radius),
        with_socket=args.with_socket,
        idle_arm=args.idle_arm,
        plug_radius=args.plug_radius,
    )


if __name__ == "__main__":
    main()
