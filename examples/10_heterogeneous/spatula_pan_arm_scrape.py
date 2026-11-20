# SPDX-License-Identifier: Apache-2.0
"""S3 of the spatula in the pan: the AIRBOT picks the spatula off the bench and scrapes the pan with it.

The spatula lies on the bench beside the pan, its handle towards the arm. The hand comes down
over the handle, closes on it and lifts it; carries it over the pan; turns it tip down, to
``--pitch`` and to the roll that levels the blade's leading edge; lowers it until the blade
touches the pan's floor and ``--depth`` further; pushes it ``--stroke`` away, tip first; lifts
it, sets it down again and drags it back; lifts it clear. The wooden handle is a free rigid body
here and the silicone head FEM: the hand holds the handle by pad friction alone, as the
Clawhauser and stamp scenes hold theirs (:mod:`arm_gripper`).

Where the hand goes is planned from the handle, not from the hand: once the spatula hangs in
the pads the handle's pose in the hand is read, and every later hand pose is the one that puts
the handle where the stroke wants it. What the spatula does in the hand after that -- turning in
the pads under the floor's push -- is not corrected, and is read as ``grip_turn_deg``.

The pan's floor is a slab, as in ``spatula_pan_press``; the pan is drawn over it, in its asset's
three colours, and touches nothing, and how far the spatula passes over its rim is read from the
rim's measured height. The spatula is drawn in its two parts' colours.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import arm_gripper
import newton
import numpy as np
import rigid_part_body
import spatula_pan_press as s1
import spatula_s0 as s0
import stamp_press
from clawhauser_scene import Trajectory, arrangement, build_sim
from mesh_quality import MeshQuality

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish
from robosimstudio.scene.handles import DeformableHandle
from robosimstudio.specs import AssetSpec, StaticSpec
from robosimstudio_materials.heterogeneous import load_labelled_tets

ASSET = s0.ASSET
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "spatula_pan_press" / "s3"
CELL = arm_gripper.CELL
HANDLE, HEAD = s0.HANDLE, s0.HEAD
PHYSICS = stamp_press.PHYSICS
CONTROL_HZ = PHYSICS.control_hz
PAN_AT = (0.0, -0.06)
"""The pan's centre on the bench [m, work-surface frame]; its handle points along +x, away from the spatula."""
PICK_AT = (-0.18, -0.20)
"""Where the hand takes the handle [m]: the spatula lies along y beside the pan, tip away from the arm."""
GRIP_X = -0.040
"""The grasp along the mesh's length [m]: 105 mm behind the seam, on the handle."""
GRIP_BAND = 0.006
"""The handle's section is measured over this far each side of the grasp [m]."""
GRIP_RAISE = 0.002
"""The pads' centre this far above the handle's mid-height [m]: keeps their lower edge off the bench."""
GRIP_SQUEEZE = 0.0002
"""How far past the handle's half-width each pad is commanded [m] (the stamp scene's)."""
GRIP_LIMIT_N = arm_gripper.GRASP_LIMIT_N
GRIP_MU = arm_gripper.GRIP_MU
HOVER = 0.060
"""The open hand waits this far above its grasp while the spatula settles [m]."""
LIFT = 0.120
HOVER_GAP = 0.030
"""Turned tip down, the blade's lowest point waits this far over the pan's floor [m]."""
BETWEEN = 0.020
"""Between the two strokes the hand lifts this far [m]."""
CLEAR_AFTER = 0.050
DEPTH = 0.006
STROKE = 0.100
NEAR = -0.030
"""The tip's near end of the stroke, along y from the pan's centre [m]."""
SETTLE_S = 0.8
OPEN_WIDTH = 2.0 * arm_gripper.OPEN_HALF_GAP
MAX_IK_ERROR_M = 0.010
SUBSTEPS, ITERATIONS, SOLVER = s1.SUBSTEPS, s1.ITERATIONS, s1.SOLVER
YAW = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
"""The mesh's length, +x, laid along the bench's +y."""
PHASES = (
    "settle",
    "reach",
    "close",
    "lift",
    "hang",
    "carry",
    "tilt",
    "poise",
    "lower",
    "dwell",
    "push",
    "stop",
    "raise",
    "lower_again",
    "dwell_again",
    "drag",
    "stop_again",
    "clear",
)


def quat_of(matrix: np.ndarray) -> np.ndarray:
    """The xyzw quaternion of a rotation matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
    return np.asarray(q, dtype=np.float64)


def matrix_of(quat) -> np.ndarray:
    """The rotation matrix of an xyzw quaternion."""
    x, y, z, w = (float(v) for v in quat)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """The angle of the rotation from ``a`` to ``b`` [deg]."""
    return float(np.degrees(np.arccos(np.clip(0.5 * (np.trace(a.T @ b) - 1.0), -1.0, 1.0))))


PAN_PARTS = (
    ("bowl", Finish(roughness=0.5, metallic=0.4)),
    ("rim", Finish(roughness=0.35, metallic=0.8)),
    ("handle", Finish(roughness=0.6)),
)
"""The pan's drawn materials, in its asset's order, and how each takes light."""
SPATULA_PARTS = {HANDLE: ("handle", Finish(roughness=0.7)), HEAD: ("head", Finish(roughness=0.35))}
"""The spatula's parts as they are drawn: oiled wood, and silicone."""


def pan_parts(at) -> list[StaticSpec]:
    """The pan as it is drawn: its visual mesh, one piece a material, each in its albedo's colour.

    The asset's texture is three flat colours, so each material is written once to the temp
    directory as its own welded mesh and drawn in the colour its faces sample. None collides.
    """
    from PIL import Image

    with np.load(s1.PAN / "appearance" / "visual_to_physics.npz") as binding:
        vertices, faces = binding["visual_rest_vertices_m"], binding["visual_faces"]
        uv, material = binding["visual_uv"], binding["face_material_ids"]
    albedo = np.asarray(Image.open(s1.PAN / "appearance" / "albedo.png").convert("RGB"), dtype=np.float64)
    height, width, _ = albedo.shape
    specs = []
    for index, (name, finish) in enumerate(PAN_PARTS):
        mine = faces[material == index]
        at_uv = uv[mine[:, 0]].mean(axis=0)
        colour = albedo[int(at_uv[1] * (height - 1)), int(at_uv[0] * (width - 1))] / 255.0
        welded, inverse = np.unique(np.round(vertices[mine.ravel()], 7), axis=0, return_inverse=True)
        triangles = inverse.reshape(-1, 3) + 1
        key = hashlib.sha1(welded.tobytes() + triangles.tobytes()).hexdigest()[:12]
        path = Path(tempfile.gettempdir()) / "spatula_pan" / f"pan_{name}_{key}.obj"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in welded]
            lines += [f"f {a} {b} {c}" for a, b, c in triangles]
            scratch = path.with_suffix(f".{os.getpid()}.tmp")
            scratch.write_text("\n".join(lines) + "\n")
            os.replace(scratch, path)
        specs.append(
            StaticSpec(
                name=f"pan_{name}",
                asset=AssetSpec(path=str(path)),
                xform=(float(at[0]), float(at[1]), 0.0, 0.0, 0.0, 0.0, 1.0),
                collide=False,
                color=tuple(float(c) for c in colour),
                finish=finish,
            )
        )
    return specs


def spatula_look(sim, entity, backend: str) -> dict:
    """Renderer arguments that draw the spatula's two parts in their own colours.

    A deformable is drawn in one colour, so the surface is handed over as two: the handle's outer
    faces and the head's, each over the particles it uses, in the asset's part colours. Empty for
    a renderer that does not take deformables this way.
    """
    from robosimstudio.solvers.render import backend_class

    if "deformables" not in getattr(backend_class(backend), "wants", ()):
        return {}
    tets, labels = np.asarray(entity.source_tets), np.asarray(entity.tet_part_labels)
    owner = {}
    for corners in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        for face, label in zip(np.sort(tets[:, corners], axis=1).tolist(), labels.tolist(), strict=True):
            owner[tuple(face)] = label
    surface = newton.TetMesh.compute_surface_triangles(tets.ravel()).reshape(-1, 3)
    part = np.array([owner[tuple(face)] for face in np.sort(surface, axis=1).tolist()])
    particles = np.asarray(entity.particles)
    handles, colors, finishes = {}, {}, {}
    for label, (name, finish) in SPATULA_PARTS.items():
        triangles = surface[part == label]
        nodes = np.unique(triangles)
        key = f"spatula_{name}"
        handles[key] = DeformableHandle(
            name=key,
            particles=particles[nodes].astype(np.int32),
            triangles=np.searchsorted(nodes, triangles).astype(np.int32),
        )
        colors[key] = tuple(float(c) / 255.0 for c in entity.part_materials[label]["part_color_rgb"])
        finishes[key] = finish
    return {"deformables": handles, "deformable_colors": colors, "deformable_finishes": finishes}


def build_scene(
    *,
    pitch_deg: float,
    solver_name: str,
    iterations: int,
    substeps: int,
    grip_mu: float,
    force_limit: float,
    mu: float | None,
):
    vertices, tets, labels, _elastic, _density = load_labelled_tets(
        ASSET / "final_mesh.mesh", ASSET / "heterogeneous_params.npz"
    )
    settings = dataclasses.replace(PHYSICS, substeps=substeps)
    physics = arrangement(
        settings, claw="arm", coarse=False, solver_name=solver_name, iterations=iterations, self_contact=False
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="spatula_pan_arm_scrape")
    radius = settings.particle_radius
    head_nodes = np.unique(tets[labels == HEAD])
    handle_nodes = np.unique(tets[labels == HANDLE])
    far = float(vertices[head_nodes, 0].max())
    tips = head_nodes[vertices[head_nodes, 0] >= far - s1.TIP_BAND]
    edge, edge_strips, _slope = s1.leading_edge(vertices, head_nodes)
    pitch = np.radians(pitch_deg)
    roll = s1.level_roll(vertices, edge, edge_strips, pitch)
    press = YAW @ s1.turn_matrix(pitch, roll)
    # The handle's section where the pads take it.
    section = handle_nodes[np.abs(vertices[handle_nodes, 0] - GRIP_X) <= GRIP_BAND]
    grip = np.array(
        (
            GRIP_X,
            0.5 * (vertices[section, 1].min() + vertices[section, 1].max()),
            0.5 * (vertices[section, 2].min() + vertices[section, 2].max()),
        )
    )
    half_width = 0.5 * float(np.ptp(vertices[section, 1]))
    lying = np.array((*(np.asarray(PICK_AT) - (YAW @ grip)[:2]), radius + stamp_press.PLACE_GAP - vertices[:, 2].min()))
    scene.add(
        rss.slab(
            name="pan_floor",
            half=(s1.PAN_FLOOR_RADIUS, s1.PAN_FLOOR_RADIUS, s1.PAN_FLOOR_Z / 2.0),
            at=(PAN_AT[0], PAN_AT[1], s1.PAN_FLOOR_Z / 2.0),
            visible=False,
        )
    )
    for piece in pan_parts(PAN_AT):
        scene.add(piece)
    obj = scene.add(
        stamp_press.StampSpec(
            rigid_parts=(HANDLE,),
            attach_damping=stamp_press.ATTACH_DAMPING,
            name="silicone_spatula",
            mesh_path=str(ASSET / "final_mesh.mesh"),
            params_path=str(ASSET / "heterogeneous_params.npz"),
            materials_path=str(ASSET / "inferred_params.json"),
            pose=tuple(float(v) for v in lying),
            quat=(0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
            contact_ke=settings.contact_ke,
            contact_kd=settings.contact_kd,
            contact_mu=settings.soft_mu,
            particle_radius=radius,
            damping_ratio=settings.damping_ratio,
        )
    )
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    pan = surface + np.array((PAN_AT[0], PAN_AT[1], 0.0))
    for camera in (
        # From beside the bench, the whole arm in profile: the pick beside the pan and the strokes in it.
        rss.CameraSpec(
            name="overview",
            width=1280,
            height=720,
            eye=tuple(float(v) for v in np.add(pan, (0.72, 0.36, 0.50))),
            look_at=tuple(float(v) for v in np.add(pan, (-0.06, -0.20, 0.20))),
            fov_y=44.0,
        ),
        # Over the pan's far rim, on the blade.
        rss.CameraSpec(
            name="close",
            width=1280,
            height=720,
            eye=tuple(float(v) for v in np.add(pan, (0.30, 0.20, 0.17))),
            look_at=tuple(float(v) for v in np.add(pan, (-0.02, -0.04, 0.04))),
            fov_y=34.0,
        ),
    ):
        scene.add(camera)
    sim = build_sim(scene, rigid_loop=False)
    entity = obj.entity
    report = stamp_press.check_stamp(sim, entity, tets, settings, mu=mu)
    report["arm"] = arm_gripper.check_arm(sim, limit=force_limit)
    soft_mu = report["friction"]["global_soft_contact_mu"]
    report["friction"].update(arm_gripper.set_contact_friction(sim, global_mu=soft_mu, grip_mu=grip_mu))
    built = sim.pipeline.state_0.particle_q.numpy()[np.asarray(entity.particles)].astype(np.float64) - surface
    if np.abs(built - (vertices @ YAW.T + lying)).max() > 1.0e-4:
        raise RuntimeError("the spatula was not built where it was laid")
    floor_shapes = [index for index, label in enumerate(sim.model.shape_label) if "pan_floor" in str(label)]
    if not floor_shapes:
        raise RuntimeError("no pan_floor shape")
    young = entity.elastic[:, 0]
    report.update(
        {
            "pitch_deg": pitch_deg,
            "roll_deg": float(np.degrees(roll)),
            "head_young_pa": float(young[labels == HEAD].max()),
            "weight_n": float((entity.mass.sum() + sum(part.body_mass for part in entity.rigid_parts)) * s0.GRAVITY),
            "lying_pose_m": [float(v) for v in lying],
            "grip_mesh_m": grip.tolist(),
            "handle_half_width_m": half_width,
            "handle_thickness_m": float(np.ptp(vertices[section, 2])),
            "pan_at_m": list(PAN_AT),
            "pick_at_m": list(PICK_AT),
        }
    )
    print(json.dumps({"geometry": report}), flush=True)
    geometry = {
        "vertices": vertices,
        "tets": tets,
        "labels": labels,
        "tips": tips,
        "head_nodes": head_nodes,
        "edge": edge,
        "edge_strips": edge_strips,
        "press": press,
        "grip": grip,
        "half_width": half_width,
        "lying": lying,
        "floor_shapes": np.asarray(floor_shapes),
    }
    return sim, obj, geometry, report, settings


def run(
    out: Path,
    *,
    pitch_deg: float = 45.0,
    depth: float = DEPTH,
    stroke: float = STROKE,
    render_every: int = 0,
    until: str = PHASES[-1],
    solver_name: str = SOLVER,
    iterations: int = ITERATIONS,
    substeps: int = SUBSTEPS,
    grip_mu: float = GRIP_MU,
    force_limit: float = GRIP_LIMIT_N,
    squeeze: float = GRIP_SQUEEZE,
    mu: float | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, geometry, report, settings = build_scene(
        pitch_deg=pitch_deg,
        solver_name=solver_name,
        iterations=iterations,
        substeps=substeps,
        grip_mu=grip_mu,
        force_limit=force_limit,
        mu=mu,
    )
    entity = obj.entity
    control = arm_gripper.ArmControl(sim)
    particles = np.asarray(entity.particles)
    vertices, tets, labels = geometry["vertices"], geometry["tets"], geometry["labels"]
    tips, head_nodes, edge, strips = geometry["tips"], geometry["head_nodes"], geometry["edge"], geometry["edge_strips"]
    press = geometry["press"]
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    radius, ke = settings.particle_radius, settings.contact_ke
    contact_z = s1.PAN_FLOOR_Z + radius
    soft_tets = labels == HEAD
    part = entity.rigid_parts[0]
    quality = MeshQuality(sim, entity)
    trajectory = Trajectory(sim, entity)
    contacts = sim.pipeline.contacts
    arms = sim.handles.arms[arm_gripper.ARM]
    finger_sides = [
        np.array([int(shape) for _, shapes in side for shape in shapes], dtype=np.int32) for side in arms.finger_groups
    ]
    shape_forces = getattr(sim.pipeline.program.solver, "shape_contact_forces", None)
    built_body = sim.pipeline.state_0.body_q.numpy()[part.body].astype(np.float64)
    # The handle's body frame: its centre of mass, and the lying spatula's axes.
    centre_mesh = YAW.T @ (built_body[:3] - surface - geometry["lying"])
    grip_in_body = YAW @ (geometry["grip"] - centre_mesh)
    pan_centre = np.asarray(PAN_AT)
    closed = 2.0 * (geometry["half_width"] - squeeze)
    turned = vertices @ press.T
    tip_turned, low_turned = turned[tips].mean(axis=0), float(turned[head_nodes, 2].min())
    # The handle's turn from lying to pressing.
    relative = press @ YAW.T

    def handle_for(land: float, gap: float) -> np.ndarray:
        """The handle body's place [world] with the rigid blade's tip ``land`` along y and ``gap`` over the floor."""
        origin = np.array((PAN_AT[0] - tip_turned[0], PAN_AT[1] + land - tip_turned[1], contact_z + gap - low_turned))
        return surface + origin + press @ centre_mesh

    rows, phases, executed = [], [], []
    frame, worst_ik = 0, 0.0
    # The hand's pose in the handle's frame once it hangs: (rotation, offset).
    held = None
    gap = OPEN_WIDTH
    target, quat = None, np.asarray(arm_gripper.GRASP_QUAT, dtype=np.float64)

    videos, renderer, video_error = None, None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet
        from robosimstudio.solvers.render import FALLBACK

        backend, _why = sim.platform.renderer_for("")
        renderer = sim.make_renderer(
            backend, cameras=list(sim.cameras), channels=["rgb"], **spatula_look(sim, entity, backend)
        )
        if backend != FALLBACK:
            renderer.render(sim.state)
        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // CONTROL_HZ))
        videos.__enter__()

    def body_pose() -> tuple[np.ndarray, np.ndarray]:
        pose = sim.pipeline.state_0.body_q.numpy()[part.body].astype(np.float64)
        return pose[:3], matrix_of(pose[3:])

    def hand_pose() -> tuple[np.ndarray, np.ndarray]:
        position, orientation = control.arm.ee_pose(sim.pipeline.state_0)
        return np.asarray(position, dtype=np.float64), np.asarray(orientation, dtype=np.float64)

    def hand_for(place: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The hand pose that has the handle's body at ``place``, turned ``rotation`` from lying."""
        return place + rotation @ held[1], quat_of(rotation @ held[0])

    def route(begin, end, q_begin, q_end, gap_begin: float, gap_end: float, seconds: float):
        frames = max(2, math.ceil(seconds * CONTROL_HZ))
        q_begin, q_end = np.asarray(q_begin, dtype=np.float64), np.asarray(q_end, dtype=np.float64)
        if np.dot(q_begin, q_end) < 0:
            q_end = -q_end
        span = (frames - 1) / CONTROL_HZ

        def blend(clock: float) -> float:
            x = min(1.0, clock / span)
            return x * x * (3.0 - 2.0 * x)

        def orientation(clock: float) -> np.ndarray:
            q = q_begin + (q_end - q_begin) * blend(clock)
            return q / np.linalg.norm(q)

        return control.plan(
            quat=orientation,
            capacity=frames,
            dt=1.0 / CONTROL_HZ,
            tcp_at=lambda clock: begin + (end - begin) * blend(clock),
            half_gap_at=lambda clock: 0.5 * (gap_begin + (gap_end - gap_begin) * blend(clock)),
        )

    def step(name: str) -> None:
        nonlocal frame
        sim.run(1).sync()
        frame += 1
        state = sim.pipeline.state_0
        q, body_q = state.particle_q.numpy(), state.body_q.numpy()
        owned = q[particles].astype(np.float64)
        local = owned - surface
        place, rotation = body_q[part.body, :3].astype(np.float64), matrix_of(body_q[part.body, 3:])
        tcp, tool = hand_pose()
        tool = matrix_of(tool)
        shapes, forces = s0.contact_forces(q, body_q, sim.model, contacts, ke)
        on_floor = np.isin(shapes, geometry["floor_shapes"])
        touching = np.zeros(len(local), dtype=bool)
        touching[head_nodes] = local[head_nodes, 2] <= contact_z + s1.TOUCH_TOLERANCE
        inverted = np.linalg.det(owned[tets[:, 1:]] - owned[tets[:, :1]]) <= 0
        over_rim = np.hypot(*(local[:, :2] - pan_centre).T)
        over_rim = (over_rim >= s1.PAN_RIM_RADII[0] - radius) & (over_rim <= s1.PAN_RIM_RADII[1] + radius)
        length = rotation @ np.array((0.0, 1.0, 0.0))
        parts = rigid_part_body.frame_record(entity, owned, body_q)
        row = {
            "frame": frame,
            "time_s": frame / CONTROL_HZ,
            "phase": name,
            "finite": bool(np.isfinite(owned).all() and np.isfinite(body_q).all()),
            "tcp_m": (tcp - surface).tolist(),
            "commanded_tcp_m": None if target is None else (target - surface).tolist(),
            "aperture_m": 2.0 * control.aperture(),
            "handle_m": (place - surface).tolist(),
            "handle_grip_z_m": float((place + rotation @ grip_in_body - surface)[2]),
            # Tip down is positive.
            "handle_pitch_deg": float(np.degrees(np.arcsin(np.clip(-length[2], -1.0, 1.0)))),
            "handle_turn_deg": angle_between(np.eye(3), rotation),
            "tip_m": local[tips].mean(axis=0).tolist(),
            "tip_along_m": float(local[tips, 1].mean() - PAN_AT[1]),
            "lowest_m": float(local[:, 2].min()),
            "head_lowest_m": float(local[head_nodes, 2].min()),
            "floor_force_n": float(forces[on_floor, 2].sum()) if on_floor.any() else 0.0,
            "floor_penetration_m": float(np.linalg.norm(forces[on_floor], axis=1).max() / ke)
            if on_floor.any()
            else 0.0,
            "edge_contact_fraction": float(len(np.unique(strips[touching[edge]])) / len(np.unique(strips))),
            "rim_clearance_m": float(local[over_rim, 2].min() - radius - s1.PAN_RIM_Z) if over_rim.any() else None,
            "inverted_soft_tets": int(np.count_nonzero(inverted & soft_tets)),
            "rigid_node_stretch_m": float(
                max(max(p["seam_stretch_max_m"], p["skin_stretch_max_m"]) for p in parts.values())
            ),
        }
        if held is not None:
            now = rotation.T @ tool
            # How far the handle has turned and slid in the pads since it first hung in them.
            row["grip_turn_deg"] = angle_between(held[0], now)
            row["grip_slip_m"] = float(np.linalg.norm(rotation.T @ (tcp - place) - held[1]))
        if shape_forces is not None:
            load, pinch = arm_gripper.pad_forces(sim, shape_forces(), finger_sides)
            row["pad_force_n"], row["squeeze_n"] = max(load), max(pinch)
        rows.append(row)
        trajectory.add(row["time_s"], owned, body_q)
        if videos is not None and frame % render_every == 0:
            videos.append_frames(renderer.render(sim.state))
        if not row["finite"]:
            raise RuntimeError(f"{name}: nonfinite state at frame {frame}")

    def finish(name: str, count: int) -> None:
        phases.append({"phase": name, "seconds": count / CONTROL_HZ, "ends_at_s": rows[-1]["time_s"]})
        print(json.dumps(rows[-1]), flush=True)

    def grasp_point() -> np.ndarray:
        """Where the pads' centre goes on the handle as it lies now [world]."""
        place, rotation = body_pose()
        return place + rotation @ grip_in_body + np.array((0.0, 0.0, GRIP_RAISE))

    def go(name: str, end, end_quat=None, *, opening: float | None = None, seconds: float | None = None) -> None:
        """One action: from the measured tool pose and gap to the commanded ones."""
        nonlocal target, quat, gap, worst_ik
        end = np.asarray(end, dtype=np.float64)
        end_quat = quat if end_quat is None else np.asarray(end_quat, dtype=np.float64)
        position, orientation = hand_pose()
        if seconds is None:
            seconds = max(0.8, float(np.linalg.norm(end - position)) / 0.06)
        end_gap = gap if opening is None else float(opening)
        begin_gap = gap if opening is None else 2.0 * control.aperture()
        path = route(position, end, orientation, end_quat, begin_gap, end_gap, seconds)
        worst_ik = max(worst_ik, path.worst_position_error_m)
        if path.worst_position_error_m > MAX_IK_ERROR_M:
            raise RuntimeError(f"{name}: the arm misses its path by {path.worst_position_error_m:.4f} m")
        target, quat, gap = end, end_quat, end_gap
        for index in range(len(path.joint_q)):
            control.follow(path, index)
            step(name)
        executed.append(
            {
                "phase": name,
                "to_m": (end - surface).tolist(),
                "quat": end_quat.tolist(),
                "seconds": seconds,
                "opening_m": end_gap,
            }
        )
        finish(name, len(path.joint_q))

    def shift(name: str, delta, *, seconds: float) -> None:
        go(name, target + np.asarray(delta, dtype=np.float64), seconds=seconds)

    def settle() -> None:
        nonlocal target
        target = arm_gripper.tcp_for_pads(grasp_point()) + np.array((0.0, 0.0, HOVER))
        hold = route(target, target, quat, quat, gap, gap, 0.1)
        if hold.worst_position_error_m > MAX_IK_ERROR_M:
            raise RuntimeError(f"the hover pose is unreachable: {hold.worst_position_error_m:.4f} m off")
        control.start(hold)
        count = round(SETTLE_S * CONTROL_HZ)
        for index in range(count):
            control.follow(hold, index)
            step("settle")
        low, high = arm_gripper.pad_bounds(sim)
        report["pads"] = {
            "low_m": (low - surface).tolist(),
            "high_m": (high - surface).tolist(),
            "tcp_m": (hand_pose()[0] - surface).tolist(),
        }
        print(json.dumps({"pads": report["pads"]}), flush=True)
        finish("settle", count)

    def close() -> None:
        go("close", target, opening=closed, seconds=1.0)

    def hang() -> None:
        nonlocal held
        go("hang", target, seconds=0.6)
        place, rotation = body_pose()
        tcp, tool = hand_pose()
        held = (rotation.T @ matrix_of(tool), rotation.T @ (tcp - place))
        report["held"] = {
            "handle_turn_deg": angle_between(np.eye(3), rotation),
            "handle_pitch_deg": rows[-1]["handle_pitch_deg"],
            "lifted_m": rows[-1]["handle_grip_z_m"] - report["lying_grip_z_m"],
        }

    def carry() -> None:
        # Level, at the lift's height, to over where the handle will be once it is turned.
        place, rotation = body_pose()
        over = handle_for(NEAR, HOVER_GAP)
        end, _ = hand_for(np.array((over[0], over[1], place[2])), rotation)
        go("carry", np.array((end[0], end[1], target[2])), seconds=2.2)

    def tilt() -> None:
        end, end_quat = hand_for(handle_for(NEAR, HOVER_GAP), relative)
        go("tilt", end, end_quat, seconds=2.2)

    def lower(name: str) -> None:
        # From where the blade's lowest point is now: to the floor, and ``depth`` further.
        drop = rows[-1]["head_lowest_m"] - contact_z + depth
        report.setdefault("touch_drop_m", []).append(rows[-1]["head_lowest_m"] - contact_z)
        shift(name, (0.0, 0.0, -drop), seconds=1.2)

    def steps():
        yield "settle", settle
        yield "reach", lambda: go("reach", arm_gripper.tcp_for_pads(grasp_point()), seconds=1.4)
        yield "close", close
        yield "lift", lambda: shift("lift", (0.0, 0.0, LIFT), seconds=1.8)
        yield "hang", hang
        yield "carry", carry
        yield "tilt", tilt
        yield "poise", lambda: go("poise", target, seconds=0.6)
        yield "lower", lambda: lower("lower")
        yield "dwell", lambda: go("dwell", target, seconds=0.4)
        yield "push", lambda: shift("push", (0.0, stroke, 0.0), seconds=2.0)
        yield "stop", lambda: go("stop", target, seconds=0.4)
        yield "raise", lambda: shift("raise", (0.0, 0.0, depth + BETWEEN), seconds=0.9)
        yield "lower_again", lambda: lower("lower_again")
        yield "dwell_again", lambda: go("dwell_again", target, seconds=0.4)
        yield "drag", lambda: shift("drag", (0.0, -stroke, 0.0), seconds=2.0)
        yield "stop_again", lambda: go("stop_again", target, seconds=0.4)
        yield "clear", lambda: shift("clear", (0.0, 0.0, CLEAR_AFTER), seconds=1.2)

    report["lying_grip_z_m"] = float((body_pose()[0] + body_pose()[1] @ grip_in_body - surface)[2])
    started = time.perf_counter()
    error = None
    try:
        for name, phase in steps():
            phase()
            if name == until:
                break
    except RuntimeError as exc:
        error = str(exc)
        print(json.dumps({"stopped": error}), flush=True)
    finally:
        if videos is not None:
            try:
                videos.__exit__(None, None, None)
            except Exception as exc:
                video_error = repr(exc)
        if renderer is not None:
            renderer.close()
    elapsed = time.perf_counter() - started

    trajectory.save(out / "trajectory.npz")
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")

    def during(*names: str) -> list[dict]:
        return [row for row in rows if row["phase"] in names]

    def middle(name: str) -> list[dict]:
        stretch = during(name)
        return stretch[len(stretch) // 5 : len(stretch) - len(stretch) // 5]

    def mean(stretch: list[dict], key: str) -> float | None:
        values = [row[key] for row in stretch if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    complete = error is None and bool(phases) and phases[-1]["phase"] == PHASES[-1]
    carried = during("hang", "carry", "tilt", "poise")
    strokes = {name: middle(name) for name in ("push", "drag")}
    result = {
        "complete": complete,
        "finite": all(row["finite"] for row in rows),
        "stopped": error,
        "reached": phases[-1]["phase"] if phases else None,
        "lifted_m": report.get("held", {}).get("lifted_m"),
        "hung_pitch_deg": report.get("held", {}).get("handle_pitch_deg"),
        "poised_pitch_deg": mean(during("poise")[-6:], "handle_pitch_deg"),
        "lowest_while_carried_m": min((row["lowest_m"] for row in carried), default=None),
        "floor_force_n": {
            name: mean(middle(name), "floor_force_n")
            for name in ("dwell", "push", "stop", "dwell_again", "drag", "stop_again")
        },
        "edge_contact_fraction": {name: mean(stretch, "edge_contact_fraction") for name, stretch in strokes.items()},
        "handle_pitch_deg": {name: mean(stretch, "handle_pitch_deg") for name, stretch in strokes.items()},
        "tip_travel_m": {
            name: (stretch[-1]["tip_along_m"] - stretch[0]["tip_along_m"]) if stretch else None
            for name, stretch in ((n, during(n)) for n in ("push", "drag"))
        },
        "max_grip_turn_deg": max((row.get("grip_turn_deg", 0.0) for row in rows), default=0.0),
        "max_grip_slip_m": max((row.get("grip_slip_m", 0.0) for row in rows), default=0.0),
        "grip_turn_deg": {
            name: mean(during(name)[-6:], "grip_turn_deg") for name in ("poise", "dwell", "push", "drag", "clear")
        },
        "peak_pad_squeeze_n": max((row.get("squeeze_n", 0.0) for row in rows), default=0.0),
        "min_rim_clearance_m": min(
            (
                row["rim_clearance_m"]
                for row in during(
                    "lower", "dwell", "push", "stop", "raise", "lower_again", "dwell_again", "drag", "stop_again"
                )
                if row["rim_clearance_m"] is not None
            ),
            default=None,
        ),
        "max_floor_force_n": max(row["floor_force_n"] for row in rows),
        "max_floor_penetration_m": max(row["floor_penetration_m"] for row in rows),
        "max_inverted_soft_tets": max(row["inverted_soft_tets"] for row in rows),
        "max_rigid_node_stretch_m": max(row["rigid_node_stretch_m"] for row in rows),
        "final_aperture_m": rows[-1]["aperture_m"],
        "worst_ik_error_m": worst_ik,
    }
    force = result["floor_force_n"]
    result["checks"] = {
        "ran": complete and result["finite"],
        "lifted": result["lifted_m"] is not None and result["lifted_m"] >= 0.8 * LIFT,
        "still_held": rows[-1]["handle_grip_z_m"] > 0.05 and result["final_aperture_m"] < closed + 0.004,
        "pushed_on_the_floor": bool(strokes["push"]) and min(row["floor_force_n"] for row in strokes["push"]) > 0,
        "dragged_on_the_floor": bool(strokes["drag"]) and min(row["floor_force_n"] for row in strokes["drag"]) > 0,
        "edge_on_floor": all(value is not None and value >= 0.9 for value in result["edge_contact_fraction"].values()),
        "handle_clears_rim": result["min_rim_clearance_m"] is not None
        and result["min_rim_clearance_m"] >= s1.RIM_CLEARANCE,
        "no_dig_in": result["max_floor_penetration_m"] <= s1.DIG_IN,
        "soft_mesh_intact": result["max_inverted_soft_tets"] == 0,
    }
    result["success"] = all(result["checks"].values())
    result["push_over_drag"] = force["push"] / force["drag"] if force["push"] and force["drag"] else None
    result["mesh_quality"] = quality.read()
    result["actions"], result["phases"] = executed, phases
    result["config"] = {
        "pitch_deg": pitch_deg,
        "roll_deg": report["roll_deg"],
        "depth_m": depth,
        "stroke_m": stroke,
        "grip_mu": grip_mu,
        "force_limit_n": force_limit,
        "grip_squeeze_m": squeeze,
        "solver": type(arm_gripper.vbd_solver(sim.pipeline.program.solver)).__name__,
        "solver_name": solver_name,
        "iterations": iterations,
        "substeps": substeps,
        "control_hz": CONTROL_HZ,
        "render_every": render_every,
        "newton_version": newton.__version__,
    }
    result["geometry"] = report
    result["video_error"] = video_error
    result["simulated_s"] = len(rows) / CONTROL_HZ
    result["elapsed_s"] = elapsed
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "S3_RESULT " + json.dumps({k: v for k, v in result.items() if k not in ("geometry", "actions", "phases")}),
        flush=True,
    )
    sim.close()
    return result


def replay(out: Path, recording: Path, *, render_every: int, renderer_name: str, limit: int | None, **scene) -> None:
    """Film a finished run again from its ``trajectory.npz``: the scene is rebuilt and posed, nothing is solved.

    Each recorded frame's particles and bodies are written into the state and rendered, so the
    pictures are of the run that was measured, whatever renderer takes them.
    """
    from robosimstudio.core.render.video import VideoSet
    from robosimstudio.solvers.render import FALLBACK

    out.mkdir(parents=True, exist_ok=True)
    sim, obj, _geometry, _report, _settings = build_scene(**scene)
    particles = np.asarray(obj.entity.particles)
    with np.load(recording) as source:
        particle_q, body_q, joint_q = source["particle_q"], source["body_q"], source["joint_q"]
    if particle_q.shape[1] != len(particles) or body_q.shape[1] != sim.model.body_count:
        raise RuntimeError("the recording is not of this scene")
    states = list({id(state): state for state in (sim.state, sim.pipeline.state_0, sim.pipeline.state_1)}.values())
    backend, why = sim.platform.renderer_for(renderer_name)
    print(json.dumps({"renderer": backend, "why": why}), flush=True)
    renderer = sim.make_renderer(
        backend, cameras=list(sim.cameras), channels=["rgb"], **spatula_look(sim, obj.entity, backend)
    )
    if backend != FALLBACK:
        renderer.render(sim.state)
    frames = range(0, len(particle_q) if limit is None else min(limit, len(particle_q)), render_every)
    started = time.perf_counter()
    with VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // CONTROL_HZ)) as videos:
        for count, index in enumerate(frames):
            for state in states:
                q = state.particle_q.numpy()
                q[particles] = particle_q[index]
                state.particle_q.assign(q)
                state.body_q.assign(body_q[index])
                state.joint_q.assign(joint_q[index])
            videos.append_frames(renderer.render(sim.state))
            if count % 60 == 0:
                print(json.dumps({"frame": index, "elapsed_s": time.perf_counter() - started}), flush=True)
    print(
        "REPLAY_DONE "
        + json.dumps({"frames": len(frames), "renderer": backend, "elapsed_s": time.perf_counter() - started}),
        flush=True,
    )
    if backend != FALLBACK:
        # The videos are written and closed. On Windows the path tracer's own shutdown did not
        # return in ten minutes (ovrtx 0.4.1), so the process ends here without it.
        os._exit(0)
    renderer.close()
    sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=Path, default=None, help="film this finished run's trajectory.npz again")
    parser.add_argument(
        "--renderer", type=str, default="", help="the renderer a replay films with (default: the machine's)"
    )
    parser.add_argument("--replay-frames", type=int, default=None, help="film only this many recorded frames")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pitch", type=float, default=45.0, help="the handle's pitch over the pan, tip down [deg]")
    parser.add_argument("--depth", type=float, default=DEPTH, help="how far past touching the hand lowers [m]")
    parser.add_argument("--stroke", type=float, default=STROKE, help="how far each stroke travels [m]")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--until", choices=PHASES, default=PHASES[-1], help="stop after this phase")
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default=SOLVER)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--substeps", type=int, default=SUBSTEPS)
    parser.add_argument("--mu", type=float, default=None, help="the spatula's soft-contact friction")
    parser.add_argument("--grip-mu", type=float, default=GRIP_MU, help="pad-handle friction")
    parser.add_argument("--force-limit", type=float, default=GRIP_LIMIT_N, help="one finger's force [N]")
    parser.add_argument(
        "--squeeze", type=float, default=GRIP_SQUEEZE, help="pad command past the handle's half-width [m]"
    )
    args = parser.parse_args()
    name = f"pitch{args.pitch:g}_{args.depth * 1e3:g}mm_{args.solver}_{args.substeps}x{args.iterations}"
    if args.replay is not None:
        replay(
            args.out if args.out is not None else args.replay.parent / "replay",
            args.replay,
            render_every=max(1, args.render_every),
            renderer_name=args.renderer,
            limit=args.replay_frames,
            pitch_deg=args.pitch,
            solver_name=args.solver,
            iterations=args.iterations,
            substeps=args.substeps,
            grip_mu=args.grip_mu,
            force_limit=args.force_limit,
            mu=args.mu,
        )
        return
    run(
        args.out if args.out is not None else OUTPUT / name,
        pitch_deg=args.pitch,
        depth=args.depth,
        stroke=args.stroke,
        render_every=args.render_every,
        until=args.until,
        solver_name=args.solver,
        iterations=args.iterations,
        substeps=args.substeps,
        grip_mu=args.grip_mu,
        force_limit=args.force_limit,
        squeeze=args.squeeze,
        mu=args.mu,
    )


if __name__ == "__main__":
    main()
