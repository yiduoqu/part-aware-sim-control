# SPDX-License-Identifier: Apache-2.0
"""The AIRBOT flips the hinged soap case open, takes the bar out and sets it down beside the case.

The case (``hinged_soap_case``) is a tray and a lid joined along one long side by a thin web. Tray
and lid (with its tab) are two rigid bodies (:mod:`rigid_part_body`); the web is FEM, and it is the
only thing between them. The bar (``soap_bar``) is a third rigid body, lying in the tray. Nothing
else holds anything: the lid stays shut because its rim rests on the tray's, the bar stays in
because the tray is under it, and both of those are the solver's contact between particles and
faces of the soft meshes. The case lies with its hinge along the bench's y and its thumb tab
towards +x.

The web's Young's modulus is a scene setting (``--hinge-young``), not the bundle's estimate: at the
bundle's 4e7 Pa the web closes the lid again from any angle and would lift the tray first.

**Flip.** The shut hand, leaned ``TILT`` off vertical with its fingers pointing down and towards
the case, slides one finger under the tab from the front and follows the tab round the hinge to
``--push-angle``. Below the finger's own pitch the tab's front edge rides on the finger's outer
face; above it the fingertip's corner is under the tab. Nothing holds the lid: once its weight is
past the hinge it falls away from the finger and lies open behind the tray.

**Pick and place.** The hand stands up, opens, comes down over the bar, closes on the two long
sides that stand out of the tray, lifts the bar out, carries it ``PLACE_AT - CASE_AT`` along x,
lowers it to the bench and lets go. It holds the bar by pad friction alone, and each action runs
from the measured tool pose to the commanded one, as ``rubber_stamp_arm_press`` runs one.

The run measures the lid's angle, whether the tray stayed put, how far the bar rose and where it
came to rest, how far it slipped in the pads, and what the web did.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import arm_gripper
import newton
import numpy as np
import rigid_part_body
import stamp_press
from clawhauser_scene import PhysicsSettings, Trajectory, arrangement, build_sim, set_contact
from mesh_quality import MeshQuality

import robosimstudio as rss
from robosimstudio_materials.heterogeneous import load_labelled_tets

ASSETS = Path(__file__).resolve().parent / "assets"
CASE, SOAP = "hinged_soap_case", "soap_bar"
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "soap_case_arm_pick"
CELL = arm_gripper.CELL
TRAY, LID, HINGE, TAB = range(4)
"""The case's part labels. The tray is one rigid group, the lid and its tab the other."""

PHYSICS = PhysicsSettings(
    control_hz=60,
    substeps=20,
    contact_ke=1.0e4,
    contact_kd=10.0,
    soft_mu=0.5,
    # The rims are 1 mm apart and the web is 1 mm thick.
    particle_radius=0.0005,
    shape_mu=0.5,
    damping_ratio=5.0e-4,
)
"""The stamp scenes' rate, penalty and damping."""
SELF_CONTACT_RADIUS = 0.0004
"""Distance at which one soft surface stops another [m]: under half the 1 mm between the rims, so
the lid is not touching the tray as built. It also caps a node's speed, at about 0.2 m/s here."""
REST_EXCLUSION = 0.0006
"""Pairs nearer than this in the rest shape never touch [m]: under the rims' 1 mm, over the 0.5 mm
between neighbouring nodes of the web."""
PLACE_GAP = 0.0005
"""How far over its rest height the case starts [m]."""
SOAP_GAP = 0.001
"""How far over the tray's floor the bar starts [m]: past the self-contact radius."""
HINGE_YOUNG = 1.5e6
"""The web's Young's modulus in this scene [Pa]. As a beam the meshed web then turns the lid back
with 0.86 mN m per radian, against the lid's 6.7 mN m of weight about the hinge: past about 82
degrees the lid's weight wins. The bundle's 4e7 gives 23 mN m per radian, which nothing here opens."""
CONTROL_HZ = PHYSICS.control_hz
CASE_AT = (-0.10, -0.06)
"""Where the case stands [m, bench frame]: where the arm's stamp scene picks its stamp up."""
PLACE_AT = (0.02, -0.06)
"""Where the bar is set down, 120 mm along x from the case [m]: where that scene's paper is."""
CASE_QUAT = (0.0, 0.0, -0.70710678, 0.70710678)
"""A quarter turn about z: the tab faces +x, and the case's own x, the hinge's axis, runs along -y."""
HINGE_AXIS = (0.0, -1.0, 0.0)
TILT = math.radians(30.0)
"""How far the pushing hand leans off vertical, its fingers towards the case [rad]."""
PITCH = 0.5 * math.pi - TILT
"""The pushing finger's outer face rises at this angle from the bench [rad]."""
LEAN = arm_gripper.axis_angle_quat((0.0, 1.0, 0.0), TILT)
PUSH_QUAT = tuple(float(v) for v in arm_gripper.quat_multiply(LEAN, arm_gripper.GRASP_QUAT))
"""The grasp frame leaned ``TILT`` about y: the jaws stay in the x-z plane, shut."""
PUSH_ANGLE = 95.0
"""How far round the hinge the finger follows the tab [deg]. The lid's weight is past the hinge at 71
degrees and, as a beam, the web's spring gives way to it at 82."""
FACE_REACH = 0.003
"""The tab's edge sits this far up the finger's face from its tip [m]."""
BACK_OFF = 0.025
"""The finger waits this far back along its own length before it slides under the tab [m]."""
HOVER = 0.060
LIFT = 0.045
"""How far the bar is lifted [m]: it stands 14 mm deep in the tray."""
GRIP_ABOVE_RIM = 0.003
"""The pads' lower ends stay this far over the tray's rim [m]."""
GRIP_SQUEEZE = 0.0002
"""How far inside the bar's sides each pad is commanded [m], the stamp scene's: the fingers stall on their 20 N."""
OPEN_WIDTH = 0.064
"""The pad faces' gap over the 55 mm bar [m], of the hand's 80: the pads' backs stay inside the
tray's 64 mm, off the open lid lying behind it."""
GRIP_LIMIT_N = arm_gripper.GRASP_LIMIT_N
GRIP_MU = arm_gripper.GRIP_MU
SETTLE_S = 0.6
MAX_IK_ERROR_M = 0.010
HELD = ("lift", "carry", "lower")
PHASES = (
    "settle",
    "reach",
    "flip",
    "clear",
    "descend",
    "close",
    "lift",
    "carry",
    "lower",
    "open",
    "retreat",
    "rest",
)


def load_bundle(folder: str):
    """A bundle's ``(vertices, tets, labels, rigid groups, part names)``, its files checked against ``SOURCE.md``."""
    path = ASSETS / folder
    listed = re.findall(r"- `([^`]+)`: `([0-9a-f]{64})`", (path / "SOURCE.md").read_text(encoding="utf-8"))
    if not listed:
        raise ValueError(f"{folder}: SOURCE.md lists no file hashes")
    for relative, expected in listed:
        if hashlib.sha256((path / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{folder} source asset changed: {relative}")
    vertices, tets, labels, _elastic, _density = load_labelled_tets(
        path / "final_mesh.mesh", path / "heterogeneous_params.npz"
    )
    meta = json.loads((path / "asset.json").read_text(encoding="utf-8"))
    groups = tuple(tuple(g) if len(g) > 1 else g[0] for g in meta["rigid_groups"])
    parts = json.loads((path / "inferred_params.json").read_text(encoding="utf-8"))["predictions"]
    names = tuple(part["part_name"] for part in sorted(parts, key=lambda part: part["part_index"]))
    return vertices, tets, labels, groups, names


def layout(vertices: np.ndarray, tets: np.ndarray, labels: np.ndarray) -> dict:
    """What the scene is placed and judged by, measured from the case's mesh (its own frame, z up)."""
    if set(np.unique(labels).tolist()) != set(range(4)):
        raise ValueError(f"{CASE}: expected part labels 0 to 3")
    nodes = [np.unique(tets[labels == part]) for part in range(4)]
    web, tab = nodes[HINGE], nodes[TAB]
    web_z = vertices[web, 2]
    web_y = vertices[web, 1]
    # The rims that face each other, on the tab's side: the tray's top and the lid's bottom there.
    front = float(vertices[tab, 1].min())
    tray_top = float(vertices[nodes[TRAY], 2].max())
    lid_bottom = float(vertices[nodes[LID], 2].min())
    tray, lid = vertices[nodes[TRAY]], vertices[nodes[LID]]
    tray_rim = nodes[TRAY][(tray[:, 2] >= tray_top - 1e-7) & (tray[:, 1] >= front - 0.0021)]
    lid_rim = nodes[LID][(lid[:, 2] <= lid_bottom + 1e-7) & (lid[:, 1] >= front - 0.0021)]
    inside = nodes[TRAY][(np.abs(tray[:, 0]) < 0.03) & (np.abs(tray[:, 1]) < 0.015)]
    if not len(tray_rim) or not len(lid_rim) or not len(inside):
        raise ValueError(f"{CASE}: a rim or the tray's floor is missing")
    thickness, width = float(np.ptp(web_y)), float(np.ptp(vertices[web, 0]))
    return {
        "nodes": nodes,
        "tray_rim": tray_rim,
        "lid_rim": lid_rim,
        "floor_top": float(vertices[inside, 2].max()),
        "report": {
            "rim_gap_m": lid_bottom - tray_top,
            "web_free_length_m": float(np.ptp(web_z)),
            "web_thickness_m": thickness,
            "web_width_m": width,
            "tab_depth_m": float(np.ptp(vertices[tab, 1])),
            "tab_thickness_m": float(np.ptp(vertices[tab, 2])),
            "tab_to_hinge_m": float(vertices[tab, 1].mean() - web_y.mean()),
        },
    }


def carried(points: np.ndarray, angle: float, hinge: tuple[float, float, float]) -> np.ndarray:
    """Where the lid has ``points`` ``[.., (across, z)]`` once it has turned ``angle`` [rad] on the web.

    ``across`` is the level axis the lid swings in, from the hinge towards the tab. ``hinge`` is the
    web's mid-thickness along it and its free length's lower and upper z, as the lid sits shut. The
    web is taken to bend at one curvature along that length and the lid to ride on its upper end; a
    positive angle takes the lid's top away from the tab.
    """
    across, low, high = hinge
    if abs(angle) < 1.0e-9:
        return np.array(points, dtype=np.float64)
    length = high - low
    end = np.array((across - length * (1.0 - np.cos(angle)) / angle, low + length * np.sin(angle) / angle))
    d = np.asarray(points, dtype=np.float64) - (across, high)
    c, s = np.cos(angle), np.sin(angle)
    return end + np.stack((d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c), axis=-1)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def lid_angle_deg(tray: np.ndarray, lid: np.ndarray) -> float:
    """How far the lid's body has turned from the tray's about ``HINGE_AXIS`` [deg]; open is positive."""
    rel = _quat_mul(np.array((-tray[3], -tray[4], -tray[5], tray[6])), lid[3:7])
    angle = float(np.degrees(2.0 * np.arctan2(float(np.dot(rel[:3], HINGE_AXIS)), rel[3])))
    angle = (angle + 180.0) % 360.0 - 180.0
    return angle + 360.0 if angle < -90.0 else angle


def spec_for(folder: str, groups, pose, settings: PhysicsSettings, overrides=()) -> rigid_part_body.RigidPartSoftSpec:
    path = ASSETS / folder
    return rigid_part_body.RigidPartSoftSpec(
        rigid_parts=groups,
        name=folder,
        mesh_path=str(path / "final_mesh.mesh"),
        params_path=str(path / "heterogeneous_params.npz"),
        materials_path=str(path / "inferred_params.json"),
        pose=pose,
        quat=CASE_QUAT,
        contact_ke=settings.contact_ke,
        contact_kd=settings.contact_kd,
        contact_mu=settings.soft_mu,
        particle_radius=settings.particle_radius,
        damping_ratio=settings.damping_ratio,
        part_overrides=tuple(overrides),
    )


def check_build(sim, entities, settings: PhysicsSettings, *, mu: float | None) -> dict:
    """Check what Newton built against each bundle's materials, apply the contact settings, and report."""
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    if not type(solver).__name__.startswith("SolverVBD"):
        raise RuntimeError(f"expected a VBD solver, got {type(solver).__name__}")
    if sim.model.tet_count != sum(len(entity.source_tets) for entity in entities):
        raise RuntimeError("RSS lost tetrahedra")
    if sim.model.attachment_particle_body_count != sum(len(rp.nodes) for e in entities for rp in e.rigid_parts):
        raise RuntimeError("a rigid group lost node attachments")
    report = {"friction": set_contact(sim, solver, settings, mu=mu, grip_mu=None), "rigid_bodies": {}}
    for entity in entities:
        span = slice(*entity.tet_range)
        if not np.array_equal(sim.model.tet_indices.numpy()[span] - entity.particles[0], entity.source_tets):
            raise RuntimeError(f"{entity.spec.name}: Newton changed the source tetrahedron order")
        young, nu = entity.elastic.T
        lame = np.column_stack((young / (2.0 * (1.0 + nu)), young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))
        if not np.allclose(sim.model.tet_materials.numpy()[span, :2], lame, rtol=1e-5):
            raise RuntimeError(f"{entity.spec.name}: Newton FEM elasticity differs from the materials")
        if not np.allclose(sim.model.particle_mass.numpy()[entity.particles], entity.mass, rtol=1e-5):
            raise RuntimeError(f"{entity.spec.name}: Newton nodal mass differs from the per-tet densities")
        for rp in entity.rigid_parts:
            report["rigid_bodies"][f"{entity.spec.name}/part{rp.part}"] = {
                "body": rp.body,
                "nodes": int(len(rp.nodes)),
                "seam_nodes": int(rp.seam.sum()),
                "part_mass_kg": rp.part_mass,
            }
    return report


def tip_at(angle: float, edge, hinge) -> np.ndarray:
    """Where the pushing fingertip's corner goes for the lid at ``angle`` [rad]: ``(x, z)`` [m, world].

    ``edge`` is the tab's lower front edge as the lid sits shut and ``hinge`` the web, both in the
    x-z plane (:func:`carried`). Up to the finger's pitch the edge is ``FACE_REACH``
    up the finger's face; past it the corner is that far in under the tab. Either way the finger's
    surface is one particle radius off the tab's nodes.
    """
    at = carried(np.asarray(edge, dtype=np.float64), angle, hinge)
    phi = max(angle, PITCH)
    along = np.array((math.cos(phi), math.sin(phi)))
    normal = np.array((-math.sin(phi), math.cos(phi)))
    return at - FACE_REACH * along - PHYSICS.particle_radius * normal


def build_scene(
    *,
    hinge_young: float = HINGE_YOUNG,
    solver_name: str = stamp_press.SOLVER,
    iterations: int = 160,
    mu: float | None = None,
    grip_mu: float = GRIP_MU,
    force_limit: float = GRIP_LIMIT_N,
):
    vertices, tets, labels, groups, names = load_bundle(CASE)
    soap_vertices, _t, _l, soap_groups, _n = load_bundle(SOAP)
    geometry = layout(vertices, tets, labels)
    settings = PHYSICS
    physics = arrangement(
        settings,
        claw="arm",
        coarse=False,
        solver_name=solver_name,
        iterations=iterations,
        self_contact=True,
        self_contact_radius=SELF_CONTACT_RADIUS,
        rest_exclusion=REST_EXCLUSION,
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="soap_case_arm_pick")
    pose = (CASE_AT[0], CASE_AT[1], settings.particle_radius + PLACE_GAP - float(vertices[:, 2].min()))
    softer = ((HINGE, "youngs_modulus_pa", hinge_young),)
    case = scene.add(spec_for(CASE, groups, pose, settings, softer))
    on_floor = (pose[0], pose[1], pose[2] + geometry["floor_top"] + SOAP_GAP)
    bar = scene.add(spec_for(SOAP, soap_groups, on_floor, settings))
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    middle = surface + np.array((0.5 * (CASE_AT[0] + PLACE_AT[0]), CASE_AT[1], 0.0))
    foot = surface + np.array((CASE_AT[0], CASE_AT[1], 0.0))
    for camera in (
        # From across the bench, the arm behind: the flip, the pick and the carry all in frame.
        rss.CameraSpec(
            name="overview",
            width=960,
            height=540,
            eye=tuple(float(v) for v in np.add(middle, (0.30, 0.42, 0.26))),
            look_at=tuple(float(v) for v in np.add(middle, (0.0, -0.04, 0.07))),
            fov_y=40.0,
        ),
        # Along the hinge from the side the arm is not on: the lid's swing and the bar's path are in plane.
        rss.CameraSpec(
            name="side",
            width=960,
            height=540,
            eye=tuple(float(v) for v in np.add(foot, (0.03, 0.34, 0.09))),
            look_at=tuple(float(v) for v in np.add(foot, (0.03, 0.0, 0.035))),
            fov_y=34.0,
        ),
    ):
        scene.add(camera)
    sim = build_sim(scene, rigid_loop=False)
    report = dict(geometry["report"])
    report.update(check_build(sim, [case.entity, bar.entity], settings, mu=mu))
    report["arm"] = arm_gripper.check_arm(sim, limit=force_limit)
    soft_mu = report["friction"]["global_soft_contact_mu"]
    report["friction"].update(arm_gripper.set_contact_friction(sim, global_mu=soft_mu, grip_mu=grip_mu))
    young = case.entity.elastic[:, 0]
    report["effective_young_pa_by_part"] = {names[p]: float(young[labels == p].max()) for p in range(4)}
    report["case_pose_m"] = [float(v) for v in pose]
    report["soap_half_width_m"] = float(0.5 * np.ptp(soap_vertices[:, 1]))
    report["soap_weight_n"] = float(bar.entity.mass.sum() * 9.81) + float(bar.entity.rigid_parts[0].body_mass * 9.81)
    print(json.dumps({"geometry": report}), flush=True)
    return sim, case, bar, geometry, report


def judge(rows: list[dict], complete: bool) -> dict:
    """What the arm did with the case and the bar, from the per-frame rows."""
    rest = next((row for row in reversed(rows) if row["phase"] == "settle"), rows[0])
    final = rows[-1]
    after_flip = [row for row in rows if row["phase"] in ("clear", "descend")]
    place = np.asarray(PLACE_AT)
    result = {
        "complete": complete,
        "finite": all(row["finite"] for row in rows),
        "rest_lid_angle_deg": rest["lid_angle_deg"],
        "rest_front_rim_gap_m": rest["front_rim_gap_m"],
        "peak_lid_angle_deg": max(row["lid_angle_deg"] for row in rows),
        "lid_angle_before_pick_deg": after_flip[-1]["lid_angle_deg"] if after_flip else final["lid_angle_deg"],
        "final_lid_angle_deg": final["lid_angle_deg"],
        "max_tray_tilt_deg": max(row["tray_tilt_deg"] for row in rows),
        "max_tray_slide_m": max(row["tray_slide_m"] for row in rows),
        "max_tray_lift_m": max(row["tray_lift_m"] for row in rows),
        "max_soap_rise_m": max(row["soap_rise_m"] for row in rows),
        "final_soap_at_m": final["soap_at_m"],
        "final_soap_miss_m": float(np.linalg.norm(np.asarray(final["soap_at_m"]) - place)),
        "final_soap_bottom_m": final["soap_bottom_m"],
        "final_soap_tilt_deg": final["soap_tilt_deg"],
        # While the pads hold it: after they open, the hand leaves and the bar does not.
        "max_grip_slip_m": max((abs(row["grip_slip_m"]) for row in rows if row["phase"] in HELD), default=None),
        "peak_squeeze_n": max((row["squeeze_n"] for row in rows if "squeeze_n" in row), default=None),
        "final_aperture_m": final["aperture_m"],
        "peak_web_stretch": max(row["web_stretch_max"] for row in rows),
        "max_inverted_web_tets": max(row["inverted_web_tets"] for row in rows),
        "final_inverted_web_tets": final["inverted_web_tets"],
        "max_rigid_node_stretch_m": max(row["rigid_node_stretch_m"] for row in rows),
    }
    result["checks"] = {
        "ran": complete and result["finite"],
        "shut_at_rest": abs(rest["lid_angle_deg"]) < 3.0 and 0.0 < rest["front_rim_gap_m"] < 0.0015,
        "lid_open_for_the_pick": result["lid_angle_before_pick_deg"] >= 120.0,
        "tray_stays_put": result["max_tray_tilt_deg"] < 3.0 and result["max_tray_slide_m"] < 0.005,
        "soap_lifted_out": result["max_soap_rise_m"] >= 0.03,
        # Down on the bench within 10 mm of where it was sent, lying flat, and let go of.
        "soap_set_down": result["final_soap_miss_m"] < 0.010
        and result["final_soap_bottom_m"] < 0.002
        and result["final_soap_tilt_deg"] < 5.0,
        "hand_open": result["final_aperture_m"] >= 0.9 * OPEN_WIDTH,
        "web_intact": result["final_inverted_web_tets"] == 0,
    }
    return result


def run(
    out: Path,
    *,
    render_every: int = 0,
    until: str = PHASES[-1],
    push_angle: float = PUSH_ANGLE,
    hinge_young: float = HINGE_YOUNG,
    solver_name: str = stamp_press.SOLVER,
    iterations: int = 160,
    mu: float | None = None,
    grip_mu: float = GRIP_MU,
    force_limit: float = GRIP_LIMIT_N,
    squeeze: float = GRIP_SQUEEZE,
):
    out.mkdir(parents=True, exist_ok=True)
    sim, case, bar, geometry, report = build_scene(
        hinge_young=hinge_young,
        solver_name=solver_name,
        iterations=iterations,
        mu=mu,
        grip_mu=grip_mu,
        force_limit=force_limit,
    )
    entity, soap = case.entity, bar.entity
    control = arm_gripper.ArmControl(sim)
    particles, soap_particles = np.asarray(entity.particles), np.asarray(soap.particles)
    tets, labels = entity.source_tets, entity.tet_part_labels
    nodes = geometry["nodes"]
    web_tets = tets[labels == HINGE]
    tray_body, lid_body = (rp.body for rp in entity.rigid_parts)
    soap_body = soap.rigid_parts[0].body
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    radius = PHYSICS.particle_radius
    quality = MeshQuality(sim, entity)
    trajectory = Trajectory(sim, entity)
    arms = sim.handles.arms[arm_gripper.ARM]
    finger_sides = [
        np.array([int(shape) for _, shapes in side for shape in shapes], dtype=np.int32) for side in arms.finger_groups
    ]
    shape_forces = getattr(sim.pipeline.program.solver, "shape_contact_forces", None)
    start = sim.pipeline.state_0.particle_q.numpy()[particles].astype(np.float64)
    web_rest_inv = np.linalg.inv(start[web_tets[:, 1:]] - start[web_tets[:, :1]])
    start_body = sim.pipeline.state_0.body_q.numpy().astype(np.float64)
    closed = 2.0 * (report["soap_half_width_m"] - squeeze)
    push_quat = np.asarray(PUSH_QUAT, dtype=np.float64)
    grasp_quat = np.asarray(arm_gripper.GRASP_QUAT, dtype=np.float64)
    # The pushing corner in the tool's frame, until the built hand is measured: the stock pad is
    # 40 mm long about the tool centre and 4 mm thick.
    corner = np.array((0.0, -2.0 * arm_gripper.PAD_HALF_THICKNESS, 0.020))

    rows, phases, executed = [], [], []
    frame, worst_ik, held = 0, 0.0, None
    owned = start
    gap = 0.0
    plan = {}

    def tool_at(tip_xz, y: float) -> np.ndarray:
        """The tool centre that puts the pushing corner at ``tip_xz`` in the x-z plane at ``y`` [m, world]."""
        return np.array((tip_xz[0], y, tip_xz[1])) - arm_gripper._rotate(push_quat, corner[None])[0]

    def read_tab() -> None:
        """Where the tab and the web are now, for the finger's path."""
        tab, web = owned[nodes[TAB]], owned[nodes[HINGE]]
        plan["edge"] = (float(tab[:, 0].max()), float(tab[:, 2].min()))
        plan["hinge"] = (float(web[:, 0].mean()), float(web[:, 2].min()), float(web[:, 2].max()))
        plan["y"] = float(tab[:, 1].mean())

    read_tab()
    axis = np.array((-math.sin(TILT), -math.cos(TILT)))  # the fingers' own direction in the x-z plane
    target = tool_at(tip_at(0.0, plan["edge"], plan["hinge"]) - BACK_OFF * axis, plan["y"])
    quat = push_quat

    videos, renderer, video_error = None, None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet
        from robosimstudio.solvers.render import FALLBACK

        backend, _why = sim.platform.renderer_for("")
        renderer = sim.make_renderer(backend, cameras=list(sim.cameras), channels=["rgb"])
        if backend != FALLBACK:
            renderer.render(sim.state)  # the first frame only builds the stage
        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // CONTROL_HZ))
        videos.__enter__()

    def blend_over(frames: int):
        span = (frames - 1) / CONTROL_HZ

        def blend(clock: float) -> float:
            x = min(1.0, clock / span)
            return x * x * (3.0 - 2.0 * x)

        return blend

    def route(begin, end, q_begin, q_end, gap_begin: float, gap_end: float, seconds: float):
        """A smoothstep from one tool pose and pad gap to another, as ``rubber_stamp_arm_press`` plans one."""
        frames = max(2, math.ceil(seconds * CONTROL_HZ))
        q_begin, q_end = np.asarray(q_begin, dtype=np.float64), np.asarray(q_end, dtype=np.float64)
        if np.dot(q_begin, q_end) < 0:
            q_end = -q_end
        blend = blend_over(frames)

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
        """One control frame, and its row."""
        nonlocal frame, owned
        sim.run(1).sync()
        frame += 1
        state = sim.pipeline.state_0
        q, body_q = state.particle_q.numpy(), state.body_q.numpy()
        owned = q[particles].astype(np.float64)
        block_nodes = q[soap_particles].astype(np.float64)
        edges = owned[web_tets[:, 1:]] - owned[web_tets[:, :1]]
        stretch = np.linalg.svd(web_rest_inv @ edges, compute_uv=False).max(axis=1)
        tray, lid = body_q[tray_body].astype(np.float64), body_q[lid_body].astype(np.float64)
        block = body_q[soap_body].astype(np.float64)
        tcp = np.asarray(control.tcp(), dtype=np.float64)
        parts = rigid_part_body.frame_record(entity, owned, body_q)
        row = {
            "frame": frame,
            "time_s": frame / CONTROL_HZ,
            "phase": name,
            "finite": bool(np.isfinite(owned).all() and np.isfinite(body_q).all()),
            "tcp_m": (tcp - surface).tolist(),
            "commanded_tcp_m": (target - surface).tolist(),
            "aperture_m": 2.0 * control.aperture(),
            "lid_angle_deg": lid_angle_deg(tray, lid),
            "front_rim_gap_m": float(owned[geometry["lid_rim"], 2].mean() - owned[geometry["tray_rim"], 2].mean()),
            "tray_tilt_deg": stamp_press._tilt_deg(tray),
            "tray_slide_m": float(np.linalg.norm(tray[:2] - start_body[tray_body, :2])),
            "tray_lift_m": float(tray[2] - start_body[tray_body, 2]),
            "soap_at_m": (block[:2] - surface[:2]).tolist(),
            "soap_rise_m": float(block[2] - start_body[soap_body, 2]),
            "soap_bottom_m": float(block_nodes[:, 2].min() - surface[2] - radius),
            "soap_tilt_deg": stamp_press._tilt_deg(block),
            "web_stretch_max": float(stretch.max()),
            "inverted_web_tets": int(np.count_nonzero(np.linalg.det(edges) <= 0)),
            "rigid_node_stretch_m": float(
                max(max(p["seam_stretch_max_m"], p["skin_stretch_max_m"]) for p in parts.values())
            ),
        }
        if held is not None:
            # How far the bar has moved down in the pads since they closed on it.
            row["grip_slip_m"] = float((tcp[2] - block[2]) - held)
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

    def follow(name: str, path) -> None:
        nonlocal worst_ik
        worst_ik = max(worst_ik, path.worst_position_error_m)
        if path.worst_position_error_m > MAX_IK_ERROR_M:
            raise RuntimeError(f"{name}: the arm misses its path by {path.worst_position_error_m:.4f} m")
        for index in range(len(path.joint_q)):
            control.follow(path, index)
            step(name)
        finish(name, len(path.joint_q))

    def settle() -> None:
        """Hold the shut, leaning hand in front of the tab while the case settles, then measure the hand."""
        nonlocal target, corner
        hold = route(target, target, push_quat, push_quat, gap, gap, 0.1)
        if hold.worst_position_error_m > MAX_IK_ERROR_M:
            raise RuntimeError(f"the waiting pose is unreachable: {hold.worst_position_error_m:.4f} m off")
        control.start(hold)
        count = round(SETTLE_S * CONTROL_HZ)
        for index in range(count):
            control.follow(hold, index)
            step("settle")
        position, orientation = control.arm.ee_pose(sim.pipeline.state_0)
        position, orientation = np.asarray(position, dtype=np.float64), np.asarray(orientation, dtype=np.float64)
        inverse = np.array((-orientation[0], -orientation[1], -orientation[2], orientation[3]))
        local = arm_gripper._rotate(inverse, arm_gripper.pad_corners(sim) - position)
        measured = np.array((0.0, float(local[:, 1].min()), float(local[:, 2].max())))
        print(json.dumps({"pushing_corner_in_tool_m": measured.tolist(), "assumed_m": corner.tolist()}), flush=True)
        if not (0.005 < measured[2] < 0.200 and -0.012 < measured[1] < -0.002):
            raise RuntimeError(f"the hand's pads are not where the path expects them: {measured.tolist()}")
        corner = measured
        report["pushing_corner_in_tool_m"] = measured.tolist()
        target = position
        read_tab()
        finish("settle", count)

    def act(name: str, end, *, end_quat=None, opening: float | None = None, seconds: float | None = None) -> None:
        """One action: from the measured tool pose and gap to the commanded ones."""
        nonlocal target, quat, gap
        end = np.asarray(end, dtype=np.float64)
        position, orientation = control.arm.ee_pose(sim.pipeline.state_0)
        position = np.asarray(position, dtype=np.float64)
        if seconds is None:
            seconds = max(0.8, float(np.linalg.norm(end - position)) / 0.06)
        end_gap = gap if opening is None else float(opening)
        # A gap that is not changing stays the commanded one: read back, a stalled pad's is wider.
        begin_gap = gap if opening is None else 2.0 * control.aperture()
        end_quat = quat if end_quat is None else np.asarray(end_quat, dtype=np.float64)
        path = route(position, end, orientation, end_quat, begin_gap, end_gap, seconds)
        target, quat, gap = end, end_quat, end_gap
        executed.append({"phase": name, "to_m": (end - surface).tolist(), "seconds": seconds, "opening_m": end_gap})
        follow(name, path)

    def reach() -> None:
        """Slide the finger down its own length until its face is under the tab's edge."""
        act("reach", tool_at(tip_at(0.0, plan["edge"], plan["hinge"]), plan["y"]), seconds=1.2)

    def flip(seconds: float = 3.0) -> None:
        """Follow the tab round the hinge; the lid goes on alone once its weight is past it."""
        nonlocal target
        frames = math.ceil(seconds * CONTROL_HZ)
        blend = blend_over(frames)
        sweep = math.radians(push_angle)
        path = control.plan(
            quat=push_quat,
            capacity=frames,
            dt=1.0 / CONTROL_HZ,
            tcp_at=lambda clock: tool_at(tip_at(sweep * blend(clock), plan["edge"], plan["hinge"]), plan["y"]),
            half_gap_at=lambda clock: 0.0,
        )
        target = tool_at(tip_at(sweep, plan["edge"], plan["hinge"]), plan["y"])
        executed.append({"phase": "flip", "push_angle_deg": push_angle, "seconds": seconds})
        follow("flip", path)

    def clear() -> None:
        """Stand the hand up over the bar, open, and give the lid time to land."""
        state = sim.pipeline.state_0
        block = state.body_q.numpy()[soap_body].astype(np.float64)
        rim = float(owned[geometry["tray_rim"], 2].mean())
        # The pads' lower ends just over the rim; the tool centre is the pads' length above them.
        plan["grasp"] = np.array((block[0], block[1], rim + radius + GRIP_ABOVE_RIM + corner[2]))
        act("clear", plan["grasp"] + np.array((0.0, 0.0, HOVER)), end_quat=grasp_quat, opening=OPEN_WIDTH, seconds=2.4)
        if rows[-1]["lid_angle_deg"] < 120.0:
            raise RuntimeError(f"the lid is at {rows[-1]['lid_angle_deg']:.1f} degrees, not open: the hand stays up")

    def close() -> None:
        nonlocal held
        act("close", target, opening=closed, seconds=1.0)
        state = sim.pipeline.state_0
        held = float(np.asarray(control.tcp())[2] - state.body_q.numpy()[soap_body][2])

    def carry() -> None:
        """Over ``PLACE_AT``, by where the bar is now: the hand took it where it lay, not where it was put."""
        block = sim.pipeline.state_0.body_q.numpy()[soap_body].astype(np.float64)
        to = surface[:2] + np.asarray(PLACE_AT) - block[:2]
        act("carry", target + np.array((to[0], to[1], 0.0)), seconds=2.2)

    def lower() -> None:
        """Down until the bar's underside is half a millimetre over where it rests on the bench."""
        block_nodes = sim.pipeline.state_0.particle_q.numpy()[soap_particles]
        drop = float(block_nodes[:, 2].min()) - (surface[2] + radius + 0.0005)
        act("lower", target - np.array((0.0, 0.0, drop)), seconds=1.6)

    def rest() -> None:
        count = round(0.6 * CONTROL_HZ)
        for _ in range(count):
            step("rest")
        finish("rest", count)

    def steps():
        """The script. Each line is one phase; a caller may stop after any of them."""
        yield "settle", settle
        yield "reach", reach
        yield "flip", flip
        yield "clear", clear
        yield "descend", lambda: act("descend", plan["grasp"], seconds=1.4)
        yield "close", close
        yield "lift", lambda: act("lift", target + np.array((0.0, 0.0, LIFT)), seconds=1.4)
        yield "carry", carry
        yield "lower", lower
        yield "open", lambda: act("open", target, opening=OPEN_WIDTH, seconds=0.8)
        yield "retreat", lambda: act("retreat", target + np.array((0.0, 0.0, HOVER)), seconds=1.0)
        yield "rest", rest

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

    if rows:
        trajectory.save(out / "trajectory.npz")
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    complete = error is None and bool(phases) and phases[-1]["phase"] == PHASES[-1]
    result = judge(rows, complete) if rows else {"checks": {"ran": False}}
    result["success"] = all(result["checks"].values())
    result["stopped"] = error
    result["mesh_quality"] = quality.read()
    result["actions"], result["phases"] = executed, phases
    result["worst_ik_error_m"] = worst_ik
    result["config"] = {
        "push_angle_deg": push_angle,
        "tilt_deg": math.degrees(TILT),
        "hinge_young_pa": hinge_young,
        "grip_mu": grip_mu,
        "force_limit_n": force_limit,
        "grip_squeeze_m": squeeze,
        "case_at_m": CASE_AT,
        "place_at_m": PLACE_AT,
        "solver": type(arm_gripper.vbd_solver(sim.pipeline.program.solver)).__name__,
        "iterations": iterations,
        "substeps": PHYSICS.substeps,
        "control_hz": CONTROL_HZ,
        "newton_version": newton.__version__,
    }
    result["geometry"] = report
    result["video_error"] = video_error
    result["simulated_s"] = len(rows) / CONTROL_HZ
    result["elapsed_s"] = elapsed
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "actions", "phases")}), flush=True)
    sim.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--until", choices=PHASES, default=PHASES[-1], help="stop after this phase")
    parser.add_argument("--push-angle", type=float, default=PUSH_ANGLE, help="how far the finger follows the tab [deg]")
    parser.add_argument("--hinge-young", type=float, default=HINGE_YOUNG, help="the web's Young's modulus [Pa]")
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default=stamp_press.SOLVER)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--mu", type=float, default=None, help="the soft-contact friction")
    parser.add_argument("--grip-mu", type=float, default=GRIP_MU, help="pad friction against the case and the bar")
    parser.add_argument("--force-limit", type=float, default=GRIP_LIMIT_N, help="one finger's force cap [N]")
    parser.add_argument("--squeeze", type=float, default=GRIP_SQUEEZE, help="pad command inside the bar's sides [m]")
    args = parser.parse_args()
    run(
        args.out,
        render_every=args.render_every,
        until=args.until,
        push_angle=args.push_angle,
        hinge_young=args.hinge_young,
        solver_name=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        grip_mu=args.grip_mu,
        force_limit=args.force_limit,
        squeeze=args.squeeze,
    )


if __name__ == "__main__":
    main()
