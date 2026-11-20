# SPDX-License-Identifier: Apache-2.0
"""Plug the USB hub's cable into a socket: grasp the rigid plug, carry it, push it in.

The hub lies flat with its cable laid in a hook on the bench (:func:`cable_scene.lay`; the rest
shape stays straight), the plug's tip pointing +y. A rigid USB-A socket block (a static
triangle mesh that keeps its concave cavity) stands on a slab straight ahead of the plug, its mouth
facing -y. The arm hovers while the laid cable settles (``SETTLE``) and plans its grasp from where
the plug came to rest. It pinches the plug's collar across its width (the pads close along x),
lifts it straight up to the cavity's axis, measures how the plug sits in the pads and turns and
shifts the hand to put it on the axis, re-centres it across the axis (``TRIMS_AT``) as it
approaches and pushes it in along +y, lets go and withdraws upwards. The cable trails behind and the hook gives the push its slack.

What the earlier runs taught: backing the plug over its cable (q1) buckled the stiff cable against
the collar and pitched the plug 35 degrees in the pinch, which turns freely about the pads'
closing axis; a sideways carry pitched it 10 degrees in both q1 and q2; and with the cavity 12 mm
over the bench (q2) the plug's contact skin rode on the bench, so in the push it yawed 25 degrees
and the pads slid 17 mm along the collar while the plug moved 3 mm. So the plug only rises and
pushes, and the cavity stands ``PEDESTAL`` up, where nothing but the socket is near it.

Why heterogeneous: the insertion needs the plug to keep its shape and orientation in the pads
while the cable bends and trails behind it, and the housing to stay on the bench at the cable's
other end. A soft plug would squash at the mouth and in the pads; a stiff cable would lever the
plug out of line with the cavity and lift or fling the housing.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import arm_gripper
import cable_scene
import hetero_scene
import newton
import numpy as np
import usb_hub_box
from clawhauser_scene import REFERENCE, arrangement, build_sim
from loop_geometry import smooth
from parallel_gripper import ParallelGripperSpec
from scipy.spatial.transform import Rotation, Slerp

import robosimstudio as rss
from robosimstudio.specs import AssetSpec, StaticSpec

HOUSING, COLLAR, TIP = 0, 2, 3
PLUG = (COLLAR, TIP)
SOCKETS = {"snug": "socket_usb_a_snug", "loose": "socket_usb_a_loose"}
"""``--socket``: 1 mm of clearance per side, or the loose 4 mm made for a 3 mm contact skin."""
PLUG_RADIUS = 0.0003
"""The plug's own contact radius [m] (``--plug-radius``; :func:`hetero_scene.set_part_radius`): a thin
skin so it fits the snug socket, while the cable and housing keep the scene's 3 mm."""
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "cable_plug_place"
CONTROL_HZ = REFERENCE.control_hz
SUBSTEPS = usb_hub_box.SUBSTEPS
DURATION = 12.4
PLUG_AT = (0.10, 0.0)
"""The plug's shoulder once the cable is laid [m, bench frame]."""
BEND_RADIUS = 0.045
HOOK = 0.75 * np.pi
BENDS = (
    (0.02, 0.0),
    (HOOK * BEND_RADIUS, BEND_RADIUS),
    (HOOK * BEND_RADIUS, -BEND_RADIUS),
)
"""The laid cable from the housing (:func:`cable_scene.bend_path`): a hook, 135 degrees left then 135
right at 45 mm radius, so the plug still points +y, about 154 mm to -x of the housing and 150 mm
nearer it than straight (the cable is only ~250 mm long, too short for two U-turns)."""
SETTLE = 0.5
"""How long the laid cable settles before the arm moves [s]; the hand hovers ``HOVER`` over the plug."""
HOVER = 0.04
GRASP_BEHIND = 0.020
"""Where the pads close: this far behind the shoulder along the collar [m]. The pads are 30 mm wide
along it, so they cover 23 of the collar's 30 mm and their front edge stays 5 mm behind the
shoulder, about 10 mm clear of the socket's face when the plug is seated."""
PEDESTAL = 0.03
"""The slab under the socket [m]: the cavity's axis 42 mm over the bench, so the lifted plug's
skin and the pads' bottoms stay well clear of it."""
PRE_INSERT = 0.0225
"""The seated shoulder lies this far ahead (+y) of the plug's start [m]: the tip's contact skin
starts 7 mm short of the slab's face, rises past it and is pushed the rest."""
MAX_TURN = 10.0
"""The most the measured correction tilts the hand [deg]; it turns about z freely. Undoing a nose-up pitch leans the fingers'
tops towards the block: at 10 degrees they stay 3 mm behind the pads' front edge at the block's
top. q1's lift pitched the plug 8 degrees."""
OVER_TRAVEL = 0.002
"""How far past the nominal seat the hand's path ends [m]: the collar's face stops the plug first."""


def tolerances(clearance: float, tip_length: float, contact_offset: float) -> dict:
    """How near the socket's seated pose a released plug must sit to count as seated, from the cavity.

    Along the axis: the offset contact leaves (the collar's face stops a plug radius before the
    block's face, 1.5 mm ahead of the shoulder) plus 2 mm. Across it: the clearance plus 1 mm, as
    the cable's weight tips the released plug within it. Tilt: what the clearance allows on the tip,
    ``atan(2 c / l)`` (snug: 1.5 mm and 9.5 deg; loose: 4.5 mm and 34 deg).
    """
    return {
        "axial_m": abs(contact_offset) + 0.002,
        "lateral_m": clearance + 0.001,
        # At least 15 deg: a seated plug in the snug cavity measured 11.8 deg (q7), its thin skin
        # pressing into the walls, so the cavity bound alone is too strict for the measure.
        "angle_deg": max(15.0, float(np.degrees(np.arctan2(2.0 * clearance, tip_length)))),
    }


WITHDRAW_Z = 0.06
PHASES = {
    "descend": (0.55, 1.0),
    "close": (1.1, 1.7),
    "lift": (2.0, 3.5),
    "align": (4.1, 4.8),
    "approach": (5.6, 6.4),
    "insert": (7.0, 9.3),
    "open": (9.8, 10.3),
    "withdraw": (10.6, 11.3),
}
"""Phase windows [s]: straight moves. The plug's pose is measured at ``MEASURE_AT`` (turn and
shift, in "align"); "approach" brings the tip's skin ``APPROACH`` nearer, 2 mm short of the face,
and "insert" pushes the rest and holds 0.5 s before "open". The last 1.1 s check the plug stays in after the release."""
MEASURE_AT = 4.0
TRIMS_AT = (5.3, 6.7, 7.3, 7.7, 8.2, 8.7)
"""When the plug is measured again and the hand shifted across the axis (over the next 0.1-0.3 s):
before the approach, before the push, just before the tip enters, just after and twice during the push. The hand creeps
about 1.5 mm/s across the axis while it holds (q4: it undid a 1.8 mm trim within a second and the
tip struck the mouth's side 2 mm off), so one trim is not enough for a 1 mm fit."""
APPROACH = 0.005
INSERT_AXIS = np.array((0.0, 1.0, 0.0))
"""The socket's +X in the bench frame: the asset's quarter turn about z, as the hub's."""


def layout(vertices, tets, labels, quat, female: dict, male: dict, plug_radius: float) -> dict:
    """Where the hub lies, where the pads close, and where the socket and its seated pose go."""
    turned = arm_gripper._rotate(np.asarray(quat), vertices)
    t_male = np.asarray(male["T_asset_from_mating_frame"])
    shoulder = arm_gripper._rotate(np.asarray(quat), t_male[None, :3, 3])[0]
    axes = Rotation.from_quat(quat).as_matrix() @ t_male[:3, :3]
    collar = turned[np.unique(tets[labels == COLLAR])]
    grasp_y = shoulder[1] - GRASP_BEHIND
    band = collar[np.abs(collar[:, 1] - grasp_y) < 0.004]
    reach = float(0.5 * (band[:, 0].max() - band[:, 0].min()))
    # Laid, the plug moves with the path's frame at its node nearest the cable's root (the housing's seam).
    root = turned[np.intersect1d(tets[labels == HOUSING], tets[labels != HOUSING])].mean(axis=0)
    direction = axes[:, 0]
    near = float(((turned[np.unique(tets[np.isin(labels, PLUG)])] - root) @ direction).min())
    point, heading = cable_scene.bend_path([near], BENDS, float(np.arctan2(direction[1], direction[0])))
    if abs(heading[0] - np.arctan2(direction[1], direction[0])) > 1e-9:
        raise ValueError("BENDS must leave the plug pointing as it did: the socket is placed along +y")
    laid = point[0] - near * direction[:2]
    shift = np.array(
        (
            PLUG_AT[0] - shoulder[0] - laid[0],
            PLUG_AT[1] - shoulder[1] - laid[1],
            -turned[:, 2].min() + hetero_scene.PLACE_GAP,
        )
    )
    # Settled: the collar's bottom a plug radius over the bench.
    settled_z = plug_radius + shoulder[2] - collar[:, 2].min()
    # Contact stops the collar's face a plug radius before the block's face.
    overhang = float(collar[:, 1].max() - shoulder[1])
    return {
        "pose": [float(c) for c in shift],
        "lay": {"root_m": (root + shift).tolist(), "direction": direction[:2].tolist(), "bends": BENDS},
        "grip_centre_m": [PLUG_AT[0], PLUG_AT[1] - GRASP_BEHIND, float(ParallelGripperSpec.pad_half_height + 0.001)],
        "reach_m": reach,
        "shoulder_rest_m": [PLUG_AT[0], PLUG_AT[1], float(shoulder[2] + shift[2])],
        "shoulder_settled_m": [PLUG_AT[0], PLUG_AT[1], float(settled_z)],
        "plug_axes": axes.tolist(),
        **socket_pose(female, male, overhang, plug_radius, PLUG_AT),
    }


def socket_pose(female: dict, male: dict, overhang: float, plug_radius: float, plug_at) -> dict:
    """The socket whose seat lies ``PRE_INSERT`` ahead (+y) of ``plug_at`` [m, bench frame], and the plug's seated pose in it.

    ``overhang`` is how far the collar's face lies ahead of the shoulder [m].
    """
    # The socket: its frame at the mouth's centre, turned like the hub so its +X runs along +y.
    t_socket = np.asarray(female["T_asset_from_mating_frame"])
    seat_x = float(female["seated_shoulder_x_in_socket_m"])
    mouth = np.array((plug_at[0], plug_at[1] + PRE_INSERT - seat_x, PEDESTAL + t_socket[2, 3]))
    socket_axes = Rotation.from_quat(hetero_scene.TURN_Z).as_matrix() @ t_socket[:3, :3]
    seat = mouth + seat_x * socket_axes[:, 0]
    contact_seat_x = -(plug_radius + overhang)
    clearance = float(female["clearance_per_side_m"])
    return {
        "tip_length_m": float(male["tip_length_m"]),
        "socket_xform": [float(mouth[0]), float(mouth[1]), PEDESTAL, *hetero_scene.TURN_Z],
        "socket_mouth_m": mouth.tolist(),
        "socket_axes": socket_axes.tolist(),
        "seat_m": seat.tolist(),
        "seat_x_m": seat_x,
        "contact_seat_x_m": contact_seat_x,
        "cavity_depth_m": float(female["cavity_depth_m"]),
        "clearance_per_side_m": clearance,
        "plug_radius_m": plug_radius,
        "tolerance": tolerances(clearance, float(male["tip_length_m"]), contact_seat_x - seat_x),
    }


def build_scene(
    asset,
    *,
    socket: str,
    plug_radius: float,
    solver: str,
    iterations: int,
    mu: float,
    grip_mu: float,
    force_limit: float,
    self_contact,
):
    vertices, tets, labels = hetero_scene.load(asset)
    mesh, connectors = hetero_scene.fixture(SOCKETS[socket])
    hub = json.loads((asset.path / "connectors.json").read_text())
    female = connectors["female_ports"][0]
    geometry = layout(vertices, tets, labels, asset.quat, female, hub["male_ends"][0], plug_radius)
    geometry["socket"] = SOCKETS[socket]
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
    scene = rss.Scene(cell=arm_gripper.CELL, physics=physics, name="cable_plug_place")
    # The block on a slab of its footprint, as a static triangle mesh: the authored triangles keep the
    # concave cavity (no hull). Its asset frame runs +X into the block and Y across it.
    block = np.load(mesh.with_suffix(".npz"))["vertices_m"]
    depth, width = float(np.ptp(block[:, 0])), float(np.ptp(block[:, 1]))
    mx, my, _ = geometry["socket_mouth_m"]
    scene.add(
        rss.slab(
            name="socket_pedestal",
            half=(width / 2.0, depth / 2.0, PEDESTAL / 2.0),
            at=(mx, my + depth / 2.0, PEDESTAL / 2.0),
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
        hetero_scene.spec(
            asset, vertices, tets, labels, name="usb_hub", pose=geometry["pose"], settings=settings, mu=mu
        )
    )
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    # From above at an angle, the whole hub (housing, bent cable, plug) and the socket in frame.
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    mouth = np.asarray(geometry["socket_mouth_m"])
    camera = scene.add(
        rss.CameraSpec(
            name="overview",
            width=960,
            height=540,
            eye=tuple(surface + mouth + (0.40, -0.30, 0.48)),
            look_at=tuple(surface + mouth + (0.05, -0.12, -0.03)),
            fov_y=35.0,
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    solver_ = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    geometry["arm"] = arm_gripper.check_arm(sim, limit=force_limit)
    geometry["friction"] = hetero_scene.set_friction(sim, solver_, settings, mu=mu, grip_mu=grip_mu)
    geometry["socket_mesh_shapes"] = socket_shapes(sim)
    geometry["plug_thin_particles"] = hetero_scene.set_part_radius(sim, obj.entity, PLUG, plug_radius)
    print(json.dumps({"geometry": geometry}), flush=True)
    return sim, obj, camera, geometry, labels


def socket_shapes(sim) -> list[int]:
    """The socket's colliding triangle-mesh shapes: one, or the cavity has been lost to a hull."""
    model = sim.model
    kinds, flags = model.shape_type.numpy(), model.shape_flags.numpy()
    return [
        s
        for s, label in enumerate(model.shape_label)
        if "socket" in str(label)
        and "pedestal" not in str(label)
        and kinds[s] == int(newton.GeoType.MESH)
        and flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES)
    ]


def motion(geometry: dict, correction=None, trims=(), phases=PHASES):
    """Hover, descend and close on the collar, lift, align, approach, push in, open, withdraw.

    Returns the hand's ``(tcp_at, quat_at, half_gap_at)`` of the time since the plug's phase began.
    ``correction`` is the hand's (shift [m], xyzw quat) at the end of "align", from the plug's
    measured pose (:func:`correct`); None keeps the nominal path and the hand pointing down.
    ``trims`` are further ``(shift [m], measured at [s])`` across the axis (:func:`lateral`).
    ``phases`` are the windows of :data:`PHASES`.
    """
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    centre = arm_gripper.tcp_for_pads(surface + np.asarray(geometry["grip_centre_m"]))
    reach = float(geometry["reach_m"])
    pre = np.asarray(geometry["seat_m"]) - PRE_INSERT * INSERT_AXIS
    move = pre - np.asarray(geometry["shoulder_settled_m"])
    push = PRE_INSERT + OVER_TRAVEL - APPROACH
    grasp = geometry.get("grasp_quat", arm_gripper.GRASP_QUAT)
    shift, quat = (np.zeros(3), grasp) if correction is None else correction
    turn = Slerp((0.0, 1.0), Rotation.from_quat([grasp, quat]))

    def tcp_at(time: float) -> np.ndarray:
        return (
            centre
            + np.array((0.0, 0.0, HOVER)) * (1.0 - smooth(time, *phases["descend"]))
            + move * smooth(time, *phases["lift"])
            + np.asarray(shift) * smooth(time, *phases["align"])
            + sum((np.asarray(d) * smooth(time, t + 0.1, t + 0.3) for d, t in trims), np.zeros(3))
            + INSERT_AXIS * APPROACH * smooth(time, *phases["approach"])
            + INSERT_AXIS * push * smooth(time, *phases["insert"])
            + np.array((0.0, 0.0, WITHDRAW_Z)) * smooth(time, *phases["withdraw"])
        )

    def quat_at(time: float) -> np.ndarray:
        return turn(smooth(time, *phases["align"])).as_quat()

    def half_gap_at(time: float) -> float:
        face = reach + usb_hub_box.GRIP_SQUEEZE - 2.0 * usb_hub_box.GRIP_SQUEEZE * smooth(time, *phases["close"])
        return face + (usb_hub_box.OPENED_HALF_GAP - face) * smooth(time, *phases["open"])

    return tcp_at, quat_at, half_gap_at


def arm_route(sim, geometry: dict, capacity: int, dt: float, correction=None, trims=()):
    """The route of :func:`motion`, planned for ``capacity`` frames."""
    control = arm_gripper.ArmControl(sim)
    tcp_at, quat_at, half_gap_at = motion(geometry, correction, trims)
    route = control.plan(quat=quat_at, capacity=capacity, dt=dt, tcp_at=tcp_at, half_gap_at=half_gap_at)
    return control, route, tcp_at


def rigid_fit(rest: np.ndarray, now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rotation and translation taking ``rest`` onto ``now`` (Kabsch)."""
    a, b = rest.mean(axis=0), now.mean(axis=0)
    u, _, vt = np.linalg.svd((rest - a).T @ (now - b))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag((1.0, 1.0, d)) @ u.T
    return rotation, b - rotation @ a


def correct(geometry: dict, frame: tuple, hand: np.ndarray, quat, max_turn: float) -> tuple[np.ndarray, np.ndarray]:
    """The hand's shift and orientation that put the measured plug ``frame`` on the pre-insert pose.

    The plug is held rigidly, so it keeps its pose relative to the hand (at ``hand``, turned by
    ``quat``): turn the hand by what takes the plug's axes onto the socket's, tilting it at most
    ``max_turn`` degrees, and shift it so the plug's shoulder lands on the socket's axis ``PRE_INSERT`` in front
    of the seat.
    """
    origin, axes = frame
    target = np.asarray(geometry["seat_m"]) - PRE_INSERT * INSERT_AXIS
    turn = Rotation.from_matrix(np.asarray(geometry["socket_axes"]) @ axes.T).as_rotvec()
    # Only the tilt (about horizontal axes) leans the fingers; a turn about z is free.
    angle = np.degrees(np.linalg.norm(turn[:2]))
    if angle > max_turn:
        turn[:2] *= max_turn / angle
    hand_axes = Rotation.from_quat(quat).as_matrix()
    turned = Rotation.from_rotvec(turn).as_matrix() @ hand_axes
    reached = target - turned @ hand_axes.T @ (origin - hand)
    return reached - hand, Rotation.from_matrix(turned).as_quat()


def lateral(geometry: dict, frame: tuple) -> np.ndarray:
    """The hand's shift that puts the measured plug's shoulder back on the socket's axis [m]."""
    socket_axes = np.asarray(geometry["socket_axes"])
    error = socket_axes.T @ (frame[0] - np.asarray(geometry["seat_m"]))
    error[0] = 0.0
    return -socket_axes @ error


def forward(geometry: dict, frame: tuple, most: float = 0.006) -> np.ndarray:
    """The hand's shift along the socket's axis that closes a plug's shortfall to its contact seat, at most ``most`` [m]."""
    socket_axes = np.asarray(geometry["socket_axes"])
    short = (
        geometry["contact_seat_x_m"]
        - geometry["seat_x_m"]
        - float(socket_axes[:, 0] @ (frame[0] - np.asarray(geometry["seat_m"])))
    )
    return socket_axes[:, 0] * float(np.clip(short, 0.0, most))


def seat_error(geometry: dict, frame: tuple) -> dict:
    """The plug's mating frame against the socket's seated pose, in the socket's frame."""
    origin, axes = frame
    socket_axes = np.asarray(geometry["socket_axes"])
    error = socket_axes.T @ (origin - np.asarray(geometry["seat_m"]))
    tip = socket_axes.T @ (origin + geometry["tip_length_m"] * axes[:, 0] - np.asarray(geometry["socket_mouth_m"]))
    angle = np.degrees(Rotation.from_matrix(socket_axes.T @ axes).magnitude())
    width, height = geometry.get("tip_profile_m", (0.012, 0.0045))
    half = 0.5 * width + geometry["clearance_per_side_m"], 0.5 * height + geometry["clearance_per_side_m"]
    return {
        "seat_error_m": [float(c) for c in error],
        "seat_position_error_m": float(np.linalg.norm(error)),
        "seat_axial_error_m": float(error[0]),
        "seat_lateral_error_m": float(np.hypot(error[1], error[2])),
        "seat_angle_error_deg": float(angle),
        "tip_inside": bool(
            0.0 < tip[0] <= geometry["cavity_depth_m"] and abs(tip[1]) < half[0] and abs(tip[2]) < half[1]
        ),
    }


def run(out: Path, asset, *, steps: int, render_every: int, config: dict, **physics) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, camera, geometry, labels = build_scene(asset, **physics)
    entity = obj.entity
    tets = entity.source_tets
    soft = hetero_scene.soft_labels(asset, labels)
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    plug = np.unique(tets[np.isin(labels, PLUG)])
    housing = np.unique(tets[labels == HOUSING])
    lay = geometry["lay"]
    geometry["laid"] = cable_scene.lay(
        sim,
        entity,
        root=surface + np.asarray(lay["root_m"]),
        direction=lay["direction"],
        bends=lay["bends"],
        keep=housing,
    )
    q0 = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
    plug_rest, housing_rest = q0[plug], q0[housing].mean(axis=0)
    shoulder_rest, axes_rest = np.asarray(geometry["shoulder_rest_m"]), np.asarray(geometry["plug_axes"])

    def plug_frame(local: np.ndarray) -> tuple:
        rotation, translation = rigid_fit(plug_rest, local[plug])
        return rotation @ shoulder_rest + translation, rotation @ axes_rest

    # The whole route, however short the run, so a smoke run checks every pose is reached.
    capacity = max(steps, round(DURATION * CONTROL_HZ))
    arm, route, tcp_at = arm_route(sim, geometry, capacity, 1.0 / CONTROL_HZ)
    if route.worst_position_error_m > 0.025:
        raise RuntimeError(f"the arm misses the path by {route.worst_position_error_m:.3f} m")
    geometry["arm_path_error_m"] = route.worst_position_error_m
    arm.start(route)
    settle, measure = round(SETTLE * CONTROL_HZ), round(MEASURE_AT * CONTROL_HZ)
    trim_frames = {round(t * CONTROL_HZ): t for t in TRIMS_AT}
    turned, trims = None, []

    def follow(frame: int):
        nonlocal arm, route, tcp_at, turned, trims
        if frame == settle:
            # Grasp the plug where the laid cable left it: the pads across the collar, behind the shoulder.
            local = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
            origin, axes = plug_frame(local)
            yaw = float(np.arctan2(-axes[0, 0], axes[1, 0]))
            along = np.array((-np.sin(yaw), np.cos(yaw)))
            geometry["grip_centre_m"][:2] = (origin[:2] - GRASP_BEHIND * along).tolist()
            geometry["shoulder_settled_m"] = origin.tolist()
            geometry["grasp_quat"] = (
                (Rotation.from_rotvec((0.0, 0.0, yaw)) * Rotation.from_quat(arm_gripper.GRASP_QUAT)).as_quat().tolist()
            )
            arm, route, tcp_at = arm_route(sim, geometry, capacity, 1.0 / CONTROL_HZ)
            geometry["settled"] = {
                "plug_shoulder_m": origin.tolist(),
                "plug_yaw_deg": float(np.degrees(yaw)),
                "arm_path_error_m": route.worst_position_error_m,
            }
            print(json.dumps({"settled": geometry["settled"]}), flush=True)
        if frame == measure or (frame in trim_frames and turned is not None):
            # Measure how the plug sits in the pads and replan: turn and shift at the first look, then
            # shift across the axis only.
            local = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
            frame_ = plug_frame(local)
            if frame == measure:
                shift, quat = correct(
                    geometry, frame_, tcp_at(frame / CONTROL_HZ) - surface, geometry["grasp_quat"], MAX_TURN
                )
                turned = (shift, quat)
            else:
                shift, quat = lateral(geometry, frame_), turned[1]
                trims = [*trims, (shift, trim_frames[frame])]
            arm, route, tcp_at = arm_route(sim, geometry, capacity, 1.0 / CONTROL_HZ, turned, trims)
            record = {
                "time_s": frame / CONTROL_HZ,
                "shift_m": shift.tolist(),
                "turn_deg": float(
                    np.degrees(
                        (Rotation.from_quat(quat) * Rotation.from_quat(arm_gripper.GRASP_QUAT).inv()).magnitude()
                    )
                ),
                "plug_error_before": seat_error(geometry, frame_),
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
        if frame == 0:
            low, high = arm_gripper.pad_bounds(sim)
            geometry["pads_first_frame_m"] = {"low": (low - surface).tolist(), "high": (high - surface).tolist()}
        centre = local[housing].mean(axis=0)
        return {
            "plug_shoulder_m": plug_frame(local)[0].tolist(),
            **seat_error(geometry, plug_frame(local)),
            "housing_bottom_m": float(local[housing, 2].min()),
            # Along the bench: the first frames' 1 mm settling drop is not a drag.
            "housing_displacement_m": float(np.linalg.norm(centre[:2] - housing_rest[:2])),
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
    # The laid cable's settle: how far the hook springs back towards straight before the arm moves.
    settled = rows[: max(1, round(SETTLE * CONTROL_HZ))]
    settle = {
        "cable_inverted_max": max(r["cable_inverted"] for r in settled),
        "plug_spring_back_m": float(np.linalg.norm(np.subtract(settled[-1]["plug_shoulder_m"], shoulder_rest))),
        "plug_spring_back_xy_m": np.subtract(settled[-1]["plug_shoulder_m"], shoulder_rest)[:2].tolist(),
        "housing_displacement_m": settled[-1]["housing_displacement_m"],
        "seconds": settled[-1]["time_s"],
    }
    print(json.dumps({"settle": settle}), flush=True)
    result = {
        "settle": settle,
        "geometry": geometry,
        "weight_n": float(entity.mass.sum() * 9.81),
        "released": max(r["finger_contacts"] for r in tail) == 0,
        **{k: final[k] for k in final if k.startswith("seat_") or k == "tip_inside"},
        "housing_bottom_final_m": final["housing_bottom_m"],
        "housing_displacement_max_m": max(r["housing_displacement_m"] for r in rows),
        "cable_stretch_p99_max": max(r["cable_stretch_p99"] for r in rows),
        "max_inverted_tetrahedra": max(r["inverted_tetrahedra"] for r in rows),
        "max_cable_inverted": max(r["cable_inverted"] for r in rows),
        "arm_path_error_m": max(route.worst_position_error_m, geometry["arm_path_error_m"]),
        "steps": steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "elapsed_s": elapsed,
        "config": config,
    }
    # Seated: let go with the plug in the cavity, in line with it, and the housing still on the bench.
    seat_axial = final["seat_axial_error_m"]
    result["success"] = bool(
        result["released"]
        and abs(seat_axial) < geometry["tolerance"]["axial_m"]
        and final["seat_lateral_error_m"] < geometry["tolerance"]["lateral_m"]
        and final["seat_angle_error_deg"] < geometry["tolerance"]["angle_deg"]
        and final["tip_inside"]
        and result["housing_bottom_final_m"] < 0.01
        # Only the soft cable's tets count: a rigid part's own tets have almost no stiffness and ride on its body.
        and result["max_cable_inverted"] == 0
    )
    # Score: how far the plug's mating frame sits from the socket's seated pose [m].
    result["score"] = final["seat_position_error_m"]
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "config")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--asset", choices=("usb_hub_gen",), default="usb_hub_gen")
    parser.add_argument("--steps", type=int, default=round(DURATION * CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default="chebyshev")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--mu", type=float, default=hetero_scene.OBJECT_MU)
    parser.add_argument("--grip-mu", type=float, default=2.0)
    parser.add_argument("--force-limit", type=float, default=30.0)
    parser.add_argument("--no-self-contact", dest="self_contact", action="store_false")
    parser.add_argument("--socket", choices=tuple(SOCKETS), default="snug")
    parser.add_argument("--plug-radius", type=float, default=PLUG_RADIUS, help="the plug's contact radius [m]")
    hetero_scene.add_lever_arguments(parser)
    args = parser.parse_args()
    asset = hetero_scene.with_levers(cable_scene.ASSETS[args.asset], args)
    run(
        args.out,
        asset,
        steps=args.steps,
        render_every=args.render_every,
        config=hetero_scene.config(args, asset, dataclasses.replace(REFERENCE, substeps=SUBSTEPS)),
        solver=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        grip_mu=args.grip_mu,
        force_limit=args.force_limit,
        self_contact=args.self_contact,
        socket=args.socket,
        plug_radius=args.plug_radius,
    )


if __name__ == "__main__":
    main()
