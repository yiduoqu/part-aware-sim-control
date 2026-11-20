# SPDX-License-Identifier: Apache-2.0
"""S1 of the spatula in the pan: the held spatula pressed onto the pan's floor.

The wooden handle is a kinematic rigid body, pitched ``--pitch`` with the blade's tip down, and
the silicone head hangs from it as FEM. The blade's leading edge is not square to the handle: it
runs back from one corner, so a spatula that is only pitched meets the floor at that corner.
``--roll level`` (the default) also turns the spatula about its own length until that edge is
level; ``--roll 0`` leaves it on the corner. Once the blade has settled under its weight the handle
comes straight down to just short of touching, then on by each of ``--depths`` in turn, holding
at each; then it lifts clear and waits. One run is the floor's force against the press depth at
one pitch.

The pan's floor is flat, so its collider is a slab with its top where the pan's inner floor is;
the pan itself is drawn over it and touches nothing. Its rim is not a collider here: how far the
spatula passes over it is read from the rim's measured height and radii.

Two gauges read the floor's force. ``floor_force_n`` is the penalty force of the stored contacts
(normal only). ``handle_*`` is what the handle's springs stopped carrying since the blade hung
free, which also has the floor's friction, along the pan.

The handle is driven whole, so its material does nothing here: this scene cannot tell the
heterogeneous model from a uniform soft one. That comparison needs only the grip zone driven.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import arm_gripper
import newton
import numpy as np
import spatula_s0 as s0
import stamp_press
import warp as wp
from clawhauser_scene import arrangement, build_sim
from loop_geometry import smooth
from mesh_quality import MeshQuality

import robosimstudio as rss
from robosimstudio.specs import AssetSpec, StaticSpec
from robosimstudio_materials.heterogeneous import load_labelled_tets

ASSET = s0.ASSET
PAN = Path(__file__).resolve().parent / "assets" / "rigid_pan"
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "spatula_pan_press" / "s1"
CELL = arm_gripper.CELL
HANDLE, HEAD = s0.HANDLE, s0.HEAD
PHYSICS = stamp_press.PHYSICS
PAN_AT = (0.0, -0.10)
"""The pan's centre on the bench [m, work-surface frame]; its handle points along +x."""
PAN_FLOOR_Z = 0.004
"""The pan's inner floor over the bench [m] (``assets/rigid_pan``)."""
PAN_FLOOR_RADIUS = 0.100
"""The floor is flat out to here [m]."""
PAN_RIM_Z = 0.038
PAN_RIM_RADII = (0.116, 0.120)
"""The rim's top over the bench and its inner and outer radius [m], away from the pan's handle."""
LAND = -0.010
"""The tip starts this far along x from the pan's centre [m]: it slides forward as it is pressed."""
START_GAP = 0.010
"""The undeformed blade's lowest node starts this far over its contact height [m]."""
CLEAR = 0.0002
"""The approach stops this short of touching [m]."""
LIFT = 0.005
"""The unload ends with the handle this far above where the blade first touched [m]."""
DEPTHS = (0.003, 0.006, 0.010, 0.015)
SETTLE_S, APPROACH_S, PRESS_S, HOLD_S, UNLOAD_S, REST_S = 2.0, 0.5, 0.8, 0.8, 1.0, 1.0
SUBSTEPS, ITERATIONS, SOLVER = 60, 160, "chebyshev"
"""What S0 read as enough to carry the head's weight (``outputs/spatula_pan_press/S0_FINDINGS.md``)."""
FORCE_CAP = 10.0
"""The handle stops going down once the floor reads this [N]."""
WINDOW_S = 0.3
"""A hold's reading is the mean over its last this long [s]."""
TIP_BAND = s0.TIP_BAND
BEND_BACK = (0.024, 0.030)
"""The blade's bend is read between its tip band and the band this far back from the tip [m]."""
EDGE_Y = (-0.027, 0.006)
"""The blade's leading edge is straight between these, across the mesh [m] (``assets/silicone_spatula``)."""
EDGE_BAND = 0.004
"""The edge's nodes are those within this of its front, one element deep [m]."""
EDGE_BIN = 0.005
"""The leading edge is cut into strips this wide across the blade [m]."""
TOUCH_TOLERANCE = stamp_press.TOUCH_TOLERANCE
RIM_CLEARANCE = 0.003
DIG_IN = 0.0005
SPRING_BACK = 0.001
TIP_STIFFNESS = s0.REFERENCE["body"]["tip_stiffness_n_per_m"]
"""The tip's stiffness on this mesh and these springs, by linear statics [N/m]."""


class DriveBody:
    """Put a kinematic body on a path given one control frame at a time, a pose per substep."""

    def __init__(self, sim, body: int, substeps: int) -> None:
        self.pipeline, self.device, self.body = sim.pipeline, sim.model.device, int(body)
        self.substeps = int(substeps)
        self.base = sim.pipeline.state_0.body_q.numpy()[self.body].astype(np.float32)
        self.poses = wp.array(np.repeat(self.base[None], self.substeps, axis=0), dtype=wp.transform, device=self.device)
        self.rates = wp.zeros(self.substeps, dtype=wp.spatial_vector, device=self.device)
        self.lengths = wp.array(np.array([self.substeps], dtype=np.int32), dtype=int, device=self.device)
        self.tick = wp.zeros(1, dtype=int, device=self.device)
        sim.pipeline.hooks.add("pre_solve", self.apply, name="drive-spatula-handle")
        sim.pipeline.hooks.add("post_solve", self.advance, name="advance-spatula-handle")
        sim.pipeline.invalidate_graph()

    def set_frame(self, drop: np.ndarray, dt: float, shift: np.ndarray | None = None) -> None:
        """The next frame's path: the handle ``drop[k]`` under its built height at substep ``k``'s start.

        ``shift[k]``, when given, is how far along x from where it was built it is then. Each has
        one more entry than there are substeps, for the last one's rate.
        """
        poses = np.repeat(self.base[None], self.substeps, axis=0)
        poses[:, 2] -= drop[:-1].astype(np.float32)
        rates = np.zeros((self.substeps, 6), dtype=np.float32)
        rates[:, 2] = -np.diff(drop) / dt
        if shift is not None:
            poses[:, 0] += shift[:-1].astype(np.float32)
            rates[:, 0] = np.diff(shift) / dt
        wp.copy(self.poses, wp.array(poses, dtype=wp.transform, device=self.device))
        wp.copy(self.rates, wp.array(rates, dtype=wp.spatial_vector, device=self.device))
        self.tick.zero_()

    def apply(self, state, dt) -> None:
        del dt
        wp.launch(
            stamp_press._hold_plate,
            dim=1,
            inputs=[
                self.tick,
                self.lengths,
                self.body,
                self.poses,
                self.rates,
                state.body_q,
                self.pipeline.state_1.body_q,
                state.body_qd,
                self.pipeline.state_1.body_qd,
            ],
            device=self.device,
        )

    def advance(self, state, dt) -> None:
        del state, dt
        wp.launch(stamp_press._advance_tick, dim=1, inputs=[self.tick, self.lengths], device=self.device)


def timeline(settle_s: float, depths: tuple[float, ...]) -> list[tuple[str, float, float]]:
    """Each phase's name, start and end [s]."""
    spans = [("settle", 0.0, settle_s), ("approach", settle_s, settle_s + APPROACH_S)]
    clock = settle_s + APPROACH_S
    for index in range(len(depths)):
        spans.append((f"press{index}", clock, clock + PRESS_S))
        spans.append((f"hold{index}", clock + PRESS_S, clock + PRESS_S + HOLD_S))
        clock += PRESS_S + HOLD_S
    spans.append(("unload", clock, clock + UNLOAD_S))
    spans.append(("rest", clock + UNLOAD_S, clock + UNLOAD_S + REST_S))
    return spans


def phase_at(spans, clock: float) -> str:
    for name, _start, end in spans:
        if clock <= end + 1.0e-9:
            return name
    return spans[-1][0]


def planned_drop(spans, depths: tuple[float, ...], touch: float, clock: float) -> float:
    """How far under its built height the handle is planned to be at ``clock`` [m]."""
    drop, last = 0.0, 0.0
    for name, start, end in spans:
        if name == "approach":
            drop += touch * smooth(clock, start, end)
        elif name.startswith("press"):
            depth = depths[int(name[5:])]
            drop += (depth - last) * smooth(clock, start, end)
            last = depth
        elif name == "unload":
            drop -= (last + LIFT) * smooth(clock, start, end)
    return drop


def tip_model_force(depth: float, normal: np.ndarray, mu: float, stiffness: float) -> float | None:
    """The floor's normal force for a sliding tip on a blade that only bends, small deflection [N].

    ``normal`` is the blade's, in the world. The tip moves ``depth / normal_z`` across the blade,
    and the floor's push with its friction has ``normal_z - mu |normal_xy|`` of itself across the
    blade. None where that is not positive: the tip cannot slide and the blade is pushed along its
    length.
    """
    up, flat = float(normal[2]), float(np.hypot(normal[0], normal[1]))
    lever = up - mu * flat
    if lever <= 0.0:
        return None
    return float(stiffness * depth / (up * lever))


def leading_edge(vertices: np.ndarray, head_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The straight leading edge's nodes, each one's strip across the blade, and the edge's slope dx/dy."""
    x, y = vertices[head_nodes, 0], vertices[head_nodes, 1]
    inside = (y >= EDGE_Y[0]) & (y <= EDGE_Y[1])
    strip = np.floor((y - EDGE_Y[0]) / EDGE_BIN).astype(int)
    front = np.array([x[inside & (strip == index)].max() for index in np.unique(strip[inside])])
    middle = EDGE_Y[0] + (np.unique(strip[inside]) + 0.5) * EDGE_BIN
    slope, offset = np.polyfit(middle, front, 1)
    on_edge = inside & (x >= offset + slope * y - EDGE_BAND)
    return head_nodes[on_edge], strip[on_edge], float(slope)


def turn_matrix(pitch: float, roll: float) -> np.ndarray:
    """Roll about the mesh's x, then pitch about y with +x turned towards -z."""
    c, s = float(np.cos(pitch)), float(np.sin(pitch))
    cr, sr = float(np.cos(roll)), float(np.sin(roll))
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))) @ np.array(
        ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr))
    )


def level_roll(vertices: np.ndarray, edge: np.ndarray, strips: np.ndarray, pitch: float) -> float:
    """The roll at which the leading edge's strips are lowest at the most even height [rad]."""
    best, best_spread = 0.0, np.inf
    for roll in np.radians(np.arange(-40.0, 40.0001, 0.05)):
        height = (vertices[edge] @ turn_matrix(pitch, roll).T)[:, 2]
        spread = float(np.ptp([height[strips == index].min() for index in np.unique(strips)]))
        if spread < best_spread:
            best, best_spread = float(roll), spread
    return best


def build_scene(
    *,
    pitch_deg: float,
    roll_deg: float | None,
    solver_name: str,
    iterations: int,
    substeps: int,
    young_scale: float,
    attach_stiffness: float,
    mu: float | None,
    land: float = LAND,
    name: str = "spatula_pan_press",
):
    vertices, tets, labels, elastic, _density = load_labelled_tets(
        ASSET / "final_mesh.mesh", ASSET / "heterogeneous_params.npz"
    )
    head_young = float(elastic[labels == HEAD, 0][0])
    overrides = ((HEAD, "youngs_modulus_pa", head_young * young_scale),) if young_scale != 1.0 else ()
    settings = dataclasses.replace(PHYSICS, substeps=substeps, part_overrides=overrides)
    physics = arrangement(
        settings, claw="claw", coarse=False, solver_name=solver_name, iterations=iterations, self_contact=False
    )
    scene = rss.Scene(cell=CELL, physics=physics, name=name)
    radius = settings.particle_radius
    head_nodes = np.unique(tets[labels == HEAD])
    far = float(vertices[head_nodes, 0].max())
    tips = head_nodes[vertices[head_nodes, 0] >= far - TIP_BAND]
    back = head_nodes[(vertices[head_nodes, 0] >= far - BEND_BACK[1]) & (vertices[head_nodes, 0] <= far - BEND_BACK[0])]
    edge, edge_strips, edge_slope = leading_edge(vertices, head_nodes)
    pitch = np.radians(pitch_deg)
    roll = level_roll(vertices, edge, edge_strips, pitch) if roll_deg is None else float(np.radians(roll_deg))
    turn = turn_matrix(pitch, roll)
    turned = vertices @ turn.T
    lowest = turned[edge, 2]
    edge_spread = float(np.ptp([lowest[edge_strips == index].min() for index in np.unique(edge_strips)]))
    tip = turned[tips].mean(axis=0)
    pose = (
        PAN_AT[0] + land - float(tip[0]),
        PAN_AT[1] - float(tip[1]),
        PAN_FLOOR_Z + radius + START_GAP - float(turned[head_nodes, 2].min()),
    )
    # The collider: a slab under the floor's flat part. The pan is drawn over it.
    scene.add(
        rss.slab(
            name="pan_floor",
            half=(PAN_FLOOR_RADIUS, PAN_FLOOR_RADIUS, PAN_FLOOR_Z / 2.0),
            at=(PAN_AT[0], PAN_AT[1], PAN_FLOOR_Z / 2.0),
            visible=False,
        )
    )
    scene.add(
        StaticSpec(
            name="pan",
            asset=AssetSpec(path=str(PAN / "physics" / "collision_mesh.obj")),
            xform=(PAN_AT[0], PAN_AT[1], 0.0, 0.0, 0.0, 0.0, 1.0),
            collide=False,
            color=(0.20, 0.21, 0.23),
        )
    )
    obj = scene.add(
        s0.HeldSpec(
            rigid_parts=(HANDLE,),
            attach_stiffness=attach_stiffness,
            attach_damping=stamp_press.ATTACH_DAMPING,
            name="silicone_spatula",
            mesh_path=str(ASSET / "final_mesh.mesh"),
            params_path=str(ASSET / "heterogeneous_params.npz"),
            materials_path=str(ASSET / "inferred_params.json"),
            pose=pose,
            # Pitch after roll, as (x, y, z, w).
            quat=(
                float(np.cos(pitch / 2.0) * np.sin(roll / 2.0)),
                float(np.sin(pitch / 2.0) * np.cos(roll / 2.0)),
                float(-np.sin(pitch / 2.0) * np.sin(roll / 2.0)),
                float(np.cos(pitch / 2.0) * np.cos(roll / 2.0)),
            ),
            contact_ke=settings.contact_ke,
            contact_kd=settings.contact_kd,
            contact_mu=settings.soft_mu,
            particle_radius=radius,
            damping_ratio=settings.damping_ratio,
            part_overrides=overrides,
        )
    )
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    floor = surface + np.array((PAN_AT[0], PAN_AT[1], PAN_FLOOR_Z))
    for camera in (
        rss.CameraSpec(
            name="side",
            width=960,
            height=540,
            # Over the near rim, low enough that the blade's bend reads against the floor.
            eye=tuple(float(v) for v in np.add(floor, (-0.03, 0.33, 0.14))),
            look_at=tuple(float(v) for v in np.add(floor, (-0.03, 0.0, 0.01))),
            fov_y=30.0,
        ),
        rss.CameraSpec(
            name="overview",
            width=960,
            height=540,
            eye=tuple(float(v) for v in np.add(floor, (0.22, -0.34, 0.26))),
            look_at=tuple(float(v) for v in np.add(floor, (-0.03, 0.0, 0.02))),
            fov_y=30.0,
        ),
    ):
        scene.add(camera)
    sim = build_sim(scene, rigid_loop=False)
    entity = obj.entity
    report = stamp_press.check_stamp(sim, entity, tets, settings, mu=mu)
    built = sim.pipeline.state_0.particle_q.numpy()[np.asarray(entity.particles)].astype(np.float64) - surface
    if np.abs(built - (turned + np.asarray(pose))).max() > 1.0e-4:
        raise RuntimeError("the spatula was not built at the pitch and place it was given")
    floor_shapes = [index for index, label in enumerate(sim.model.shape_label) if "pan_floor" in str(label)]
    if not floor_shapes:
        raise RuntimeError(f"no pan_floor shape among {[str(label) for label in sim.model.shape_label]}")
    young = entity.elastic[:, 0]
    faces = stamp_press.boundary_faces(tets)
    report.update(
        {
            "pitch_deg": pitch_deg,
            "roll_deg": float(np.degrees(roll)),
            "roll": "level" if roll_deg is None else "given",
            "edge_nodes": int(len(edge)),
            "edge_strips": int(len(np.unique(edge_strips))),
            "edge_slope": edge_slope,
            # How uneven the edge's strips are in height as built: what the press has to flatten.
            "edge_height_spread_m": edge_spread,
            "head_young_pa": float(young[labels == HEAD].max()),
            "handle_young_pa": float(young[labels == HANDLE].max()),
            "particle_weight_n": float(entity.mass.sum() * s0.GRAVITY),
            "pose_m": [float(v) for v in pose],
            "tip_nodes": int(len(tips)),
            "floor_shapes": floor_shapes,
            "pan_at_m": list(PAN_AT),
        }
    )
    print(json.dumps({"geometry": report}), flush=True)
    geometry = {
        "tets": tets,
        "labels": labels,
        "tips": tips,
        "back": back,
        "head_nodes": head_nodes,
        "head_faces": faces[np.isin(faces, head_nodes).all(axis=1)],
        "edge": edge,
        "edge_strips": edge_strips,
        "normal": turn[:, 2].copy(),
        "along": turn[:, 0].copy(),
        "floor_shapes": np.asarray(floor_shapes),
        "built": built,
    }
    return sim, obj, geometry, report, settings


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def run(
    out: Path,
    *,
    pitch_deg: float,
    roll_deg: float | None = None,
    depths: tuple[float, ...] = DEPTHS,
    solver_name: str = SOLVER,
    iterations: int = ITERATIONS,
    substeps: int = SUBSTEPS,
    young_scale: float = 1.0,
    attach_stiffness: float = 1.0e4,
    mu: float | None = None,
    force_cap: float = FORCE_CAP,
    settle_s: float = SETTLE_S,
    render_every: int = 0,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, geometry, report, settings = build_scene(
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        solver_name=solver_name,
        iterations=iterations,
        substeps=substeps,
        young_scale=young_scale,
        attach_stiffness=attach_stiffness,
        mu=mu,
    )
    entity = obj.entity
    particles = np.asarray(entity.particles)
    tets, labels = geometry["tets"], geometry["labels"]
    tips, back, head_nodes, head_faces = (
        geometry["tips"],
        geometry["back"],
        geometry["head_nodes"],
        geometry["head_faces"],
    )
    normal, along = geometry["normal"], geometry["along"]
    edge, strips = geometry["edge"], geometry["edge_strips"]
    hz, radius, ke = settings.control_hz, settings.particle_radius, settings.contact_ke
    dt = 1.0 / (hz * settings.substeps)
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    soft_tets = labels == HEAD
    quality = MeshQuality(sim, entity)
    contacts = sim.pipeline.contacts
    part = entity.rigid_parts[0]
    drive = DriveBody(sim, part.body, settings.substeps)
    start = sim.pipeline.state_0.particle_q.numpy()[particles].astype(np.float64) - surface
    spans = timeline(settle_s, depths)
    duration = spans[-1][2]
    settle_frames, steps = round(settle_s * hz), round(duration * hz)
    contact_z = PAN_FLOOR_Z + radius
    centre = np.array((PAN_AT[0], PAN_AT[1]))
    rest_bend = start[tips].mean(axis=0) - start[back].mean(axis=0)
    rest_bend_angle = float(np.arctan2(rest_bend @ normal, rest_bend @ along))
    rows, local = [], start
    touch, limit, capped_at = None, None, None

    videos, renderer, video_error = None, None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet
        from robosimstudio.solvers.render import FALLBACK

        backend, _why = sim.platform.renderer_for("")
        renderer = sim.make_renderer(backend, cameras=list(sim.cameras), channels=["rgb"])
        if backend != FALLBACK:
            renderer.render(sim.state)
        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // hz))
        videos.__enter__()

    def drop_at(clock: float) -> float:
        if touch is None:
            return 0.0
        drop = planned_drop(spans, depths, touch, clock)
        return drop if limit is None else min(drop, limit)

    started = time.perf_counter()
    try:
        for frame in range(steps):
            if frame == settle_frames:
                # From where the blade hangs now.
                touch = float(local[head_nodes, 2].min()) - contact_z - CLEAR
                report["touch_drop_m"] = touch
            drops = np.array([drop_at(frame / hz + k * dt) for k in range(settings.substeps + 1)])
            drive.set_frame(drops, dt)
            sim.run(1).sync()
            state = sim.pipeline.state_0
            q, body_q = state.particle_q.numpy(), state.body_q.numpy()
            owned = q[particles].astype(np.float64)
            local = owned - surface
            clock = (frame + 1) / hz
            phase = phase_at(spans, clock)
            drop = float(drops[-1])
            carried = start - np.array((0.0, 0.0, drop))
            shapes, forces = s0.contact_forces(q, body_q, sim.model, contacts, ke)
            on_floor = np.isin(shapes, geometry["floor_shapes"])
            floor_force = float(forces[on_floor, 2].sum()) if on_floor.any() else 0.0
            # A stored contact is a pair within the collision margin; one that pushes is in the floor.
            pushing = np.linalg.norm(forces, axis=1) > 0.0
            depth = np.linalg.norm(forces[on_floor], axis=1) / ke if on_floor.any() else np.zeros(0)
            touching = np.zeros(len(local), dtype=bool)
            touching[head_nodes] = local[head_nodes, 2] <= contact_z + TOUCH_TOLERANCE
            touched = head_nodes[touching[head_nodes]]
            down = touching[head_faces].all(axis=1)
            corners = local[head_faces[down]]
            area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
            moved = local[tips].mean(axis=0) - carried[tips].mean(axis=0)
            bend = local[tips].mean(axis=0) - local[back].mean(axis=0)
            over_rim = np.hypot(*(local[:, :2] - centre).T)
            over_rim = (over_rim >= PAN_RIM_RADII[0] - radius) & (over_rim <= PAN_RIM_RADII[1] + radius)
            pose = body_q[part.body]
            stretch = s0.rotate(pose, part.body_point) - owned[part.nodes]
            pull = entity.spec.attach_stiffness * stretch.sum(axis=0)
            inverted = np.linalg.det(owned[tets[:, 1:]] - owned[tets[:, :1]]) <= 0
            row = {
                "frame": frame + 1,
                "time_s": clock,
                "phase": phase,
                "finite": bool(np.isfinite(owned).all() and np.isfinite(body_q).all()),
                "handle_drop_m": drop,
                "press_m": None if touch is None else drop - touch,
                "handle_z_error_m": float(pose[2] - (drive.base[2] - drop)),
                "floor_force_n": floor_force,
                "floor_contacts": int(np.count_nonzero(on_floor & pushing)),
                "other_contacts": int(np.count_nonzero(~on_floor & pushing)),
                "floor_penetration_m": float(depth.max()) if len(depth) else 0.0,
                # What the springs pull the handle's nodes with [N, world].
                "handle_pull_x_n": float(pull[0]),
                "handle_pull_z_n": float(pull[2]),
                "seam_stretch_m": float(np.linalg.norm(stretch[part.seam], axis=1).max()),
                "tip_across_m": float(moved @ normal),
                "tip_along_m": float(moved @ along),
                "tip_forward_m": float(moved[0]),
                "tip_up_m": float(moved[2]),
                "tip_x_m": float(local[tips, 0].mean() - PAN_AT[0]),
                # Towards the blade's upper face is positive.
                "blade_bend_deg": float(np.degrees(np.arctan2(bend @ normal, bend @ along) - rest_bend_angle)),
                "edge_contact_fraction": float(len(np.unique(strips[touching[edge]])) / len(np.unique(strips))),
                "edge_contact_nodes": int(np.count_nonzero(touching[edge])),
                "contact_nodes": int(len(touched)),
                "contact_length_m": float(np.ptp(local[touched, 0])) if len(touched) else 0.0,
                "contact_from_tip_m": float(local[tips, 0].mean() - local[touched, 0].min()) if len(touched) else 0.0,
                "contact_area_m2": float(area.sum()),
                "rim_clearance_m": float(local[over_rim, 2].min() - radius - PAN_RIM_Z) if over_rim.any() else None,
                "inverted_soft_tets": int(np.count_nonzero(inverted & soft_tets)),
                "speed_m_s": float(np.linalg.norm(local - (rows[-1]["_local"] if rows else start), axis=1).max() * hz),
                "capped": limit is not None,
            }
            row["_local"] = local
            if rows:
                del rows[-1]["_local"]
            rows.append(row)
            if limit is None and floor_force > force_cap and phase.startswith(("approach", "press", "hold")):
                limit, capped_at = drop, clock
            if videos is not None and frame % render_every == 0:
                sim.sync()
                videos.append_frames(renderer.render(sim.state))
            if frame % 30 == 0 or frame == steps - 1:
                print(json.dumps({k: v for k, v in row.items() if k != "_local"}), flush=True)
            if not row["finite"]:
                break
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
        rows[-1].pop("_local", None)
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")

    window = round(WINDOW_S * hz)
    by_phase = {name: [row for row in rows if row["phase"] == name] for name, _a, _b in spans}
    free = by_phase["settle"][-window:]
    result = {
        "complete": len(rows) == steps,
        "finite": all(row["finite"] for row in rows),
        "max_inverted_soft_tets": max(row["inverted_soft_tets"] for row in rows),
        "max_floor_force_n": max(row["floor_force_n"] for row in rows),
        "max_floor_penetration_m": max(row["floor_penetration_m"] for row in rows),
        "max_seam_stretch_m": max(row["seam_stretch_m"] for row in rows),
        "max_handle_z_error_m": max(abs(row["handle_z_error_m"]) for row in rows),
        "other_contacts": max(row["other_contacts"] for row in rows),
        "capped_at_s": capped_at,
        "touch_drop_m": report.get("touch_drop_m"),
    }
    soft_mu = report["friction"]["global_soft_contact_mu"]
    if free:
        result["free"] = {
            "tip_sag_across_m": -mean(free, "tip_across_m"),
            "tip_sag_swing_m": float(np.ptp([row["tip_across_m"] for row in free])),
            # Linear statics has the level blade sag this; pitched, its weight across the blade is cos of it.
            "tip_sag_reference_m": s0.REFERENCE["body"]["sag_m"] / young_scale * float(normal[2]),
            "handle_pull_z_n": mean(free, "handle_pull_z_n"),
            "handle_pull_x_n": mean(free, "handle_pull_x_n"),
            "hanging_weight_n": report["particle_weight_n"],
            "floor_contacts": max(row["floor_contacts"] for row in by_phase["settle"]),
        }
        result["free"]["sag_over_reference"] = (
            result["free"]["tip_sag_across_m"] / result["free"]["tip_sag_reference_m"]
        )
    holds = []
    for index, depth in enumerate(depths):
        held = by_phase[f"hold{index}"][-window:]
        if not held or not free:
            continue
        # Since the blade hung free, the springs carry this much less upward and push this much along x.
        lift = result["free"]["handle_pull_z_n"] - mean(held, "handle_pull_z_n")
        shove = mean(held, "handle_pull_x_n") - result["free"]["handle_pull_x_n"]
        stage = {
            "depth_m": depth,
            "press_m": mean(held, "press_m"),
            "floor_force_n": mean(held, "floor_force_n"),
            "floor_force_swing_n": float(np.ptp([row["floor_force_n"] for row in held])),
            "handle_floor_force_n": lift,
            "handle_floor_friction_n": shove,
            "friction_over_normal": shove / lift if lift > 0 else None,
            "tip_model_force_n": tip_model_force(depth, normal, soft_mu, TIP_STIFFNESS * young_scale),
            "edge_contact_fraction": mean(held, "edge_contact_fraction"),
            "edge_contact_nodes": held[-1]["edge_contact_nodes"],
            "contact_nodes": held[-1]["contact_nodes"],
            "contact_length_m": mean(held, "contact_length_m"),
            "contact_from_tip_m": mean(held, "contact_from_tip_m"),
            "contact_area_m2": mean(held, "contact_area_m2"),
            "tip_across_m": mean(held, "tip_across_m") - mean(free, "tip_across_m"),
            "tip_forward_m": mean(held, "tip_forward_m") - mean(free, "tip_forward_m"),
            "blade_bend_deg": mean(held, "blade_bend_deg") - mean(free, "blade_bend_deg"),
            "floor_penetration_m": max(row["floor_penetration_m"] for row in held),
            "rim_clearance_m": min(row["rim_clearance_m"] for row in held),
            "speed_m_s": max(row["speed_m_s"] for row in held),
            "capped": held[-1]["capped"],
        }
        holds.append(stage)
    result["holds"] = holds
    rested = by_phase["rest"][-window:]
    if rested and free:
        result["rest"] = {
            "tip_return_m": abs(mean(rested, "tip_across_m") - mean(free, "tip_across_m")),
            "floor_contacts": rested[-1]["floor_contacts"],
            "floor_force_n": mean(rested, "floor_force_n"),
        }
    result["checks"] = {
        "ran": result["complete"] and result["finite"],
        "hung_free_before": bool(free) and result["free"]["floor_contacts"] == 0,
        "pressed_every_depth": len(holds) == len(depths) and all(stage["floor_force_n"] > 0 for stage in holds),
        "under_force_cap": capped_at is None,
        "edge_on_floor": bool(holds) and all(stage["edge_contact_fraction"] >= 0.9 for stage in holds),
        "handle_clears_rim": bool(holds) and min(stage["rim_clearance_m"] for stage in holds) >= RIM_CLEARANCE,
        "no_dig_in": result["max_floor_penetration_m"] <= DIG_IN,
        "soft_mesh_intact": result["max_inverted_soft_tets"] == 0,
        "springs_back": "rest" in result and result["rest"]["tip_return_m"] <= SPRING_BACK,
        "only_the_floor_touched": result["other_contacts"] == 0,
    }
    result["passed"] = all(result["checks"].values())
    result["mesh_quality"] = quality.read()
    result["config"] = {
        "pitch_deg": pitch_deg,
        "roll_deg": report["roll_deg"],
        "roll": report["roll"],
        "depths_m": list(depths),
        "solver": type(arm_gripper.vbd_solver(sim.pipeline.program.solver)).__name__,
        "solver_name": solver_name,
        "iterations": iterations,
        "substeps": substeps,
        "control_hz": hz,
        "young_scale": young_scale,
        "attach_stiffness_n_m": attach_stiffness,
        "soft_mu": soft_mu,
        "shape_mu": report["friction"]["shape_mu"],
        "force_cap_n": force_cap,
        "settle_s": settle_s,
        "newton_version": newton.__version__,
    }
    result["geometry"] = report
    result["video_error"] = video_error
    result["simulated_s"] = len(rows) / hz
    result["elapsed_s"] = elapsed
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "geometry"}), flush=True)
    brief = {
        "pitch_deg": pitch_deg,
        "roll_deg": round(report["roll_deg"], 2),
        "substeps": substeps,
        "iterations": iterations,
        "solver_name": solver_name,
        "passed": result["passed"],
        "failed": [name for name, ok in result["checks"].items() if not ok],
        "floor_force_n": [round(stage["floor_force_n"], 4) for stage in holds],
        "edge_contact_fraction": [round(stage["edge_contact_fraction"], 3) for stage in holds],
        "elapsed_s": round(elapsed, 1),
    }
    print("S1_RESULT " + json.dumps(brief), flush=True)
    sim.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pitch", type=float, default=30.0, help="the handle's pitch, tip down [deg]")
    parser.add_argument("--roll", type=str, default="level", help="about the spatula's length [deg], or 'level'")
    parser.add_argument("--depths", type=str, default=",".join(str(d) for d in DEPTHS), help="press depths [m]")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default=SOLVER)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--substeps", type=int, default=SUBSTEPS)
    parser.add_argument("--young-scale", type=float, default=1.0, help="multiplies the head's Young's modulus")
    parser.add_argument("--attach-stiffness", type=float, default=1.0e4, help="each node-body spring [N/m]")
    parser.add_argument("--mu", type=float, default=None, help="the spatula's friction coefficient")
    parser.add_argument("--force-cap", type=float, default=FORCE_CAP, help="stop going down at this floor force [N]")
    parser.add_argument("--settle", type=float, default=SETTLE_S, help="how long the blade hangs first [s]")
    parser.add_argument("--render-every", type=int, default=0)
    args = parser.parse_args()
    depths = tuple(float(value) for value in args.depths.split(","))
    if not depths or any(b <= a for a, b in zip((0.0, *depths), depths, strict=False)):
        raise SystemExit("--depths must be positive and increasing")
    roll = None if args.roll == "level" else float(args.roll)
    name = f"pitch{args.pitch:g}_{'level' if roll is None else f'roll{roll:g}'}_{args.solver}_{args.substeps}x{args.iterations}"
    run(
        args.out if args.out is not None else OUTPUT / name,
        pitch_deg=args.pitch,
        roll_deg=roll,
        depths=depths,
        solver_name=args.solver,
        iterations=args.iterations,
        substeps=args.substeps,
        young_scale=args.young_scale,
        attach_stiffness=args.attach_stiffness,
        mu=args.mu,
        force_cap=args.force_cap,
        settle_s=args.settle,
        render_every=args.render_every,
    )


if __name__ == "__main__":
    main()
