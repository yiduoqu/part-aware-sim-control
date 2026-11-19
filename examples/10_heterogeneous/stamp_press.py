# SPDX-License-Identifier: Apache-2.0
"""Press the retractable stamp onto a sheet of paper and read the mark its die leaves.

The stamp (``retractable_stamp_spring``) has a hard shell: a cap whose skirt slides over the
sleeve, a shaft in a guide bore, and a steel helix between the cap and a seat inside the sleeve.
It stands on its open sleeve rim on a paper pad. Its cap and plunger are one rigid body and its
sleeve another (:mod:`rigid_part_body`); the foam cushion and the rubber die are FEM.

The helix is in the mesh but this solver cannot run it (``HELIX_YOUNG_PA``), so the spring is
two things here. What a spring does -- hold the cap up, give under a push, bring it back, keep
it in line -- is :class:`SpringGuide`, forces between the two rigid groups. What a spring looks
like is the meshed helix, drawn between its two seats and carrying nothing (:class:`StampSpec`).

A kinematic plate comes down on the cap, pushes it down until the die is ``--overpress`` into
the paper, holds, and lifts off again. Nothing is attached to the plate: the cap goes down
because the plate is on it, the die reaches the paper because the spring gives, and it comes
back because the spring pushes.

The run measures what the asset's static checks could not: whether the stamp stands, how
far the cap travels before the die touches, how much of the raised "1" and its border
meets the paper, whether the core returns when the plate leaves, and whether the moving
parts stay out of each other. Every die face whose three nodes are on the paper inks it;
the union over the run is ``imprint.png``.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import arm_gripper
import newton
import numpy as np
import rigid_part_body
import warp as wp
from clawhauser_scene import PhysicsSettings, Trajectory, arrangement, build_sim, set_contact
from loop_geometry import smooth
from mesh_quality import MeshQuality

import robosimstudio as rss
from robosimstudio_materials.heterogeneous import load_labelled_tets

ASSET_ROOT = Path(__file__).resolve().parent / "assets"
ASSET = "retractable_stamp_spring"
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "stamp_press"
CELL = arm_gripper.CELL
CAP, PLUNGER, SLEEVE, RETURN, FOAM, DIE = range(6)
"""The stamp's part labels. Cap + plunger and the sleeve are the two rigid groups; ``RETURN`` is the helix."""
SOFT = (RETURN, FOAM, DIE)

PHYSICS = PhysicsSettings(
    control_hz=60,
    substeps=20,
    contact_ke=1.0e4,
    contact_kd=10.0,
    soft_mu=0.6,
    # The die is recessed 3 mm and its relief is 3 mm wide: at the Clawhauser's 3 mm radius
    # the die would touch the paper at rest.
    particle_radius=0.0005,
    shape_mu=0.6,
    damping_ratio=5.0e-4,
)
"""The Clawhauser reference's rate, penalty and damping, at a radius this stamp's features need."""
STAMP_AT = (0.0, -0.12)
"""Where the stamp stands [m, bench frame]: where the hub scene's grasp is, in the arm's reach."""
PAPER_HALF = (0.05, 0.05, 0.0005)
"""A 100 mm square pad, 1 mm thick [m, half extents]."""
PAPER_COLOR = (0.97, 0.97, 0.94)
INK_RGB = (150, 28, 40)
PLACE_GAP = 0.0005
"""How far over its rest height the stamp starts [m]."""
PRESSER_HALF = (0.018, 0.018, 0.004)
APPROACH = 0.003
"""The plate waits this far over the cap [m]."""
OVERPRESS = 0.0005
"""How far the plate goes on once the die would touch [m]: into the foam and the rubber. Its whole
travel is the die's gap over the paper where the stamp settled, plus this."""
HELIX_YOUNG_PA = 3.0e5
"""The helix's Young's modulus in this scene [Pa].

Its source is steel, 2e11: the solve diverges there and down to 1e8 (the cap is on its side 0.3 s
in), and where it is stable, 3e7, a 1.2 mm wire carries 1 mN/mm, not the cap. Nor does it hold
itself up: a sweep passes a pull one node along the wire, the wire is 330 nodes long, and at 80
sweeps a substep its middle never learns that its ends are held -- it lay on the paper within 0.6 s
at steel's density and at 1/400 of it, and tied node by node to the two groups at 0.05 N/m it still
hung 5 to 13 mm low. So the helix is drawn, not solved (:class:`StampSpec`), and the force a spring
gives comes from :class:`SpringGuide`. Its tets still pull on the seats they are bonded to wherever a
drawn node is a few micrometres off: at 2e7 that kicked the stamp 1.1 degrees over as it was set
down and added 0.14 N to what the spring carries. At the foam's 3e5 it is centinewtons.
"""
INSIDE_DEPTH = 2.0e-5
"""A node is inside a rigid group once it is this deep in it [m]. As the stamp is set down the drawn
helix runs 3 um ahead of the cap it is bonded to, for one frame; that is not one part in another."""
SOLVER = "vbd"
"""The particle solver. Under the other scenes' Chebyshev acceleration the free-standing stamp turned
about its own axis at 4 degrees a second, with nothing to turn it (0.08 degrees in all under plain VBD):
the mark came out 6 degrees round. It stopped only while the plate was on the cap."""
HELIX_LAND = 0.0013
"""How far in from each seat the helix moves with the seat alone [m]: its flat end turns, one 1.2 mm
wire thick, are bonded there."""


@dataclass(frozen=True)
class SpringGuide:
    """A stamp's coil spring and sliding fit, as forces between its cap group and its sleeve.

    Along the sleeve's axis: a spring of ``rate`` holding the cap up against a top stop with
    ``preload``, and a bottom stop ``travel`` below it, where the cap's shoulder meets the shroud.
    Across it: a ``guide`` spring and an ``align`` torque, which are the shaft in its bore and the
    skirt on its shroud. The meshed guides cannot do this themselves: without self-contact the
    stamp's parts do not touch each other.
    """

    rate: float = 1000.0
    """The spring's rate [N/m]: 1 N/mm, a pocket stamp's."""
    preload: float = 1.0
    """Its force at rest, on the top stop [N]: over three times what it carries."""
    travel: float = 0.0045
    """From the top stop to the bottom one [m]: the asset's ``mechanical_stop_mm``."""
    stop: float = 5.0e3
    """Either stop's stiffness [N/m]."""
    damping: float = 2.0
    """Axial damping [N*s/m]."""
    guide_height: float = 0.027
    """Where the guide acts, up the stamp's axis from its rim [m]: the middle of the bore the shaft runs in."""
    guide: float = 3.0e3
    """Lateral stiffness there [N/m]. The forces are explicit and the sleeve weighs 6 g: at 5e3, acting
    at the cap's centre of mass 29 mm up, the sleeve was at 61 degrees inside the first frame."""
    guide_damping: float = 1.0
    align: float = 0.5
    """Stiffness against tilt and twist [N*m/rad]."""
    align_damping: float = 4.0e-4


ATTACH_DAMPING = 0.0
"""Damping of the nodes tied to each rigid body [N*s/m]. At the cable scenes' 100 the cap of an earlier
stamp, held up by a foam collar, sank at g*dt and never came back: its die was on the paper in 0.27 s."""
PHASES = {
    "settle": (0.0, 0.6),
    "press": (0.6, 2.4),
    "hold": (2.4, 3.2),
    "release": (3.2, 4.4),
    "rest": (4.4, 5.4),
}
"""Phase windows [s]. The plate's path is planned at the end of ``settle``, from where the cap came to rest."""
DURATION = PHASES["rest"][1]
TOUCH_TOLERANCE = 5.0e-5
"""A die node is on the paper when it is within this of its contact distance [m]."""
INK_PIXEL = 5.0e-5
INK_HALF = 0.012
"""The imprint image: this pixel size, this far each side of the stamp's axis [m]."""


@wp.kernel
def _hold_plate(
    tick: wp.array[int],
    lengths: wp.array[int],
    body: int,
    poses: wp.array[wp.transform],
    rates: wp.array[wp.spatial_vector],
    q0: wp.array[wp.transform],
    q1: wp.array[wp.transform],
    qd0: wp.array[wp.spatial_vector],
    qd1: wp.array[wp.spatial_vector],
):
    frame = wp.min(tick[0], lengths[0] - 1)
    q0[body] = poses[frame]
    q1[body] = poses[frame]
    qd0[body] = rates[frame]
    qd1[body] = rates[frame]


@wp.kernel
def _advance_tick(tick: wp.array[int], lengths: wp.array[int]):
    tick[0] = wp.min(tick[0] + 1, lengths[0] - 1)


@wp.kernel
def _spring_guide(
    core: int,
    sleeve: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    rest: float,
    up_core: float,
    up_sleeve: float,
    rate: float,
    preload: float,
    travel: float,
    stop: float,
    damping: float,
    guide: float,
    guide_damping: float,
    align: float,
    align_damping: float,
    inv_mass: wp.array[float],
    inv_inertia: wp.array[wp.mat33],
    centre: wp.array[wp.vec3],
    linear: wp.array[wp.vec3],
    angular: wp.array[wp.vec3],
    reading: wp.array[float],
):
    """The spring, stops and guide between the two groups, as each group's rigid acceleration.

    Every force acts at the guide point: ``up_core`` up the cap group's axis from its centre of
    mass and ``up_sleeve`` up the sleeve's from its own, the same place when the two are in line.
    """
    q_core = wp.transform_get_rotation(body_q[core])
    q_sleeve = wp.transform_get_rotation(body_q[sleeve])
    p_core = wp.transform_point(body_q[core], body_com[core])
    p_sleeve = wp.transform_point(body_q[sleeve], body_com[sleeve])
    axis = wp.quat_rotate(q_sleeve, wp.vec3(0.0, 0.0, 1.0))
    arm_core = wp.quat_rotate(q_core, wp.vec3(0.0, 0.0, up_core))
    arm_sleeve = up_sleeve * axis
    w_core = wp.spatial_bottom(body_qd[core])
    w_sleeve = wp.spatial_bottom(body_qd[sleeve])
    v = (wp.spatial_top(body_qd[core]) + wp.cross(w_core, arm_core)) - (
        wp.spatial_top(body_qd[sleeve]) + wp.cross(w_sleeve, arm_sleeve)
    )
    rate_along = wp.dot(v, axis)
    x = wp.dot(p_core - p_sleeve, axis) - rest  # the cap over its rest height on the sleeve: negative as it is pressed
    push = preload - rate * x - damping * rate_along
    if x > 0.0:
        push -= stop * x
    if x < -travel:
        push += stop * (-travel - x)
    apart = (p_core + arm_core) - (p_sleeve + arm_sleeve)
    force = push * axis - guide * (apart - wp.dot(apart, axis) * axis) - guide_damping * (v - rate_along * axis)
    error = q_core * wp.quat_inverse(q_sleeve)
    if error[3] < 0.0:
        error = wp.quat(-error[0], -error[1], -error[2], -error[3])
    turn_axis = wp.vec3(error[0], error[1], error[2])
    sine = wp.length(turn_axis)
    turn = wp.vec3(0.0)
    if sine > 1.0e-8:
        turn = turn_axis * (2.0 * wp.atan2(sine, error[3]) / sine)
    torque = -align * turn - align_damping * (w_core - w_sleeve)
    r_core = wp.quat_to_matrix(q_core)
    r_sleeve = wp.quat_to_matrix(q_sleeve)
    centre[0] = p_core
    linear[0] = force * inv_mass[0]
    angular[0] = r_core * (inv_inertia[0] * (wp.transpose(r_core) * (torque + wp.cross(arm_core, force))))
    centre[1] = p_sleeve
    linear[1] = -force * inv_mass[1]
    angular[1] = r_sleeve * (inv_inertia[1] * (wp.transpose(r_sleeve) * (-torque - wp.cross(arm_sleeve, force))))
    reading[0] = x
    reading[1] = push


@wp.kernel
def _accelerate_nodes(
    nodes: wp.array[int],
    group: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    centre: wp.array[wp.vec3],
    linear: wp.array[wp.vec3],
    angular: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
):
    particle = nodes[wp.tid()]
    g = group[wp.tid()]
    acceleration = linear[g] + wp.cross(angular[g], particle_q[particle] - centre[g])
    particle_f[particle] = particle_f[particle] + particle_mass[particle] * acceleration


@wp.kernel
def _accelerate_bodies(
    bodies: wp.array[int],
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    linear: wp.array[wp.vec3],
    angular: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    g = wp.tid()
    body = bodies[g]
    r = wp.quat_to_matrix(wp.transform_get_rotation(body_q[body]))
    torque = r * (body_inertia[body] * (wp.transpose(r) * angular[g]))
    wp.atomic_add(body_f, body, wp.spatial_vector(body_mass[body] * linear[g], torque))


@wp.func
def _pose_after(pose: wp.transform, rate: wp.spatial_vector, dt: float) -> wp.transform:
    """``pose`` carried on by ``rate`` for ``dt``: where the body is headed by the end of the substep."""
    q = wp.transform_get_rotation(pose)
    w = wp.spatial_bottom(rate)
    turned = wp.normalize(q + 0.5 * dt * (wp.quat(w[0], w[1], w[2], 0.0) * q))
    return wp.transform(wp.transform_get_translation(pose) + wp.spatial_top(rate) * dt, turned)


@wp.kernel
def _pose_carried(
    nodes: wp.array[int],
    along: wp.array[float],
    low: wp.array[wp.vec3],
    high: wp.array[wp.vec3],
    lower: int,
    upper: int,
    dt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    q0: wp.array[wp.vec3],
    q1: wp.array[wp.vec3],
    qd0: wp.array[wp.vec3],
    qd1: wp.array[wp.vec3],
):
    """Put each carried node between where the lower group and the upper one will have it."""
    i = wp.tid()
    particle = nodes[i]
    below = wp.transform_point(_pose_after(body_q[lower], body_qd[lower], dt), low[i])
    above = wp.transform_point(_pose_after(body_q[upper], body_qd[upper], dt), high[i])
    x = (1.0 - along[i]) * below + along[i] * above
    q0[particle] = x
    q1[particle] = x
    qd0[particle] = wp.vec3(0.0)
    qd1[particle] = wp.vec3(0.0)


@dataclass(frozen=True, kw_only=True)
class PresserSpec:
    """A flat kinematic plate on a substep-sampled path; it holds its last pose."""

    entity: str = "scripted_presser"
    name: str = "presser"
    center: tuple[float, float, float] = (0.0, 0.0, 0.1)
    half: tuple[float, float, float] = PRESSER_HALF
    control_hz: int = PHYSICS.control_hz
    substeps: int = PHYSICS.substeps
    duration: float = DURATION


@rss.register_entity("scripted_presser")
class Presser(rss.Entity):
    domain = "rigid"

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        self.center = np.asarray(self.placed((*spec.center, 0.0, 0.0, 0.0, 1.0))[:3], dtype=np.float64)
        hx, hy, hz = (float(v) for v in spec.half)
        mass = 0.2
        inertia = np.diag(np.array((hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy)) * mass / 3)
        self.body = builder.add_body(
            xform=wp.transform(self.center, wp.quat_identity()),
            mass=mass,
            inertia=wp.mat33(inertia),
            lock_inertia=True,
            is_kinematic=True,
            label=spec.name,
        )
        cfg = newton.ModelBuilder.ShapeConfig(mu=PHYSICS.shape_mu, ke=1.0e5, kd=25.0, gap=0.001, density=0.0)
        self.shape = builder.add_shape_box(
            self.body, hx=hx, hy=hy, hz=hz, cfg=cfg, color=(0.16, 0.30, 0.39), label=f"{spec.name}_plate"
        )

    def on_scene_ready(self, pipeline) -> None:
        self.pipeline = pipeline
        self._model = self.stepping_model if self.stepping_model is not None else pipeline.model
        spec, device = self.spec, self._model.device
        self.dt = 1.0 / (spec.control_hz * spec.substeps)
        self.capacity = int(np.ceil(spec.duration / self.dt)) + 2
        poses = np.zeros((self.capacity, 7), dtype=np.float32)
        poses[:, :3], poses[:, 6] = self.center, 1.0
        self._poses = wp.array(poses, dtype=wp.transform, device=device)
        self._rates = wp.zeros(self.capacity, dtype=wp.spatial_vector, device=device)
        self._lengths = wp.ones(1, dtype=int, device=device)
        self._tick = wp.zeros(1, dtype=int, device=device)

    def set_trajectory(self, poses: np.ndarray) -> None:
        """Replace the path with ``[N, 7]`` poses at the physics rate, from the next substep on."""
        poses = np.asarray(poses, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 7 or not 1 <= len(poses) <= self.capacity:
            raise ValueError(f"the plate's path needs 1 to {self.capacity} poses of 7 numbers")
        rates = np.zeros((len(poses), 6), dtype=np.float32)
        rates[:-1, :3] = np.diff(poses[:, :3], axis=0) / self.dt
        device = self._model.device
        source = wp.array(poses, dtype=wp.transform, device=device)
        wp.copy(self._poses, source, count=source.size)
        source = wp.array(rates, dtype=wp.spatial_vector, device=device)
        wp.copy(self._rates, source, count=source.size)
        self._lengths.assign(np.array([len(poses)], dtype=np.int32))
        self._tick.zero_()

    def contribute_hooks(self, hooks) -> None:
        hooks.add("pre_solve", self._drive, name="hold-presser-plate")
        hooks.add("post_solve", self._advance, name="advance-presser-plate")

    def _drive(self, state, dt) -> None:
        del dt
        wp.launch(
            _hold_plate,
            dim=1,
            inputs=[
                self._tick,
                self._lengths,
                self.body,
                self._poses,
                self._rates,
                state.body_q,
                self.pipeline.state_1.body_q,
                state.body_qd,
                self.pipeline.state_1.body_qd,
            ],
            device=self._model.device,
        )

    def _advance(self, state, dt) -> None:
        del state, dt
        wp.launch(_advance_tick, dim=1, inputs=[self._tick, self._lengths], device=self._model.device)

    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        del pipeline, episode, world
        if hasattr(self, "_tick"):
            self._tick.zero_()


@dataclass(frozen=True, kw_only=True)
class StampSpec(rigid_part_body.RigidPartSoftSpec):
    """A :class:`rigid_part_body.RigidPartSoftSpec` that can carry one part between its two rigid groups."""

    entity: str = "stamp_rigid_parts"
    carried_part: int | None = None
    """A part bonded to both rigid groups that is drawn between them, not solved: its other nodes
    have no mass, which is how this solver leaves a particle alone, and :class:`CarriedPose` puts each
    one, every substep, where the two groups' poses have it, weighted by how far from one seat to
    the other it is. So the part shortens evenly between them, as a coil does. None carries nothing."""
    carried_land: float = 0.0
    """Within this of either seat a carried node moves with that seat's group alone [m]."""


@rss.register_entity("stamp_rigid_parts")
class EntityStamp(rigid_part_body.EntityRigidParts):
    spec: StampSpec

    def build(self, builder, episode) -> None:
        super().build(builder, episode)
        spec = self.spec
        self.carried = np.zeros(0, dtype=np.int64)
        if spec.carried_part is None:
            return
        if len(self.rigid_parts) != 2:
            raise ValueError(f"{spec.name}: a carried part goes between exactly two rigid groups")
        tets, labels = self.source_tets, self.tet_part_labels
        n = len(self.source_vertices)
        start = int(builder.particle_count) - n
        x = np.array([[float(c) for c in p] for p in builder.particle_q[start : start + n]], dtype=np.float64)
        part = np.unique(tets[labels == spec.carried_part])
        centres = [x[rp.nodes[0]] - rp.body_point[0] for rp in self.rigid_parts]
        seats = [np.intersect1d(part, rp.nodes) for rp in self.rigid_parts]
        if min(len(seat) for seat in seats) == 0:
            raise ValueError(f"{spec.name}: part {spec.carried_part} is not bonded to both rigid groups")
        heights = [float(x[seat, 2].mean()) for seat in seats]
        bottom, top = (0, 1) if heights[0] < heights[1] else (1, 0)
        free = np.setdiff1d(part, np.concatenate(seats))
        self.carried = free
        first, last = heights[bottom] + spec.carried_land, heights[top] - spec.carried_land
        self.carried_along = np.clip((x[free, 2] - first) / (last - first), 0.0, 1.0)
        self.carried_low, self.carried_high = x[free] - centres[bottom], x[free] - centres[top]
        self.carried_bodies = (int(self.rigid_parts[bottom].body), int(self.rigid_parts[top].body))
        for node in free:
            builder.particle_mass[start + int(node)] = 0.0
        self.mass[free] = 0.0


class CarriedPose:
    """Pose an :class:`EntityStamp`'s carried part from its two rigid groups, before every substep's solve."""

    def __init__(self, sim, entity: EntityStamp, dt: float) -> None:
        """``dt`` is the substep [s]: the nodes go where the groups will be at its end, not where they were."""
        self.pipeline, device = sim.pipeline, sim.model.device
        self.device, self.dt = device, float(dt)
        self.lower, self.upper = entity.carried_bodies
        self.nodes = wp.array(np.asarray(entity.particles)[entity.carried], dtype=int, device=device)
        self.along = wp.array(entity.carried_along, dtype=float, device=device)
        self.low = wp.array(entity.carried_low, dtype=wp.vec3, device=device)
        self.high = wp.array(entity.carried_high, dtype=wp.vec3, device=device)
        sim.pipeline.hooks.add("pre_solve", self.apply, name="pose-carried-part")
        sim.pipeline.invalidate_graph()

    def apply(self, state, dt) -> None:
        del dt
        other = self.pipeline.state_1
        wp.launch(
            _pose_carried,
            dim=len(self.nodes),
            inputs=[
                self.nodes,
                self.along,
                self.low,
                self.high,
                self.lower,
                self.upper,
                self.dt,
                state.body_q,
                state.body_qd,
            ],
            outputs=[state.particle_q, other.particle_q, state.particle_qd, other.particle_qd],
            device=self.device,
        )


def check_stamp(sim, entity, tets: np.ndarray, settings: PhysicsSettings, *, mu: float | None) -> dict:
    """Check what Newton built against the stamp's materials, apply the contact settings, and report."""
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    if not type(solver).__name__.startswith("SolverVBD"):
        raise RuntimeError(f"expected a VBD solver, got {type(solver).__name__}")
    if sim.model.tet_count != len(tets) or len(entity.tet_part_labels) != len(tets):
        raise RuntimeError("RSS lost stamp tetrahedra or part labels")
    if sim.model.attachment_particle_body_count != sum(len(rp.nodes) for rp in entity.rigid_parts):
        raise RuntimeError("a rigid group lost node attachments")
    report = {
        "rigid_body_parts": {
            f"part{rp.part}": {
                "body": rp.body,
                "nodes": int(len(rp.nodes)),
                "seam_nodes": int(rp.seam.sum()),
                "body_mass_kg": rp.body_mass,
                "part_mass_kg": rp.part_mass,
            }
            for rp in entity.rigid_parts
        },
        "attach_stiffness_n_m": entity.spec.attach_stiffness,
        "carried_nodes": int(len(entity.carried)),
        "friction": set_contact(sim, solver, settings, mu=mu, grip_mu=None),
    }
    span = slice(*entity.tet_range)
    if not np.array_equal(sim.model.tet_indices.numpy()[span] - entity.particles[0], tets):
        raise RuntimeError("Newton changed the source tetrahedron order")
    young, nu = entity.elastic.T
    lame = np.column_stack((young / (2.0 * (1.0 + nu)), young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))
    if not np.allclose(sim.model.tet_materials.numpy()[span, :2], lame, rtol=1e-5):
        raise RuntimeError("Newton FEM elasticity differs from the stamp's materials")
    if not np.allclose(sim.model.tet_materials.numpy()[span, 2], entity.damping, rtol=1e-5):
        raise RuntimeError("Newton FEM damping differs from the stamp's materials")
    if not np.allclose(sim.model.particle_radius.numpy()[entity.particles], settings.particle_radius):
        raise RuntimeError("Newton particle radius differs from the settings'")
    if not np.allclose(sim.model.particle_mass.numpy()[entity.particles], entity.mass, rtol=1e-5):
        raise RuntimeError("Newton nodal mass differs from the stamp's per-tet densities")
    return report


def load_stamp(asset: str):
    """The source ``(vertices, tets, labels, rigid groups, part names)``, after checking the files against ``SOURCE.md``."""
    path = ASSET_ROOT / asset
    listed = re.findall(r"- `([^`]+)`: `([0-9a-f]{64})`", (path / "SOURCE.md").read_text(encoding="utf-8"))
    if not listed:
        raise ValueError(f"{asset}: SOURCE.md lists no file hashes")
    for relative, expected in listed:
        if hashlib.sha256((path / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{asset} source asset changed: {relative}")
    vertices, tets, labels, _elastic, _density = load_labelled_tets(
        path / "final_mesh.mesh", path / "heterogeneous_params.npz"
    )
    if set(np.unique(labels).tolist()) != set(range(6)):
        raise ValueError(f"{asset}: expected part labels 0 to 5")
    meta = json.loads((path / "asset.json").read_text(encoding="utf-8"))
    groups = tuple(tuple(g) if len(g) > 1 else g[0] for g in meta["rigid_groups"])
    parts = json.loads((path / "inferred_params.json").read_text(encoding="utf-8"))["predictions"]
    names = tuple(part["part_name"] for part in sorted(parts, key=lambda part: part["part_index"]))
    return vertices, tets, labels, groups, names


def boundary_faces(tets: np.ndarray) -> np.ndarray:
    """The outward triangles of a tet mesh: the faces one tet owns."""
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]])
    _, first, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    return faces[first[count == 1]]


def layout(vertices: np.ndarray, tets: np.ndarray, labels: np.ndarray) -> dict:
    """What the scene is placed and judged by, measured from the mesh (the mesh's own frame, z up)."""
    radius = np.hypot(vertices[:, 0], vertices[:, 1])
    nodes = [np.unique(tets[labels == part]) for part in range(6)]
    rim_z, cap_top = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    die = nodes[DIE]
    tip_z = float(vertices[die, 2].min())
    tips = die[vertices[die, 2] <= tip_z + 1.0e-7]
    # The relief is a digit inside a border ring: the widest radial gap between tip nodes parts them.
    order = np.sort(radius[tips])
    widest = int(np.argmax(np.diff(order)))
    if order[widest + 1] - order[widest] < 0.0015:
        raise ValueError("no gap between the die's digit and its border")
    split = 0.5 * float(order[widest] + order[widest + 1])
    faces = boundary_faces(tets)
    tip_faces = faces[np.isin(faces, tips).all(axis=1)]
    corners = vertices[tip_faces]
    area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    return {
        "nodes": nodes,
        "rim_z": rim_z,
        "cap_top": cap_top,
        "cap_top_nodes": nodes[CAP][vertices[nodes[CAP], 2] >= cap_top - 1.0e-7],
        "tips": tips,
        "tip_faces": tip_faces,
        "tip_face_area": area,
        "tip_face_digit": radius[tip_faces].max(axis=1) < split,
        "report": {
            "height_m": cap_top - rim_z,
            "die_recess_m": tip_z - rim_z,
            "tip_nodes": int(len(tips)),
            "tip_faces": int(len(tip_faces)),
            "relief_area_m2": float(area.sum()),
            "digit_border_split_radius_m": split,
        },
    }


def build_scene(
    *,
    asset: str = ASSET,
    solver_name: str = SOLVER,
    iterations: int = 80,
    mu: float | None = None,
    self_contact: bool = False,
    self_contact_radius: float = 0.0004,
    attach_stiffness: float = 1.0e4,
    attach_damping: float = ATTACH_DAMPING,
    return_young: float | None = None,
    return_density: float | None = None,
    core_density: float | None = None,
    spring: SpringGuide | None = None,
    draw_return: bool = True,
):
    vertices, tets, labels, groups, names = load_stamp(asset)
    geometry = layout(vertices, tets, labels)
    overrides = ((RETURN, "youngs_modulus_pa", float(return_young)),) if return_young is not None else ()
    if return_density is not None:
        overrides += ((RETURN, "density_kg_m3", float(return_density)),)
    if core_density is not None:
        overrides += tuple((part, "density_kg_m3", float(core_density)) for part in (CAP, PLUNGER))
    settings = dataclasses.replace(PHYSICS, part_overrides=overrides)
    physics = arrangement(
        settings,
        claw="claw",
        coarse=False,
        solver_name=solver_name,
        iterations=iterations,
        self_contact=self_contact,
        self_contact_radius=self_contact_radius,
        # The sliding pairs are 0.75 to 1 mm apart at rest: the exclusion has to stay inside that.
        rest_exclusion=2.0 * self_contact_radius,
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="stamp_press")
    paper_top = 2.0 * PAPER_HALF[2]
    scene.add(
        rss.slab(
            name="paper",
            half=PAPER_HALF,
            at=(STAMP_AT[0], STAMP_AT[1], PAPER_HALF[2]),
            color=PAPER_COLOR,
        )
    )
    # A node rests one particle radius over the paper; the stamp's origin is its rim's centre.
    rest_z = paper_top + settings.particle_radius
    pose = (STAMP_AT[0], STAMP_AT[1], rest_z + PLACE_GAP - geometry["rim_z"])
    obj = scene.add(
        StampSpec(
            carried_part=RETURN if spring is not None and draw_return else None,
            carried_land=HELIX_LAND,
            rigid_parts=groups,
            attach_stiffness=attach_stiffness,
            attach_damping=attach_damping,
            name=asset,
            mesh_path=str(ASSET_ROOT / asset / "final_mesh.mesh"),
            params_path=str(ASSET_ROOT / asset / "heterogeneous_params.npz"),
            materials_path=str(ASSET_ROOT / asset / "inferred_params.json"),
            pose=pose,
            contact_ke=settings.contact_ke,
            contact_kd=settings.contact_kd,
            contact_mu=settings.soft_mu,
            particle_radius=settings.particle_radius,
            damping_ratio=settings.damping_ratio,
            part_overrides=settings.part_overrides,
        )
    )
    # Where the plate's bottom face first pushes the cap's nodes, before the stamp settles.
    touch_z = pose[2] + geometry["cap_top"] + settings.particle_radius
    presser = scene.add(PresserSpec(center=(STAMP_AT[0], STAMP_AT[1], touch_z + APPROACH + PRESSER_HALF[2])))
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    foot = surface + np.array((STAMP_AT[0], STAMP_AT[1], paper_top))
    for camera in (
        rss.CameraSpec(
            name="overview",
            width=960,
            height=540,
            eye=tuple(float(v) for v in np.add(foot, (0.17, -0.20, 0.13))),
            look_at=tuple(float(v) for v in np.add(foot, (0.0, 0.0, 0.034))),
            fov_y=30.0,
        ),
        # Level with the cap's skirt and side on, so its travel reads against the sleeve.
        rss.CameraSpec(
            name="side",
            width=720,
            height=720,
            eye=tuple(float(v) for v in np.add(foot, (0.0, -0.24, 0.042))),
            look_at=tuple(float(v) for v in np.add(foot, (0.0, 0.0, 0.040))),
            fov_y=24.0,
        ),
    ):
        scene.add(camera)
    sim = build_sim(scene, rigid_loop=False)
    entity = obj.entity
    report = dict(geometry["report"])
    report.update(check_stamp(sim, entity, tets, settings, mu=mu))
    young = entity.elastic[:, 0]
    if len(entity.carried):
        CarriedPose(sim, entity, 1.0 / (settings.control_hz * settings.substeps))
    forces = None
    if spring is not None:
        core = next(rp for rp in entity.rigid_parts if str(CAP) in rp.part.split("+"))
        sleeve = next(rp for rp in entity.rigid_parts if rp.part == str(SLEEVE))
        forces = SpringGuideForces(
            sim,
            entity,
            core,
            sleeve,
            spring,
            rim_z=float(surface[2] + pose[2] + geometry["rim_z"]),
            dt=1.0 / (settings.control_hz * settings.substeps),
        )
        report["spring_guide"] = dataclasses.asdict(spring) | {
            "cap_group_mass_kg": forces.group_mass_kg[0],
            "sleeve_group_mass_kg": forces.group_mass_kg[1],
        }
    report["asset"] = asset
    report["source_stiffness_ratio"] = float(entity.source_elastic[:, 0].max() / entity.source_elastic[:, 0].min())
    # What Newton was given: a rigid group's tets keep part_stiffness_scale of their modulus.
    report["effective_young_pa_by_part"] = {names[p]: float(young[labels == p].max()) for p in range(6)}
    report["effective_stiffness_ratio"] = float(young.max() / young.min())
    report["paper_top_m"] = paper_top
    report["stamp_pose_m"] = [float(v) for v in pose]
    print(json.dumps({"geometry": report}), flush=True)
    return sim, obj, presser, forces, geometry, report


class SpringGuideForces:
    """Apply a :class:`SpringGuide` between the cap group and the sleeve, every substep.

    A force on a rigid group reaches the rest of it only through the nodes' attachments, and this
    solve passes little of that on per substep: an earlier stamp's cap, on a foam collar, settled as if it weighed
    2-3 N whatever its mass. So the wrench is applied as one rigid acceleration of the whole
    group -- the body and every node tied to it, each by its own mass -- which is how gravity
    reaches them, and is in the solver's prediction before its first sweep.
    """

    def __init__(self, sim, entity, core, sleeve, spec: SpringGuide, *, rim_z: float, dt: float) -> None:
        """``rim_z`` is the stamp's rim in the world [m] as built, which the guide's height is measured from."""
        model = sim.model
        device = model.device
        self.spec, self.model = spec, model
        self.core, self.sleeve = int(core.body), int(sleeve.body)
        body_q = sim.pipeline.state_0.body_q.numpy()
        self.rest = float(body_q[self.core, 2] - body_q[self.sleeve, 2])
        self.up_core = float(rim_z + spec.guide_height - body_q[self.core, 2])
        self.up_sleeve = float(rim_z + spec.guide_height - body_q[self.sleeve, 2])
        particle_mass, body_mass = model.particle_mass.numpy(), model.body_mass.numpy()
        body_inertia = model.body_inertia.numpy()
        particles = np.asarray(entity.particles)
        inv_mass, inv_inertia, nodes, group = [], [], [], []
        for index, part in enumerate((core, sleeve)):
            m = particle_mass[particles[part.nodes]].astype(np.float64)
            r = part.body_point  # each node from the body's centre of mass, in its frame
            spread = (r * r).sum(axis=1)[:, None, None] * np.eye(3) - np.einsum("ni,nj->nij", r, r)
            inertia = body_inertia[part.body].astype(np.float64).reshape(3, 3) + (m[:, None, None] * spread).sum(axis=0)
            inv_mass.append(1.0 / float(body_mass[part.body] + m.sum()))
            inv_inertia.append(np.linalg.inv(inertia))
            # The forces are explicit: each stiffness has to stay well inside what the group's own
            # inertia can follow in one substep, or the group rings up instead of settling.
            up = (self.up_core, self.up_sleeve)[index]
            turning = (spec.align + spec.guide * up * up) / float(np.linalg.eigvalsh(inertia).min())
            pushing = max(spec.stop + spec.rate, spec.guide) * inv_mass[-1]
            if max(turning, pushing) * dt * dt > 1.0:
                raise ValueError(
                    f"SpringGuide is too stiff for a {1e3 / inv_mass[-1]:.1f} g group at {1.0 / dt:.0f} Hz: "
                    f"(omega*dt)^2 is {turning * dt * dt:.2f} turning and {pushing * dt * dt:.2f} pushing"
                )
            nodes.append(particles[part.nodes])
            group.append(np.full(len(part.nodes), index))
        self.group_mass_kg = [1.0 / value for value in inv_mass]
        self.bodies = wp.array([self.core, self.sleeve], dtype=int, device=device)
        self.inv_mass = wp.array(inv_mass, dtype=float, device=device)
        self.inv_inertia = wp.array(np.asarray(inv_inertia, dtype=np.float32), dtype=wp.mat33, device=device)
        self.nodes = wp.array(np.concatenate(nodes), dtype=int, device=device)
        self.group = wp.array(np.concatenate(group), dtype=int, device=device)
        self.centre = wp.zeros(2, dtype=wp.vec3, device=device)
        self.linear = wp.zeros(2, dtype=wp.vec3, device=device)
        self.angular = wp.zeros(2, dtype=wp.vec3, device=device)
        self.reading = wp.zeros(2, dtype=float, device=device)
        sim.pipeline.hooks.add("rigid_force", self.apply, name="stamp-spring-guide")
        sim.pipeline.invalidate_graph()

    def apply(self, state, dt) -> None:
        del dt
        spec, model = self.spec, self.model
        wp.launch(
            _spring_guide,
            dim=1,
            inputs=[
                self.core,
                self.sleeve,
                state.body_q,
                state.body_qd,
                model.body_com,
                self.rest,
                self.up_core,
                self.up_sleeve,
                spec.rate,
                spec.preload,
                spec.travel,
                spec.stop,
                spec.damping,
                spec.guide,
                spec.guide_damping,
                spec.align,
                spec.align_damping,
                self.inv_mass,
                self.inv_inertia,
            ],
            outputs=[self.centre, self.linear, self.angular, self.reading],
            device=model.device,
        )
        wp.launch(
            _accelerate_nodes,
            dim=len(self.nodes),
            inputs=[
                self.nodes,
                self.group,
                state.particle_q,
                model.particle_mass,
                self.centre,
                self.linear,
                self.angular,
            ],
            outputs=[state.particle_f],
            device=model.device,
        )
        wp.launch(
            _accelerate_bodies,
            dim=2,
            inputs=[self.bodies, state.body_q, model.body_mass, model.body_inertia, self.linear, self.angular],
            outputs=[state.body_f],
            device=model.device,
        )

    def read(self) -> tuple[float, float]:
        """The cap's height over its rest on the sleeve [m] and the axial force on it [N], last substep."""
        x, push = self.reading.numpy()
        return float(x), float(push)


def press_path(*, touch_z: float, travel: float, start: float, dt: float, at) -> np.ndarray:
    """The plate's centre poses from ``start`` [s] on: down ``travel`` past ``touch_z``, hold, and back up."""
    count = int(np.ceil((DURATION - start) / dt)) + 2
    poses = np.zeros((count, 7), dtype=np.float32)
    poses[:, 0], poses[:, 1], poses[:, 6] = at[0], at[1], 1.0
    for index in range(count):
        clock = start + index * dt
        down = smooth(clock, *PHASES["press"]) - smooth(clock, *PHASES["release"])
        poses[index, 2] = touch_z + APPROACH - (APPROACH + travel) * down + PRESSER_HALF[2]
    return poses


def _in_body(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    """World ``points`` in the frame of a body at ``pose`` (p, xyzw)."""
    v, w = -pose[3:6].astype(np.float64), float(pose[6])
    d = points - pose[:3].astype(np.float64)
    t = 2.0 * np.cross(v, d)
    return d + w * t + np.cross(v, t)


class RigidVolume:
    """A rigid group's tets in its body's frame, for asking which points have got inside it.

    The moving parts have no contact with each other unless self-contact is on, so a shaft can
    pass through its guide and a skirt through its shroud with nothing to say so but this.
    """

    def __init__(self, part: rigid_part_body.RigidPart, tets: np.ndarray, rows: np.ndarray, cell: float = 0.002):
        index = np.full(int(tets.max()) + 1, -1)
        index[part.nodes] = np.arange(len(part.nodes))
        corners = part.body_point[index[tets[rows]]]  # (n, 4, 3)
        self.origin = corners[:, 0]
        self.inverse = np.linalg.inv(np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2))
        # Each corner's height over the face opposite it: a barycentric weight times this is a depth.
        six_volumes = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1]))
        opposite = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
        doubled = [
            np.linalg.norm(np.cross(corners[:, b] - corners[:, a], corners[:, c] - corners[:, a]), axis=1)
            for a, b, c in opposite
        ]
        self.height = six_volumes[:, None] / np.stack(doubled, axis=1)
        low, high = corners.min(axis=1), corners.max(axis=1)
        self.low, self.cell = low.min(axis=0), float(cell)
        first = np.floor((low - self.low) / cell).astype(int)
        last = np.floor((high - self.low) / cell).astype(int)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for tet, (a, b) in enumerate(zip(first, last, strict=True)):
            for i in range(a[0], b[0] + 1):
                for j in range(a[1], b[1] + 1):
                    for k in range(a[2], b[2] + 1):
                        buckets.setdefault((i, j, k), []).append(tet)
        self.buckets = {key: np.asarray(value) for key, value in buckets.items()}

    def inside(self, pose: np.ndarray, points: np.ndarray, deeper: float = INSIDE_DEPTH) -> int:
        """How many world ``points`` are more than ``deeper`` inside the volume, its body at ``pose``."""
        local = _in_body(pose, points)
        cells = np.floor((local - self.low) / self.cell).astype(int)
        count = 0
        for point, cell in zip(local, cells, strict=True):
            near = self.buckets.get((int(cell[0]), int(cell[1]), int(cell[2])))
            if near is None:
                continue
            bary = np.einsum("tij,tj->ti", self.inverse[near], point - self.origin[near])
            weights = np.column_stack((1.0 - bary.sum(axis=1), bary))
            count += bool(np.any((weights * self.height[near]).min(axis=1) > deeper))
        return count


def _tilt_deg(pose: np.ndarray) -> float:
    """How far a body's z axis is off vertical [deg]."""
    x, y = float(pose[3]), float(pose[4])
    return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


def _yaw_deg(pose: np.ndarray) -> float:
    """How far a body has turned about the vertical [deg]."""
    x, y, z, w = (float(v) for v in pose[3:7])
    return float(np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))))


def _ink(mask: np.ndarray, triangles: np.ndarray) -> None:
    """Fill ``triangles`` ``[n, 3, 2]`` (pixel coordinates) into the boolean ``mask``."""
    height, width = mask.shape
    for a, b, c in triangles:
        lo = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        hi = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        x0, y0, x1, y1 = max(lo[0], 0), max(lo[1], 0), min(hi[0], width - 1), min(hi[1], height - 1)
        if x0 > x1 or y0 > y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d0 = (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])
        d1 = (c[0] - b[0]) * (ys - b[1]) - (c[1] - b[1]) * (xs - b[0])
        d2 = (a[0] - c[0]) * (ys - c[1]) - (a[1] - c[1]) * (xs - c[0])
        inside = ((d0 >= 0) & (d1 >= 0) & (d2 >= 0)) | ((d0 <= 0) & (d1 <= 0) & (d2 <= 0))
        mask[y0 : y1 + 1, x0 : x1 + 1] |= inside


def phase_at(clock: float) -> str:
    return next((name for name, (_, end) in PHASES.items() if clock <= end + 1.0e-9), "rest")


def judge(rows: list[dict], rest: dict, report: dict, steps: int) -> dict:
    """What the press did, from the per-frame rows and the settled reference ``rest``."""
    planned = [row for row in rows if row["frame"] > rest["frame"]]
    touched = [row for row in planned if row["tip_nodes_touching"] > 0]
    first = touched[0] if touched else None
    peak = max(planned, key=lambda row: row["relief_inked_now"]) if planned else None
    final = rows[-1]
    settled = rows[-round(0.3 * PHYSICS.control_hz) :]
    result = {
        "complete": steps >= round(DURATION * PHYSICS.control_hz),
        "finite": all(row["finite"] for row in rows),
        "rest_die_gap_m": rest["die_gap_m"],
        "rest_cap_sag_m": rest["cap_sag_m"],
        "max_tilt_deg": max(max(row["core_tilt_deg"], row["sleeve_tilt_deg"]) for row in rows),
        "max_yaw_deg": max(max(abs(row["core_yaw_deg"]), abs(row["sleeve_yaw_deg"])) for row in rows),
        "touched_paper": first is not None,
        "first_touch_time_s": None if first is None else first["time_s"],
        "press_travel_at_first_touch_m": None if first is None else first["press_travel_m"],
        "peak_press_travel_m": max((row["press_travel_m"] for row in planned), default=0.0),
        "peak_relief_touching": 0.0 if peak is None else peak["relief_inked_now"],
        "peak_paper_force_n": max((row["paper_force_n"] for row in planned), default=0.0),
        "peak_die_force_n": max((row["die_force_n"] for row in planned), default=0.0),
        "peak_rim_force_n": max((row["rim_force_n"] for row in planned), default=0.0),
        "peak_sleeve_lift_m": max((row["sleeve_lift_m"] for row in planned), default=0.0),
        "final_tip_nodes_touching": final["tip_nodes_touching"],
        "final_die_gap_m": final["die_gap_m"],
        "final_press_travel_m": final["press_travel_m"],
        "final_plate_contacts": final["plate_contacts"],
        "rest_spring_force_n": rest["spring_force_n"],
        "peak_spring_force_n": max((row["spring_force_n"] for row in planned), default=0.0),
        "peak_spring_travel_m": max((row["spring_travel_m"] for row in planned), default=0.0),
        "final_drift_m_s": max(abs(row["core_speed_m_s"]) for row in settled),
        "max_core_nodes_in_sleeve": max(row["core_nodes_in_sleeve"] for row in rows),
        "max_return_nodes_in_sleeve": max(row["return_nodes_in_sleeve"] for row in rows),
        "max_return_nodes_in_core": max(row["return_nodes_in_core"] for row in rows),
        "max_rigid_node_stretch_m": max(row["rigid_node_stretch_m"] for row in rows),
        "max_inverted_soft_tets": max(row["inverted_soft_tets"] for row in rows),
        "max_inverted_rigid_tets": max(row["inverted_rigid_tets"] for row in rows),
        "final_inverted_soft_tets": final["inverted_soft_tets"],
    }
    checks = {
        "ran": result["complete"] and result["finite"],
        "stands": result["max_tilt_deg"] < 2.0 and result["peak_sleeve_lift_m"] < 0.0005,
        # Nothing turns the stamp: one that has turned is being pushed by something that is not there.
        "does_not_turn": result["max_yaw_deg"] < 1.0,
        "die_clear_at_rest": rest["die_gap_m"] > 0.0015 and rest["tip_nodes_touching"] == 0,
        "die_reaches_paper": result["touched_paper"],
        "springs_back": (
            final["tip_nodes_touching"] == 0
            and abs(final["press_travel_m"]) < 0.0003
            and final["die_gap_m"] > 0.9 * rest["die_gap_m"]
        ),
        "parts_stay_clear": (
            result["max_core_nodes_in_sleeve"] == 0
            and result["max_return_nodes_in_sleeve"] == 0
            and result["max_return_nodes_in_core"] == 0
        ),
        "soft_mesh_intact": result["max_inverted_soft_tets"] == 0,
    }
    result["checks"] = checks
    result["effective_stiffness_ratio"] = report["effective_stiffness_ratio"]
    return result


def run(
    out: Path,
    *,
    steps: int,
    render_every: int = 0,
    asset: str = ASSET,
    overpress: float = OVERPRESS,
    solver_name: str = SOLVER,
    iterations: int = 80,
    mu: float | None = None,
    self_contact: bool = False,
    self_contact_radius: float = 0.0004,
    attach_stiffness: float = 1.0e4,
    attach_damping: float = ATTACH_DAMPING,
    return_young: float | None = None,
    return_density: float | None = None,
    core_density: float | None = None,
    spring: SpringGuide | None = None,
    body_spring: bool = True,
    draw_return: bool = True,
):
    out.mkdir(parents=True, exist_ok=True)
    if return_young is None:
        return_young = HELIX_YOUNG_PA
    if spring is None and body_spring:
        spring = SpringGuide()
    sim, obj, presser, forces, geometry, report = build_scene(
        asset=asset,
        solver_name=solver_name,
        iterations=iterations,
        mu=mu,
        self_contact=self_contact,
        self_contact_radius=self_contact_radius,
        attach_stiffness=attach_stiffness,
        attach_damping=attach_damping,
        return_young=return_young,
        return_density=return_density,
        core_density=core_density,
        spring=spring if body_spring else None,
        draw_return=draw_return,
    )
    entity = obj.entity
    particles = np.asarray(entity.particles)
    tets, labels = entity.source_tets, entity.tet_part_labels
    nodes, tips, tip_faces = geometry["nodes"], geometry["tips"], geometry["tip_faces"]
    area, digit = geometry["tip_face_area"], geometry["tip_face_digit"]
    info = geometry["report"]
    rigid_tets = np.isin(labels, (CAP, PLUNGER))
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    paper_top, radius, ke = 2.0 * PAPER_HALF[2], PHYSICS.particle_radius, PHYSICS.contact_ke
    soft_tets = np.isin(labels, SOFT)
    core = next(rp for rp in entity.rigid_parts if str(CAP) in rp.part.split("+"))
    sleeve = next(rp for rp in entity.rigid_parts if rp.part == str(SLEEVE))
    # What must stay out of what: the moving core's surface out of the sleeve, and the return
    # element out of both rigid groups, but for the nodes it is bonded to them by.
    sleeve_volume = RigidVolume(sleeve, tets, labels == SLEEVE)
    core_volume = RigidVolume(core, tets, rigid_tets)
    skin = np.unique(boundary_faces(tets))
    movers = np.intersect1d(skin, np.concatenate((nodes[CAP], nodes[PLUNGER], nodes[FOAM], nodes[DIE])))
    return_off_sleeve = np.setdiff1d(nodes[RETURN], nodes[SLEEVE])
    return_off_core = np.setdiff1d(nodes[RETURN], np.concatenate((nodes[CAP], nodes[PLUNGER])))
    quality = MeshQuality(sim, entity)
    contacts = sim.pipeline.contacts
    trajectory = Trajectory(sim, entity)
    dt = 1.0 / (PHYSICS.control_hz * PHYSICS.substeps)
    settle_frames = round(PHASES["settle"][1] * PHYSICS.control_hz)
    # Where the cap's top would stand on the paper if what holds it up carried no weight [m, bench frame].
    unloaded_cap_top = report["stamp_pose_m"][2] - PLACE_GAP + geometry["cap_top"]

    size = round(2.0 * INK_HALF / INK_PIXEL)
    mask = np.zeros((size, size), dtype=bool)
    inked_at = np.full((len(tip_faces), 2), np.inf)
    face_inked = np.zeros(len(tip_faces), dtype=bool)
    rows, rest, previous_core_z, travel = [], None, None, None

    videos, renderer, video_error = None, None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet
        from robosimstudio.solvers.render import FALLBACK

        # One renderer over both cameras: ``sim.render`` keeps one camera's, and rebuilds it on a change.
        backend, _why = sim.platform.renderer_for("")
        renderer = sim.make_renderer(backend, cameras=list(sim.cameras), channels=["rgb"])
        if backend != FALLBACK:
            renderer.render(sim.state)  # the first frame only builds the stage
        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // PHYSICS.control_hz))
        videos.__enter__()
    started = time.perf_counter()
    try:
        for frame in range(steps):
            if frame == settle_frames and rows:
                # Plan from where the stamp came to rest: it dropped the placing gap and the cap sagged.
                rest = dict(rows[-1])
                touch_z = rest["cap_top_z_m"] + surface[2] + radius
                at = surface[:2] + np.asarray(STAMP_AT)
                travel = max(rest["die_gap_m"], 0.0) + overpress
                presser.entity.set_trajectory(
                    press_path(touch_z=touch_z, travel=travel, start=frame / PHYSICS.control_hz, dt=dt, at=at)
                )
            sim.run(1).sync()
            state = sim.pipeline.state_0
            q, body_q = state.particle_q.numpy(), state.body_q.numpy()
            owned = q[particles].astype(np.float64)
            local = owned - surface
            over = local[:, 2] - paper_top - radius  # each node's gap to its contact distance over the paper
            push = np.maximum(-over, 0.0) * ke
            tip_touch = over[tips] <= TOUCH_TOLERANCE
            touching = np.zeros(len(owned), dtype=bool)
            touching[tips[tip_touch]] = True
            face_touch = touching[tip_faces].all(axis=1)
            if face_touch.any():
                centroid = local[tip_faces][:, :, :2].mean(axis=1)
                fresh = face_touch & (np.linalg.norm(centroid - inked_at, axis=1) > 0.4 * INK_PIXEL)
                pixels = (local[tip_faces[fresh]][:, :, :2] - (np.asarray(STAMP_AT) - INK_HALF)) / INK_PIXEL
                _ink(mask, pixels)
                inked_at[fresh] = centroid[fresh]
                face_inked |= face_touch
            edges = owned[tets[:, 1:]] - owned[tets[:, :1]]
            inverted = np.linalg.det(edges) <= 0
            core_pose, sleeve_pose = body_q[core.body], body_q[sleeve.body]
            count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
            shapes = contacts.soft_contact_shape.numpy()[:count]
            core_z, sleeve_z = float(core_pose[2] - surface[2]), float(sleeve_pose[2] - surface[2])
            clock = (frame + 1) / PHYSICS.control_hz
            row = {
                "frame": frame + 1,
                "time_s": clock,
                "phase": phase_at(clock),
                "finite": bool(np.isfinite(owned).all() and np.isfinite(body_q).all()),
                "plate_bottom_z_m": float(body_q[presser.entity.body, 2] - surface[2] - PRESSER_HALF[2]),
                "plate_contacts": int(np.count_nonzero(shapes == presser.entity.shape)),
                "spring_travel_m": 0.0 if forces is None else -forces.read()[0],
                "spring_force_n": 0.0 if forces is None else forces.read()[1],
                "cap_top_z_m": float(local[geometry["cap_top_nodes"], 2].mean()),
                "core_z_m": core_z,
                "sleeve_z_m": sleeve_z,
                "core_speed_m_s": 0.0 if previous_core_z is None else (core_z - previous_core_z) * PHYSICS.control_hz,
                "core_tilt_deg": _tilt_deg(core_pose),
                "sleeve_tilt_deg": _tilt_deg(sleeve_pose),
                "core_yaw_deg": _yaw_deg(core_pose),
                "sleeve_yaw_deg": _yaw_deg(sleeve_pose),
                "minimum_z_m": float(local[:, 2].min()),
                "die_gap_m": float(over[tips].min()),
                "tip_nodes_touching": int(tip_touch.sum()),
                "relief_inked_now": float(area[face_touch].sum() / area.sum()),
                "digit_inked_now": float(area[face_touch & digit].sum() / area[digit].sum()),
                "border_inked_now": float(area[face_touch & ~digit].sum() / area[~digit].sum()),
                "paper_force_n": float(push.sum()),
                "die_force_n": float(push[nodes[DIE]].sum()),
                "rim_force_n": float(push[nodes[SLEEVE]].sum()),
                "core_nodes_in_sleeve": sleeve_volume.inside(sleeve_pose, owned[movers]),
                "return_nodes_in_sleeve": sleeve_volume.inside(sleeve_pose, owned[return_off_sleeve]),
                "return_nodes_in_core": core_volume.inside(core_pose, owned[return_off_core]),
                "inverted_soft_tets": int(np.count_nonzero(inverted & soft_tets)),
                "inverted_rigid_tets": int(np.count_nonzero(inverted & ~soft_tets)),
                "soft_contacts": count,
            }
            # Measured from the settled stamp: how far the cap has come down on the sleeve, and
            # how far the sleeve has left the paper.
            reference = rest if rest is not None else row
            row["press_travel_m"] = float((reference["core_z_m"] - core_z) - (reference["sleeve_z_m"] - sleeve_z))
            row["sleeve_lift_m"] = float(local[nodes[SLEEVE], 2].min() - paper_top - radius)
            row["cap_sag_m"] = float(unloaded_cap_top - row["cap_top_z_m"])
            # How far the rigid groups' nodes sit from their bodies: what "approximately rigid" costs.
            parts = rigid_part_body.frame_record(entity, owned, body_q)
            row["rigid_node_stretch_m"] = float(
                max(max(p["seam_stretch_max_m"], p["skin_stretch_max_m"]) for p in parts.values())
            )
            previous_core_z = core_z
            rows.append(row)
            trajectory.add(clock, owned, body_q)
            if videos is not None and frame % render_every == 0:
                sim.sync()
                videos.append_frames(renderer.render(sim.state))
            if frame % 15 == 0 or frame == steps - 1:
                print(json.dumps(row), flush=True)
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

    trajectory.save(out / "trajectory.npz")
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    image[mask] = INK_RGB
    try:
        from PIL import Image

        Image.fromarray(image[::-1]).save(out / "imprint.png")  # row 0 at the top: +y up, as seen from above
    except ImportError:
        np.save(out / "imprint.npy", mask)
    result = judge(rows, rest if rest is not None else rows[0], report, steps)
    result["imprint"] = {
        "relief_inked": float(area[face_inked].sum() / area.sum()),
        "digit_inked": float(area[face_inked & digit].sum() / area[digit].sum()),
        "border_inked": float(area[face_inked & ~digit].sum() / area[~digit].sum()),
        "inked_area_m2": float(mask.sum() * INK_PIXEL**2),
        "relief_area_m2": info["relief_area_m2"],
    }
    result["checks"]["imprint_complete"] = (
        result["imprint"]["digit_inked"] >= 0.95 and result["imprint"]["border_inked"] >= 0.95
    )
    result["success"] = all(result["checks"].values())
    result["mesh_quality"] = quality.read()
    result["config"] = {
        "overpress_m": overpress,
        "plate_travel_m": travel,
        "solver": type(arm_gripper.vbd_solver(sim.pipeline.program.solver)).__name__,
        "iterations": iterations,
        "substeps": PHYSICS.substeps,
        "control_hz": PHYSICS.control_hz,
        "self_contact": self_contact,
        "attach_stiffness_n_m": attach_stiffness,
        "attach_damping_n_s_m": attach_damping,
        "asset": asset,
        "return_young_pa": return_young,
        "return_density_kg_m3": return_density,
        "body_spring": forces is not None,
        "return_element": "drawn between the rigid groups" if len(entity.carried) else "solved",
        "core_density_kg_m3": core_density,
        "newton_version": newton.__version__,
    }
    result["geometry"] = report
    result["video_error"] = video_error
    result["simulated_s"] = len(rows) / PHYSICS.control_hz
    result["elapsed_s"] = elapsed
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "geometry"}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--steps", type=int, default=round(DURATION * PHYSICS.control_hz))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--overpress", type=float, default=OVERPRESS, help="plate travel past the die's touch [m]")
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default=SOLVER)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--mu", type=float, default=None, help="the stamp's soft-contact friction")
    parser.add_argument("--self-contact", action="store_true", help="turn the stamp's self-contact on")
    parser.add_argument("--self-contact-radius", type=float, default=0.0004, help="self-contact radius and margin [m]")
    parser.add_argument("--attach-stiffness", type=float, default=1.0e4, help="rigid-body node stiffness [N/m]")
    parser.add_argument("--attach-damping", type=float, default=ATTACH_DAMPING, help="rigid-body node damping [N*s/m]")
    parser.add_argument(
        "--return-young",
        type=float,
        default=None,
        help="the helix's Young's modulus [Pa]; the scene's when unset",
    )
    parser.add_argument(
        "--core-density", type=float, default=None, help="override the cap and plunger density [kg/m^3]"
    )
    parser.add_argument(
        "--return-density", type=float, default=None, help="the helix's density [kg/m^3]; the source's when unset"
    )
    parser.add_argument("--no-body-spring", dest="body_spring", action="store_false", help="leave the SpringGuide out")
    parser.add_argument("--spring-rate", type=float, default=None, help="the SpringGuide's rate [N/m]")
    parser.add_argument(
        "--solve-return", dest="draw_return", action="store_false", help="leave the helix to the solver, not drawn"
    )
    parser.add_argument("--spring-preload", type=float, default=None, help="the SpringGuide's force at rest [N]")
    args = parser.parse_args()
    levers = {"rate": args.spring_rate, "preload": args.spring_preload}
    spring = dataclasses.replace(SpringGuide(), **{k: v for k, v in levers.items() if v is not None})
    run(
        args.out,
        steps=args.steps,
        render_every=args.render_every,
        overpress=args.overpress,
        solver_name=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        self_contact=args.self_contact,
        self_contact_radius=args.self_contact_radius,
        attach_stiffness=args.attach_stiffness,
        attach_damping=args.attach_damping,
        return_young=args.return_young,
        return_density=args.return_density,
        spring=spring,
        body_spring=args.body_spring,
        draw_return=args.draw_return,
        core_density=args.core_density,
    )


if __name__ == "__main__":
    main()
