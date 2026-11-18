# SPDX-License-Identifier: Apache-2.0
"""Cabled objects for the cable scenes: rigid housings and plugs joined by a soft FEM cable.

Every asset is a labelled tet volume (:func:`load_labelled_tets`) whose rigid groups become
rigid bodies (:mod:`rigid_part_body`); every other label is soft. :data:`ASSETS` holds what each
one needs to lie flat on the bench and the material levers tuned for it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import rigid_part_body

from robosimstudio_materials.heterogeneous import load_labelled_tets

ASSET_ROOT = Path(__file__).resolve().parent / "assets"
TURN_Z = (0.0, 0.0, 0.70710678, 0.70710678)
"""A quarter turn about z: a Z-up asset long along x then lies long along the bench's y."""
REST_EXCLUSION = 0.009
"""Self-contact rest-shape exclusion [m]: at the Clawhauser's 6 mm the cable bunching against the
housing and collar kept kicking the rigid housing, and the charger never settled."""
OBJECT_MU = 0.5
"""The objects' friction against the bench, scenery and each other; only the pads grip harder (``grip_mu``)."""
PLACE_GAP = 0.004
"""How far over the bench an object starts [m]: past the 3 mm particle radius, or the bench
launches the housing on the first substep."""


@dataclasses.dataclass(frozen=True)
class CableAsset:
    folder: str
    quat: tuple[float, float, float, float]
    """Lays the asset flat, long along the bench's y [xyzw]."""
    cable_young: float
    """The cable's Young's modulus [Pa]. The strands are a few tets thick, so the source's rubber stretches."""
    cable_damping: float
    """The cable's tet damping [Pa*s]: a cable's slow, heavy sway."""
    plug_end_mass: float | None
    """Each plug end's mass [kg], or None for the source densities. The meshes fill plugs solid."""
    rigid_groups: tuple = ()
    """Rigid groups when the package has no ``asset.json``."""
    attach_damping: float = 100.0
    """Damping of the nodes tied to each rigid body [N*s/m] (100: q8, swing probe). Undamped, a body and its nodes pass
    energy back and forth and a hanging hub's swing grows instead of dying out."""

    @property
    def path(self) -> Path:
        return ASSET_ROOT / self.folder

    def groups(self) -> tuple:
        meta = self.path / "asset.json"
        if not meta.exists():
            return self.rigid_groups
        return tuple(tuple(g) if len(g) > 1 else g[0] for g in json.loads(meta.read_text())["rigid_groups"])


ASSETS = {
    # The scanned hub; its tuning is usb_hub_box.py's original.
    "usb_hub": CableAsset("usb_hub", (0.5, 0.5, 0.5, 0.5), 2.0e7, 3.0e4, 0.002, rigid_groups=(0, (2, 3))),
    # 4e7 Pa: at 2e7 the 4-4.5 mm cables hung like rope; at 1e8 (EI 1.3e-3 N*m^2, 5-10x a real
    # cable) the cable propped the charger brick up 8 deg. 4e7 lies flat and still holds a bend (q9, user).
    "usb_hub_gen": CableAsset("usb_hub_gen", TURN_Z, 4.0e7, 6.0e4, 0.004),
    "charger": CableAsset("charger", TURN_Z, 4.0e7, 6.0e4, 0.004),
    "power_strip": CableAsset("power_strip", TURN_Z, 2.0e7, 3.0e4, None),
    "y_splitter_cable": CableAsset("y_splitter_cable", TURN_Z, 4.0e7, 6.0e4, 0.004),
    "coiled_usb_cable": CableAsset("coiled_usb_cable", TURN_Z, 4.0e7, 6.0e4, 0.004),
}


def _members(group) -> tuple[int, ...]:
    return tuple(group) if isinstance(group, tuple) else (group,)


def load(asset: CableAsset):
    """The asset's vertices, tets and labels, after checking its files against ``SOURCE.md``."""
    listed = re.findall(r"- `([^`]+)`: `([0-9a-f]{64})`", (asset.path / "SOURCE.md").read_text())
    for relative, expected in listed:
        if hashlib.sha256((asset.path / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{asset.folder} source asset changed: {relative}")
    vertices, tets, labels, _elastic, _density = load_labelled_tets(
        asset.path / "final_mesh.mesh", asset.path / "heterogeneous_params.npz"
    )
    return vertices, tets, labels


def soft_labels(asset: CableAsset, labels: np.ndarray) -> np.ndarray:
    rigid = [p for g in asset.groups() for p in _members(g)]
    return np.setdiff1d(np.unique(labels), rigid)


def spec(asset: CableAsset, vertices, tets, labels, *, name: str, pose, settings, mu: float, overrides=()):
    """The object's :class:`rigid_part_body.RigidPartSoftSpec`, with the asset's levers applied."""
    groups = asset.groups()
    levers = [(p, "youngs_modulus_pa", asset.cable_young) for p in soft_labels(asset, labels)]
    levers += [(p, "damping_pa_s", asset.cable_damping) for p in soft_labels(asset, labels)]
    if asset.plug_end_mass is not None:
        # Plug ends: every rigid group but a housing (label 0, when rigid).
        for group in [g for g in groups if 0 not in _members(g)]:
            part = tets[np.isin(labels, _members(group))]
            volume = float(np.abs(np.linalg.det(vertices[part[:, 1:]] - vertices[part[:, :1]])).sum() / 6.0)
            levers += [(p, "density_kg_m3", asset.plug_end_mass / volume) for p in _members(group)]
    return rigid_part_body.RigidPartSoftSpec(
        rigid_parts=groups,
        attach_damping=asset.attach_damping,
        name=name,
        mesh_path=str(asset.path / "final_mesh.mesh"),
        params_path=str(asset.path / "heterogeneous_params.npz"),
        materials_path=str(asset.path / "inferred_params.json"),
        pose=pose,
        quat=asset.quat,
        contact_ke=settings.contact_ke,
        contact_kd=settings.contact_kd,
        contact_mu=mu,
        particle_radius=settings.particle_radius,
        damping_ratio=settings.damping_ratio,
        part_overrides=tuple(levers) + tuple(overrides),
    )


def episode(
    sim,
    entity,
    camera,
    out: Path,
    *,
    soft: np.ndarray,
    steps: int,
    render_every: int,
    control_hz: int,
    follow,
    task_row,
    stop=None,
):
    """Step ``steps`` frames, writing ``trajectory.npz``, ``diagnostics.json`` and ``overview.mp4``.

    ``follow(frame)`` drives the arm before each frame and returns the route's frames so far;
    ``task_row(frame, local, body_q)`` adds the task's own fields to each frame's row (``local``: the
    object's particles in the bench frame). ``soft`` are the soft labels, whose strain is measured.
    ``stop(row)``, when given, ends the run early once it returns true (a blown-up state).
    Returns the rows, the run's seconds and any video error.
    """
    import arm_gripper
    from clawhauser_scene import Trajectory

    particles, tets, labels = entity.particles, entity.source_tets, entity.tet_part_labels
    soft = np.isin(labels, soft)
    contacts = sim.pipeline.contacts
    finger_shapes = arm_gripper.finger_shapes(sim)
    q0 = sim.pipeline.state_0.particle_q.numpy()[particles]
    rest_inv = np.linalg.inv((q0[tets[:, 1:]] - q0[tets[:, :1]])[soft])
    surface = np.asarray(sim_surface(), dtype=np.float64)
    rows, trajectory = [], Trajectory(sim, entity)
    videos, video_error = None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet

        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // control_hz))
        videos.__enter__()
    started = time.perf_counter()
    try:
        for frame in range(steps):
            arm = follow(frame)
            sim.run(1).sync()
            state = sim.pipeline.state_0
            q, body_q = state.particle_q.numpy(), state.body_q.numpy()
            owned = q[particles]
            edges = owned[tets[:, 1:]] - owned[tets[:, :1]]
            # The cable's strain: each tet's largest principal stretch; its 99th percentile skips slivers.
            finite = bool(np.isfinite(owned).all())
            stretch = (
                np.linalg.svd(rest_inv @ edges[soft], compute_uv=False).max(axis=1) if finite else np.full(1, np.nan)
            )
            count = int(contacts.soft_contact_count.numpy()[0])
            shapes = contacts.soft_contact_shape.numpy()[: min(count, contacts.soft_contact_max)]
            row = {
                "frame": frame + 1,
                "time_s": (frame + 1) / control_hz,
                "minimum_z_m": float(owned[:, 2].min() - surface[2]),
                "inverted_tetrahedra": int(np.count_nonzero(np.linalg.det(edges) <= 0)),
                "cable_inverted": int(np.count_nonzero(np.linalg.det(edges[soft]) <= 0)),
                "cable_stretch_p99": float(np.percentile(stretch, 99.0)),
                "finger_contacts": int(np.count_nonzero(np.isin(shapes, finger_shapes))),
                "aperture_m": arm.aperture(),
                **task_row(frame, owned - surface, body_q),
                **rigid_part_body.frame_record(entity, owned, body_q),
            }
            trajectory.add(row["time_s"], owned, body_q)
            rows.append(row)
            if stop is not None and stop(row):
                print(json.dumps({"stopped_at_frame": frame + 1}), flush=True)
                break
            if videos is not None and frame % render_every == 0:
                videos.append_frames({camera.name: sim.render(camera)})
            if frame % 30 == 0 or frame == steps - 1:
                print(json.dumps({k: v for k, v in row.items() if not k.startswith("part")}), flush=True)
    finally:
        if videos is not None:
            try:
                videos.__exit__(None, None, None)
            except Exception as exc:
                video_error = repr(exc)
    elapsed = time.perf_counter() - started
    trajectory.save(out / "trajectory.npz")
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows, elapsed, video_error


def sim_surface():
    """The bench's top in the world frame [m]."""
    import arm_gripper

    import robosimstudio as rss

    return rss.get_cell(arm_gripper.CELL).surface_frame[:3]


def with_levers(asset: CableAsset, args) -> CableAsset:
    """The asset with ``--cable-young`` / ``--cable-damping`` / ``--attach-damping`` applied, when given."""
    names = ("cable_young", "cable_damping", "attach_damping")
    levers = {k: getattr(args, k) for k in names if getattr(args, k, None) is not None}
    return dataclasses.replace(asset, **levers)


def add_lever_arguments(parser) -> None:
    parser.add_argument("--cable-young", type=float, default=None, help="cable Young's modulus [Pa]")
    parser.add_argument("--cable-damping", type=float, default=None, help="cable damping [Pa*s]")
    parser.add_argument("--attach-damping", type=float, default=None, help="rigid-body node damping [N*s/m]")


def set_friction(sim, solver, settings, *, mu: float, grip_mu: float) -> dict:
    """Friction ``mu`` between the object and every shape, ``grip_mu`` with the pads.

    Shapes take ``mu`` as well, since particle-shape friction mixes as sqrt(mu * shape_mu).
    """
    from clawhauser_scene import set_contact

    return set_contact(sim, solver, dataclasses.replace(settings, shape_mu=mu), mu=mu, grip_mu=grip_mu)


def set_shape_mu(sim, solver, static: str, target: float, mu: float) -> None:
    """Give one static's collision shapes friction ``target`` against the soft object.

    Particle-shape friction mixes as sqrt(mu * shape_mu), with ``mu`` the object's.
    """
    shapes = sim.handles.statics[static].collision_shapes
    for model in {id(sim.model): sim.model, id(solver.model): solver.model}.values():
        shape_mu = model.shape_material_mu.numpy()
        shape_mu[shapes] = target**2 / mu
        model.shape_material_mu.assign(shape_mu)


def config(args, asset: CableAsset, settings) -> dict:
    """What a run was: its arguments, the asset's levers and the solver's rate, for ``result.json``."""
    return {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "asset_levers": dataclasses.asdict(asset),
        "substeps": settings.substeps,
        "control_hz": settings.control_hz,
    }


def fixture(folder: str) -> tuple[Path, dict]:
    """A rigid fixture's collision mesh and ``connectors.json``, after checking its files against ``SOURCE.md``."""
    path = ASSET_ROOT / folder
    listed = re.findall(r"- `([^`]+)`: `([0-9a-f]{64})`", (path / "SOURCE.md").read_text())
    for relative, expected in listed:
        if hashlib.sha256((path / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{folder} source asset changed: {relative}")
    meta = json.loads((path / "asset.json").read_text())
    return path / meta["collision_geometry"], json.loads((path / "connectors.json").read_text())


def bend_path(s, bends, heading: float = 0.0):
    """Points and headings along a planar path of ``bends`` from the origin, at arclengths ``s`` [m].

    ``bends`` is ``((length, radius), ...)``: radius 0 runs straight, a positive one turns left
    (counter-clockwise from above), a negative one right; past the last bend the path runs straight.
    ``heading`` is the start's direction [rad]. Returns the (N, 2) points and (N,) headings.
    """
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    points, headings = np.zeros((len(s), 2)), np.full(len(s), float(heading))
    start, origin, angle = 0.0, np.zeros(2), float(heading)

    def along(t, radius, origin, angle):
        if radius == 0.0:
            return origin + t[:, None] * (np.cos(angle), np.sin(angle)), np.full(len(t), angle)
        theta = angle + t / radius
        return origin + radius * np.stack((np.sin(theta) - np.sin(angle), np.cos(angle) - np.cos(theta)), 1), theta

    for length, radius in (*bends, (np.inf, 0.0)):
        rows = s >= start
        points[rows], headings[rows] = along(np.minimum(s[rows] - start, length), radius, origin, angle)
        if np.isfinite(length):
            end, turned = along(np.array((length,)), radius, origin, angle)
            origin, angle, start = end[0], float(turned[0]), start + length
    return points, headings


def lay(sim, entity, *, root, direction, bends, keep) -> dict:
    """Lay the built (straight) cable along a planar path on the bench, before the first step.

    The rest shape stays straight; only the initial state moves. Each particle past ``root`` [m,
    world] along the horizontal ``direction`` goes to the point of :func:`bend_path` at its
    arclength, its sideways offset turned with the path and its height kept. A rigid part moves
    rigidly with the path's frame at its node nearest the root, its body and free joint with it;
    the ``keep`` nodes (local ids: the housing) and parts made of them stay put. Both state
    buffers are written, as :meth:`arm_gripper.ArmControl.start` does, and velocities zeroed.
    """
    import newton

    model, pipeline = sim.model, sim.pipeline
    particles = np.asarray(entity.particles)
    root = np.asarray(root, dtype=np.float64)
    u = np.array((direction[0], direction[1], 0.0), dtype=np.float64)
    u /= np.linalg.norm(u)
    v = np.array((-u[1], u[0], 0.0))
    heading = float(np.arctan2(u[1], u[0]))
    q = pipeline.state_0.particle_q.numpy()
    x = q[particles].astype(np.float64)
    d = x - root
    s, n = d @ u, d @ v
    points, headings = bend_path(s, bends, heading)
    new = x.copy()
    move = s > 0.0
    move[np.asarray(keep)] = False
    new[move, 0] = root[0] + points[move, 0] - n[move] * np.sin(headings[move])
    new[move, 1] = root[1] + points[move, 1] + n[move] * np.cos(headings[move])
    body_q, joint_q = pipeline.state_0.body_q.numpy(), pipeline.state_0.joint_q.numpy()
    body_qd = pipeline.state_0.body_qd.numpy()
    joint_qd = pipeline.state_0.joint_qd.numpy()
    starts, dof_starts = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    children, kinds = model.joint_child.numpy(), model.joint_type.numpy()
    moved = {}
    for rp in entity.rigid_parts:
        if np.isin(rp.nodes, keep).all():
            continue
        near = float(s[rp.nodes].min())
        point, angle = bend_path([near], bends, heading)
        turn = float(angle[0]) - heading
        c, w = np.cos(turn), np.sin(turn)
        rotation = np.array(((c, -w, 0.0), (w, c, 0.0), (0.0, 0.0, 1.0)))
        centre = root + near * u
        target = root + np.array((point[0, 0], point[0, 1], 0.0))
        new[rp.nodes] = (x[rp.nodes] - centre) @ rotation.T + target
        spin = np.array((0.0, 0.0, np.sin(0.5 * turn), np.cos(0.5 * turn)))
        pose = np.concatenate(
            (rotation @ (body_q[rp.body, :3] - centre) + target, _quat_mul(spin, body_q[rp.body, 3:]))
        )
        moved[rp.part] = {"translation_m": (pose[:3] - body_q[rp.body, :3]).tolist(), "turn_rad": turn}
        body_q[rp.body], body_qd[rp.body] = pose, 0.0
        for j in np.flatnonzero((children == rp.body) & (kinds == int(newton.JointType.FREE))):
            joint_q[starts[j] : starts[j] + 7] = pose
            joint_qd[dof_starts[j] : dof_starts[j] + 6] = 0.0
    q[particles] = new
    qd = pipeline.state_0.particle_qd.numpy()
    qd[particles] = 0.0
    for state in (pipeline.state_0, pipeline.state_1):
        state.particle_q.assign(q)
        state.particle_qd.assign(qd)
        state.body_q.assign(body_q)
        state.body_qd.assign(body_qd)
        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)
    # The graph is captured on the first step; any captured earlier is dropped so it re-reads nothing stale.
    pipeline.invalidate_graph()
    return {"moved_particles": int(move.sum()), "rigid_parts": moved}


def _quat_mul(a, b) -> np.ndarray:
    """``a * b`` for xyzw quaternions."""
    (ax, ay, az, aw), (bx, by, bz, bw) = a, b
    return np.array(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def set_part_radius(sim, entity, labels, radius: float) -> int:
    """Give the nodes of the tets labelled ``labels`` their own contact radius [m], after the build.

    Newton reads ``model.particle_radius`` per particle in every particle-shape contact (broad and
    narrow phase, mesh contacts and VBD's contact response); only bounds read the maximum, which
    a thinner part leaves unchanged. Self-contact uses the solver's own radius. Written in place, so
    a captured CUDA graph keeps reading it. Returns how many particles changed.
    """
    nodes = np.unique(entity.source_tets[np.isin(entity.tet_part_labels, labels)])
    if float(radius) > float(sim.model.particle_max_radius):
        raise ValueError("a part radius above the model's maximum would outgrow its contact bounds")
    radii = sim.model.particle_radius.numpy()
    radii[np.asarray(entity.particles)[nodes]] = float(radius)
    sim.model.particle_radius.assign(radii)
    return len(nodes)
