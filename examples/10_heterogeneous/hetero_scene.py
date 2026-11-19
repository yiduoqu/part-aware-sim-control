# SPDX-License-Identifier: Apache-2.0
"""Labelled heterogeneous objects on the bench: the layer the cable and tool scenes share.

Every asset is a labelled tet volume (:func:`load_labelled_tets`) whose rigid groups (``asset.json``)
become rigid bodies (:mod:`rigid_part_body`); every other label is soft. This holds the asset record
(:class:`HeteroAsset`), loading, the object's spec and material levers, friction, the per-frame episode
loop and the run's config. :mod:`cable_scene` and :mod:`tool_scene` add their families' assets and helpers.
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
class HeteroAsset:
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


def _members(group) -> tuple[int, ...]:
    return tuple(group) if isinstance(group, tuple) else (group,)


def load(asset: HeteroAsset):
    """The asset's vertices, tets and labels, after checking its files against ``SOURCE.md``."""
    listed = re.findall(r"- `([^`]+)`: `([0-9a-f]{64})`", (asset.path / "SOURCE.md").read_text())
    for relative, expected in listed:
        if hashlib.sha256((asset.path / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{asset.folder} source asset changed: {relative}")
    vertices, tets, labels, _elastic, _density = load_labelled_tets(
        asset.path / "final_mesh.mesh", asset.path / "heterogeneous_params.npz"
    )
    return vertices, tets, labels


def soft_labels(asset: HeteroAsset, labels: np.ndarray) -> np.ndarray:
    rigid = [p for g in asset.groups() for p in _members(g)]
    return np.setdiff1d(np.unique(labels), rigid)


def spec(
    asset: HeteroAsset,
    vertices,
    tets,
    labels,
    *,
    name: str,
    pose,
    settings,
    mu: float,
    overrides=(),
    scale: float = 1.0,
    part_stiffness_scale: float = 1.0e-3,
    attach_stiffness: float = 1.0e4,
):
    """The object's :class:`rigid_part_body.RigidPartSoftSpec`, with the asset's levers applied, its mesh scaled ``scale``.

    ``part_stiffness_scale`` and ``attach_stiffness`` are the rigid parts' tet stiffness fraction and node ties [N/m].
    """
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
        attach_stiffness=attach_stiffness,
        part_stiffness_scale=part_stiffness_scale,
        name=name,
        mesh_path=str(asset.path / "final_mesh.mesh"),
        params_path=str(asset.path / "heterogeneous_params.npz"),
        materials_path=str(asset.path / "inferred_params.json"),
        pose=pose,
        quat=asset.quat,
        scale=scale,
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
    objects=None,
    extra=None,
):
    """Step ``steps`` frames, writing ``trajectory.npz``, ``diagnostics.json`` and ``overview.mp4``.

    ``follow(frame)`` drives the arm before each frame and returns the route's frames so far;
    ``task_row(frame, local, body_q)`` adds the task's own fields to each frame's row (``local``: the
    object's particles in the bench frame). ``soft`` are the soft labels, whose strain is measured.
    ``stop(row)``, when given, ends the run early once it returns true (a blown-up state).
    ``objects`` ({name: entity}): the scene's other soft bodies, each recorded to ``trajectory_<name>.npz``.
    ``extra``: arrays the task adds to ``trajectory.npz``, read when it is saved.
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
    others = {name: (other, Trajectory(sim, other)) for name, other in (objects or {}).items()}
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
            for other, recording in others.values():
                recording.add(row["time_s"], q[other.particles], body_q)
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
    trajectory.save(out / "trajectory.npz", extra)
    for name, (_, recording) in others.items():
        recording.save(out / f"trajectory_{name}.npz")
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows, elapsed, video_error


def sim_surface():
    """The bench's top in the world frame [m]."""
    import arm_gripper

    import robosimstudio as rss

    return rss.get_cell(arm_gripper.CELL).surface_frame[:3]


def with_levers(asset: HeteroAsset, args) -> HeteroAsset:
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


def config(args, asset: HeteroAsset, settings) -> dict:
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
