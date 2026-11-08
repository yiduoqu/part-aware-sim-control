# SPDX-License-Identifier: Apache-2.0
"""Simulate parts of a labelled soft mesh as rigid bodies, tied to the rest by attachments.

Where :mod:`rigid_projection` snaps a part's nodes onto a rigid pose after each
solver step, here each part in ``rigid_parts`` *is* a rigid body that
``SolverVBD`` integrates (AVBD) in the same solve as the soft mesh:

1. one free body per part takes the part's mass, centre of mass and inertia,
   computed exactly from its tets at their density;
2. every node of the part is tied to its body at its rest position by a
   particle-body attachment (``ModelBuilder.add_attachment_particle_body``), a
   spring whose force acts on both ends. The *seam* nodes, which the part shares
   with the rest of the mesh, are the joint between the two. The *skin* nodes,
   which belong to the part alone, ride on the body and carry its contacts: pads,
   rods and the bench still touch particles, so sticky matching, slide feedback
   and every judge that reads the part's particles work unchanged;
3. the part's own tets are scaled to ``part_stiffness_scale`` of their stiffness
   (not zero: Newton's Neo-Hookean divides by the Lame parameter), so they no
   longer resist the body; the skin keeps ``skin_mass_fraction`` of its nodal
   mass and the body carries the rest, so the total mass is unchanged.

Two rigid parts may not share a node that touches no soft tet. The bodies have no
collision shapes. With an arm, the one-way coupler gives the bodies to VBD and
holds the arm's where MuJoCo puts them (``SolverOneWay``'s ``soft_bodies``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from robosimstudio.scene.registry import register_entity
from robosimstudio_materials.heterogeneous import EntityHeterogeneousSoft, HeterogeneousSoftSpec

__all__ = ["ENTITY", "EntityRigidParts", "RigidPart", "RigidPartSoftSpec", "frame_record", "tet_mass_properties"]

ENTITY = "heterogeneous_soft_rigid_parts"


@dataclass(frozen=True, kw_only=True)
class RigidPartSoftSpec(HeterogeneousSoftSpec):
    """A :class:`HeterogeneousSoftSpec` whose parts ``rigid_parts`` are built as rigid bodies."""

    entity: str = ENTITY
    rigid_parts: tuple[int, ...] = ()
    """The part labels simulated as rigid bodies, one body each."""
    attach_stiffness: float = 1.0e4
    """Each node-body attachment's stiffness [N/m]."""
    attach_damping: float = 0.0
    """Each node-body attachment's damping [N·s/m]."""
    skin_mass_fraction: float = 0.1
    """The share of its nodal mass a part-only node keeps; the body carries the rest."""
    part_stiffness_scale: float = 1.0e-3
    """A rigid part's tets keep this fraction of their stiffness and damping."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.rigid_parts or len(set(self.rigid_parts)) != len(self.rigid_parts):
            raise ValueError("rigid_parts must name at least one part, each once")
        if self.attach_stiffness <= 0 or self.attach_damping < 0:
            raise ValueError("attachment stiffness must be positive and damping non-negative")
        if not 0 < self.skin_mass_fraction < 1 or not 0 < self.part_stiffness_scale <= 1:
            raise ValueError("skin_mass_fraction must be in (0, 1) and part_stiffness_scale in (0, 1]")


@dataclass(frozen=True)
class RigidPart:
    """One part built as a body: its body index and the nodes tied to it (local particle ids)."""

    part: int
    body: int
    nodes: np.ndarray
    seam: np.ndarray
    """Mask over ``nodes``: shared with a soft tet."""
    body_point: np.ndarray
    """Each node's anchor in the body frame [m]."""
    body_mass: float
    part_mass: float


def tet_mass_properties(x: np.ndarray, tets: np.ndarray, density: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, centre of mass and inertia about it of uniform-density tets at positions ``x``."""
    corners = x[tets]  # (n, 4, 3)
    volume = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.0
    mass = volume * density
    total = float(mass.sum())
    com = (mass[:, None] * corners.mean(axis=1)).sum(axis=0) / total
    r = corners - com
    s = r.sum(axis=1)
    # A uniform tet's second moment: m/20 (sum_i r_i r_i^T + s s^T), s = sum_i r_i.
    second = (mass[:, None, None] / 20.0) * (np.einsum("nij,nik->njk", r, r) + np.einsum("nj,nk->njk", s, s))
    second = second.sum(axis=0)
    return total, com, np.trace(second) * np.eye(3) - second


@register_entity(ENTITY)
class EntityRigidParts(EntityHeterogeneousSoft):
    """A labelled soft body with some parts as rigid bodies; see the module docstring."""

    spec: RigidPartSoftSpec

    def build(self, builder, episode) -> None:
        super().build(builder, episode)
        spec = self.spec
        tets, labels = self.source_tets, self.tet_part_labels
        n = len(self.source_vertices)
        start = int(builder.particle_count) - n
        x = np.array([[float(c) for c in p] for p in builder.particle_q[start : start + n]], dtype=np.float64)
        rigid = np.isin(labels, spec.rigid_parts)
        volume = np.abs(np.linalg.det(x[tets[:, 1:]] - x[tets[:, :1]])) / 6.0

        def nodal(per_tet: np.ndarray) -> np.ndarray:
            out = np.zeros(n, dtype=np.float64)
            np.add.at(out, tets.ravel(), np.repeat(per_tet * volume / 4.0, 4))
            return out

        # Every node keeps its share of the soft tets; a part-only (skin) node also keeps
        # skin_mass_fraction of its share of its part's.
        masses = nodal(np.where(rigid, 0.0, self.density))
        soft_nodes = np.unique(tets[~rigid])
        elastic, damping = self.elastic.copy(), self.damping.copy()
        scale = spec.part_stiffness_scale
        t0 = self.tet_range[0]
        self.rigid_parts = []
        for part in spec.rigid_parts:
            rows = labels == part
            if not rows.any():
                raise ValueError(f"{spec.name}: no tets in part {part}")
            nodes = np.unique(tets[rows])
            seam = np.isin(nodes, np.unique(tets[~rows]))
            if np.any(seam & ~np.isin(nodes, soft_nodes)):
                raise ValueError(f"{spec.name}: rigid part {part} shares nodes with another rigid part only")
            part_mass, com, inertia = tet_mass_properties(x, tets[rows], self.density[rows])
            skin = nodes[~seam]
            masses[skin] = spec.skin_mass_fraction * nodal(np.where(rows, self.density, 0.0))[skin]
            body_mass = float(part_mass - masses[skin].sum())
            # The skin's share comes off the body; its inertia goes with it in proportion.
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*(float(c) for c in com)), wp.quat_identity()),
                mass=body_mass,
                inertia=wp.mat33(*(float(c) for c in (inertia * (body_mass / part_mass)).ravel())),
                lock_inertia=True,
                label=f"{spec.name}/part{part}_body",
            )
            # The part's tets stop resisting the body: scale their stiffness and damping.
            for t in np.flatnonzero(rows):
                mu, lam, damp = builder.tet_materials[t0 + t]
                builder.tet_materials[t0 + t] = (mu * scale, lam * scale, damp * scale)
            elastic[rows, 0] *= scale
            damping[rows] *= scale
            local = x[nodes] - com
            for node, point in zip(nodes, local, strict=True):
                builder.add_attachment_particle_body(
                    start + int(node),
                    body,
                    body_point=wp.vec3(*(float(c) for c in point)),
                    stiffness=float(spec.attach_stiffness),
                    damping=float(spec.attach_damping),
                )
            self.rigid_parts.append(RigidPart(part, int(body), nodes, seam, local, body_mass, part_mass))
        if np.any(masses <= 0):
            raise RuntimeError(f"{spec.name}: a node would have no mass")
        builder.particle_mass[start : start + n] = masses.tolist()
        # What the build checks compare Newton's arrays with.
        self.elastic, self.damping, self.mass = elastic, damping, masses


def frame_record(entity: EntityRigidParts, owned: np.ndarray, body_q: np.ndarray) -> dict:
    """Each rigid part's body pose and how far its nodes sit from their anchors, for a per-frame record.

    ``owned`` is the entity's particle positions, ``body_q`` every body's ``(p, q)`` row.
    """
    record = {}
    for rp in entity.rigid_parts:
        pose = body_q[rp.body]
        p, q = pose[:3].astype(np.float64), pose[3:].astype(np.float64)  # q is (x, y, z, w)
        v, w = q[:3], q[3]
        t = 2.0 * np.cross(v, rp.body_point)
        gap = np.linalg.norm(owned[rp.nodes] - (p + rp.body_point + w * t + np.cross(v, t)), axis=1)
        record[f"part{rp.part}_body"] = {
            "position_m": p.tolist(),
            "quat_xyzw": q.tolist(),
            "seam_stretch_max_m": float(gap[rp.seam].max()) if rp.seam.any() else 0.0,
            "skin_stretch_max_m": float(gap[~rp.seam].max()) if (~rp.seam).any() else 0.0,
        }
    return record
