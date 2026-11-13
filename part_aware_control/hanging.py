"""Ring/rod observations and the hanging task's measured success gate."""

from __future__ import annotations


def threading_geometry(contour, rod_origin):
    """Winding of a closed material hole contour around the rod's +Y axis.

    The contour keeps its source ordering through deformation. Together with
    all its points lying inboard of the finite rod tip, winding distinguishes
    an axis through the hole from an exterior contact/drape.
    """
    import numpy as np

    delta = np.asarray(contour) - np.asarray(rod_origin)
    projected = delta[:, [0, 2]]
    following = np.roll(projected, -1, axis=0)
    cross = projected[:, 0] * following[:, 1] - projected[:, 1] * following[:, 0]
    angles = np.arctan2(cross, (projected * following).sum(axis=1))
    return {
        "rod_axis_winding": float(angles.sum() / (2 * np.pi)),
        "minimum_hole_engagement_m": float(delta[:, 1].min()),
        "maximum_hole_engagement_m": float(delta[:, 1].max()),
    }


class HangingState:
    def __init__(self, sim, obj, rod, ring, geometry):
        import numpy as np

        from .tasks import example

        self.sim, self.obj, self.ring = sim, obj, ring
        self.rod_shapes = np.asarray(rod.shapes, dtype=np.int32)
        self.loop_particles = obj.entity.particles[ring["loop_ids"]]
        arms = sim.handles.arms[example("arm_gripper").ARM]
        self.finger_sides = [
            np.asarray([int(shape) for _, shapes in side for shape in shapes], dtype=np.int32)
            for side in arms.finger_groups
        ]
        self.finger_shapes = np.unique(np.concatenate(self.finger_sides))
        self.shape_ke = sim.model.shape_material_ke.numpy()
        self.rod_origin = np.asarray(geometry["rod_free_end_world_m"])
        self.weight = float(obj.entity.mass.sum() * 9.81)
        vertices, _, _ = example("clawhauser_scene").load_toy()
        inner = ring["inner_ids"]
        local = vertices[inner, :2] - np.asarray(ring["hole_centre_local"][:2])
        bins = np.floor((np.arctan2(local[:, 1], local[:, 0]) + np.pi) / (2 * np.pi) * 72).astype(int) % 72
        self.contour_groups = [inner[bins == sector] for sector in range(72) if np.any(bins == sector)]

    def read(self, all_points, points):
        import numpy as np

        from .tasks import example

        hole_points = points[self.ring["inner_ids"]]
        hole = hole_points.mean(axis=0)
        centered = hole_points - hole
        normal = np.linalg.eigh(centered.T @ centered)[1][:, 0]
        if normal[1] < 0:
            normal = -normal
        contacts = self.sim.pipeline.contacts
        count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
        shapes = contacts.soft_contact_shape.numpy()[:count]
        particles = contacts.soft_contact_particle.numpy()[:count]
        depth = example("mesh_quality").penetration(
            all_points, self.sim.pipeline.state_0.body_q.numpy(), self.sim.model, contacts
        )
        touching = depth > 0
        on_rod = np.isin(shapes, self.rod_shapes)
        ring_contact = on_rod & np.isin(particles, self.loop_particles)
        # Match Hanson's readback: mean penalty stiffness times real depth,
        # projected onto world +Z. A proximity record is not a contact/load.
        load = 0.5 * (float(self.sim.model.soft_contact_ke) + self.shape_ke[shapes]) * depth
        support = load * contacts.soft_contact_normal.numpy()[:count, 2]
        finger = np.isin(shapes, self.finger_shapes)
        finger_contacts = int(np.count_nonzero(finger & touching))
        if not np.isfinite(support).all():
            raise RuntimeError("nonfinite contact support measurement")
        shape_forces = getattr(self.sim.pipeline.program.solver, "shape_contact_forces", None)
        squeeze = None
        if shape_forces is not None:
            _, squeeze = example("arm_gripper").pad_forces(self.sim, shape_forces(), self.finger_sides)
        contour = np.asarray([points[group].mean(axis=0) for group in self.contour_groups])
        return {
            **threading_geometry(contour, self.rod_origin),
            "loop_center_world_m": points[self.ring["loop_ids"]].mean(axis=0).tolist(),
            "hole_center_world_m": hole.tolist(),
            "hole_axis_world": normal.tolist(),
            "rod_engagement_m": float(hole[1] - self.rod_origin[1]),
            "rod_loop_contacts": int(np.count_nonzero(ring_contact & touching)),
            "rod_support_n": float(support[on_rod].sum()),
            "rod_loop_support_n": float(support[ring_contact].sum()),
            "bench_and_rod_support_n": float(support[~finger].sum()),
            "finger_contacts": finger_contacts,
            "finger_squeeze_n": squeeze,
            "max_contact_penetration_m": float(depth.max()) if depth.size else 0.0,
        }


def measure_hanging(task, samples, geometry, quality):
    import numpy as np

    name = task.name
    end = samples[-1]["time_s"]
    tail = [s for s in samples if s["time_s"] >= end - task.settle_window_seconds]
    positions = np.asarray([s["objects"][name]["position"] for s in tail])
    times = np.asarray([s["time_s"] for s in tail])
    drift = float(np.linalg.norm(positions - positions[-1], axis=1).max())
    speed = (
        float((np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)).max())
        if len(tail) > 1
        else float("inf")
    )
    weight = geometry["weight_n"]
    support = float(np.mean([s["hanging"]["rod_loop_support_n"] for s in tail]))
    clearance = min(s["objects"][name]["minimum_z"] for s in tail) - geometry["rest_surface_z_world_m"]
    engagements = [s["hanging"]["rod_engagement_m"] for s in tail]
    rod_length = geometry["rod_length_m"]
    engaged = all(
        s["hanging"]["rod_loop_contacts"] > 0
        and abs(s["hanging"]["rod_axis_winding"]) >= 0.5
        and s["hanging"]["minimum_hole_engagement_m"] >= task.minimum_rod_engagement
        and s["hanging"]["maximum_hole_engagement_m"] <= rod_length
        and abs(s["hanging"]["loop_center_world_m"][0] - geometry["rod_free_end_world_m"][0]) < 0.010
        for s in tail
    )
    lifted = max(s["objects"][name]["minimum_z"] for s in samples) - samples[0]["objects"][name]["minimum_z"]
    penetration = max(s["hanging"]["max_contact_penetration_m"] for s in samples)
    squeeze = max((max(s["hanging"]["finger_squeeze_n"] or [0]) for s in samples), default=0.0)
    force_limit = geometry["finger_effort_limit_n"] * (1 + geometry["finger_force_tolerance"])
    valid = bool(
        not sum(quality["inverted_tet_substeps_by_part"])
        and not sum(quality["nonfinite_tet_substeps_by_part"])
        and min(quality["minimum_J_by_part"]) > 0
        and penetration <= task.max_contact_penetration
        and squeeze <= force_limit
    )
    gates = {
        "valid": valid,
        "lifted": lifted >= task.minimum_lift,
        "placed": bool(engaged),
        "released": all(s["hanging"]["finger_contacts"] == 0 and s["gripper"]["width"] >= 0.060 for s in tail),
        "supported": bool(
            support >= task.minimum_rod_support_fraction * weight and clearance >= task.minimum_hang_clearance
        ),
        "settled": bool(
            times[-1] - times[0] >= 0.45 and drift <= task.max_settled_drift and speed <= task.max_settled_speed
        ),
    }
    return {
        "success": all(gates.values()),
        **gates,
        "lifted_m": lifted,
        "rod_loop_support_n": support,
        "rod_loop_support_fraction": support / weight,
        "minimum_final_engagement_m": min(engagements),
        "minimum_final_hole_engagement_m": min(s["hanging"]["minimum_hole_engagement_m"] for s in tail),
        "minimum_final_absolute_winding": min(abs(s["hanging"]["rod_axis_winding"]) for s in tail),
        "final_clearance_m": clearance,
        "final_position_world_m": positions[-1].tolist(),
        "final_drift_m": drift,
        "final_peak_speed_m_s": speed,
        "mesh_quality": quality,
        "max_contact_penetration_m": penetration,
        "max_finger_squeeze_n": squeeze,
        "reason": ", ".join(k for k, v in gates.items() if not v) or "all measured hanging gates passed",
    }
