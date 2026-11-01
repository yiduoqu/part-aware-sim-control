# SPDX-License-Identifier: Apache-2.0
"""Pick up and transport the labelled Clawhauser volume, then release its ring.

Run on a CUDA node with Newton 1.8 development and RoboSimStudio on PYTHONPATH.
The two kinematic fingers use only collision contact; no particle attachment or
direct object motion is added. Outputs are written under ``outputs/clawhauser``.
Use ``--lower-before-release`` to descend before the same release sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import newton
import numpy as np
from gripper import ScriptedGripperSpec

import robosimstudio as rss
from robosimstudio_materials.heterogeneous import HeterogeneousSoftSpec, load_labelled_tets

ASSET = Path(__file__).resolve().parent / "assets" / "clawhauser"
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "clawhauser"
CONTROL_HZ = 30
SUBSTEPS = 80
DURATION = 12.0
TABLE_Z = 0.75
# RSS positions are relative to the bench's work-surface frame, whose origin
# is world x=0.6 m. These values place the toy at world x=0.47 and 0.65 m.
SOURCE_X = -0.13
TARGET_X = 0.05
SOURCE_HASHES = {
    "final_mesh.mesh": "df3d1505d96d36fbfa76291e7ab94269a30500ba5e4b87516d0bf8f57f0c6458",
    "heterogeneous_params.npz": "7cfc8ca3cc7c08d261aa90b08832230801fec1ac62848d0e9239d9e2f318e8c6",
    "inferred_params.json": "45aacd698f2b38a6fd8bf13b46cc9c33497fc90c6c1bdf088ed12d915ec3d992",
    "appearance/visual_to_physics.npz": "4b015d1141794211e4e2c6e2b39fadef6762a73ffe2d76b921e47d5676f3d6c4",
}


def build_scene(*, lower_before_release: float = 0.0):
    for relative, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((ASSET / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Clawhauser source asset changed: {relative}")
    vertices, tets, labels, elastic, density = load_labelled_tets(
        ASSET / "final_mesh.mesh", ASSET / "heterogeneous_params.npz"
    )
    if (len(vertices), len(tets)) != (1840, 6850) or set(np.unique(labels)) != {0, 1, 2}:
        raise ValueError("Clawhauser source geometry or labels changed")
    if not np.array_equal(np.unique(elastic[:, 0]), [5e5, 5e6]) or not np.array_equal(np.unique(density), [300, 1100]):
        raise ValueError("Clawhauser source material estimates changed")

    # A flat pose makes the loop accessible and the plush body stable on the
    # table. The asset's local +y points toward its loop; rotate 180° about x.
    pose = (SOURCE_X, 0.0, float(vertices[:, 2].max() + 0.002))
    rotation = (1.0, 0.0, 0.0, 0.0)
    with np.load(ASSET / "appearance" / "visual_to_physics.npz") as binding:
        faces = binding["visual_faces"][binding["boundary_face_part_ids"] == 2]
        ring_surface = np.unique(binding["physics_vertex_indices"][faces])
    grasp_ids = ring_surface[vertices[ring_surface, 1] > 0.095]
    if len(grasp_ids) < 8:
        raise ValueError("Clawhauser loop grasp patch is missing")
    local_center = vertices[grasp_ids].mean(axis=0)
    grasp_center = (SOURCE_X + local_center[0], -local_center[1], pose[2] - local_center[2])
    half_span = np.max(np.abs(vertices[grasp_ids, 0] - local_center[0]))

    vbd = rss.backend(
        "vbd",
        iterations=48,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=True,
        rigid_body_particle_contact_buffer_size=4096,
    )
    physics = rss.ArrangementSpec(
        control_hz=CONTROL_HZ,
        substeps=SUBSTEPS,
        use_cuda_graph=True,
        backends=(("soft", vbd), ("rigid", vbd), ("*", vbd)),
        collision=rss.CollisionSpec(
            soft_contact_margin=0.002,
            full_surface_contact=False,
            contact_ke=100000.0,
            contact_kd=25.0,
            broad_phase="nxn",
        ),
    )
    scene = rss.Scene(cell="bench_075", physics=physics, name="clawhauser_airborne_release")
    # Omit decorative zone slabs: this RSS revision's collide=False still
    # allows particle contacts. The bench is the only intended support.
    obj = scene.add(
        HeterogeneousSoftSpec(
            name="clawhauser",
            mesh_path=str(ASSET / "final_mesh.mesh"),
            params_path=str(ASSET / "heterogeneous_params.npz"),
            materials_path=str(ASSET / "inferred_params.json"),
            pose=pose,
            quat=rotation,
            damping_time=0.0005,
            contact_ke=100000.0,
        )
    )
    grip = scene.add(
        ScriptedGripperSpec(
            center=grasp_center,
            half_gap=float(half_span + 0.012),
            # Pad half-thickness is 3 mm: allow approximately 1 mm of
            # indentation rather than squeezing the whole loop to 14 mm.
            closed_half_gap=float(half_span + 0.002),
            settle_s=1.0,
            airborne_release=True,
            release_lower_z=lower_before_release,
            travel_x=TARGET_X - SOURCE_X,
            control_hz=CONTROL_HZ,
            substeps=SUBSTEPS,
            duration=DURATION + (2.0 if lower_before_release else 0.0),
        )
    )
    camera = rss.CameraSpec(
        name="overview",
        width=640,
        height=480,
        eye=(0.90, -0.50, 1.30),
        look_at=(0.57, -0.04, 0.95),
        fov_y=50.0,
    )
    scene.add(camera)
    sim = scene.build(device="cuda:0", warmup=True, use_cuda_graph=True)
    entity = obj.entity
    if type(sim.pipeline.program.solver).__name__ != "SolverVBD":
        raise RuntimeError(f"expected Newton VBD, got {type(sim.pipeline.program.solver).__name__}")
    if sim.model.tet_count != len(tets) or len(entity.tet_part_labels) != len(tets):
        raise RuntimeError("RSS lost Clawhauser tetrahedra or part labels")
    if sim.model.attachment_particle_body_count:
        raise RuntimeError("the contact grasp scene must not contain particle attachments")
    model_tets = sim.model.tet_indices.numpy()[entity.tet_range[0] : entity.tet_range[1]]
    if not np.array_equal(model_tets - entity.particles[0], tets):
        raise RuntimeError("Newton changed the source tetrahedron order")
    young, nu = entity.source_elastic.T
    source_mu = young / (2.0 * (1.0 + nu))
    source_lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    model_elastic = sim.model.tet_materials.numpy()[entity.tet_range[0] : entity.tet_range[1], :2]
    if not np.allclose(model_elastic, np.column_stack((source_mu, source_lambda)), rtol=1e-5):
        raise RuntimeError("Newton FEM elasticity differs from the source estimates")
    if not np.allclose(sim.model.particle_mass.numpy()[entity.particles], entity.source_mass, rtol=1e-5):
        raise RuntimeError("Newton nodal mass differs from source per-tet densities")
    return sim, obj, grip, camera, grasp_ids, labels


def run(out: Path, *, steps: int | None = None, render_every: int = 0, lower_before_release: float = 0.0):
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, grip, camera, ring_ids, labels = build_scene(lower_before_release=lower_before_release)
    duration = grip.entity.spec.duration
    if steps is None:
        steps = round(duration * CONTROL_HZ)
    task = "pickup_transport_lowered_release" if lower_before_release else "pickup_transport_airborne_release"
    entity = obj.entity
    particle_ids = entity.particles
    patch_ids = particle_ids[ring_ids]
    # Contacts may move around the loop after pickup; the original tip patch
    # is only for initial alignment, not the definition of the whole ring.
    ring_ids = particle_ids[np.unique(entity.source_tets[labels == 2])]
    body_ids = particle_ids[np.unique(entity.source_tets[labels == 1])]
    q0 = sim.pipeline.state_0.particle_q.numpy()[particle_ids]
    rest_det = np.linalg.det(q0[entity.source_tets[:, 1:]] - q0[entity.source_tets[:, :1]])
    if np.any(rest_det <= 0):
        raise RuntimeError("source tetrahedra were not positively oriented after build")
    initial_body = q0[np.unique(entity.source_tets[labels == 1])].mean(axis=0)
    positions, body_poses, diagnostics = [], [], []

    videos = None
    video_error = None
    if render_every:
        from robosimstudio.core.render.video import VideoSet

        videos = VideoSet(str(out), fps=max(1, CONTROL_HZ // render_every))
        videos.__enter__()
    try:
        for frame in range(steps):
            sim.run(1).sync()
            q = sim.pipeline.state_0.particle_q.numpy()
            if frame + 1 == round(grip.entity.spec.settle_s * CONTROL_HZ):
                grip.entity.fit_grasp(q[patch_ids])
            body_q = sim.pipeline.state_0.body_q.numpy()
            owned = q[particle_ids]
            if not np.isfinite(owned).all():
                raise RuntimeError(f"non-finite Clawhauser state at frame {frame}")
            det = np.linalg.det(owned[entity.source_tets[:, 1:]] - owned[entity.source_tets[:, :1]])
            contacts = sim.pipeline.contacts
            count = int(contacts.soft_contact_count.numpy()[0])
            contact_shapes = contacts.soft_contact_shape.numpy()[: min(count, contacts.soft_contact_max)]
            finger_mask = np.isin(contact_shapes, grip.entity.finger_shapes)
            finger_contacts = int(np.count_nonzero(finger_mask))
            contact_particles = contacts.soft_contact_particle.numpy()[: len(contact_shapes)]
            ring_mask = np.isin(contact_particles, ring_ids)
            row = {
                "frame": frame + 1,
                "time_s": (frame + 1) / CONTROL_HZ,
                "ring_center_m": q[patch_ids].mean(axis=0).tolist(),
                "body_center_m": q[body_ids].mean(axis=0).tolist(),
                "center_of_mass_m": np.average(owned, weights=entity.source_mass, axis=0).tolist(),
                "center_of_mass_velocity_m_s": np.average(
                    sim.pipeline.state_0.particle_qd.numpy()[particle_ids], weights=entity.source_mass, axis=0
                ).tolist(),
                "body_translation_x_m": float(q[body_ids, 0].mean() - initial_body[0]),
                "minimum_z_m": float(owned[:, 2].min()),
                "minimum_tet_volume_ratio": float((det / rest_det).min()),
                "inverted_tetrahedra": int(np.count_nonzero(det <= 0)),
                "soft_contacts": count,
                "finger_contacts": finger_contacts,
                "finger_contacts_by_shape": [
                    int(np.count_nonzero(contact_shapes == shape)) for shape in grip.entity.finger_shapes
                ],
                "finger_contacts_by_side": [
                    int(np.count_nonzero(np.isin(contact_shapes, shapes))) for shapes in grip.entity.shapes_by_side
                ],
                "ring_contacts_by_side": [
                    int(np.count_nonzero(np.isin(contact_shapes, shapes) & ring_mask))
                    for shapes in grip.entity.shapes_by_side
                ],
                "grasp_patch_contacts": int(np.count_nonzero(finger_mask & np.isin(contact_particles, patch_ids))),
                "gripper_body_x_m": body_q[grip.entity.bodies, 0].tolist(),
                "gripper_body_y_m": body_q[grip.entity.bodies, 1].tolist(),
                "gripper_body_z_m": body_q[grip.entity.bodies, 2].tolist(),
            }
            diagnostics.append(row)
            positions.append(owned.astype(np.float32))
            body_poses.append(body_q.astype(np.float32))
            if videos is not None and frame % render_every == 0:
                try:
                    videos.append_frames({camera.name: sim.render(camera)})
                except Exception as exc:
                    video_error = repr(exc)
                    try:
                        videos.__exit__(type(exc), exc, exc.__traceback__)
                    except Exception:
                        pass
                    videos = None
            if frame % 15 == 0 or frame == steps - 1:
                print(json.dumps(row), flush=True)
            if frame + 1 == round(6.5 * CONTROL_HZ):
                transit = [r for r in diagnostics if 4.5 <= r["time_s"] <= 6.5]
                if any(r["inverted_tetrahedra"] for r in diagnostics) or not all(
                    min(r["ring_contacts_by_side"]) > 0 for r in transit
                ):
                    print("Pickup check failed; stopping before airborne release.", flush=True)
                    break
    finally:
        if videos is not None:
            try:
                videos.__exit__(None, None, None)
            except Exception as exc:
                video_error = repr(exc)

    np.savez_compressed(out / "states.npz", particle_q=np.asarray(positions), body_q=np.asarray(body_poses))
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    newton_root = Path(newton.__file__).resolve().parents[1]
    newton_commit = subprocess.check_output(["git", "-C", str(newton_root), "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "asset": "clawhauser",
        "task": task,
        "lower_before_release_m": lower_before_release,
        "actuation": "kinematic, uncapped, contact only",
        "newton_version": newton.__version__,
        "newton_source_commit": newton_commit,
        "solver": type(sim.pipeline.program.solver).__name__,
        "tetrahedra": int(sim.model.tet_count),
        "part_tet_counts": {str(i): int(np.count_nonzero(labels == i)) for i in np.unique(labels)},
        "source_young_pa": sorted(float(v) for v in np.unique(entity.source_elastic[:, 0])),
        "source_poisson": sorted(float(v) for v in np.unique(entity.source_elastic[:, 1])),
        "source_density_kg_m3": sorted(float(v) for v in np.unique(entity.source_density)),
        "global_soft_contact_mu": float(sim.model.soft_contact_mu),
        "grasp_attachments": int(sim.model.attachment_particle_body_count),
        "completed_steps": len(diagnostics),
        "video_error": video_error,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if diagnostics:
        transit = [row for row in diagnostics if 4.5 <= row["time_s"] <= 6.5]
        lifted = float(max((row["body_center_m"][2] - initial_body[2] for row in transit), default=0.0))
        clearance = float(max((row["minimum_z_m"] - TABLE_Z for row in transit), default=0.0))
        bilateral_ring_grasp = bool(transit and all(min(row["ring_contacts_by_side"]) > 0 for row in transit))
        opening_s = grip.entity.spec.settle_s + 6.0 + (2.0 if lower_before_release else 0.0)
        after_open = [row for row in diagnostics if row["time_s"] >= opening_s]
        impact = next(
            (
                row["time_s"]
                for row in diagnostics
                if row["time_s"] >= 6.5
                if row["soft_contacts"] > row["finger_contacts"] and row["minimum_z_m"] <= TABLE_Z + 0.004
            ),
            None,
        )
        # Only a genuinely contact-free airborne interval tests gravity. A
        # later floor impact is reported separately from finger separation.
        flight_runs, flight = [], []
        for row in after_open:
            if impact is not None and row["time_s"] >= impact:
                break
            if row["soft_contacts"] == 0 and row["minimum_z_m"] > TABLE_Z + 0.004:
                flight.append(row)
            elif flight:
                flight_runs.append(flight)
                flight = []
        if flight:
            flight_runs.append(flight)
        flight = max(flight_runs, key=len, default=[])
        gravity = sim.model.gravity.numpy()[0].astype(float)
        acceleration = None
        gravity_error = None
        fall = 0.0
        if len(flight) >= 3:
            velocities = np.asarray([row["center_of_mass_velocity_m_s"] for row in flight])
            acceleration = np.diff(velocities, axis=0).mean(axis=0) * CONTROL_HZ
            gravity_error = float(np.linalg.norm(acceleration - gravity) / np.linalg.norm(gravity))
            fall = float(flight[0]["center_of_mass_m"][2] - flight[-1]["center_of_mass_m"][2])
        tail = [row for row in diagnostics if row["time_s"] >= duration - 0.5]
        released = bool(tail and all(row["finger_contacts"] == 0 for row in tail))
        before_impact = [row for row in diagnostics if impact is None or row["time_s"] < impact]
        pre_impact_inverted = max((row["inverted_tetrahedra"] for row in before_impact), default=0)
        max_inverted = max(row["inverted_tetrahedra"] for row in diagnostics)
        airborne_success = bool(
            len(diagnostics) == steps
            and steps >= round(duration * CONTROL_HZ)
            and lifted > 0.08
            and clearance > 0.05
            and bilateral_ring_grasp
            and released
            and len(flight) >= 3
            and fall > 0.01
            and gravity_error is not None
            and gravity_error < 0.20
            and pre_impact_inverted == 0
        )
        # A near-table release can correctly have no airborne interval. Keep
        # its motion result separate from both free-fall and mesh integrity.
        table_supported = bool(
            tail
            and all(
                TABLE_Z - 0.002 <= row["minimum_z_m"] <= TABLE_Z + 0.004
                and row["soft_contacts"] > row["finger_contacts"]
                for row in tail
            )
        )
        body_drift = (
            float(
                np.linalg.norm(
                    np.asarray([row["body_center_m"] for row in tail]) - tail[0]["body_center_m"], axis=1
                ).max()
            )
            if tail
            else None
        )
        # The final uninterrupted zero-contact interval is sustained release;
        # a brief earlier gap followed by recontact must not establish its time.
        last_contact = max((i for i, row in enumerate(after_open) if row["finger_contacts"]), default=-1)
        first_clear = (
            after_open[last_contact + 1]["time_s"] if released and last_contact + 1 < len(after_open) else None
        )
        lowered_success = bool(
            lower_before_release
            and len(diagnostics) == steps
            and steps >= round(duration * CONTROL_HZ)
            and lifted > 0.08
            and clearance > 0.05
            and bilateral_ring_grasp
            and released
            and table_supported
            and body_drift is not None
            and body_drift < 0.02
            and pre_impact_inverted == 0
        )
        result = {
            "task": task,
            "transit_body_lift_m": lifted,
            "transit_clearance_m": clearance,
            "bilateral_ring_grasp": bilateral_ring_grasp,
            "released": released,
            "first_finger_free_s": first_clear,
            "table_contact_before_release": impact is not None and first_clear is not None and impact < first_clear,
            "table_supported": table_supported,
            "final_half_second_body_drift_m": body_drift,
            "target_error_xy_m": float(
                np.linalg.norm(
                    np.asarray(diagnostics[-1]["body_center_m"][:2]) - [initial_body[0] + TARGET_X - SOURCE_X, -0.10]
                )
            ),
            "contact_free_flight_samples": len(flight),
            "contact_free_flight_window_s": [flight[0]["time_s"], flight[-1]["time_s"]] if flight else None,
            "contact_free_com_fall_m": fall,
            "mean_contact_free_com_acceleration_m_s2": acceleration.tolist() if acceleration is not None else None,
            "model_gravity_m_s2": gravity.tolist(),
            "gravity_relative_error": gravity_error,
            "first_table_contact_after_transport_s": impact,
            "max_pre_impact_inverted_tetrahedra": pre_impact_inverted,
            "max_pre_release_inverted_tetrahedra": max(
                (
                    row["inverted_tetrahedra"]
                    for row in diagnostics
                    if first_clear is None or row["time_s"] < first_clear
                ),
                default=0,
            ),
            "max_inverted_tetrahedra": max_inverted,
            "minimum_tet_volume_ratio": min(row["minimum_tet_volume_ratio"] for row in diagnostics),
            "airborne_release_success": airborne_success,
            "lowered_release_success": lowered_success,
            "all_episode_mesh_integrity_pass": max_inverted == 0,
            "success": (lowered_success if lower_before_release else airborne_success) and max_inverted == 0,
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--lower-before-release", type=float, default=0.0, metavar="METRES")
    args = parser.parse_args()
    if not 0.0 <= args.lower_before_release < 0.4:
        parser.error("--lower-before-release must be between 0 and 0.4 metres")
    run(args.out, steps=args.steps, render_every=args.render_every, lower_before_release=args.lower_before_release)


if __name__ == "__main__":
    main()
