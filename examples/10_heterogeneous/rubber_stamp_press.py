# SPDX-License-Identifier: Apache-2.0
"""Press the wooden-handle rubber stamp onto a sheet of paper and read the mark its die leaves.

The stamp (``rubber_stamp``) is a turned handle on a block, a foam cushion under the block and a
rubber die under that, its raised border and three bars down. Handle and block are one rigid
body (:mod:`rigid_part_body`); the cushion and the die are FEM, and nothing else holds the
stamp together or up: it stands on its own relief.

A kinematic plate comes down on the handle, pushes it ``--overpress`` further, holds, and lifts
off again. Nothing is attached to the plate. The run measures whether the stamp stands and stays
where it was put, how much of the relief is on the paper, how far the cushion and the die give
under the press and how much of that comes back, and what the mesh does. Every relief face whose
three nodes are on the paper inks it; the union over the run is ``imprint.png``.

The plate, its path, the ink and the build checks are :mod:`stamp_press`'s.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import arm_gripper
import newton
import numpy as np
import rigid_part_body
import stamp_press
from clawhauser_scene import Trajectory, arrangement, build_sim
from mesh_quality import MeshQuality
from stamp_press import INK_PIXEL, INK_RGB, PAPER_COLOR, PAPER_HALF, PHASES, PHYSICS, PLACE_GAP, STAMP_AT

import robosimstudio as rss
from robosimstudio_materials.heterogeneous import load_labelled_tets

ASSET = Path(__file__).resolve().parent / "assets" / "rubber_stamp"
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "rubber_stamp_press"
CELL = arm_gripper.CELL
HANDLE, MOUNT, FOAM, DIE = range(4)
"""The stamp's part labels. Handle + mount are its one rigid group."""
SOFT = (FOAM, DIE)
OVERPRESS = 0.001
"""How far the plate pushes the handle down once it is on it [m]: a third of the cushion's 3 mm."""
DURATION = PHASES["rest"][1]
TOUCH_TOLERANCE = stamp_press.TOUCH_TOLERANCE
INK_HALF = 0.036
"""The imprint image reaches this far each side of the stamp's centre [m]: the die is 60 mm long."""


def load_stamp():
    """The source ``(vertices, tets, labels, rigid groups, part names)``, after checking the files against ``SOURCE.md``."""
    listed = re.findall(r"- `([^`]+)`: `([0-9a-f]{64})`", (ASSET / "SOURCE.md").read_text(encoding="utf-8"))
    if not listed:
        raise ValueError("rubber_stamp: SOURCE.md lists no file hashes")
    for relative, expected in listed:
        if hashlib.sha256((ASSET / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"rubber_stamp source asset changed: {relative}")
    vertices, tets, labels, _elastic, _density = load_labelled_tets(
        ASSET / "final_mesh.mesh", ASSET / "heterogeneous_params.npz"
    )
    if set(np.unique(labels).tolist()) != set(range(4)):
        raise ValueError("rubber_stamp: expected part labels 0 to 3")
    meta = json.loads((ASSET / "asset.json").read_text(encoding="utf-8"))
    groups = tuple(tuple(g) if len(g) > 1 else g[0] for g in meta["rigid_groups"])
    parts = json.loads((ASSET / "inferred_params.json").read_text(encoding="utf-8"))["predictions"]
    names = tuple(part["part_name"] for part in sorted(parts, key=lambda part: part["part_index"]))
    return vertices, tets, labels, groups, names


def layout(vertices: np.ndarray, tets: np.ndarray, labels: np.ndarray) -> dict:
    """What the scene is placed and judged by, measured from the mesh (the mesh's own frame, z up)."""
    nodes = [np.unique(tets[labels == part]) for part in range(4)]
    base_z, top = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    die = nodes[DIE]
    tips = die[vertices[die, 2] <= base_z + 1.0e-7]
    faces = stamp_press.boundary_faces(tets)
    tip_faces = faces[np.isin(faces, tips).all(axis=1)]
    corners = vertices[tip_faces]
    area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    # The two bonded faces the layers are measured between: block on cushion, cushion on die.
    under_mount = np.intersect1d(nodes[MOUNT], nodes[FOAM])
    under_foam = np.intersect1d(nodes[FOAM], nodes[DIE])
    if not len(tips) or not len(under_mount) or not len(under_foam):
        raise ValueError("rubber_stamp: the relief or a bonded layer is missing")
    return {
        "nodes": nodes,
        "base_z": base_z,
        "top": top,
        "top_nodes": nodes[HANDLE][vertices[nodes[HANDLE], 2] >= top - 1.0e-7],
        "tips": tips,
        "tip_faces": tip_faces,
        "tip_face_area": area,
        "under_mount": under_mount,
        "under_foam": under_foam,
        "report": {
            "height_m": top - base_z,
            "tip_nodes": int(len(tips)),
            "tip_faces": int(len(tip_faces)),
            "relief_area_m2": float(area.sum()),
            "foam_thickness_m": float(vertices[under_mount, 2].mean() - vertices[under_foam, 2].mean()),
            "die_thickness_m": float(vertices[under_foam, 2].mean() - base_z),
        },
    }


def build_scene(*, solver_name: str = stamp_press.SOLVER, iterations: int = 160, mu: float | None = None):
    vertices, tets, labels, groups, names = load_stamp()
    geometry = layout(vertices, tets, labels)
    settings = PHYSICS
    physics = arrangement(
        settings, claw="claw", coarse=False, solver_name=solver_name, iterations=iterations, self_contact=False
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="rubber_stamp_press")
    paper_top = 2.0 * PAPER_HALF[2]
    scene.add(rss.slab(name="paper", half=PAPER_HALF, at=(STAMP_AT[0], STAMP_AT[1], PAPER_HALF[2]), color=PAPER_COLOR))
    # A node rests one particle radius over the paper; the stamp's origin is the middle of its printing face.
    pose = (STAMP_AT[0], STAMP_AT[1], paper_top + settings.particle_radius + PLACE_GAP - geometry["base_z"])
    obj = scene.add(
        stamp_press.StampSpec(
            rigid_parts=groups,
            attach_damping=stamp_press.ATTACH_DAMPING,
            name="rubber_stamp",
            mesh_path=str(ASSET / "final_mesh.mesh"),
            params_path=str(ASSET / "heterogeneous_params.npz"),
            materials_path=str(ASSET / "inferred_params.json"),
            pose=pose,
            contact_ke=settings.contact_ke,
            contact_kd=settings.contact_kd,
            contact_mu=settings.soft_mu,
            particle_radius=settings.particle_radius,
            damping_ratio=settings.damping_ratio,
        )
    )
    touch_z = pose[2] + geometry["top"] + settings.particle_radius
    half = stamp_press.PRESSER_HALF
    presser = scene.add(
        stamp_press.PresserSpec(center=(STAMP_AT[0], STAMP_AT[1], touch_z + stamp_press.APPROACH + half[2]))
    )
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    foot = surface + np.array((STAMP_AT[0], STAMP_AT[1], paper_top))
    for camera in (
        rss.CameraSpec(
            name="overview",
            width=960,
            height=540,
            eye=tuple(float(v) for v in np.add(foot, (0.17, -0.20, 0.13))),
            look_at=tuple(float(v) for v in np.add(foot, (0.0, 0.0, 0.030))),
            fov_y=30.0,
        ),
        # Level with the cushion, along the stamp's short side: the two soft layers fill the frame's width.
        rss.CameraSpec(
            name="side",
            width=960,
            height=540,
            eye=tuple(float(v) for v in np.add(foot, (0.0, -0.16, 0.012))),
            look_at=tuple(float(v) for v in np.add(foot, (0.0, 0.0, 0.012))),
            fov_y=24.0,
        ),
    ):
        scene.add(camera)
    sim = build_sim(scene, rigid_loop=False)
    entity = obj.entity
    report = dict(geometry["report"])
    report.update(stamp_press.check_stamp(sim, entity, tets, settings, mu=mu))
    young = entity.elastic[:, 0]
    report["asset"] = "rubber_stamp"
    report["source_stiffness_ratio"] = float(entity.source_elastic[:, 0].max() / entity.source_elastic[:, 0].min())
    # What Newton was given: the rigid group's tets keep part_stiffness_scale of their modulus.
    report["effective_young_pa_by_part"] = {names[p]: float(young[labels == p].max()) for p in range(4)}
    report["effective_stiffness_ratio"] = float(young.max() / young.min())
    report["paper_top_m"] = paper_top
    report["stamp_pose_m"] = [float(v) for v in pose]
    print(json.dumps({"geometry": report}), flush=True)
    return sim, obj, presser, geometry, report


def judge(rows: list[dict], rest: dict, report: dict, steps: int) -> dict:
    """What the press did, from the per-frame rows and the settled reference ``rest``."""
    planned = [row for row in rows if row["frame"] > rest["frame"]]
    hold = [row for row in planned if row["phase"] == "hold"]
    final = rows[-1]
    peak = max(planned, key=lambda row: row["squeeze_m"]) if planned else rest
    result = {
        "complete": steps >= round(DURATION * PHYSICS.control_hz),
        "finite": all(row["finite"] for row in rows),
        "max_tilt_deg": max(row["tilt_deg"] for row in rows),
        "max_yaw_deg": max(abs(row["yaw_deg"]) for row in rows),
        "max_slide_m": max(row["slide_m"] for row in rows),
        "rest_relief_on_paper": rest["relief_on_paper"],
        "rest_foam_squeeze_m": rest["foam_squeeze_m"],
        "rest_die_squeeze_m": rest["die_squeeze_m"],
        "hold_relief_on_paper": min((row["relief_on_paper"] for row in hold), default=0.0),
        "peak_press_travel_m": max((row["press_travel_m"] for row in planned), default=0.0),
        "peak_squeeze_m": peak["squeeze_m"],
        "peak_foam_squeeze_m": peak["foam_squeeze_m"],
        "peak_die_squeeze_m": peak["die_squeeze_m"],
        "peak_foam_strain": peak["foam_squeeze_m"] / report["foam_thickness_m"],
        "peak_bulge_m": max((row["bulge_m"] for row in planned), default=0.0),
        "final_squeeze_m": final["squeeze_m"],
        "final_relief_on_paper": final["relief_on_paper"],
        "final_plate_contacts": final["plate_contacts"],
        "max_rigid_node_stretch_m": max(row["rigid_node_stretch_m"] for row in rows),
        "max_inverted_soft_tets": max(row["inverted_soft_tets"] for row in rows),
        "max_inverted_rigid_tets": max(row["inverted_rigid_tets"] for row in rows),
    }
    result["checks"] = {
        "ran": result["complete"] and result["finite"],
        "stands": result["max_tilt_deg"] < 2.0,
        # Nothing turns or pushes the stamp sideways: one that has moved is being pushed by something that is not there.
        "stays_put": result["max_yaw_deg"] < 1.0 and result["max_slide_m"] < 0.001,
        "prints_under_the_press": result["hold_relief_on_paper"] >= 0.95,
        # The two soft layers give by at least half of what the plate asked for, and it can be seen.
        "soft_layers_give": result["peak_squeeze_m"] - rest["squeeze_m"] >= 0.5 * report["overpress_m"],
        "springs_back": abs(final["squeeze_m"] - rest["squeeze_m"]) < 1.0e-4,
        "soft_mesh_intact": result["max_inverted_soft_tets"] == 0,
    }
    result["effective_stiffness_ratio"] = report["effective_stiffness_ratio"]
    return result


def run(
    out: Path,
    *,
    steps: int,
    render_every: int = 0,
    overpress: float = OVERPRESS,
    solver_name: str = stamp_press.SOLVER,
    iterations: int = 160,
    mu: float | None = None,
):
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, presser, geometry, report = build_scene(solver_name=solver_name, iterations=iterations, mu=mu)
    report["overpress_m"] = overpress
    entity = obj.entity
    particles = np.asarray(entity.particles)
    tets, labels = entity.source_tets, entity.tet_part_labels
    nodes, tips, tip_faces, area = geometry["nodes"], geometry["tips"], geometry["tip_faces"], geometry["tip_face_area"]
    info = geometry["report"]
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    paper_top, radius = 2.0 * PAPER_HALF[2], PHYSICS.particle_radius
    half = stamp_press.PRESSER_HALF
    soft_tets = np.isin(labels, SOFT)
    soft_nodes = np.concatenate((nodes[FOAM], nodes[DIE]))
    body = entity.rigid_parts[0]
    quality = MeshQuality(sim, entity)
    contacts = sim.pipeline.contacts
    trajectory = Trajectory(sim, entity)
    dt = 1.0 / (PHYSICS.control_hz * PHYSICS.substeps)
    settle_frames = round(PHASES["settle"][1] * PHYSICS.control_hz)
    start = sim.pipeline.state_0.particle_q.numpy()[particles].astype(np.float64) - surface
    width = float(np.ptp(start[soft_nodes, 0]))

    size = round(2.0 * INK_HALF / INK_PIXEL)
    mask = np.zeros((size, size), dtype=bool)
    inked_at = np.full((len(tip_faces), 2), np.inf)
    face_inked = np.zeros(len(tip_faces), dtype=bool)
    rows, rest = [], None

    videos, renderer, video_error = None, None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet
        from robosimstudio.solvers.render import FALLBACK

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
                # Plan from where the stamp came to rest: the plate goes ``overpress`` past the handle's top.
                rest = dict(rows[-1])
                touch_z = rest["top_z_m"] + surface[2] + radius
                at = surface[:2] + np.asarray(STAMP_AT)
                presser.entity.set_trajectory(
                    stamp_press.press_path(
                        touch_z=touch_z, travel=overpress, start=frame / PHYSICS.control_hz, dt=dt, at=at
                    )
                )
            sim.run(1).sync()
            state = sim.pipeline.state_0
            q, body_q = state.particle_q.numpy(), state.body_q.numpy()
            owned = q[particles].astype(np.float64)
            local = owned - surface
            over = local[:, 2] - paper_top - radius  # each node's gap to its contact distance over the paper
            touching = np.zeros(len(owned), dtype=bool)
            touching[tips[over[tips] <= TOUCH_TOLERANCE]] = True
            face_touch = touching[tip_faces].all(axis=1)
            if face_touch.any():
                centroid = local[tip_faces][:, :, :2].mean(axis=1)
                fresh = face_touch & (np.linalg.norm(centroid - inked_at, axis=1) > 0.4 * INK_PIXEL)
                pixels = (local[tip_faces[fresh]][:, :, :2] - (np.asarray(STAMP_AT) - INK_HALF)) / INK_PIXEL
                stamp_press._ink(mask, pixels)
                inked_at[fresh] = centroid[fresh]
                face_inked |= face_touch
            inverted = np.linalg.det(owned[tets[:, 1:]] - owned[tets[:, :1]]) <= 0
            pose = body_q[body.body]
            count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
            shapes = contacts.soft_contact_shape.numpy()[:count]
            # Each soft layer's thickness, between the bonded faces above and below it.
            foam = float(local[geometry["under_mount"], 2].mean() - local[geometry["under_foam"], 2].mean())
            die = float(local[geometry["under_foam"], 2].mean() - local[tips, 2].mean())
            clock = (frame + 1) / PHYSICS.control_hz
            parts = rigid_part_body.frame_record(entity, owned, body_q)
            row = {
                "frame": frame + 1,
                "time_s": clock,
                "phase": stamp_press.phase_at(clock),
                "finite": bool(np.isfinite(owned).all() and np.isfinite(body_q).all()),
                "plate_bottom_z_m": float(body_q[presser.entity.body, 2] - surface[2] - half[2]),
                "plate_contacts": int(np.count_nonzero(shapes == presser.entity.shape)),
                "top_z_m": float(local[geometry["top_nodes"], 2].mean()),
                "body_z_m": float(pose[2] - surface[2]),
                "tilt_deg": stamp_press._tilt_deg(pose),
                "yaw_deg": stamp_press._yaw_deg(pose),
                "slide_m": float(
                    np.linalg.norm(local[soft_nodes, :2].mean(axis=0) - start[soft_nodes, :2].mean(axis=0))
                ),
                "tip_nodes_touching": int(touching.sum()),
                "relief_on_paper": float(area[face_touch].sum() / area.sum()),
                "foam_squeeze_m": info["foam_thickness_m"] - foam,
                "die_squeeze_m": info["die_thickness_m"] - die,
                "bulge_m": 0.5 * (float(np.ptp(local[soft_nodes, 0])) - width),
                "inverted_soft_tets": int(np.count_nonzero(inverted & soft_tets)),
                "inverted_rigid_tets": int(np.count_nonzero(inverted & ~soft_tets)),
                "rigid_node_stretch_m": float(
                    max(max(p["seam_stretch_max_m"], p["skin_stretch_max_m"]) for p in parts.values())
                ),
                "soft_contacts": count,
            }
            row["squeeze_m"] = row["foam_squeeze_m"] + row["die_squeeze_m"]
            reference = rest if rest is not None else row
            row["press_travel_m"] = float(reference["body_z_m"] - row["body_z_m"])
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
        "inked_area_m2": float(mask.sum() * INK_PIXEL**2),
        "relief_area_m2": info["relief_area_m2"],
    }
    result["checks"]["imprint_complete"] = result["imprint"]["relief_inked"] >= 0.95
    result["success"] = all(result["checks"].values())
    result["mesh_quality"] = quality.read()
    result["config"] = {
        "overpress_m": overpress,
        "solver": type(arm_gripper.vbd_solver(sim.pipeline.program.solver)).__name__,
        "iterations": iterations,
        "substeps": PHYSICS.substeps,
        "control_hz": PHYSICS.control_hz,
        "attach_damping_n_s_m": stamp_press.ATTACH_DAMPING,
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
    parser.add_argument("--overpress", type=float, default=OVERPRESS, help="plate travel past the handle's top [m]")
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default=stamp_press.SOLVER)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--mu", type=float, default=None, help="the stamp's soft-contact friction")
    args = parser.parse_args()
    run(
        args.out,
        steps=args.steps,
        render_every=args.render_every,
        overpress=args.overpress,
        solver_name=args.solver,
        iterations=args.iterations,
        mu=args.mu,
    )


if __name__ == "__main__":
    main()
