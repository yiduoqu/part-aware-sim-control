# SPDX-License-Identifier: Apache-2.0
"""S0 for the silicone spatula: what this solver makes of the head's bending and of the spatula's weight.

Two tests, each against a number the static reference already has
(``outputs/spatula_pan_press/S0_FINDINGS.md``, linear statics on this same mesh):

``cantilever``
    The handle is held level, 100 mm over the bench, and the head hangs from it. Once it has
    settled the tip's sag is read. Then a narrow kinematic plate comes up under the last 6 mm of
    the blade, lifts it ``--push`` and holds; the plate's force over the tip's rise is the tip's
    stiffness. ``--hold pin`` fixes the handle's nodes outright (no mass: this solver leaves such a
    particle alone); ``--hold body`` makes the handle the kinematic rigid body the press scene
    will drive, its nodes tied to it by ``RigidPartSoftSpec``'s springs, and reads the handle's
    reaction from those springs' stretch as a second gauge.

``flat``
    The whole spatula, handle a free rigid body, lies on the bench. The bench's reading -- the
    penalty force of every contact -- should be the spatula's weight.

What differs from the reference at a given ``--substeps`` and ``--iterations`` is the solver's.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import arm_gripper
import newton
import numpy as np
import stamp_press
import warp as wp
from clawhauser_scene import arrangement, build_sim, set_contact
from loop_geometry import smooth
from mesh_quality import MeshQuality, penetration

import robosimstudio as rss
from robosimstudio_materials.heterogeneous import EntityHeterogeneousSoft, HeterogeneousSoftSpec, load_labelled_tets

ASSET = Path(__file__).resolve().parent / "assets" / "silicone_spatula"
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "spatula_s0"
CELL = arm_gripper.CELL
HANDLE, HEAD = 0, 1
PHYSICS = stamp_press.PHYSICS
"""The stamp scenes' rate, penalty, radius and damping; ``--substeps`` replaces the substep count."""
AT = (-0.02, -0.12)
"""The mesh origin on the bench [m, work-surface frame]; the spatula runs along x from it."""
HEIGHT = 0.10
"""The held spatula's underside over the bench [m]."""
GRAVITY = 9.81
TIP_BAND = 0.006
"""The tip is the head's nodes within this of its far end [m], as in the reference."""
PLATE_HALF = (0.003, 0.040, 0.004)
"""The plate under the tip [m, half extents]: as long as the tip band, wider than the blade."""
PARK = 0.080
"""The plate waits this far under the blade while it settles [m]."""
CLEAR = 0.0002
"""The plate stops this short of touching before it starts its push [m]."""
PUSH = 0.005
SETTLE_S, APPROACH_S, PUSH_S, HOLD_S = 2.5, 0.4, 1.5, 1.0
FLAT_S = 2.5
ITERATIONS = 160
"""Solver sweeps per substep, unless ``--iterations`` says otherwise."""
WINDOW_S = 0.5
"""A reading is the mean over the last this long of its phase [s]."""
REFERENCE = {
    # Linear statics on this mesh, head E 1.5 MPa, nu 0.48 (reference.json).
    "pin": {"sag_m": 3.178e-3, "tip_stiffness_n_per_m": 49.80},
    # The shared nodes on 1e4 N/m springs.
    "body": {"sag_m": 3.501e-3, "tip_stiffness_n_per_m": 46.57},
}


@dataclass(frozen=True, kw_only=True)
class PinnedSpec(HeterogeneousSoftSpec):
    """A labelled soft body with one part's nodes fixed where they are built."""

    entity: str = "spatula_pinned_part"
    pinned_part: int = HANDLE


@rss.register_entity("spatula_pinned_part")
class EntityPinned(EntityHeterogeneousSoft):
    spec: PinnedSpec

    def build(self, builder, episode) -> None:
        super().build(builder, episode)
        n = len(self.source_vertices)
        start = int(builder.particle_count) - n
        self.pinned = np.unique(self.source_tets[self.tet_part_labels == self.spec.pinned_part])
        for node in self.pinned:
            builder.particle_mass[start + int(node)] = 0.0
        self.mass[self.pinned] = 0.0


@dataclass(frozen=True, kw_only=True)
class HeldSpec(stamp_press.StampSpec):
    """A :class:`stamp_press.StampSpec` whose rigid groups are kinematic bodies."""

    entity: str = "spatula_held_parts"


@rss.register_entity("spatula_held_parts")
class EntityHeld(stamp_press.EntityStamp):
    spec: HeldSpec

    def build(self, builder, episode) -> None:
        super().build(builder, episode)
        for part in self.rigid_parts:
            builder.body_flags[part.body] = int(newton.BodyFlags.KINEMATIC)


class HoldBody:
    """Keep a kinematic body at the pose it was built at, before every substep's solve."""

    def __init__(self, sim, body: int) -> None:
        self.pipeline, self.device, self.body = sim.pipeline, sim.model.device, int(body)
        pose = sim.pipeline.state_0.body_q.numpy()[self.body].astype(np.float32)
        self.poses = wp.array(pose[None], dtype=wp.transform, device=self.device)
        self.rates = wp.zeros(1, dtype=wp.spatial_vector, device=self.device)
        self.lengths = wp.ones(1, dtype=int, device=self.device)
        self.tick = wp.zeros(1, dtype=int, device=self.device)
        sim.pipeline.hooks.add("pre_solve", self.apply, name="hold-spatula-handle")
        sim.pipeline.invalidate_graph()

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


def contact_forces(q: np.ndarray, body_q: np.ndarray, model, contacts, ke: float):
    """Each stored soft contact's shape and its penalty force on the particle [N, world]."""
    count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    if not count:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    depth = penetration(q, body_q, model, contacts)
    shapes = contacts.soft_contact_shape.numpy()[:count]
    normals = contacts.soft_contact_normal.numpy()[:count]
    return shapes, ke * depth[:, None] * normals


def rotate(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Body-frame ``points`` in the world, for a ``(p, q xyzw)`` pose."""
    p, v, w = pose[:3].astype(np.float64), pose[3:6].astype(np.float64), float(pose[6])
    t = 2.0 * np.cross(v, points)
    return p + points + w * t + np.cross(v, t)


def build_scene(
    test: str,
    hold: str,
    *,
    solver_name: str,
    iterations: int,
    substeps: int,
    young_scale: float,
    attach_stiffness: float,
    damping_ratio: float,
    duration: float,
):
    vertices, tets, labels, elastic, _density = load_labelled_tets(
        ASSET / "final_mesh.mesh", ASSET / "heterogeneous_params.npz"
    )
    head_young = float(elastic[labels == HEAD, 0][0])
    overrides = ((HEAD, "youngs_modulus_pa", head_young * young_scale),) if young_scale != 1.0 else ()
    settings = dataclasses.replace(PHYSICS, substeps=substeps, damping_ratio=damping_ratio, part_overrides=overrides)
    physics = arrangement(
        settings, claw="claw", coarse=False, solver_name=solver_name, iterations=iterations, self_contact=False
    )
    scene = rss.Scene(cell=CELL, physics=physics, name=f"spatula_s0_{test}")
    head_nodes = np.unique(tets[labels == HEAD])
    tips = head_nodes[vertices[head_nodes, 0] >= vertices[head_nodes, 0].max() - TIP_BAND]
    base_z = float(vertices[:, 2].min())
    common = {
        "name": "silicone_spatula",
        "mesh_path": str(ASSET / "final_mesh.mesh"),
        "params_path": str(ASSET / "heterogeneous_params.npz"),
        "materials_path": str(ASSET / "inferred_params.json"),
        "contact_ke": settings.contact_ke,
        "contact_kd": settings.contact_kd,
        "contact_mu": settings.soft_mu,
        "particle_radius": settings.particle_radius,
        "damping_ratio": settings.damping_ratio,
        "part_overrides": overrides,
    }
    presser = None
    if test == "flat":
        pose = (AT[0], AT[1], settings.particle_radius + stamp_press.PLACE_GAP - base_z)
        spec = stamp_press.StampSpec(
            rigid_parts=(HANDLE,),
            attach_stiffness=attach_stiffness,
            attach_damping=stamp_press.ATTACH_DAMPING,
            pose=pose,
            **common,
        )
    else:
        pose = (AT[0], AT[1], HEIGHT - base_z)
        if hold == "pin":
            spec = PinnedSpec(pose=pose, **common)
        else:
            spec = HeldSpec(
                rigid_parts=(HANDLE,),
                attach_stiffness=attach_stiffness,
                attach_damping=stamp_press.ATTACH_DAMPING,
                pose=pose,
                **common,
            )
        plate_x = float(vertices[tips, 0].max()) - PLATE_HALF[0]
        plate_y = float(vertices[tips, 1].mean())
        under = float(vertices[tips, 2].min())
        presser = scene.add(
            stamp_press.PresserSpec(
                center=(pose[0] + plate_x, pose[1] + plate_y, pose[2] + under - PARK - PLATE_HALF[2]),
                half=PLATE_HALF,
                control_hz=settings.control_hz,
                substeps=settings.substeps,
                duration=duration,
            )
        )
    obj = scene.add(spec)
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    middle = surface + np.array((pose[0] + 0.11, pose[1], pose[2]))
    scene.add(
        rss.CameraSpec(
            name="side",
            width=960,
            height=540,
            eye=tuple(float(v) for v in np.add(middle, (0.0, -0.30, 0.03))),
            look_at=tuple(float(v) for v in middle),
            fov_y=30.0,
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    entity = obj.entity
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    if not type(solver).__name__.startswith("SolverVBD"):
        raise RuntimeError(f"expected a VBD solver, got {type(solver).__name__}")
    if test == "cantilever" and hold == "pin":
        report = {"friction": set_contact(sim, solver, settings, mu=None, grip_mu=None)}
        inverse = sim.model.particle_inv_mass.numpy()[np.asarray(entity.particles)[entity.pinned]]
        if np.any(inverse != 0.0):
            raise RuntimeError("a pinned node kept its mass")
        report["pinned_nodes"] = int(len(entity.pinned))
    else:
        report = stamp_press.check_stamp(sim, entity, tets, settings, mu=None)
    young = entity.elastic[:, 0]
    report.update(
        {
            "test": test,
            "hold": hold if test == "cantilever" else "free",
            "head_young_pa": float(young[labels == HEAD].max()),
            "handle_young_pa": float(young[labels == HANDLE].max()),
            "particle_weight_n": float(entity.mass.sum() * GRAVITY),
            "pose_m": [float(v) for v in pose],
            "tip_nodes": int(len(tips)),
        }
    )
    print(json.dumps({"geometry": report}), flush=True)
    return sim, obj, presser, {"tips": tips, "head_nodes": head_nodes, "tets": tets, "labels": labels}, report, settings


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def run(
    out: Path,
    *,
    test: str,
    hold: str,
    solver_name: str,
    iterations: int,
    substeps: int,
    young_scale: float = 1.0,
    attach_stiffness: float = 1.0e4,
    damping_ratio: float = PHYSICS.damping_ratio,
    push: float = PUSH,
    settle_s: float = SETTLE_S,
    render_every: int = 0,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cantilever = test == "cantilever"
    duration = settle_s + APPROACH_S + PUSH_S + HOLD_S if cantilever else FLAT_S
    sim, obj, presser, geometry, report, settings = build_scene(
        test,
        hold,
        solver_name=solver_name,
        iterations=iterations,
        substeps=substeps,
        young_scale=young_scale,
        attach_stiffness=attach_stiffness,
        damping_ratio=damping_ratio,
        duration=duration,
    )
    entity = obj.entity
    particles = np.asarray(entity.particles)
    tets, labels, tips, head_nodes = geometry["tets"], geometry["labels"], geometry["tips"], geometry["head_nodes"]
    hz, radius, ke = settings.control_hz, settings.particle_radius, settings.contact_ke
    dt = 1.0 / (hz * settings.substeps)
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    soft_tets = labels == HEAD
    quality = MeshQuality(sim, entity)
    contacts = sim.pipeline.contacts
    rigid = getattr(entity, "rigid_parts", [])
    holder = HoldBody(sim, rigid[0].body) if cantilever and hold == "body" else None
    start = sim.pipeline.state_0.particle_q.numpy()[particles].astype(np.float64) - surface
    tip_rest = float(start[tips, 2].mean())
    weight = report["particle_weight_n"] + sum(part.body_mass for part in rigid) * GRAVITY * (0.0 if holder else 1.0)
    settle_frames = round(settle_s * hz)
    steps = round(duration * hz)
    plate_xy = None if presser is None else np.asarray(presser.entity.center[:2]) - surface[:2]
    rows, local = [], start

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

    def phase_at(clock: float) -> str:
        if not cantilever:
            return "settle"
        for name, end in (
            ("settle", settle_s),
            ("approach", settle_s + APPROACH_S),
            ("push", settle_s + APPROACH_S + PUSH_S),
        ):
            if clock <= end + 1.0e-9:
                return name
        return "hold"

    started = time.perf_counter()
    try:
        for frame in range(steps):
            if cantilever and frame == settle_frames:
                # From where the blade hangs now: up to just under it, then ``push`` further.
                near = head_nodes[
                    (np.abs(local[head_nodes, 0] - plate_xy[0]) <= PLATE_HALF[0] + radius)
                    & (np.abs(local[head_nodes, 1] - plate_xy[1]) <= PLATE_HALF[1])
                ]
                if not len(near):
                    # The blade has swung clear of the plate: there is nothing over it to lift.
                    report["plate_missed"] = True
                    near = tips
                touch = float(local[near, 2].min()) - radius - CLEAR
                parked = float(presser.entity.center[2] - surface[2] + PLATE_HALF[2])
                count = int(np.ceil((duration - settle_s) / dt)) + 2
                poses = np.zeros((count, 7), dtype=np.float32)
                poses[:, 0], poses[:, 1], poses[:, 6] = presser.entity.center[0], presser.entity.center[1], 1.0
                for index in range(count):
                    clock = index * dt
                    top = (
                        parked
                        + (touch - parked) * smooth(clock, 0.0, APPROACH_S)
                        + push * smooth(clock, APPROACH_S, APPROACH_S + PUSH_S)
                    )
                    poses[index, 2] = surface[2] + top - PLATE_HALF[2]
                presser.entity.set_trajectory(poses)
                report["plate_touch_z_m"] = touch
            sim.run(1).sync()
            state = sim.pipeline.state_0
            q, body_q = state.particle_q.numpy(), state.body_q.numpy()
            owned = q[particles].astype(np.float64)
            local = owned - surface
            shapes, forces = contact_forces(q, body_q, sim.model, contacts, ke)
            inverted = np.linalg.det(owned[tets[:, 1:]] - owned[tets[:, :1]]) <= 0
            clock = (frame + 1) / hz
            row = {
                "frame": frame + 1,
                "time_s": clock,
                "phase": phase_at(clock),
                "finite": bool(np.isfinite(owned).all() and np.isfinite(body_q).all()),
                "tip_z_m": float(local[tips, 2].mean()),
                "sag_m": tip_rest - float(local[tips, 2].mean()),
                "soft_contacts": int(len(shapes)),
                "contact_force_z_n": float(forces[:, 2].sum()) if len(forces) else 0.0,
                "inverted_soft_tets": int(np.count_nonzero(inverted & soft_tets)),
                "speed_m_s": float(np.linalg.norm(local - (rows[-1]["_local"] if rows else start), axis=1).max() * hz),
            }
            row["_local"] = local
            if rows:
                del rows[-1]["_local"]
            if presser is not None:
                on_plate = shapes == presser.entity.shape
                row["plate_top_z_m"] = float(body_q[presser.entity.body, 2] - surface[2] + PLATE_HALF[2])
                row["plate_contacts"] = int(np.count_nonzero(on_plate))
                row["plate_force_n"] = float(forces[on_plate, 2].sum()) if on_plate.any() else 0.0
            if rigid:
                part = rigid[0]
                pose = body_q[part.body]
                stretch = rotate(pose, part.body_point) - owned[part.nodes]
                # What the handle's springs carry, upward: the hanging particles' weight less the plate's push.
                row["handle_reaction_n"] = float(entity.spec.attach_stiffness * stretch[:, 2].sum())
                row["handle_tilt_deg"] = stamp_press._tilt_deg(pose)
                row["handle_z_m"] = float(pose[2] - surface[2])
            rows.append(row)
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
    result = {
        "complete": len(rows) == steps,
        "finite": all(row["finite"] for row in rows),
        "max_inverted_soft_tets": max(row["inverted_soft_tets"] for row in rows),
    }
    if cantilever:
        settled = [row for row in rows if row["phase"] == "settle"][-window:]
        held = [row for row in rows if row["phase"] == "hold"][-window:]
        target = REFERENCE[hold]
        result["sag_m"] = mean(settled, "sag_m")
        # A blade that reached the waiting plate was stopped by it: its sag is then a lower bound.
        result["reached_parked_plate"] = any(row["plate_contacts"] for row in rows if row["phase"] == "settle")
        result["sag_swing_m"] = float(np.ptp([row["sag_m"] for row in settled]))
        result["sag_reference_m"] = target["sag_m"] / young_scale
        result["sag_over_reference"] = result["sag_m"] / result["sag_reference_m"]
        result["plate_missed"] = bool(report.get("plate_missed", False))
        if held:
            rise = mean(settled, "tip_z_m") - mean(held, "tip_z_m")
            result["tip_rise_m"] = -rise
            result["plate_force_n"] = mean(held, "plate_force_n")
            result["plate_force_swing_n"] = float(np.ptp([row["plate_force_n"] for row in held]))
            result["plate_contacts"] = held[-1]["plate_contacts"]
            if "handle_reaction_n" in held[-1]:
                result["handle_reaction_settled_n"] = mean(settled, "handle_reaction_n")
                result["handle_reaction_held_n"] = mean(held, "handle_reaction_n")
                # The second gauge: what the springs stopped carrying when the plate took it.
                result["plate_force_from_handle_n"] = (
                    result["handle_reaction_settled_n"] - result["handle_reaction_held_n"]
                )
                result["hanging_weight_n"] = report["particle_weight_n"]
            if result["tip_rise_m"] > 0:
                result["tip_stiffness_n_per_m"] = result["plate_force_n"] / result["tip_rise_m"]
                result["tip_stiffness_reference_n_per_m"] = target["tip_stiffness_n_per_m"] * young_scale
                result["tip_stiffness_over_reference"] = (
                    result["tip_stiffness_n_per_m"] / result["tip_stiffness_reference_n_per_m"]
                )
        # The spring reference is for 1e4 N/m and the source modulus; elsewhere it is only scaled.
        result["reference_exact"] = young_scale == 1.0 and (hold == "pin" or attach_stiffness == 1.0e4)
    else:
        rest = rows[-window:]
        result["bench_force_n"] = mean(rest, "contact_force_z_n")
        result["bench_force_swing_n"] = float(np.ptp([row["contact_force_z_n"] for row in rest]))
        result["weight_n"] = weight
        result["bench_over_weight"] = result["bench_force_n"] / weight
        result["final_speed_m_s"] = max(row["speed_m_s"] for row in rest)
        result["final_handle_tilt_deg"] = rest[-1].get("handle_tilt_deg")
        result["soft_contacts"] = rest[-1]["soft_contacts"]
    result["mesh_quality"] = quality.read()
    result["config"] = {
        "test": test,
        "hold": report["hold"],
        "solver": type(arm_gripper.vbd_solver(sim.pipeline.program.solver)).__name__,
        "solver_name": solver_name,
        "iterations": iterations,
        "substeps": substeps,
        "control_hz": hz,
        "young_scale": young_scale,
        "attach_stiffness_n_m": attach_stiffness,
        "damping_ratio_s": damping_ratio,
        "push_m": push,
        "settle_s": settle_s,
        "newton_version": newton.__version__,
    }
    result["geometry"] = report
    result["video_error"] = video_error
    result["simulated_s"] = len(rows) / hz
    result["elapsed_s"] = elapsed
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "geometry"}), flush=True)
    brief = {k: v for k, v in result.items() if isinstance(v, (int, float, bool)) or v is None}
    brief.update({k: result["config"][k] for k in ("test", "hold", "solver_name", "substeps", "iterations")})
    print("S0_RESULT " + json.dumps(brief), flush=True)
    sim.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", choices=("cantilever", "flat"), default="cantilever")
    parser.add_argument("--hold", choices=("pin", "body"), default="pin", help="how the handle is held (cantilever)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default=stamp_press.SOLVER)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--substeps", type=int, default=PHYSICS.substeps)
    parser.add_argument("--young-scale", type=float, default=1.0, help="multiplies the head's Young's modulus")
    parser.add_argument("--attach-stiffness", type=float, default=1.0e4, help="each node-body spring [N/m]")
    parser.add_argument(
        "--damping-ratio", type=float, default=PHYSICS.damping_ratio, help="tet damping over shear modulus [s]"
    )
    parser.add_argument("--push", type=float, default=PUSH, help="how far the plate lifts the tip [m]")
    parser.add_argument(
        "--settle", type=float, default=SETTLE_S, help="how long the blade hangs before the plate moves [s]"
    )
    parser.add_argument("--render-every", type=int, default=0)
    args = parser.parse_args()
    name = f"{args.test}_{args.hold if args.test == 'cantilever' else 'free'}_{args.solver}_{args.substeps}x{args.iterations}"
    run(
        args.out if args.out is not None else OUTPUT / name,
        test=args.test,
        hold=args.hold,
        solver_name=args.solver,
        iterations=args.iterations,
        substeps=args.substeps,
        young_scale=args.young_scale,
        attach_stiffness=args.attach_stiffness,
        damping_ratio=args.damping_ratio,
        push=args.push,
        settle_s=args.settle,
        render_every=args.render_every,
    )


if __name__ == "__main__":
    main()
