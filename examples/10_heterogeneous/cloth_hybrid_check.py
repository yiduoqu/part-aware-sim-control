# SPDX-License-Identifier: Apache-2.0
"""Behaviour check for the cloth hybrids as VBD FEM (``cloth_to_tets.py``): the AIRBOT holds the rigid grip.

``duster``: the ABS grip is pinched, lifted, lowered until the panels press on the bench and swept side to side
so they drag. ``bag``: the aluminium handle is pinched and the bag lifted 15 cm and held. Each run records whether
the rigid parts stay tied to the cloth (anchor gaps), the cloth's stretch, inverted tets, NaNs and how the cloth
hangs. The cloth is a thin solid shell of tets (soft label 0) and the rigid parts are rigid bodies
(:mod:`rigid_part_body`), as in the cable scenes.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import arm_gripper
import cable_scene
import numpy as np
from clawhauser_scene import REFERENCE, arrangement, build_sim
from loop_geometry import smooth

import robosimstudio as rss

OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "cloth_hybrid_check"
CONTROL_HZ = REFERENCE.control_hz
SUBSTEPS = 40
SETTLE = 1.2
"""How long the cloth settles on the bench before the hand moves in [s]; the grasp is measured at its end."""
HOVER = 0.04
SQUEEZE = 0.012
"""How far past the grip's width the closed pads aim [m]; the clamp stops them."""
PHASES = {"descend": (1.3, 1.8), "close": (1.9, 2.5), "lift": (2.8, 4.0)}
SWEEP = {"start": 5.0, "period": 2.4, "cycles": 2, "amplitude": 0.08, "lower": (4.2, 4.9)}
DRAG_Z = 0.030
"""The duster's grip centre over the bench while it sweeps [m]: its lowest panel tips are 40 mm below it."""
LIFT_BAG = 0.15
TURN_Z = cable_scene.TURN_Z
GRASP = arm_gripper.GRASP_QUAT
TILTS = (0.0, 25.0, 50.0, 75.0, -25.0)
"""Hand pitches tried in turn [deg]: the bag's lifted handle is out of reach of a hand pointing straight down."""


@dataclasses.dataclass(frozen=True)
class Task:
    asset: cable_scene.CableAsset
    at: tuple[float, float]
    """Where the grasp point lies in the bench frame [m]."""
    grasp: tuple[float, float, float]
    """The grasp point in the asset frame [m]."""
    band: float
    """Half the length of the grip's nodes that are measured and pinched [m]."""
    duration: float
    force_limit: float
    rigid: int
    """The rigid part held (a label)."""


TASKS = {
    "duster": Task(
        cable_scene.CableAsset("cloth_duster_fem", TURN_Z, 5.0e5, 200.0, None), (0.0, -0.12), (-0.09, 0.0, 0.006), 0.006, 11.5, 30.0, 1
    ),
    "bag": Task(
        cable_scene.CableAsset("cloth_bag_fem", (0.0, 0.0, 0.0, 1.0), 1.0e5, 100.0, None), (0.0, -0.12), (0.0, 0.0, 0.390), 0.012, 7.5, 50.0, 1
    ),
}  # fmt: skip


def lift_z(task: str, center_z: float):
    """Height of the grasp point over time: lift, then (duster) lower onto the bench, or (bag) hold."""

    def z_at(t: float) -> float:
        up = smooth(t, *PHASES["lift"])
        if task == "bag":
            return center_z + LIFT_BAG * up
        return center_z + 0.06 * up + (DRAG_Z - center_z - 0.06) * smooth(t, *SWEEP["lower"])

    return z_at


def motion(task: str, center: np.ndarray, reach: float, surface: np.ndarray):
    """The hand's ``(tcp_at, half_gap_at)`` for a grasp point ``center`` [m, world] with half width ``reach``."""
    z_at = lift_z(task, center[2] - surface[2])

    def tcp_at(t: float) -> np.ndarray:
        point = np.array((center[0], center[1], surface[2] + z_at(t) + HOVER * (1.0 - smooth(t, *PHASES["descend"]))))
        if task == "duster":
            s = SWEEP
            ramp = smooth(t, s["start"], s["start"] + 0.6)
            span = 2.0 * np.pi * (t - s["start"]) / s["period"]
            point[0] += ramp * s["amplitude"] * np.sin(np.clip(span, 0.0, 2.0 * np.pi * s["cycles"]))
        return arm_gripper.tcp_for_pads(point)

    def half_gap_at(t: float) -> float:
        return reach + SQUEEZE - 2.0 * SQUEEZE * smooth(t, *PHASES["close"])

    return tcp_at, half_gap_at


def build_scene(task: Task, *, solver: str, iterations: int, mu: float, grip_mu: float, self_contact: bool, asset):
    vertices, tets, labels = cable_scene.load(asset)
    settings = dataclasses.replace(REFERENCE, substeps=SUBSTEPS, particle_radius=0.0015)
    physics = arrangement(
        settings,
        claw="arm",
        coarse=False,
        solver_name=solver,
        iterations=iterations,
        self_contact=self_contact,
        feedback_smoothing=0.15,
        feedback_clamp=1.5,
        self_contact_radius=0.002,
        rest_exclusion=0.006,
    )
    scene = rss.Scene(cell=arm_gripper.CELL, physics=physics, name="cloth_hybrid")
    turned = arm_gripper._rotate(np.asarray(asset.quat), vertices)
    grasp = arm_gripper._rotate(np.asarray(asset.quat), np.asarray(task.grasp)[None])[0]
    pose = (task.at[0] - grasp[0], task.at[1] - grasp[1], -turned[:, 2].min() + cable_scene.PLACE_GAP)
    obj = scene.add(cable_scene.spec(asset, vertices, tets, labels, name="cloth", pose=pose, settings=settings, mu=mu))
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=task.force_limit)
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    camera = scene.add(
        rss.CameraSpec(
            name="overview",
            width=960,
            height=540,
            eye=tuple(surface + np.array((task.at[0] + 0.55, task.at[1] - 0.45, 0.45))),
            look_at=tuple(surface + np.array((task.at[0], task.at[1] + 0.03, 0.12))),
            fov_y=40.0,
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    solver_ = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    geometry = {"pose": list(pose), "arm": arm_gripper.check_arm(sim, limit=task.force_limit)}
    geometry["friction"] = cable_scene.set_friction(sim, solver_, settings, mu=mu, grip_mu=grip_mu)
    print(json.dumps({"geometry": geometry}), flush=True)
    return sim, obj, camera, geometry, labels


def run(out: Path, name: str, *, steps: int, render_every: int, config: dict, **physics) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    task = TASKS[name]
    asset = cable_scene.with_levers(task.asset, argparse.Namespace(**physics.pop("levers")))
    sim, obj, camera, geometry, labels = build_scene(task, asset=asset, **physics)
    entity = obj.entity
    tets = entity.source_tets
    soft = cable_scene.soft_labels(asset, labels)
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    q0 = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
    part = next(rp for rp in entity.rigid_parts if rp.part == str(task.rigid))
    # The nodes the pads close on and that are tracked: the rigid part's within ``band`` of the grasp point along its length.
    rigid_nodes = part.nodes
    near = rigid_nodes[np.linalg.norm(q0[rigid_nodes][:, :2] - task.at, axis=1) < task.band]
    cloth_nodes = np.unique(tets[np.isin(labels, soft)])
    base = None
    if name == "bag":
        other = next(rp for rp in entity.rigid_parts if rp.part != str(task.rigid))
        base = other.nodes
    dt = 1.0 / CONTROL_HZ
    capacity = max(steps, round(task.duration * CONTROL_HZ))
    control = arm_gripper.ArmControl(sim)
    state = {}

    def plan(center: np.ndarray, reach: float) -> None:
        """The hand pitched by the first of ``TILTS`` [deg, about x: fingers forward] that reaches the whole path."""
        tcp_at, half_gap_at = motion(name, center, reach, surface)
        best = None
        for tilt in TILTS:
            quat = arm_gripper.quat_multiply(arm_gripper.axis_angle_quat((1.0, 0.0, 0.0), np.radians(tilt)), GRASP)
            route = control.plan(quat=quat, capacity=capacity, dt=dt, tcp_at=tcp_at, half_gap_at=half_gap_at)
            best = min(best or (route, tilt), (route, tilt), key=lambda r: r[0].worst_position_error_m)
            if route.worst_position_error_m < 0.015:
                break
        state["route"], geometry["tilt_deg"] = best
        if best[0].worst_position_error_m > 0.025:
            raise RuntimeError(f"the arm misses the path by {best[0].worst_position_error_m:.3f} m")

    centre0 = q0[near].mean(axis=0) + surface
    plan(centre0, float(0.5 * np.ptp(q0[near][:, 0])))
    control.start(state["route"])
    settle = round(SETTLE * CONTROL_HZ)

    def follow(frame: int):
        if frame == settle:
            now = sim.pipeline.state_0.particle_q.numpy()[entity.particles][near]
            reach = float(0.5 * np.ptp(now[:, 0]))
            plan(now.mean(axis=0), reach)
            geometry["settled"] = {"grasp_m": (now.mean(axis=0) - surface).tolist(), "reach_m": reach}
            print(json.dumps({"settled": geometry["settled"]}), flush=True)
        control.follow(state["route"], frame)
        return control

    def task_row(frame: int, local: np.ndarray, body_q: np.ndarray) -> dict:
        tool = local[near].mean(axis=0)
        row = {
            "finite": bool(np.isfinite(local).all()),
            "tool_m": tool.tolist(),
            "cloth_min_z_m": float(local[cloth_nodes, 2].min()),
            "cloth_on_bench": int(np.count_nonzero(local[cloth_nodes, 2] < 0.004)),
            "cloth_lowest_below_tool_m": float(tool[2] - local[cloth_nodes, 2].min()),
        }
        if base is not None:
            row["base_z_m"] = float(local[base, 2].mean())
        return row

    rows, elapsed, video_error = cable_scene.episode(
        sim, entity, camera, out, soft=soft, steps=steps, render_every=render_every, control_hz=CONTROL_HZ,
        follow=follow, task_row=task_row,
    )  # fmt: skip
    parts = [k for k in rows[0] if k.startswith("part") and k.endswith("_body")]
    tool = np.array([r["tool_m"] for r in rows])
    result = {
        "finite": all(r["finite"] for r in rows),
        "anchor_gap_max_m": max(
            max(r[k]["seam_stretch_max_m"], r[k]["skin_stretch_max_m"]) for r in rows for k in parts
        ),
        "seam_gap_max_m": {k: max(r[k]["seam_stretch_max_m"] for r in rows) for k in parts},
        "cloth_stretch_p99_max": max(r["cable_stretch_p99"] for r in rows),
        "cloth_inverted_max": max(r["cable_inverted"] for r in rows),
        "tool_range_m": (tool.max(0) - tool.min(0)).tolist(),
        "tool_end_m": tool[-1].tolist(),
        "cloth_on_bench_max": max(r["cloth_on_bench"] for r in rows),
        "finger_contacts_end": rows[-1]["finger_contacts"],
        "steps": steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "elapsed_s": elapsed,
        "geometry": geometry,
        "config": config,
    }
    if name == "bag":
        lifted = float(tool[-1, 2] - rows[round(PHASES["descend"][1] * CONTROL_HZ) - 1]["tool_m"][2])
        result["handle_lift_m"] = lifted
        result["base_lift_m"] = float(rows[-1]["base_z_m"] - rows[0]["base_z_m"])
        result["score"] = lifted
        held = lifted > 0.9 * LIFT_BAG and result["base_lift_m"] > 0.8 * LIFT_BAG
    else:
        sweep = float(np.ptp(tool[round((SWEEP["start"] + 0.6) * CONTROL_HZ) :, 0])) if len(tool) > 300 else 0.0
        result["sweep_m"] = sweep
        result["score"] = sweep
        held = sweep > 1.4 * SWEEP["amplitude"] and result["cloth_on_bench_max"] > 0
    result["success"] = bool(
        held
        and result["finite"]
        and result["anchor_gap_max_m"] < 0.01
        and result["cloth_stretch_p99_max"] < 1.5
        and result["cloth_inverted_max"] == 0
        and result["finger_contacts_end"] > 0
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "config")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(TASKS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=0, help="frames (0: the whole route)")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default="chebyshev")
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--mu", type=float, default=cable_scene.OBJECT_MU)
    parser.add_argument("--grip-mu", type=float, default=2.0)
    parser.add_argument("--no-self-contact", dest="self_contact", action="store_false")
    cable_scene.add_lever_arguments(parser)
    args = parser.parse_args()
    task = TASKS[args.task]
    asset = cable_scene.with_levers(task.asset, args)
    run(
        args.out or OUTPUT / args.task,
        args.task,
        steps=args.steps or round(task.duration * CONTROL_HZ),
        render_every=args.render_every,
        config=cable_scene.config(args, asset, dataclasses.replace(REFERENCE, substeps=SUBSTEPS)),
        solver=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        grip_mu=args.grip_mu,
        self_contact=args.self_contact,
        levers={k: getattr(args, k) for k in ("cable_young", "cable_damping", "attach_damping")},
    )


if __name__ == "__main__":
    main()
