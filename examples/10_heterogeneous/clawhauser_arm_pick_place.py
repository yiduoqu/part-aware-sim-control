# SPDX-License-Identifier: Apache-2.0
"""Pick the labelled Clawhauser volume up by its loop, lift it, and set it back down.

This is ``python -m part_aware_control script --task clawhauser`` without its
sideways carry, and nothing else changed:

- the scene is ``clawhauser_pick_drop.build_scene()`` with its defaults, as
  ``part_aware_control.tasks.build_scene`` builds it;
- the arm waits at the same common start, and the action list is
  ``part_aware_control.tasks.scripted_plan``'s with the carry left out;
- each action runs as ``part_aware_control.rollout`` runs one: from the measured
  tool pose and jaw opening to the commanded ones, on the same smoothstep, for
  max(0.8 s, distance / 60 mm/s) when the plan states no duration;
- the gates and their thresholds are that task's. ``scripted_plan`` carries the
  grasp over the target; with no carry the target is centred where the grasp
  already is.

Run on a CUDA node with Newton 1.8 development and RoboSimStudio on PYTHONPATH.
Outputs are written under ``outputs/clawhauser_arm_pick_place``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import arm_gripper
import numpy as np
from clawhauser_pick_drop import CELL, CONTROL_HZ, REST_Z, TABLE_Z, build_scene
from clawhauser_scene import loop_shift
from mesh_quality import MeshQuality

import robosimstudio as rss

OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "clawhauser_arm_pick_place"

# The plan: part_aware_control.tasks.scripted_plan for the Clawhauser, less the carry.
INITIAL_TCP_M = (0.47, 0.0, 1.00)
"""Where the open hand waits while the scene settles, in world metres
(``common_initial_tcp_world_m``)."""
INITIAL_SETTLE_S = 1.0
OPEN_WIDTH_M = 2.0 * arm_gripper.OPEN_HALF_GAP
"""The open hand's total pad-face gap [m]: 80 mm, the CLI's ``MAX_APERTURE_M``."""
LOWER_M = 0.140
"""How far the pads come back down before the hand opens [m]. With the scene's
``lift_m`` (about 301 mm) the hand opens about 160 mm above where it closed."""
LOWER_S = 8.0
RETREAT_M = 0.120
RETREAT_S = 2.0
FINAL_SETTLE_S = 2.0
"""How long the run continues after the last action, the arm holding still [s]."""

# The gates: part_aware_control.tasks.Task as the CLI sets it for --task clawhauser.
TARGET_HALF_M = (0.060, 0.100)
LIFTED_MIN_M = 0.030
SETTLE_WINDOW_S = 0.5
MAX_SETTLED_DRIFT_M = 0.003
MAX_SETTLED_SPEED_M_S = 0.010
SURFACE_TOLERANCE_M = 0.002
MAX_TABLE_PENETRATION_M = 0.005
MAX_IK_ERROR_M = 0.010
OPEN_MIN_M = 0.060
"""The jaws count as open at this total pad-face gap [m]."""
PAD_CLEAR_M = 0.010
"""The pads count as off the toy with this much space between their bounds [m]."""

PHASES = ("settle", "align", "reach", "close", "lift_1", "lift_2", "hang", "lower", "open", "retreat", "rest")


def worst(windows: list[dict]) -> dict:
    """Several ``MeshQuality`` windows as one: the smallest J and the summed counts, per part."""
    return {
        "minimum_J_by_part": np.min([w["minimum_J_by_part"] for w in windows], axis=0).tolist(),
        "inverted_tet_substeps_by_part": np.sum([w["inverted_tet_substeps_by_part"] for w in windows], axis=0).tolist(),
        "nonfinite_tet_substeps_by_part": np.sum(
            [w["nonfinite_tet_substeps_by_part"] for w in windows], axis=0
        ).tolist(),
    }


def judge(samples: list[dict], pads: tuple[np.ndarray, np.ndarray], target: np.ndarray, quality: dict) -> dict:
    """Score the run as ``part_aware_control.rollout.Rollout.measure`` does.

    ``samples`` start with the state after the initial settle, as the CLI's do.
    """
    first, last = samples[0], samples[-1]
    tail = [s for s in samples if s["time_s"] >= last["time_s"] - SETTLE_WINDOW_S]
    centres = np.asarray([s["centre_m"] for s in tail])
    times = np.asarray([s["time_s"] for s in tail])
    elapsed = float(times[-1] - times[0])
    drift = float(np.linalg.norm(centres - centres[-1], axis=1).max())
    speed = (
        float(np.linalg.norm(np.diff(centres, axis=0), axis=1).max() / max(np.diff(times).min(), 1e-9))
        if len(tail) > 1
        else float("inf")
    )
    lifted = max(s["minimum_z_m"] for s in samples) - first["minimum_z_m"]
    low, high = pads
    clear = bool(
        np.any(low > np.asarray(last["bounds_max_m"]) + PAD_CLEAR_M)
        or np.any(high < np.asarray(last["bounds_min_m"]) - PAD_CLEAR_M)
    )
    penetration = max(0.0, TABLE_Z - min(s["minimum_z_m"] for s in samples))
    valid = not (sum(quality["inverted_tet_substeps_by_part"]) or sum(quality["nonfinite_tet_substeps_by_part"]))
    gates = {
        "valid": bool(valid and penetration <= MAX_TABLE_PENETRATION_M),
        "lifted": bool(lifted >= LIFTED_MIN_M),
        "placed": bool(np.all(np.abs(centres[:, :2] - target[:2]) <= np.asarray(TARGET_HALF_M))),
        "released": bool(min(s["aperture_m"] for s in tail) >= OPEN_MIN_M and clear),
        "supported": all(abs(s["minimum_z_m"] - REST_Z) <= SURFACE_TOLERANCE_M for s in tail),
        "settled": bool(elapsed >= 0.45 and drift <= MAX_SETTLED_DRIFT_M and speed <= MAX_SETTLED_SPEED_M_S),
    }
    moved = np.asarray(last["centre_m"]) - np.asarray(first["centre_m"])
    squeezes = [s["squeeze_n"] for s in samples if "squeeze_n" in s]
    return {
        "success": all(gates.values()),
        **gates,
        "reason": ", ".join(name for name, passed in gates.items() if not passed) or "all measured gates passed",
        "lifted_m": float(lifted),
        "moved_m": moved.tolist(),
        "final_centre_m": last["centre_m"],
        "target_centre_m": target.tolist(),
        "final_drift_m": drift,
        "final_peak_speed_m_s": speed,
        "max_table_penetration_m": float(penetration),
        "peak_squeeze_n": max(squeezes) if squeezes else None,
        "mesh_quality": quality,
    }


def run(
    out: Path, *, render_every: int = 2, until: str = PHASES[-1], lift: float | None = None, lower: float = LOWER_M
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sim, obj, _, camera, geometry = build_scene()
    entity = obj.entity
    control = arm_gripper.ArmControl(sim)
    quality = MeshQuality(sim, entity)
    built = sim.pipeline.state_0.particle_q.numpy()[entity.particles]
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    nominal = arm_gripper.tcp_for_pads(surface + np.asarray(geometry["grip_centre_m"], dtype=np.float64))
    closed = max(0.0, 2.0 * (float(geometry["loop_reach_m"]) - float(geometry["grip_squeeze_m"])))
    lift = float(geometry["lift_m"]) if lift is None else float(lift)

    arms = sim.handles.arms[arm_gripper.ARM]
    finger_sides = [
        np.array([int(shape) for _, shapes in side for shape in shapes], dtype=np.int32) for side in arms.finger_groups
    ]
    shape_forces = getattr(sim.pipeline.program.solver, "shape_contact_forces", None)

    video = None
    if render_every:
        from robosimstudio.core.render.video import VideoWriter

        video = VideoWriter(out / "overview.mp4", fps=CONTROL_HZ // render_every, preset="veryfast", threads=2)

    samples, phases, windows, executed = [], [], [], []
    frame = 0
    worst_ik = 0.0
    # The commanded tool pose and jaw opening; each action's end is the last one's plus its delta.
    target, quat, width = np.asarray(INITIAL_TCP_M, dtype=np.float64), None, OPEN_WIDTH_M
    start = grasp = None

    def toy() -> np.ndarray:
        return sim.pipeline.state_0.particle_q.numpy()[entity.particles]

    def route(begin, end, q_begin, q_end, width_begin: float, width_end: float, seconds: float):
        """``part_aware_control.rollout``'s ``route_between``: a smoothstep from one pose and gap to another."""
        frames = max(2, math.ceil(seconds * CONTROL_HZ))
        q_begin, q_end = np.asarray(q_begin, dtype=np.float64), np.asarray(q_end, dtype=np.float64)
        if np.dot(q_begin, q_end) < 0:
            q_end = -q_end
        span = (frames - 1) / CONTROL_HZ

        def blend(clock: float) -> float:
            x = min(1.0, clock / span)
            return x * x * (3.0 - 2.0 * x)

        def orientation(clock: float) -> np.ndarray:
            q = q_begin + (q_end - q_begin) * blend(clock)
            return q / np.linalg.norm(q)

        return control.plan(
            quat=orientation,
            capacity=frames,
            dt=1.0 / CONTROL_HZ,
            tcp_at=lambda clock: begin + (end - begin) * blend(clock),
            half_gap_at=lambda clock: 0.5 * (width_begin + (width_end - width_begin) * blend(clock)),
        )

    def step(name: str) -> None:
        """One control frame, and its sample."""
        nonlocal frame
        sim.run(1).sync()
        frame += 1
        points = toy()
        if not np.isfinite(points).all():
            raise RuntimeError(f"{name}: nonfinite particle state at frame {frame}")
        sample = {
            "frame": frame,
            "time_s": frame / CONTROL_HZ,
            "phase": name,
            "centre_m": points.mean(axis=0).tolist(),
            "minimum_z_m": float(points[:, 2].min()),
            "bounds_min_m": points.min(axis=0).tolist(),
            "bounds_max_m": points.max(axis=0).tolist(),
            "tcp_m": np.asarray(control.tcp()).tolist(),
            "aperture_m": 2.0 * control.aperture(),
            "commanded_aperture_m": 2.0 * arm_gripper.half_gap_of(float(control.hand.command)),
        }
        if shape_forces is not None:
            load, squeeze = arm_gripper.pad_forces(sim, shape_forces(), finger_sides)
            sample["pad_force_n"], sample["squeeze_n"] = max(load), max(squeeze)
        window = quality.read()
        quality.clear()
        windows.append(window)
        sample["minimum_j"] = min(window["minimum_J_by_part"])
        sample["inverted_tet_substeps"] = sum(window["inverted_tet_substeps_by_part"])
        samples.append(sample)
        if video is not None and frame % render_every == 0:
            video.append(sim.render(camera.spec)["rgb"])

    def finish(name: str, count: int) -> None:
        last = samples[-1]
        phases.append(
            {
                "phase": name,
                "seconds": count / CONTROL_HZ,
                "ends_at_s": last["time_s"],
                "pads_at_m": target.tolist(),
                "mesh_quality": worst(windows[-count:]),
            }
        )
        print(json.dumps({k: v for k, v in last.items() if not k.startswith("bounds")}), flush=True)
        print(json.dumps({"phase": name, "mesh_quality": phases[-1]["mesh_quality"]}), flush=True)

    def settle() -> None:
        """Hold the open hand at the common start, then read the pose and the loop's shift."""
        nonlocal target, quat, start, grasp
        hold = route(target, target, arm_gripper.GRASP_QUAT, arm_gripper.GRASP_QUAT, width, width, 0.1)
        if hold.worst_position_error_m > MAX_IK_ERROR_M:
            raise RuntimeError("the common initial AIRBOT pose is unreachable")
        control.start(hold)
        count = round(INITIAL_SETTLE_S * CONTROL_HZ)
        for index in range(count):
            control.follow(hold, index)
            step("settle")
        position, orientation = control.arm.ee_pose(sim.pipeline.state_0)
        target, quat = np.asarray(position, dtype=np.float64), np.asarray(orientation, dtype=np.float64)
        start = target.copy()
        shift = loop_shift(entity, built, toy())
        print(json.dumps({"loop_shift_mm": np.round(1000.0 * shift, 2).tolist()}), flush=True)
        grasp = nominal + shift
        finish("settle", count)

    def act(name: str, delta, *, opening: float | None = None, seconds: float | None = None) -> None:
        """One action, run as ``part_aware_control.rollout`` runs it."""
        nonlocal target, width, worst_ik
        delta = np.asarray(delta, dtype=np.float64)
        if seconds is None:
            seconds = max(0.8, float(np.linalg.norm(delta)) / 0.06)
        end = target + delta
        end_width = width if opening is None else float(opening)
        position, orientation = control.arm.ee_pose(sim.pipeline.state_0)
        path = route(
            np.asarray(position, dtype=np.float64), end, orientation, quat, 2.0 * control.aperture(), end_width, seconds
        )
        worst_ik = max(worst_ik, path.worst_position_error_m)
        if path.worst_position_error_m > MAX_IK_ERROR_M:
            raise RuntimeError(f"{name}: the arm misses its path by {path.worst_position_error_m:.4f} m")
        for index in range(len(path.joint_q)):
            control.follow(path, index)
            step(name)
        target, width = end, end_width
        executed.append(
            {
                "phase": name,
                "delta_m": delta.tolist(),
                "seconds": seconds,
                "target_position_world_m": end.tolist(),
                "commanded_total_opening_m": end_width,
            }
        )
        finish(name, len(path.joint_q))

    def rest() -> None:
        """The CLI's closing settle: plain steps, the arm holding its last targets."""
        count = round(FINAL_SETTLE_S * CONTROL_HZ)
        for _ in range(count):
            step("rest")
        finish("rest", count)

    def steps():
        """The script. Each line is one phase; a caller may stop after any of them."""
        yield "settle", settle
        yield "align", lambda: act("align", (grasp[0] - start[0], grasp[1] - start[1], 0.0))
        yield "reach", lambda: act("reach", (0.0, 0.0, grasp[2] - start[2]))
        yield "close", lambda: act("close", (0.0, 0.0, 0.0), opening=closed, seconds=1.0)
        yield "lift_1", lambda: act("lift_1", (0.0, 0.0, lift / 3), seconds=6.0)
        yield "lift_2", lambda: act("lift_2", (0.0, 0.0, 2 * lift / 3), seconds=8.0)
        yield "hang", lambda: act("hang", (0.0, 0.0, 0.0), seconds=0.8)
        # scripted_plan carries the grasp over its target here, in 3 s; this script does not.
        yield "lower", lambda: act("lower", (0.0, 0.0, -lower), seconds=LOWER_S)
        yield "open", lambda: act("open", (0.0, 0.0, 0.0), opening=OPEN_WIDTH_M, seconds=1.0)
        yield "retreat", lambda: act("retreat", (0.0, 0.0, RETREAT_M), seconds=RETREAT_S)
        yield "rest", rest

    try:
        for name, phase in steps():
            phase()
            if name == until:
                break
        if video is not None:
            video.close()
        complete = phases[-1]["phase"] == PHASES[-1]
        # The CLI scores from the state after the initial settle on.
        judged = samples[round(INITIAL_SETTLE_S * CONTROL_HZ) - 1 :]
        place = grasp if grasp is not None else nominal
        result = judge(judged, arm_gripper.pad_bounds(sim), np.array((place[0], place[1], TABLE_Z)), worst(windows))
        if not complete:
            result.update({"success": False, "reason": f"stopped after {until}"})
        result.update(
            {
                "complete": complete,
                "lift_m": lift,
                "lower_m": lower,
                "release_clearance_m": next(
                    (s["minimum_z_m"] - REST_Z for s in reversed(samples) if s["phase"] == "lower"), None
                ),
                "actions": executed,
                "phases": phases,
                "worst_ik_error_m": worst_ik,
                "simulated_s": frame / CONTROL_HZ,
                "elapsed_s": time.perf_counter() - started,
                "geometry": geometry,
            }
        )
        (out / "diagnostics.json").write_text(json.dumps(samples, indent=2) + "\n")
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "phases", "actions")}), flush=True)
        return result
    finally:
        try:
            if video is not None:
                video.close()
        finally:
            sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument(
        "--render-every", type=int, default=2, help="render one control frame in this many to overview.mp4; 0 for none"
    )
    parser.add_argument("--until", choices=PHASES, default=PHASES[-1], help="stop after this phase")
    parser.add_argument(
        "--lift", type=float, default=None, help="how far the pads rise with the toy [m]; default the scene's lift_m"
    )
    parser.add_argument("--lower", type=float, default=LOWER_M, help="how far they come back down before opening [m]")
    args = parser.parse_args()
    run(args.out, render_every=args.render_every, until=args.until, lift=args.lift, lower=args.lower)


if __name__ == "__main__":
    main()
