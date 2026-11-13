"""Fresh-scene AIRBOT executions. No grasp, release or target motion is implicit."""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

from .actions import MAX_APERTURE_M, InvalidPlan, normalize_plan, rotate_world
from .tasks import Task, build_scene, example


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


class Rollout:
    def __init__(self, task: Task, *, video=False):
        self.task = task
        self.video = video

    def run(self, plan, index, out_dir):
        """Execute one entire sequence, starting from the common initial state."""
        import imageio.v3 as iio
        import numpy as np

        actions = normalize_plan(plan) if plan is not None else []
        out = Path(out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        if (out / "rollout.json").exists() or (out / "error.json").exists():
            raise FileExistsError(f"use a fresh rollout directory: {out}")
        started = time.perf_counter()
        sim, obj, camera, geometry, hz, hanging = build_scene(self.task)
        setup_seconds = time.perf_counter() - started
        arm_module = example("arm_gripper")
        control = arm_module.ArmControl(sim)
        snapshots, samples, executed = [], [], []
        quality = None
        if self.task.name != "cube":
            quality = example("mesh_quality").MeshQuality(sim, obj.entity)
            q_built = sim.pipeline.state_0.particle_q.numpy()[obj.entity.particles]
        target = np.asarray(geometry["common_initial_tcp_world_m"], dtype=float)
        quat = np.asarray(geometry["common_initial_quaternion_xyzw"], dtype=float)
        width = MAX_APERTURE_M
        max_ik_error = 0.0
        max_ik_orientation_error = 0.0
        video = None
        frame_count = 0
        if self.video:
            from robosimstudio.core.render.video import VideoWriter

            video = VideoWriter(out / "replay.mp4", fps=hz // 2, preset="veryfast", threads=2)

        def step():
            nonlocal frame_count
            sim.run(1).sync()
            frame_count += 1
            if video is not None and frame_count % 2 == 0:
                video.append(sim.render(camera.spec)["rgb"])

        def route_between(begin, end, q_begin, q_end, width_begin, width_end, duration):
            nonlocal max_ik_orientation_error
            frames = max(2, math.ceil(duration * hz))
            if np.dot(q_begin, q_end) < 0:
                q_end = -q_end

            def blend(t):
                x = min(1.0, t / ((frames - 1) / hz))
                return x * x * (3 - 2 * x)

            def orientation(t):
                q = q_begin + (q_end - q_begin) * blend(t)
                return q / np.linalg.norm(q)

            def tcp_at(t):
                if hanging is None:
                    return begin + (end - begin) * blend(t)
                offset = np.asarray(geometry["pad_midpoint_offset_tool_m"])
                pivot_begin = begin + arm_module._rotate(q_begin, offset)
                pivot_end = end + arm_module._rotate(q_end, offset)
                return pivot_begin + (pivot_end - pivot_begin) * blend(t) - arm_module._rotate(orientation(t), offset)

            route = control.plan(
                quat=orientation,
                capacity=frames,
                dt=1 / hz,
                tcp_at=tcp_at,
                half_gap_at=lambda t: (width_begin + (width_end - width_begin) * blend(t)) / 2,
            )
            if hanging is not None:
                full_q = sim.pipeline.state_0.joint_q.numpy().copy()
                with control.arm.host_fk():
                    for frame, joint_q in enumerate(route.joint_q):
                        full_q[control.coords] = joint_q
                        _, angular_error = control.arm.pose_errors(full_q, tcp_at(frame / hz), orientation(frame / hz))
                        max_ik_orientation_error = max(max_ik_orientation_error, float(angular_error))
                if max_ik_orientation_error > self.task.max_ik_orientation_error:
                    raise InvalidPlan(f"IK orientation misses by {math.degrees(max_ik_orientation_error):.2f} degrees")
            return route

        def state():
            position, orientation = control.arm.ee_pose(sim.pipeline.state_0)
            if self.task.name == "cube":
                pose = np.asarray(obj.pose)
                p = pose[:3]
                # Rotated cube corners give real bounds after a fall or rotation.
                signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                points = p + arm_module._rotate(pose[3:], signs * self.task.cube_half)
                item = {"position": p.tolist(), "orientation": pose[3:].tolist()}
            else:
                all_points = sim.pipeline.state_0.particle_q.numpy()
                points = all_points[obj.entity.particles]
                p = points.mean(axis=0)
                item = {"position": p.tolist()}
            finite = bool(
                np.isfinite(points).all()
                and np.isfinite(position).all()
                and np.isfinite(orientation).all()
                and np.isfinite(control.aperture())
            )
            if not finite:
                raise RuntimeError("nonfinite simulator state")
            item.update(
                {
                    "minimum_z": float(points[:, 2].min()),
                    "maximum_z": float(points[:, 2].max()),
                    "bounds_min": points.min(axis=0).tolist(),
                    "bounds_max": points.max(axis=0).tolist(),
                }
            )
            sample = {
                "time_s": float(sim.time),
                "gripper": {
                    "position": np.asarray(position).tolist(),
                    "orientation": np.asarray(orientation).tolist(),
                    "width": 2 * control.aperture(),
                    "commanded_width": 2 * arm_module.half_gap_of(float(control.hand.command)),
                },
                "objects": {self.task.name: item},
            }
            if hanging is not None:
                sample["hanging"] = hanging.read(all_points, points)
                item.update(
                    {k: v for k, v in sample["hanging"].items() if k.endswith("world_m") or k == "hole_axis_world"}
                )
            return sample

        def record(label, action_index=None):
            sample = state()
            if quality is not None:
                sample["mesh_quality"] = quality.read()
            image_path = out / f"{len(snapshots):03d}_{label}.png"
            iio.imwrite(image_path, sim.render(camera.spec)["rgb"])
            sample.update(
                {
                    "label": label,
                    "waypoint_index": len(snapshots),
                    "action_index": action_index,
                    "screenshot": str(image_path),
                }
            )
            snapshots.append(sample)
            print(json.dumps({"snapshot": label, **sample}), flush=True)
            return sample

        try:
            # Only this shared initialization uses start(). Candidate-specific
            # targets are reached exclusively by following routes and stepping.
            initial_route = route_between(target, target, quat, quat, width, width, 0.1)
            if initial_route.worst_position_error_m > self.task.max_ik_error:
                raise RuntimeError("common initial AIRBOT pose is unreachable")
            control.start(initial_route)
            for frame in range(round(self.task.initial_settle_seconds * hz)):
                control.follow(initial_route, frame)
                step()
            initial = record("initial")
            target = np.asarray(initial["gripper"]["position"])
            quat = np.asarray(initial["gripper"]["orientation"])
            if self.task.name != "cube":
                q_settled = sim.pipeline.state_0.particle_q.numpy()[obj.entity.particles]
                shift = example("clawhauser_scene").loop_shift(obj.entity, q_built, q_settled)
                initial["objects"][self.task.name]["settled_loop_shift_m"] = np.asarray(shift).tolist()
                if hanging is not None:
                    tail_shift = example("clawhauser_scene").loop_shift(obj.entity, q_built, q_settled, part=0)
                    initial["objects"][self.task.name]["settled_part_shifts_m"] = {
                        "loop": np.asarray(shift).tolist(),
                        "tail": np.asarray(tail_shift).tolist(),
                    }
            samples.append(initial)
            execution_started = time.perf_counter()
            for index_action, action in enumerate(actions):
                end = target + np.asarray(action.delta)
                end_quat = np.asarray(rotate_world(quat, action.rotation))
                end_width = width if action.width is None else action.width
                current = state()["gripper"]
                route = route_between(
                    np.asarray(current["position"]),
                    end,
                    np.asarray(current["orientation"]),
                    end_quat,
                    width if hanging is not None else current["width"],
                    end_width,
                    action.duration,
                )
                max_ik_error = max(max_ik_error, route.worst_position_error_m)
                if route.worst_position_error_m > self.task.max_ik_error:
                    raise InvalidPlan(f"action {index_action} IK misses by {route.worst_position_error_m:.4f} m")
                for frame in range(len(route.joint_q)):
                    control.follow(route, frame)
                    step()
                    samples.append(state())
                executed.append(
                    {
                        "action_index": index_action,
                        **action.to_dict(),
                        "target_position_world_m": end.tolist(),
                        "target_orientation_xyzw": end_quat.tolist(),
                        "commanded_total_opening_m": end_width,
                    }
                )
                record("waypoint", index_action)
                target, quat, width = end, end_quat, end_width
            if actions:
                for _ in range(round(self.task.settle_seconds * hz)):
                    step()
                    samples.append(state())
                record("final")
            execution_seconds = time.perf_counter() - execution_started
            if video is not None:
                video.close()
            outcome = (
                self.measure(samples, control, geometry, quality)
                if actions
                else {"success": False, "valid": True, "reason": "observation only"}
            )
            if not actions and quality is not None:
                quality_read = quality.read()
                outcome["mesh_quality"] = quality_read
                outcome["valid"] = not (
                    sum(quality_read["inverted_tet_substeps_by_part"])
                    or sum(quality_read["nonfinite_tet_substeps_by_part"])
                )
            root = Path(__file__).resolve().parents[1]
            result = {
                "schema_version": 1,
                "index": index,
                "timestamp": str(time.time_ns()),
                "task": self.task.to_dict(),
                "geometry": geometry,
                "privileged_simulation_state": True,
                "revisions": {"controller": revision(root), "newton": revision(root.parent / "newton")},
                "controller_has_local_changes": bool(
                    subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
                ),
                "plan": plan.to_dict() if hasattr(plan, "to_dict") else plan,
                "executed_actions": executed,
                "snapshots": snapshots,
                "outcome": outcome,
                "max_ik_position_error_m": max_ik_error,
                "max_ik_orientation_error_rad": max_ik_orientation_error,
                "timing": {
                    "build_seconds": setup_seconds,
                    "execution_with_readback_and_images_seconds": execution_seconds,
                    "wall_seconds": time.perf_counter() - started,
                },
            }
            if video is not None:
                result["video"] = {"path": str(video.path), "fps": video.fps, "frames": video.frames}
            path = out / "rollout.json"
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            if plan is None:
                (out / "observation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            return path
        except Exception as error:
            # Preserve the physical execution evidence even if a later waypoint
            # is unreachable or simulation/rendering fails.
            (out / "error.json").write_text(
                json.dumps(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "executed_actions": executed,
                        "snapshots": snapshots,
                        "wall_seconds": time.perf_counter() - started,
                    },
                    indent=2,
                )
                + "\n"
            )
            raise
        finally:
            try:
                if video is not None:
                    video.close()
            finally:
                sim.close()

    def measure(self, samples, control, geometry, quality):
        import numpy as np

        if self.task.name == "clawhauser_hang":
            from .hanging import measure_hanging

            return measure_hanging(self.task, samples, geometry, quality.read())

        name = self.task.name
        start, final = samples[0]["objects"][name], samples[-1]["objects"][name]
        end_time = samples[-1]["time_s"]
        tail = [s for s in samples if s["time_s"] >= end_time - self.task.settle_window_seconds]
        positions = np.asarray([s["objects"][name]["position"] for s in tail])
        elapsed = tail[-1]["time_s"] - tail[0]["time_s"]
        drift = float(np.linalg.norm(positions - positions[-1], axis=1).max())
        speed = (
            float(
                np.linalg.norm(np.diff(positions, axis=0), axis=1).max()
                / max(np.diff([s["time_s"] for s in tail]).min(), 1e-9)
            )
            if len(tail) > 1
            else float("inf")
        )
        lifted = max(s["objects"][name]["minimum_z"] for s in samples) - start["minimum_z"]
        center = np.asarray(geometry["target_center_world_m"])
        half = np.asarray(self.task.target_half)
        if name == "cube":
            placed = bool(
                np.all(np.asarray(final["bounds_min"])[:2] >= center[:2] - half)
                and np.all(np.asarray(final["bounds_max"])[:2] <= center[:2] + half)
            )
        else:
            placed = bool(np.all(np.abs(positions[:, :2] - center[:2]) <= half))
        low, high = example("arm_gripper").pad_bounds(control.sim)
        # Disjoint finger/object bounds prove no final pad contact, independently
        # of contact-buffer conventions in the two different physics backends.
        clear = bool(
            np.any(low > np.asarray(final["bounds_max"]) + 0.010)
            or np.any(high < np.asarray(final["bounds_min"]) - 0.010)
        )
        released = bool(min(s["gripper"]["width"] for s in tail) >= 0.060 and clear)
        resting_height = 0.75 + (0.003 if name == "clawhauser" else 0.0)
        supported = all(
            abs(s["objects"][name]["minimum_z"] - resting_height) <= self.task.surface_tolerance for s in tail
        )
        settled = bool(
            elapsed >= 0.45 and drift <= self.task.max_settled_drift and speed <= self.task.max_settled_speed
        )
        quality_read = quality.read() if quality else None
        valid = (
            True
            if quality is None
            else not (
                sum(quality_read["inverted_tet_substeps_by_part"])
                or sum(quality_read["nonfinite_tet_substeps_by_part"])
            )
        )
        table_penetration = max(0.0, 0.75 - min(s["objects"][name]["minimum_z"] for s in samples))
        valid = valid and table_penetration <= self.task.max_table_penetration
        gates = {
            "valid": bool(valid),
            "lifted": lifted >= self.task.minimum_lift,
            "placed": placed,
            "released": released,
            "supported": supported,
            "settled": settled,
        }
        return {
            "success": all(gates.values()),
            **gates,
            "lifted_m": lifted,
            "final_position_world_m": final["position"],
            "final_drift_m": drift,
            "final_peak_speed_m_s": speed,
            "mesh_quality": quality_read,
            "max_table_penetration_m": table_penetration,
            "reason": ", ".join(k for k, v in gates.items() if not v) or "all measured gates passed",
        }
