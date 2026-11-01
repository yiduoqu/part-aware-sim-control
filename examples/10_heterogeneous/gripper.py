# SPDX-License-Identifier: Apache-2.0
"""A two-finger kinematic gripper driven through RSS substep hooks."""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

import robosimstudio as rss


def smooth(time: float, start: float, end: float) -> float:
    t = float(np.clip((time - start) / (end - start), 0.0, 1.0))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


@wp.kernel
def _drive_fingers(
    tick: wp.array[int],
    bodies: wp.array[int],
    poses: wp.array2d[wp.transform],
    velocities: wp.array2d[wp.spatial_vector],
    q0: wp.array[wp.transform],
    q1: wp.array[wp.transform],
    qd0: wp.array[wp.spatial_vector],
    qd1: wp.array[wp.spatial_vector],
):
    finger = wp.tid()
    body = bodies[finger]
    frame = tick[0]
    q0[body] = poses[frame, finger]
    q1[body] = poses[frame, finger]
    qd0[body] = velocities[frame, finger]
    qd1[body] = velocities[frame, finger]


@wp.kernel
def _advance_tick(tick: wp.array[int]):
    tick[0] = tick[0] + 1


@dataclass(frozen=True, kw_only=True)
class ScriptedGripperSpec:
    entity: str = "scripted_loop_gripper"
    name: str = "loop_gripper"
    center: tuple[float, float, float] = (0.0, 0.0, 0.02)
    half_gap: float = 0.015
    closed_half_gap: float = 0.005
    release_half_gap: float = 0.070
    approach_z: float = -0.030
    travel_x: float = 0.18
    lift_z: float = 0.40
    lower_offset_z: float = 0.295
    release_lower_z: float = 0.17
    settle_s: float = 0.0
    airborne_release: bool = False
    control_hz: int = 30
    substeps: int = 20
    duration: float = 16.0


@rss.register_entity("scripted_loop_gripper")
class ScriptedGripper(rss.Entity):
    domain = "rigid"

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        center = self.placed((*spec.center, 0.0, 0.0, 0.0, 1.0))[:3]
        self.center = np.asarray(center, dtype=np.float64)
        cfg = newton.ModelBuilder.ShapeConfig(mu=2.0, ke=1.0e5, kd=25.0, gap=0.001)
        self.bodies = []
        self.finger_shapes = []
        self.shapes_by_side = []
        for side in (-1, 1):
            pos = self.center + np.array([side * spec.half_gap, 0.0, 0.0])
            body = builder.add_body(
                xform=wp.transform(pos, wp.quat_identity()),
                is_kinematic=True,
                label=f"{spec.name}_{side:+d}",
            )
            shape = builder.add_shape_box(
                body,
                hx=0.003,
                hy=0.020,
                hz=0.006,
                cfg=cfg,
                color=(0.16, 0.30, 0.39),
                label=f"{spec.name}_{side:+d}_pad",
            )
            self.bodies.append(body)
            self.finger_shapes.append(shape)
            # A shallow ledge slides under the flat loop as the jaws close.
            # It carries the loop during lift through ordinary rigid-soft
            # contact, without welding particles to either finger.
            ledge = builder.add_shape_box(
                body,
                xform=wp.transform(wp.vec3(0.0, 0.0, -0.0068), wp.quat_identity()),
                hx=0.006,
                hy=0.020,
                hz=0.0012,
                cfg=cfg,
                color=(0.16, 0.30, 0.39),
                label=f"{spec.name}_{side:+d}_ledge",
            )
            self.finger_shapes.append(ledge)
            self.shapes_by_side.append((shape, ledge))

    def on_scene_ready(self, pipeline) -> None:
        self.pipeline = pipeline
        spec = self.spec
        count = int(np.ceil(spec.duration * spec.control_hz * spec.substeps)) + 2
        dt = 1.0 / (spec.control_hz * spec.substeps)
        poses = np.zeros((count, 2, 7), dtype=np.float32)
        poses[:, :, 6] = 1.0
        angles = np.zeros(count, dtype=np.float32)
        for index in range(count):
            time = index * dt - spec.settle_s
            # Reserve two seconds after transport for an optional descent.
            release_time = time - (2.0 if spec.airborne_release and spec.release_lower_z else 0.0)
            close = smooth(time, 0.40, 0.85)
            grasp_gap = spec.half_gap + (spec.closed_half_gap - spec.half_gap) * close
            left_open = (6.0, 6.5) if spec.airborne_release else (10.7, 11.2)
            # Clear the left jaw, tip the loaded right finger, then withdraw
            # it. Opening the right jaw first carries the loop sideways.
            right_open = (8.0, 8.5) if spec.airborne_release else (12.5, 13.0)
            left_gap = grasp_gap + (spec.release_half_gap - grasp_gap) * smooth(release_time, *left_open)
            right_gap = grasp_gap + (spec.release_half_gap - grasp_gap) * smooth(release_time, *right_open)
            approach = spec.approach_z * smooth(time, 0.05, 0.35)
            lower = 0.0 if spec.airborne_release else smooth(time, 5.6, 9.4)
            lift = spec.lift_z * smooth(time, 1.2, 2.8) * (1.0 - lower)
            move = spec.travel_x * smooth(time, 3.0, 4.8)
            retreat = 0.0 if spec.airborne_release else smooth(time, 13.6, 14.4)
            release_lower = (
                spec.release_lower_z * smooth(time, 5.5, 7.5)
                if spec.airborne_release
                else (spec.release_lower_z * smooth(time, 8.5, 9.4))
            )
            center = self.center + np.array(
                [move, -0.12 * retreat, approach + lift + spec.lower_offset_z * lower - release_lower + 0.08 * retreat]
            )
            poses[index, 0, :3] = center + np.array([-left_gap, 0.0, 0.0])
            poses[index, 1, :3] = center + np.array([right_gap, 0.0, 0.0])
            # Tip the supporting right ledge downward so the loop can slide
            # off. Release holds the midpoint fixed while tipping.
            tip = (
                smooth(release_time, 7.0, 7.5)
                if spec.airborne_release
                else smooth(time, 11.7, 12.5) * (1.0 - smooth(time, 13.0, 13.5))
            )
            angle = -0.5 * np.pi * tip
            angles[index] = angle
            poses[index, 1, 4] = np.sin(angle / 2.0)
            poses[index, 1, 6] = np.cos(angle / 2.0)
        velocities = np.zeros((count, 2, 6), dtype=np.float32)
        velocities[:-1, :, :3] = np.diff(poses[:, :, :3], axis=0) / dt
        # The right finger rotates about the world y axis. Contact friction
        # needs its angular velocity as well as its prescribed orientation.
        velocities[:-1, 1, 4] = np.diff(angles) / dt
        self._poses = wp.array(poses, dtype=wp.transform, device=pipeline.model.device)
        self._velocities = wp.array(velocities, dtype=wp.spatial_vector, device=pipeline.model.device)
        self._body_indices = wp.array(self.bodies, dtype=int, device=pipeline.model.device)
        self._tick = wp.zeros(1, dtype=int, device=pipeline.model.device)

    def fit_grasp(self, points: np.ndarray) -> None:
        """Align the approach once, after settling and before finger motion."""
        target = points.mean(axis=0) + np.array([0.0, 0.0, 0.003])
        offset = target - self.center - np.array([0.0, 0.0, self.spec.approach_z])
        poses, velocities = self._poses.numpy(), self._velocities.numpy()
        dt = 1.0 / (self.spec.control_hz * self.spec.substeps)
        blend = np.array([smooth(i * dt - self.spec.settle_s, 0.05, 0.35) for i in range(len(poses))])
        poses[:, :, :3] += blend[:, None, None] * offset
        velocities[:-1, :, :3] = np.diff(poses[:, :, :3], axis=0) / dt
        self._poses.assign(poses)
        self._velocities.assign(velocities)

    def contribute_hooks(self, hooks) -> None:
        hooks.add("pre_solve", self._drive, name="drive-loop-fingers")
        hooks.add("post_solve", self._advance, name="advance-loop-fingers")

    def _drive(self, state, dt) -> None:
        del dt
        wp.launch(
            _drive_fingers,
            dim=2,
            inputs=[
                self._tick,
                self._body_indices,
                self._poses,
                self._velocities,
                state.body_q,
                self.pipeline.state_1.body_q,
                state.body_qd,
                self.pipeline.state_1.body_qd,
            ],
        )

    def _advance(self, state, dt) -> None:
        del state, dt
        wp.launch(_advance_tick, dim=1, inputs=[self._tick])

    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        del pipeline, episode, world
        if hasattr(self, "_tick"):
            self._tick.zero_()
