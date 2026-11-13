"""Small task definitions. Simulator imports stay out of the planner process."""

from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "10_heterogeneous"


def example(name):
    # Reuse the existing example modules and their sibling imports as shipped.
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    return importlib.import_module(name)


@dataclass(frozen=True)
class Task:
    name: str = "cube"
    cube_start: tuple[float, float] = (-0.13, 0.0)  # Work-surface frame.
    target_xy: tuple[float, float] = (0.02, 0.0)
    target_half: tuple[float, float] = (0.06, 0.06)
    cube_half: float = 0.020
    minimum_lift: float = 0.040
    settle_seconds: float = 2.0
    initial_settle_seconds: float = 1.0
    settle_window_seconds: float = 0.5
    max_settled_speed: float = 0.01
    max_settled_drift: float = 0.003
    surface_tolerance: float = 0.002
    max_table_penetration: float = 0.005
    max_ik_error: float = 0.010
    max_ik_orientation_error: float = 0.08726646259971647  # Five degrees; hanging only.
    minimum_hang_clearance: float = 0.020
    minimum_rod_support_fraction: float = 0.90
    minimum_rod_engagement: float = 0.010
    max_contact_penetration: float = 0.0005

    def to_dict(self):
        return asdict(self)


def build_scene(task):
    import robosimstudio as rss

    arm = example("arm_gripper")
    hanging = None
    if task.name == "clawhauser_hang":
        from .hanging import HangingState

        native = example("clawhauser_on_rod")
        sim, obj, _, camera, rod, ring, geometry = native.build_scene()
        origin = rss.get_cell(arm.CELL).surface_frame[:3]
        rod_origin = [
            origin[0] + native.SOURCE_X + float(ring["hole_centre_local"][0]),
            origin[1] + native.ROD_FREE_Y,
            origin[2] + geometry["rod_z_m"],
        ]
        geometry.update(
            {
                "physics": "clawhauser_on_rod.build_scene defaults (unchanged)",
                "material_settings": asdict(example("clawhauser_scene").REFERENCE),
                "rod_free_end_world_m": rod_origin,
                "rod_axis_world": [0.0, 1.0, 0.0],
                "rod_length_m": native.POST_Y - native.POST_RADIUS - native.ROD_FREE_Y,
                "rest_surface_z_world_m": native.REST_Z,
                "weight_n": float(obj.entity.mass.sum() * 9.81),
                "finger_force_tolerance": arm.GRIP_TOLERANCE,
                "target_kind": "ring threaded onto rod; rod-supported after release, clear of table",
                "pose_interpolation": "TCP endpoints in world metres; pad midpoint follows a straight line while wrist rotates",
            }
        )
        hanging = HangingState(sim, obj, rod, ring, geometry)
        hz = native.CONTROL_HZ
    elif task.name == "clawhauser":
        sim, obj, _, camera, geometry = example("clawhauser_pick_drop").build_scene()
        geometry["physics"] = "clawhauser_pick_drop.build_scene defaults (unchanged)"
        geometry["material_settings"] = asdict(example("clawhauser_scene").REFERENCE)
        hz = 60
    elif task.name == "cube":
        scene = rss.Scene(cell=arm.CELL, physics=rss.MUJOCO_500HZ, name="simpact_cube")
        obj = scene.add(
            rss.box(
                name="cube",
                half=(task.cube_half,) * 3,
                at=(*task.cube_start, task.cube_half + 0.001),
                mass=0.05,
                friction=1.0,
                color=(0.9, 0.35, 0.12),
            )
        )
        scene.add(
            rss.slab(
                name="target", half=(*task.target_half, 0.0001), at=(*task.target_xy, 0.0001), color=(0.15, 0.65, 0.25)
            )
        )
        arm.add_arm(scene, base_offset=arm.MOUNT)
        camera = scene.add(
            rss.CameraSpec(
                name="overview", width=640, height=480, eye=(1.05, -0.55, 1.18), look_at=(0.53, 0.0, 0.85), fov_y=45
            )
        )
        sim = scene.build(device="cuda:0", warmup=False, use_cuda_graph=False)
        geometry = {
            "physics": "MUJOCO_500HZ",
            "cube_side_m": 2 * task.cube_half,
            "cube_mass_kg": 0.05,
            "cube_friction": 1.0,
        }
        hz = 50
    else:
        raise ValueError(f"unknown task {task.name!r}")
    geometry.update(
        {
            "arrangement": asdict(sim.request.physics),
            "max_aperture_m": arm.SPAN,
            "pad_size_tool_xyz_m": [0.0105, 0.004, 0.040],
            "close_axis_tool": list(arm.CLOSE_AXIS),
            "finger_effort_limit_n": arm.GRASP_LIMIT_N,
            "quaternion_order": "xyzw",
            "action_frame": "world",
            "width_convention": "total jaw opening in metres",
            "surface_origin_world_m": list(rss.get_cell(arm.CELL).surface_frame[:3]),
            "target_center_world_m": [0.6 + task.target_xy[0], task.target_xy[1], 0.75],
            "target_half_xy_m": list(task.target_half),
            "common_initial_tcp_world_m": [0.47, 0.0, 1.00],
            "common_initial_quaternion_xyzw": list(arm.GRASP_QUAT),
        }
    )
    if task.name in {"clawhauser", "clawhauser_hang"}:
        geometry.update(
            {
                "grip_centre_frame": "work_surface, nominal before initial settling",
                "nominal_grip_centre_world_m": [
                    x + y for x, y in zip(geometry["surface_origin_world_m"], geometry["grip_centre_m"], strict=True)
                ],
                "tcp_for_pads_offset_world_m": [0.0, 0.0, -arm.PAD_RISE],
                "grasp_reference_note": (
                    "For the supplied downward grasp orientation, add the observed world-frame "
                    "settled_loop_shift_m and tcp_for_pads_offset_world_m to the nominal grip centre."
                ),
            }
        )
    if hanging is not None:
        geometry.pop("target_center_world_m")
        geometry.pop("target_half_xy_m")
        inverse_grasp = (*(-v for v in arm.GRASP_QUAT[:3]), arm.GRASP_QUAT[3])
        geometry["pad_midpoint_offset_tool_m"] = arm._rotate(inverse_grasp, [0.0, 0.0, arm.PAD_RISE]).tolist()
        vertices, tets, labels = example("clawhauser_scene").load_toy()
        tail_xy, tail_reach = example("clawhauser_scene").tail_grasp(vertices, tets, labels)
        geometry["grasp_candidates"] = {
            "loop": {
                "part_id": 2,
                "nominal_pad_center_world_m": geometry["nominal_grip_centre_world_m"],
                "outer_width_m": 2 * geometry["reach_m"],
            },
            "tail": {
                "part_id": 0,
                "nominal_pad_center_world_m": [
                    origin[0] + native.SOURCE_X + float(tail_xy[0]),
                    origin[1] + float(tail_xy[1] - ring["hole_centre_local"][1]),
                    origin[2] + geometry["grip_z_m"],
                ],
                "outer_width_m": 2 * tail_reach,
            },
        }
        geometry["grasp_reference_note"] = (
            "Choose a loop or tail pickup; final success still requires hanging by the loop. "
            "For a downward grasp add the selected nominal pad center, its matching "
            "settled_part_shifts_m and tcp_for_pads_offset_world_m. "
            "Moves and holds maintain the previous commanded jaw opening; only an explicit "
            "gripper_control changes it."
        )
    return sim, obj, camera, geometry, hz, hanging


def scripted_plan(observation):
    """An explicit input plan; the rollout executor never adds grasp/transport actions."""
    task, geometry = observation["task"], observation["geometry"]
    if task["name"] == "clawhauser_hang":
        raise ValueError("author an explicit hanging plan from the saved Hanson reference, then use worker --plan")
    initial = observation["snapshots"][0]
    position = initial["gripper"]["position"]
    if task["name"] == "cube":
        obj = initial["objects"]["cube"]
        grasp = list(obj["position"])
        grasp[2] = max(grasp[2], 0.773)
        closed, lift = 0.030, 0.12
    else:
        origin = geometry["surface_origin_world_m"]
        grasp = [x + y for x, y in zip(origin, geometry["grip_centre_m"], strict=True)]
        grasp[2] += 0.014  # Existing tcp_for_pads convention.
        # Account for the settled loop in task definition, before any execution.
        shift = initial["objects"]["clawhauser"].get("settled_loop_shift_m", [0, 0, 0])
        grasp = [x + y for x, y in zip(grasp, shift, strict=True)]
        closed = max(0.0, 2 * (geometry["loop_reach_m"] - geometry["grip_squeeze_m"]))
        lift = geometry["lift_m"]
    target = geometry["target_center_world_m"]
    # Center the held grasp over the target. The hanging toy's centroid differs
    # from its centroid while lying flat in the initial observation.
    transport = [target[i] - grasp[i] for i in range(2)]
    actions = []

    def move(delta, duration=None):
        action = {
            "type": "move",
            "delta_x": delta[0],
            "delta_y": delta[1],
            "delta_z": delta[2],
            "delta_roll": 0.0,
            "delta_pitch": 0.0,
            "delta_yaw": 0.0,
        }
        if duration is not None:
            action["duration"] = duration
        actions.append(action)

    move((grasp[0] - position[0], grasp[1] - position[1], 0.0))
    move((0.0, 0.0, grasp[2] - position[2]))
    actions.append({"type": "gripper_control", "width": closed, "duration": 1.0})
    if task["name"] == "clawhauser":
        move((0.0, 0.0, lift / 3), 6.0)
        move((0.0, 0.0, 2 * lift / 3), 8.0)
    else:
        move((0.0, 0.0, lift), 2.0)
    move((0.0, 0.0, 0.0), 0.8)
    move((*transport, 0.0), 3.0)
    if task["name"] == "clawhauser":
        # Release above first table contact, letting the hanging body settle
        # without forcing it flat while its loop remains clamped.
        move((0.0, 0.0, -0.14), 8.0)
    else:
        move((0.0, 0.0, -lift), 2.0)
    actions.append({"type": "RELEASE", "duration": 1.0})
    move((0.0, 0.0, 0.12), 2.0)
    return {
        "description": "Explicit approach, pinch, lift, transport, lower, release and withdraw",
        "action_sequence": actions,
    }
