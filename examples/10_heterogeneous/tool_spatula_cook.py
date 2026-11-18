# SPDX-License-Identifier: Apache-2.0
"""TT4: cook with the silicone spatula, pushing food across a rigid frying pan.

The AIRBOT picks the spatula up by its handle (near the butt), lifts it high, tilts it tip down about 40
degrees, lowers the head steeply onto the pan floor behind the food and pushes straight across the bowl.
Soft tet cubes and rigid crumbs start in the near half; success is >= 70 % of them in the far half and none
out of the bowl.

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
import rigid_part_body
import tool_scene
from clawhauser_scene import REFERENCE
from loop_geometry import smooth
from scipy.spatial.transform import Rotation
from tool_asset_check import kabsch

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish
from robosimstudio.specs import AssetSpec, StaticSpec
from robosimstudio_materials.heterogeneous import HeterogeneousSoftSpec

DURATION = 11.0
SETTLE = 0.4
HOVER, PAD_DROP = 0.04, 0.0225
"""The pads hover this far over the grip, and the hand's tool point sits this high [m] when the pad bottoms are 2.5 mm over the bench."""
PHASES = {"descend": (0.55, 1.0), "close": (1.1, 1.7), "lift": (2.0, 3.2)}
SQUEEZE = 0.016
SETTINGS = dataclasses.replace(REFERENCE, substeps=tool_scene.SUBSTEPS)
HOLDER = dataclasses.replace(tool_scene.TOOLS["silicone_spatula"], grip_frac=0.2)
"""The spatula, gripped near the butt of its handle."""
SPATULA_AT = (0.0, -0.15)
"""Where the grip sits on the bench; the spatula lies along +x with its head toward the pan."""
PAN_ASSET = Path(__file__).resolve().parent / "assets" / "rigid_pan"
PAN_AT, PAN_FLOOR, PAN_TOP, BOWL_R = (0.0, 0.06), 0.004, 0.038, 0.115
"""The rigid pan's bowl centre (handle along +y, away from the arm), its floor and rim heights and inner radius at the rim [m]."""
LOW = PAN_FLOOR - 0.020
"""The head's commanded lowest point on the floor: 20 mm under it: the head stops about 3 mm over the floor whatever the depth, the depth sets the press."""
SPAWN_GAP = 0.004
"""Soft food spawns this far over the floor: touching it threw the cubes 200 mm up."""
FLAT_TILT = np.radians(25.0)
"""The tilt on the floor, so the head's broad edge leads the push."""
ENTRY_TILT = np.radians(58.0)
"""The spatula's tip-down tilt through the carry and touchdown, steep enough for the handle to clear the rim."""
RIM_R, RIM_Z = 0.1175, PAN_TOP
"""The pan rim's top circle: radius from the bowl axis and height [m]."""
CUBE_AT = ((-0.025, 0.045), (-0.025, 0.075), (0.0, 0.06))
CUBE = 0.02
CRUMB_X, CRUMB_Y = (0.01, 0.025), (0.03, 0.09)
START_X, END_X = -0.07, 0.06
"""The head's tip starts on the floor 35 mm behind the food's rear edge (rim-safe: its corners stay inside r 0.08)."""
FAR_X = 0.03
"""Food past this x counts as moved to the far half."""
CAMERA = ((0.22, 0.30, 0.22), (0.0, 0.06, 0.02))


@dataclasses.dataclass(frozen=True, kw_only=True)
class FoodSpec(HeterogeneousSoftSpec):
    """A soft body with a display colour: the renderer reads ``spec.color``, which the stock spec lacks (it draws white)."""

    color: tuple | None = None
    finish: Finish | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ColouredToolSpec(rigid_part_body.RigidPartSoftSpec):
    color: tuple | None = None
    finish: Finish | None = None


def colour_tool(color) -> None:
    """Make the tool scene's soft tool spec carry ``color`` (patching ``tool_scene.spec`` for this process)."""
    plain = tool_scene.spec

    def coloured(*args, **kwargs):
        spec = plain(*args, **kwargs)
        return ColouredToolSpec(
            **{f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)},
            color=color,
            finish=Finish(roughness=0.5),
        )

    tool_scene.spec = coloured


def steel_pan() -> list:
    """The rigid pan asset as a concave static triangle mesh in dark steel, its handle turned along +y."""
    yaw = (0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4)))
    return [
        StaticSpec(
            name="pan",
            asset=AssetSpec(path=str(PAN_ASSET / "physics" / "collision_mesh.obj")),
            xform=(PAN_AT[0], PAN_AT[1], 0.0, *yaw),
            color=(0.30, 0.32, 0.35),
            finish=Finish(roughness=0.25, metallic=1.0),
        )
    ]


def lattice(size, cells, keep=None):
    """A box of ``cells`` lattice cells (six tets each) ``size`` [m] across, centred on x and y with its bottom at z = 0.

    ``keep(centre_xy)`` optionally drops cells (a round disc); returns ``(vertices, tets)`` without unused vertices.
    """
    n = np.asarray(cells)
    grid = np.stack(np.meshgrid(*[np.arange(k + 1) for k in n], indexing="ij"), axis=-1).reshape(-1, 3)
    vertices = grid * (np.asarray(size) / n) - (0.5 * size[0], 0.5 * size[1], 0.0)
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    six = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
    cell = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), axis=-1).reshape(-1, 3)
    if keep is not None:
        centre = vertices[np.ravel_multi_index(tuple(cell.T), n + 1)] + 0.5 * np.asarray(size) / n
        cell = cell[[keep(c[:2]) for c in centre]]
    ids = np.ravel_multi_index(tuple(np.moveaxis(cell[:, None] + corners, -1, 0)), n + 1)
    tets = ids[:, six].reshape(-1, 4)
    used, tets = np.unique(tets, return_inverse=True)
    return vertices[used], tets.reshape(-1, 4)


def soft_solid(name, vertices, tets, *, at, young, settings, mu, density=1000.0, nu=0.45, color=(0.9, 0.7, 0.3)):
    """A single-material soft body spec for a tet mesh ``vertices``, ``tets``, resting with its origin at ``at`` [m].

    Written under ``outputs/soft_solids`` as the labelled-mesh files the tool loader reads.
    """
    folder = tool_scene.OUTPUT / "soft_solids" / name
    folder.mkdir(parents=True, exist_ok=True)
    mesh = ["MeshVersionFormatted 1", "Dimension 3", "Vertices", str(len(vertices))]
    mesh += [f"{x:.9g} {y:.9g} {z:.9g} 0" for x, y, z in vertices]
    mesh += ["Tetrahedra", str(len(tets))] + [" ".join(str(int(i) + 1) for i in t) + " 0" for t in tets] + ["End"]
    (folder / "final_mesh.mesh").write_text("\n".join(mesh) + "\n")
    np.savez(
        folder / "heterogeneous_params.npz",
        tet_part_labels=np.zeros(len(tets), dtype=np.int64),
        tet_E_nu=np.tile((young, nu), (len(tets), 1)),
        tet_density=np.full(len(tets), density),
    )
    part = {
        "part_index": 0,
        "youngs_modulus_pa": young,
        "poisson_ratio": nu,
        "density_kg_m3": density,
        "part_color_rgb": [round(255 * c) for c in color],
    }
    (folder / "inferred_params.json").write_text(json.dumps({"predictions": [part]}))
    return FoodSpec(
        color=color,
        finish=Finish(roughness=0.6),
        name=name,
        mesh_path=str(folder / "final_mesh.mesh"),
        params_path=str(folder / "heterogeneous_params.npz"),
        materials_path=str(folder / "inferred_params.json"),
        pose=tuple(float(c) for c in at),
        contact_ke=settings.contact_ke,
        contact_kd=settings.contact_kd,
        contact_mu=mu,
        particle_radius=settings.particle_radius,
        damping_ratio=settings.damping_ratio,
        part_overrides=((0, "damping_pa_s", 50.0),),
    )


def pan_extras(scene, settings, mu, cubes: list, crumb_names: list, young: float, n_crumbs: int) -> list:
    """The pan and its food; ``cubes`` and ``crumb_names`` are filled with what was added."""
    specs = steel_pan()
    for i, (x, y) in enumerate(CUBE_AT):
        vertices, tets = lattice((CUBE,) * 3, (4, 4, 4))
        cubes.append(
            scene.add(
                soft_solid(
                    f"food{i}",
                    vertices,
                    tets,
                    at=(x, y, PAN_FLOOR + SPAWN_GAP),
                    young=young,
                    settings=settings,
                    mu=mu,
                    color=(0.93, 0.72, 0.30),
                )
            )
        )
    crumb_names += scatter_crumbs(scene, n_crumbs)
    return specs


def scatter_crumbs(scene, n: int) -> list[str]:
    """``n`` rigid spheres and cubes (6-8 mm, bigger than the head's contact gap) on a jittered grid in the pan's near end, so none starts overlapping."""
    rng = np.random.default_rng(2)
    rows = 2
    xs = np.linspace(*CRUMB_X, rows)
    ys = np.linspace(*CRUMB_Y, -(-n // rows))
    names = []
    for i in range(n):
        edge = float(rng.uniform(0.006, 0.008))
        at = (
            float(xs[i % rows]),
            float(ys[i // rows]) + float(rng.uniform(-0.003, 0.003)),
            PAN_FLOOR + 0.5 * edge + 0.003,
        )
        shade = float(rng.uniform(0.25, 0.45))
        kwargs = {"name": f"crumb{i}", "at": at, "color": (shade + 0.15, shade, shade * 0.6)}
        if i < n // 2:
            scene.add(rss.sphere(radius=0.5 * edge, mass=1500.0 * 4 / 3 * np.pi * (0.5 * edge) ** 3, **kwargs))
        else:
            scene.add(rss.box(half=(0.5 * edge,) * 3, mass=1500.0 * edge**3, **kwargs))
        names.append(f"crumb{i}")
    return names


def food_positions(sim, cubes, crumb_names) -> np.ndarray:
    """Centres of the cubes and the crumbs [m, bench frame]."""
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    q = sim.pipeline.state_0.particle_q.numpy()
    centres = [q[c.entity.particles].mean(axis=0) for c in cubes]
    return (
        np.vstack([np.asarray(centres, dtype=np.float64).reshape(-1, 3), tool_scene.crumb_positions(sim, crumb_names)])
        - surface
    )


def judge_food(p: np.ndarray) -> dict:
    """The food in the far half of the pan, and what is out of the bowl."""
    inside = (np.hypot(p[:, 0] - PAN_AT[0], p[:, 1] - PAN_AT[1]) < BOWL_R) & (p[:, 2] > 0.0)
    far = inside & (p[:, 0] > FAR_X)
    return {
        "moved_fraction": float(far.mean()),
        "moved": int(far.sum()),
        "out_of_pan": int((~inside).sum()),
        "food": len(p),
        "mean_x_m": float(p[:, 0].mean()),
        "positions_mm": np.round(1e3 * p).astype(int).tolist(),
    }


def rim_distance(local: np.ndarray) -> float:
    """The least distance [m] from any tool node (bench frame) to the pan rim's top circle."""
    r = np.hypot(local[:, 0] - PAN_AT[0], local[:, 1] - PAN_AT[1])
    return float(np.hypot(r - RIM_R, local[:, 2] - RIM_Z).min())


def hand_motion(grasp: dict, head: np.ndarray, tip: int, target, surface: np.ndarray):
    """``tcp_at``, ``quat_at`` and ``half_gap_at`` that carry the tool's head as ``target(t)`` says.

    ``head`` are the head's nodes (bench frame, resting) and ``tip`` the front one; ``target(t)`` gives
    the tip's xy, the head's lowest z and the tilt [rad] about the closing axis, from the lift on. Before it
    the hand holds still at the grip; the tool is rigid in the hand, so the head's pose follows from the tilt.
    """
    across = np.asarray(grasp["across"])
    pivot = np.array(grasp["point_m"])
    base = surface + np.array((*grasp["point_m"][:2], PAD_DROP))
    open_half = min(0.04, grasp["half_width_m"] + 0.014)
    half_close = grasp["half_width_m"] - SQUEEZE

    def at(t: float):
        tip_xy, low, angle = target(max(t, PHASES["lift"][0]))
        turned = Rotation.from_rotvec(across * angle).apply(head - pivot) + pivot
        shift_xy = np.asarray(tip_xy) - turned[tip, :2]
        return shift_xy, low - float(turned[:, 2].min()), angle

    def tcp_at(t: float) -> np.ndarray:
        shift, rise, _ = at(t)
        hover = HOVER * (1.0 - smooth(t, *PHASES["descend"]))
        return base + np.array((*shift, rise + hover))

    quat = tool_scene.tilt_quat(grasp, lambda t: at(t)[2])

    def half_gap_at(t: float) -> float:
        return open_half + (half_close - open_half) * smooth(t, *PHASES["close"])

    return tcp_at, quat, half_gap_at


def rim_tilt(x: float, base: float = FLAT_TILT) -> float:
    """The flattest tilt [rad] that keeps the tool 10 mm over the near rim with the tip on the floor at ``x``."""
    return max(base, float(np.arctan2(PAN_TOP + 0.010 - (PAN_FLOOR + 0.003), x - (PAN_AT[0] - RIM_R))))


def keyframes(tip0: np.ndarray, low0: float, level: float, variant: int = 0):
    entry = ENTRY_TILT - np.radians(8.0 * variant)
    """Lift the spatula high, tilt it tip down, carry the head over the food's near side, lower it steeply onto
    the floor, push straight across the bowl, then lift."""
    lift, over, down, push, up = PHASES["lift"], (3.4, 5.0), (5.0, 5.8), (5.8, 9.5), (9.5, 10.3)
    start, end = np.array((START_X, PAN_AT[1])), np.array((END_X, PAN_AT[1]))
    carry = 0.10

    def target(t: float):
        if t < over[0]:
            s = smooth(t, *lift)
            return tip0, low0 + 0.12 * s, level * s
        if t < down[0]:
            s = smooth(t, *over)
            return (1 - s) * tip0 + s * start, low0 + 0.12 + s * (carry - low0 - 0.12), level + s * (entry - level)
        if t < push[0]:
            return start, carry + (LOW - carry) * smooth(t, *down), entry
        if t < up[0]:
            xy = start + (end - start) * smooth(t, *push)
            tilt = entry + (min(entry, rim_tilt(xy[0])) - entry) * smooth(t, push[0], push[0] + 0.8)
            return xy, LOW, tilt
        return end, LOW + 0.06 * smooth(t, *up), rim_tilt(END_X)

    return target


def run(out: Path, asset, args, task: dict) -> dict:
    """Pick up the spatula and drive ``task``: ``name``, ``extras(scene, args)``, ``keyframes(tip0, low0, level)``,
    ``duration`` [s], ``camera``, ``measure(sim)`` (the task's state as a dict) and ``verdict(state, rows)`` (score, success)."""
    out.mkdir(parents=True, exist_ok=True)
    colour_tool((0.82, 0.50, 0.38))
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    vertices, tets, labels = cable_scene.load(asset)
    grip = tool_scene.grasp_handle(asset, vertices, tets, labels)
    pose = tool_scene.rest_pose(
        vertices, asset.quat, at=(SPATULA_AT[0] - grip["point_m"][0], SPATULA_AT[1] - grip["point_m"][1])
    )
    sim, obj, camera, vertices, tets, labels, _ = tool_scene.build_scene(
        asset,
        name=task["name"],
        pose=pose,
        extras=lambda scene: task["extras"](scene, args),
        camera=task["camera"],
        **{k: getattr(args, k) for k in ("solver", "iterations", "mu", "grip_mu", "force_limit", "self_contact")},
    )
    entity = obj.entity
    src_tets = entity.source_tets
    soft = cable_scene.soft_labels(asset, labels)
    rigid = np.unique(src_tets[~np.isin(labels, soft)])
    handle = np.unique(src_tets[labels == asset.handle])
    head_nodes = np.unique(src_tets[np.isin(labels, soft)])
    turned = arm_gripper._rotate(np.asarray(asset.quat), vertices)
    arm = arm_gripper.ArmControl(sim)
    capacity = max(args.steps, round(task["duration"] * tool_scene.CONTROL_HZ))
    grasp = tool_scene.grasp_handle(asset, vertices, tets, labels, pose=pose)
    state = {"route": None, "grasp": grasp}

    def plan(grasp: dict, now: np.ndarray, level: float):
        """The route of the first of the task's ``variants`` (milder motions) the arm can follow."""
        head = now[head_nodes]
        tip = int(np.argmax(head[:, 0]))
        best = None
        for variant in range(task.get("variants", 1)):
            raw = task["keyframes"](head[tip, :2], float(head[:, 2].min()), level, variant)
            # Aim the head's middle, not its tip node, at the target line.
            y_off = float(head[tip, 1] - 0.5 * (head[:, 1].min() + head[:, 1].max()))

            def target(t, raw=raw, y_off=y_off):
                xy, low, tilt = raw(t)
                return xy + np.array((0.0, y_off)), low, tilt

            if task.get("pile_rear") is not None:
                # The whole head footprint at touchdown against the food's rear edge, from the head's nodes.
                xy, _, tilt = target(task["touchdown"])
                pivot = np.array(grasp["point_m"])
                turned = Rotation.from_rotvec(np.asarray(grasp["across"]) * tilt).apply(head - pivot) + pivot
                x = turned[:, 0] + (xy[0] - turned[tip, 0])
                print(
                    json.dumps(
                        {
                            "touchdown_head_x_range_m": [float(x.min()), float(x.max())],
                            "pile_rear_x_m": task["pile_rear"],
                            "gap_m": task["pile_rear"] - float(x.max()),
                        }
                    ),
                    flush=True,
                )
            tcp_at, quat_at, half_gap_at = hand_motion(grasp, head, tip, target, surface)
            route = arm.plan(
                quat=quat_at, capacity=capacity, dt=1.0 / tool_scene.CONTROL_HZ, tcp_at=tcp_at, half_gap_at=half_gap_at
            )
            print(json.dumps({"variant": variant, "plan_error_m": route.worst_position_error_m}), flush=True)
            if best is None or route.worst_position_error_m < best.worst_position_error_m:
                best = route
            if route.worst_position_error_m < 0.01:
                break
        return best

    q0 = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
    state["route"] = plan(grasp, q0, 0.0)
    print(json.dumps({"initial_plan_error_m": state["route"].worst_position_error_m}), flush=True)
    arm.start(state["route"])
    settle = round(SETTLE * tool_scene.CONTROL_HZ)
    errors = []

    def follow(frame: int):
        if frame == settle:
            now = sim.pipeline.state_0.particle_q.numpy()[entity.particles] - surface
            axis = now[head_nodes].mean(axis=0) - now[handle].mean(axis=0)
            state["grasp"] = tool_scene.grasp_handle(asset, now, src_tets, labels, axis=axis / np.linalg.norm(axis))
            pitch = kabsch(turned[rigid], now[rigid]).as_rotvec() @ np.asarray(state["grasp"]["across"])
            state["route"] = plan(state["grasp"], now, -float(pitch))
            errors.append(state["route"].worst_position_error_m)
            print(
                json.dumps(
                    {"settled_grasp": state["grasp"], "pitch_deg": float(np.degrees(pitch)), "plan_error_m": errors[-1]}
                ),
                flush=True,
            )
        arm.follow(state["route"], frame)
        return arm

    def task_row(frame: int, local: np.ndarray, body_q: np.ndarray) -> dict:
        return {
            "head_low_m": float(local[head_nodes, 2].min()),
            "head_tip_x_m": float(local[head_nodes, 0].max()),
            "rim_dist_m": rim_distance(local),
            **task["measure"](sim),
        }

    rows, elapsed, video_error = cable_scene.episode(
        sim,
        entity,
        camera,
        out,
        soft=soft,
        steps=args.steps,
        render_every=args.render_every,
        control_hz=tool_scene.CONTROL_HZ,
        follow=follow,
        task_row=task_row,
    )
    final = task["measure"](sim)
    result = {
        **final,
        "head_max_height_m": max(r["head_low_m"] for r in rows),
        "max_finger_contacts": max(r["finger_contacts"] for r in rows),
        "max_inverted_tetrahedra": max(r["inverted_tetrahedra"] for r in rows),
        "rim_distance_min_m": min(r["rim_dist_m"] for r in rows),
        "min_z_m": min(r["minimum_z_m"] for r in rows),
        "arm_path_error_m": errors[-1] if errors else None,
        "steps": args.steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "config": tool_scene.config(args, asset, SETTINGS),
    }
    result["score"], result["success"] = task["verdict"](final, rows)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "config"}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=tool_scene.OUTPUT / "tool_spatula_cook" / "run")
    parser.add_argument("--steps", type=int, default=round(DURATION * tool_scene.CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--food-young", type=float, default=3.0e4, help="Young's modulus of the food cubes [Pa]")
    parser.add_argument("--crumbs", type=int, default=10, help="rigid crumbs")
    parser.add_argument(
        "--grip-frac", type=float, default=HOLDER.grip_frac, help="grip position along the handle from its butt"
    )
    tool_scene.add_arguments(parser)
    args = parser.parse_args()
    asset = cable_scene.with_levers(dataclasses.replace(HOLDER, grip_frac=args.grip_frac), args)
    cubes, crumb_names = [], []

    def extras(scene, args):
        return pan_extras(scene, SETTINGS, args.mu, cubes, crumb_names, args.food_young, args.crumbs)

    task = {
        "name": "tool_spatula_cook",
        "extras": extras,
        "keyframes": keyframes,
        "duration": DURATION,
        "camera": CAMERA,
        "touchdown": 5.8,
        "variants": 4,
        "pile_rear": min(x for x, _ in CUBE_AT) - 0.5 * CUBE,
        "measure": lambda sim: judge_food(food_positions(sim, cubes, crumb_names)),
        "verdict": lambda final, rows: (
            final["moved_fraction"],
            bool(final["moved_fraction"] >= 0.7 and final["out_of_pan"] == 0),
        ),
    }
    run(args.out, asset, args, task)


if __name__ == "__main__":
    main()
