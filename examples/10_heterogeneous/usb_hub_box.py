# SPDX-License-Identifier: Apache-2.0
"""Pick the USB hub up by its rigid housing and lower its soft cable into an open box.

The hub lies flat. Its housing (ABS) and its connector and plug (one 2 g body) are rigid
bodies (:mod:`rigid_part_body`); the cable between them is FEM. The AIRBOT pinches the housing
near its cable end and lifts it, so the housing hangs from the pads and the cable, connector
and plug hang beside it. Measured once they hang, the hand carries them over the
box and lowers them in -- the plug lands, the cable piles on the floor and against the walls,
the housing comes down after -- then opens. Every move is a straight line; the hand does not turn.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import arm_gripper
import cable_scene
import hetero_scene
import numpy as np
import rigid_part_body
from clawhauser_scene import REFERENCE, Trajectory, arrangement, build_sim
from loop_geometry import smooth
from parallel_gripper import ParallelGripperSpec

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish

HOUSING, CABLE, CONNECTOR, PLUG = 0, 1, 2, 3
"""The labels of every hub-like asset (``--asset``); the housing and connector+plug are rigid."""
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "usb_hub_box"
CELL = arm_gripper.CELL
CONTROL_HZ = REFERENCE.control_hz
SUBSTEPS = 40
"""Substeps per frame: twice the Clawhauser scenes', so the thin cable's tension carries further per sweep."""
DURATION = 13.0
GRASP_AT = (0.0, -0.12)
"""Where the pads close on the housing [m, bench frame]: clear of the arm's riser."""
BOX_AT = (0.16, -0.04)
"""The box's centre [m, bench frame]: 160 mm along x and 80 mm along y from the grasp, about the
farthest the arm reaches at the lift's height (200 mm along x alone; 150 by 100 mm). Nearer, the
arm reached over its wall to pick the hub up."""
BOX_INNER = (0.16, 0.26)
"""The box's inside, x by y [m]: the housing (48 mm wide, 200 mm long) fits, so it drops in."""
BOX_WALL = 0.006
BOX_FLOOR = 0.006
BOX_RIM = 0.066
"""The box's wall and floor thickness and the rim's height over the bench [m]."""
GRASP_FROM_ROOT = 0.030
"""Where the pads close: this far along the housing from its cable end [m]. The pinch turns freely
about its closing axis, so the housing hangs straight down from it; pinched at its middle or its
centre of mass it still turned 60 degrees, the cable end down, and the plug stayed on the bench."""
LIFT = 0.40
"""How high the hand rises [m], about the most the arm reaches over the carry. The cable leaves the
housing at the pads, and cable, connector and plug hang about 0.3 m under them, clear of the rim."""
OVER_FLOOR = 0.010
"""Where the lowering stops: the hanging housing's bottom this far over the box's floor [m]."""
GRIP_SQUEEZE = 0.016
"""How far past the housing's width the closed pads aim [m]; the clamp stops them."""
OPENED_HALF_GAP = 0.038
WITHDRAW_Z = 0.040
PHASES = {
    "close": (0.3, 1.0),
    "lift": (1.3, 4.3),
    "carry": (4.6, 6.6),
    "lower": (6.9, 8.9),
    "open": (9.4, 9.8),
    "withdraw": (10.0, 10.6),
}
"""Phase windows [s]: one straight move each."""
GLASS = Finish(roughness=0.05, opacity=0.35, ior=1.5)


def layout(vertices: np.ndarray, tets: np.ndarray, labels: np.ndarray, quat) -> dict:
    """Where the hub lies, where the pads close, and where the box goes, in the bench frame."""
    turned = arm_gripper._rotate(np.asarray(quat), vertices)
    housing = turned[np.unique(tets[labels == HOUSING])]
    root = turned[np.intersect1d(tets[labels == HOUSING], tets[labels == CABLE])].mean(axis=0)
    # Near the housing's cable end, which ``root`` marks.
    near = housing[:, 1].max() if root[1] > housing[:, 1].mean() else housing[:, 1].min()
    grasp_y = near - np.sign(near - housing[:, 1].mean()) * GRASP_FROM_ROOT
    far = housing[:, 1].min() if near == housing[:, 1].max() else housing[:, 1].max()
    band = housing[np.abs(housing[:, 1] - grasp_y) < 0.005]
    centre = np.array((0.5 * (band[:, 0].min() + band[:, 0].max()), grasp_y))
    reach = float(0.5 * (band[:, 0].max() - band[:, 0].min()))
    shift = np.asarray(GRASP_AT) - centre
    pose = (float(shift[0]), float(shift[1]), float(-turned[:, 2].min()) + hetero_scene.PLACE_GAP)
    root_xy = root[:2] + shift
    return {
        "pose": pose,
        "grip_centre_m": [float(GRASP_AT[0]), float(GRASP_AT[1]), float(ParallelGripperSpec.pad_half_height + 0.001)],
        "reach_m": reach,
        "housing_height_m": float(housing[:, 2].max() - housing[:, 2].min()),
        "hang_m": float(abs(far - grasp_y)),
        "cable_root_m": [float(c) for c in root_xy],
        "box_centre_m": [float(c) for c in BOX_AT],
    }


def build_scene(
    asset: hetero_scene.HeteroAsset,
    *,
    solver: str = "chebyshev",
    iterations: int,
    mu: float,
    grip_mu: float,
    force_limit: float,
    self_contact: bool,
):
    vertices, tets, labels = hetero_scene.load(asset)
    geometry = layout(vertices, tets, labels, asset.quat)
    settings = dataclasses.replace(REFERENCE, substeps=SUBSTEPS)
    physics = arrangement(
        settings,
        claw="arm",
        coarse=False,
        solver_name=solver,
        iterations=iterations,
        self_contact=self_contact,
        feedback_smoothing=0.15,
        feedback_clamp=1.5,
        rest_exclusion=hetero_scene.REST_EXCLUSION,
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="usb_hub_box")
    bx, by = geometry["box_centre_m"]
    half_x, half_y = BOX_INNER[0] / 2.0, BOX_INNER[1] / 2.0
    wall_z = (BOX_RIM + BOX_FLOOR) / 2.0
    wall_half = (BOX_RIM - BOX_FLOOR) / 2.0
    for name, half, at, finish in (
        ("box_floor", (half_x + BOX_WALL, half_y + BOX_WALL, BOX_FLOOR / 2.0), (bx, by, BOX_FLOOR / 2.0), None),
        (
            "box_wall_xn",
            (BOX_WALL / 2.0, half_y + BOX_WALL, wall_half),
            (bx - half_x - BOX_WALL / 2.0, by, wall_z),
            GLASS,
        ),
        (
            "box_wall_xp",
            (BOX_WALL / 2.0, half_y + BOX_WALL, wall_half),
            (bx + half_x + BOX_WALL / 2.0, by, wall_z),
            GLASS,
        ),
        ("box_wall_yn", (half_x, BOX_WALL / 2.0, wall_half), (bx, by - half_y - BOX_WALL / 2.0, wall_z), GLASS),
        ("box_wall_yp", (half_x, BOX_WALL / 2.0, wall_half), (bx, by + half_y + BOX_WALL / 2.0, wall_z), GLASS),
    ):
        extra = {"finish": finish} if finish is not None else {}
        scene.add(rss.slab(name=name, half=half, at=at, color=(0.55, 0.60, 0.65), **extra))
    obj = scene.add(
        hetero_scene.spec(
            asset, vertices, tets, labels, name="usb_hub", pose=geometry["pose"], settings=settings, mu=mu
        )
    )
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    camera = scene.add(
        rss.CameraSpec(
            name="overview", width=960, height=540, eye=(1.25, -0.70, 1.35), look_at=(0.66, -0.05, 0.95), fov_y=50.0
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    geometry["arm"] = arm_gripper.check_arm(sim, limit=force_limit)
    geometry["friction"] = hetero_scene.set_friction(sim, solver, settings, mu=mu, grip_mu=grip_mu)
    geometry["rigid_parts"] = {
        f"part{rp.part}": {"nodes": int(len(rp.nodes)), "mass_kg": rp.part_mass} for rp in obj.entity.rigid_parts
    }
    print(json.dumps({"geometry": geometry}), flush=True)
    return sim, obj, camera, geometry


def arm_route(sim, geometry: dict, capacity: int, dt: float, carry=None):
    """Close on the housing, lift, carry, lower, open, withdraw: straight moves, the hand level.

    ``carry`` is the carry's (x, y) [m], the grasp to the box when None; :func:`run` replans it
    from where the hub hangs, since the housing tilts in the pinch and the cable does not hang
    under the grasp.
    """
    if carry is None:
        carry = np.asarray(geometry["box_centre_m"]) - np.asarray(geometry["grip_centre_m"][:2])
    control = arm_gripper.ArmControl(sim)
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    centre = arm_gripper.tcp_for_pads(surface + np.asarray(geometry["grip_centre_m"]))
    reach = float(geometry["reach_m"])
    # The pads stop with the hanging housing's bottom ``OVER_FLOOR`` over the box's floor.
    lower = LIFT + geometry["grip_centre_m"][2] - (BOX_FLOOR + OVER_FLOOR + geometry["hang_m"])

    def tcp_at(time: float) -> np.ndarray:
        return centre + np.array(
            (
                carry[0] * smooth(time, *PHASES["carry"]),
                carry[1] * smooth(time, *PHASES["carry"]),
                LIFT * smooth(time, *PHASES["lift"])
                - lower * smooth(time, *PHASES["lower"])
                + WITHDRAW_Z * smooth(time, *PHASES["withdraw"]),
            )
        )

    def half_gap_at(time: float) -> float:
        face = reach + GRIP_SQUEEZE - 2.0 * GRIP_SQUEEZE * smooth(time, *PHASES["close"])
        return face + (OPENED_HALF_GAP - face) * smooth(time, *PHASES["open"])

    route = control.plan(quat=arm_gripper.GRASP_QUAT, capacity=capacity, dt=dt, tcp_at=tcp_at, half_gap_at=half_gap_at)
    return control, route


def inside_box(points: np.ndarray, geometry: dict, surface: np.ndarray, top: float = BOX_RIM) -> np.ndarray:
    """Which points are inside the box, below ``top`` over the bench [mask]."""
    local = points - surface
    bx, by = geometry["box_centre_m"]
    return (
        (np.abs(local[:, 0] - bx) <= BOX_INNER[0] / 2.0)
        & (np.abs(local[:, 1] - by) <= BOX_INNER[1] / 2.0)
        & (local[:, 2] <= top)
    )


def run(
    out: Path,
    asset: hetero_scene.HeteroAsset,
    *,
    steps: int,
    render_every: int,
    solver: str = "chebyshev",
    iterations: int,
    mu: float,
    grip_mu: float,
    force_limit: float,
    self_contact: bool,
    config: dict | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, camera, geometry = build_scene(
        asset,
        solver=solver,
        iterations=iterations,
        mu=mu,
        grip_mu=grip_mu,
        force_limit=force_limit,
        self_contact=self_contact,
    )
    entity = obj.entity
    particles, tets, labels = entity.particles, entity.source_tets, entity.tet_part_labels
    cable = np.unique(tets[labels == CABLE])
    housing_nodes = np.unique(tets[labels == HOUSING])
    dangling = np.setdiff1d(np.unique(tets[labels != HOUSING]), np.unique(tets[labels == HOUSING]))
    soft = labels == CABLE
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    arm, route = arm_route(sim, geometry, steps, 1.0 / CONTROL_HZ)
    if route.worst_position_error_m > 0.025:
        raise RuntimeError(f"the arm misses the path by {route.worst_position_error_m:.3f} m")
    arm.start(route)
    contacts = sim.pipeline.contacts
    finger_shapes = arm_gripper.finger_shapes(sim)
    q0 = sim.pipeline.state_0.particle_q.numpy()[particles]
    rest = q0[tets[:, 1:]] - q0[tets[:, :1]]
    rest_inv = np.linalg.inv(rest[soft])
    diagnostics, trajectory = [], Trajectory(sim, entity)
    videos, video_error = None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet

        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // CONTROL_HZ))
        videos.__enter__()
    started = time.perf_counter()
    try:
        for frame in range(steps):
            arm.follow(route, frame)
            sim.run(1).sync()
            state = sim.pipeline.state_0
            q, body_q = state.particle_q.numpy(), state.body_q.numpy()
            owned = q[particles]
            if frame + 1 == round(PHASES["carry"][0] * CONTROL_HZ):
                # Carry what hangs -- cable, connector and plug -- over the box's centre.
                hanging = owned[dangling, :2].mean(axis=0) - surface[:2]
                carry = np.asarray(geometry["box_centre_m"]) - hanging
                arm, route = arm_route(sim, geometry, steps, 1.0 / CONTROL_HZ, carry)
                print(json.dumps({"carry_m": np.round(carry, 4).tolist()}), flush=True)
            edges = owned[tets[:, 1:]] - owned[tets[:, :1]]
            # The cable's strain: each tet's largest principal stretch; its 99th percentile skips slivers.
            stretch = np.linalg.svd(rest_inv @ edges[soft], compute_uv=False).max(axis=1)
            count = int(contacts.soft_contact_count.numpy()[0])
            shapes = contacts.soft_contact_shape.numpy()[: min(count, contacts.soft_contact_max)]
            row = {
                "frame": frame + 1,
                "time_s": (frame + 1) / CONTROL_HZ,
                "minimum_z_m": float(owned[:, 2].min() - surface[2]),
                "housing_minimum_z_m": float(owned[housing_nodes, 2].min() - surface[2]),
                "cable_inside": float(inside_box(owned[cable], geometry, surface).mean()),
                "inverted_tetrahedra": int(np.count_nonzero(np.linalg.det(edges) <= 0)),
                "cable_inverted": int(np.count_nonzero(np.linalg.det(edges[soft]) <= 0)),
                "cable_stretch_p99": float(np.percentile(stretch, 99.0)),
                "finger_contacts": int(np.count_nonzero(np.isin(shapes, finger_shapes))),
                "aperture_m": arm.aperture(),
                **rigid_part_body.frame_record(entity, owned, body_q),
            }
            trajectory.add(row["time_s"], owned, body_q)
            diagnostics.append(row)
            if videos is not None and frame % render_every == 0:
                videos.append_frames({camera.name: sim.render(camera)})
            if frame % 30 == 0 or frame == steps - 1:
                print(json.dumps({k: v for k, v in row.items() if not k.startswith("part")}), flush=True)
    finally:
        if videos is not None:
            try:
                videos.__exit__(None, None, None)
            except Exception as exc:
                video_error = repr(exc)
    elapsed = time.perf_counter() - started
    trajectory.save(out / "trajectory.npz")
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    lift_rows = [r for r in diagnostics if PHASES["lift"][1] <= r["time_s"] <= PHASES["carry"][1]]
    tail = diagnostics[-round(0.5 * CONTROL_HZ) :]
    skin = max(v["skin_stretch_max_m"] for r in diagnostics for k, v in r.items() if k.endswith("_body"))
    result = {
        "geometry": geometry,
        "weight_n": float(entity.mass.sum() * 9.81),
        "lifted_clearance_m": min((r["minimum_z_m"] for r in lift_rows), default=None),
        "housing_lifted_m": min((r["housing_minimum_z_m"] for r in lift_rows), default=None),
        "released": max(r["finger_contacts"] for r in tail) == 0,
        "cable_inside_final": diagnostics[-1]["cable_inside"],
        # The housing's centre, which lies on the cable it drops onto: within a housing height of the rim.
        "housing_inside_final": bool(
            inside_box(body_q[None, entity.rigid_parts[0].body, :3], geometry, surface, BOX_RIM + 0.02)[0]
        ),
        "cable_stretch_p99_max": max(r["cable_stretch_p99"] for r in diagnostics),
        "rigid_skin_offset_max_m": skin,
        "max_inverted_tetrahedra": max(r["inverted_tetrahedra"] for r in diagnostics),
        "max_cable_inverted": max(r["cable_inverted"] for r in diagnostics),
        "arm_path_error_m": route.worst_position_error_m,
        "steps": steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(diagnostics)),
        "mean_step_ms": 1000.0 * elapsed / max(1, len(diagnostics) * SUBSTEPS),
        "elapsed_s": elapsed,
        "config": config,
    }
    result["success"] = bool(
        # The housing must leave the bench; a cable longer than the lift may still trail on it.
        (result["housing_lifted_m"] or 0.0) > 0.02
        and result["released"]
        # Most of the cable: a 350 mm cable can leave a loop over the rim.
        and result["cable_inside_final"] >= 0.8
        and result["housing_inside_final"]
        # Only the soft cable's tets count: a rigid part's own tets have almost no stiffness and ride on its body.
        and result["max_cable_inverted"] == 0
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "geometry"}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--asset", choices=("usb_hub", "usb_hub_gen", "charger", "power_strip"), default="usb_hub")
    parser.add_argument("--steps", type=int, default=round(DURATION * CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default="chebyshev")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--mu", type=float, default=hetero_scene.OBJECT_MU)
    parser.add_argument("--grip-mu", type=float, default=2.0)
    parser.add_argument("--force-limit", type=float, default=30.0)
    parser.add_argument("--no-self-contact", dest="self_contact", action="store_false")
    hetero_scene.add_lever_arguments(parser)
    args = parser.parse_args()
    asset = hetero_scene.with_levers(cable_scene.ASSETS[args.asset], args)
    run(
        args.out,
        asset,
        steps=args.steps,
        render_every=args.render_every,
        solver=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        grip_mu=args.grip_mu,
        force_limit=args.force_limit,
        self_contact=args.self_contact,
        config=hetero_scene.config(args, asset, dataclasses.replace(REFERENCE, substeps=SUBSTEPS)),
    )


if __name__ == "__main__":
    main()
