# SPDX-License-Identifier: Apache-2.0
"""Retrieve the power strip by pulling its cord round a post.

The strip's housing (about 300 g) lies beyond a round post, out of the arm's reach. Its 500 mm
cord runs from the housing to the post, wraps a quarter turn round it and runs on toward the arm,
to the mains plug within reach (laid so with :func:`cable_scene.lay`; its rest shape stays
straight). The housing and the collar+plug are rigid bodies (:mod:`rigid_part_body`); the cord is
FEM.

The AIRBOT pinches the rigid collar once and drags it straight away from the post along the
bench, then reels it straight up. The cord stays wrapped round the post while its tension turns
the corner and drags the housing along to the post and round it, out on the hand's side. The
hand then sets the plug down and opens. Every move is a straight line.

Why the parts matter: the soft cord bends round the post and passes the pull on, the rigid
housing slides as one body. A stiff cord would not bend round it and would lever the housing
sideways; a cord as soft as rubber would stretch and leave it where it lay.

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
from parallel_gripper import ParallelGripperSpec

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish

HOUSING, CABLE, COLLAR, PLUG = 0, 1, 2, 3
"""The power strip's labels; the housing and the collar+plug are rigid."""
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "cable_retrieve"
WIDE = 1.5
"""Half-width of the bench [m]: a strip dragged round the post must not reach an edge (at 0.62 it tipped off)."""
_BASE = rss.get_cell(arm_gripper.CELL)
CELL = rss.register_cell(
    dataclasses.replace(
        _BASE,
        name=f"{arm_gripper.CELL}_wide",
        half=(WIDE, WIDE, _BASE.half[2]),
        statics=tuple(
            dataclasses.replace(s, half_extents=(WIDE, WIDE, s.half_extents[2])) if s.name == "table" else s
            for s in _BASE.statics
        ),
    )
).name
CONTROL_HZ = REFERENCE.control_hz
SUBSTEPS = 40
"""Substeps per frame, as usb_hub_box: the thin cord's tension carries further per sweep."""
DURATION = 15.7
STRIP_MU = cable_scene.OBJECT_MU
CORD_DAMPING = 1.5e5
"""The cord's damping [Pa*s]: the laid cord straightens as the housing comes round the post; at
6e4 and mu 0.5 that flung the 0.3 kg strip off the bench, at 1.5e5 it comes round smoothly (q7)."""
STRIP_QUAT = (0.0, 0.0, 0.0, 1.0)
"""The strip as built, before the cord is laid round the post: housing to plug along the bench's +x."""
POST_AT = (0.12, 0.33)
POST_RADIUS = 0.045
POST_HEIGHT = 0.15
"""The post's centre [m, bench frame], radius and height [m]: round, so the cord slides round it
and the housing slides past it; thick, so the laid bend stores less to fling the housing with;
tall, so the reeled cord stays on it. Half a metre in from the bench's +x edge, room for the strip."""
POST_MU = 0.5
"""The post's effective friction with the strip: painted wood. The cord's tension grows as
exp(mu * angle) round it, so a grippier post would take the grip's whole budget."""
CORD_CLEAR = 0.006
"""How far the cord's surface lies outside the post's [m]: a particle radius and a little."""
WRAP = np.radians(90.0)
"""How far the laid cord turns round the post [rad]: from running along +x past its far side to
running toward the arm (-y) past its side."""
ROOT_TO_POST = 0.15
"""How far along the cord from the housing it reaches the post [m]: the cord, not the housing,
takes the wrap, which stays while the housing is dragged this far to the post."""
DRAG_TO_Y = -0.28
"""Where the drag along the bench ends [m, bench y]: straight away from the post, as far as the
arm reaches."""
DRAG_Z = 0.030
"""How high the plug is carried across the bench [m]: clear of it, so it does not snag."""
REEL_Z = 0.40
"""How high the hand reels the plug [m], about the most it reaches there."""
PUT_DOWN_Z = 0.030
"""Where the hand stops lowering, over its pinch height [m]: the hanging collar touches down
first. Lowered all the way, the bench pushed the collar up out of the pads."""
PAST_POST = 0.06
"""How far past the post's centre, along the pull, the housing's centre must end [m]: round it."""
TARGET_PAST = 0.12
"""The target for the housing's centre: this far past the post along the pull [m], for the score."""
OPENED_HALF_GAP = 0.040
"""The pads fully open [m]: the most clearance round the collar."""
CLOSED_HALF_GAP = 0.0
"""What the pinch aims at [m]; the clamp stops the pads on the collar."""
WITHDRAW_Z = 0.12
PHASES = {
    "close": (0.3, 1.0),
    "drag": (1.3, 6.3),
    "reel": (6.7, 11.2),
    "lower": (11.7, 13.3),
    "open": (13.6, 14.0),
    "withdraw": (14.2, 15.0),
}
"""Phase windows [s]: one straight move each; the drag and reel are slow, since they pull the
housing, and the pull on the pinch grows gradually."""
TRANSLUCENT = Finish(roughness=0.3, opacity=0.6, ior=1.5)


def layout(vertices: np.ndarray, tets: np.ndarray, labels: np.ndarray) -> dict:
    """Where the strip lies, how its cord is laid round the post and where the pads close [bench frame].

    The strip is built straight along +x, its cord past the post's far (+y) side; :func:`build_scene`
    then lays the cord from ``lay_root_m`` (where it meets the post) round the post and on toward -y.
    """
    post = np.asarray(POST_AT)
    seam = np.intersect1d(tets[labels == HOUSING], tets[labels == CABLE])
    root_x = float(vertices[seam, 0].mean())
    cord_radius = float(np.abs(vertices[np.unique(tets[labels == CABLE]), 1]).max())
    bend = POST_RADIUS + cord_radius + CORD_CLEAR
    meet = post + np.array((0.0, bend))
    leave = post + np.array((bend, 0.0))
    shift = meet - np.array((root_x + ROOT_TO_POST, vertices[seam, 1].mean()))
    collar = vertices[np.unique(tets[labels == COLLAR])]
    centre = 0.5 * (collar.min(axis=0) + collar.max(axis=0))
    # The collar's centre, past the wrap, down the straight run toward the arm.
    grasp = leave + np.array((0.0, -(centre[0] - root_x - ROOT_TO_POST - WRAP * bend)))
    housing = vertices[np.unique(tets[labels == HOUSING]), :2].mean(axis=0) + shift
    pull_to = np.array((leave[0], DRAG_TO_Y))
    # Cord the drag and reel draw, against what the housing needs: to the post, round it and PAST_POST on.
    drawn = float(np.hypot(np.linalg.norm(leave - pull_to), REEL_Z) - np.linalg.norm(leave - grasp))
    needed = ROOT_TO_POST + WRAP * bend + root_x - float(vertices[np.unique(tets[labels == HOUSING]), 0].mean())
    mount = np.asarray(arm_gripper.MOUNT[:2])
    return {
        "pose": (float(shift[0]), float(shift[1]), float(-vertices[:, 2].min()) + cable_scene.PLACE_GAP),
        "lay_root_m": [float(c) for c in meet],
        "lay_bend_radius_m": bend,
        "grip_centre_m": [float(grasp[0]), float(grasp[1]), float(ParallelGripperSpec.pad_half_height + 0.001)],
        "pull_to_m": [float(c) for c in pull_to],
        "cord_by_post_m": [float(c) for c in leave],
        "root_asset_x_m": root_x,
        "planned_wrap_deg": float(np.degrees(WRAP)),
        "planned_pull_m": drawn,
        "planned_pull_needed_m": needed + PAST_POST,
        "housing_start_m": [float(c) for c in housing],
        "housing_from_base_m": float(np.linalg.norm(housing - mount)),
        "grasp_from_base_m": float(np.linalg.norm(grasp - mount)),
        "post_m": list(POST_AT),
    }


def build_scene(
    asset: cable_scene.CableAsset,
    *,
    solver: str = "chebyshev",
    iterations: int,
    mu: float,
    grip_mu: float,
    post_mu: float,
    force_limit: float,
    self_contact: bool,
):
    vertices, tets, labels = cable_scene.load(asset)
    geometry = layout(vertices, tets, labels)
    settings = dataclasses.replace(REFERENCE, substeps=SUBSTEPS)
    physics = arrangement(
        settings,
        claw="arm",
        coarse=False,
        solver_name=solver,
        iterations=iterations,
        self_contact=self_contact,
        rest_exclusion=cable_scene.REST_EXCLUSION,
        feedback_smoothing=0.15,
        feedback_clamp=1.5,
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="cable_retrieve")
    half = POST_HEIGHT / 2.0
    scene.add(
        rss.post(
            name="post",
            radius=POST_RADIUS,
            half_height=half,
            at=(POST_AT[0], POST_AT[1], half),
            color=(0.62, 0.48, 0.34),
            finish=TRANSLUCENT,
        )
    )
    obj = scene.add(
        cable_scene.spec(
            asset, vertices, tets, labels, name="power_strip", pose=geometry["pose"], settings=settings, mu=mu
        )
    )
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    # High over the bench's far +x corner, opposite the arm's base: the wrap is on the post's +x side,
    # facing the camera, with the strip beyond and the hand below it.
    camera = scene.add(
        rss.CameraSpec(
            name="overview", width=960, height=540, eye=(1.16, 0.62, 1.50), look_at=(0.70, 0.08, 0.75), fov_y=55.0
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    geometry["arm"] = arm_gripper.check_arm(sim, limit=force_limit)
    geometry["friction"] = cable_scene.set_friction(sim, solver, settings, mu=mu, grip_mu=grip_mu)
    cable_scene.set_shape_mu(sim, solver, "post", post_mu, mu)
    geometry["friction"]["post_mu"] = post_mu
    entity = obj.entity
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    geometry["laid"] = cable_scene.lay(
        sim,
        entity,
        root=surface + np.array((*geometry["lay_root_m"], 0.0)),
        direction=(1.0, 0.0),
        bends=((WRAP * geometry["lay_bend_radius_m"], -geometry["lay_bend_radius_m"]),),
        keep=np.unique(entity.source_tets[entity.tet_part_labels == HOUSING]),
    )
    geometry["rigid_parts"] = {
        f"part{rp.part}": {"nodes": int(len(rp.nodes)), "mass_kg": rp.part_mass} for rp in obj.entity.rigid_parts
    }
    print(json.dumps({"geometry": geometry}), flush=True)
    return sim, obj, camera, geometry


def arm_route(sim, geometry: dict, capacity: int, dt: float):
    """Pinch the collar, drag it away from the post, reel it up, set it down, open, withdraw.

    The pads close across the collar, and the hand turns so the collar points along the cord, back
    at the post: held askew, the collar swung crosswise, pried the pads open and fell out.
    """
    control = arm_gripper.ArmControl(sim)
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    grip = np.asarray(geometry["grip_centre_m"])
    start = arm_gripper.tcp_for_pads(surface + grip)
    drag = np.asarray(geometry["pull_to_m"]) - grip[:2]
    lifts = (
        ("drag", DRAG_Z),
        ("reel", REEL_Z - DRAG_Z),
        ("lower", PUT_DOWN_Z - REEL_Z),
        ("withdraw", WITHDRAW_Z),
    )

    def tcp_at(time: float) -> np.ndarray:
        xy = drag * smooth(time, *PHASES["drag"])
        z = sum(step * smooth(time, *PHASES[name]) for name, step in lifts)
        return start + np.array((xy[0], xy[1], z))

    def quat_at(time: float) -> np.ndarray:
        toward = np.asarray(geometry["cord_by_post_m"]) - (grip[:2] + drag * smooth(time, *PHASES["drag"]))
        yaw = float(np.arctan2(toward[1], toward[0])) - 0.5 * np.pi
        return arm_gripper.quat_multiply(arm_gripper.axis_angle_quat((0.0, 0.0, 1.0), yaw), arm_gripper.GRASP_QUAT)

    def half_gap_at(time: float) -> float:
        shut = smooth(time, *PHASES["close"]) - smooth(time, *PHASES["open"])
        return OPENED_HALF_GAP + (CLOSED_HALF_GAP - OPENED_HALF_GAP) * shut

    route = control.plan(quat=quat_at, capacity=capacity, dt=dt, tcp_at=tcp_at, half_gap_at=half_gap_at)
    return control, route


def wrap_deg(local: np.ndarray, cord: np.ndarray, arc: np.ndarray) -> float:
    """How far the strip turns round the post [deg]: between its directions 70-130 mm either side
    of its node nearest the post, past the 90 mm of cord a quarter turn lays on it.

    ``arc`` is every node's distance along the strip from the cord's root, so the run behind the
    post may be the housing's.
    """
    xy = local[:, :2]
    at = arc[cord][np.argmin(np.linalg.norm(xy[cord] - np.asarray(POST_AT), axis=1))]

    def run(offset: float) -> np.ndarray:
        near = np.abs(arc - at - offset) < 0.01
        return xy[near].mean(axis=0) if near.any() else np.full(2, np.nan)

    into, out = run(-0.08) - run(-0.12), run(0.12) - run(0.08)
    cos = np.dot(into, out) / np.linalg.norm(into) / np.linalg.norm(out)
    # Past the cord's ends (the strip mostly round the post) there is no run to measure.
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))) if np.isfinite(cos) else 0.0


def run(
    out: Path,
    asset: cable_scene.CableAsset,
    *,
    steps: int,
    render_every: int,
    solver: str = "chebyshev",
    iterations: int,
    mu: float,
    grip_mu: float,
    post_mu: float,
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
        post_mu=post_mu,
        force_limit=force_limit,
        self_contact=self_contact,
    )
    entity = obj.entity
    tets, labels = entity.source_tets, entity.tet_part_labels
    housing_body = entity.rigid_parts[0].body
    # The cord's nodes and how far along it each lies from the housing, from the straight rest mesh.
    vertices, _, _ = cable_scene.load(asset)
    cord = np.setdiff1d(np.unique(tets[labels == CABLE]), np.unique(tets[labels != CABLE]))
    arc = vertices[:, 0] - geometry["root_asset_x_m"]
    capacity = max(steps, round(DURATION * CONTROL_HZ))
    # The whole route, however short the run, so a smoke run checks every move is reached.
    arm, route = arm_route(sim, geometry, capacity, 1.0 / CONTROL_HZ)
    print(json.dumps({"arm_path_error_m": route.worst_position_error_m}), flush=True)
    if route.worst_position_error_m > 0.025:
        raise RuntimeError(f"the arm misses the path by {route.worst_position_error_m:.3f} m")
    arm.start(route)
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    post, pull = np.asarray(POST_AT), np.array((0.0, -1.0))  # the pull runs from the post toward -y
    target = post + TARGET_PAST * pull

    def follow(frame: int):
        arm.follow(route, frame)
        return arm

    def task_row(frame: int, local: np.ndarray, body_q: np.ndarray) -> dict:
        housing = body_q[housing_body, :3] - surface
        return {
            "housing_m": [float(c) for c in housing],
            "housing_past_post_m": float(np.dot(housing[:2] - post, pull)),
            "housing_to_target_m": float(np.linalg.norm(housing[:2] - target)),
            "cord_wrap_deg": wrap_deg(local, cord, arc),
        }

    rows, elapsed, video_error = cable_scene.episode(
        sim,
        entity,
        camera,
        out,
        soft=cable_scene.soft_labels(asset, labels),
        steps=steps,
        render_every=render_every,
        control_hz=CONTROL_HZ,
        follow=follow,
        task_row=task_row,
    )
    tail = rows[-round(0.5 * CONTROL_HZ) :]
    final = rows[-1]["housing_m"]
    dragged = rows[min(len(rows), round(PHASES["drag"][1] * CONTROL_HZ)) - 1]
    skin = max(v["skin_stretch_max_m"] for r in rows for k, v in r.items() if k.endswith("_body"))
    result = {
        "geometry": geometry,
        "weight_n": float(entity.mass.sum() * 9.81),
        "cord_wrap_after_drag_deg": dragged["cord_wrap_deg"],
        "cord_wrap_max_deg": max(r["cord_wrap_deg"] for r in rows),
        "housing_final_m": final,
        "housing_travel_m": float(np.linalg.norm(np.subtract(final[:2], rows[0]["housing_m"][:2]))),
        "housing_past_post_final_m": rows[-1]["housing_past_post_m"],
        "housing_round_post": rows[-1]["housing_past_post_m"] >= PAST_POST,
        "housing_to_target_final_m": rows[-1]["housing_to_target_m"],
        "housing_to_target_min_m": min(r["housing_to_target_m"] for r in rows),
        "released": max(r["finger_contacts"] for r in tail) == 0,
        "cable_stretch_p99_max": max(r["cable_stretch_p99"] for r in rows),
        "rigid_skin_offset_max_m": skin,
        # The rigid parts' own tets only ride on their bodies: a squeezed collar tet can invert harmlessly.
        "max_inverted_tetrahedra": max(r["inverted_tetrahedra"] for r in rows),
        "max_cable_inverted": max(r["cable_inverted"] for r in rows),
        "arm_path_error_m": route.worst_position_error_m,
        "steps": steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "mean_step_ms": 1000.0 * elapsed / max(1, len(rows) * SUBSTEPS),
        "elapsed_s": elapsed,
        "config": config,
    }
    # A sane cord stretches a few percent round the post; far more and it is pulling apart.
    result["success"] = bool(
        result["housing_round_post"]
        and result["released"]
        and result["max_cable_inverted"] == 0
        # The strain measure counts bending: the 10 mm cord on the post reads ~1.09 at rest.
        and result["cable_stretch_p99_max"] < 1.3
    )
    # Continuous: the housing's final distance to the target's centre [m], lower is better.
    result["score"] = result["housing_to_target_final_m"]
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "config")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--steps", type=int, default=round(DURATION * CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default="chebyshev")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--mu", type=float, default=STRIP_MU)
    parser.add_argument("--grip-mu", type=float, default=2.0)
    parser.add_argument("--post-mu", type=float, default=POST_MU)
    parser.add_argument("--force-limit", type=float, default=50.0)
    parser.add_argument("--no-self-contact", dest="self_contact", action="store_false")
    cable_scene.add_lever_arguments(parser)
    args = parser.parse_args()
    strip = dataclasses.replace(cable_scene.ASSETS["power_strip"], quat=STRIP_QUAT, cable_damping=CORD_DAMPING)
    asset = cable_scene.with_levers(strip, args)
    run(
        args.out,
        asset,
        steps=args.steps,
        render_every=args.render_every,
        solver=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        grip_mu=args.grip_mu,
        post_mu=args.post_mu,
        force_limit=args.force_limit,
        self_contact=args.self_contact,
        config=cable_scene.config(args, asset, dataclasses.replace(REFERENCE, substeps=SUBSTEPS)),
    )


if __name__ == "__main__":
    main()
