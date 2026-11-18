# SPDX-License-Identifier: Apache-2.0
"""Hang the USB hub on a horizontal rod by its cable.

The hub lies flat, its cable straight along the bench. A rod runs parallel to it, raised beside
it. The AIRBOT pinches the rigid housing near its cable end and lifts it, so the cable hangs
under the pads; it carries the hub across the rod, so the hanging cable catches on the rod and
drapes over it, lowers the housing past the rod's far side and lets go. The housing hangs from
the cable on the far side, the plug end on the near side: the cable bends round the rod while
both ends keep their shape and their weight keeps it taut.

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
import scenery
import usb_hub_box
from clawhauser_scene import REFERENCE, arrangement, build_sim
from loop_geometry import smooth

import robosimstudio as rss

HOUSING = usb_hub_box.HOUSING
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "cable_hang_stick"
CONTROL_HZ = REFERENCE.control_hz
SUBSTEPS = usb_hub_box.SUBSTEPS
DURATION = 14.0
ROD_X = 0.10
"""The rod's axis, beside the hub [m, bench frame]: clear of the open pads at the grasp."""
ROD_Z = 0.25
"""The rod's height [m]: under the carry, and over the hanging housing once it is lowered."""
ROD_RADIUS = 0.008
ROD_MU = 1.0
"""A coated rod: over a half wrap the cable holds the housing only up to exp(mu*pi), about 4.8 at mu 0.5,
and the 52 g housing outweighs the plug side 6.5 times, so a bare rod lets it slide off; mu 1 gives 23."""
ROD_Y = (-0.20, 0.10)
"""The rod's span along y [m]: round the hanging cable, short of the arm's riser."""
LIFT = 0.40
"""How high the hand rises [m]: the most the arm reaches over the carry (usb_hub_box's lift)."""
CARRY_X = 0.19
"""How far the hand carries across [m]: the rod's far side, the inner pad clear of it."""
UNDER_ROD = 0.01
"""Where the lowering stops: the pads this far under the rod's axis [m]."""
WITHDRAW_Z = 0.06
PHASES = {
    "close": (0.3, 1.0),
    "lift": (1.3, 4.3),
    "carry": (4.6, 6.8),
    "lower": (7.1, 8.6),
    "open": (9.1, 9.5),
    "withdraw": (9.7, 10.3),
}
"""Phase windows [s]: one straight move each; the last 3.7 s let the hub stop swinging on the rod."""
STEEL = {"color": scenery.STEEL_COLOR, "finish": scenery.STEEL}
POST_RADIUS = 0.010


def build_scene(
    asset, *, solver: str, iterations: int, mu: float, grip_mu: float, rod_mu: float, force_limit: float, self_contact
):
    vertices, tets, labels = cable_scene.load(asset)
    geometry = usb_hub_box.layout(vertices, tets, labels, asset.quat)
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
        rest_exclusion=cable_scene.REST_EXCLUSION,
    )
    scene = rss.Scene(cell=arm_gripper.CELL, physics=physics, name="cable_hang_stick")
    # Stand as in clawhauser_on_rod: hidden collision base under a round foot, post, elbow, then the rod
    # running from the post's face to its free end.
    post_y = ROD_Y[1] + POST_RADIUS
    scene.add(
        rss.slab(name="rod_base", half=(0.030, 0.030, 0.006), at=(ROD_X, post_y, 0.006), collide=True, visible=False)
    )
    scene.add(scenery.rod_foot(at=(ROD_X, post_y)))
    scene.add(
        rss.post(name="rod_post", radius=POST_RADIUS, half_height=ROD_Z / 2.0, at=(ROD_X, post_y, ROD_Z / 2.0), **STEEL)
    )
    scene.add(scenery.rod_elbow(at=(ROD_X, post_y, ROD_Z), radius=POST_RADIUS))
    span = ROD_Y[1] - ROD_Y[0]
    scene.add(
        rss.post(
            name="rod",
            radius=ROD_RADIUS,
            half_height=span / 2.0,
            # A quarter turn about x lays the cylinder's own z axis along y.
            at=(ROD_X, sum(ROD_Y) / 2.0, ROD_Z, -np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0)),
            **STEEL,
        )
    )
    obj = scene.add(
        cable_scene.spec(
            asset, vertices, tets, labels, name="cable_object", pose=geometry["pose"], settings=settings, mu=mu
        )
    )
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    camera = scene.add(
        rss.CameraSpec(
            name="overview", width=960, height=540, eye=(1.25, -0.75, 1.30), look_at=(0.70, -0.08, 0.92), fov_y=45.0
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    solver_ = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    geometry["arm"] = arm_gripper.check_arm(sim, limit=force_limit)
    geometry["friction"] = cable_scene.set_friction(sim, solver_, settings, mu=mu, grip_mu=grip_mu)
    cable_scene.set_shape_mu(sim, solver_, "rod", rod_mu, mu)
    geometry["friction"]["rod_mu"] = rod_mu
    geometry["rod"] = {"x_m": ROD_X, "z_m": ROD_Z, "radius_m": ROD_RADIUS, "y_m": list(ROD_Y)}
    print(json.dumps({"geometry": geometry}), flush=True)
    return sim, obj, camera, geometry, labels


def arm_route(sim, geometry: dict, capacity: int, dt: float):
    """Close on the housing, lift, carry across the rod, lower past it, open, withdraw."""
    control = arm_gripper.ArmControl(sim)
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    grip = np.asarray(geometry["grip_centre_m"])
    centre = arm_gripper.tcp_for_pads(surface + grip)
    reach = float(geometry["reach_m"])
    lower = grip[2] + LIFT - (ROD_Z - UNDER_ROD)

    def tcp_at(time: float) -> np.ndarray:
        return centre + np.array(
            (
                (CARRY_X - grip[0]) * smooth(time, *PHASES["carry"]),
                0.0,
                LIFT * smooth(time, *PHASES["lift"])
                - lower * smooth(time, *PHASES["lower"])
                + WITHDRAW_Z * smooth(time, *PHASES["withdraw"]),
            )
        )

    def half_gap_at(time: float) -> float:
        face = reach + usb_hub_box.GRIP_SQUEEZE - 2.0 * usb_hub_box.GRIP_SQUEEZE * smooth(time, *PHASES["close"])
        return face + (usb_hub_box.OPENED_HALF_GAP - face) * smooth(time, *PHASES["open"])

    route = control.plan(quat=arm_gripper.GRASP_QUAT, capacity=capacity, dt=dt, tcp_at=tcp_at, half_gap_at=half_gap_at)
    return control, route


def on_rod(local: np.ndarray) -> np.ndarray:
    """Which points rest on the rod: on its top half, within a particle of its surface [mask]."""
    radial = np.hypot(local[:, 0] - ROD_X, local[:, 2] - ROD_Z)
    along = (local[:, 1] >= ROD_Y[0]) & (local[:, 1] <= ROD_Y[1])
    return along & (local[:, 2] >= ROD_Z) & (radial <= ROD_RADIUS + REFERENCE.particle_radius + 0.003)


def run(out: Path, asset, *, steps: int, render_every: int, config: dict, **physics) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, camera, geometry, labels = build_scene(asset, **physics)
    entity = obj.entity
    tets = entity.source_tets
    soft = cable_scene.soft_labels(asset, labels)
    cable = np.unique(tets[np.isin(labels, soft)])
    housing = np.unique(tets[labels == HOUSING])
    arm, route = arm_route(sim, geometry, steps, 1.0 / CONTROL_HZ)
    if route.worst_position_error_m > 0.025:
        raise RuntimeError(f"the arm misses the path by {route.worst_position_error_m:.3f} m")
    arm.start(route)

    def follow(frame: int):
        arm.follow(route, frame)
        return arm

    def task_row(frame: int, local: np.ndarray, body_q: np.ndarray) -> dict:
        return {
            "housing_bottom_m": float(local[housing, 2].min()),
            "housing_centre_m": [float(c) for c in local[housing].mean(axis=0)],
            "cable_on_rod": int(np.count_nonzero(on_rod(local[cable]))),
        }

    rows, elapsed, video_error = cable_scene.episode(
        sim,
        entity,
        camera,
        out,
        soft=soft,
        steps=steps,
        render_every=render_every,
        control_hz=CONTROL_HZ,
        follow=follow,
        task_row=task_row,
    )
    tail = rows[-round(0.5 * CONTROL_HZ) :]
    drift = np.ptp(np.asarray([r["housing_centre_m"] for r in tail]), axis=0)
    result = {
        "geometry": geometry,
        "weight_n": float(entity.mass.sum() * 9.81),
        "released": max(r["finger_contacts"] for r in tail) == 0,
        "housing_bottom_final_m": rows[-1]["housing_bottom_m"],
        "minimum_z_final_m": rows[-1]["minimum_z_m"],
        "cable_on_rod_final": rows[-1]["cable_on_rod"],
        "housing_drift_last_0_5s_m": float(np.linalg.norm(drift)),
        "cable_stretch_p99_max": max(r["cable_stretch_p99"] for r in rows),
        "max_inverted_tetrahedra": max(r["inverted_tetrahedra"] for r in rows),
        "max_cable_inverted": max(r["cable_inverted"] for r in rows),
        "arm_path_error_m": route.worst_position_error_m,
        "steps": steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "elapsed_s": elapsed,
        "config": config,
    }
    # Hanging: let go, the housing off the bench and still, the rod carrying the cable.
    result["success"] = bool(
        result["released"]
        and result["housing_bottom_final_m"] > 0.02
        # A hanging hub keeps a small sway after 3.7 s of settling; 35 mm is still hanging, not slipping.
        and result["housing_drift_last_0_5s_m"] < 0.035
        and result["cable_on_rod_final"] > 0
        # Only the soft cable's tets count: a rigid part's own tets have almost no stiffness and ride on its body.
        and result["max_cable_inverted"] == 0
    )
    # Score: how far the housing hangs clear of the bench [m]; negative when it rests on it.
    result["score"] = result["housing_bottom_final_m"]
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "config")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--asset", choices=("usb_hub_gen", "charger"), default="usb_hub_gen")
    parser.add_argument("--steps", type=int, default=round(DURATION * CONTROL_HZ))
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default="chebyshev")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--mu", type=float, default=cable_scene.OBJECT_MU)
    parser.add_argument("--grip-mu", type=float, default=2.0)
    parser.add_argument("--rod-mu", type=float, default=ROD_MU, help="cable-rod friction")
    parser.add_argument(
        "--force-limit", type=float, default=50.0, help="at 30 N the hub can slip as the cable drags over the rod"
    )
    parser.add_argument("--no-self-contact", dest="self_contact", action="store_false")
    cable_scene.add_lever_arguments(parser)
    args = parser.parse_args()
    asset = cable_scene.with_levers(cable_scene.ASSETS[args.asset], args)
    run(
        args.out,
        asset,
        steps=args.steps,
        render_every=args.render_every,
        config=cable_scene.config(args, asset, dataclasses.replace(REFERENCE, substeps=SUBSTEPS)),
        solver=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        grip_mu=args.grip_mu,
        rod_mu=args.rod_mu,
        force_limit=args.force_limit,
        self_contact=args.self_contact,
    )


if __name__ == "__main__":
    main()
