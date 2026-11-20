# SPDX-License-Identifier: Apache-2.0
"""The AIRBOT picks the wooden-handle rubber stamp off the bench, carries it over a sheet of paper and presses it.

The stamp (``rubber_stamp``) stands on the bench on its own relief. The hand comes down over its
knob, closes on it, lifts it, holds, carries it sideways over the paper, sets it down and pushes
``--overpress`` further, holds, takes the push off, lets go and backs away. The arm is the
Clawhauser scenes' (:mod:`arm_gripper`): it holds the stamp by pad friction alone, and each action
runs as ``clawhauser_arm_pick_place`` runs one, from the measured tool pose to the commanded one.

The run measures whether the stamp left the bench and arrived over the paper, how much of the
relief is on the paper under the push, how far the cushion and the die give, how far the knob
slips in the pads, and what the mesh does. Every relief face whose three nodes are on the paper
inks it; the union over the run is ``imprint.png``.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import arm_gripper
import newton
import numpy as np
import rigid_part_body
import stamp_press
from clawhauser_scene import Trajectory, arrangement, build_sim
from mesh_quality import MeshQuality
from rubber_stamp_press import ASSET, DIE, FOAM, INK_HALF, SOFT, layout, load_stamp
from stamp_press import INK_PIXEL, INK_RGB, PAPER_COLOR, PAPER_HALF, PHYSICS, PLACE_GAP, TOUCH_TOLERANCE

import robosimstudio as rss

OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "rubber_stamp_arm_press"
CELL = arm_gripper.CELL
CONTROL_HZ = PHYSICS.control_hz
PICK_AT = (-0.10, -0.06)
"""Where the stamp stands on the bench, work-surface frame [m]."""
PAPER_AT = (0.02, -0.06)
"""The middle of the paper, 120 mm along x from the stamp [m]."""
GRIP_Z = 0.054
"""The pads' centre above the stamp's printing face [m]: the knob is the 32 mm cylinder from 46 to 66 mm."""
KNOB_RADIUS = 0.016
GRIP_SQUEEZE = 0.0002
"""How far past the knob's radius each pad is commanded [m].

Just past, so the fingers reach the hand's 20 N and stop within a fraction of a millimetre of
where they are told. Commanded 4 mm past, both stalled far from their places and nothing placed
the pair, so the knob slid 9 mm down the pads and tipped 5 degrees while it hung. The push goes
through the grip's friction: about 32 N a pad at the scene's cushion, of the 57 N the grip holds.
"""
GRIP_LIMIT_N = arm_gripper.GRASP_LIMIT_N
"""One finger's force cap [N], the hand's own."""
GRIP_MU = arm_gripper.GRIP_MU
HOVER = 0.060
"""The open hand waits this far above its grasp while the stamp settles [m]."""
LIFT = 0.050
OVERPRESS = 0.0015
"""How far the hand pushes down once the stamp is back at its standing height [m].

Half the 3 mm cushion's thickness, and about 64 N of push at the scene's cushion. At the asset's
modulus the same push takes about 160 N, and 3 mm squeezes the cushion 1.74 mm and inverts 82 tets.
"""
FOAM_YOUNG = 1.0e5
"""The cushion's Young's modulus in this scene [Pa], a third of the asset's 3e5.

A scene override; the asset's files are untouched. The push that squeezes the cushion is its
modulus times its strain times its 64 x 29 mm face, about 160 N for 0.9 mm at the asset's value,
which no AIRBOT hand gives. At 1e5 the stiffest part is 100 times the softest, the most allowed.
"""
RETREAT = 0.060
SETTLE_S = 0.6
OPEN_WIDTH = 2.0 * arm_gripper.OPEN_HALF_GAP
MAX_IK_ERROR_M = 0.010
PHASES = (
    "settle",
    "reach",
    "close",
    "lift",
    "hang",
    "carry",
    "lower",
    "press",
    "hold",
    "unload",
    "open",
    "retreat",
    "rest",
)


def build_scene(
    *,
    solver_name: str = stamp_press.SOLVER,
    iterations: int = 160,
    mu: float | None = None,
    grip_mu: float = GRIP_MU,
    force_limit: float = GRIP_LIMIT_N,
    foam_young: float = FOAM_YOUNG,
):
    vertices, tets, labels, groups, names = load_stamp()
    geometry = layout(vertices, tets, labels)
    settings = PHYSICS
    physics = arrangement(
        settings, claw="arm", coarse=False, solver_name=solver_name, iterations=iterations, self_contact=False
    )
    scene = rss.Scene(cell=CELL, physics=physics, name="rubber_stamp_arm_press")
    paper_top = 2.0 * PAPER_HALF[2]
    scene.add(rss.slab(name="paper", half=PAPER_HALF, at=(PAPER_AT[0], PAPER_AT[1], PAPER_HALF[2]), color=PAPER_COLOR))
    # On the bench itself: a node rests one particle radius over it.
    pose = (PICK_AT[0], PICK_AT[1], settings.particle_radius + PLACE_GAP - geometry["base_z"])
    obj = scene.add(
        stamp_press.StampSpec(
            rigid_parts=groups,
            attach_damping=stamp_press.ATTACH_DAMPING,
            name="rubber_stamp",
            mesh_path=str(ASSET / "final_mesh.mesh"),
            params_path=str(ASSET / "heterogeneous_params.npz"),
            materials_path=str(ASSET / "inferred_params.json"),
            pose=pose,
            contact_ke=settings.contact_ke,
            contact_kd=settings.contact_kd,
            contact_mu=settings.soft_mu,
            particle_radius=settings.particle_radius,
            damping_ratio=settings.damping_ratio,
            part_overrides=((FOAM, "youngs_modulus_pa", foam_young),),
        )
    )
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    middle = surface + np.array((0.5 * (PICK_AT[0] + PAPER_AT[0]), PICK_AT[1], 0.0))
    foot = surface + np.array((PAPER_AT[0], PAPER_AT[1], paper_top))
    for camera in (
        # From across the bench, the arm behind: the pick, the carry and the press all in frame.
        rss.CameraSpec(
            name="overview",
            width=960,
            height=540,
            eye=tuple(float(v) for v in np.add(middle, (0.30, 0.42, 0.26))),
            look_at=tuple(float(v) for v in np.add(middle, (0.0, -0.04, 0.07))),
            fov_y=40.0,
        ),
        # Level with the cushion over the paper, from the side the arm is not on: the two soft layers fill the frame.
        rss.CameraSpec(
            name="side",
            width=960,
            height=540,
            eye=tuple(float(v) for v in np.add(foot, (0.0, 0.16, 0.012))),
            look_at=tuple(float(v) for v in np.add(foot, (0.0, 0.0, 0.012))),
            fov_y=24.0,
        ),
    ):
        scene.add(camera)
    sim = build_sim(scene, rigid_loop=False)
    entity = obj.entity
    report = dict(geometry["report"])
    report.update(stamp_press.check_stamp(sim, entity, tets, settings, mu=mu))
    report["arm"] = arm_gripper.check_arm(sim, limit=force_limit)
    soft_mu = report["friction"]["global_soft_contact_mu"]
    report["friction"].update(arm_gripper.set_contact_friction(sim, global_mu=soft_mu, grip_mu=grip_mu))
    young = entity.elastic[:, 0]
    report["asset"] = "rubber_stamp"
    report["source_stiffness_ratio"] = float(entity.source_elastic[:, 0].max() / entity.source_elastic[:, 0].min())
    report["effective_young_pa_by_part"] = {names[p]: float(young[labels == p].max()) for p in range(4)}
    report["effective_stiffness_ratio"] = float(young.max() / young.min())
    report["weight_n"] = float(entity.mass.sum() * 9.81)
    report["paper_top_m"] = paper_top
    report["stamp_pose_m"] = [float(v) for v in pose]
    print(json.dumps({"geometry": report}), flush=True)
    return sim, obj, geometry, report


def judge(rows: list[dict], report: dict, complete: bool) -> dict:
    """What the arm did with the stamp, from the per-frame rows."""

    def during(*names: str) -> list[dict]:
        return [row for row in rows if row["phase"] in names]

    stood = during("settle")[-1]
    hold, hang, final = during("hold"), during("hang"), rows[-1]
    peak = max(during("press", "hold"), key=lambda row: row["squeeze_m"], default=stood)
    over_paper = hold[-1] if hold else final
    result = {
        "complete": complete,
        "finite": all(row["finite"] for row in rows),
        "lifted_m": max((row["base_z_m"] for row in hang), default=stood["base_z_m"]) - stood["base_z_m"],
        "carried_to_m": over_paper["centre_m"],
        "miss_m": float(np.linalg.norm(np.asarray(over_paper["centre_m"]) - np.asarray(PAPER_AT))),
        "standing_squeeze_m": stood["squeeze_m"],
        "hold_relief_on_paper": min((row["relief_on_paper"] for row in hold), default=0.0),
        "peak_squeeze_m": peak["squeeze_m"],
        "peak_foam_squeeze_m": peak["foam_squeeze_m"],
        "peak_die_squeeze_m": peak["die_squeeze_m"],
        "peak_foam_strain": peak["foam_squeeze_m"] / report["foam_thickness_m"],
        "peak_bulge_m": max((row["bulge_m"] for row in during("press", "hold")), default=0.0),
        "peak_pad_squeeze_n": max((row.get("squeeze_n", 0.0) for row in rows), default=0.0),
        # While the pads hold it: after they open the same number is only how far the hand has gone.
        "grip_slip_m": max(
            (abs(row["grip_slip_m"]) for row in during("lift", "hang", "carry", "lower", "press", "hold", "unload")),
            default=0.0,
        ),
        "max_tilt_held_deg": max((row["tilt_deg"] for row in during("lift", "hang", "carry", "lower")), default=0.0),
        "final_tilt_deg": final["tilt_deg"],
        "final_squeeze_m": final["squeeze_m"],
        "final_aperture_m": final["aperture_m"],
        "max_inverted_soft_tets": max(row["inverted_soft_tets"] for row in rows),
        "max_rigid_node_stretch_m": max(row["rigid_node_stretch_m"] for row in rows),
    }
    result["checks"] = {
        "ran": result["complete"] and result["finite"],
        "lifted": result["lifted_m"] >= 0.030,
        "over_the_paper": result["miss_m"] <= 0.005,
        "prints_under_the_press": result["hold_relief_on_paper"] >= 0.95,
        # The two soft layers give by at least half of what the hand pushed, past what they give standing.
        "soft_layers_give": result["peak_squeeze_m"] - stood["squeeze_m"] >= 0.5 * report["overpress_m"],
        "let_go_and_stands": result["final_aperture_m"] >= 0.060 and result["final_tilt_deg"] < 2.0,
        "soft_mesh_intact": result["max_inverted_soft_tets"] == 0,
    }
    result["effective_stiffness_ratio"] = report["effective_stiffness_ratio"]
    return result


def run(
    out: Path,
    *,
    render_every: int = 0,
    until: str = PHASES[-1],
    overpress: float = OVERPRESS,
    solver_name: str = stamp_press.SOLVER,
    iterations: int = 160,
    mu: float | None = None,
    grip_mu: float = GRIP_MU,
    force_limit: float = GRIP_LIMIT_N,
    squeeze: float = GRIP_SQUEEZE,
    foam_young: float = FOAM_YOUNG,
):
    out.mkdir(parents=True, exist_ok=True)
    sim, obj, geometry, report = build_scene(
        solver_name=solver_name,
        iterations=iterations,
        mu=mu,
        grip_mu=grip_mu,
        force_limit=force_limit,
        foam_young=foam_young,
    )
    report["overpress_m"] = overpress
    entity = obj.entity
    control = arm_gripper.ArmControl(sim)
    particles = np.asarray(entity.particles)
    tets, labels = entity.source_tets, entity.tet_part_labels
    nodes, tips, tip_faces, area = geometry["nodes"], geometry["tips"], geometry["tip_faces"], geometry["tip_face_area"]
    info = geometry["report"]
    surface = np.asarray(rss.get_cell(CELL).surface_frame[:3], dtype=np.float64)
    paper_top, radius = 2.0 * PAPER_HALF[2], PHYSICS.particle_radius
    soft_tets = np.isin(labels, SOFT)
    soft_nodes = np.concatenate((nodes[FOAM], nodes[DIE]))
    body = entity.rigid_parts[0]
    quality = MeshQuality(sim, entity)
    trajectory = Trajectory(sim, entity)
    arms = sim.handles.arms[arm_gripper.ARM]
    finger_sides = [
        np.array([int(shape) for _, shapes in side for shape in shapes], dtype=np.int32) for side in arms.finger_groups
    ]
    shape_forces = getattr(sim.pipeline.program.solver, "shape_contact_forces", None)
    start = sim.pipeline.state_0.particle_q.numpy()[particles].astype(np.float64) - surface
    width = float(np.ptp(start[soft_nodes, 0]))
    grasp = surface + np.array((PICK_AT[0], PICK_AT[1], float(start[tips, 2].mean()) + GRIP_Z))
    closed = 2.0 * (KNOB_RADIUS - squeeze)

    size = round(2.0 * INK_HALF / INK_PIXEL)
    mask = np.zeros((size, size), dtype=bool)
    inked_at = np.full((len(tip_faces), 2), np.inf)
    face_inked = np.zeros(len(tip_faces), dtype=bool)
    rows, phases, executed = [], [], []
    frame, worst_ik, held = 0, 0.0, None
    target, quat, gap = grasp + np.array((0.0, 0.0, HOVER)), None, OPEN_WIDTH

    videos, renderer, video_error = None, None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet
        from robosimstudio.solvers.render import FALLBACK

        backend, _why = sim.platform.renderer_for("")
        renderer = sim.make_renderer(backend, cameras=list(sim.cameras), channels=["rgb"])
        if backend != FALLBACK:
            renderer.render(sim.state)  # the first frame only builds the stage
        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // CONTROL_HZ))
        videos.__enter__()

    def route(begin, end, q_begin, q_end, gap_begin: float, gap_end: float, seconds: float):
        """A smoothstep from one tool pose and pad gap to another, as ``clawhauser_arm_pick_place`` plans one."""
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
            half_gap_at=lambda clock: 0.5 * (gap_begin + (gap_end - gap_begin) * blend(clock)),
        )

    def step(name: str) -> None:
        """One control frame, and its row."""
        nonlocal frame
        sim.run(1).sync()
        frame += 1
        state = sim.pipeline.state_0
        q, body_q = state.particle_q.numpy(), state.body_q.numpy()
        owned = q[particles].astype(np.float64)
        local = owned - surface
        # Ink: a relief node over the sheet, at its contact distance from it.
        over = local[:, 2] - paper_top - radius
        on_sheet = (np.abs(local[:, :2] - np.asarray(PAPER_AT)) <= np.asarray(PAPER_HALF[:2])).all(axis=1)
        touching = np.zeros(len(owned), dtype=bool)
        touching[tips[(over[tips] <= TOUCH_TOLERANCE) & on_sheet[tips]]] = True
        face_touch = touching[tip_faces].all(axis=1)
        if face_touch.any():
            centroid = local[tip_faces][:, :, :2].mean(axis=1)
            fresh = face_touch & (np.linalg.norm(centroid - inked_at, axis=1) > 0.4 * INK_PIXEL)
            pixels = (local[tip_faces[fresh]][:, :, :2] - (np.asarray(PAPER_AT) - INK_HALF)) / INK_PIXEL
            stamp_press._ink(mask, pixels)
            inked_at[fresh] = centroid[fresh]
            face_inked[:] |= face_touch
        inverted = np.linalg.det(owned[tets[:, 1:]] - owned[tets[:, :1]]) <= 0
        pose = body_q[body.body]
        foam = float(local[geometry["under_mount"], 2].mean() - local[geometry["under_foam"], 2].mean())
        die = float(local[geometry["under_foam"], 2].mean() - local[tips, 2].mean())
        tcp = np.asarray(control.tcp(), dtype=np.float64)
        parts = rigid_part_body.frame_record(entity, owned, body_q)
        row = {
            "frame": frame,
            "time_s": frame / CONTROL_HZ,
            "phase": name,
            "finite": bool(np.isfinite(owned).all() and np.isfinite(body_q).all()),
            "tcp_m": (tcp - surface).tolist(),
            "commanded_tcp_z_m": float(target[2] - surface[2]),
            "aperture_m": 2.0 * control.aperture(),
            "centre_m": local[soft_nodes, :2].mean(axis=0).tolist(),
            "base_z_m": float(local[tips, 2].mean()),
            "body_z_m": float(pose[2] - surface[2]),
            "tilt_deg": stamp_press._tilt_deg(pose),
            "yaw_deg": stamp_press._yaw_deg(pose),
            "relief_on_paper": float(area[face_touch].sum() / area.sum()),
            "foam_squeeze_m": info["foam_thickness_m"] - foam,
            "die_squeeze_m": info["die_thickness_m"] - die,
            "bulge_m": 0.5 * (float(np.ptp(local[soft_nodes, 0])) - width),
            "inverted_soft_tets": int(np.count_nonzero(inverted & soft_tets)),
            "inverted_rigid_tets": int(np.count_nonzero(inverted & ~soft_tets)),
            "rigid_node_stretch_m": float(
                max(max(p["seam_stretch_max_m"], p["skin_stretch_max_m"]) for p in parts.values())
            ),
        }
        row["squeeze_m"] = row["foam_squeeze_m"] + row["die_squeeze_m"]
        if held is not None:
            # How far the knob has moved down in the pads since they closed on it.
            row["grip_slip_m"] = float((tcp[2] - pose[2]) - held)
        if shape_forces is not None:
            load, pinch = arm_gripper.pad_forces(sim, shape_forces(), finger_sides)
            row["pad_force_n"], row["squeeze_n"] = max(load), max(pinch)
        rows.append(row)
        trajectory.add(row["time_s"], owned, body_q)
        if videos is not None and frame % render_every == 0:
            videos.append_frames(renderer.render(sim.state))
        if not row["finite"]:
            raise RuntimeError(f"{name}: nonfinite state at frame {frame}")

    def finish(name: str, count: int) -> None:
        phases.append({"phase": name, "seconds": count / CONTROL_HZ, "ends_at_s": rows[-1]["time_s"]})
        print(json.dumps(rows[-1]), flush=True)

    def settle() -> None:
        """Hold the open hand over the knob while the stamp settles, then read the tool's pose."""
        nonlocal target, quat
        hold = route(target, target, arm_gripper.GRASP_QUAT, arm_gripper.GRASP_QUAT, gap, gap, 0.1)
        if hold.worst_position_error_m > MAX_IK_ERROR_M:
            raise RuntimeError(f"the hover pose is unreachable: {hold.worst_position_error_m:.4f} m off")
        control.start(hold)
        count = round(SETTLE_S * CONTROL_HZ)
        for index in range(count):
            control.follow(hold, index)
            step("settle")
        position, orientation = control.arm.ee_pose(sim.pipeline.state_0)
        target, quat = np.asarray(position, dtype=np.float64), np.asarray(orientation, dtype=np.float64)
        low, high = arm_gripper.pad_bounds(sim)
        print(
            json.dumps({"pads_below_tcp_m": float(target[2] - low[2]), "pads_above_tcp_m": float(high[2] - target[2])})
        )
        finish("settle", count)

    def act(name: str, delta, *, opening: float | None = None, seconds: float | None = None) -> None:
        """One action: from the measured tool pose and gap to the commanded ones."""
        nonlocal target, gap, worst_ik
        delta = np.asarray(delta, dtype=np.float64)
        if seconds is None:
            seconds = max(0.8, float(np.linalg.norm(delta)) / 0.06)
        end = target + delta
        end_gap = gap if opening is None else float(opening)
        position, orientation = control.arm.ee_pose(sim.pipeline.state_0)
        # A gap that is not changing stays the commanded one. Read back, a stalled pad's gap is wider
        # than its command, and planning from it let the grip off for the first frames of every action.
        begin_gap = gap if opening is None else 2.0 * control.aperture()
        path = route(np.asarray(position, dtype=np.float64), end, orientation, quat, begin_gap, end_gap, seconds)
        worst_ik = max(worst_ik, path.worst_position_error_m)
        if path.worst_position_error_m > MAX_IK_ERROR_M:
            raise RuntimeError(f"{name}: the arm misses its path by {path.worst_position_error_m:.4f} m")
        target, gap = end, end_gap
        for index in range(len(path.joint_q)):
            control.follow(path, index)
            step(name)
        executed.append({"phase": name, "delta_m": delta.tolist(), "seconds": seconds, "opening_m": end_gap})
        finish(name, len(path.joint_q))

    def close() -> None:
        nonlocal held
        act("close", (0.0, 0.0, 0.0), opening=closed, seconds=1.0)
        state = sim.pipeline.state_0
        held = float(np.asarray(control.tcp())[2] - state.body_q.numpy()[body.body][2])

    def rest() -> None:
        count = round(0.6 * CONTROL_HZ)
        for _ in range(count):
            step("rest")
        finish("rest", count)

    def steps():
        """The script. Each line is one phase; a caller may stop after any of them."""
        yield "settle", settle
        yield "reach", lambda: act("reach", (0.0, 0.0, -HOVER), seconds=1.4)
        yield "close", close
        yield "lift", lambda: act("lift", (0.0, 0.0, LIFT), seconds=1.5)
        yield "hang", lambda: act("hang", (0.0, 0.0, 0.0), seconds=0.6)
        yield "carry", lambda: act("carry", (PAPER_AT[0] - PICK_AT[0], PAPER_AT[1] - PICK_AT[1], 0.0), seconds=2.0)
        # Back to the height it stood at, on the paper now.
        yield "lower", lambda: act("lower", (0.0, 0.0, paper_top - LIFT), seconds=2.0)
        yield "press", lambda: act("press", (0.0, 0.0, -overpress), seconds=1.0)
        yield "hold", lambda: act("hold", (0.0, 0.0, 0.0), seconds=1.0)
        yield "unload", lambda: act("unload", (0.0, 0.0, overpress), seconds=0.8)
        yield "open", lambda: act("open", (0.0, 0.0, 0.0), opening=OPEN_WIDTH, seconds=0.8)
        yield "retreat", lambda: act("retreat", (0.0, 0.0, RETREAT), seconds=1.0)
        yield "rest", rest

    started = time.perf_counter()
    error = None
    try:
        for name, phase in steps():
            phase()
            if name == until:
                break
    except RuntimeError as exc:
        error = str(exc)
        print(json.dumps({"stopped": error}), flush=True)
    finally:
        if videos is not None:
            try:
                videos.__exit__(None, None, None)
            except Exception as exc:
                video_error = repr(exc)
        if renderer is not None:
            renderer.close()
    elapsed = time.perf_counter() - started

    trajectory.save(out / "trajectory.npz")
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    image[mask] = INK_RGB
    try:
        from PIL import Image

        Image.fromarray(image[::-1]).save(out / "imprint.png")  # row 0 at the top: +y up, as seen from above
    except ImportError:
        np.save(out / "imprint.npy", mask)
    complete = error is None and bool(phases) and phases[-1]["phase"] == PHASES[-1]
    result = judge(rows, report, complete)
    result["imprint"] = {
        "relief_inked": float(area[face_inked].sum() / area.sum()),
        "inked_area_m2": float(mask.sum() * INK_PIXEL**2),
        "relief_area_m2": info["relief_area_m2"],
    }
    result["checks"]["imprint_complete"] = result["imprint"]["relief_inked"] >= 0.95
    result["success"] = all(result["checks"].values())
    result["stopped"] = error
    result["mesh_quality"] = quality.read()
    result["actions"], result["phases"] = executed, phases
    result["worst_ik_error_m"] = worst_ik
    result["config"] = {
        "overpress_m": overpress,
        "grip_mu": grip_mu,
        "force_limit_n": force_limit,
        "grip_squeeze_m": squeeze,
        "solver": type(arm_gripper.vbd_solver(sim.pipeline.program.solver)).__name__,
        "iterations": iterations,
        "substeps": PHYSICS.substeps,
        "control_hz": CONTROL_HZ,
        "newton_version": newton.__version__,
    }
    result["geometry"] = report
    result["video_error"] = video_error
    result["simulated_s"] = len(rows) / CONTROL_HZ
    result["elapsed_s"] = elapsed
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "actions", "phases")}), flush=True)
    sim.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--until", choices=PHASES, default=PHASES[-1], help="stop after this phase")
    parser.add_argument("--overpress", type=float, default=OVERPRESS, help="the hand's push past standing height [m]")
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default=stamp_press.SOLVER)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--mu", type=float, default=None, help="the stamp's soft-contact friction")
    parser.add_argument("--grip-mu", type=float, default=GRIP_MU, help="pad-knob friction")
    parser.add_argument("--force-limit", type=float, default=GRIP_LIMIT_N, help="one finger's force [N]")
    parser.add_argument("--squeeze", type=float, default=GRIP_SQUEEZE, help="pad command past the knob's radius [m]")
    parser.add_argument("--foam-young", type=float, default=FOAM_YOUNG, help="the cushion's Young's modulus [Pa]")
    args = parser.parse_args()
    run(
        args.out,
        render_every=args.render_every,
        until=args.until,
        overpress=args.overpress,
        solver_name=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        grip_mu=args.grip_mu,
        force_limit=args.force_limit,
        squeeze=args.squeeze,
        foam_young=args.foam_young,
    )


if __name__ == "__main__":
    main()
