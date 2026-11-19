# SPDX-License-Identifier: Apache-2.0
"""Pack the bag and lift it: the AIRBOT drops the USB hub into an open cloth bag, then carries the bag off the bench.

The bag (``cloth_bag_fem``: a soft 3 mm FEM cloth shell, an aluminium rim with an arched handle, a base board) collapses
under its frame to a pile with the frame ring on top. A sponge wand and a silicone spatula start inside the ring, resting
on the cloth (``--pick-all``: all three lie beside the bag and the arm packs each). The hand takes the hub by its housing,
carries it over the ring and lets it go. Then it pinches the handle's apex, lifts the bag ``--lift`` m and holds.

Success: every packed item's rigid part is inside the bag's volume at the end and the base board is lifted at least
0.10 m. Score: items inside / items packed x min(1, base lift / 0.10).

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import arm_gripper
import cable_scene
import cable_tray_multi as tray
import hetero_scene
import numpy as np
import tool_scene
from clawhauser_scene import REFERENCE, Trajectory, arrangement, build_sim
from loop_geometry import smooth

import robosimstudio as rss

CONTROL_HZ = tool_scene.CONTROL_HZ
SUBSTEPS = tool_scene.SUBSTEPS
BAG = hetero_scene.HeteroAsset("cloth_bag_fem_coarse", hetero_scene.TURN_Z, 1.0e5, 1000.0, None)
"""Soft cloth (E 1e5, damping 1000 Pa*s): the bag may sag, the frame keeps the mouth open. Turned a quarter about z, so the handle's arch lies along x and the bag is narrow in x."""
BAG_AT = (0.22, -0.12)
"""The bag's centre [m, bench frame]: where the hand-down arm reaches the handle apex."""
RIM_Z = 0.272
"""The rim's top over the bag's own origin [m]; the bag's volume is below it."""
APEX_Z = 0.30
"""Frame nodes above this (at the handle's middle) are the grip's apex [m, asset frame]."""
ITEMS = {
    "sponge": ("sponge_wand", (-0.19, -0.14)),
    "spatula": ("silicone_spatula", (-0.10, -0.24)),
    "hub": ("usb_hub_gen", (-0.12, -0.22)),
}
"""Item -> asset and where its grip lands in the bench frame [m], in picking order."""
PRE = {"sponge": (0.17, -0.12), "spatula": (0.27, -0.12)}
"""Items that start inside the bag (x, bench frame [m]); the arm picks the others."""
PRE_TILT = np.radians(25.0)
"""They lie diagonally, the -y end on the base board and the +y end up towards the rim, so they fit between the walls."""
DROP_DY = -0.12
"""The drop's y from the bag's centre [m]: the middle of the arm's side of the mouth, clear of the handle's arch."""
SQUEEZE = 0.016
TILTS = (0.0, 25.0, 50.0, 75.0, -25.0)
"""Hand pitches tried in turn for the bag handle [deg, about x]."""
OPEN = 0.04
HUB_HANG = 0.40
"""How far the hub's cable and plug hang under the pad point [m]."""
IK_STRIDE = 6
"""Control frames per IK solve of the arm's route."""


class Route:
    """Keyframes of the hand: pad point, pitch about x and half gap, each move a smoothstep."""

    def __init__(self, point, gap: float) -> None:
        self.k = [(0.0, np.asarray(point, dtype=np.float64), 0.0, gap, 0.0)]
        self.marks: dict[str, float] = {}

    def go(self, dt: float, point=None, tilt=None, gap=None, yaw=None) -> None:
        t, p, a, g, w = self.k[-1]
        self.k.append(
            (
                t + dt,
                p if point is None else np.asarray(point, dtype=np.float64),
                a if tilt is None else tilt,
                g if gap is None else gap,
                w if yaw is None else yaw,
            )
        )

    @property
    def end(self) -> float:
        return self.k[-1][0]

    def at(self, time: float):
        for (t0, p0, a0, g0, w0), (t1, p1, a1, g1, w1) in zip(self.k, self.k[1:], strict=False):
            if time < t1:
                s = smooth(time, t0, t1)
                return p0 + (p1 - p0) * s, a0 + (a1 - a0) * s, g0 + (g1 - g0) * s, w0 + (w1 - w0) * s
        return self.k[-1][1:]


def hand_quat(tilt: float, yaw: float = 0.0) -> np.ndarray:
    """The grasp orientation turned ``yaw`` about z, then pitched ``tilt`` about world x."""
    turned = arm_gripper.quat_multiply(arm_gripper.axis_angle_quat((0.0, 0.0, 1.0), yaw), arm_gripper.GRASP_QUAT)
    return arm_gripper.quat_multiply(arm_gripper.axis_angle_quat((1.0, 0.0, 0.0), tilt), turned)


def item_specs(args, settings, scene):
    """Add the items to the scene; returns ``{name: dict}`` with each one's asset, mesh and handle."""
    items = {}
    for name in args.items:
        key, (gx, gy) = ITEMS[name]
        if name == "hub":
            asset = cable_scene.ASSETS[key]
            vertices, tets, labels = hetero_scene.load(asset)
            geometry = tray.layout(vertices, tets, labels, asset.quat, (gx, gy))
            spec = hetero_scene.spec(
                asset, vertices, tets, labels, name=name, pose=geometry["pose"], settings=settings, mu=args.mu
            )
            items[name] = {"asset": asset, "geometry": geometry, "rigid": 0}
        else:
            base = tool_scene.TOOLS[key]
            yaw = arm_gripper.axis_angle_quat((0.0, 0.0, 1.0), np.pi / 2)
            asset = dataclasses.replace(base, quat=tuple(float(c) for c in arm_gripper.quat_multiply(yaw, base.quat)))
            vertices, tets, labels = hetero_scene.load(asset)
            pre = name in PRE and not args.pick_all
            # Items packed in advance shrink only as far as they must to lie diagonally in the bag (0.48 m at full size):
            # a small sponge is a few grams and its foam inverts.
            length = np.ptp(arm_gripper._rotate(np.asarray(asset.quat), vertices)[:, 1])
            scale = min(1.0, 0.9 * 0.48 * args.bag_scale / length) if pre else 1.0
            vertices = vertices * scale
            grip = tool_scene.grasp_handle(asset, vertices, tets, labels)
            at = (gx - grip["point_m"][0], gy - grip["point_m"][1])
            pose = tool_scene.rest_pose(vertices, asset.quat, at=at)
            if pre:
                k = args.bag_scale
                tilt = arm_gripper.axis_angle_quat((1.0, 0.0, 0.0), PRE_TILT)
                asset = dataclasses.replace(
                    asset, quat=tuple(float(c) for c in arm_gripper.quat_multiply(tilt, asset.quat))
                )
                turned = arm_gripper._rotate(np.asarray(asset.quat), vertices)
                # The low end just inside the base board's -y edge (0.181 m from the centre), 5 mm over the board (15 mm thick).
                x = args.bag_at[0] + (PRE[name][0] - BAG_AT[0]) * k - turned[:, 0].mean()
                y = args.bag_at[1] - 0.17 * k - turned[:, 1].min()
                pose = (x, y, 0.015 * k + 0.005 - turned[:, 2].min())
            spec = hetero_scene.spec(
                asset, vertices, tets, labels, name=name, pose=pose, settings=settings, mu=args.mu,
                overrides=asset.overrides(), scale=scale,
            )  # fmt: skip
            items[name] = {"asset": asset, "rigid": asset.handle, "pre": pre}
        items[name].update(labels=labels, vertices=vertices, handle=scene.add(spec))
    return items


def build(args):
    settings = dataclasses.replace(REFERENCE, substeps=args.substeps, particle_radius=0.0015)
    physics = arrangement(
        settings,
        claw="arm",
        coarse=False,
        solver_name=args.solver,
        iterations=args.iterations,
        self_contact=args.self_contact,
        feedback_smoothing=0.15,
        feedback_clamp=1.5,
        self_contact_radius=0.002,
        rest_exclusion=0.006,
    )
    scene = rss.Scene(cell=arm_gripper.CELL, physics=physics, name="pack_bag")
    bag = dataclasses.replace(
        BAG, cable_young=args.bag_young or BAG.cable_young, cable_damping=args.bag_damping or BAG.cable_damping
    )
    vertices, tets, labels = hetero_scene.load(bag)
    vertices = vertices * args.bag_scale
    overrides = ((1, "density_kg_m3", args.frame_density),) if args.frame_density else ()
    pose = (args.bag_at[0], args.bag_at[1], hetero_scene.PLACE_GAP)
    bag_obj = scene.add(
        hetero_scene.spec(
            bag,
            vertices,
            tets,
            labels,
            name="bag",
            pose=pose,
            settings=settings,
            mu=args.mu,
            overrides=overrides,
            scale=args.bag_scale,
            part_stiffness_scale=args.part_stiffness,
            attach_stiffness=args.attach_stiffness,
        )
    )
    items = item_specs(args, settings, scene)
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=args.force_limit)
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    camera = scene.add(
        rss.CameraSpec(
            name="overview",
            width=960,
            height=540,
            eye=tuple(surface + np.array((0.80, -0.62, 0.62))),
            look_at=tuple(surface + np.array((-0.08, -0.14, 0.18))),
            fov_y=45.0,
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    geometry = {"arm": arm_gripper.check_arm(sim, limit=args.force_limit)}
    geometry["friction"] = hetero_scene.set_friction(sim, solver, settings, mu=args.mu, grip_mu=args.grip_mu)
    return sim, bag_obj, vertices, labels, items, camera, geometry


def run(out: Path, args) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, bag_obj, vertices, labels, items, camera, geometry = build(args)
    surface = np.asarray(hetero_scene.sim_surface(), dtype=np.float64)
    bag = bag_obj.entity
    btets = bag.source_tets
    frame_nodes = np.unique(btets[labels == 1])
    base_nodes = np.unique(btets[labels == 2])
    k = args.bag_scale
    rim_nodes = frame_nodes[vertices[frame_nodes, 2] < RIM_Z * k]
    apex_nodes = frame_nodes[(vertices[frame_nodes, 2] > APEX_Z * k) & (np.abs(vertices[frame_nodes, 1]) < 0.012 * k)]
    half = vertices[rim_nodes, :2].max(axis=0)[::-1]
    cloth = hetero_scene.soft_labels(BAG, labels)
    picked = [n for n, i in items.items() if not i.get("pre")]
    arm = arm_gripper.ArmControl(sim)
    dt = 1.0 / CONTROL_HZ

    def local_all() -> np.ndarray:
        return sim.pipeline.state_0.particle_q.numpy().astype(np.float64) - surface

    # What each item needs to be taken: nodes and rigid part, from the entity's own indexing.
    for name, item in items.items():
        entity = item["handle"].entity
        t, lab = entity.source_tets, entity.tet_part_labels
        item["entity"] = entity
        item["rigid_nodes"] = np.unique(t[lab == item["rigid"]])
        if name == "hub":
            item["soft_nodes"] = np.unique(t[lab == 1])
        else:
            item["soft_nodes"] = np.unique(t[np.isin(lab, hetero_scene.soft_labels(item["asset"], item["labels"]))])
        item["trajectory"] = Trajectory(sim, entity)

    def measure(q: np.ndarray, settled: bool = False) -> dict:
        """Where and how every grasp happens, from the particles ``q`` [m, bench frame]."""
        out_ = {}
        for name, item in items.items():
            if item.get("pre"):
                continue
            now = q[item["entity"].particles]
            if name == "hub":
                g = item["geometry"]
                out_[name] = {"point": np.array(g["grip_centre_m"]), "half": g["reach_m"], "open": tray.hub.OPENED_HALF_GAP,
                              "shut": g["reach_m"] - tray.hub.GRIP_SQUEEZE, "tilt": 0.0, "lo": 0.0, "hi": 0.4}  # fmt: skip
                continue
            axis = now[item["soft_nodes"]].mean(axis=0) - now[item["rigid_nodes"]].mean(axis=0)
            axis /= np.linalg.norm(axis)
            grip = tool_scene.grasp_handle(item["asset"], now, item["entity"].source_tets, item["labels"], axis=axis)
            point = np.array(grip["point_m"])
            point[2] = max(point[2], 0.025)
            along = now @ axis
            origin = point @ axis
            # The pitch about x that points the working end up by ``args.drop_deg``, the handle end down: a hand pitched forward reaches highest.
            tilt = np.radians(args.drop_deg) - np.arctan2(axis[2], axis[1])
            out_[name] = {"point": point, "half": grip["half_width_m"], "open": min(OPEN, grip["half_width_m"] + 0.014),
                          "shut": grip["half_width_m"] - SQUEEZE, "tilt": float(tilt), "lo": float(along.min() - origin),
                          "hi": float(along.max() - origin)}  # fmt: skip
        now = q[bag.particles]
        apex = now[apex_nodes]
        out_["bag"] = {
            "point": apex.mean(axis=0),
            "half": float(0.5 * np.ptp(apex[:, 1])),
            "centre": now[rim_nodes, :2].mean(axis=0),
            "rim_top": float(now[rim_nodes, 2].max()) if settled else args.rim_top,
        }
        return out_

    def route(m: dict, bag_tilt: float, settle: float) -> Route:
        top = args.carry_z
        r = Route(m[picked[0]]["point"] + (0.0, 0.0, 0.08) if picked else m["bag"]["point"] + (0.0, 0.0, 0.04), OPEN)
        r.go(settle, gap=OPEN)
        bag_xy = m["bag"]["centre"]
        b_rim = m["bag"]["rim_top"]
        for name in picked:
            c = m[name]
            g, hover = c["point"], c["point"] + (0.0, 0.0, 0.08)
            drop = np.array((bag_xy[0], bag_xy[1] + DROP_DY * k))
            if name == "hub":
                # As in the hub-into-box task: lower until the housing is just over the pile, so it isn't dropped.
                z = b_rim + 0.02 + g[2]
                top_here = max(top, b_rim + HUB_HANG)
            else:
                # The tilted tool's handle end clears the rim by 0.05 m.
                z = b_rim + 0.05 - c["lo"] * np.sin(np.radians(args.drop_deg))
                top_here = top
            if name != picked[0]:
                r.go(1.8, (hover[0], hover[1], top), tilt=0.0)
            r.go(1.2, hover, gap=c["open"], tilt=0.0)
            r.go(0.9, g)
            r.go(0.8, gap=c["shut"])
            r.go(0.3)
            if name == "hub":
                # The hub-into-box task's slow phases (lift 3 s, carry 2 s, lower 2 s, 0.3 s pauses) keep the cable from swinging.
                r.go(3.0, (g[0], g[1], top_here), tilt=c["tilt"])
                r.go(0.3)
                r.go(2.0, (drop[0], drop[1], top_here))
                r.go(0.3)
                r.go(2.0, (drop[0], drop[1], z))
            else:
                r.go(1.0, (g[0], g[1], g[2] + 0.12))
                r.go(1.2, (g[0], g[1], top_here), tilt=None)
                r.go(2.0, (drop[0], drop[1], top_here))
                r.go(1.5, (drop[0], drop[1], z), tilt=c["tilt"])
            r.go(0.4)
            r.go(0.6, gap=OPEN)
            r.marks[f"{name}_release"] = r.end
            r.go(0.8, (drop[0], drop[1], max(z, top_here)))
        b = m["bag"]
        point = b["point"] + (0.0, 0.0, arm_gripper.PAD_RISE)  # the pads' centre on the tube, not its lower edge
        hover = point + (0.0, 0.0, 0.04 * k)
        r.go(2.5, hover, tilt=bag_tilt, gap=b["half"] + 0.012 * k, yaw=np.pi / 2)
        r.marks["bag_hover_end"] = r.end
        r.go(0.9, point)
        r.go(0.8, gap=b["half"] - 0.012 * k)
        r.go(args.pre_lift_hold)
        r.marks["lift_start"] = r.end
        r.go(args.lift_s, point + (0.0, 0.0, args.lift))
        r.go(args.hold)
        return r

    steps_cap = {}

    def solve(r: Route, start: float = 0.0, until: float | None = None):
        """The arm's route for ``r`` from ``start`` s: IK at 10 Hz, the joints interpolated to the control rate."""
        at = r.at
        frames = round(((r.end if until is None else until) - start) * CONTROL_HZ) + 1
        samples = -(-frames // IK_STRIDE) + 1
        coarse = arm.plan(
            quat=lambda t: hand_quat(at(t + start)[1], at(t + start)[3]),
            capacity=samples,
            dt=IK_STRIDE * dt,
            tcp_at=lambda t: arm_gripper.tcp_for_pads(surface + at(t + start)[0]),
            half_gap_at=lambda t: at(t + start)[2],
        )
        grid, fine = np.arange(samples), np.arange(frames) / IK_STRIDE
        joints = np.stack([np.interp(fine, grid, coarse.joint_q[:, c]) for c in range(coarse.joint_q.shape[1])], axis=1)
        command = np.interp(fine, grid, coarse.command)
        return arm_gripper.Route(joints.astype(np.float32), command.astype(np.float32), coarse.worst_position_error_m)

    def plan(m: dict, settle: float, strict: bool = True):
        # The hand pitch for the bag's handle is probed on that stretch alone, then the whole route is solved.
        probe = route(m, 0.0, settle)
        start = probe.end - (5.9 + args.pre_lift_hold + args.lift_s + args.hold)
        tilt, error = 0.0, 1e9
        angles = TILTS if args.bag_tilt_deg is None else (args.bag_tilt_deg,)
        for angle in angles:
            e = solve(route(m, np.radians(angle), settle), start).worst_position_error_m
            if e < error:
                tilt, error = angle, e
            if e < 0.015:
                break
        r = route(m, np.radians(tilt), settle)
        planned = solve(r)
        steps_cap["marks"] = r.marks
        steps_cap["route"], geometry["bag_tilt_deg"], geometry["arm_path_error_m"] = (
            planned,
            tilt,
            planned.worst_position_error_m,
        )
        print(
            json.dumps({"tilt_deg": tilt, "path_error_m": planned.worst_position_error_m, "duration_s": r.end}),
            flush=True,
        )
        if strict and planned.worst_position_error_m > 0.025:
            # Which move first misses: the worst error over the route up to each keyframe.
            for k in r.k[1:]:
                print(
                    json.dumps(
                        {
                            "to_t": round(k[0], 1),
                            "p": np.round(k[1], 3).tolist(),
                            "tilt": round(k[2], 2),
                            "error_m": round(solve(r, 0.0, k[0]).worst_position_error_m, 3),
                        }
                    ),
                    flush=True,
                )
            raise RuntimeError(f"the arm misses the path by {planned.worst_position_error_m:.3f} m")

    settle = args.settle
    measured = measure(local_all())
    # The first plan sees the bag standing, its handle higher than it will be once the bag has collapsed: only the later plans must reach.
    plan(measured, settle, strict=False)
    if args.plan_only:
        return {}
    steps = args.steps or len(steps_cap["route"].joint_q)
    arm.start(steps_cap["route"])
    settle_frame = round(settle * CONTROL_HZ)

    last_m = {}
    hover_frame = round(steps_cap["marks"]["bag_hover_end"] * CONTROL_HZ)

    def follow(frame: int):
        if frame == settle_frame and frame > 0:
            m = measure(local_all(), settled=True)
            last_m.update(m)
            plan(m, settle)
            geometry["settled"] = {k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()} for k, v in m.items()}
        if frame == hover_frame and last_m:
            # Over the handle: aim at its apex as the bag stands now (the pile may have drifted since the settle).
            m = dict(last_m)
            m["bag"] = measure(local_all(), settled=True)["bag"]
            last_m.update(m)
            geometry["apex_at_grasp"] = np.asarray(m["bag"]["point"]).tolist()
            plan(m, settle)
        arm.follow(steps_cap["route"], frame)
        return arm

    q0 = local_all()
    base_rest = float(q0[bag.particles][base_nodes, 2].mean())

    def inside(p: np.ndarray, rim_c: np.ndarray, rim_z: float, base_z: float) -> np.ndarray:
        return (
            (np.abs(p[:, 0] - rim_c[0]) < half[0] - 0.01)
            & (np.abs(p[:, 1] - rim_c[1]) < half[1] - 0.01)
            & (p[:, 2] < rim_z + 0.03)
            & (p[:, 2] > base_z - 0.01)
        )

    def task_row(frame: int, local: np.ndarray, body_q: np.ndarray) -> dict:
        q = local_all()
        rim_c, rim_z = local[rim_nodes, :2].mean(axis=0), float(local[rim_nodes, 2].max())
        base_z = float(local[base_nodes, 2].mean())
        row = {
            "finite": bool(np.isfinite(q).all()),
            "base_z_m": base_z,
            "rim_z_m": float(local[rim_nodes, 2].mean()),
            "rim_c_m": rim_c.tolist(),
        }
        for name, item in items.items():
            p = q[item["entity"].particles]
            centre = p[item["rigid_nodes"]].mean(axis=0)
            row[name] = {
                "centre_m": centre.tolist(),
                "to_ring_m": (centre - np.array((*rim_c, rim_z))).tolist(),
                "inside": bool(inside(centre[None], rim_c, rim_z, base_z)[0]),
                "inside_fraction": float(inside(p, rim_c, rim_z, base_z).mean()),
                "inverted": int(
                    np.count_nonzero(
                        np.linalg.det(p[item["entity"].source_tets[:, 1:]] - p[item["entity"].source_tets[:, :1]]) <= 0
                    )
                ),
            }
            item["trajectory"].add((frame + 1) / CONTROL_HZ, p + surface, body_q)
        return row

    blown = {"hit": False}

    def stop(row: dict) -> bool:
        """End the run when the state is not finite or something fell through the bench: a blown-up bag."""
        blown["hit"] = (not row["finite"]) or row["minimum_z_m"] < -0.05
        return blown["hit"]

    soft = cloth
    rows, elapsed, video_error = hetero_scene.episode(
        sim, bag, camera, out, soft=soft, steps=steps, render_every=args.render_every, control_hz=CONTROL_HZ,
        follow=follow, task_row=task_row, stop=stop,
    )  # fmt: skip
    for name, item in items.items():
        item["trajectory"].save(out / f"trajectory_{name}.npz")
    last = rows[-1]
    base_lift = float(last["base_z_m"] - base_rest)
    inside_now = {n: last[n]["inside"] for n in items}
    lift_fraction = float(min(1.0, max(0.0, base_lift) / 0.10))
    score = (sum(inside_now.values()) / max(1, len(items))) * lift_fraction
    parts = [k for k in rows[0] if k.startswith("part") and k.endswith("_body")]
    marks = steps_cap["marks"]

    def at_time(t: float, name: str) -> dict:
        row = rows[min(len(rows), max(1, round(t * CONTROL_HZ))) - 1]
        return {"time_s": row["time_s"], "to_ring_m": row[name]["to_ring_m"], "inside": row[name]["inside"]}

    peak = max(range(len(rows)), key=lambda i: rows[i]["base_z_m"])
    phases = {
        n: {
            "before_lift": at_time(marks["lift_start"], n),
            "peak_lift": {
                "time_s": rows[peak]["time_s"],
                "to_ring_m": rows[peak][n]["to_ring_m"],
                "inside": rows[peak][n]["inside"],
            },
            "end": {"time_s": last["time_s"], "to_ring_m": last[n]["to_ring_m"], "inside": last[n]["inside"]},
        }
        for n in items
    }
    release = {n: at_time(marks[f"{n}_release"] + 1.0, n) for n in picked if f"{n}_release" in marks}
    result = {
        "item_phases": phases,
        "released_inside_ring": {n: v["inside"] for n, v in release.items()},
        "released_to_ring_m": {n: v["to_ring_m"] for n, v in release.items()},
        "items": list(items),
        "items_inside": inside_now,
        "inside_fraction": {n: last[n]["inside_fraction"] for n in items},
        "item_inverted_max": {n: max(r[n]["inverted"] for r in rows) for n in items},
        "base_lift_m": base_lift,
        "base_lift_max_m": float(max(r["base_z_m"] for r in rows) - base_rest),
        "lift_fraction": lift_fraction,
        "bag_mass_kg": float(bag.mass.sum() + sum(rp.part_mass for rp in bag.rigid_parts)),
        "bag_rigid_mass_kg": {f"part{rp.part}": float(rp.part_mass) for rp in bag.rigid_parts},
        "items_mass_kg": {n: float(i["entity"].mass.sum()) for n, i in items.items()},
        "anchor_gap_max_m": max(
            max(r[k]["seam_stretch_max_m"], r[k]["skin_stretch_max_m"]) for r in rows for k in parts
        ),
        "cloth_stretch_p99_max": max(r["cable_stretch_p99"] for r in rows),
        "cloth_inverted_max": max(r["cable_inverted"] for r in rows),
        "finger_contacts_end": last["finger_contacts"],
        "finite": all(r["finite"] for r in rows),
        "blown_up": bool(blown["hit"]),
        "steps": steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "elapsed_s": elapsed,
        "geometry": geometry,
        "config": json.loads(
            json.dumps(
                hetero_scene.config(args, BAG, dataclasses.replace(REFERENCE, substeps=args.substeps)), default=str
            )
        ),
        "score": float(score),
        "success": bool(all(inside_now.values()) and base_lift >= 0.10),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "config")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=tool_scene.OUTPUT / "tool_pack_bag" / "run")
    parser.add_argument(
        "--pick-all",
        action="store_true",
        help="the arm picks every item (default: sponge and spatula start in the bag)",
    )
    parser.add_argument(
        "--items",
        type=lambda s: [i for i in s.split(",") if i not in ("", "none")],
        default=list(ITEMS),
        help="comma list of: sponge,spatula,hub (none: lift only)",
    )
    parser.add_argument("--steps", type=int, default=0, help="frames (0: the whole route)")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument(
        "--settle", type=float, default=1.0, help="the scene rests this long before the grasps are measured [s]"
    )
    parser.add_argument("--lift", type=float, default=0.15, help="how far the bag is lifted [m]")
    parser.add_argument("--bag-scale", type=float, default=0.6, help="the bag's size (and the items packed in it)")
    parser.add_argument(
        "--part-stiffness",
        type=float,
        default=1.0e-3,
        help="the bag's rigid frame and board tets keep this stiffness fraction",
    )
    parser.add_argument("--attach-stiffness", type=float, default=1.0e4, help="the bag's rigid-part node ties [N/m]")
    parser.add_argument("--substeps", type=int, default=SUBSTEPS, help="solver substeps per frame")
    parser.add_argument("--bag-at", type=float, nargs=2, default=BAG_AT, help="the bag's centre, bench frame [m]")
    parser.add_argument("--plan-only", action="store_true", help="plan the arm route, print its error and stop")
    parser.add_argument(
        "--bag-tilt-deg",
        type=lambda v: None if v == "search" else float(v),
        default=0.0,
        help="hand pitch at the handle [deg], or 'search'",
    )
    parser.add_argument("--lift-s", type=float, default=5.0, help="how long the lift takes [s]")
    parser.add_argument(
        "--pre-lift-hold", type=float, default=0.4, help="the closed hand rests this long before the lift [s]"
    )
    parser.add_argument("--hold", type=float, default=1.0, help="how long the lifted bag is held [s]")
    parser.add_argument(
        "--carry-z", type=float, default=0.46, help="pad height of a tool over the bench while carried [m]"
    )
    parser.add_argument(
        "--rim-top", type=float, default=0.06, help="the collapsed bag's rim height assumed by the first plan [m]"
    )
    parser.add_argument(
        "--drop-deg",
        type=float,
        default=60.0,
        help="a tool's working end points up this far at the release (handle end down) [deg]",
    )
    parser.add_argument("--bag-damping", type=float, default=None, help="the bag cloth's damping [Pa*s] (default 1000)")
    parser.add_argument(
        "--bag-young", type=float, default=None, help="the bag cloth's Young's modulus [Pa] (default 1e5)"
    )
    parser.add_argument(
        "--frame-density", type=float, default=None, help="the bag frame's density [kg/m^3] (source: aluminium 2700)"
    )
    tool_scene.add_arguments(parser)
    parser.set_defaults(grip_mu=3.0, force_limit=100.0)
    args = parser.parse_args()
    unknown = set(args.items) - set(ITEMS)
    if unknown:
        parser.error(f"unknown items {sorted(unknown)}")
    run(args.out, args)


if __name__ == "__main__":
    main()
