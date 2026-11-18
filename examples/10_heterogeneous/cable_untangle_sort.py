# SPDX-License-Identifier: Apache-2.0
"""Untangle cabled objects whose cables cross on the bench, and sort each into its own bin.

The generated USB hub, the charger and a Y splitter cable lie in a pile with their cables crossed,
the splitter's lowest, the charger's over it and the hub's on top. Each is rigid ends joined by an
FEM cable (:mod:`rigid_part_body`), and all share one VBD solve, so the cables touch each other
through self-contact. The hub and the charger have a rigid housing, the AIRBOT's handle; the splitter
(and the optional coiled USB cable) has none, so the hand pinches a rigid plug end instead (the
splitter's mini-B trunk, the coil's USB-A). The hand takes the objects from the bottom of the pile up:
lifting one peels its cable off the bench and its tail slides lengthwise out from under the cables
above, while the plug end passes under them. It carries the object to its bin, lowers and opens, then
finds the next where it now lies, turns its wrist to it, comes down on it and takes it away.

Why parts: sorting needs cable-to-cable contact -- one cable sliding along and out from under the
other -- while the rigid ends are handles that keep their shape in the pinch. A stiff homogeneous
object could not lie interleaved or slide its cable out; a soft one could not be lifted cleanly by
its housing or plug.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import time
from pathlib import Path
from typing import NamedTuple

import arm_gripper
import cable_scene
import numpy as np
import rigid_part_body
import usb_hub_box as hub
from clawhauser_scene import REFERENCE, Trajectory, arrangement, build_sim
from loop_geometry import smooth
from scipy.spatial import cKDTree
from usb_hub_box import BOX_FLOOR, BOX_RIM, BOX_WALL, GLASS, HOUSING, LIFT, PHASES, SUBSTEPS

import robosimstudio as rss

OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "cable_untangle_sort"
CELL = arm_gripper.CELL
CONTROL_HZ = REFERENCE.control_hz


class Item(NamedTuple):
    asset: str
    yaw: float
    """Turn about z [rad]; the cable heads to (-sin, cos)."""
    along: float
    """Cable root to ``through``, along the heading [m]."""
    raised: float
    """Start height over the bench's lowest object [m]: the higher cable lies on the lower."""
    through: tuple
    """A bench point the cable passes [m, bench frame]."""
    bin: str
    grasp: tuple
    """Labels of the rigid end the pads pinch: the housing, or a plug end when there is none."""
    cable: int
    """The cable's label."""
    inset: float
    """Where the pads close: this far in from the grasp end's cable end [m]."""
    bends: tuple = ()
    """The cable's laid path from its root (:func:`cable_scene.bend_path`): it starts curved, not straight."""


CROSSING = (-0.066, -0.014)
"""Where the hub's and the charger's cables cross [m, bench frame]."""
SPLITTER_LAID = ((0.10, 0.0), (0.10, 0.2))
CHARGER_LAID = ((0.14, 0.0), (0.06, 0.12))
HUB_LAID = ((0.06, 0.0), (0.12, 0.12), (0.12, -0.12))
"""Each cable's laid path [m]: a gentle bend (radius 0.12-0.2 m) or the hub's S, short of the crossing
shifted by at most a few cm, so the three still cross with the raised one on top."""
OBJECTS = {
    "splitter": Item(
        "y_splitter_cable", 1.296, 0.195, 0.0, (-0.1285, 0.0161), "cables", (1, 2), 0, 0.012, SPLITTER_LAID
    ),
    "charger": Item(
        "charger", -np.pi / 4.0, 0.17, 0.016, CROSSING, "charger", (HOUSING,), 1, hub.GRASP_FROM_ROOT, CHARGER_LAID
    ),
    "hub": Item("usb_hub_gen", np.pi / 4.0, 0.10, 0.032, CROSSING, "hub", (HOUSING,), 1, hub.GRASP_FROM_ROOT, HUB_LAID),
    "coil": Item("coiled_usb_cable", -0.211, 0.0, 0.048, (-0.094, -0.1744), "cables", (1, 2), 0, 0.012),
}
"""The pile, in picking order (``--objects`` picks from it; the default leaves the coil out, 71k tets).

The hub's cable heads to -x+y, the charger's to +x+y, so they cross at right angles, 0.10 m along
the hub's 0.36 m cable and 0.17 m along the charger's 0.23 m one (laid straight; the cables then bend, see ``SPLITTER_LAID``). The charger starts 16 mm over the
splitter and the hub 16 mm over the charger, so each cable lies on the one below. The splitter's
trunk is the lowest and crosses both; the coil lies across all three, its USB-A end clear of them.

Why this way round (q1-sort and 30-frame probes at mu 0.5): an object whose cable is crossed turns
on the bench at about 0.2 rad/s while the cables settle onto each other, the hub either way round
and the 85 g charger hardly at all. Crossed far from the plug end, the turn walks the crossing off
it within about a second.

Picking order, bottom of the pile first: the splitter's lift draws its trunk out from under the
charger's and the hub's cables (and the coil), the charger's then draws its cable out from under the
hub's; the hub's and the coil's just lift off. Taken from the top, nothing would be untangled."""
BINS = {
    "hub": ((0.20, -0.09), (0.16, 0.22)),
    "charger": ((0.22, -0.32), (0.12, 0.14)),
    "cables": ((0.34, -0.48), (0.16, 0.14)),
}
"""Bin centre and inside (x, y) [m, bench frame]. Arm base at (-0.05, -0.42); all stand on +x, where the
hand reaches at the lift's height: the hub's 0.41 m from the base, about usb_hub_box's box (40 mm
shorter), the charger's 0.29 m, beside the arm, the cables' 0.39 m. Towards -x the arm misses a carry
at the lift's height by 40-100 mm, even 0.32 m from the base. The splitter and the coil share the
cables' bin, which is smaller than the hub's: a cable folds into it."""
SETTLE = 1.0
"""Seconds before the first block: the pile settles while the hand comes down over the first grasp."""
FIRST_DESCEND = (0.5, 1.2)
"""When the hand comes down over the first grasp [s]; the pile is still turning (see ``OBJECTS``), so
the first pick too is aimed where its grasp lies just before."""
BLOCK = 13.4
"""Seconds per object: usb_hub_box's phases, then the move to the next object [s]."""
RETURN = (10.9, 12.4)
DESCEND = (12.6, 13.4)
"""Local windows [s]: from over the bin to ``APPROACH_Z`` over the next grasp, turning the wrist,
then down onto it, open."""
REAIM = (RETURN[0], 12.2)
"""When the next grasp is measured again [s]: as the hand leaves the bin, and just before it comes
down (the object can keep moving during the return). The second correction eases in over ``EASE``."""
EASE = 0.4
APPROACH_Z = 0.12
TAIL = 1.4
"""Seconds after the last withdraw."""
TOUCH = 0.004
"""Two objects' particles this close count as a contact [m]: twice the 2 mm self-contact radius."""
DRAG_SCALE = 0.05
"""Score: each pick loses the other objects' largest displacement over this [m], at most 1.

Drag is recorded and scored, not a failure: pulling one cable out from under the others IS the
tangle being pulled apart, and re-aiming at the measured grasp absorbs what it moves."""
CABLE_OK = 0.5
"""A housing-less object is in its bin when its grasp end is and this fraction of its cable is."""
REACH_OK = 0.025
"""The worst distance the arm may miss a planned path by [m]."""


def duration() -> float:
    return SETTLE + BLOCK * (len(OBJECTS) - 1) + PHASES["withdraw"][1] + TAIL


def used_bins() -> dict:
    return {b: BINS[b] for b in dict.fromkeys(o.bin for o in OBJECTS.values())}


def layout(vertices, tets, labels, base_quat, item: Item) -> dict:
    """Where the object lies and where the pads close, in the bench frame.

    Laid flat and long along y, then turned by the item's yaw and moved so the cable root lies
    ``along`` before ``through``.
    """
    turned = arm_gripper._rotate(np.asarray(base_quat), vertices)
    ends = tets[np.isin(labels, item.grasp)]
    grasp = turned[np.unique(ends)]
    root = turned[np.intersect1d(ends, tets[labels == item.cable])].mean(axis=0)
    # Near the grasp end's cable end, which ``root`` marks.
    mid_y = grasp[:, 1].mean()
    near = grasp[:, 1].max() if root[1] > mid_y else grasp[:, 1].min()
    grasp_y = near - np.sign(near - mid_y) * item.inset
    far = grasp[:, 1].min() if near == grasp[:, 1].max() else grasp[:, 1].max()
    band = grasp[np.abs(grasp[:, 1] - grasp_y) < 0.005]
    centre = np.array((0.5 * (band[:, 0].min() + band[:, 0].max()), grasp_y))
    turn = np.array(((np.cos(item.yaw), -np.sin(item.yaw)), (np.sin(item.yaw), np.cos(item.yaw))))
    heading = turn @ (0.0, 1.0)
    new_root = np.asarray(item.through) - item.along * heading
    origin = new_root - turn @ root[:2]
    grip = turn @ centre + origin
    return {
        "pose": (float(origin[0]), float(origin[1]), float(-turned[:, 2].min() + cable_scene.PLACE_GAP + item.raised)),
        "grip_centre_m": [float(grip[0]), float(grip[1]), float(hub.ParallelGripperSpec.pad_half_height + 0.001)],
        "reach_m": float(0.5 * (band[:, 0].max() - band[:, 0].min())),
        "hang_m": float(abs(far - grasp_y)),
        "cable_root_m": [float(c) for c in new_root],
        "yaw_rad": float(item.yaw),
        "grasp_from_base_m": float(np.linalg.norm(grip - arm_gripper.MOUNT[:2])),
    }


def measured_grasps(points, grasp, root, near: float, inset: float) -> list:
    """Where to pinch the grasp end as it lies now, as (centre, hand yaw) candidates, best first.

    The end's long axis in plan, pointed at its cable root; the pads close ``inset`` in from its
    cable end, as :func:`layout` places them. Then the other turn of the wrist (the pads close the
    same either way round), then ``inset`` in from the far end, for when the arm cannot reach the first.
    Yaws are the closest to ``near``.
    """
    flat = points[grasp, :2]
    centre = flat.mean(axis=0)
    axis = np.linalg.eigh(np.cov((flat - centre).T))[1][:, -1]
    if np.dot(points[root, :2].mean(axis=0) - centre, axis) < 0.0:
        axis = -axis
    along = (flat - centre) @ axis
    yaw = float(np.arctan2(-axis[0], axis[1]))
    yaw = near + (yaw - near + np.pi / 2.0) % np.pi - np.pi / 2.0
    other = yaw + np.pi if yaw <= near else yaw - np.pi
    return [
        (centre + (edge - side * inset) * axis, y)
        for side, edge in ((1.0, along.max()), (-1.0, along.min()))
        for y in (yaw, other)
    ]


def check_layout(placed: dict) -> dict:
    """The grasps lie clear, nothing lies under a bin, and every cable crosses another's with the raised one on top.

    ``placed`` holds each object's rest vertices in the bench frame and its grasp-end and cable vertex ids.
    """
    out = {}
    for name, (points, grasp, _) in placed.items():
        for bin_name, (at, inner) in used_bins().items():
            half = np.asarray(inner) / 2.0 + BOX_WALL
            if np.all(np.abs(points[:, :2] - at) < half, axis=1).any():
                raise ValueError(f"{name} lies under {bin_name}'s bin")
            out[f"{name}_to_{bin_name}_bin_m"] = float(np.min(np.max(np.abs(points[:, :2] - at) - half, axis=1)))
        others = np.concatenate([p[:, :2] for n, (p, *_) in placed.items() if n != name] or [np.full((1, 2), 9.0)])
        out[f"{name}_grasp_clear_m"] = float(cKDTree(others).query(points[grasp, :2])[0].min())
        if out[f"{name}_grasp_clear_m"] < 0.015:
            raise ValueError(f"{name}'s grasp end lies {out[f'{name}_grasp_clear_m']:.3f} m from another object")
    return {**out, **check_crossings(placed)}


def check_crossings(placed: dict) -> dict:
    """Every cable crosses another's, with the raised one on top; ``placed`` as in :func:`check_layout`."""
    out, crossed = {}, set()
    for a, b in itertools.combinations(placed, 2):
        lower, upper = sorted((a, b), key=lambda n: OBJECTS[n].raised)
        (l_points, _, l_cable), (u_points, _, u_cable) = placed[lower], placed[upper]
        out[f"{lower}-{upper}_initial_gap_m"] = float(cKDTree(l_points).query(u_points)[0].min())
        # The cables' crossing: the upper cable's vertices over the lower's, within a cable radius in plan.
        distance, index = cKDTree(l_points[l_cable, :2]).query(u_points[u_cable, :2])
        over = distance < 0.002
        if not over.any():
            continue
        crossed.update((a, b))
        above = u_points[u_cable][over, 2] - l_points[l_cable][index[over], 2]
        out[f"{upper}_over_{lower}"] = {
            "at_m": u_points[u_cable][over, :2].mean(axis=0).tolist(),
            "min_above_m": float(above.min()),
        }
        if above.min() <= 0.0:
            raise ValueError(f"the {upper}'s cable starts under the {lower}'s")
    if len(placed) > 1 and crossed != set(placed):
        raise ValueError(f"{sorted(set(placed) - crossed)} cross no other cable")
    return out


def lay_pile(sim, entities: dict, objects: dict) -> dict:
    """Lay every cable on its bent path (:func:`cable_scene.lay`) before the first step, then check they still cross.

    The grasp end stays where :func:`layout` put it; ``objects`` holds each one's ``cable_root_m`` and ``yaw_rad``.
    """
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    laid = {}
    for name, entity in entities.items():
        item, yaw = OBJECTS[name], OBJECTS[name].yaw
        laid[name] = cable_scene.lay(
            sim,
            entity,
            root=surface + np.array((*objects[name]["cable_root_m"], 0.0)),
            direction=(-np.sin(yaw), np.cos(yaw)),
            bends=item.bends,
            keep=np.unique(entity.source_tets[np.isin(entity.tet_part_labels, item.grasp)]),
        )
    q = sim.pipeline.state_0.particle_q.numpy() - surface
    placed = {
        n: (q[e.particles], (), np.unique(e.source_tets[e.tet_part_labels == OBJECTS[n].cable]))
        for n, e in entities.items()
    }
    return {"lay": laid, "crossings": check_crossings(placed)}


def build_scene(
    *, solver: str, iterations: int, mu: float, grip_mu: float, force_limit: float, self_contact: bool, levers
):
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
    scene = rss.Scene(cell=CELL, physics=physics, name="cable_untangle_sort")
    wall_z = (BOX_RIM + BOX_FLOOR) / 2.0
    wall_half = (BOX_RIM - BOX_FLOOR) / 2.0
    for owner, ((bx, by), inner) in used_bins().items():
        half_x, half_y = inner[0] / 2.0, inner[1] / 2.0
        for name, half, at, finish in (
            ("floor", (half_x + BOX_WALL, half_y + BOX_WALL, BOX_FLOOR / 2.0), (bx, by, BOX_FLOOR / 2.0), None),
            (
                "wall_xn",
                (BOX_WALL / 2.0, half_y + BOX_WALL, wall_half),
                (bx - half_x - BOX_WALL / 2.0, by, wall_z),
                GLASS,
            ),
            (
                "wall_xp",
                (BOX_WALL / 2.0, half_y + BOX_WALL, wall_half),
                (bx + half_x + BOX_WALL / 2.0, by, wall_z),
                GLASS,
            ),
            ("wall_yn", (half_x, BOX_WALL / 2.0, wall_half), (bx, by - half_y - BOX_WALL / 2.0, wall_z), GLASS),
            ("wall_yp", (half_x, BOX_WALL / 2.0, wall_half), (bx, by + half_y + BOX_WALL / 2.0, wall_z), GLASS),
        ):
            extra = {"finish": finish} if finish is not None else {}
            scene.add(rss.slab(name=f"{owner}_bin_{name}", half=half, at=at, color=(0.55, 0.60, 0.65), **extra))
    objects, placed, handles, assets = {}, {}, {}, {}
    for name, item in OBJECTS.items():
        asset = cable_scene.with_levers(cable_scene.ASSETS[item.asset], levers)
        vertices, tets, labels = cable_scene.load(asset)
        objects[name] = layout(vertices, tets, labels, asset.quat, item)
        quat = arm_gripper.quat_multiply(arm_gripper.axis_angle_quat((0.0, 0.0, 1.0), item.yaw), asset.quat)
        asset = dataclasses.replace(asset, quat=tuple(float(c) for c in quat))
        placed[name] = (
            arm_gripper._rotate(np.asarray(asset.quat), vertices) + objects[name]["pose"],
            np.unique(tets[np.isin(labels, item.grasp)]),
            np.unique(tets[labels == item.cable]),
        )
        spec = cable_scene.spec(
            asset, vertices, tets, labels, name=name, pose=objects[name]["pose"], settings=settings, mu=mu
        )
        handles[name], assets[name] = scene.add(spec), asset
    pile = check_layout(placed)
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    camera = scene.add(
        rss.CameraSpec(
            name="overview", width=960, height=540, eye=(1.20, -0.85, 1.40), look_at=(0.55, -0.05, 0.88), fov_y=55.0
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    solver = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    geometry = {
        "objects": objects,
        "pile": pile,
        "laid": lay_pile(sim, {n: h.entity for n, h in handles.items()}, objects),
        "bins": {n: {"centre_m": list(at), "inner_m": list(inner)} for n, (at, inner) in used_bins().items()},
        "arm": arm_gripper.check_arm(sim, limit=force_limit),
        "friction": cable_scene.set_friction(sim, solver, settings, mu=mu, grip_mu=grip_mu),
    }
    for name, handle in handles.items():
        objects[name]["rigid_parts"] = {
            f"part{rp.part}": {"nodes": int(len(rp.nodes)), "mass_kg": rp.part_mass} for rp in handle.entity.rigid_parts
        }
    print(json.dumps({"geometry": geometry}), flush=True)
    return sim, {n: h.entity for n, h in handles.items()}, assets, camera, geometry


def arm_route(control, geometry: dict, order: tuple, plans: list, start: int, capacity: int, dt: float, ease=None):
    """Down over the first object, then each one's close, lift, carry, lower, open and withdraw,
    then over and down onto the next, the wrist turning to its yaw.

    ``plans`` holds each pick's carry (x, y) and lowering [m]; the route covers frames ``start`` on.
    ``ease`` = (pick, centre shift, yaw shift, time) eases that pick's re-aimed over-point in from the old one.
    """
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    objects = [geometry["objects"][name] for name in order]
    centres = [arm_gripper.tcp_for_pads(surface + np.asarray(g["grip_centre_m"])) for g in objects]
    reaches = [float(g["reach_m"]) for g in objects]
    yaws = [float(g["yaw_rad"]) for g in objects]
    last = len(order) - 1

    def own(k: int, local: float) -> np.ndarray:
        carry, lower = plans[k]
        return centres[k] + np.array(
            (
                carry[0] * smooth(local, *PHASES["carry"]),
                carry[1] * smooth(local, *PHASES["carry"]),
                LIFT * smooth(local, *PHASES["lift"])
                - lower * smooth(local, *PHASES["lower"])
                + hub.WITHDRAW_Z * smooth(local, *PHASES["withdraw"]),
            )
        )

    def split(time: float) -> tuple[int, float]:
        local = time + start * dt - SETTLE
        k = min(max(int(local // BLOCK), 0), last)
        return k, local - k * BLOCK

    def tcp_at(time: float) -> np.ndarray:
        k, local = split(time)
        at = own(k, local)
        blend = 0.0 if ease is None or k != ease[0] - 1 else 1.0 - smooth(time + start * dt, ease[3], ease[3] + EASE)
        at[2] += APPROACH_Z * (1.0 - smooth(local, *np.subtract(FIRST_DESCEND, SETTLE)))
        if k < last:
            over = centres[k + 1] + np.array((0.0, 0.0, APPROACH_Z))
            at = at + (over - own(k, BLOCK)) * smooth(local, *RETURN)
            if blend:
                at[:2] += blend * ease[1]
            at[2] -= APPROACH_Z * smooth(local, *DESCEND)
        return at

    def quat_at(time: float) -> np.ndarray:
        k, local = split(time)
        yaw = yaws[k] + ((yaws[k + 1] - yaws[k]) * smooth(local, *RETURN) if k < last else 0.0)
        if ease is not None and k == ease[0] - 1:
            yaw += ease[2] * (1.0 - smooth(time + start * dt, ease[3], ease[3] + EASE))
        return arm_gripper.quat_multiply(arm_gripper.axis_angle_quat((0.0, 0.0, 1.0), yaw), arm_gripper.GRASP_QUAT)

    def half_gap_at(time: float) -> float:
        k, local = split(time)
        opened = hub.OPENED_HALF_GAP
        face = opened + (max(reaches[k] - hub.GRIP_SQUEEZE, 0.002) - opened) * smooth(local, *PHASES["close"])
        return face + (opened - face) * smooth(local, *PHASES["open"])

    return control.plan(quat=quat_at, capacity=capacity, dt=dt, tcp_at=tcp_at, half_gap_at=half_gap_at)


def lowering(geometry: dict, name: str) -> float:
    """How far the hand comes down: the hanging housing's bottom ``OVER_FLOOR`` over the bin's floor."""
    g = geometry["objects"][name]
    return LIFT + g["grip_centre_m"][2] - (BOX_FLOOR + hub.OVER_FLOOR + g["hang_m"])


def inside_bin(points: np.ndarray, name: str, top: float = BOX_RIM) -> np.ndarray:
    """Which bench-frame points are inside ``name``'s bin, below ``top`` over the bench [mask]."""
    at, inner = BINS[OBJECTS[name].bin]
    return (
        (np.abs(points[:, 0] - at[0]) <= inner[0] / 2.0)
        & (np.abs(points[:, 1] - at[1]) <= inner[1] / 2.0)
        & (points[:, 2] <= top)
    )


def run(
    out: Path,
    *,
    steps: int,
    render_every: int,
    order: tuple,
    solver: str,
    iterations: int,
    mu: float,
    grip_mu: float,
    force_limit: float,
    self_contact: bool,
    levers=None,
    config: dict | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim, entities, assets, camera, geometry = build_scene(
        solver=solver,
        iterations=iterations,
        mu=mu,
        grip_mu=grip_mu,
        force_limit=force_limit,
        self_contact=self_contact,
        levers=levers,
    )
    names = list(OBJECTS)
    pairs = list(itertools.combinations(names, 2))
    dt = 1.0 / CONTROL_HZ
    total = max(steps, round(duration() * CONTROL_HZ))
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    # Planned before anything is measured: straight carries to each bin's centre.
    plans = [
        (np.subtract(BINS[OBJECTS[n].bin][0], geometry["objects"][n]["grip_centre_m"][:2]), lowering(geometry, n))
        for n in order
    ]
    control = arm_gripper.ArmControl(sim)
    route, origin = arm_route(control, geometry, order, plans, 0, total, dt), 0
    errors = {"planned": route.worst_position_error_m}
    print(json.dumps({"arm_path_error_m": errors["planned"], "duration_s": duration(), "order": order}), flush=True)
    if route.worst_position_error_m > REACH_OK:
        raise RuntimeError(f"the arm misses the path by {route.worst_position_error_m:.3f} m")
    control.start(route)
    replan_at = {round((SETTLE + k * BLOCK + PHASES["carry"][0]) * CONTROL_HZ) - 1: k for k in range(len(order))}
    # Each pick is aimed where its grasp lies before the hand comes down: the first once the pile has
    # settled, the later ones as the hand leaves the one before and again just before it descends.
    regrasp_at = {round(FIRST_DESCEND[0] * CONTROL_HZ) - 1: 0}
    for k in range(1, len(order)):
        regrasp_at.update({round((SETTLE + (k - 1) * BLOCK + t) * CONTROL_HZ) - 1: k for t in REAIM})

    # The grasp end (the housing, or a housing-less cable's plug end) is tracked by its particles' centroid:
    # the body's origin is the asset's. "housing" below means that grasp end.
    parts, housing_before = {}, {}
    for name, entity in entities.items():
        tets, labels, item = entity.source_tets, entity.tet_part_labels, OBJECTS[name]
        soft = np.isin(labels, cable_scene.soft_labels(assets[name], labels))
        q0 = sim.pipeline.state_0.particle_q.numpy()[entity.particles]
        held = np.isin(labels, item.grasp)
        parts[name] = {
            "cable": np.unique(tets[labels == item.cable]),
            "housing": np.unique(tets[held]),
            "root": np.intersect1d(tets[held], tets[labels == item.cable]),
            "dangling": np.setdiff1d(np.unique(tets[~held]), np.unique(tets[held])),
            "soft": soft,
            "rest_inv": np.linalg.inv((q0[tets[:, 1:]] - q0[tets[:, :1]])[soft]),
            "trajectory": Trajectory(sim, entity),
        }
        housing_before[name] = q0[parts[name]["housing"]].mean(axis=0) - surface
    contacts = sim.pipeline.contacts
    finger_shapes = arm_gripper.finger_shapes(sim)
    rows, videos, video_error, aborted = [], None, None, None
    if render_every:
        from robosimstudio.core.render.video import VideoSet

        videos = VideoSet(str(out), fps=60, holds=max(1, 60 * render_every // CONTROL_HZ))
        videos.__enter__()
    started = time.perf_counter()
    try:
        for frame in range(steps):
            control.follow(route, frame - origin)
            sim.run(1).sync()
            state = sim.pipeline.state_0
            q, body_q = state.particle_q.numpy(), state.body_q.numpy()
            local = {n: q[e.particles] - surface for n, e in entities.items()}
            if frame in regrasp_at:
                k = regrasp_at[frame]
                name = order[k]
                g, item = geometry["objects"][name], OBJECTS[name]
                old_grip, old_yaw = np.array(g["grip_centre_m"][:2]), g["yaw_rad"]
                candidates = measured_grasps(
                    local[name], parts[name]["housing"], parts[name]["root"], old_yaw, item.inset
                )
                for tried, (grip, yaw) in enumerate(candidates):
                    g["grip_centre_m"][:2], g["yaw_rad"] = grip.tolist(), yaw
                    plans[k] = (np.asarray(BINS[item.bin][0]) - grip, lowering(geometry, name))
                    ease = (k, old_grip - grip, old_yaw - yaw, (frame + 1) * dt) if k else None
                    route = arm_route(control, geometry, order, plans, frame + 1, total - frame - 1, dt, ease)
                    origin = frame + 1
                    errors[f"{name}_grasp"] = route.worst_position_error_m
                    regrasp = {
                        "object": name,
                        "grip_m": grip.round(4).tolist(),
                        "yaw_rad": round(yaw, 4),
                        "moved_m": float(np.linalg.norm(grip - old_grip)),
                        "candidate": tried,
                    }
                    print(json.dumps({**regrasp, "arm_path_error_m": route.worst_position_error_m}), flush=True)
                    if route.worst_position_error_m <= REACH_OK:
                        break
                else:
                    aborted = {**regrasp, "reason": f"unreachable: misses by {route.worst_position_error_m:.3f} m"}
                    print(json.dumps({"aborted": aborted}), flush=True)
                    break
            if frame in replan_at:
                # Carry what hangs over the bin's centre.
                k = replan_at[frame]
                name = order[k]
                hanging = local[name][parts[name]["dangling"], :2].mean(axis=0)
                plans[k] = (np.asarray(BINS[OBJECTS[name].bin][0]) - hanging, lowering(geometry, name))
                route, origin = arm_route(control, geometry, order, plans, frame + 1, total - frame - 1, dt), frame + 1
                errors[name] = route.worst_position_error_m
                print(
                    json.dumps(
                        {"object": name, "carry_m": np.round(plans[k][0], 4).tolist(), "arm_path_error_m": errors[name]}
                    ),
                    flush=True,
                )
            trees = {n: cKDTree(local[n]) for n in names}
            count = int(contacts.soft_contact_count.numpy()[0])
            shapes = contacts.soft_contact_shape.numpy()[: min(count, contacts.soft_contact_max)]
            row = {
                "frame": frame + 1,
                "time_s": (frame + 1) * dt,
                "finger_contacts": int(np.count_nonzero(np.isin(shapes, finger_shapes))),
                "aperture_m": control.aperture(),
                "gap_m": {f"{a}-{b}": float(trees[b].query(local[a])[0].min()) for a, b in pairs},
                "touching_pairs": {f"{a}-{b}": int(trees[a].count_neighbors(trees[b], TOUCH)) for a, b in pairs},
            }
            for name, entity in entities.items():
                p, tets = local[name], entity.source_tets
                edges = p[tets[:, 1:]] - p[tets[:, :1]]
                # The cable's strain: each tet's largest principal stretch; its 99th percentile skips slivers.
                stretch = np.linalg.svd(parts[name]["rest_inv"] @ edges[parts[name]["soft"]], compute_uv=False)
                housing = p[parts[name]["housing"]].mean(axis=0)
                row[name] = {
                    "minimum_z_m": float(p[:, 2].min()),
                    "cable_inside": float(inside_bin(p[parts[name]["cable"]], name).mean()),
                    "housing_m": housing.tolist(),
                    "housing_speed_m_s": float(np.linalg.norm(housing - housing_before[name]) / dt),
                    "inverted_tetrahedra": int(np.count_nonzero(np.linalg.det(edges) <= 0)),
                    "cable_inverted": int(np.count_nonzero(np.linalg.det(edges[parts[name]["soft"]]) <= 0)),
                    "cable_stretch_p99": float(np.percentile(stretch.max(axis=1), 99.0)),
                    **rigid_part_body.frame_record(entity, p + surface, body_q),
                }
                housing_before[name] = housing
                parts[name]["trajectory"].add(row["time_s"], p + surface, body_q)
            rows.append(row)
            if videos is not None and frame % render_every == 0:
                videos.append_frames({camera.name: sim.render(camera)})
            if frame % 30 == 0 or (frame < 30 and frame % 5 == 4) or frame == steps - 1:
                brief = {n: {k: v for k, v in row[n].items() if not k.startswith("part")} for n in names}
                print(json.dumps({**{k: v for k, v in row.items() if k not in names}, **brief}), flush=True)
    finally:
        if videos is not None:
            try:
                videos.__exit__(None, None, None)
            except Exception as exc:
                video_error = repr(exc)
    elapsed = time.perf_counter() - started
    for name in names:
        parts[name]["trajectory"].save(out / f"trajectory_{name}.npz")
    (out / "diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")

    final, tail, settled = rows[-1], rows[-round(0.5 * CONTROL_HZ) :], rows[min(29, len(rows) - 1)]
    per_object = {}
    for k, name in enumerate(order):
        waiting = order[k + 1 :]
        begin = SETTLE + k * BLOCK
        block = [r for r in rows if begin + PHASES["close"][0] <= r["time_s"] <= begin + PHASES["withdraw"][1]]
        lift = [r for r in block if begin + PHASES["lift"][0] <= r["time_s"] <= begin + PHASES["carry"][1]]
        # The untangle: how far the grasp ends still waiting move while this one is taken.
        drag = (
            max(
                float(np.linalg.norm(np.subtract(r[o]["housing_m"], block[0][o]["housing_m"])))
                for r in block
                for o in waiting
            )
            if block and waiting
            else None
        )
        # When this cable parts from the others' for good during the lift and carry.
        mine = [f"{a}-{b}" for a, b in pairs if name in (a, b)]
        touching = [sum(r["touching_pairs"][p] for p in mine) for r in lift]
        apart = [i for i in range(len(lift)) if not any(touching[i:])]
        housing = np.asarray(final[name]["housing_m"])
        per_object[name] = {
            "pick": k,
            "cable_inside_final": final[name]["cable_inside"],
            "housing_inside_final": bool(inside_bin(housing[None], name, BOX_RIM + 0.02)[0]),
            "housing_final_m": housing.tolist(),
            "other_housing_drag_m": drag,
            "touching_at_lift_start": touching[0] if touching else None,
            "separated_s": lift[apart[0]]["time_s"] if apart else None,
            "lifted_clearance_m": min(
                (r[name]["minimum_z_m"] for r in lift if r["time_s"] >= begin + PHASES["lift"][1]), default=None
            ),
            "cable_stretch_p99_max": max(r[name]["cable_stretch_p99"] for r in rows),
            "max_inverted_tetrahedra": max(r[name]["inverted_tetrahedra"] for r in rows),
            "max_cable_inverted": max(r[name]["cable_inverted"] for r in rows),
            "rigid_skin_offset_max_m": max(
                v["skin_stretch_max_m"] for r in rows for key, v in r[name].items() if key.endswith("_body")
            ),
        }
    drags = [o["other_housing_drag_m"] or 0.0 for o in per_object.values()]
    result = {
        "geometry": geometry,
        "order": list(order),
        "objects": per_object,
        "released": max(r["finger_contacts"] for r in tail) == 0,
        "aborted": aborted,
        "max_inverted_tetrahedra": max(o["max_inverted_tetrahedra"] for o in per_object.values()),
        "max_cable_inverted": max(o["max_cable_inverted"] for o in per_object.values()),
        "settle": {
            "frame": settled["frame"],
            "speed_m_s": {n: settled[n]["housing_speed_m_s"] for n in names},
            "gap_m": settled["gap_m"],
            "cable_inverted": {n: max(r[n]["cable_inverted"] for r in rows[: settled["frame"]]) for n in names},
        },
        "gap_final_m": final["gap_m"],
        "gap_min_m": {p: min(r["gap_m"][p] for r in rows) for p in final["gap_m"]},
        "touching_pairs_initial": rows[0]["touching_pairs"],
        "touching_pairs_final": final["touching_pairs"],
        "arm_path_error_m": errors,
        "steps": steps,
        "video_error": video_error,
        "mean_frame_ms": 1000.0 * elapsed / max(1, len(rows)),
        "mean_step_ms": 1000.0 * elapsed / max(1, len(rows) * SUBSTEPS),
        "elapsed_s": elapsed,
        "config": config,
    }
    result["success"] = bool(
        aborted is None
        and all(
            o["housing_inside_final"] and (HOUSING in OBJECTS[n].grasp or o["cable_inside_final"] >= CABLE_OK)
            for n, o in per_object.items()
        )
        # A lone tet flipping for a frame under a three-way cable contact is not a broken cable.
        and result["max_cable_inverted"] <= 2
        and result["released"]
    )
    # Mean cable fraction in its own bin, less each pick's drag of the others' grasp ends (DRAG_SCALE costs 1).
    result["score"] = float(
        np.mean([o["cable_inside_final"] for o in per_object.values()])
        - np.mean([min(1.0, d / DRAG_SCALE) for d in drags])
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("geometry", "config")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT / "run")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--objects", default="hub,charger,splitter", help=f"from {', '.join(OBJECTS)}; coil is costly")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--order", help="picking order (default: bottom of the pile first, see OBJECTS)")
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default="chebyshev")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--mu", type=float, default=cable_scene.OBJECT_MU)
    parser.add_argument("--grip-mu", type=float, default=2.0)
    parser.add_argument("--force-limit", type=float, default=30.0)
    parser.add_argument("--no-self-contact", dest="self_contact", action="store_false")
    cable_scene.add_lever_arguments(parser)
    args = parser.parse_args()
    chosen = args.objects.split(",")
    if not set(chosen) <= set(OBJECTS):
        parser.error(f"--objects must be from {', '.join(OBJECTS)}")
    for name in set(OBJECTS) - set(chosen):
        del OBJECTS[name]
    order = tuple(args.order.split(",")) if args.order else tuple(OBJECTS)
    if sorted(order) != sorted(OBJECTS):
        parser.error(f"--order must name each of {', '.join(OBJECTS)} once")
    settings = dataclasses.replace(REFERENCE, substeps=SUBSTEPS)
    run(
        args.out,
        steps=args.steps or round(duration() * CONTROL_HZ),
        render_every=args.render_every,
        order=order,
        solver=args.solver,
        iterations=args.iterations,
        mu=args.mu,
        grip_mu=args.grip_mu,
        force_limit=args.force_limit,
        self_contact=args.self_contact,
        levers=args,
        config={
            **cable_scene.config(args, cable_scene.ASSETS["usb_hub_gen"], settings),
            "asset_levers": {
                name: dataclasses.asdict(cable_scene.with_levers(cable_scene.ASSETS[o.asset], args))
                for name, o in OBJECTS.items()
            },
            "objects": {name: [*o[:4], list(o.through), *o[5:]] for name, o in OBJECTS.items()},
        },
    )


if __name__ == "__main__":
    main()
