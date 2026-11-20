# SPDX-License-Identifier: Apache-2.0
"""Cabled objects for the cable scenes: rigid housings and plugs joined by a soft FEM cable.

:data:`ASSETS` holds what each one needs to lie flat on the bench and the material levers tuned for it;
:func:`lay` bends a built cable along a path on the bench. Loading, the spec and the episode loop are
:mod:`hetero_scene`'s.
"""

from __future__ import annotations

import numpy as np
from hetero_scene import TURN_Z, HeteroAsset

ASSETS = {
    # The scanned hub; its tuning is usb_hub_box.py's original.
    "usb_hub": HeteroAsset("usb_hub", (0.5, 0.5, 0.5, 0.5), 2.0e7, 3.0e4, 0.002, rigid_groups=(0, (2, 3))),
    # 4e7 Pa: at 2e7 the 4-4.5 mm cables hung like rope; at 1e8 (EI 1.3e-3 N*m^2, 5-10x a real
    # cable) the cable propped the charger brick up 8 deg. 4e7 lies flat and still holds a bend (q9, user).
    "usb_hub_gen": HeteroAsset("usb_hub_gen", TURN_Z, 4.0e7, 6.0e4, 0.004),
    "charger": HeteroAsset("charger", TURN_Z, 4.0e7, 6.0e4, 0.004),
    "power_strip": HeteroAsset("power_strip", TURN_Z, 2.0e7, 3.0e4, None),
    "y_splitter_cable": HeteroAsset("y_splitter_cable", TURN_Z, 4.0e7, 6.0e4, 0.004),
    "coiled_usb_cable": HeteroAsset("coiled_usb_cable", TURN_Z, 4.0e7, 6.0e4, 0.004),
}


def bend_path(s, bends, heading: float = 0.0):
    """Points and headings along a planar path of ``bends`` from the origin, at arclengths ``s`` [m].

    ``bends`` is ``((length, radius), ...)``: radius 0 runs straight, a positive one turns left
    (counter-clockwise from above), a negative one right; past the last bend the path runs straight.
    ``heading`` is the start's direction [rad]. Returns the (N, 2) points and (N,) headings.
    """
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    points, headings = np.zeros((len(s), 2)), np.full(len(s), float(heading))
    start, origin, angle = 0.0, np.zeros(2), float(heading)

    def along(t, radius, origin, angle):
        if radius == 0.0:
            return origin + t[:, None] * (np.cos(angle), np.sin(angle)), np.full(len(t), angle)
        theta = angle + t / radius
        return origin + radius * np.stack((np.sin(theta) - np.sin(angle), np.cos(angle) - np.cos(theta)), 1), theta

    for length, radius in (*bends, (np.inf, 0.0)):
        rows = s >= start
        points[rows], headings[rows] = along(np.minimum(s[rows] - start, length), radius, origin, angle)
        if np.isfinite(length):
            end, turned = along(np.array((length,)), radius, origin, angle)
            origin, angle, start = end[0], float(turned[0]), start + length
    return points, headings


def lay(sim, entity, *, root, direction, bends, keep) -> dict:
    """Lay the built (straight) cable along a planar path on the bench, before the first step.

    The rest shape stays straight; only the initial state moves. Each particle past ``root`` [m,
    world] along the horizontal ``direction`` goes to the point of :func:`bend_path` at its
    arclength, its sideways offset turned with the path and its height kept. A rigid part moves
    rigidly with the path's frame at its node nearest the root, its body and free joint with it;
    the ``keep`` nodes (local ids: the housing) and parts made of them stay put. Both state
    buffers are written, as :meth:`arm_gripper.ArmControl.start` does, and velocities zeroed.
    """
    import newton

    model, pipeline = sim.model, sim.pipeline
    particles = np.asarray(entity.particles)
    root = np.asarray(root, dtype=np.float64)
    u = np.array((direction[0], direction[1], 0.0), dtype=np.float64)
    u /= np.linalg.norm(u)
    v = np.array((-u[1], u[0], 0.0))
    heading = float(np.arctan2(u[1], u[0]))
    q = pipeline.state_0.particle_q.numpy()
    x = q[particles].astype(np.float64)
    d = x - root
    s, n = d @ u, d @ v
    points, headings = bend_path(s, bends, heading)
    new = x.copy()
    move = s > 0.0
    move[np.asarray(keep)] = False
    new[move, 0] = root[0] + points[move, 0] - n[move] * np.sin(headings[move])
    new[move, 1] = root[1] + points[move, 1] + n[move] * np.cos(headings[move])
    body_q, joint_q = pipeline.state_0.body_q.numpy(), pipeline.state_0.joint_q.numpy()
    body_qd = pipeline.state_0.body_qd.numpy()
    joint_qd = pipeline.state_0.joint_qd.numpy()
    starts, dof_starts = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    children, kinds = model.joint_child.numpy(), model.joint_type.numpy()
    moved = {}
    for rp in entity.rigid_parts:
        if np.isin(rp.nodes, keep).all():
            continue
        near = float(s[rp.nodes].min())
        point, angle = bend_path([near], bends, heading)
        turn = float(angle[0]) - heading
        c, w = np.cos(turn), np.sin(turn)
        rotation = np.array(((c, -w, 0.0), (w, c, 0.0), (0.0, 0.0, 1.0)))
        centre = root + near * u
        target = root + np.array((point[0, 0], point[0, 1], 0.0))
        new[rp.nodes] = (x[rp.nodes] - centre) @ rotation.T + target
        spin = np.array((0.0, 0.0, np.sin(0.5 * turn), np.cos(0.5 * turn)))
        pose = np.concatenate(
            (rotation @ (body_q[rp.body, :3] - centre) + target, _quat_mul(spin, body_q[rp.body, 3:]))
        )
        moved[rp.part] = {"translation_m": (pose[:3] - body_q[rp.body, :3]).tolist(), "turn_rad": turn}
        body_q[rp.body], body_qd[rp.body] = pose, 0.0
        for j in np.flatnonzero((children == rp.body) & (kinds == int(newton.JointType.FREE))):
            joint_q[starts[j] : starts[j] + 7] = pose
            joint_qd[dof_starts[j] : dof_starts[j] + 6] = 0.0
    q[particles] = new
    qd = pipeline.state_0.particle_qd.numpy()
    qd[particles] = 0.0
    for state in (pipeline.state_0, pipeline.state_1):
        state.particle_q.assign(q)
        state.particle_qd.assign(qd)
        state.body_q.assign(body_q)
        state.body_qd.assign(body_qd)
        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)
    # The graph is captured on the first step; any captured earlier is dropped so it re-reads nothing stale.
    pipeline.invalidate_graph()
    return {"moved_particles": int(move.sum()), "rigid_parts": moved}


def _quat_mul(a, b) -> np.ndarray:
    """``a * b`` for xyzw quaternions."""
    (ax, ay, az, aw), (bx, by, bz, bw) = a, b
    return np.array(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )
