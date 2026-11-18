# SPDX-License-Identifier: Apache-2.0
"""Hand tools for the tool scenes: a rigid handle and backing carrying a soft working part.

Tools are labelled tet assets like the cable ones (:mod:`cable_scene`): the rigid groups become rigid
bodies, every other label is soft. This adds what the tool tasks share: the tool table
(:data:`TOOLS`), a grasp on the handle and a hand that tilts, free rigid crumbs and their scoring,
a stain map the soft part rubs off, and counter, pan, basin, whiteboard and shelf scenery.

Free rigid bodies (crumbs) land in the "rigid" island, which :func:`clawhauser_scene.arrangement`
gives to the same VBD solver as the tool's particles: they collide with the soft tool and the bench.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import arm_gripper
import cable_scene
import numpy as np
from clawhauser_scene import REFERENCE, arrangement, build_sim
from scenery import STEEL, revolved_obj
from scipy.spatial.transform import Rotation

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish
from robosimstudio.specs import AssetSpec, StaticSpec

SUBSTEPS = 40
CONTROL_HZ = REFERENCE.control_hz


@dataclasses.dataclass(frozen=True)
class ToolAsset(cable_scene.CableAsset):
    """A tool: ``cable_young`` / ``cable_damping`` are the levers of every soft label, ``soft_young`` per label."""

    handle: int = 0
    """The grip's rigid label."""
    axis: tuple = (1.0, 0.0, 0.0)
    """Handle's long axis in the asset frame, pointing to the working end."""
    grip_frac: float = 0.5
    """Where the pads close, as a fraction of the handle's length from its far end."""
    soft_young: tuple = ()
    """``((label, Young's modulus), ...)`` applied after the common lever."""
    pitch_deg: float = 0.0
    """Pitch about y the tool starts with: a wand as built has its handle rising 26 degrees and comes to rest with its handle on the bench, about this far over."""
    pad: int = -1
    """The rigid label whose lowest node measures compression (the backing); -1 uses the whole rigid body."""

    def overrides(self) -> tuple:
        return tuple((label, "youngs_modulus_pa", e) for label, e in self.soft_young)


def _tool(folder, cable_young, cable_damping, pitch_deg: float = 0.0, **kwargs) -> ToolAsset:
    quat = tuple(arm_gripper.axis_angle_quat((0.0, 1.0, 0.0), np.radians(pitch_deg)))
    return ToolAsset(folder, quat, cable_young, cable_damping, None, pitch_deg=pitch_deg, **kwargs)


TOOLS = {
    # Source Es: sponge 8e4 / scouring face 3e5, felt 1.2e5, cup 5e6, silicone head 1.5e6; the common lever replaces them.
    "sponge_wand": _tool("sponge_wand", 8.0e4, 50.0, soft_young=((3, 3.0e5),), grip_frac=0.45, pad=1, pitch_deg=-23.0),
    "gap_sponge_wand": _tool(
        "gap_sponge_wand", 8.0e4, 50.0, soft_young=((3, 3.0e5),), grip_frac=0.45, pad=1, pitch_deg=-23.0
    ),
    "whiteboard_eraser": _tool("whiteboard_eraser", 1.2e5, 50.0, pad=1),
    "toilet_plunger": _tool("toilet_plunger_generated", 5.0e6, 500.0, axis=(0.0, 0.0, 1.0), grip_frac=0.8, pad=-1),
    "silicone_spatula": _tool("silicone_spatula", 1.5e6, 200.0, grip_frac=0.4, pad=-1),
    "corded_soldering_iron": _tool("corded_soldering_iron", 4.0e7, 6.0e4, handle=1, grip_frac=0.5, pad=-1),
}


def spec(asset: ToolAsset, vertices, tets, labels, *, name: str, pose, settings, mu: float):
    return cable_scene.spec(
        asset, vertices, tets, labels, name=name, pose=pose, settings=settings, mu=mu, overrides=asset.overrides()
    )


def grasp_handle(asset: ToolAsset, vertices, tets, labels, *, pose=(0.0, 0.0, 0.0), axis=None) -> dict:
    """Where the pads close on the handle [m, asset frame shifted by ``pose``] and how.

    ``point`` is the handle's section centre at ``grip_frac`` along its axis, ``half_width`` its half
    extent along the closing axis ``across`` (horizontal, across the handle), ``quat`` the hand's
    orientation (xyzw) and ``yaw`` that closing axis's turn from world x. A tool that has settled
    tilted passes its current ``vertices`` and the handle's current ``axis``.
    """
    if axis is None:
        vertices = arm_gripper._rotate(np.asarray(asset.quat), vertices)
        axis = arm_gripper._rotate(np.asarray(asset.quat), np.asarray(asset.axis)[None])[0]
    axis = np.asarray(axis, dtype=np.float64)
    handle = vertices[np.unique(tets[labels == asset.handle])]
    along = handle @ axis
    at = along.min() + asset.grip_frac * np.ptp(along)
    # The section's nodes, widening the window while the mesh is too sparse to have any (a spatula's long tets).
    width = 0.004
    while (np.abs(along - at) < width).sum() < 12:
        width *= 1.5
    band = handle[np.abs(along - at) < width]
    across = np.cross((0.0, 0.0, 1.0), axis)
    across = np.array((1.0, 0.0, 0.0)) if np.linalg.norm(across) < 1e-6 else across / np.linalg.norm(across)
    low, high = band.min(axis=0), band.max(axis=0)
    centre = 0.5 * (low + high)
    point = centre + axis * (at - centre @ axis) + np.asarray(pose)
    yaw = float(np.arctan2(across[1], across[0]))
    quat = (Rotation.from_rotvec((0.0, 0.0, yaw)) * Rotation.from_quat(arm_gripper.GRASP_QUAT)).as_quat()
    return {
        "point_m": point.tolist(),
        "half_width_m": float(0.5 * np.ptp(band @ across)),
        "across": across.tolist(),
        "yaw_rad": yaw,
        "quat": quat.tolist(),
        "height_m": float(np.ptp(band[:, 2])),
    }


def tilt_quat(grasp: dict, angle_at):
    """The hand's ``quat(t)``: its grasp orientation turned ``angle_at(t)`` [rad] about the closing axis.

    The pads pivot about the axis they close along, so the tool pitches about its grip, as in the plug scene.
    """
    base, across = np.asarray(grasp["quat"]), np.asarray(grasp["across"])
    return lambda t: arm_gripper.quat_multiply(arm_gripper.axis_angle_quat(across, float(angle_at(t))), base)


# --- Free rigid crumbs ---------------------------------------------------------------------------


def crumbs(
    scene,
    *,
    region,
    n: int,
    size=(0.003, 0.006),
    seed: int = 0,
    density: float = 1500.0,
    z0=0.0,
    prefix: str = "crumb",
    spacing: float = 0.0,
    spheres: bool = True,
) -> list[str]:
    """Add ``n`` free spheres and cubes of ``size`` [m] across, resting over ``region`` ``(x0, x1, y0, y1)``.

    The first half are spheres, the rest cubes; ``density`` [kg/m^3] sets their masses; ``z0`` is the surface they rest on, ``prefix`` their names (``z0`` may be a function of x; ``spheres=False`` makes every crumb a cube, which stays on a slope); ``spacing`` [m] keeps centres apart (crumbs that spawn overlapping fling each other off).
    """
    rng = np.random.default_rng(seed)
    names, placed = [], []
    for i in range(n):
        edge = float(rng.uniform(*size))
        for _ in range(200):
            xy = (float(rng.uniform(*region[:2])), float(rng.uniform(*region[2:])))
            if all(np.hypot(xy[0] - q[0], xy[1] - q[1]) >= spacing for q in placed):
                break
        placed.append(xy)
        at = (*xy, (z0(xy[0]) if callable(z0) else z0) + 0.5 * edge + 0.0005)
        name = f"{prefix}{i}"
        shade = float(rng.uniform(0.25, 0.45))
        kwargs = {"at": at, "color": (shade + 0.15, shade, shade * 0.6)}
        if spheres and i < n // 2:
            scene.add(
                rss.sphere(name=name, radius=0.5 * edge, mass=density * 4 / 3 * np.pi * (0.5 * edge) ** 3, **kwargs)
            )
        else:
            scene.add(rss.box(name=name, half=(0.5 * edge,) * 3, mass=density * edge**3, **kwargs))
        names.append(name)
    return names


def crumb_positions(sim, names) -> np.ndarray:
    """The crumbs' world positions [m, (n, 3)]."""
    body_q = sim.pipeline.state_0.body_q.numpy()
    return body_q[[int(sim.handles.objects[name].body) for name in names], :3].astype(np.float64)


def crumb_score(sim, names, *, region=None, below: float | None = None) -> dict:
    """The fraction of crumbs outside ``region`` (world ``(x0, x1, y0, y1)``) or lower than ``below`` (world z)."""
    p = crumb_positions(sim, names)
    out = np.zeros(len(p), dtype=bool)
    if region is not None:
        out |= (p[:, 0] < region[0]) | (p[:, 0] > region[1]) | (p[:, 1] < region[2]) | (p[:, 1] > region[3])
    if below is not None:
        out |= p[:, 2] < below
    return {"cleared_fraction": float(out.mean()) if len(p) else 0.0, "cleared": int(out.sum()), "crumbs": len(p)}


# --- Stain map -------------------------------------------------------------------------------------

STAIN_COLOR = (0.08, 0.08, 0.10)


class StainMap:
    """Marks on a horizontal surface that the soft part rubs off, modelled on ``cloth.ForceGatedWipe``.

    A mark is erased on a control step in which a soft particle within ``reach`` of it (horizontally)
    presses into the surface deeper than ``force_min / contact_ke`` (the penalty contact's force on
    that particle, so the gate is the force), and the part moved at least ``rub`` along the surface
    since the last step, so pressing or lifting alone erases nothing. The marks are thin dark visual
    tiles (:meth:`specs`); :meth:`hide` repaints an erased one as the surface.
    """

    def __init__(
        self,
        *,
        z: float,
        radius: float,
        contact_ke: float,
        centre=None,
        size=None,
        grid=None,
        points=None,
        tile=0.006,
        force_min=0.05,
        reach=None,
        half_z=0.0003,
    ):
        """A ``grid`` of marks over ``size`` about ``centre``, or the given ``points`` (n, 2); ``half_z`` thins the tiles."""
        if points is None:
            xs = centre[0] + np.linspace(-0.5, 0.5, grid[0]) * (size[0] - tile)
            ys = centre[1] + np.linspace(-0.5, 0.5, grid[1]) * (size[1] - tile)
            points = np.array([(x, y) for x in xs for y in ys])
            reach = 0.6 * tile + 0.5 * np.ptp(xs) / max(1, grid[0] - 1)
        self.points = np.asarray(points, dtype=np.float64)
        self.z, self.tile, self.reach, self.half_z = z, tile, 0.6 * tile if reach is None else reach, half_z
        self.depth = radius - force_min / contact_ke
        self.erased = np.zeros(len(self.points), dtype=bool)
        self.last = None

    def specs(self) -> list:
        """One dark thin static tile per mark, named ``stain<i>``."""
        return [
            rss.slab(
                name=f"stain{i}",
                half=(0.5 * self.tile, 0.5 * self.tile, self.half_z),
                at=(x, y, self.z + self.half_z),
                collide=False,
                color=STAIN_COLOR,
            )
            for i, (x, y) in enumerate(self.points)
        ]

    def update(self, soft: np.ndarray, rub: float = 0.0002) -> np.ndarray:
        """Erase the marks the soft particles ``soft`` (bench frame, (n, 3)) press and rub; returns the new ones."""
        pressed = soft[soft[:, 2] - self.z < self.depth, :2]
        moved = self.last is None or np.linalg.norm(pressed.mean(axis=0) - self.last) >= rub if len(pressed) else False
        self.last = pressed.mean(axis=0) if len(pressed) else None
        if not moved:
            return np.zeros(0, dtype=int)
        near = (np.linalg.norm(self.points[:, None] - pressed[None], axis=2) < self.reach).any(axis=1)
        new = np.flatnonzero(near & ~self.erased)
        self.erased[new] = True
        return new

    def fraction_erased(self) -> float:
        return float(self.erased.mean())

    def make_live(self, sim) -> None:
        """Declare the tiles' colour live, before the first render, so the renderer re-reads it each frame."""
        shapes = np.concatenate([sim.handles.statics[f"stain{i}"].visual_shapes for i in range(len(self.points))])
        sim.handles.live_shapes = np.union1d(sim.handles.live_shapes, shapes).astype(np.int32)

    def hide(self, sim, indices, color=(0.95, 0.95, 0.93)) -> None:
        """Paint the erased tiles the surface's colour (the renderer reads ``shape_color`` live)."""
        colors = sim.model.shape_color.numpy()
        for i in indices:
            colors[sim.handles.statics[f"stain{i}"].visual_shapes] = color
        sim.model.shape_color.assign(colors)


# --- Scenery -------------------------------------------------------------------------------------

COUNTER_COLOR = (0.62, 0.60, 0.55)
RIDGE_COLOR = (0.42, 0.45, 0.50)
SINK_STEEL = Finish(roughness=0.25, metallic=0.9)


def counter_with_gap(*, at, size, thickness: float, gap: float) -> list:
    """A counter ``size`` ``(x, y)`` centred on ``at`` and ``thickness`` thick, with a ``gap`` wide slot along x at its centre.

    The slot opens onto the bench below, the sink the crumbs fall into.
    """
    half_y = 0.5 * (size[1] - gap) / 2.0
    offset = 0.25 * (size[1] + gap)
    return [
        rss.slab(
            name=f"counter_{side}",
            half=(0.5 * size[0], half_y, 0.5 * thickness),
            at=(at[0], at[1] + sign * offset, 0.5 * thickness),
            color=COUNTER_COLOR,
        )
        for side, sign in (("near", -1.0), ("far", 1.0))
    ]


def ridged_plate(
    *,
    at,
    length: float,
    n: int,
    ridge_w: float,
    groove_w: float,
    ridge_h: float,
    base: float,
    across: str = "x",
    tilt: float = 0.0,
    pivot=(0.0, 0.0),
) -> list:
    """A plate of square-wave ridges: ``n`` grooves with a ridge on both sides, on a ``base`` thick slab.

    The ridges repeat along ``across`` ("x" or "y") over ``n * groove_w + (n + 1) * ridge_w`` and run
    ``length`` the other way, centred on ``at``; the ridge tops are ``base + ridge_h`` over the bench.
    The first and last features are ridges, and an odd ``n`` puts a groove on the centre.
    ``tilt`` [rad] turns the plate about y through ``pivot`` ``(x, z)`` so +x slopes down.
    """
    span = n * groove_w + (n + 1) * ridge_w
    flip = across == "y"

    def slab(name, offset, extent, z0, z1):
        half, centre = [0.5 * extent, 0.5 * length], [at[0] + offset, at[1]]
        if flip:
            half, centre = half[::-1], centre[::-1]
            centre[0], centre[1] = at[0], at[1] + offset
        x, z = centre[0], 0.5 * (z0 + z1)
        pose = (x, centre[1], z)
        if tilt:
            c, sn = np.cos(tilt), np.sin(tilt)
            pose = (
                pivot[0] + (x - pivot[0]) * c + (z - pivot[1]) * sn,
                centre[1],
                pivot[1] - (x - pivot[0]) * sn + (z - pivot[1]) * c,
                0.0,
                float(np.sin(0.5 * tilt)),
                0.0,
                float(np.cos(0.5 * tilt)),
            )
        return rss.slab(name=name, half=(*half, 0.5 * (z1 - z0)), at=pose, color=RIDGE_COLOR)

    specs = [slab("plate_base", 0.0, span, 0.0, base)]
    for i in range(n + 1):
        offset = -0.5 * span + 0.5 * ridge_w + i * (ridge_w + groove_w)
        specs.append(slab(f"plate_ridge{i}", offset, ridge_w, base, base + ridge_h))
    return specs


def pan(*, at, inner, wall: float = 0.004, height: float = 0.04, floor: float = 0.003) -> list:
    """A flat, long metallic pan: ``inner`` ``(x, y)`` floor centred on ``at``, walls ``height`` high."""
    ix, iy = 0.5 * inner[0], 0.5 * inner[1]
    kw = {"color": (0.72, 0.73, 0.76), "finish": STEEL}
    slabs = [rss.slab(name="pan_floor", half=(ix + wall, iy + wall, 0.5 * floor), at=(at[0], at[1], 0.5 * floor), **kw)]
    for name, half, centre in (
        ("pan_xn", (0.5 * wall, iy + wall, 0.5 * height), (-ix - 0.5 * wall, 0.0)),
        ("pan_xp", (0.5 * wall, iy + wall, 0.5 * height), (ix + 0.5 * wall, 0.0)),
        ("pan_yn", (ix, 0.5 * wall, 0.5 * height), (0.0, -iy - 0.5 * wall)),
        ("pan_yp", (ix, 0.5 * wall, 0.5 * height), (0.0, iy + 0.5 * wall)),
    ):
        slabs.append(rss.slab(name=name, half=half, at=(at[0] + centre[0], at[1] + centre[1], 0.5 * height), **kw))
    return slabs


def basin(*, at, radius: float, rim: float, drain: float, floor: float, wall: float = 0.006) -> StaticSpec:
    """A round basin on the bench: a concave static triangle mesh (no hull) with a through drain of radius ``drain``.

    The floor is ``floor`` thick and the drain a pipe through it; the rim stands ``rim`` high.
    """
    outer = radius + wall
    profile = [(drain, 0.0), (drain, floor), (radius, floor), (radius, rim), (outer, rim), (outer, 0.0), (drain, 0.0)]
    return StaticSpec(
        name="basin",
        asset=AssetSpec(path=str(revolved_obj("basin", profile, segments=48))),
        xform=(float(at[0]), float(at[1]), 0.0, 0.0, 0.0, 0.0, 1.0),
        color=(0.88, 0.89, 0.90),
        finish=Finish(roughness=0.15, metallic=0.1),
    )


def whiteboard(*, at, size, thickness: float = 0.008) -> StaticSpec:
    """A whiteboard lying on the bench, ``size`` ``(x, y)``, its top ``thickness`` over it."""
    return rss.slab(
        name="whiteboard",
        half=(0.5 * size[0], 0.5 * size[1], 0.5 * thickness),
        at=(at[0], at[1], 0.5 * thickness),
        color=(0.95, 0.95, 0.93),
        finish=Finish(roughness=0.2),
    )


def shelf(*, at, size, height: float, thickness: float = 0.015) -> list:
    """A shelf top ``size`` ``(x, y)`` with its top ``height`` over the bench, on two side panels."""
    wood = {"color": (0.55, 0.40, 0.27)}
    panel = 0.5 * (height - thickness)
    return [
        rss.slab(
            name="shelf_top",
            half=(0.5 * size[0], 0.5 * size[1], 0.5 * thickness),
            at=(at[0], at[1], height - 0.5 * thickness),
            **wood,
        )
    ] + [
        rss.slab(
            name=f"shelf_side{sign}",
            half=(0.004, 0.5 * size[1] - 0.01, panel),
            at=(at[0] + sign * (0.5 * size[0] - 0.02), at[1], panel),
            **wood,
        )
        for sign in (-1, 1)
    ]


# --- Scene -----------------------------------------------------------------------------------------


def add_arguments(parser) -> None:
    parser.add_argument("--solver", choices=("vbd", "chebyshev"), default="chebyshev")
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--mu", type=float, default=cable_scene.OBJECT_MU)
    parser.add_argument("--grip-mu", type=float, default=2.0, help="pad-handle friction (what a handle's mu is)")
    parser.add_argument("--force-limit", type=float, default=50.0, help="grip force [N]")
    parser.add_argument("--no-self-contact", dest="self_contact", action="store_false")
    cable_scene.add_lever_arguments(parser)


def build_scene(
    asset: ToolAsset,
    *,
    name: str,
    pose,
    extras=(),
    camera: tuple,
    solver: str = "chebyshev",
    iterations: int = 90,
    mu: float = cable_scene.OBJECT_MU,
    grip_mu: float = 2.0,
    force_limit: float = 50.0,
    self_contact: bool = True,
):
    """The tool at ``pose`` (xyz, bench frame) with ``extras`` (specs, or a function of the scene) and the AIRBOT.

    ``camera`` is ``(eye, look_at)`` in the bench frame. Returns ``(sim, obj, camera, vertices, tets, labels, settings)``.
    """
    vertices, tets, labels = cable_scene.load(asset)
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
    scene = rss.Scene(cell=arm_gripper.CELL, physics=physics, name=name)
    obj = scene.add(spec(asset, vertices, tets, labels, name="tool", pose=pose, settings=settings, mu=mu))
    for extra in extras(scene) if callable(extras) else extras:
        scene.add(extra)
    arm_gripper.add_arm(scene, base_offset=arm_gripper.MOUNT, force_limit=force_limit)
    surface = np.asarray(cable_scene.sim_surface(), dtype=np.float64)
    cam = scene.add(
        rss.CameraSpec(
            name="overview",
            width=960,
            height=540,
            eye=tuple(surface + camera[0]),
            look_at=tuple(surface + camera[1]),
            fov_y=35.0,
        )
    )
    sim = build_sim(scene, rigid_loop=False)
    solver_ = arm_gripper.vbd_solver(sim.pipeline.program.solver)
    arm_gripper.check_arm(sim, limit=force_limit)
    cable_scene.set_friction(sim, solver_, settings, mu=mu, grip_mu=grip_mu)
    return sim, obj, cam, vertices, tets, labels, settings


def rest_pose(vertices, quat=(0.0, 0.0, 0.0, 1.0), at=(0.0, 0.0)) -> tuple:
    """The tool's pose [m] that lies its lowest point :data:`cable_scene.PLACE_GAP` over the bench at ``at``."""
    turned = arm_gripper._rotate(np.asarray(quat), vertices)
    return (at[0], at[1], float(-turned[:, 2].min() + cable_scene.PLACE_GAP))


def config(args, asset: ToolAsset, settings) -> dict:
    return json.loads(json.dumps(cable_scene.config(args, asset, settings), default=str))


OUTPUT = Path(__file__).resolve().parents[2] / "outputs"
