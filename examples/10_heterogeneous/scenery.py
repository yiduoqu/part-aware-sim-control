# SPDX-License-Identifier: Apache-2.0
"""How the Clawhauser scenes' fixtures look: a glass cup, and a steel rod on a round foot.

The colliders stay what the scenes measured them as -- the cup's 24 flat panels and
square floor, the rod stand's square base -- and are hidden; these are drawn in their
place and touch nothing (``collide=False``), so how a run looks cannot change what
it does. The meshes are surfaces of revolution written once to the temp directory
and keyed by their profile, so concurrent runs share them.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from robosimstudio.core.render.finish import Finish
from robosimstudio.specs import AssetSpec, StaticSpec
from robosimstudio.specs.asset import ShapeSpec

__all__ = ["GLASS", "STEEL", "cup", "rod_elbow", "rod_foot"]

GLASS = Finish(roughness=0.05, opacity=0.35, ior=1.5)
GLASS_COLOR = (0.82, 0.91, 0.96)
STEEL = Finish(roughness=0.3, metallic=0.9)
STEEL_COLOR = (0.74, 0.76, 0.79)
FOOT_COLOR = (0.22, 0.23, 0.26)
SEGMENTS = 96


def _arc(centre, radius: float, start: float, end: float, count: int = 8) -> list[tuple[float, float]]:
    angles = np.linspace(start, end, count)
    return [(centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)) for a in angles]


def revolved_obj(name: str, profile, segments: int = SEGMENTS) -> Path:
    """The surface of revolution of ``profile`` ``[(r, z), ...]`` about z, as an OBJ path."""
    profile = np.asarray(profile, dtype=np.float64)
    key = hashlib.sha1(profile.tobytes() + str(segments).encode()).hexdigest()[:12]
    path = Path(tempfile.gettempdir()) / "clawhauser_scenery" / f"{name}_{key}.obj"
    if path.exists():
        return path
    angles = 2.0 * np.pi * np.arange(segments) / segments
    rings = [np.column_stack((r * np.cos(angles), r * np.sin(angles), np.full(segments, z))) for r, z in profile]
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in np.concatenate(rings)]
    for i in range(len(profile) - 1):
        for j in range(segments):
            a, b = i * segments + j + 1, i * segments + (j + 1) % segments + 1
            c, d = a + segments, b + segments
            lines += [f"f {a} {b} {d}", f"f {a} {d} {c}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(f".{os.getpid()}.tmp")
    scratch.write_text("\n".join(lines) + "\n")
    os.replace(scratch, path)  # atomic, so a run reading it never sees half a file
    return path


def cup(*, bore: float, wall: float, rim: float, floor: float, at) -> StaticSpec:
    """A glass cup over the panel container: ``bore`` inner radius, ``rim`` tall, on the bench at ``at``."""
    outer = bore + wall
    fillet = min(0.004, 0.5 * floor + wall)
    profile = [(0.0, 0.0), (outer - fillet, 0.0)]
    profile += _arc((outer - fillet, fillet), fillet, -0.5 * np.pi, 0.0)
    profile += _arc((bore + 0.5 * wall, rim - 0.5 * wall), 0.5 * wall, 0.0, np.pi)  # the rolled rim
    profile += [(bore, floor + 0.002), (bore - 0.002, floor), (0.0, floor)]
    return StaticSpec(
        name="container_glass",
        asset=AssetSpec(path=str(revolved_obj("cup", profile))),
        xform=(float(at[0]), float(at[1]), 0.0, 0.0, 0.0, 0.0, 1.0),
        collide=False,
        color=GLASS_COLOR,
        finish=GLASS,
    )


def rod_foot(*, at, radius: float = 0.040, height: float = 0.012) -> StaticSpec:
    """A round foot under the rod's post, with a bevelled top edge, standing on the bench."""
    bevel = 0.003
    profile = [(0.0, 0.0), (radius, 0.0), (radius, height - bevel), (radius - bevel, height), (0.0, height)]
    return StaticSpec(
        name="rod_foot",
        asset=AssetSpec(path=str(revolved_obj("rod_foot", profile))),
        xform=(float(at[0]), float(at[1]), 0.0, 0.0, 0.0, 0.0, 1.0),
        collide=False,
        color=FOOT_COLOR,
        finish=STEEL,
    )


def rod_elbow(*, at, radius: float) -> StaticSpec:
    """A steel ball where the rod meets its post, so the bend reads as one bent bar."""
    return StaticSpec(
        name="rod_elbow",
        shape=ShapeSpec(kind="sphere", radius=radius),
        xform=(*(float(v) for v in at), 0.0, 0.0, 0.0, 1.0),
        collide=False,
        color=STEEL_COLOR,
        finish=STEEL,
    )
