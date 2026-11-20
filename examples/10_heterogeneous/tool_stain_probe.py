# SPDX-License-Identifier: Apache-2.0
"""Probe: does repainting a stain tile show in the render, and do crumbs rest on the bench and the board?

Writes ``before.png`` and ``after.png`` (every third mark erased) and ``result.json`` under ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hetero_scene
import numpy as np
import tool_scene
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    asset = tool_scene.TOOLS["whiteboard_eraser"]
    vertices, _, _ = hetero_scene.load(asset)
    stain = tool_scene.StainMap(
        centre=(0.10, 0.0), size=(0.20, 0.12), grid=(10, 6), z=0.008, radius=0.003, contact_ke=1e4
    )
    names = []

    def extras(scene):
        scene.add(tool_scene.whiteboard(at=(0.10, 0.0), size=(0.26, 0.18)))
        names.extend(tool_scene.crumbs(scene, region=(0.0, 0.05, -0.05, 0.05), n=6, seed=1))
        return stain.specs()

    sim, _, camera, *_ = tool_scene.build_scene(
        asset,
        name="stain_probe",
        pose=tool_scene.rest_pose(vertices, at=(0.10, -0.25)),
        extras=extras,
        camera=((0.45, -0.30, 0.40), (0.10, 0.0, 0.0)),
    )
    stain.make_live(sim)
    for _ in range(args.steps):
        sim.run(1).sync()
    surface = np.asarray(hetero_scene.sim_surface())
    for label, erase in (("before", []), ("after", list(range(0, len(stain.points), 3)))):
        stain.hide(sim, erase)
        Image.fromarray(np.asarray(sim.render(camera)["rgb"])[..., :3].astype(np.uint8)).save(args.out / f"{label}.png")
    positions = tool_scene.crumb_positions(sim, names) - surface
    (args.out / "result.json").write_text(json.dumps({"crumb_z_m": positions[:, 2].tolist()}) + "\n")


if __name__ == "__main__":
    main()
