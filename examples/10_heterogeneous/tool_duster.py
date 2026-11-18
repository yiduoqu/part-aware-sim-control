# SPDX-License-Identifier: Apache-2.0
"""Cloth duster: the AIRBOT brushes the duster's cloth panels along a groove of the wipe's ridged plate into the sink gap.

The scene and route are the sponge wipe's (:mod:`tool_sponge_wipe`): the ridged plate with crumbs in its grooves,
a gap over a sink, the hand pinching the rigid handle near the butt, tilting the head down and making one straight
pass per groove. The tool is ``cloth_duster_fem``: eight cloth panels (2.5 mm VBD tet shells) on a rigid grip.
``--grooves N`` passes over the first N grooves. Score: the fraction of those grooves' crumbs out of the groove
(on a ridge, in the sink or off the plate); success at 60%.

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
import tool_scene
import tool_sponge_wipe as wipe

DUSTER = tool_scene._tool("cloth_duster_fem", 1.0e5, 200.0, handle=1, pad=-1, pitch_deg=-8.0)
"""Soft cloth (E 1e5, softer than the asset's 5e5); lying, it rests on its butt and two panels."""
# Rolled half a panel spacing about its axis: balanced on one panel's edge, the head topples sideways and flattens.
DUSTER = dataclasses.replace(
    DUSTER, quat=tuple(arm_gripper.quat_multiply(DUSTER.quat, arm_gripper.axis_angle_quat((1.0, 0.0, 0.0), np.pi / 8)))
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, conflict_handler="resolve")
    parser.add_argument("--out", type=Path, default=tool_scene.OUTPUT / "tool_duster" / "run")
    parser.add_argument("--asset", default="cloth_duster_fem")
    wipe.add_arguments(parser)
    parser.add_argument("--grooves", type=int, default=1, help="pass over the first N grooves")
    parser.set_defaults(force_limit=120.0, grip_mu=2.5, grip_frac=0.25, tilt_deg=15.0, press_mm=10.0, crumbs=24)
    args = parser.parse_args()
    asset = cable_scene.with_levers(dataclasses.replace(DUSTER, grip_frac=args.grip_frac), args)
    print(json.dumps({"grasp": tool_scene.grasp_handle(asset, *cable_scene.load(asset))}), flush=True)

    seen = []

    def clear(p):
        seen.append(p)
        return ~wipe.in_groove(p, x_max=wipe.PLATE_X1, half_y=0.5 * wipe.SPAN)

    base = wipe.wipe(args)
    sweep = dataclasses.replace(base, name="dust", ys=base.ys[: args.grooves], clear=clear, lay_front=0.09)
    result = wipe.run(args.out, asset, args, sweep)
    if not result:
        return
    # Score only the brushed grooves' crumbs: they are added groove by groove.
    brushed = np.arange(len(seen[-1])) < args.crumbs // wipe.GROOVES * args.grooves
    cleared = clear(seen[-1])[brushed]
    result.update(
        grooves=args.grooves,
        groove_crumbs=int(brushed.sum()),
        groove_cleared=int(cleared.sum()),
        score=float(cleared.mean()),
        success=bool(cleared.mean() >= 0.6),
    )
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("groove_crumbs", "groove_cleared", "score", "success")}), flush=True)


if __name__ == "__main__":
    main()
