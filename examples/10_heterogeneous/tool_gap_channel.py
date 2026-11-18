# SPDX-License-Identifier: Apache-2.0
"""Gap sponge in a groove: the AIRBOT drags the narrow sponge along a groove of a ridged plate to clean it.

The plate is the wipe scene's (:func:`tool_scene.ridged_plate`) turned so its 24 mm grooves run along x,
the drag direction, and open at the far end. The hand grasps the gap sponge wand like
:mod:`tool_sponge_wipe`, lowers the 16 mm sponge into the centre groove and drags it along, pushing the
crumbs out of the end. Score: the fraction of crumbs that left the groove.

Why heterogeneous: the sponge compresses against the groove walls and keeps pushing along them; a
rigid tool of the same width jams, and a soft one with no handle buckles.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tool_scene
from tool_sponge_wipe import BASE, GROOVE_W, THICK, X0, X1, Sweep, Y, add_arguments, in_groove, plate, run, soft_sponge

GROOVES = 3


def channel(args) -> Sweep:
    def extras(scene, names):
        names.extend(
            tool_scene.crumbs(
                scene,
                region=(0.15, 0.27, Y - 0.006, Y + 0.006),
                spacing=0.009,
                n=args.crumbs,
                size=(0.003, 0.005),
                seed=args.seed,
                z0=BASE,
            )
        )
        return plate(across="y", at=(0.5 * (X0 + X1), Y), length=X1 - X0, n=GROOVES)

    return Sweep(
        "channel",
        extras,
        start_front=0.13,
        end_front=X1 + 0.015,
        contact_z=BASE,
        hover_z=THICK + 0.05,
        # Cleared: out of the groove's end, or onto a ridge.
        clear=lambda p: ~in_groove(p, x_max=X1 - 0.004, half_y=0.5 * GROOVE_W),
        camera=((0.45, -0.40, 0.20), (0.24, -0.10, 0.02)),
        soft_radius=0.0015,
        lay_front=0.04,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=tool_scene.OUTPUT / "tool_gap_channel" / "run")
    parser.add_argument("--asset", default="gap_sponge_wand")
    add_arguments(parser)
    parser.set_defaults(tilt_deg=0.0, press_mm=4.0, crumbs=12)
    args = parser.parse_args()
    asset = soft_sponge(tool_scene.TOOLS[args.asset], args)
    run(args.out, asset, args, channel(args))


if __name__ == "__main__":
    main()
