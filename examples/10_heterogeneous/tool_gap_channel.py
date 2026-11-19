# SPDX-License-Identifier: Apache-2.0
"""Gap sponge in a groove: the AIRBOT drags the narrow sponge along a groove of a ridged plate to clean it.

The plate and the arm's motion are :mod:`tool_sponge_wipe`'s (24 mm grooves along x, open at the far end).
The hand grasps the gap sponge wand, lowers the narrow sponge into the centre groove and drags it along,
pushing the crumbs out of the end. Score: the fraction of crumbs that left the groove.

Why heterogeneous: the sponge compresses against the groove walls and keeps pushing along them; a
rigid tool of the same width jams, and a soft one with no handle buckles.

Run on a CUDA node with RoboSimStudio on PYTHONPATH.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tool_scene
from tool_sponge_wipe import (
    GROOVE_W,
    PLATE_X1,
    X0,
    Sweep,
    add_arguments,
    in_groove,
    plate_specs,
    run,
    soft_sponge,
    wipe,
)


def channel(args) -> Sweep:
    """The wipe scene's plate, one pass down its centre groove with every crumb in that groove."""
    sweep = wipe(args)
    y = args.plate_y

    def extras(scene, names):
        names.extend(
            tool_scene.crumbs(
                scene,
                region=(X0 + 0.05, PLATE_X1 - 0.05, y - 0.008, y + 0.008),
                n=args.crumbs,
                spacing=0.012,
                size=(0.005, 0.008),
                seed=args.seed,
                z0=0.020,
            )
        )
        return plate_specs(y=y, x0=args.plate_start)

    sweep.extras, sweep.ys, sweep.scored, sweep.soft_radius = extras, (y,), (), None
    sweep.clear = lambda p: ~in_groove(p, x_max=PLATE_X1, half_y=0.5 * GROOVE_W, y=y, x_min=args.plate_start)
    return sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=tool_scene.OUTPUT / "tool_gap_channel" / "run")
    parser.add_argument("--asset", default="gap_sponge_wand")
    add_arguments(parser)
    parser.set_defaults(
        force_limit=120.0,
        grip_mu=2.5,
        roll_deg=0.0,
        tilt_deg=0.0,
        press_mm=12.0,
        plate_y=0.0,
        plate_start=0.06,
        grip_frac=0.5,
        crumbs=6,
        grooves=1,
    )
    args = parser.parse_args()
    asset = soft_sponge(tool_scene.TOOLS[args.asset], args)
    run(args.out, asset, args, channel(args))


if __name__ == "__main__":
    main()
