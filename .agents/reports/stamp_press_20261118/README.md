# Two stamps pressed onto paper

Two runs, 2026-11-18, each of its scene at its defaults, on Newton 1.8.0.dev0 with plain VBD at 60 Hz and 20 substeps. The videos are the scenes' own tiled renders, uncaptioned; neither run was path traced.

## `spring_stamp/`: a plate presses the retractable stamp

[`stamp_press.py`](../../../examples/10_heterogeneous/stamp_press.py) with `retractable_stamp_spring`, 80 iterations. The cap and plunger are one rigid body and the sleeve another; a prismatic joint the solver holds keeps them in line and stops the travel, the coil is a force along it, and the meshed helix is drawn, not solved. All nine checks passed.

| | |
|---|---|
| Die over the paper at rest | 3.06 mm |
| Plate travel at first touch | 3.01 mm |
| Relief on the paper at the peak | 1.00 |
| Spring force, rest / peak | 0.25 N / 4.13 N |
| Die over the paper after release | 3.06 mm |
| Tilt / yaw, at most | 0.023 deg / 0.040 deg |
| Inverted tets | 0 |
| Solve | 209 s for 5.4 s |

With the coil as the joint's drive too (`--joint-spring`) the cap sinks 3.3 mm and the die rests on the paper; with no joint (`--no-joint`) the run passes as well, the two groups up to 0.010 deg out of line.

## `rubber_stamp_arm/`: the arm picks the rubber stamp up, carries it and presses it

[`rubber_stamp_arm_press.py`](../../../examples/10_heterogeneous/rubber_stamp_arm_press.py) with `rubber_stamp`, 160 iterations. The AIRBOT Play closes on the knob at 20 N a finger, lifts the stamp 50 mm, carries it 120 mm to the paper, sets it down, pushes 1.5 mm further, holds, lets go and backs away. The cushion's modulus is 1e5 Pa by a scene override, a third of the asset's, which puts the stiffest part at 100 times the softest. All eight checks passed.

| | |
|---|---|
| Lifted | 49.7 mm |
| Set down off its mark by | 0.33 mm |
| Cushion squeezed at the peak | 1.28 mm of 3 mm, of which 0.27 mm standing |
| Die squeezed at the peak | 0.30 mm |
| Relief on the paper under the press | 1.00 |
| Pad squeeze, peak | 20.2 N |
| Push through the grip (friction, both pads) | about 64 N |
| Slip in the grip | 0.18 mm |
| Tilt while held, at most | 0.27 deg |
| Inverted tets | 0 |
| Solve | 1007 s for 14.3 s |

The standing squeeze is the solve's residual, not the stamp's 0.38 N weight: it halves when the iterations double. The 64 N push is over the arm's 60 N; the arm is driven by position here and is not pushed back.

## What is here

In each folder: `overview.mp4` and `side.mp4`, the two cameras; `imprint.png`, the mark the die left; `result.json`, the readings and checks; `diagnostics.json.gz`, every control frame's row. `trajectory.npz` (30 and 64 MB) is not archived.

## How to run it again

```
python examples/10_heterogeneous/stamp_press.py --render-every 2
python examples/10_heterogeneous/rubber_stamp_arm_press.py --render-every 2
```
