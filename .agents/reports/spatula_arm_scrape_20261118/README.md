# The arm picks the spatula off the bench and scrapes the pan

One run, 2026-11-18, of [`spatula_pan_arm_scrape.py`](../../../examples/10_heterogeneous/spatula_pan_arm_scrape.py) at its defaults. The AIRBOT Play closes on the wooden handle of `silicone_spatula` where it lies on the bench, lifts it 120 mm, carries it over `rigid_pan`, tilts it to 45 degrees with the blade's leading edge rolled level (11.65 degrees), presses 6 mm past touching, pushes 100 mm along the pan's floor, lifts, sets down, and drags 100 mm back.

The handle is a free rigid body held only by pad friction (20 N a finger, pad friction 2.83); the silicone head is FEM. Chebyshev VBD, 60 substeps of 160 iterations, Newton 1.8.0.dev0 (5cd406a1); the arm is on MuJoCo, coupled one way. RTX 5080, 3,190 s for 20.7 s simulated.

## What is here

- `overview.mp4`, `close.mp4`: the two cameras, path traced from the recorded trajectory and captioned.
- `result.json`: the readings and the nine checks, all passed.
- `diagnostics.json.gz`: every control frame's row, lossless.
- `caption.py`: what wrote the title, the materials line and the clock onto the videos, from the Windows machine that filmed them (its font paths are that machine's).

## Readings

| | |
|---|---|
| Lifted | 119 mm of 120 mm |
| Handle pitch poised over the pan | 45.0 deg |
| Floor force, push (middle of the stroke) | 0.88 N, standard deviation 0.06 N |
| Floor force, drag (middle of the stroke) | 0.32 N, standard deviation 0.03 N |
| Push over drag | 2.7 |
| Leading edge on the floor, push / drag | 1.00 / 1.00 |
| Tip travel | +100.1 mm / -100.1 mm |
| Handle turned in the grip, at most | 0.78 deg, slipped 0.23 mm |
| Peak pad squeeze | 20.5 N |
| Least clearance over the rim | 40 mm (computed; see below) |
| Worst IK error | 0.4 mm |
| Inverted tets | 0 |

## How to run it again

```
python examples/10_heterogeneous/spatula_pan_arm_scrape.py
python examples/10_heterogeneous/spatula_pan_arm_scrape.py --replay <run>/trajectory.npz --render-every 1
python .agents/reports/spatula_arm_scrape_20261118/caption.py <replay folder> overview overview_captioned.mp4
```

The second line films a finished run again without solving it (1,242 frames, two 1280x720 cameras, 139 s with `ovrtx 0.4.1`). `trajectory.npz` is 99 MB and is not archived here.

## What this does not show

- One run: one pitch, one press depth, one speed, one grip point.
- The pan is drawn but only its floor collides, as a flat slab. Rim clearance is computed from the pan's dimensions, and nothing scrapes into the corner.
- The handle is rigid, so this scene does not tell a heterogeneous model from a homogeneous one.
- The head's Young's modulus (1.5 MPa) is the asset's and is not calibrated, and this mesh bends 54% stiffer than a refined one. The forces hold for this asset and mesh only.
- The push force is about 20% under the same stroke with the handle driven kinematically (1.09 N) and drifts from 0.94 N to 0.76 N along the stroke. The grip turning by about 0.5 deg and a hand-height error of 0.5 mm are both of that size; they were not separated. The first dwell (0.4 s) ends before the floor force settles.
- The coupling is one way: the arm does not feel the spatula's load.
- The spatula lying flat on a bench reads 2.9 times its weight in `spatula_s0.py`; the cause is not known. Here it rested 0.5 mm over the bench with the handle 0.7 deg off level, and the pick was not affected.

## Provenance

The solve ran on a working copy at 9813224 with the stamp scripts that became 2164f42. The pan's and spatula's colours and `--replay` were added to the script after the solve; the pan parts are drawn with `collide=False`, so the solve was not run again. The three committed scripts were then passed through `ruff format` (and one unused import removed from `spatula_s0.py`); the formatted files import on the same machine but the run was not repeated with them.
