# The arm flips the soap case open, takes the bar out and sets it down

The rendered run of two, 2026-11-19, of [`soap_case_arm_pick.py`](../../../examples/10_heterogeneous/soap_case_arm_pick.py) at its defaults with `--render-every 2`. The AIRBOT Play, its hand shut and leaned 30 degrees, slides one finger under the thumb tab of `hinged_soap_case` and follows the tab round the hinge to 95 degrees; the lid falls open on its own. The hand stands up, opens to 64 mm, closes on the two long sides of `soap_bar` that stand 11 mm out of the tray, lifts the bar 45 mm, carries it 120 mm along x, lowers it to the bench and lets go.

Tray, lid with its tab, and bar are three rigid bodies; the 7 mm web between tray and lid is FEM and the only thing joining them. The lid stays shut and the bar stays in by soft-on-soft contact (0.4 mm radius). The web runs at 1.5e6 Pa by a scene override, a 27th of the asset's 4e7 Pa; nothing was run at the asset's value. Plain VBD, 20 substeps of 160 iterations at 60 Hz, Newton 1.8.0.dev0; the arm is on MuJoCo, coupled one way. RTX 5080, 1,022 s for 17.2 s simulated. All eight checks passed.

## Readings

| | |
|---|---|
| Shut at rest | lid -0.3 deg to the tray, front rims 0.39 mm apart |
| Pad force on the tab, at most | 3.4 N through the push; 8.0 N for one frame as the finger slid in under it |
| Lid | past 90 deg 2.33 s into the push, 170 deg 0.4 s later, lying at 180.0 deg |
| Tray tilt / slide, whole run, at most | 0.011 deg / 0.007 mm |
| Bar when taken | 1.5 mm along the hinge from where it was put |
| Pad squeeze while held | 18.8 to 19.3 N |
| Slip / tilt in the grip | 0.02 mm / 0.10 deg |
| Lifted | 44.1 mm |
| Set down off its mark by | 0.35 mm |
| Web, largest stretch / smallest volume ratio | 1.83 / 0.51 |
| Inverted web tets | 0 |
| Worst IK error | 0.007 mm |

The finger's path is planned from the tab's and the web's node positions once the case has settled; nothing is perceived. Only the pads collide. The bar's 1.5 mm of creep before it is touched is the soft-on-soft contact's, not the arm's. The limits are in the example folder's [README](../../../examples/10_heterogeneous/README.md#hinged-soap-case).

## What is here

- `overview.mp4`, `side.mp4`: the two cameras, the scene's own tiled renders, uncaptioned.
- `result.json`: the readings and the eight checks.
- `diagnostics.json.gz`: every control frame's row.

`trajectory.npz` is not archived.

## The second run

The same command without rendering, 994 s, after the scene's shared pieces were moved into its own file. All eight checks passed again, with the same settings and build report. The lid, the tray and the web read within a few percent of the table. Two readings do not: the bar crept 0.12 mm before it was taken, not 1.5 mm, and the pads held it at 10.3 to 11.9 N, not 18.8 to 19.3 N (slip 0.03 mm, tilt 0.34 deg, set down 0.27 mm off its mark). Why the two runs part was not looked into; its files are not archived here.

## How to run it again

```
python examples/10_heterogeneous/soap_case_arm_pick.py --render-every 2
```
