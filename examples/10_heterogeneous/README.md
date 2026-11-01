# Clawhauser loop pick and place

`clawhauser_pick_place.py` builds an RSS scene with one labelled tetrahedral
Clawhauser, a bench, and two scripted kinematic fingers. The fingers pinch its
loop by Newton contact, lift it, and move it 180 mm across the table. To release,
the left jaw opens, the right finger tips downward, then the right jaw withdraws.
Opening the right jaw before tipping carries the loop sideways and can leave it
hanging on the finger. There are no particle-body attachments or scripted
object transforms. The mesh remains one face-connected, part-labelled FEM
volume throughout the run; tail, body, and loop are not separate objects.

The source estimate is used directly: parts 0 and 1 have E=0.5 MPa, ν=0.45,
ρ=300 kg/m³; part 2 has E=5 MPa, ν=0.49, ρ=1100 kg/m³. The builder maps each
tetrahedron to its source part and gathers density-weighted mass at vertices.
`damping_time` and the contact penalty are solver settings. Newton VBD currently
exposes one global soft friction coefficient; this scene uses the loop's 0.9
estimate during loop grasping and keeps all per-part friction values in the
source JSON for later contact work.

Run from an allocated CUDA compute node with the sibling Newton checkout at
commit `d960ec3b8be23d7acac01c0ba10b8d04dca8b549` (1.8.0.dev0). In the
current cluster workspace:

```bash
rss_root="$PWD"
source ../newton/outputs/stage1_setup/activate.sh
export PYTHONPATH="$rss_root:$rss_root/../newton"
python "$rss_root/examples/10_heterogeneous/clawhauser_pick_place.py" \
  --out "$rss_root/outputs/clawhauser/drop" --render-every 1

# Lower the closed gripper before the same release sequence (metres).
python "$rss_root/examples/10_heterogeneous/clawhauser_pick_place.py" \
  --lower-before-release 0.20 \
  --out "$rss_root/outputs/clawhauser/lowered_release" --render-every 1
```

On the GH200's Linux ARM environment, `uv sync` of the entire RSS project is
currently blocked by an unrelated PyRoKi dependency chain: `yourdfpy` requests
`trimesh[easy]`, whose `embreex` wheel is unavailable for Linux ARM. The command
above uses the validated Newton environment and the RSS source checkout. It
needs `imageio-ffmpeg` there for video output.

The output contains state arrays, per-frame diagnostics, a manifest, and a
result summary. The `overview.mp4` video is written when rendering is enabled.
The example runs at 30 Hz with 80 Newton substeps and 48 VBD iterations per
substep. The default 12-second episode releases in the air; lowering adds a
two-second descent after transport and records 14 seconds. Materials, mesh,
finger geometry, friction and solver settings are identical in both modes.
The fingers remain kinematic, with no actuator force cap.

The result separates airborne release, lowered release and mesh integrity.
Airborne release requires a contact-free falling interval; lowered release
requires final finger clearance, table contact and less than 20 mm body drift
over the final half-second. Target error is reported without a landing-accuracy
gate. Overall `success` also requires zero recorded inverted tetrahedra: a
successful release can therefore coexist with a failed impact-integrity check.
Diagnostics sample 30 Hz control frames, not every internal solver substep.
Table contact before release is recorded separately; contact counts are not
measurements of table load. A lower-height drop is not proof of supported
placement before opening.
The current 200 mm descent releases and settles, but finishes about 160 mm from
the nominal target and briefly inverts elements after release. It is not an
accepted accurate pick-and-place result.

## AIRBOT Play arm variant

`airbot_play_pick_place.py` imports the six-joint AIRBOT Play arm and parallel
finger visual model from DISCOVERSE. Its `assets/airbot_play/SOURCE.md` records
the upstream revision, MIT license and local gripper changes. RSS controls the
articulated arm with its kinematic backend and couples finger contact one way
into the same VBD Clawhauser volume. Its visual shell, six joint axes and
fingers come from the AIRBOT model; this mode does not validate AIRBOT motors
or two-way force response. The task-specific fingertip pads and shelves are
visible and documented in `SOURCE.md`.

Run it on the same CUDA node and Python environment as above:

```bash
python "$rss_root/examples/10_heterogeneous/airbot_play_pick_place.py" \
  --out "$rss_root/outputs/clawhauser/airbot_play" --render-every 1
```

The default 390 steps record 13 seconds at 30 fps. `diagnostics.json` reports
arm pose, joint positions, per-side loop contacts and tetrahedral integrity at
each frame; `result.json` scores actual lift, transfer, release and settling.
Shorten `--steps` for a smoke run. The baseline script above remains the
verified contact-grasp reference while the AIRBOT arm variant is tuned. The
current full AIRBOT recording under `outputs/clawhauser/airbot_play_final/`
shows the arm reaching, closing and moving, but `result.json` reports
`success: false`: the loop slips off the fingertips during lift and remains at
the source zone.
