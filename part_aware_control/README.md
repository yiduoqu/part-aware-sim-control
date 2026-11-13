# SIMPACT over RoboSimStudio

A small source-checkout integration. RoboSimStudio builds, steps, renders and
measures the scene; pinned SIMPACT generates, verifies and refines full action
sequences. The planner and simulator exchange JSON and PNG files. No perception
reconstruction is performed: the initial object and robot state is privileged
simulation state.

## Simulator: one scripted placement and replay

Run on an allocated ARM GH200, with this checkout and its sibling Newton main
on the Python path. The old activation helper changes directory, so return here:

```bash
# First allocate with srun/salloc, for example:
# srun -A ehpc-dev-2026d08-170-gpu -p gpu --gres=gpu:nvidia_gh200_120gb:1 \
#   --cpus-per-task=8 --mem=32G --time=00:45:00 --pty bash
rss_root="$PWD"
source /nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/newton/outputs/stage1_setup/activate.sh
cd "$rss_root"
export PYTHONPATH="$rss_root:$rss_root/../newton"
export ROBOSIMSTUDIO_RENDERER=tiled
python -u -m part_aware_control script --task cube --out outputs/simpact/cube_run
```

`script` records a common initial observation, writes the entire input `plan.json`,
executes it, and replays once in a fresh scene only if measured success passes.
Each execution has an initial image, one image per action, a final image, measured
poses/opening/object state, commanded actions, outcome gates, revisions and timing.
`result.json` separates initial execution from confirmation and records whether
the common initial state was reproduced. Images and readback are included in the
execution timing. Dense trajectories are omitted; video is optional.

To replay a plan directly, including the planner's `selected_plan.json`:

```bash
python -m part_aware_control worker --task cube \
  --plan outputs/simpact/cube_run/plan.json --out outputs/simpact/cube_replay --video
```

`--video` records `replay.mp4` at scene speed, rendering every second control
frame (25 fps for cube, 30 fps for Clawhauser). It also works with `observe` and
`script`; recording is off by default. Exact task settings can be preserved by
passing the original planning run's `task.json` with `--task-config`.

`--cube-start X Y` changes the cube's start in work-surface metres. The default
cube is 40 mm across and 50 g. The green target is 120 x 120 mm; the entire cube
must finish inside it. Success requires at least 40 mm lift, clean release, support
on the table, and at least half a second with drift <=3 mm and peak speed <=10 mm/s.
These thresholds live in `Task` and are saved before execution. To replay exact
custom thresholds/geometry use `--task-config task.json` (a serialized `Task`).

For the deformable scene use `--task clawhauser`; it calls the merged
`clawhauser_pick_drop.build_scene()` defaults. The executor uses the same AIRBOT
boundary, without that demonstration's fixed trajectory or runtime loop-aim
correction. Script generation may use the measured loop position after the common
settling interval; it is explicit in the saved initial state and input plan.
Scene context labels the nominal grip centre's work-surface frame, provides its
world equivalent and the existing downward TCP offset, and records the measured
world-frame loop shift after settling.
Final target placement is measured after release, using the whole object's mean
position within a 120 x 200 mm target. Lift uses the object's lowest particle,
not its held transport distance. The existing GPU mesh-quality latch retains
nonfinite/inverted-element detection at every substep; full diagnostic dumps are
not collected. A failed script is reported as failure, not as a planner result.

The validated Clawhauser script lifts in two slow stages, centers the held loop
over the target, and opens above first table contact so the hanging body can
land freely. Script and fresh confirmation passed the unchanged placement,
release, settling and mesh-validity gates. This is an explicit input sequence;
model-generated plans supply their own motions through the same executor.

## Hanging Clawhauser on the rod

`--task clawhauser_hang` calls Hanson's current `clawhauser_on_rod.build_scene()`
defaults: the fully deformable toy, loop grasp geometry, horizontal 6 mm rod,
and the current AIRBOT mount. No material, contact or solver overrides are added.
The worker begins at the same open elevated pose in every fresh scene.
The planner receives both native loop and tail pickup locations, their widths,
and separately measured settling shifts. Either pickup is allowed; the final
support must be the ring on the rod. Moves/holds retain the commanded jaw target
so a stalled grip does not relax at each action boundary.

```bash
python -m part_aware_control observe --task clawhauser_hang \
  --out outputs/simpact/hang_scene
/path/to/planner/python -m part_aware_control plan \
  --observation outputs/simpact/hang_scene/observation.json \
  --worker-command '/path/to/newton/python -m part_aware_control --video' \
  --provider codex --model gpt-6-sol --out outputs/simpact/hang_planning
```

Use the same Slurm setup below for ARM workers. The planner chooses the hanging
instruction and prompts from the observation's task. Both generation stages use
the existing six-axis `move` and `gripper_control` schema; nonzero world roll and
pitch are allowed. Translation deltas still specify TCP endpoints. While turning,
the virtual pad midpoint follows a straight line between those endpoints, using
the saved tool-local offset. A turn about a fixed pad pivot therefore requires
an explicit compensating TCP translation. IK checks position within 2 mm and
orientation within five degrees before execution.

Success requires the ring actually threaded onto and supported by the rod after
release: upward rod-to-loop support >=90% of weight, the entire inner-hole
contour at least 10 mm beyond the tip and within the rod span, measured contour
winding around the rod axis, actual loop contact, no touching fingers, jaws open
at least 60 mm, and the whole toy at least 20 mm off the table. The final half
second must satisfy the existing 3 mm drift / 10 mm/s speed limits. Any inverted
or nonfinite element, contact penetration over 0.5 mm, or excessive finger load
fails validity. Hole pose, winding, engagement, contacts and support are included
in waypoint feedback for refinement. Images alone cannot override these gates.

The native Hanson demo uses measured loop aim, turn and threading corrections.
Those callbacks remain confined to that separately labelled reference. Our
workers execute only the supplied complete sequence. To test a static reference
plan use `worker --plan ... --task-config ... --video`; the placement-oriented
`script` command does not author hanging plans. The recorded-reference baseline
and model campaign are separate in `outputs/simpact/hanging_loop_tail_20261112`.

The completed campaign used main `3393ff5`, the existing Codex login and
`gpt-6-sol`: ten initial plans and three refinements, with 17 actual model calls.
All initial plans were rejected during IK execution. Rejected plans retain their
partial traces for refinement; the original batch was continued from those exact
cached artifacts without repeating proposals or simulations.

The final 40-action plan executed the complete grasp, lift, turn, thread, release
and withdrawal sequence. The toy remained threaded, with no finger contact,
57.5 mm table clearance and zero inverted element-substeps. Full validation
failed: peak penetration was 1.477 mm against 0.5 mm, average loop support was
86.2% of weight against 90%, and final drift/speed exceeded settling limits.
No fresh confirmation ran because no candidate passed every gate. The final
21.8 N finger readback passes the native 30 N allowance; the saved VLM verdict
incorrectly compared it with the 20 N actuator setting. The verifier prompt now
states that distinction explicitly; saved campaign prompts and outcomes remain
unchanged.

Final artifacts are `recovery/result.json`, `recovery/selected_plan.json` and
`recovery/rollouts/attempt_012/replay.mp4` below that campaign directory. The
40.67-second, 30 fps video decoded all 1,220 frames. `experiment.json` records
source hashes, common-state checks and released GPU allocations.
The [tracked results bundle](../.agents/reports/simpact_hanging_20261112/README.md)
contains the final video, plan, measured outcome and archived campaign evidence
for browsing from a fresh clone.

## Planner: ten candidates, at most three refinements

The user-selected default is the existing Codex login with exact `gpt-6-sol`.
Install the separate environment described in [PLANNER_SETUP.md](PLANNER_SETUP.md).
Keep the Newton interpreter as the worker and call the planner interpreter
separately (both must resolve this source checkout):

```bash
python -m part_aware_control observe --task cube --out outputs/simpact/scene
# Reuse an existing Codex ChatGPT login; verify it with codex login status.
/path/to/planner/python -m part_aware_control plan \
  --observation outputs/simpact/scene/observation.json \
  --instruction 'Lift the orange cube and place it inside the green target.' \
  --worker-command '/path/to/newton/python -m part_aware_control' \
  --provider codex --model gpt-6-sol --out outputs/simpact/planned_cube
```

On this cluster the installed Codex CLI runs on the x86 login host. The planner
can run there without importing the simulator; send each worker to the existing
ARM GPU allocation with `srun`. With a live allocation ID in `sim_job`:

```bash
export PYTHONPATH="$rss_root:$rss_root/../newton"
export WARP_CACHE_PATH=/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/newton/.cache/warp
export OMP_NUM_THREADS=8
/path/to/planner/python -m part_aware_control plan \
  --observation outputs/simpact/scene/observation.json \
  --instruction 'Lift the orange cube and place it inside the green target.' \
  --worker-command "srun --jobid=$sim_job --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 /path/to/newton/python -m part_aware_control" \
  --provider codex --model gpt-6-sol --out outputs/simpact/planned_cube
```

For unattended runs, couple the allocation to the planner command using
`salloc --kill-command=TERM`, rather than leaving an interactive allocation open.
The planner remains on the login host and its `srun` workers use that allocation:

```bash
salloc -A ehpc-dev-2026d08-170-gpu -p gpu --gres=gpu:nvidia_gh200_120gb:1 \
  --cpus-per-task=8 --mem=32G --time=01:00:00 --kill-command=TERM \
  /path/to/planner/python -m part_aware_control plan \
  --observation outputs/simpact/scene/observation.json \
  --worker-command 'srun --nodes=1 --ntasks=1 --cpus-per-task=8 /path/to/newton/python -m part_aware_control' \
  --provider codex --model gpt-6-sol --out outputs/simpact/planned_cube
```

The allocation is released when the planner command exits, including failures.
Use a duration appropriate to the task; the unchanged Clawhauser solver takes
substantially longer per execution than the rigid cube.

The allocation must remain active for the full run. GPU workers execute
sequentially; model calls happen on the planner host. The CLI/auth and planner
Python must be executable on their host, and the Newton Python on the ARM node.

Use a new output directory per run. The exact observation task configuration is
passed to every worker. All ten initial complete sequences execute before the
first passing candidate is selected in proposal order. If none pass, each of up
to three refinements proposes one sequence with accumulated history. Verification
requires VLM validity AND measured success. Successful plans get exactly one
confirmation execution; failed planning gets none. API/schema errors are reported
separately, malformed responses get at most one retry, and invalid/unreachable
candidate commands are logged. No model substitution or extra optimization
budget follows a failed confirmation.

## Action contract

- Translation: signed world-frame metres, +z up. RSS construction coordinates are
  converted using the saved work-surface origin `(0.6, 0, 0.75)`.
- Proposal `DESCEND.delta_z` is a positive distance down; refined `move.delta_z`
  is signed. Both uppercase proposal and lowercase refined schemas are accepted.
- Orientations: xyzw. Refined rotations apply about fixed world X, Y, then Z axes.
- Jaw opening: total distance between pad faces, 0 to 80 mm. A `RELEASE` opens to
  the actual AIRBOT maximum. `ArmControl.aperture()` is half that distance.
- Motion uses the existing IK, smooth Cartesian interpolation and joint targets.
  Default average translation rate is 60 mm/s; average rotation rate is 0.5 rad/s
  (smooth interpolation has a higher peak).
  Gripper changes take at least 0.8 s. Script inputs can state duration explicitly.
  Every plan gets the same final two-second settling interval, holding its final
  commanded arm/gripper targets. It does not get any implicit release/withdrawal.
- Every candidate builds afresh, initializes the same open elevated pose, settles,
  then physically follows its supplied actions. An unreachable waypoint is an
  invalid input; no object attachment or candidate-specific pose teleport occurs.

## Validated runs

The saved Codex login and exact `gpt-6-sol` completed these model-planned runs.
These placement results predate the hanging update to main `3393ff5`; their exact
revisions remain in the saved artifacts. The hanging campaign uses the updated
source and is reported separately.

| Task | Initial candidates | Refinements | Confirmation | Result |
| --- | ---: | ---: | ---: | --- |
| Nominal cube | 10 | 0 | 1 | passed |
| Shifted cube start (-0.11, +0.02) m | 10 | 1 | 1 | passed |
| Clawhauser | 10 | 1 | 1 | passed |

Scripted cube and Clawhauser placements also passed fresh confirmation. The
model's Clawhauser refinement lifted 64.4 mm, lowered only 50 mm and opened above
the table after the initial full descents caused inversions. Its replay lifted
64.6 mm; both left the toy settled inside the predefined target with zero inverted
or nonfinite tet-substeps and zero table penetration. Hanson’s physics settings
were preserved. These are integration demonstrations using privileged state.

Local result artifacts, relative to the repository root:

- `outputs/simpact/cube_codex_20261111/planning/result.json`
- `outputs/simpact/cube_shifted_clawdiag_20261111/planning/result.json`
- `outputs/simpact/clawhauser_centeredrelease_20261112/script_validation/result.json`
- `outputs/simpact/clawhauser_centeredrelease_20261112/planning/result.json`

Each run retains raw model calls, proposals, executed actions, images, measured
gates and confirmation. The Clawhauser model run used 11 planning executions and
one confirmation, with no schema retries or additional optimization budget.

Continuous videos of the same selected model plans are saved under
`outputs/simpact/videos_20261112/{cube,cube_shifted,clawhauser}/replay.mp4`.
All three recorded replays passed their measured gates. The videos are 16.28 s,
17.28 s and 22.67 s respectively; `manifest.json` records source plans, exact task
settings, replay outcomes and decoded frame counts. These visualization replays
made no model calls and preserve Hanson’s physics defaults.

## Core checks

```bash
python -m unittest tests.test_simpact_actions
# In the planner environment with pinned SIMPACT importable:
python -m unittest tests.test_simpact_planner tests.test_simpact_hanging
```

No complete toolkit suite or model calls run inside the controller. Physical
validation still requires the scripted task execution and replay on the GPU.
