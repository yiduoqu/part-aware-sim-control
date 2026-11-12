# SIMPACT integration — completed

Source: /nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/handoffs/simpact_integration_plan_20261110.md.
Completion audit: 12 November 2026.

## Outcome and scope

All three integration milestones pass: scripted cube with replay, model-planned
cube with replay (nominal and one shifted start), and scripted plus model-planned
Clawhauser with fresh confirmation. Independent review accepted the code boundary,
planning behavior, states, images, selected-plan identity and confirmation.

The implementation is in simpact-integration-20261110/part-aware-sim-control,
branch yiduo/simpact-integration-20261110. Publication to yiduo/dev and main
was subsequently authorized by the user on 12 November 2026.
The original robosimstudio checkout and its untracked AIRBOT G2 work were preserved.
No simulator rewrite, physics tuning, reconstruction, test framework or broad
test campaign was introduced.

## Model, dependencies and Hanson fidelity

The user explicitly selected existing Codex ChatGPT login and exact gpt-6-sol,
superseding the handoff's Gemini default. The provider invokes official codex exec
with image inputs and stage-specific structured JSON. Codex manages its saved
login; the bridge does not extract credentials, use private endpoints or fall
back to another provider/model. Google remains an explicit optional backend.
The planner environment has no google-genai installed; actual image/schema calls
and full task runs succeeded without a Google key.

Validated source revisions:

- Controller base: 1190c756688d62bc50071cfa80b3ca7d2c333092.
- Sibling Newton: 5cd406a10a4d9160132fa115124254005f675d8f.
- Imported SIMPACT: 692aab2bb92e4df324d28c76630d17b08e814287.

The controller/Newton merge trees match Hanson heterogeneous-tests heads
8a2b83910e1da406b89933059ba961d1ddbbc88e and
c30507d2ba41fd081953f08a45886b73daef60f0. Final fresh fetches confirm those heads
and Newton main are unchanged. Controller main is now
937912dceb3e104cd19e3156bcd2ad67739edd83; its AIRBOT, Clawhauser scene/pick-drop
and mesh-quality modules are identical to the validation pin. Its USB assets and
rigid-part additions do not require changing this fully deformable task.
No merge or repeated migration audit was needed.

Clawhauser calls clawhauser_pick_drop.build_scene() without physical overrides:
fully deformable labelled toy, REFERENCE materials, 60 Hz, 20 substeps,
Chebyshev 160 iterations, one-way coupling, sticky/self contacts and 20 N fingers.
No rigid-loop projection, rigid-part substitution or AIRBOT G2 replacement.

Official Codex references checked for the provider decision:
- https://learn.chatgpt.com/docs/non-interactive-mode
- https://learn.chatgpt.com/docs/auth?surface=cli
- https://developers.openai.com/api/docs/models/gpt-6-sol

## Implementation and execution contract

- part_aware_control/actions.py normalizes uppercase proposal primitives and
  lowercase move/gripper_control into world-frame metre deltas, xyzw rotations
  and 0–80 mm total jaw opening. Positive DESCEND magnitudes move down.
- tasks.py builds the procedural cube or reuses Clawhauser defaults.
  rollout.py uses existing ArmControl planning/following and simulator stepping.
- planner.py imports the pinned SIMPACT generator/optimizer/verifier, injects
  the rollout callback, AIRBOT trace parser and measured success gate.
- providers.py supplies the Codex CLI transport; prompts and separate planner
  requirements avoid installing SIMPACT simulator/perception extras into Newton.
- __main__.py provides observe/script/worker/plan. README.md and PLANNER_SETUP.md
  document source-checkout usage, environment separation, Slurm and saved login.

Every whole-plan execution builds a fresh scene, initializes the same open,
elevated TCP (0.47, 0, 1.00) world metres, and settles for one second.
ArmControl.start() is used only for that shared pose; supplied actions physically
approach, close, lift, transport, lower, open and withdraw. The runner adds only
the common final two-second settling interval. There is no attachment,
candidate-specific teleportation or hidden successful trajectory.

The work-surface origin is (0.6, 0, 0.75). Initial observations are explicitly
privileged simulation state. Clawhauser context labels the nominal work-surface
grip centre, world equivalent, existing +14 mm downward-grasp TCP offset and
observed world-frame settled loop XY shift. The scripted input uses that observed
shift before execution; no runtime loop-aim correction is hidden in the worker.

The fixed default duration rule is max(0.8 s, translation/0.06 m/s,
maximum rotation/0.5 rad/s), also stated in model prompts. Explicit script
durations are allowed. Plans are bounded to 60 actions and 180 simulation seconds.

## Budget and output behavior

The exact configured budget is ten initial proposals and at most three
single-plan refinements. All initial full sequences execute before verification;
the first verified success in proposal order stops optimization. Refinements
receive accumulated actions, observations, measured failures and VLM feedback.
Success requires both VLM acceptance and measured task gates.

Only a successful plan receives one fresh confirmation. Failed confirmation
starts no extra budget. Exhausted failures retain their last plan/history without
claiming that a best_plan field proves success. Invalid/unreachable commands are
logged; no substitute plan is invented. Malformed responses get at most one
retry; API/schema/execution errors are distinct from physical task failure.

Saved evidence includes raw model responses, proposals, input/selected plans,
actual commands, initial/waypoint/final JSON and PNGs, measured outcomes, source
revisions, configuration, actual counts and timing. Dense trajectories/contact
dumps/video are omitted. Existing mesh-quality hooks retain every-substep
inverted/nonfinite detection. Waypoint progress streams existing summary data
without additional GPU reads.

Worker callback indices are forwarded through --index. Live Clawhauser workers
record correct indices 0–10, fixing historical cube worker JSON that always used
local index 0. Historical planner filenames/history/counts were already correct;
those archived results remain unchanged.

## Task criteria fixed before judging

Cube: 40 mm, 50 g procedural box; whole-box containment in a 120 x 120 mm target,
at least 40 mm lift, release, table support and final stability.

Clawhauser: whole-object vertex-mean position within the predefined 120 x 200 mm
target after release; at least 30 mm lift of its lowest particle, release,
support and stability. Held transport alone cannot certify placement.

Both use a final 0.5 s window with drift <=3 mm and centroid speed <=10 mm/s,
surface tolerance 2 mm, table penetration <=5 mm and IK position error <=10 mm.
Any inverted/nonfinite tet-substep invalidates Clawhauser, even if it later rests.

## Verified milestones

Paths below are relative to this implementation checkout.

| Evidence | Initial model plans | Refinements | Fresh confirmation | Outcome |
| --- | ---: | ---: | ---: | --- |
| outputs/simpact/cube_nominal/result.json | scripted | 0 | 1 | passed |
| outputs/simpact/cube_codex_20261111/planning/result.json | 10 | 0 | 1 | passed |
| outputs/simpact/cube_shifted_clawdiag_20261111/planning/result.json | 10 | 1 | 1 | passed |
| outputs/simpact/clawhauser_centeredrelease_20261112/script_validation/result.json | scripted | 0 | 1 | passed |
| outputs/simpact/clawhauser_centeredrelease_20261112/planning/result.json | 10 | 1 | 1 | passed |

Cube scripted lift 119.831 mm; final-source cube_final_replay/rollout.json also
passed, including table penetration 4.9 micrometres. Its later duplicate-directory
error did not overwrite the successful rollout. Nominal model cube used ten
successful initial executions, selected candidate 0 and confirmed (wall 1041.14 s).
The shifted start(-0.11,+0.02) m produced ten real initial failures; one model
refinement corrected grasp height 0.81→0.78 m and confirmed (wall 1260.01 s).
No extra rounds, provider fallback or adapter compensation occurred.

Clawhauser scripted input lifts in two stages (6+8 s), centers the held grasp over
the target, lowers140 mm over8 s, opens above first contact and withdraws.
First/replay lifts 161.564/160.910 mm; zero inversions, nonfinite states or table
penetration. Both pass the same target/settling gates. Exact saved initial fields,
Task, geometry, input plan and commands match. Final means differ11.98 mm,
demonstrating contact variation within the predefined target, not exact dynamics.

The Clawhauser model run completed all 10 original plans and rejected all 10.
Candidate 8 lifted44.959 mm and passed every task gate except mesh validity:
9 BODY inverted tet-substeps first occurred during lowering. Others missed lift
and sometimes placement/validity. All saved initial measurements/configuration
match exactly, and raw proposals match saved input plans with correct indices.

Actual gpt-6-sol refinement 1 cited candidate 8's clearance and prior compression:
lift200 mm, hold/carry, lower only50 mm, hold, open80 mm above the table,
withdraw and settle. The model supplied its own14-action sequence; it did not
receive the scripted input plan or executor corrections.

First refined execution / confirmation:
- Whole-toy lift 64.366 / 64.620 mm.
- Final mean(0.596651,0.036467,0.782022) /
  (0.608348,0.050281,0.785941) world metres; both inside the unchanged target.
- Final drift 6.976 / 6.937 micrometres; peak speed0.262 / 0.308 mm/s.
- Zero inverted/nonfinite tet-substeps and zero table penetration in every phase.
- BODY minimum Jacobian 0.370868 / 0.368590.
- All measured gates pass; VLM accepts both (confidence 0.96 / 0.98).
- Saved initial gripper/object/quality, Task, geometry, selected/input plan and
  executed commands match exactly. Final means differ18.52 mm; both satisfy criteria.

The final model result reports codex/gpt-6-sol, actual 10/10 proposals,
11 planning executions, one refinement, one confirmation and success=true.
Exactly 14 API calls occurred: one proposal, ten initial verifiers, one
refinement, one refined verifier and one confirmation verifier; all retry 0.
Independent review accepted identities, images, physical gates, budget and stopping.
Model wall 12351.79 s; confirmation worker wall 1047.52 s.

## Core checks and completion audit

Ten focused offline checks passed in 0.396 s using the actual pinned planner:
four action/frame/aperture checks, five planner budget/trace/error checks and
one mocked CLI schema/image/model/auth contract. Ruff passed the package and both
focused test files; the later two-line script reference change also passed Ruff.
Actual scripted/model execution validates that reference and the worker index.
No tests were added for timing experiments, logging, documentation or conversions
already covered; no complete toolkit suite was run.

| Requirement | Authoritative evidence |
| --- | --- |
| Small pinned adapter; environments separate | package, two requirement files, README/PLANNER_SETUP; actual x86 planner/ARM workers |
| Correct frames, opening and trace | actions.py, AIRBOT parser, four action checks, live primitive and refined executions |
| Common fresh state and explicit physical approach | rollout.py; all 10 initial, refinement and replay saved state/configuration comparisons |
| Exact 10+3 policy, error handling and stopping | planner constants/imported loop, focused failure-path checks, actual three model runs and raw call logs |
| Scripted cube and replay | cube_nominal summary plus final-source replay |
| Model nominal and shifted cube with replay | both cube planning result.json files and independent review |
| Scripted Clawhauser without physical overrides | script_validation summary, states/images/quality and source comparison |
| Model Clawhauser placed after release and confirmed | planning result.json, selected plan, both positive outcomes/VLM verdicts and independent review |
| Compact reproducible evidence | task/geometry/revisions, actual counts, commands, JSON/PNGs, raw responses and timings |
| Existing Codex login/exact model | provider source, live smoke and all model-run metadata; no Google SDK/key required |
| Allocation cleanup | job 5189054 COMPLETED with exit code 0 after 04:26:25; all steps completed, launcher gone, salloc relinquished |
| Preserve existing work and avoid scope expansion | only new package/plan/two focused test files in isolated checkout; Newton clean |

No required implementation or validation work remains. No GPU allocation is
left running for this task. The user subsequently requested publication to
yiduo/dev followed by merging into remote main.

## Useful paths and deferred work

ARM simulator: /nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/newton/.venv/bin/python.
Planner: /tmp/simpact-planner-venv/bin/python, with sibling SIMPACT importable.
Source-checkout worker commands and Slurm examples are in part_aware_control/README.md.
Use fresh output directories. Preserved earlier Clawhauser diagnostic runs under
outputs/simpact/clawhauser_* explain the invalid lift/setdown attempts and the
early-release placement reference correction; none is claimed as model success.

Deferred as specified: labelled part-targeted action/schema support,
reconstruction, homogeneous/heterogeneous benchmark comparisons, broader task
families, batching/reset performance work and real hardware.

## Requested video follow-up — 12 November 2026

Added an optional `--video` flag to simulator commands. It reuses RSS VideoWriter,
records every second control frame at scene speed (cube 25 fps, Clawhauser 30 fps),
and leaves recording off by default. No simulation or action settings changed.

Replayed the exact saved nominal cube, shifted cube and Clawhauser model plans,
using their original Task JSON, into outputs/simpact/videos_20261112. All three
passed all measured gates. Input plans, tasks, geometry, executed commands and
initial states were checked against the original confirmations. No model calls.
Videos decoded completely: 407/432/680 frames, 16.28/17.28/22.67 seconds, H.264
with yuv420p. Clawhauser lifted 63.940 mm, settled inside the target, and retained
zero inverted/nonfinite elements and zero table penetration. Its final drift was
17.55 micrometres and peak speed 0.518 mm/s. Existing ten core checks and Ruff
passed; no tests were added. Slurm job 5199251 released its allocation after
recording and video validation. The manifest and worker results preserve evidence.
