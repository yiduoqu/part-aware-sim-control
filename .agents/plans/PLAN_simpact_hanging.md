# SIMPACT Clawhauser hanging — implementation and experiment

Requested 12 November 2026: pull Hanson's new main, then try a complete
SIMPACT-style VLM optimization campaign for hanging the toy's ring on the rod.
Work autonomously, preserve the simple integration, and test only core behavior.

## Source and constraints

- Integration branch: yiduo/simpact-integration-20261110, fast-forwarded from
  2fe88f4 to origin/main 3393ff5 (Hanson heterogeneous-tests 650015e).
- New upstream changes include arm MOUNT=(-0.05,-0.42,0.232), tail grasp options,
  trajectory joint angles and a revised bottle scene. Build the default physical scene
  and allow either supplied loop or tail pickup; preserve the fully deformable toy
  and its materials.
- Reuse clawhauser_on_rod.build_scene defaults, including the 6 mm rod, original
  label identities, 60 Hz/20 substeps/Chebyshev160, contact/self-contact settings,
  force limits and existing AIRBOT hand. Native example motions are a separately
  labelled reference; its measured turn/thread callbacks are not VLM actions.
- Newton remains 5cd406a1 after a fresh fetch; pinned SIMPACT 692aab2b stays fixed.
- Existing saved Codex login and exact gpt-6-sol. Ten full initial plans, at most
  three one-plan refinements, one fresh confirmation only after planning success.
  No Google key, provider substitution or extra optimization campaigns.

## Implementation

1. Run the unmodified Hanson loop-grasp example on an allocated ARM GH200, with
   video, and preserve its result/diagnostics/configuration. Already submitted
   under outputs/simpact/hanging_hanson_reference_20261112.
2. Add task clawhauser_hang. Reuse the native builder, expose rod endpoints/axis,
   ring inner center/normal, source settings, pad offset and common fresh pose.
3. Add compact per-control-frame hanging measurements using Hanson's penetration
   and mixed-stiffness support calculation: loop/rod contacts, upward rod support,
   finger contact, actual hole pose, engagement and collision depth. Keep the
   existing mesh-quality substep latch and record waypoint feedback.
4. Use existing lowercase six-axis move and gripper_control for both proposal and
   refinement stages of this task; preserve placement profiles. Turns must be
   physically commanded, with reachable orientations and consistent TCP/pad
   geometry. No attachments, implicit grasp, auto alignment or scripted thread.
5. Hanging success requires rod support >=90%weight, actual loop/rod contact,
   absolute winding >=0.5 of the ordered material inner-hole contour around the
   rod axis, its entire contour >=10mm inboard of the tip and <=rod length,
   >=20 mm whole-toy table clearance, no final
   finger contact, released jaws, stable final half-second, valid mesh and the
   reference's <=0.5 mm contact penetration. Freeze thresholds before the model
   campaign; do not weaken them to obtain success.
6. Add hanging-specific propose/refine/verify prompts so airborne rod-supported
   success is accepted. Model receives measured failures and supplies every
   action of every independent fresh-scene sequence.

## Core validation and experiment

- Reuse existing ten offline checks. Add only focused pose/profile and hanging
  gate checks that reject a toy on the table, a toy still held by fingers, or an
  invalid/disengaged mesh. Ruff on changed files; no broad new test framework.
- Inspect the native reference, then execute an explicit baseline through our
  worker when its pose commands can be derived from the saved reference. Any
  reference feedback used to author that input is labelled separately.
- Obtain our own common initial observation. Execute all ten generated plans,
  verify in order, refine up to three times with accumulated evidence, then
  confirm the first verified success in a fresh scene. Preserve failure if the
  budget is exhausted; the goal is to try the specified campaign honestly.
- Save raw model calls, task/geometry/revisions, plans, measured outcomes,
  screenshots, timing and continuous videos. Distinguish errors from physical
  failures, and reference motions from model-generated solutions.
- Use x86 planner and srun ARM workers. Tie each Slurm allocation to its command
  with salloc --kill-command=TERM so completion/failure releases the GPU.
- Independently review the adapter, gating and final experimental evidence.

## Status

The implementation and bounded experiment are complete. No fully validated
hanging success was obtained. The third refinement executed all 40 actions and
left the toy threaded and released, but failed support, settling and penetration
gates. No confirmation ran. Thirteen focused checks and Ruff pass; independent
review accepted the implementation and final experimental provenance.

Final results: outputs/simpact/hanging_loop_tail_20261112/recovery/result.json.
Complete model plan: recovery/selected_plan.json in that campaign directory.
Final video: recovery/rollouts/attempt_012/replay.mp4. Both campaign GPU allocations
are released. The execution record below preserves intermediate findings.
The publication bundle is [the tracked results report](../reports/simpact_hanging_20261112/README.md),
including the final video, complete plan, measured rollout and campaign evidence.

## Execution record

Native reference job5212884 completed and released its allocation. Native default
loop-grasp run failed: no final rod support, toy fell, max contact penetration
1.74791mm, two BODY inverted element-substeps. This is not a model attempt.
An initial adapter observation completed successfully; the actual campaign will
obtain a fresh observation with the final winding schema before generating plans.

Actual experiment output: outputs/simpact/hanging_loop_tail_20261112. The explicit
baseline input is authored from the saved native trajectory's TCP poses, with
recorded initial settled-grasp compensation. This static baseline has no runtime
turn/thread correction, and is not guaranteed to reproduce a feedback demo.
It is separate from the ten model-generated candidates.

Preflight allocation5213382 was cancelled after6m50s before any model call. It
exposed a controller contract issue: every move interpolated jaw targets from
measured rather than previous commanded opening, briefly unloading a stalled
pinch. Independent source review confirmed the native demo explicitly avoids this.
Hanging moves now preserve the commanded jaw target; placement behavior is kept.
Measured and commanded openings are recorded separately. The cancelled baseline
is retained in outputs/simpact/hanging_codex_20261112 and is not a model attempt.

The final campaign exposes both loop and tail source grasp locations and distinct
settled part shifts, as supported by Hanson's latest code. Choosing a pickup part
does not alter the physical scene; the final support must still be the loop on
the rod. All original acceptance thresholds and the10+3budget remain fixed.
A fresh observation and28-action explicit loop baseline will run with the final
controller before the model campaign. The x86 planner then runs automatically
with gpt-6-sol and all workers record continuous video. Code snapshots and
experiment metadata are saved alongside each run.

Final campaign allocation: Slurm 5213853 on n127; baseline running.

The explicit baseline completed: lifted160.8mm, released, then fell. Rod support0N;
peak penetration1.56344mm, peak finger readback27.677N, zero inverted/nonfinite
tet-substeps. Its valid=false comes from penetration; no thresholds changed.
GPT-6 sol proposal generation has started using the final fresh observation.

Candidate0 lifted but its pure world-X turn was rejected by shipped IK (51.42deg
angular residual). Independent review found no FK/frame mismatch; the native
coordinated-turn baseline passed at0.0114deg maximum residual. This is a rejected
seeded route, not proof that every IK branch is unreachable.

A minimal planner feedback correction now retains known InvalidPlan partial
traces and requested complete plans as explicitly rejected artifacts. Rejected
counts remain separate from completed simulation counts; runtime failures stay
fatal, and physical/IK gates and the3-refinement budget remain unchanged.
Thirteen focused checks and Ruff pass, including an all-rejected batch and rejected
refinement followed by success. The active batch uses the original loaded planner;
recovery will reuse the exact saved proposals and initial artifacts, with explicit
cache provenance, so neither a proposal call nor initial execution is repeated.

All ten initial model plans choose the same pure-world-X turn. Refinement memory
now also contains the clearly labelled explicit baseline (not a model candidate),
which supplies measured feasible coordinated-turn poses and its failed physical
hanging outcome. This uses existing evidence, adds no execution or proposal,
and does not change any gate. Invalid initial partial traces are mirrored into
the running batch memory so any refinement can see their actual rejected route.

Guarded continuation Slurm5214671 is queued with afterany:5213853, so it cannot
allocate until the original job terminates. Its reviewed one-off script proceeds
only for the exact all-ten-rejected/no-executable terminal result, reuses the
original raw proposal and ten partial executions, and permits only the original
three refinements and success-only confirmation. No public resume API was added.
If the original job instead completes planning, cancel this pending continuation.

Original batch5213853 terminated after1h39m42s: ten attempts, ten typed
InvalidPlan rejections, no completed model execution. Five loop turns missed
IK by51.4–53.2degrees; tail turns missed by5.67–9.22degrees. Some tail grips
inverted elements; others retained valid mesh. Original runtime error is the old
loaded planner excluding every rejected candidate, not a GPU failure.

Slurm confirms original allocation released. Dependent continuation5214671
started automatically on n127. Every frozen-input/source and plan-identity guard
passed. It reused the exact original proposal response (no API call) and all ten
partial executions (no simulation replay), then began real GPT-6 sol verification.
The existing three-refinement budget and all acceptance thresholds remain fixed.

Refinement1 supplied32 explicit actions and a34mm loop command. Its staged
turn executed; hole-axis winding reached1 outside the free tip. BODY inversions
first appeared during action14 lowering (minJ−1.2065; final638inverted
element-substeps). Insertion jammed before full contour engagement, penetration
at a waypoint reached1.383mm, and the toy fell after release. Withdrawal was
then rejected by IK at19.34degrees. The typed rejected trace consumed the first
refinement and returned feedback normally; no simulator/provider failure.

Refinement2 is running (41actions):35mm grip, segmented lowering, corrected
hole-center alignment, shorter entry steps and divided withdrawal. One further
refinement remains if needed; no additional initial proposals or executions.

Refinement2 threaded the entire ring contour inboard (minimum37.8mm at its last
waypoint), released to zero finger contacts, and retained the toy on the rod.
Upward rod-loop support was1.298N immediately after release (weight1.1766N),
then0.923N at the last partial snapshot. Whole-toy minimumZ remained0.81195m,
59mm clear of the bench. One BODY element-substep had inverted during the turn;
segmented lowering added none. Contact penetration during insertion/seating
also exceeded0.5mm. The final withdrawal move was rejected at10.71degrees,
after action38. This is a physical partial hang, not a validated complete sequence.
The third/final refinement is being generated; no extra optimization budget.

Runtime module-path check on the allocated ARM node confirms RSS imports from
this updated controller checkout and Newton from the pinned sibling checkout.

Refinement 3 supplied 40 actions with a 37 mm grip, 4 mm higher entry and a shorter
7 mm seating motion. Every action executed, including release and withdrawal,
followed by the default 2 s settling. The ring remained threaded (absolute winding 1;
whole contour >=36.2 mm inboard), fingers were clear and the whole toy remained
57.5 mm above the bench. All three parts had zero inverted/nonfinite element-
substeps. Final upward loop support averaged 1.014 N, 86.2% of 1.177 N weight against
the 90% requirement. Peak contact penetration 1.477 mm exceeds 0.5 mm; final drift
8.84 mm and peak speed 53.15 mm/s exceed 3 mm/10 mm/s. Those three failed gates keep
success=false. Peak finger readback 21.8 N passes the allowed 30 N; the raw VLM
verdict incorrectly used the 20 N actuator setting as the readback threshold.
The future verifier prompt now states the native tolerance explicitly. This
post-campaign wording correction changes no saved prompt, result or physical gate.

The campaign consumed exactly 10 initial candidates and 3 refinements: 12 rejected
executions and 1 completed execution, with zero confirmation executions. Actual
Codex calls were 1 proposal, 3 refinements and 13 verifications (17 total), all exact
gpt-6-sol with retry 0. The guarded continuation's cached proposal was not an API
call, and no initial candidate was re-executed. Refinement memory included the
separately labelled failed static reference baseline; this is not a claim of
planning without reference evidence.

Final independent review matched raw model plans, all 40 executed action endpoints,
the selected plan, cached proposal/error bytes and physical source hashes.
All 13 fresh physical initial states match the common observation within 1.20e-7
(tolerance 1e-6). Physical adapter source and native scene settings stayed fixed
through the campaign. The final video decoded all 1220 frames at 30 fps/40.67 s.
Slurm 5214671 completed 0:0 after 1h39m06; original 5213853 and the continuation
are absent from the queue, with allocations released. The original terminal
error is the documented planner feedback issue, not a cluster failure.

Core validation passed 13 focused unit checks, Ruff lint/format and git diff
whitespace checks. The remaining physical work is reducing insertion/seating
penetration and allowing stable load transfer within the unchanged thresholds;
no extra optimization campaign or additional physical test was started.
