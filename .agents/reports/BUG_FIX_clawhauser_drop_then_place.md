# Clawhauser: clear the right finger, then lower for placement

## Scope and baseline

2026-10-28. User prioritizes ordinary contact-based release; impact inversion is
recorded separately, not treated as a sticking gripper. After confirming drop,
try simple placement. Stock Newton/RSS only, same labelled 1,840-vertex /
6,850-tet object, source material profile, geometry, friction, actuation and
physics settings. No AIRBOT, solver changes, attachments, object motion commands,
or general infrastructure. Any necessary large solver change requires reporting
back first. Existing fingers remain kinematic and uncapped.

Source: RSS yiduo/dev with existing uncommitted demo changes; Newton d960ec3b,
1.8.0.dev0, Warp 1.18.0, VBD 48 iterations, 80 substeps at 30 Hz on allocated
GH200. Existing reports record opening-only retained support, followed by a
successful open-and-right-tip drop. Lowering previously created shelf/body and
ring/body interference before release. Preserve those findings and artifacts.

## Plan and acceptance

1. Repeat the exact current open-and-tip drop without editing the trajectory.
   Reuse the passing pickup/transport guard, contact/COM diagnostics and 30 Hz
   mesh checks. Confirm sustained finger separation and gravity-driven flight;
   record post-impact integrity separately. Compare saved poses to the earlier
   accepted release. Save a full video and numerical results.
2. Only if release repeats, try one small trajectory change: lower the suspended
   object partway before the same open-and-tip release. Select clearance from
   the actual saved hanging geometry to avoid replaying the old deep-lowering
   crowding. Start with a short phase-specific check if useful. Do not add load
   controllers, new collision modes or material/solver changes.
3. Continue only against concrete observed failures. A placement trial must
   actually release, clear the fingers, come to table support and settle; report
   target position and whole-episode integrity independently. Do not call a
   near-table drop a fully supported place or silently discard bad mesh samples.

Evidence: outputs/debug/clawhauser_drop_then_place/. One bounded read-only agent
reviews the existing release evidence and minimal placement strategy. Main agent
owns source edits and simulations. No commit/push unless requested.

## H0 — existing open-and-tip drop reproduces

Prediction: exact current source repeats pickup and transport, clears all finger
contacts after tipping, and gives at least three contact-free airborne frames,
COM descent >10 mm and mean acceleration error <20% relative to gravity. No
pre-impact inversion. Subsequent impact inversions do not refute release.
Control: prior 02_right_tip run; no parameter or motion changes. Both current
source files copied to 00_drop_repeat before running. This tests repeatability
of the original failure remedy, not robustness to unseen poses.

Verdict: CONFIRMED. Job 4576101 completed the full 10 s on n550. Every saved
particle and rigid-body pose is bit-identical to the previous 02_right_tip run.
Release at 8.5667 s, five contact-free samples, COM fall 144.6 mm, gravity error
7.25%, no pre-impact inversions. Same two post-impact tets invert briefly.
Airborne release passes; whole-episode integrity remains reported as failed.

## H1 — tip before right-jaw withdrawal avoids sideways dragging

Recorded before implementation; conditional on H0 passing. Previous saved
release moves mass COM from x=0.6657 m at 6.5 s to 0.7447 m at release, while
the right jaw opens outward. Only then does its ledge tip. Hypothesis: the same
tip can free the ring at the closed right-jaw X, reducing this lateral motion.

One variable: right outward opening moves from 7.0–7.5 s to 9.0–9.5 s. Left
opening stays 7.0–7.5 s, right tip stays 8.0–8.5 s; pickup, translation, height,
geometry and all physics unchanged. Extend recording to 12 s for final clearance
and settling, without changing earlier commands. No new controller mechanism.

Predictions: contact-free descent occurs before right outward opening at 9 s;
the right jaw stays at its closed X until then; sustained zero finger contact
afterward; smaller lateral COM displacement from 6.5 s to first free-flight
sample than H0. Existing flight/acceleration and pre-impact integrity gates
remain. If the ring remains on the tipped finger until outward opening, the
claimed release mechanism is insufficient; do not count a later dragged drop as
confirmation. Post-impact inversion is still reported separately.

Feature scope: existing airborne branch in gripper.py plus episode duration in
the existing standalone demo. Preserve H0 source snapshots and video. Compare
saved states through the first changed right-jaw opening command (~7 s), then
check actual right X, angle and contacts rather than only a final result flag.

Verdict: CONFIRMED in this scene. Job 4576163 completed 12 s. Prefix through
7 s exactly matches H0. Right X is constant until 9 s; first zero-finger sample
8.5667 s and no later recontact. Six contact-free airborne samples, COM fall
179.0 mm, gravity error 5.47%, no pre-impact inversions. Lateral COM displacement
from transport to release falls from 79.0 mm to 47.8 mm. Three tets invert
briefly after impact; this does not change the release verdict. Keep this
release order for lowering tests. It reduces, but does not eliminate, lateral
motion because the toy can swing/roll around the retained finger.
Read-only boundary-triangle/box checks find no physical crossings during all
six flight samples. Whole-object AABBs slightly overlap one finger AABB in the
first sample (0.090 mm), so unlike H0 there is not an all-frame AABB-separation
proof. Do not mistake overlapping bounds for actual surface intersection.
An independent read-only reviewer confirmed H1's timing/COM/contact verdict.

## H2 — stop lowering above the table, then use the verified release order

Prediction: lowering 120 mm over 6.5–8.5 s, with both jaws closed, leaves
approximately 80 mm bottom clearance and avoids the previous loaded placement
interference. Delay left opening, tip and right withdrawal by exactly 2 s so
the descent finishes before release; retain relative release timings and height
thereafter. This necessary timing/height bundle tests the same release at a
lower position. No other settings change. Extend recording 12→14 s for the
extra descent. This is explicitly a low-height drop, not supported placement.

Reuse existing release_lower_z in the airborne trajectory, defaulting to zero
in the standalone scene, exposed by one --lower-before-release argument.
AIRBOT/default non-airborne trajectory stays unchanged. No new controller or
solver class. Acceptance: pickup prefix through 6.5 s unchanged; no table contact
or inversion before release sequence; first sustained zero-finger contact before
delayed right withdrawal at 11 s; no recontact; final table support and body drift
below original 20 mm settling limit. Report target error rather than gate on an
unrequested landing spot. Record whole-episode zero-inversion separately and
do not weaken it. When lowering is used, an airborne-flight sample count cannot
serve as the placement predicate; report the lowered release and supported-
before-release observation separately. A short flight alone is not full support.

Verdict: CONFIRMED for low-height release. Job 4576232 completed 14 s. Pickup
prefix through 6.5 s is bit-identical to H1. Minimum bottom clearance after
descent and before opening is 82.55 mm, no nonfinger contact or inversion before
opening. First zero-finger contact 10.5667 s, before right withdrawal at 11 s;
no recontact. Three contact-free samples, gravity error 2.84%, first table
contact 10.6667 s. Final half-second table contact throughout, body drift
0.261 mm, mass COM speed 0.687 mm/s. All-episode integrity fails (peak six
post-impact inverted tets, one remains inverted at the end). This does not
refute release, but the run is not a clean-physics placement result.

## H3 — near-table release with descent stopped before crowding

One physical change: lower distance 120→200 mm, identical 14 s timings and
all other settings. Measured 82.55 mm pre-opening clearance in H2 predicts
about 2.55 mm remaining at 200 mm descent, near the 2.7 mm detection envelope.
The gripper stops descent at 8.5 s; left opens 9–9.5, right tips 10–10.5,
right withdraws 11–11.5. Unlike the old 275 mm lowering, this does not continue
commanding descent to force the suspended body flat. Surface contact may precede
finger release; do not call candidate counts a measured support force.

Predictions: successful original pickup; release before right withdrawal with no
later recontact; table contact starts at/before separation and persists after
withdrawal; final 0.5 s body drift <20 mm and no remaining finger contacts.
Report pre-release and post-impact mesh integrity separately, retain whole-episode
zero-inversion flag. Do not make any impact-related solver/material changes.

Read-only scoring refinement before execution: preserve the original airborne
flight predicate unchanged and add a separate lowered-release motion predicate
(complete pickup, finger-free final window, table contact and original <20 mm
drift limit). Near-table placement need not have three airborne frames. Overall
success still also requires zero recorded inverted tets. Record table contact
before finger separation explicitly, and target error as information only,
consistent with the user removing a precise landing spot requirement.

Verdict: CONFIRMED for near-table release and final settling, not precise
placement or clean all-episode FEM. Job 4576341 completed 14 s. Prefix through
6.5 s exactly matches H2; pre-opening bottom clearance 2.561 mm, zero inversions
before opening or finger separation. Table contact begins 9.0 s; sustained finger
separation 10.7333 s, before right withdrawal at 11 s; no later recontact. Final
table contact throughout the final half-second, body drift 0.0435 mm, COM speed
0.264 mm/s. No contact-free airborne interval is expected because the table
already contacts the toy. Four bad mesh samples only AFTER finger release,
10.9333–11.1 s, peak three inverted tets, minimum J -0.3236; none at the end.
Target error is 159.5 mm: this remains an untargeted lowered release and settle,
not accurate pick-and-place. Do not tune impact or target tracking in this turn.

## Final verification / read-only scoring corrections

No further physical intervention is planned. Repeat the final default drop and
200 mm lowered release on final code in fresh GPU allocations. These
verify reproducibility and final code paths, not robustness across poses/assets.

Independent review found three analysis-label issues. Rename the table-contact
time to first_table_contact_after_transport_s (it can precede opening); use the
start of the final uninterrupted zero-finger interval as sustained release;
use strict table-time < release-time to distinguish same-frame ambiguous order.
Also report max inversion before finger release, because first table contact in
a supported lowering run is not the same event as post-release impact. These
are read-only results computed from existing diagnostics; physics/trajectory
is unchanged. No additional measurement, solver or controller infrastructure.

Verdict: CONFIRMED. Jobs 4576416 (drop, 360 frames) and 4576417 (lowered,
420 frames) both completed with exit 0. They ran in separate allocations on
host n101. Every saved particle and rigid-body pose exactly reproduces H1/H3
respectively (maximum difference 0); both source files exactly match the final
run snapshots, and all states are finite. Drop video fully decodes; the lowered
video contains 420 samples and reports no encoding error. Main agent inspected
release and final-support images, and a separate read-only reviewer checked
source, hypotheses and final numerical evidence. Simulation execution was
performed by the main agent, not an independent runtime validator.

## Final outcome

**Primary release bug fixed for this scene.** The left jaw opens, the right
finger tips while still at its closed X, then withdraws after the ring separates.
Opening the right jaw before tipping contributed to sideways dragging. The
controller only commands finger poses; the object follows native contact,
gravity and deformable dynamics, with no attachments or prescribed particle
motion. Two matched runs of each final release mode reproduce exactly.

| Result | Default airborne drop | 200 mm lowered release |
| --- | ---: | ---: |
| Sustained finger separation | 8.5667 s | 10.7333 s |
| Right-jaw withdrawal starts | 9 s | 11 s |
| Later finger recontact | none | none |
| First table contact after transport | 8.7667 s | 9 s |
| Recorded inversions before release | 0 | 0 |
| Final-half-second body drift | 0.0964 mm | 0.0435 mm |
| Target error (informational) | 92.2 mm | 159.5 mm |
| Correct release motion predicate | pass | pass |
| Whole-episode mesh integrity | fail, peak 3 after release | fail, peak 3 after release |

The lowered case completes basic lower/release/withdraw/settle, with geometric
table contact before release and table support after the fingers clear. Table
load share before release is not measured. This is not accurate targeted
pick-and-place, and the whole-episode success flag remains false. No transient
mesh failures were hidden or relaxed. The lowered case's four bad samples occur
10.9333–11.1 s, after separation, and all tets are positive again at the end.
Impact/mesh tuning and target tracking were deliberately not pursued further.

**Retained change scope:** existing gripper trajectory release order and optional
descent using its existing release_lower_z field; existing standalone demo CLI,
duration, and separate release/settling/quality summaries; its README. No Newton
or RSS library changes, custom solvers, new controller framework, material/mesh/
friction/force changes, AIRBOT work, or general infrastructure. Fingers remain
kinematic and uncapped. Checks remain 30 Hz, not every internal substep. All six
owned simulation jobs completed and none remains active. Ruff lint/format and
git diff --check pass. No commit or push was made.

## Final evidence and reproduction

Root: outputs/debug/clawhauser_drop_then_place/

- 04_final_drop/: full 12 s overview.mp4, result.json, diagnostics.json,
  states.npz, manifest.json, source snapshots and validation.json.
- 05_final_lowered/: corresponding full 14 s recording and all numerical files;
  validation includes exact source hashes, trajectory repeat comparison,
  no-recontact check, inversion times and video sample count.
- 03_lower_200mm/: earlier identical motion plus table_contact.png, released.png,
  settled.png and placement_assessment.json.
- 00–02: preserved baseline and intervening trials; no failed evidence deleted.

On an allocated ARM64 GPU node, activate the existing Newton environment, put
the RSS and Newton roots on PYTHONPATH, return to the RSS root, and run:

```bash
python -u examples/10_heterogeneous/clawhauser_pick_place.py \
  --out outputs/clawhauser/drop --render-every 1
python -u examples/10_heterogeneous/clawhauser_pick_place.py \
  --lower-before-release 0.20 \
  --out outputs/clawhauser/lowered_release --render-every 1
```

Airborne release success is a dedicated result field; lowered release success
is separate. Overall success still also requires the all-episode integrity flag.
