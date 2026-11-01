# Airborne Clawhauser release

## Authorized scope / baseline

2026-10-28 local date: use squeeze-a-bug/build-a-feature, implement autonomously
until success or a genuine blocker, without expanding engineering. User now
requests pickup/transport then simply open and let gravity act; no landing target
or pre-release table support. No AIRBOT, custom solvers, infrastructure, material,
mesh, friction or force changes. Existing standalone fingers remain kinematic
and uncapped. Baseline is the restored05_cycle source on yiduo/dev. Source
snapshots copied to outputs/debug/clawhauser_airborne_release/ before editing.
Reuse stockNewton1.8.0.dev0 d960ec3b8be23d7acac01c0ba10b8d04dca8b549,
Warp1.18,80substeps/30Hz/48iterations and the passingpickup/transport.
Main agent handles this small change; no independent validation claim.

## H0 — prior lowering caused interference; airborne opening can release cleanly

Prediction: preserving every motion through6.5s, then holding the midpoint and
opening both jaws7.0–7.5s clears the shelves while the toy is still above the
table. It then has a contact-free downward flight. Earlier findings about
lowering cannot prove this simpler release fails.

Feature: one opt-in airborne_release boolean in existing gripper; pure scene
selects it. Disable lowering, right tipping and withdrawal in this mode only;
retain existing pickup and one-time alignment. Run10s including subsequent
impact; do not add placement gates or require landing in a spot. Existing
30Hz mesh checks remain. Record one additional physical measurement: mass-
weighted COM position/velocity, to distinguish true gravity-driven flight from
retained support or slow numerical drift. No contact-force exporter needed if
all external contact records are absent during flight.

Acceptance before execution: pickup/transport guard unchanged; verify its saved
particle/body prefix against05_cycle. At least3 consecutive contact-free airborne
samples, COM descends>10mm during that run; mean COM acceleration within20% of
modelgravity; sustained final finger clearance; no pre-impact inverted sample.
Record post-impact inversions separately AND retain zero-inversion all-episode
check. Full success requires both airborne release and all-episode integrity.
A target miss cannot fail this task. If impact alone corrupts the mesh, report
release success separately rather than calling it a sticking gripper.

Evidence: full30fps video, original diagnostics plusCOM, savedstates, manifest,
result and exact source snapshots under outputs/debug/clawhauser_airborne_release.
Only retry against a concrete failure. No broad test suite or extra framework.


## H0 result — REFUTED: aperture opening alone does not release this gripper

Job4525683 completes300frames/10s. Pickup/transport particle/body prefix through
6.5s exactly matches05_cycle (maximumdifference0); midpoint remains stationary
afterward. Both jaws fullyopen7.5s, but by10s14right-ringcontactrecords remain,
left0; COMheight1.04855m, toyminimumz0.9513m. No freeflight/tablecontact; zero
inversions in all300recordedframes, minJ0.3997. The toy has moved along with the
rightjaw and remains suspended. This isolates actual retained right-jaw contact
from the previous lowering/interpenetration failures. Exact force split between
pad and ledge is not yet measured; the ledge is a plausible support mechanism,
not a force conclusion based solely on counts.

## H1 — tipping the remaining right support allows the loop to slide off

Prediction: keep identical H0trajectory, including symmetricopening7–7.5s,
then apply the EXISTING rightjaw -90degYtip8–8.5s, holding the midpoint fixed.
The right contact should clear during/after tip and produce contact-free descent
without any attachment or prescribedobjectmotion. Only that angular trajectory
changes; no geometry/force/material/solverchanges. Add counts by the four
existingfinger shapes to distinguish pad from shelf records, reusing the same
contactarray. Counts still are not activeforce measurements. Native angular
velocity is already computed by the existingcontroller.

Also fix two analysis-only details before trial2: remove unusedlocal final;
restrict selected flight to before first tablecontact so a later bounce cannot
stand in for the initialrelease. H0 had noflight, so its result is unaffected.
Do not rerun physics for those scoring/lint fixes. Retain original30Hz integrity
check and report impact separately. Same10s case/video. No new infrastructure.


## H1 result / final status

**Airborne release works in this observed scene. Full-episode mesh integrity
still fails after impact; do not report the entire episode as physically clean.**

Exactly two dynamic trials were run. No material, geometry, friction, force,
collision mode, solver, timestep or iteration changes were needed. No new
solver/controller framework, attachment, object prescription or infrastructure.
Only existing gripper/demo files changed; baseline and both source variants
are preserved in the evidence directory. AIRBOT and archived solver unchanged.
No commit/push was made. Main agent performed implementation and validation;
this was not an independent review. All checks below sample30Hz, not every
substep. Lint/format pass on both finalfiles.

Trial2 job4525948: COMPLETED exit0. Trial1job4525683: COMPLETED exit0. Codec-only
job4526256 verifies media; it does not run physics. All owned jobs have ended.

| Observation | Opening alone | Opening then right tip |
| --- | ---: | ---: |
| Original pickup/transport retained | yes | yes |
| Body lift during transport | 283.3mm | 283.3mm |
| Contact-free airborne samples | 0 | 5 |
| Sustained finger separation | no | from8.567s through10s |
| Inverted tets before first tablecontact | 0 | 0 |
| Peak inverted tets across whole10s | 0 | 2 (afterimpact) |
| Airborne release success | false | true |
| Whole-episode success/integrity | false | false |

**Controlled comparison:** trial1matches original05_cycle particle/body poses
through6.5s exactly; trial2matches trial1 through8.0s exactly. Only the existing
right-jaw angular trajectory changes after8s (plus read-only shape counters
and analysis-only fixes). The midpoint stays fixed within float32roundoff
(<0.1micrometre), opening7.0–7.5s, righttip8.0–8.5s. No lowering/withdrawal,
no table-support/release gate and no desired landingposition. Default helper
mode remains unchanged for other consumers.

**Flight evidence:** five consecutive samples8.5667–8.7000s have no external
contact records. Whole-mesh/finger box AABBs are separated at allfive samples,
minimumgap3.217mm (>0.7mm particle radius +2mm detectiongap); boundary-triangle
box tests also find zero crossings. Thus this is not merely missed point
records while the mesh still passes through a finger. COMdescends144.575mm.
Mean COMacceleration[-0.4371,-0.2675,-10.3034]m/s2, modelgravity[0,0,-9.81],
7.25% vector-relative error, within preregistered20% diagnostic tolerance. This
is approximate native numerical freefall, not a claim of exact momentum balance.

**Causal verdict:** H1confirmed for freeing retained right-jaw contact in this
matchedpair. Aperture opening alone carries the ring along with the rightfinger
and leaves it suspended; rotation frees it. Images show the loop around the
finger end. This supports geometric retention by the finger/pad/ledge assembly,
not a permanent attachment or evidence that a new contact solver is required.
Pad-versus-ledge load allocation was not measured; shape counts alone cannot
assign forces or fully prove the exact local friction/topology mechanism.
One matchedpair does not establish general robustness to other poses/assets.

**Separate remaining failure:** first tablecontact8.7333s; bodyparttet5753
inverts at8.8333s, tets3639/3640 at9.0667s and3640at9.1000s. Three recorded
badframes, peak2tets, minimumJ-0.2521, all afterimpact. No recorded inversion
at10s, allstatesfinite, minimum postcontactvertexz0.750591m (bench0.75m).
The original all-episode zero-inversioncheck remains failed in result.json;
no thresholds were relaxed or badframes hidden. This prevents certifying a
fully intact fall-and-impact episode with the preserved configuration. Fixing
impact response would be separate work on the retained mesh/material/native
contact setup, not a lingering failure to open/release. Do not resume the old
placement/controller/solver project to solve it. No impact tuning performed.

## Retained code and reproduction

`examples/10_heterogeneous/gripper.py`: one opt-in airborne_release boolean,
reuse pickup/path hooks, symmetric opening and existing rightfinger tip at an
earlier time; bypass lowering/withdrawal in this mode.
`examples/10_heterogeneous/clawhauser_pick_place.py`: select mode,10s episode;
replace target/support/settling success criteria with the now-authorized air-
release criteria. Retain original source/material/mass/noattachment assertions,
pickup guard, finitepositions and zero recorded inversioncheck. Add COMposition/
velocity measurement and reuse contactarray for per-shape counts. Record release
success separately from full-episode integrity. No new permanent files/classes
or general controller systems.

Run on allocatedARM64GH200 node with existing Newton environment:
source ../newton/outputs/stage1_setup/activate.sh
Set PYTHONPATH to thisRSSroot and siblingNewtonroot, return toRSSroot, then:
`python -u examples/10_heterogeneous/clawhauser_pick_place.py --out outputs/debug/clawhauser_airborne_release/<new-run> --render-every 1`
StockNewton1.8.0.dev0 commitd960ec3b8be23d7acac01c0ba10b8d04dca8b549,
Warp1.18/CUDA13.4, SolverVBD48iterations,80substeps at30Hz, source-labelled
1840vertices/6850tets, unchanged E/nu/rho/damping/contact settings. Fingers
remain kinematic/uncapped. Native self-contact and full-surface modes remain off
as in the passing pickup configuration; no compiler option override was added.

## Artifacts

- [Full10s video](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_airborne_release/02_right_tip/overview.mp4): H264640x480,30fps, all300frames;
  includes actualimpact rather than hiding the remaining quality failure.
- [Numerical result](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_airborne_release/02_right_tip/result.json): airborne_release_success=true,
  all_episode_mesh_integrity_pass=false and overall success=false.
- [Full diagnostics](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_airborne_release/02_right_tip/diagnostics.json).
- [Saved states](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_airborne_release/02_right_tip/states.npz).
- [Validation/prefix/impactcell identities](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_airborne_release/02_right_tip/validation.json).
- [Flight geometry check](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_airborne_release/02_right_tip/flight_geometry.json).
- [Manifest](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_airborne_release/02_right_tip/manifest.json).
- [Opening-only failed video](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_airborne_release/01_symmetric/overview.mp4).
- [Opening-only results](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_airborne_release/01_symmetric/result.json).
- [Original prefix comparison](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_airborne_release/01_symmetric/prefix_comparison.json).

Final source files byte-match the02_right_tip snapshots; their SHA256values
are recorded in validation.json. All model states and failed trials preserved.
