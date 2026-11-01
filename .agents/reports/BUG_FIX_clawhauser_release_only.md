# Clawhauser dropping investigation — final status

**Not fixed. No complete pick-and-place passed the existing task and physics checks.**
Continuation ran from 2026-10-27 20:31 UTC to approximately 21:20 UTC.
All compute jobs ended. No commit or push was made.

## Scope and final workspace

Standalone contact gripper and fused labelled Clawhauser only. AIRBOT, materials,
mesh, force caps, solvers, infrastructure and policy work were not changed.
Temporary changes were confined to the two existing demo files. Every rejected
variant is preserved under the evidence root below. Both demo files are now
byte-identical to their state at the start of this turn (05_cycle snapshots):

- clawhauser_pick_place.py SHA256 `a22792514a29d47c915d645b8154366dee2084a10b6181c44b15fbfd8217969a`
- gripper.py SHA256 `9329a92316b82b356e49defa960acde0cc3ad67948cd2b7b8aac388e6514a72a`

Existing uncommitted changes on yiduo/dev remain; this turn leaves no additional
production repair. Diagnostics/reports remain saved. Targeted Ruff lint and
format checks passed on the restored files. No unrelated test suite was run.
Validation was performed by the implementor; it was not independent.

## Supported findings

1. **The toy is already in invalid contact geometry before release.** Original
   saved states show a right-shelf corner inside body tetrahedra at 9.567 s;
   first recorded body inversion is later, at 12.033 s. Native force export at
   11.5 s shows the right shelf pressing downward on body vertices (~20.96 N),
   while the table supports tail vertices. These are native instantaneous
   contact-law evaluations, not validated actuator forces or a global force
   balance. Kinematic fingers have no effort cap.
2. **Ring/body self-intersection precedes inversion.** Strict intersections
   between nonadjacent exterior ring/body triangles are absent at 9.0 s but
   present at 9.5 s in the original run (76 crossing face pairs). Slower and
   wider arcs also develop crossings before the recurring body cell 4856 flips.
   Shared-vertex face pairs and tangential endpoint hits are excluded. This
   supports a missing self-collision mechanism; a working causal repair was not
   established because tested native self-contact configurations failed settling.
3. **Point-only rigid/soft detection misses relevant shelf features.** At exactly
   the saved 11.5 s pose, stock full-surface queries add 41 penetrating edge/face
   records on the right shelf against exclusive body vertices, with maximum
   computed contact depth 1.899 mm. This demonstrates missing coverage; it does
   not establish that enabling that option fixes the complete task. The dynamic
   test instead invalidates the existing pickup.

## Experiments and verdicts

Each hypothesis and prediction was recorded before its change; the complete
chronological records and exact reproduction details are in
[investigation_history.md](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only/investigation_history.md).

| Hypothesis / intervention | Controlled comparison | Result / verdict |
| --- | --- | --- |
| H0: retained non-ring force | Original native CONTACT_F export, no motion change | Right shelf presses body before opening; confirmed loaded interference, not simply a ring sticking to pads. |
| H1: point contacts miss shelf features | Identical saved q/body poses, point vs native full-surface queries | Extra penetrating edge/face contacts found; coverage concern confirmed. Dynamic repair unverified. |
| H2: delayed right opening prolongs loading | Only right opening time advanced | Clears right-body load ~1.33 s earlier, but 5 bad recorded frames remain. Partial, not a fix. |
| H3: lay-down arc instead of vertical crowding | Midpoint y/z trajectory only; pickup prefix exactly equal | Near target, but 10 bad samples and severe ring/table pinch. Rejected. |
| H4: add 30 mm endpoint clearance | Same arc and jaw times; only vertical endpoint changes | Both fingers clear by 13.8 s; 27.5 mm XY error at 14 s. Ten bad samples, so rejected. Withdrawal/settling untested. |
| H5: slower arc and simultaneous opening | Explicit two-phase bundle from H2/H4 evidence; full 18 s scene | 79 bad samples, peak 4 inverted tets; release and target checks fail. Rejected. |
| H6: native self-contact | Same H5 path; flag only, then separate 2 mm rest-exclusion check | Flag-only case fails before touch; rest-exclusion case has 29/30 bad samples, peak 16 tets. Not a usable configuration; no custom filters added. |
| H7: extend arc sideways by 50 mm | H5 settings; only sideways radius changes | 15 bad samples through 14 s; same internal body cell flips. Rejected; stop trajectory tuning. |
| H8: stock full-surface detection | Original 05_cycle path, mesh/materials and native solver settings | Pickup fails; guard stops at 6.5 s. One bad sample, peak 2 tets. Cannot validate release; rejected. |

No stiffness, Poisson ratio, density, damping, friction, contact stiffness, force
cap, success threshold, attachment or prescribed object motion was introduced
or changed. Native self-contact and full-surface flags were tested separately
and removed after failure. No Newton/RSS library source was edited.

## Numerical coverage and limitations

Checks sample the saved 30 Hz control frames, not every solver substep. Existing
zero-inversion, pickup/transport, XY target, release, table support, withdrawal
and settling requirements were retained. A release-only clip cannot establish
complete-task success. The complete 18 s arc run failed; no new successful video
exists. The restored original configuration still fails placement.

Source remains 1,840 vertices / 6,850 labelled tets. Body/tail E=0.5 MPa,
nu=0.45, rho=300 kg/m3; ring E=5 MPa, nu=0.49, rho=1100 kg/m3. Newton
1.8.0.dev0, commit d960ec3b8be23d7acac01c0ba10b8d04dca8b549; Warp 1.18.0;
stock SolverVBD, 30 Hz, 80 substeps, 48 iterations, native tiled solve;
GH200 ARM64, CUDA toolkit 13.4 / driver 13.0. Material estimates are preserved,
not experimentally validated or ruled out as a contributing factor.

For static full-surface queries and H8 only, Warp's public per-module
optimization_level=1 compiled Newton's face-contact kernel in about 10 seconds.
The default setting returned no face result after 26m19s and was canceled.
This compiler-only option changed no collision algorithm or solver code. It is
not retained in the restored demo, and parity with a completed default-compiler
query was not measured. The native self-contact interaction margin was 2 mm,
with zero extra detection gap; this was not a 200 mm default-units mistake.

## Exact remaining failure and smallest next step

With the original passing pickup configuration, lowering permits ring/body
interpenetration and shelf/body interference before opening. Trajectory changes
can get the body onto the table and free the fingers, but do not keep the source
volume valid. Enabling more complete native rigid/soft detection changes the
pickup itself; enabling native self-contact with the tested settings destabilizes
the untouched toy before grasping. There is no verified release-only patch.

The smallest next experiment is a **short approach/closure check under stock
full-surface detection**, using measured clearance of the whole loop against
both pad and shelf, before any lift. This would reopen the grasp configuration,
so it was not pursued in this release-focused turn. Keep the same labelled mesh,
materials and solver. Native volume self-contact setup remains a separate
unresolved issue; do not substitute a custom solver or collision implementation.

## Evidence

Root: `/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only`

- [Numerical continuation summary](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only/continuation_results.json)
- [Original matched-force comparison](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only/comparison.json)
- [Full-surface static query records](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only/point_vs_surface_queries_o1.json)
- [Original ring/body crossing measurements](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only/ring_body_surface_crossings.json)
- [Additional crossing measurements](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only/additional_self_crossings.json)
- [Closest release-only clip — FAILED integrity, 8.5–14.0 s](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only/clearance30/overview.mp4)
- [Closest release-only diagnostics](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only/clearance30/diagnostics.json)
- [Complete 18 s arc — FAILED task/physics](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only/slow_arc_cycle/overview.mp4)
- [Complete arc numerical result](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only/slow_arc_cycle/result.json)
- [Native full-surface attempt — stopped by pickup guard](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/debug/clawhauser_release_only/full_surface_cycle/result.json)
- [Best prior complete-motion failed baseline](/nobackup/proj/disk/ehpc-dev-2026d08-170/personal/yiduo/robosimstudio/outputs/stock_clawhauser/05_cycle/overview.mp4)

Per-run folders contain saved states, source snapshots where applicable,
diagnostics and videos. Canceled self-contact case retains printed initial
samples and source snapshots, not a complete state trace or complete video.
`run_release.py`, `query_surface.py`, `check_geometry.py`, and
`check_self_crossings.py` are isolated diagnostic scripts under ignored outputs.
Jobs4521159/4521815 were canceled deliberately; all other continuation jobs
completed with exit0. Successful execution is separate from task success.
