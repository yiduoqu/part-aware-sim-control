# Contact merge: one combined validation

## Current configuration — user update on 2026-11-03

The user subsequently requested: "replicate his default coarse translation solver". The active candidate now restores Hanson's published pick/drop runtime from RSS `d50f86f`: SolverVBDCoarse with its common-translation block, external RSS collision calls, the original edge tolerance and 20-second duration. Commit `ac28ac3` is reverted. Its prior implementation and evidence are preserved on `yiduo/archive-chebyshev-merge-20261103` at `3f088be`.

This supersedes the Chebyshev selection, owned-pipeline integration and 1e-12 scene override in the original plan below. The source mesh/materials, 20 x 12 stepping and configured force limits remain Hanson's. Newton remains `3d3f9e4c`. No new solver or simulation is added in this restoration. The exact published configuration was already run for 600 frames in job 4860245; see [baseline comparison](../reports/BUG_FIX_hanson_baseline_20261103.md).

The acceptance criteria remain those below. This known baseline lifts but loses the toy at 2.57 seconds and does not pass the complete task. Restoring the original example also restores its native result format; that flag does not replace the stricter assessment archived with the comparison. Both main branches remain unchanged.

## Original approved plan and initial implementation

User-authorized scope: merge Hanson's Newton and RSS work, retaining narrow fixes and historical evidence; validate once. No parameter sweeps, new solver, AIRBOT, or broader test infrastructure.

## Sources and merge resolution

- Newton main `1a640e4b5e6ce9e530b359ea1d99d07600703198` fast-forwards to `hanson/sticky-contacts` at `3d3f9e4c13fd0e60e7e53a9870946c0983d1d14c` (both upstream contact-history ports plus tracker reset fix).
- RSS main `f1b648c1d987aee080e2eff742752e9f64048fab` merges `hanson/vbd-improvements-3` at `d50f86f51b31a1fbc42844ef78a4b851eb693159`.
- Choose Hanson scenes, parallel gripper, Chebyshev and coarse implementation. Keep existing shared coupled infrastructure dormant. Preserve original checkouts and unrelated untracked AIRBOT work.
- Carry historical report and saved evidence from `73e0601`, not its expensive scene settings or ADMM/friction routing edits. Archived H15 evidence is not validation of this merge.

## Narrow integration

Connect the existing sticky collision pipeline to a single VBD solver, use its owned contacts for diagnostics, and avoid duplicate collision calls. This enables both rigid-soft and self-contact history through the existing Newton API. Keep other solver routes unchanged.

The standalone pick/drop uses one SolverVBDChebyshev, 20 substeps x 12 iterations, rho=0.9, gamma=1, start=0, restart=False, viscosity 5000 Pa.s, original source mesh/materials/labels, 5 N jaw caps, existing 200 N carrier cap, vertex-only rigid-soft detection and X carry. These are explicit integration settings, not a claim about unpublished Hanson runs. Keep scene-level particle_edge_parallel_epsilon=1e-12 for mesh-scale edge queries. No adhesion or grasp attachments.

## One acceptance run

Run 300 control frames at 30 Hz (10 simulated seconds): close, lift, hold, carry, open at 5.3-5.7 s, then settle. Reuse existing per-frame and substep diagnostics. Assert actual solver/configuration/import paths and both contact-history paths before stepping.

Acceptance: whole-object clearance >=20 mm during hold/carry; retained until opening; X travel 80 +/-20 mm over 3.5-4.7 s; zero final finger contacts; final half-second table clearance -0.5..2 mm; peak frame-to-frame vertex-centroid speed <=5 mm/s in that window; no substep inversion/nonfinite state; recorded penetration <=0.5 mm; jaw drive <=5 N. Save one video, states, numeric results, source revisions and separate stepping/readback/render timings.

After success publish Newton main then RSS main. On failure preserve and push merge candidates, report exact failed criteria, and do not rerun or tune. User's one-run scope overrides generic broad test/rerun guidance in repository skills.
