# Merge Hanson development into main

Status: revised plan only, 2026-10-29. User selected hanson/dev as the default
integration base. This supersedes the earlier stock-first recommendation.
No source changes, commits, branch checkout, push or merge have been performed.

## Branch state inspected

- main: eb3d46a839bf8d9b8df6e9c3f0205e141f5b3f7f.
- yiduo/dev: 6d60d7ba50801fae112059877f1f8afbc6f75182, one commit beyond main.
- hanson/dev: 7476ea2777c8de80b58f89a92e583503e350979c, contains yiduo/dev
  and 53 further commits.
- Local uncommitted work: the stock example, gripper helper, README and three
  bug reports; these changes are not in either remote branch.
- Assets, heterogeneous.py and skills are identical on the two committed branches.

## Default behavior decision

Use Hanson as the candidate for main, retaining its parallel claw, scene
controllers, backend selection and existing solver correction. Do not replace
these with the local kinematic gripper or impose the local 80/48 solver settings.
The earlier no-new-solver-redesign constraint remains: reconciling existing
code is authorized for planning; developing a larger solver is not part of it.

## What to retain from the uncommitted local work

| Local item | Treatment |
| --- | --- |
| Exact stock drop/release source | Preserve on a separate checkpoint/archive branch; do not overlay it onto Hanson by default. |
| Three bug reports | Keep as historical evidence, labelled stock-Newton/kinematic-gripper results. Retain video/state paths and source hashes. |
| Videos, saved states and numerical results | Preserve under existing outputs; do not add large generated files to Git automatically. |
| Release versus mesh-integrity interpretation | Retain in documentation and result interpretation. Hanson already reports released, local_inversion, quality and success, so avoid a second reporting implementation. |
| Ring contacts versus initial grasp patch | Retain the semantic correction where an active example still calls patch-only contacts whole-ring contacts. Use source label 2 for the ring and the apex patch only for alignment. |
| README limitations/provenance | Retain accurate limits, exact tested configuration and clear separation of historical versus current results. Rewrite around Hanson defaults rather than copying stock timings. |
| New stock release order and independently tipping finger | Archive with the stock reference; do not port onto the level parallel claw whose pads cannot rotate. |
| One-second fit_grasp, grip gap, 80 substeps/48 iterations, disabled self-contact | Archive as configuration-specific stock tuning, not new global defaults. |
| Optional --lower-before-release and 12/14 s schedules | Archive with the stock reference; not required by Hanson's current scene controllers. |
| Omitted decorative slabs workaround | Do not propagate as the solution; Hanson already fixes collide=False for particle contacts. |
| Angular velocity correction | Preserve Hanson's existing trajectory_velocities implementation; it already covers this, so no duplicate patch. |

The material estimates, labels and fused object representation were never
changed locally and already agree with Hanson; no new representation merge
is needed.

## Proposed integration sequence

1. Preserve the exact local work in a dedicated checkpoint before changing the
   working branch. Keep the three reports with it. This is a future implementation
   step, not a commit performed by this planning turn.
2. Create an isolated merge worktree from the current hanson/dev tip; leave the
   present dirty checkout intact. Do not merge the whole local stock patch.
3. Make only selective documentation or clearly applicable correctness changes:
   whole-ring contact labelling where needed, truthful result summaries and
   current configuration/provenance. Preserve Hanson's gripper API; this avoids
   the stock helper's textual/API conflicts entirely.
4. Reconcile dependency setup: pyproject/lock currently require editable
   ../newton, while the correction documents revision 454fc3e and the local
   reference used d960ec3. Choose and record one reproducible source revision
   for the candidate. Do not silently upgrade Newton or assume the shared
   1.8.0.dev0 label proves identical implementations.
5. Clean current README claims: standing-start success counts cannot establish
   success of today's lying-start scenes. Distinguish duplicated 60 fps output
   frames from true added temporal samples. Preserve failed/limited outcomes.
6. Run the repository-required fast/lint and applicable library checks, focused
   custom-backend tests and current scene smokes on allocated compute. Run full
   rod/drop/cup cycles only after the intended grasp executes. Evaluate the
   candidate on its actual Hanson defaults, not the local stock configuration.
7. Confirm result success flags agree with all-substep quality results; for
   example pick_drop currently reports local_inversion using substep quality
   but success checks frame-sampled max_inverted. Resolve any actual mismatch
   with a small predicate correction, not a new quality framework or relaxed gate.
8. Prepare a PR from the validated Hanson-based integration branch into main.
   Record exact revisions, retained local items, test evidence and known scene
   limitations. Merging main is outside this plan-only request.

## Boundaries

- Preserve Hanson's existing work rather than reimplementing its controllers.
- No global import of archived custom-backend work from yiduo/archive-clawhauser-backend-20261027.
- No new solver rewrite, batching, policy interfaces, material optimization or
  AIRBOT development as part of reconciliation.
- Report a concrete compatibility or runtime blocker if it demands larger work.
- The local stock drop proved release only for its own independently tipping
  fingers. It is evidence, not a replacement for Hanson's parallel-claw results.

## Inspection coverage

History, source differences, temporary three-way merge, helper callers, asset
identity, dependency configuration and result predicates inspected. No simulator
or test suite was run in this planning audit. Only this plan was revised; the
three existing source/doc edits, reports and generated evidence remain intact.
