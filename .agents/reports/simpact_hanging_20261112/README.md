# SIMPACT Clawhauser hanging campaign

The bounded experiment completed without a fully validated hanging success.
The last model plan executed all 40 actions: approach, grasp, lift, coordinated
turn, alignment, insertion, seating, release, withdrawal and settling. The toy
remained threaded by its ring, with no finger contact, 57.5 mm table clearance
and zero inverted or nonfinite element-substeps.

| Final measurement | Observed | Required |
| --- | ---: | ---: |
| Peak contact penetration | 1.477 mm | <=0.5 mm |
| Average upward rod-to-loop support | 86.2% of weight | >=90% |
| Final drift | 8.84 mm | <=3 mm |
| Final peak speed | 53.15 mm/s | <=10 mm/s |
| Minimum whole-hole engagement | 36.2 mm | >=10 mm |
| Absolute hole contour winding | 1.0 | >=0.5 |
| Whole-toy table clearance | 57.5 mm | >=20 mm |
| Inverted/nonfinite element-substeps | 0 | 0 |

The saved VLM verdict incorrectly calls the 21.8 N peak finger readback excessive.
The native 20 N actuator setting with 0.5 tolerance permits 30 N measured readback,
so this measurement passes. The deterministic penetration, support and settling
gates still fail. The current verifier prompt explains the tolerance; this
post-campaign clarification changes no saved result or acceptance threshold.

The campaign used the existing Codex login and exact `gpt-6-sol`: ten initial
plans, three refinements and 17 actual API calls (one proposal batch, three
refinements and 13 verifications), all with retry 0. Twelve executions were
rejected, one completed, and none earned a fresh confirmation. All initial plans
were rejected at an IK turn. A minimal planner feedback fix preserved those
partial traces for refinement. Its guarded continuation reused the original
proposal bytes and ten partial executions without new proposal calls or replays.
Refinement evidence also included a separately labelled failed static reference
baseline; that baseline is not a model candidate.

The physical scene is Hanson's native default at controller `3393ff5`, Newton
`5cd406a1` and SIMPACT `692aab2b`. No material, contact or solver overrides were
introduced. All 13 physical initial states matched the common observation within
1.20e-7 (tolerance 1e-6), and physical source hashes remained fixed. Independent
review matched cached evidence, raw refined plans and all 40 executed endpoints.
Thirteen focused core checks and Ruff passed. GPU allocations 5213853 and 5214671
are released; the continuation completed with exit 0.

- [Final video](replay.mp4): 40.67 seconds, 30 fps, all 1,220 frames decoded.
- [Planner result](result.json), [complete selected plan](selected_plan.json) and
  [measured rollout](rollout.json.gz).
- [Experiment manifest](experiment.json), [continuation provenance](provenance.json)
  and [actual model-call log](campaign_vlm_calls.json).
- [Initial observation](observation.json), [task configuration](task.json),
  [initial image](initial.png) and [final image](final.png).
- [Evidence archive](campaign_evidence.tar.gz): JSON, raw model responses,
  campaign source/prompt snapshots and launchers under `campaign/`; native
  reference JSON under `native_reference/`. Intermediate videos, screenshots
  and verbose worker logs remain in the original output directories.
- [Checksums](checksums.json) cover the copied artifacts and every archive member.

Copied evidence is byte-identical to the original artifacts. JSON and launchers
retain absolute paths to the original shared filesystem, recorded in the
checksums file. Use this report's relative links when browsing a fresh clone;
the archived scripts are historical provenance, not portable entry points.
The feature usage is in [the adapter README](../../../part_aware_control/README.md)
and the execution record is in [the plan](../../plans/PLAN_simpact_hanging.md).
