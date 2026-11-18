# toilet_plunger_generated source asset

Re-imported from `generated_tool_assets/assets/toilet_plunger` (regenerated finer meshes; generated_tool_assets HEAD `66768d3` plus its uncommitted working-tree regeneration, 2026-11-16). Procedural geometry, Manifold union + TetGen, centroid part labels. Static mesh checks only there; behaviour is checked by the scenes here.

110 mm diameter hollow rubber cup with a 20 mm wooden shaft; total height 377 mm. Mouth remains open.

- `final_mesh.mesh`: Medit tetrahedral mesh, 7,251 vertices and 20,735 tetrahedra, Z up, metres.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density. Parts:
  - 0: wooden_handle (wood, 1.0e+10 Pa, 650 kg/m^3)
  - 1: flexible_cup (rubber, 5.0e+06 Pa, 1100 kg/m^3)
  - 2: sealing_lip (rubber, 5.0e+06 Pa, 1100 kg/m^3)
- `asset.json`: rigid groups [[0]] (for `RigidPartSoftSpec`).

## SHA-256

- `appearance/albedo.png`: `62017135481c3c586068161349bc1a42ae18a6c4c92ed7bea1cd3633cc34b7c8`
- `appearance/visual_mesh.glb`: `d03bc944b170a69b9e101a912c6f5b53b1a9ad50320142751bee14035bdb745b`
- `appearance/visual_to_physics.npz`: `b0e9f14e7abae66ffaa08b6949ac5ebe0280f1983f98e843af6f64c4e81aadb7`
- `asset.json`: `9289d558729381e94e036b56c309564d6e862eab062b59e4a91b9a66120fb696`
- `final_mesh.mesh`: `1a2175e77dea82ad15aebc265309d63703f7fc2ba48cf1460cb3822aebab7a37`
- `heterogeneous_params.npz`: `0f63fd5bd39d5568e5e53df651734134fc389840132e5d0a808d7457ce3e4f9c`
- `inferred_params.json`: `34bbb923050fb4e5c7f7eaf080f30e4dd264c509929a672431e7156d9edf856c`
