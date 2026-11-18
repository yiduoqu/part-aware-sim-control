# silicone_spatula source asset

Re-imported from `generated_tool_assets/assets/silicone_spatula` (regenerated finer meshes; generated_tool_assets HEAD `66768d3` plus its uncommitted working-tree regeneration, 2026-11-16). Procedural geometry, Manifold union + TetGen, centroid part labels. Static mesh checks only there; behaviour is checked by the scenes here.

NVIDIA wooden-handle silicone spatula, approximately 295 mm long. Source attachment openings are capped and unioned; the blade remains flexible.

- `final_mesh.mesh`: Medit tetrahedral mesh, 7,658 vertices and 29,121 tetrahedra, Z up, metres.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density. Parts:
  - 0: wooden_handle (wood, 1.0e+10 Pa, 650 kg/m^3)
  - 1: flexible_silicone_head (silicone, 1.5e+06 Pa, 1100 kg/m^3)
- `asset.json`: rigid groups [[0]] (for `RigidPartSoftSpec`).

## SHA-256

- `appearance/albedo.png`: `46882d65c07c7494e2a43998ec57d17e49d85b520c3c2f550efe2691ee9b06cd`
- `appearance/visual_mesh.glb`: `78f3c18bc1f87d42d722235e5ecfce170680b1ae4ec614a7a58dc233cfef1508`
- `appearance/visual_to_physics.npz`: `bec0c6387cc8c153237499ec1f42abbe9b0458566b9ab9c9d1b2cd60b9541fca`
- `asset.json`: `710edf515343201f4508c4775f24c896c2b79f5a9ca4e476436454637a53f8d5`
- `final_mesh.mesh`: `d30a3a8ed2393b3143b59b86fd5a0963d651c78b133948049eeb3631cc3d3e17`
- `heterogeneous_params.npz`: `82673c3d99c6bbc225b1615ba788af8b51f30c9c20eada676635da282d86c86f`
- `inferred_params.json`: `c3d4d9fc1d78bd8baa1d65036f29d0232b8a5cfc098120d04955b57fea0109e2`
