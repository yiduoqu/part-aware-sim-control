# gap_sponge_wand source asset

Re-imported from `generated_tool_assets/assets/gap_sponge_wand` (regenerated finer meshes; generated_tool_assets HEAD `66768d3` plus its uncommitted working-tree regeneration, 2026-11-16). Procedural geometry, Manifold union + TetGen, centroid part labels. Static mesh checks only there; behaviour is checked by the scenes here.

Reference-informed dishwand with a continuous contoured rigid grip/neck, a small inset backing, and a rounded tapered 65 x 16 x 12 mm gap sponge.

- `final_mesh.mesh`: Medit tetrahedral mesh, 4,352 vertices and 14,813 tetrahedra, Z up, metres.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density. Parts:
  - 0: handle (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 1: backing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 2: compressible_sponge (open-cell foam, 8.0e+04 Pa, 80 kg/m^3)
  - 3: scouring_face (compliant scouring pad, 3.0e+05 Pa, 180 kg/m^3)
- `asset.json`: rigid groups [[0, 1]] (for `RigidPartSoftSpec`).

## SHA-256

- `appearance/albedo.png`: `92d0c6d2bb01b076537cd6c22e54b89dccbc9c8c57edcd6c3bf4264c8ff44bd9`
- `appearance/visual_mesh.glb`: `f6be47f1269664db86a02d96d9992cabeea623d0a511e31f13c6ebf7e9fa072c`
- `appearance/visual_to_physics.npz`: `a2c6af1de890cb98f79f121956318540c1fa336e49ef282e041284dfbc163993`
- `asset.json`: `be2702667feb462217cd578e7c7af011b744f00942cac04c3e403dc66997dab9`
- `final_mesh.mesh`: `25ed873b258d1a7ff89d705d0b428b6b0399e41d3c3461b3d150593daba8246c`
- `heterogeneous_params.npz`: `e8d94496d51659df25dcee645ca8bad2a37a3223101e0744311c8dd60febc15a`
- `inferred_params.json`: `3ce4aa4c082b265970e123d86461e72931654938c507d4ff99c3ee4973b09408`
