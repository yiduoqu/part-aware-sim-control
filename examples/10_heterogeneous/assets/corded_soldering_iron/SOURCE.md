# corded_soldering_iron source asset

Re-imported from `generated_tool_assets/assets/corded_soldering_iron` (regenerated finer meshes; generated_tool_assets HEAD `66768d3` plus its uncommitted working-tree regeneration, 2026-11-16). Procedural geometry, Manifold union + TetGen, centroid part labels. Static mesh checks only there; behaviour is checked by the scenes here.

Corded soldering iron: original reference-informed rigid/deformable geometry.

- `final_mesh.mesh`: Medit tetrahedral mesh, 6,246 vertices and 19,117 tetrahedra, Z up, metres.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density. Parts:
  - 0: flexible_cable_and_strain_reliefs (silicone rubber, 5.0e+06 Pa, 1250 kg/m^3)
  - 1: iron_insulated_grip (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 2: iron_barrel_and_tip (steel, 2.0e+11 Pa, 7850 kg/m^3)
  - 3: station_end_1_housing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 4: station_end_1_metal (steel, 2.0e+11 Pa, 7850 kg/m^3)
- `asset.json`: rigid groups [[1, 2], [3, 4]] (for `RigidPartSoftSpec`).

## SHA-256

- `appearance/albedo.png`: `98d0e54534670321211668ca575d742b03217d5294413a68b01370f724456014`
- `appearance/visual_mesh.glb`: `21d6c797c9c127c6c3850209cb3c48c9a28076707719a08826ffb58a62c10a59`
- `appearance/visual_to_physics.npz`: `631e6a63cf51870b3e94eac750b3f03bdf81d12238cafff90349381f007f98f8`
- `asset.json`: `b59de64cba97fda713e9735ac5d14fc4091ca0cbe6d9ba26cd8e510d4a7b799a`
- `connectors.json`: `ea0233d2d0ecfb1cdff1513442fdfa6614e062d710c69d5a9a7c61c38a36f32c`
- `final_mesh.mesh`: `a1d3a19abeb47819b916a4086dae5c9fe7fa5852c9fa63a205e6c92bf4e36624`
- `heterogeneous_params.npz`: `29a0337677089f02eb0d81fe4b7a11dba64553cfc742f186a24c2d1262a110b0`
- `inferred_params.json`: `4daec7fe9decf692bd2f9bf16d6a9bbde58ae7ace766232e93d001c05adad2fc`
