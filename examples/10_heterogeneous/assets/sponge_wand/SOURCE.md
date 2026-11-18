# sponge_wand source asset

Re-imported from `generated_tool_assets/assets/sponge_wand` (regenerated finer meshes; generated_tool_assets HEAD `66768d3` plus its uncommitted working-tree regeneration, 2026-11-16). Procedural geometry, Manifold union + TetGen, centroid part labels. Static mesh checks only there; behaviour is checked by the scenes here.

Reference-informed dishwand with a continuous contoured rigid grip/neck, a small inset backing, and a rounded tapered 85 x 50 x 20 mm cleaning sponge.

- `final_mesh.mesh`: Medit tetrahedral mesh, 5,012 vertices and 16,874 tetrahedra, Z up, metres.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density. Parts:
  - 0: handle (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 1: backing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 2: compressible_sponge (open-cell foam, 8.0e+04 Pa, 80 kg/m^3)
  - 3: scouring_face (compliant scouring pad, 3.0e+05 Pa, 180 kg/m^3)
- `asset.json`: rigid groups [[0, 1]] (for `RigidPartSoftSpec`).

## SHA-256

- `appearance/albedo.png`: `92d0c6d2bb01b076537cd6c22e54b89dccbc9c8c57edcd6c3bf4264c8ff44bd9`
- `appearance/visual_mesh.glb`: `8cf35df253178d37730059024dd2e90102eda7943a6e18728a3805157f78f9e7`
- `appearance/visual_to_physics.npz`: `8d1a3fe51c9f7d41952bde0ee4b35548854210d981e606c67bdea74b42b7ec26`
- `asset.json`: `aed2882c50a1059d28411dda5244207671cc2ab71598bf89daafec29dd6c9b51`
- `final_mesh.mesh`: `6e2940bd199ab18e2a808c07313ffd182fc0e1ba756a04a76eacd884c37c959d`
- `heterogeneous_params.npz`: `bf8b435e3aa10d226099e3cf453c74969430d2d634a700a8c7bcec47a2e6b8fd`
- `inferred_params.json`: `e98a1ac1d477fe0d48cfc4888ddcfc9b2514d94b6d5da6d8b112aab587c5ace1`
