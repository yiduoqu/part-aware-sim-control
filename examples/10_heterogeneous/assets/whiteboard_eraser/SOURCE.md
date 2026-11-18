# whiteboard_eraser source asset

Re-imported from `generated_tool_assets/assets/whiteboard_eraser` (regenerated finer meshes; generated_tool_assets HEAD `66768d3` plus its uncommitted working-tree regeneration, 2026-11-16). Procedural geometry, Manifold union + TetGen, centroid part labels. Static mesh checks only there; behaviour is checked by the scenes here.

110 x 48 mm eraser with a rigid rounded palm grip and a 10 mm compressible wiping pad.

- `final_mesh.mesh`: Medit tetrahedral mesh, 8,915 vertices and 36,910 tetrahedra, Z up, metres.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density. Parts:
  - 0: palm_grip (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 1: pad_backing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 2: felt_foam_wiping_pad (felt over compressible foam, 1.2e+05 Pa, 150 kg/m^3)
- `asset.json`: rigid groups [[0, 1]] (for `RigidPartSoftSpec`).

## SHA-256

- `appearance/albedo.png`: `6ee4f2989f5b00000e181430699d819e1436fa32a453a02e61b7f698a49c74e1`
- `appearance/visual_mesh.glb`: `915871dda7547b9744a2bbb64e7e228794622a4a1dd2ac225515a76f627cd465`
- `appearance/visual_to_physics.npz`: `fe9658fb5738066e9eedb3a296db898b31ce6c101c9ec91ef8413f2674c13546`
- `asset.json`: `72ba7612389ea74e325c4335107382ed4cbb812a58fad554e10698847d7c9ad8`
- `final_mesh.mesh`: `d4f48a4d04080d77611507f9a027af8026e73d2606e3b9c0ad9ce8a3d05512f9`
- `heterogeneous_params.npz`: `b8449b0125e9d822aaa8e9798a40dd74aa00c109a4b928fc2d653a6869485271`
- `inferred_params.json`: `65cb86f8012816d122797c5a485f63f27abeef0ad04e8a9261cf47bc2f8df271`
