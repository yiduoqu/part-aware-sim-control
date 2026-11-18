# y_splitter_cable source asset

Copied unchanged from `generated_tool_assets/assets/y_splitter_cable` (procedural geometry, Manifold union + TetGen, centroid part labels). Static mesh checks only there; behaviour is checked by the scenes here.

Y splitter cable: original reference-informed rigid/deformable geometry.

- `final_mesh.mesh`: Medit tetrahedral mesh, 6,701 vertices and 20,969 tetrahedra, Z up, metres.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density. Parts:
  - 0: flexible_cable_and_strain_reliefs (PVC, 5e+06 Pa, 1250 kg/m^3)
  - 1: mini_b_end_1_housing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 2: mini_b_end_1_metal (steel, 2e+11 Pa, 7850 kg/m^3)
  - 3: usb_a_end_2_housing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 4: usb_a_end_2_metal (steel, 2e+11 Pa, 7850 kg/m^3)
  - 5: usb_a_end_3_housing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 6: usb_a_end_3_metal (steel, 2e+11 Pa, 7850 kg/m^3)
- `asset.json`: rigid groups [[1, 2], [3, 4], [5, 6]] (for `RigidPartSoftSpec`).
- `connectors.json`: plug mating frames (see `generated_tool_assets/CONNECTORS.md`).
- `appearance/`: visual mesh, albedo, and its binding to the physics mesh.

## SHA-256

- `final_mesh.mesh`: `34ce5f6f957f8092f715075844fcaf4601b6b81a053d9a01e4839126be146942`
- `heterogeneous_params.npz`: `b73a79c3e31af69a8c2bfe56e2a633f0149a3caac3556acc56403ae0c52a7d69`
- `inferred_params.json`: `7a01c194c854e28ae9ac17e62540948e9325f57b0952645e81c396b7251ef9eb`
- `asset.json`: `d5f149e71b6680174f35986004cafcc3841d382d0f4dc4b9d89a8217957c279d`
- `connectors.json`: `021cb07c91e4eab9d45f9c28a49abfe201e95bdfb29d2e38926fb8975f62fb75`
- `appearance/albedo.png`: `f0fe5cbea0140ebd1c343504db536025500dede3800be739253b94fc180cfa27`
- `appearance/visual_mesh.glb`: `88c1d7532a330e9ce655a0d03bce701f581d41404e61d70a1b3388c4ca5c6f63`
- `appearance/visual_to_physics.npz`: `1bb4e858ffb8f6fae5a3b14ee1db6d68fea60f8a73e94dacfd9b124886e8ff9b`
