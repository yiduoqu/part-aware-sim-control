# coiled_usb_cable source asset

Copied unchanged from `generated_tool_assets/assets/coiled_usb_cable` (procedural geometry, Manifold union + TetGen, centroid part labels). Static mesh checks only there; behaviour is checked by the scenes here.

Coiled usb cable: original reference-informed rigid/deformable geometry.

- `final_mesh.mesh`: Medit tetrahedral mesh, 24,065 vertices and 71,030 tetrahedra, Z up, metres.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density. Parts:
  - 0: flexible_cable_and_strain_reliefs (silicone rubber, 5e+06 Pa, 1250 kg/m^3)
  - 1: usb_a_end_1_housing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 2: usb_a_end_1_metal (steel, 2e+11 Pa, 7850 kg/m^3)
  - 3: usb_c_end_2_housing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 4: usb_c_end_2_metal (steel, 2e+11 Pa, 7850 kg/m^3)
- `asset.json`: rigid groups [[1, 2], [3, 4]] (for `RigidPartSoftSpec`).
- `connectors.json`: plug mating frames (see `generated_tool_assets/CONNECTORS.md`).
- `appearance/`: visual mesh, albedo, and its binding to the physics mesh.

## SHA-256

- `final_mesh.mesh`: `4cfa8949facfc58fe840327779589dc5d6c251dabd6a8d775730a1e0f031bdce`
- `heterogeneous_params.npz`: `cb7112b26a7c12ecfa42c1ddaa9d43b9cb4e0c6b1c3b7d628b82c15b5ef475ec`
- `inferred_params.json`: `ee7eb82decc636c3746d1c9d5a5db5068af0dff1c4dff9d2a3db69ef56ff812f`
- `asset.json`: `6ed83faa1fbbaf2a9a529444a070ecda634beaa193d20358ae8f5025a76d9fc9`
- `connectors.json`: `94018401a3c9931e8dd6d9eb25bcc155634f7c942817162e9e730990100a5a90`
- `appearance/albedo.png`: `147462671d1cf202035dbe4671b7040d8ee494344f1382d24151153fc2f832bd`
- `appearance/visual_mesh.glb`: `78f42bb7fef3beab872da6924afbcf988de1bd8f0bea7ac9674a5e219ea74321`
- `appearance/visual_to_physics.npz`: `44580b4ff5dbe5ddc91f1c7ec71256a9630d1c2a6b72e43b06f2e0ddf7cc2272`
