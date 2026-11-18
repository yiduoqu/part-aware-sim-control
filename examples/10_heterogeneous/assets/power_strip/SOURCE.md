# power_strip source asset

Copied unchanged from `generated_tool_assets/assets/power_strip` (procedural geometry, Manifold union + TetGen, centroid part labels). Static mesh checks only there; behaviour is checked by the scenes here.

Power Strip: rounded housing, straight 500 mm cable, 10 mm diameter, two strain reliefs, mains endpoint.

- `final_mesh.mesh`: Medit tetrahedral mesh, 10,619 vertices and 35,632 tetrahedra, Z up, metres.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density. Parts:
  - 0: housing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 1: cable_and_root_strain_relief (rubber, 5e+06 Pa, 1100 kg/m^3)
  - 2: distal_collar_connector (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 3: plug (steel, 2e+11 Pa, 7850 kg/m^3)
- `asset.json`: rigid groups [[0], [2, 3]] (for `RigidPartSoftSpec`).
- `connectors.json`: plug mating frames (see `generated_tool_assets/CONNECTORS.md`).
- `appearance/`: visual mesh, albedo, and its binding to the physics mesh.

## SHA-256

- `final_mesh.mesh`: `428631e871e4b5a9c71d13a5d7489c261415bb5c9c6b7043c09e5891cc0751f4`
- `heterogeneous_params.npz`: `837cea606effb352aad053c8842df62a3358494b67eb2d9a1001c27e8d1c81b5`
- `inferred_params.json`: `de9c019bd007190dbe7bf98d3ab8f283f177c336dc642527a0dfe1d22bdb5300`
- `asset.json`: `c89c6ed89cfb1e91dbea1d1680358a8a8341ea2778b4dfedf33b533d4cb526a6`
- `connectors.json`: `0edfedcd0813f7863d4354c5023e65c2315dceb7e6a36116ba8eeba677461c49`
- `appearance/albedo.png`: `e568af9dbdb760ff49e1ccf792db8d85e4c1fe5b137d8d96e0a36606360e42c8`
- `appearance/visual_mesh.glb`: `7a82dda7361152d686ee6df98dff5cc8e9423c074fe6ea2d8b6fab3f2e464295`
- `appearance/visual_to_physics.npz`: `5fe8a3c559e799764d27c02cdf6a01668aaf77be6c31ecde332444f7b172c375`
