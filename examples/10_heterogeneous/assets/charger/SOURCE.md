# charger source asset

Copied unchanged from `generated_tool_assets/assets/charger` (procedural geometry, Manifold union + TetGen, centroid part labels). Static mesh checks only there; behaviour is checked by the scenes here.

Charger: rounded housing, straight 220 mm cable, 4 mm diameter, two strain reliefs, usb_c endpoint.

- `final_mesh.mesh`: Medit tetrahedral mesh, 5,067 vertices and 18,156 tetrahedra, Z up, metres.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density. Parts:
  - 0: housing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 1: cable_and_root_strain_relief (rubber, 5e+06 Pa, 1100 kg/m^3)
  - 2: distal_collar_connector (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 3: plug (steel, 2e+11 Pa, 7850 kg/m^3)
- `asset.json`: rigid groups [[0], [2, 3]] (for `RigidPartSoftSpec`).
- `connectors.json`: plug mating frames (see `generated_tool_assets/CONNECTORS.md`).
- `appearance/`: visual mesh, albedo, and its binding to the physics mesh.

## SHA-256

- `final_mesh.mesh`: `8243c1a8d8222c01b6fb91719ad71654f3f84d7059a2b319a2a9c0cbc70790f4`
- `heterogeneous_params.npz`: `ac466b322752c91a8c2ad5f6e908e28c547643ac9e1b11f943d5509d2b0048b8`
- `inferred_params.json`: `4c6c7ec912fccdcaa22774dfd03c16aa5ebb1c7134c1c181c6d8e536f2644b1e`
- `asset.json`: `2e34ea7e92b7f1c0adf20c43d20c4dacad60e5932e9175e01a584bb430ad9d50`
- `connectors.json`: `29090829cb53311c9ef2bde55b6d72e77c432448b9a983095eb4c81384a719f0`
- `appearance/albedo.png`: `6f77b30fef3f6cd0ef57da0103947556f0c36e9db66cbc1353ee785c971677a3`
- `appearance/visual_mesh.glb`: `e616d32c9a3aebb62542368b0631fa96dfe7841bae4e8ea0d88848d09688c479`
- `appearance/visual_to_physics.npz`: `93b238779a1c7b2de9eac4fb468f22cc2b71dde69dd493944eb7b72ca8ab7929`
