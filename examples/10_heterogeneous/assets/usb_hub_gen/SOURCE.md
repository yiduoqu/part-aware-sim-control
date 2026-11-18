# usb_hub_gen source asset

Copied unchanged from `generated_tool_assets/assets/usb_hub` (procedural geometry, Manifold union + TetGen, centroid part labels). Static mesh checks only there; behaviour is checked by the scenes here.

Usb Hub: rounded housing, straight 350 mm cable, 4.5 mm diameter, two strain reliefs, usb_a endpoint.

- `final_mesh.mesh`: Medit tetrahedral mesh, 6,052 vertices and 19,025 tetrahedra, Z up, metres.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density. Parts:
  - 0: housing (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 1: cable_and_root_strain_relief (rubber, 5e+06 Pa, 1100 kg/m^3)
  - 2: distal_collar_connector (ABS, 2.2e+09 Pa, 1100 kg/m^3)
  - 3: plug (steel, 2e+11 Pa, 7850 kg/m^3)
- `asset.json`: rigid groups [[0], [2, 3]] (for `RigidPartSoftSpec`).
- `connectors.json`: plug mating frames (see `generated_tool_assets/CONNECTORS.md`).
- `appearance/`: visual mesh, albedo, and its binding to the physics mesh.

## SHA-256

- `final_mesh.mesh`: `79df5b54178cadb8bd5b4cf4e5683eef98de7052e93b167ee14615ec2accdae3`
- `heterogeneous_params.npz`: `8ba61d9bf29c86d4138b094ab22e61f6e0f0fec54efa36825eec9bcb02d6d48a`
- `inferred_params.json`: `381c9e1957557b13a59a0865462de0367ab4ed9c5c611dfe3a5d81530d235bb9`
- `asset.json`: `358d2cb6a313c075040b897e56d6485baee74859b91b547fe07abef9336c6c0e`
- `connectors.json`: `e1cac2fb6a37f92e1c03d274695272def80edef03cfebaeb7be58bcc2fbd4ca8`
- `appearance/albedo.png`: `a269c1a6ae2cdf0fde81a91c64fa24fb7c4231878c1f178ce15b0bef9141843d`
- `appearance/visual_mesh.glb`: `af61f80a2369ea98441dc18fda970022a5ad3faf122d6847eb72a3abb6be78b3`
- `appearance/visual_to_physics.npz`: `259b3f7edd6b176fc9b79dd83febbf960be263437be67dafcca6b7fe0da41397`
