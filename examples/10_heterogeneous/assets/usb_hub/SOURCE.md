# USB hub source asset

Copied from the `usb_hub_revised` export (`final_export_manifest.json`, schema
`hag4r-staged-object-package-v1`, accepted revision `revision_0001`). The mesh and the
JSON were saved with LF line endings, which is what the manifest's digests are of.

- `final_mesh.mesh`: Medit tetrahedral mesh, 3,022 vertices and 9,652 tetrahedra, 0.25 m long.
- `heterogeneous_params.npz`: per-tet labels, Young's modulus, Poisson ratio and density.
  IDs 0, 1, 2, 3 are the housing (ABS, 2.2 GPa), cable (rubber, 5 MPa), inline connector
  and USB-A plug (steel, 200 GPa).
- `inferred_params.json`: part names and material estimates.
- `appearance/`: visual mesh, albedo, and its binding to the physics mesh.

`usb_hub_box.py` checks the manifest's SHA-256 digests before building:

- `final_mesh.mesh`: `12614f42f0da9bb66ef34405e04c555d0b29bfb1540bb828fb54606a5676240c`
- `heterogeneous_params.npz`: `673d52126a6ee06ca0488e48a7a31cdf103da97dfa2542dddede5c4c13a7403d`
- `inferred_params.json`: `d1efc934ca5514b35eba3b8b01d080e9fb9bfc28a621513c19260631ad27b7db`
