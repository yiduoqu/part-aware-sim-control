# Soap bar

The bar for [`hinged_soap_case`](../hinged_soap_case/SOURCE.md): 90 x 55 x 25 mm with a
10 mm plan-view corner radius, one part and one material. Built by `../generate_soap_bar.py` on the case
generator's grid. It follows no reference photo and no particular product.

Coordinates are in metres, Z up. The origin is the centre of the bar's lower face, X its long axis.

| Index | Part | Material | E in Pa | Poisson ratio | Density in kg/m^3 | Friction | Tetrahedra | Mass in g |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | soap | bar soap | 5.0e+07 | 0.40 | 1050 | 0.35 | 13,440 | 127.6 |

Rigid groups `[[0]]`: the whole bar is one rigid body, which is how the scenes run it. The
modulus is a starting estimate for a dry bar, not a measurement; a wet or worn bar is neither modelled.
No simulation was run when the bundle was written.

## Verification from the written files

| Measurement | Result |
|---|---|
| Physics vertices / tetrahedra | 2,835 / 13,440 |
| Boundary triangles | 2,208 |
| Bounding box in m, min to max | (-0.0450, -0.0275, 0.0000) to (0.0450, 0.0275, 0.0250) |
| Smallest / largest tetrahedron | 1.841 / 17.723 mm^3 |
| Dihedral angles | 22.1 to 114.4 degrees |
| Summed tet volume / volume enclosed by the boundary | 121.547729 / 121.547729 cm^3 |
| Boundary Euler characteristic | 2 |

Every tetrahedron has positive signed volume, no face has more than two owners, and every boundary edge
has exactly two boundary triangles.

## SHA-256

- `appearance/albedo.png`: `7dff85341592ea4d3121ebc7e3a2a571d0777fc95ae72213167a3e4a419a153e`
- `appearance/visual_mesh.glb`: `9f193e6538ae67577aa5207180a1c234e41d7edebc51052bbd23126289d80288`
- `appearance/visual_to_physics.npz`: `b4a09f8901d5d7dfe72e64732d2fa8ee2fd923947b6ebc40fa49e73c9f2d37d9`
- `asset.json`: `49aa36e7f465ff589cec02baad12948073dd33dc330c075a954176fcca51aa26`
- `final_mesh.mesh`: `884cd5d892700d1c95b51095d09acfaf3b9ae4109aaf8f48939bbc0098e5b190`
- `heterogeneous_params.npz`: `6e261a0feb3207bece5b30a30e1d3773dc953adfece381a30672b1b1df2e53fb`
- `inferred_params.json`: `620b302e0b0ecc3004fef978638d099f3ee62d988fe0d65fa159cbf0755c857e`
