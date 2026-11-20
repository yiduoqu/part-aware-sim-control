# Hinged soap case

A travel soap case in its closed pose: a tray and a lid of the same plan and about the same depth, joined
along the -Y side by a thin web that is part of the same mesh. Built by `../generate_hinged_soap_case.py` (numpy only: a tensor-product
grid of hexahedra, six tetrahedra each, corners rounded by moving vertices). It follows no reference photo
and no particular product; the dimensions are ordinary ones for this kind of case.

Coordinates are in metres, Z up. The origin is the centre of the tray's footprint on the plane z = 0. X is
the long axis, the hinge is on -Y and the thumb tab on +Y. The lid opens by turning about X, its top going
toward -Y.

## Dimensions and materials

- Tray and lid: 100 x 64 mm in plan, 12 mm corner radius, 2 mm walls, floor and lid top.
- Tray z = 0 to 16 mm, lid z = 17 to 32 mm. The rims face each other across a 1 mm gap and share no vertex.
- Cavity: 96 x 60 x 28 mm, enough for the 90 x 55 x 25 mm [`soap_bar`](../soap_bar/SOURCE.md), which stands 11 mm out of the tray.
- Web: 48 mm along X, 1 mm thick, free from z = 13 to 20 mm, standing 1 mm off the back wall. A 3 mm knuckle joins each end to the tray (below) and the lid (above); the knuckles carry the tray's and the lid's labels.
- Thumb tab: 24 x 8 x 2 mm, flush with the lid's lower rim on +Y.

| Index | Part | Material | E in Pa | Poisson ratio | Density in kg/m^3 | Friction | Tetrahedra | Mass in g |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | base_tray | polypropylene | 1.5e+09 | 0.42 | 910 | 0.30 | 18,720 | 19.35 |
| 1 | lid | polypropylene | 1.5e+09 | 0.42 | 910 | 0.30 | 21,408 | 18.81 |
| 2 | living_hinge | polypropylene web (bending-equivalent) | 4.0e+07 | 0.42 | 910 | 0.30 | 1,344 | 0.31 |
| 3 | lid_tab | polypropylene | 1.5e+09 | 0.42 | 910 | 0.30 | 384 | 0.35 |

Rigid groups `[[0], [1, 3]]`: the tray is one body, the lid and its tab another, and the web is
the only part left deformable. Leave them out to run every part as finite elements.

The values are engineering starting estimates, not measurements. The web's modulus is not polypropylene's:
a real living hinge is about 0.3 mm thick, this one is meshed at 1 mm, and 4.0e7 Pa gives the meshed web
the bending stiffness of the thin one (1.5e9 x 0.3^3). A real hinge also yields and stays where it is put;
this elastic web springs back toward the closed pose.

## What is not modelled or tested

- No latch, no overlapping lip and no seal. Only contact between the rims and the web's stiffness hold the
  lid shut, so a scene needs lid-to-tray contact: as two rigid groups, or as soft self-contact whose
  rest-shape exclusion is under the 1 mm rim gap.
- No simulation was run when the bundle was written. What the lid, the web and the tray do when the case is
  opened is measured by the scenes that use it, and reported in `../../README.md`.
- The walls are thicker than a moulded case's and the plan is extruded straight up, with no draft or
  rounded top edge.

## Verification from the written files

| Measurement | Result |
|---|---|
| Physics vertices / tetrahedra | 10,872 / 41,856 |
| Boundary triangles | 15,088 |
| Bounding box in m, min to max | (-0.0500, -0.0340, 0.0000) to (0.0500, 0.0400, 0.0320) |
| Mass from volume x density | 38.81 g |
| Smallest / largest tetrahedron | 0.0869 / 2.0056 mm^3 |
| Dihedral angles | 6.7 to 117.6 degrees |
| Summed tet volume / volume enclosed by the boundary | 42.652764 / 42.652764 cm^3 |
| Boundary Euler characteristic | 2 |

Every tetrahedron has positive signed volume, no face has more than two owners, and every boundary edge
has exactly two boundary triangles. The repository's `load_labelled_tets` accepts the volume as one
face-connected solid. Labels come from the grid cell a tetrahedron was cut from.

## SHA-256

- `appearance/albedo.png`: `5f979dac0082736b6de72ec6cfbaa908ecb90bd09f1eb9e2cb056e1ba0ec2dd2`
- `appearance/visual_mesh.glb`: `12faac2d56a540d6c941db9620def6efb925d14866bc582077d961c96220fc82`
- `appearance/visual_to_physics.npz`: `223aada3c9e8c469c5eb427a0c1f462d02a6e093d6958d3f4da70473c21960fd`
- `asset.json`: `dbaf5298dd49d8f8d0454caf4d0345c0441db292d766be285ed759d2573d68a5`
- `final_mesh.mesh`: `de528b871ca46da987ac5f6f9b452fc6c2ad47ca21dd95e210af632a2bfaffa7`
- `heterogeneous_params.npz`: `66692f0f364870017ba9957d9ef145ecc6fcee184f69c6caa520ecf34348d0ca`
- `inferred_params.json`: `e791b7f372f7f3868701eab035750ee532315a71fe753dd1b3bb98897a41f918`
