# Spring-guided retractable stamp

The red top cap slides down over the grey sleeve. Its central shaft carries the foam-backed die. A discrete, three-and-a-half-turn steel helix stores strain energy between the upper spring seat under the cap and the fixed lower seat in the sleeve. The shaft passes through the sleeve’s guide bore. A second concentric pair of surfaces guides the overlapping outer skirt and shroud. The sleeve rim bears on the paper.

## Dimensions and assembly

All mesh coordinates are in metres with +Z up and the rim centered on z = 0. The resting height is 70 mm, the cap diameter is 26 mm, and the sleeve diameter is 30 mm with 32 mm crests. The ribbed band covers z = 19–22 mm. The removable bottom cover is omitted. The relief tips lie at z = 3 mm. The 18 mm die has a 2 mm base and 2 mm raised, mirrored numeral and border. The numeral is 7 mm tall, the stem is 3 mm wide and the 2.5 mm nominal border has at least 2.5 mm clearance to the numeral.

The earlier solid Ø20 plunger and foam return collar were redesigned. The moving carrier has a Ø20 mm flange at z = 10–13 mm and a Ø8 mm central shaft from z = 13–40 mm. The fixed guide has a nominal Ø10 mm bore from z = 20–34 mm. Shaft clearance is 1 mm radial. The carrier and die have 2 and 3 mm nominal clearance to the lower sleeve respectively. The fixed outer shroud ends at z = 35.5 mm and the moving skirt starts at z = 30 mm. Their nominal radial clearance is 0.75 mm. At 4.5 mm travel the cap shoulder meets the fixed shroud. The 3.5 mm bench stroke leaves 1 mm before the stop.

The helix mean diameter is 15 mm. Wire has a 1.2 × 1.2 mm octagonal cross-section. Each end has a flat quarter-turn land, attached directly to its seat, at z = 22 and 40 mm. This attachment is an intentional simplification of a loose, preloaded commercial coil. The gaps between turns, shaft/bore and sliding skirt are meshed as empty space. Steel and plastic share faces only on the end lands, with 32 shared triangles at each seat. The shaft and sleeve share zero nodes. Material segmentation uses six labels and rigid groups [[0,1],[2]].

## Parts and mesh

| Index | Part | Material | E (Pa) | ν | Density (kg/m³) | Friction | Tets |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | top_cap | ABS (red) | 2.2e+09 | 0.37 | 1100 | 0.35 | 6,880 |
| 1 | plunger | ABS (grey) | 2.2e+09 | 0.37 | 1100 | 0.35 | 1,904 |
| 2 | outer_sleeve | ABS (grey) | 2.2e+09 | 0.37 | 1100 | 0.35 | 8,832 |
| 3 | return_spring | spring steel | 2e+11 | 0.3 | 7850 | 0.2 | 10,752 |
| 4 | foam_cushion | closed-cell foam | 3e+05 | 0.25 | 120 | 0.7 | 6,264 |
| 5 | rubber_die | natural rubber | 1.5e+06 | 0.47 | 1100 | 0.9 | 8,760 |

The volume has 11,429 vertices, 43,392 positive tetrahedra and 15,024 exterior faces. Total mass is 32.355 g. Maximum boundary edge is 4.263 mm overall and 1.475 mm across the three compliant parts. The foam and die base each have three axial element layers. The wire section has a center, inner ring and outer ring of nodes. The die outline is resolved by a planar constrained triangulation. Meshing sweeps the spring cross-section around the helix and joins angular and axial prism subdivisions along shared face diagonals. The GLB and its UV atlas use the actual exterior triangles. Maximum rest binding error is 3.60e−9 m or less. All FEM material properties are starting estimates.

## Press and release

A CPU bench solver uses first-order 3D tetrahedral linear elasticity for the steel spring and the foam/rubber assembly. End-land displacements condense the spring to one guided axial mode. The sleeve is supported at rest, and the cap/shaft and sleeve are ideal rigid groups. A rigid, flat paper plane supports compression only; node gap and compressive reactions were checked at the tips. An imposed top force drives a damped axial equation including gravity and unilateral paper reaction. Motion is computed both during pressing and after removing all external cap load. The scalar guide assumes axial loading without tilt or guide friction. Stiffness, damping and force are model estimates.

The settled clearance was 2.969 mm. At 3.500 mm cap travel the measured model cap load was 34.94 N and the die’s paper reaction was 9.43 N. Both relief regions reached 100% face contact. All 416 contact triangles had compressive nodal reactions. With zero cap load the die returned to a clearance of 2.969 mm and zero near-surface paper reaction. Static tet/tet separating-axis checks passed at rest, initial contact, and full load. In these three states the smallest signed volume ratio was 0.880. The complete Newton model, frictional guidance, buckling, physical ink transfer, real material calibration and off-axis hand operation were not tested. The imprint image in ../verification is a rasterization of computed contact triangles in paper coordinates.

## Use and provenance

Use scale 1.0 and RigidPartSoftSpec with rigid_parts=((0,1),(2,)). The stock Clawhauser loader successfully read this file. In a scene adapted from the old collar revision, label 3 must use the spring steel material. A clearance test using the maximum carrier radius at every axial height would mistakenly report spring interference; measure overlapping heights of each sliding pair. The previous collar material overrides and full-length Ø20 mm plunger dimensions belong to the prior design.

The exterior is based on the original user-supplied red/grey stamp photograph. The later transparent-housing photo guides the shaft/spring arrangement. Internal dimensions, steel grade approximation and end attachment are engineering assumptions. Reference image names and SHA-256 hashes are recorded in asset.json.

## SHA-256

The final export manifest also records the final byte hash of this document.

- `appearance/albedo.png`: `8a727b2042b71522fec8e217b3275e5ae49d3e55134268a6c9508961924d648b`
- `appearance/visual_mesh.glb`: `f33fdc10605ccc2932f7a3091b1c0267061e22e71c07227298e6e68e0261067b`
- `appearance/visual_to_physics.npz`: `19c8a5ca03ad019344629a22439d8b46c117fe583b4a01d546608fefe73519da`
- `asset.json`: `008e50722f7529e16d8737fc9c8841295433a64d0fb31901d2d4fd208ba2a1dc`
- `final_mesh.mesh`: `e88f644ca5d8fff12b68417ea4063f7c30ff581b5417ca8f461a92c0d6f5d279`
- `heterogeneous_params.npz`: `f25f335e77104f26c15c62492389ecbbda31d273595492b57bf5d7f4d710b22e`
- `inferred_params.json`: `2d96a70d567ce2aac38bc50bd0978394cccf3eab41368b1369698a1a51dc91a1`
- `metric_mesh_scaling.json`: `aa362eefdd3696fa86b25ec64993ffa78e2bb03b597d599f4595abd80e9a890b`
- `volume_topology.json`: `fa6910da3a6786feb4ab4248e94d59479d7bec8d940220eac1b2823a94f8503e`
