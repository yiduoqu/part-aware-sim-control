# Retractable rubber stamp

A 70 mm tall, round push-down stamp in its unloaded pose. The physics and visual meshes both use metres, +Z up and a common identity transform. The origin is the center of the open sleeve rim. The die points down.

## Reference and design decisions

The user supplied a side photograph of a pocket stamp with a red cylindrical cap, a grey body and a fluted ring. The photo informed the silhouette and palette. The specified millimetre dimensions govern the reconstructed geometry. The reference folder's sponge_wand asset provided the file conventions, including one-based Medit material references, zero-based NPZ labels, RGB values on a 0–255 scale and solid-fill code 0. No mesh was copied from it.

- Reference image: `codex-clipboard-eeb54766-c4b7-41cc-b2cd-4f0393c21d31.png`
- Reference image SHA-256: `8d772d6b4c9a0993fd225f86eccebc46106cb819648ff4c83c7a371ce5827196`
- The removable bottom cover is omitted. The sleeve is open at z = 0.
- A bonded elastomer collar replaces the metal coil spring and sliding fit. The intended load path runs from cap through collar to sleeve. The cap and plunger move together.
- The user selected a 7 mm digit height. An 18 mm die with a 2.5 mm border and 2.5 mm feature clearance leaves an 8 mm diameter area for the digit.
- The user specified 2 mm as the minimum nominal core-to-sleeve clearance. The plunger has 2 mm clearance and the smaller foam and die have 3 mm.
- The rib pattern and digit break continuous rotational symmetry. Forty equally spaced flutes and a custom bold sans-serif numeral provide these details. A geometrically right-reading digit as seen from +Z produces a mirrored die viewed from below with +Y upright.

## Parts and dimensions

Lengths in the following table are in mm. Diameters refer to the nominal circular profiles.

| ID | Part | Geometry | Material | E (Pa) | ν | Density (kg/m³) | Friction | Tets | Max exterior edge (mm) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | top_cap | Ø26, z 40–70, top edge rounded R3 | red ABS | 2.2e9 | 0.37 | 1100 | 0.35 | 6,656 | 4.12819 |
| 1 | plunger | solid Ø20, z 10–40 | light grey ABS | 2.2e9 | 0.37 | 1100 | 0.35 | 2,680 | 4.32672 |
| 2 | outer_sleeve | Ø30 outside, Ø24 bore, z 0–22. Fluted ring Ø32 at crests, z 19–22 | light grey ABS | 2.2e9 | 0.37 | 1100 | 0.35 | 3,600 | 4.16980 |
| 3 | return_collar | Ø26 outside, Ø22 bore, z 22–40 | mid-grey soft elastomer foam | 2.0e5 | 0.20 | 200 | 0.60 | 16,320 | 1.47073 |
| 4 | foam_cushion | solid Ø18, z 7–10 | dark closed-cell foam | 3.0e5 | 0.25 | 120 | 0.70 | 6,264 | 1.33420 |
| 5 | rubber_die | Ø18 base z 5–7, border and digit z 3–5 | dark red natural rubber | 1.5e6 | 0.47 | 1100 | 0.90 | 8,760 | 1.47444 |

All six parts use solid_fill. Rigid groups are [[0,1],[2]]. The collar bonds to the cap across the collar's upper annulus. Only the radially overlapping land of the sleeve and collar is bonded at z = 22 (nominal radius 12–13 mm). The collar's inward overhang faces air. The other bonds are cap to plunger, plunger to foam, and foam to die.

The digit has a 3 mm vertical stem and a 3 mm perpendicular stroke on the slanted flag. The polygonal die has a 2.5 mm nominal border. The minimum distance from digit to inner border is 2.61931 mm. The border and digit tips share the plane z = 3 mm. The cap offers a 26 mm diameter gripping surface from z = 40 to 67, and the sleeve offers a 30 mm diameter gripping surface from z = 0 to 19. Both fit an 85 mm jaw opening.

## Geometry and meshing

Circular profiles were discretized as regular polygons. The clearance-critical sleeve and collar bores were circumscribed about the nominal circles to maintain minimum radial clearance on flat facets. At their vertices the bore radii exceed nominal by less than 0.010 mm. Other circular dimensions are measured at profile vertices. The top fillet samples a 3 mm quarter circle. The fluted sleeve is a swept radial–axial section. The die has a constrained planar triangulation of the digit and concentric annular strips. The relief is actual tetrahedral geometry.

Shared section vertices and matching face diagonals form conforming interfaces. Triangular prisms were split into tetrahedra. Two transition slabs connect different end triangulations in the plastic. Tetrahedra were positively oriented and assigned a material index from their centroid. The GLB comes directly from the outward-oriented exterior triangles. Duplicated visual corners allow smooth normals, sharp creases and independent UVs at material seams. A six-swatch PNG supplies the albedo and is embedded byte-for-byte in the GLB.

## Rest-pose verification

The Medit mesh, both NPZ files, the JSON material table and the GLB buffers were re-read for the checks below.

- 10,821 physics vertices, 44,280 tetrahedra, 13,088 exterior triangles.
- Bounding box in metres: (−0.016, −0.016, 0) to (+0.016, +0.016, 0.070).
- Total mass: 0.0355080127223 kg, calculated as the sum of signed tet volume times the corresponding part density.
- One face-connected volume. Every interior face has two opposite windings. Every boundary edge has two incident triangles, and every boundary vertex link is a single cycle. Zero duplicate vertices, duplicate tets, isolated vertices, inverted tets, zero-volume tets or faces with more than two owners. The smallest tet volume is 2.14583333e−11 m³.
- An independent tetrahedron separating-axis test examined 3,199,220 bounding-box candidate pairs. It found zero volumetric overlaps and zero contacts between pairs without a shared vertex, using a 1e−9 mm numerical tolerance.
- Every tet was checked against its centroid label. All exterior triangle edges satisfy the requested per-material limits. Two conforming radial bands span the collar wall. Three axial layers span each of the foam and the die base, and two layers span the raised relief. No tetrahedron crosses the internal layer planes or the collar's division surface.
- Shared interface faces: cap/plunger 80, cap/collar 320, collar/sleeve 160, plunger/foam 696, foam/die 696. All occur at their specified axial heights. The sleeve shares zero faces, edges or vertices with the moving core.
- Rest-pose polygon-to-polygon radial clearances: plunger/sleeve 2.000 mm, foam and die/sleeve 3.000 mm, plunger/collar 1.000 mm. Throughout their overlapping heights, the moving profiles lie inside their nominal outer cylinders and the bores enclose their nominal inner cylinders.
- The NPZ arrays have the requested dtypes and shapes. The per-tet labels, E, ν and density agree with the Medit material references and JSON part entries. Colors and fill modes agree with the part table. Friction is stored in the JSON.
- GLB positions, triangle indices and UV values match the visual binding arrays. The surface maps resolve to the correct physics vertices and face owners. Maximum Euclidean rest-pose binding error is 3.41634e−9 m.
- Material values are starting estimates. Press travel, return force, self-contact under compression, grasp stability and inking have not been simulated or measured. The asset records behavior_validated = false.

## Export records

This edition adds metric scaling, volume topology and a final export manifest for the procedural stamp. The final manifest records relative paths, sizes and SHA-256 digests for all other delivered files, including this document. The mesh, materials and appearance payloads are byte-identical to the reviewed eight-file asset.

## SHA-256

- `appearance/albedo.png`: `42138c3848ca777fd2e3fe6132386eadac239947274048d11d6b17535fd01fac`
- `appearance/visual_mesh.glb`: `e5d6274e5e4c7176402b90e417f69f8b881ce53f0ae3f6e6c33a388df6eed1b9`
- `appearance/visual_to_physics.npz`: `9f39d1fb967e3764125ab3335da072e7970a65068d3f67d1dc88fe901fce140c`
- `asset.json`: `c120705b34ef30bd64f76cf1c9a9a15560a2ca065f981391aab27b716296848a`
- `final_mesh.mesh`: `acc6f5c191e5b6c09ec2723280b2ef05d3dcd6da7dc731ea2834167619073ae1`
- `heterogeneous_params.npz`: `9d80f57ab78e673224adac893264b1278e07947403a2360cca952a7b7e646c28`
- `inferred_params.json`: `48eceaea9ce4388fc12a79cf10aec2ae3396345108ee79035378c0143cd4a466`
- `metric_mesh_scaling.json`: `b01b9fd4f1800c48599e2ade47de00ca19842fea7740c5f9c44d671e68506769`
- `volume_topology.json`: `cd59f108c04dc5f42d8b140eeb200a20f53e29b2867b8ca781c9b335376afcd0`
