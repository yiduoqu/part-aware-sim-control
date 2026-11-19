# Traditional wooden-handle rubber stamp

The coordinates are in metres. The origin is the centre of the 60 × 25 mm printing footprint. At rest the raised border and three bars contact the plane z = 0 and face −Z. The centre bar covers the origin. The handle points along +Z.

## Dimensions and materials

| Index | Part | Size in mm | Axial interval in mm | Material | E in Pa | Poisson ratio | Density in kg/m³ | Friction | Tetrahedra |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| 0 | wooden_handle | 45 tall, Ø22 neck, Ø32 maximum | 21–66 | Dark-stained beech wood | 1.0e10 | 0.30 | 650 | 0.40 | 3,927 |
| 1 | wooden_mount | 64 × 29 × 12 | 9–21 | Pale beech wood | 1.0e10 | 0.30 | 650 | 0.40 | 6,728 |
| 2 | foam_cushion | 60 × 25 × 3 | 6–9 | Closed-cell foam | 3.0e5 | 0.25 | 120 | 0.70 | 11,340 |
| 3 | rubber_die | 60 × 25 × 6, including relief | 0–6 | Red natural rubber | 1.5e6 | 0.47 | 1100 | 0.90 | 17,280 |

All parts use `solid_fill` (code 0). Rigid group `[[0, 1]]` combines the handle and mount. The axial handle profile in `asset.json` gives the radius at each sampled height. Diameters are measured across opposite perimeter vertices. The 32 mm maximum grip diameter fits inside an 85 mm parallel-jaw opening with 53 mm of opening to spare. Material and friction values are the supplied engineering estimates. Grasp stability and dynamic stamping behaviour have not been tested.

The rubber backing is solid from z = 3 to 6 mm. Its rectangular frame and three bars project 3 mm below it. The frame is 4.286 mm wide on the short ends and 4.167 mm wide on the long sides. Each bar measures 5.714 mm along X and 8.333 mm along Y. The bar centres are at X = −17.143, 0 and +17.143 mm on Y = 0. The minimum clear gap is 4.167 mm. The combined printing area is 785.714 mm².

## Reference and construction

The user supplied the 447 × 447 pixel reference image `codex-clipboard-df5d38ec-c39a-4149-9463-e99af8d36fa9.png` (SHA-256 `c47ebe04081b4ea6f860d98dae98dad9c980392f288ccb570affe1f07fe211bd`). It shows a traditional stamp with a dark turned handle, pale rectangular mount and thin dark cushion. The turned silhouette and wood tones follow that image. The prescribed dimensions set the overall proportions. The die face is hidden in the photo. The red rubber and border-and-bars pattern are choices from the options in the specification. The photo's label holder and lettering are omitted.

The backing and cushion share a rectangular grid. Only cells inside the border or bars extend down to the printing plane. An alternating five-tetrahedron subdivision gives matching diagonals on every soft-layer interface. Three full 1 mm mesh layers span the backing and three more span the cushion. Graded blocks connect the fine cushion interface to the wood interior. Axial sections form the grip, and perimeter strips and corner sectors form the mount's 2 mm plan-view corner radius and 0.8 mm top and bottom edge roundovers. Circular curves are faceted in the linear mesh. Local edge bisection splits all incident tetrahedra, including across material interfaces.

The exterior triangles are the tet faces with exactly one owner. Their original winding and owner indices are preserved in the GLB binding. Visual vertices are split where normals or UVs differ. The embedded GLB texture is byte-identical to the standalone sRGB albedo. The wood grain, foam pores and rubber texture are procedurally generated. Handle and mount colours are based on the median RGB of sampled regions of the reference. The GLB node has an identity transform, and its stored positions are already in the Z-up physics frame. The two review renders were generated from the exported triangles, normals, UVs and embedded albedo and are delivered outside this asset directory.

## File conventions

- Medit uses 1-based vertex indices. Its tetrahedron reference equals the zero-based part index plus one. Its vertex references are zero.
- Both NPZ files use zero-based indices. Material arrays follow the tetrahedron order in the ASCII file. RGB values in the material files use the 0–255 sRGB scale.
- For visual vertex v, `physics_vertex_indices[v]` equals `surface_vertex_to_physics_vertex[visual_vertex_to_surface_vertex[v]]`. Each visual face has its owner tet index and part ID in matching face order.

## Verification from the delivered files

| Measurement | Result |
|---|---|
| Physics vertices / tetrahedra | 9,312 / 39,275 |
| Surface vertices / boundary triangles | 3,455 / 6,906 |
| Visual vertices after UV and normal splits | 4,446 |
| Bounding box in m, min → max | (−0.032, −0.016, 0) → (0.032, 0.016, 0.066) |
| Total mass from signed volume × density | 0.039223572966 kg |
| Minimum signed tetrahedron volume | 1.288730163e-11 m³ |
| Maximum part-boundary edges, including interfaces | Handle 5.939697 mm, mount 5.971912 mm, foam 1.992443 mm, rubber 1.992443 mm |
| Maximum visual-to-physics rest error | 1.956751664e-09 m |
| Shared triangles across consecutive interfaces | Handle/mount 80, mount/foam 1,512, foam/die 1,512 |

All tetrahedra have positive signed volumes. The mesh has zero duplicate tetrahedra, duplicate vertex positions, unused vertices or faces with more than two owners. Each paired interior face has opposite winding. Face adjacency gives one connected solid. Each boundary edge has two incident triangles. Each boundary vertex has a single cyclic link. The boundary Euler characteristic is 2, and the signed volume enclosed by the boundary agrees with the summed tet volume.

A bounding-box sweep and tetrahedron separating-axis test checked 2,388,604 candidate pairs with zero intersection violations, using a 10⁻⁹ mm geometric tolerance. The test checks for positive-volume overlaps between incident tetrahedra and intersections between disjoint tetrahedra. Centroid classification agrees with every Medit and NPZ part label. Every exported per-tet elastic constant and density equals the value in its material prediction. All requested array keys, data types, shapes, owner indices, binding indices, UVs, render triangles and rest positions were checked on readback. The setting `behavior_validated: false` records that verification covers the rest asset and its data.

## SHA-256

Each entry hashes the file's exact bytes. The list was rewritten on import into the form the catalog's tools read; the digests are the delivered ones.

- `appearance/albedo.png`: `fbf552d392a869f521bb92d2415243938d657e20d9e34a69d1d35188e00a6599`
- `appearance/visual_mesh.glb`: `55080dbf3300d3525ff9dc1bcbbac31856248bf5d5bc924565ce2719c787b4a0`
- `appearance/visual_to_physics.npz`: `641247e28010748c3be275158ead7d5201fdd317e771bbb359692ac5d77ddd54`
- `asset.json`: `1bd83355d10b00b3bd26acafae4b0ec688d142fe0a0792a0f840a3961a956bc3`
- `final_mesh.mesh`: `5a24523fbe47a585699efa20d93877db5473f001e83ddff4ecacf640df5d1543`
- `heterogeneous_params.npz`: `6033293c025488bd0c02e166eb62d76f75fb1244e1ff24ec154a9d2e160897f1`
- `inferred_params.json`: `9cebf3f51fc58712b34da17a28498cb27ff852de41a1995fadb17aa8d81a8bf5`
