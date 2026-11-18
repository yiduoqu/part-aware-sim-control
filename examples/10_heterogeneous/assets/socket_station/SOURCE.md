# socket_station source asset

Re-imported from `generated_tool_assets/assets/socket_station` (regenerated finer meshes; generated_tool_assets HEAD `66768d3` plus its uncommitted working-tree regeneration, 2026-11-16). Procedural geometry, Manifold union + TetGen, centroid part labels. Static mesh checks only there; behaviour is checked by the scenes here.

Original procedural geometry generated in WSL by `../../tools/generate_sockets.py`. No copied mesh. Entirely rigid task fixture; simplified mechanical socket, no electrical internals. Static geometry checks only.

- Fully rigid socket block: `physics/collision_mesh.{obj,npz}` keeps the concave cavity (do not replace it by a convex hull).
- `connectors.json`: the socket's mating frame.

## SHA-256

- `appearance/albedo.png`: `879634d9af576396a945f6f53733560bc1fc1e51878c28f63cf3a75d26612263`
- `appearance/visual_mesh.glb`: `abff69d0f98ef9f1fd8ad8588c901673d38d978dfda1578fcd3f7c2dbed070db`
- `appearance/visual_to_physics.npz`: `e475148caabe66231018f374089d204e51e18e04845c8601dfd677aa72de5b3e`
- `asset.json`: `6c242bdfbf66aba6b854abd09b8c86247cf6acabedfd12283265359730afacd6`
- `connectors.json`: `5ec0d7b206816734ef8a212679298e09b8a38484f57dec2ab3d6fcab316102b7`
- `materials.json`: `3aa5ae90ca249ac077903e525a9c2ec0fcb170ff13619c5b0eee83187704fbe1`
- `physics/collision_mesh.npz`: `19181ccf0295f825fbc8865f6a97d4d976cd02dfef7a63e5735263000ee88044`
- `physics/collision_mesh.obj`: `5a02dd2855c32140d696a8c553e12e9fa6d5b3acf60c0984ccf0f1266ec04d70`
- `physics/rigid_tets.npz`: `7d5ce131da88649a7a26b268693c9ec0d13f3a5c7a76ea8044334b5c00002fac`
