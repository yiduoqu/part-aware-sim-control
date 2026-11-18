# socket_station_loose source asset

Re-imported from `generated_tool_assets/assets/socket_station_loose` (regenerated finer meshes; generated_tool_assets HEAD `66768d3` plus its uncommitted working-tree regeneration, 2026-11-16). Procedural geometry, Manifold union + TetGen, centroid part labels. Static mesh checks only there; behaviour is checked by the scenes here.

Original procedural geometry generated in WSL by `../../tools/generate_sockets.py`. No copied mesh. Entirely rigid task fixture; simplified mechanical socket, no electrical internals. Static geometry checks only.

- Fully rigid socket block: `physics/collision_mesh.{obj,npz}` keeps the concave cavity (do not replace it by a convex hull).
- `connectors.json`: the socket's mating frame.

## SHA-256

- `appearance/albedo.png`: `879634d9af576396a945f6f53733560bc1fc1e51878c28f63cf3a75d26612263`
- `appearance/visual_mesh.glb`: `d95571f9197e2c080834ad3716278e97488d268c56c0c33fe568c6134f832944`
- `appearance/visual_to_physics.npz`: `da7d5d5d01274201f1dec8245ed10745595c3e87f7c7255e9b1ac29e3c8bda50`
- `asset.json`: `58a50df1571670b82213a4dd841ca2048cc1d2a4ea869f8a975ea61323eb3dad`
- `connectors.json`: `cc2b334e8f79e811774227af20d189dfc054bc0c28414d823a14c28e222f85d1`
- `materials.json`: `3aa5ae90ca249ac077903e525a9c2ec0fcb170ff13619c5b0eee83187704fbe1`
- `physics/collision_mesh.npz`: `5e72cc856459138cd642fabdadf1e2f737e6e0042aa90277cb8669ed2a21f45a`
- `physics/collision_mesh.obj`: `ef904e98683082a0438233c241e62082bf58c9b4a24efc74a2f48dc3a7da151e`
- `physics/rigid_tets.npz`: `b3d2dca3b4a896f016a620f668626094205a3b62d654a289665c21bb4f6c7d09`
