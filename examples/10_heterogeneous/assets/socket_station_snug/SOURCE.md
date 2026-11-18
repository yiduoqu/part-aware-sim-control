# socket_station_snug source asset

Re-imported from `generated_tool_assets/assets/socket_station_snug` (regenerated finer meshes; generated_tool_assets HEAD `66768d3` plus its uncommitted working-tree regeneration, 2026-11-16). Procedural geometry, Manifold union + TetGen, centroid part labels. Static mesh checks only there; behaviour is checked by the scenes here.

Original procedural geometry generated in WSL by `../../tools/generate_sockets.py`. No copied mesh. Entirely rigid task fixture; simplified mechanical socket, no electrical internals. Static geometry checks only.

- Fully rigid socket block: `physics/collision_mesh.{obj,npz}` keeps the concave cavity (do not replace it by a convex hull).
- `connectors.json`: the socket's mating frame.

## SHA-256

- `appearance/albedo.png`: `879634d9af576396a945f6f53733560bc1fc1e51878c28f63cf3a75d26612263`
- `appearance/visual_mesh.glb`: `19e6e2b7cccf9f919a124eb37894a61a4bf1955917fe77208c6e404848528dfb`
- `appearance/visual_to_physics.npz`: `5524d56a20fa3f47b5625cebdf2615091ecaae5128161cf1c743134588838ba1`
- `asset.json`: `ff839aab9761c021f4388986453a2c4d48488747ded77498d7027c551050b52c`
- `connectors.json`: `c3a02cd34b5e032c9a86af9a32ef7cc2ec776d5af44a68d12e8dd266f4d4fdc1`
- `materials.json`: `3aa5ae90ca249ac077903e525a9c2ec0fcb170ff13619c5b0eee83187704fbe1`
- `physics/collision_mesh.npz`: `7b5b4323f416785d82a296761788f2ef2d90969fa403516393293e78ff0cd055`
- `physics/collision_mesh.obj`: `7a975bf7880f59c7384f7d054bad8e3758d4ae46f2b79ac029d616189f57f226`
- `physics/rigid_tets.npz`: `a22b0f30de949db3d3ca0c112eb216fa830f1e2e6b5276ff9539d42cf0a3cadc`
