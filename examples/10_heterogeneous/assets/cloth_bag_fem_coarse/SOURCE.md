# cloth_bag_fem_coarse

Converted by `cloth_to_tets.py` from `assets/handled_fabric_bag` (cloth mid-surface plus rigid meshes): cloth as a 5 mm solid shell (E 100000 Pa), rigid parts as solids, one TetGen volume. 71787 tets; labels: 0 cloth, 1, 2 rigid. Not behaviour-validated.

## SHA-256

- `final_mesh.mesh`: `3f51647a75dc4640ce6b1084f9e7b914b9c3e91dd1631ed50935e8ab1e158192`
- `heterogeneous_params.npz`: `c41f61817676950009aa449164a3e43770eb8d5b9b9bfc33fc71c79cb5f0b689`
- `inferred_params.json`: `cf03f69be53644ac8742e95c3a426123cd27e5c92ee64b85626ec2d85e0dbbd8`
- `asset.json`: `c81f57f6dad8dc1b3d7249d201dfa14a0a4fc87f860071e20079caa9120a75a2`

Command (from `examples/10_heterogeneous`); cloth remeshed isotropically to 8 mm edges, rigid meshes decimated to 6000 faces (fewer breaks the rim), base board (body 2) 2.5x thicker (6 to 15 mm) about its underside; tets under quality 0.01 dropped (min quality 0.01 in every label):

```
uv run --with tetgen --with fast-simplification --with pymeshlab python cloth_to_tets.py assets/handled_fabric_bag assets/cloth_bag_fem_coarse --thickness 0.005 --young 1e5 --cloth-edge 0.008 --rigid-faces 6000 --pad-box -1 -1 0.33 1 1 1 --pad-edge 0.01 --thicken 2 2.5 --sliver 1e-2
```
