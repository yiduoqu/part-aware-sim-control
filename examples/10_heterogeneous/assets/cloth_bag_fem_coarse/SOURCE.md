# cloth_bag_fem_coarse

Converted by `cloth_to_tets.py` from `assets/handled_fabric_bag` (cloth mid-surface plus rigid meshes): cloth as a 3 mm solid shell (E 100000 Pa), rigid parts as solids, one TetGen volume. 48225 tets; labels: 0 cloth, 1, 2 rigid. Not behaviour-validated.

## SHA-256

- `final_mesh.mesh`: `84b706c29387f691fbe2006a34fd920b2b5611dd2b7226aecc4f2aec732a6ebc`
- `heterogeneous_params.npz`: `556db119320117ff76a7d2a230cc8b8480b4dc94575f928d9e6b027d7dab81f7`
- `inferred_params.json`: `289cf6df41c5b18b973d58ecc487b71473a69f6e4e5b3b35bd72b6d079cd8d21`
- `asset.json`: `9807a0d4222cd2d29fc641311e838eaf676f870ff891f4b66477987287f8ca35`

Command (from `examples/10_heterogeneous`); cloth decimated to 4000 faces, rigid meshes to 6000 (fewer breaks the rim), base board (body 2) 2.5x thicker (6 to 15 mm) about its underside; `--sliver` 1e-5 is the largest that keeps one connected volume:

```
uv run --with tetgen --with fast-simplification python cloth_to_tets.py assets/handled_fabric_bag assets/cloth_bag_fem_coarse --thickness 0.003 --young 1e5 --cloth-faces 4000 --rigid-faces 6000 --pad-box -1 -1 0.33 1 1 1 --pad-edge 0.01 --thicken 2 2.5 --sliver 1e-5
```
