# cloth_bag_fem_coarse

Converted by `cloth_to_tets.py` from `assets/handled_fabric_bag` (cloth mid-surface plus rigid meshes): cloth as a 3 mm solid shell (E 100000 Pa), rigid parts as solids, one TetGen volume. 40904 tets; labels: 0 cloth, 1, 2 rigid. Not behaviour-validated.

## SHA-256

- `final_mesh.mesh`: `d3c1a6a574445d3505d136423b6e3db5c5bd2c441e4ea90cc8c1834e786878ce`
- `heterogeneous_params.npz`: `376facf8b993fdd89c0b757a07e65cfc98b5055132f0b42fb03e8a3cce0807ee`
- `inferred_params.json`: `289cf6df41c5b18b973d58ecc487b71473a69f6e4e5b3b35bd72b6d079cd8d21`
- `asset.json`: `e972b720376a1f98fa6017290dcf979dd1fa776538203923aa77fcac038af1fa`

Command (from `examples/10_heterogeneous`); cloth decimated to 1500 faces, rigid meshes to 6000 (fewer breaks the rim), base board (body 2) 2.5x thicker (6 to 15 mm) about its underside:

```
uv run --with tetgen --with fast-simplification python cloth_to_tets.py assets/handled_fabric_bag assets/cloth_bag_fem_coarse --thickness 0.003 --young 1e5 --cloth-faces 1500 --rigid-faces 6000 --pad-box -1 -1 0.33 1 1 1 --pad-edge 0.01 --thicken 2 2.5
```
