# cloth_bag_fem

Converted by `cloth_to_tets.py` from `assets/handled_fabric_bag` (cloth mid-surface plus rigid meshes): cloth as a 3 mm solid shell (E 100000 Pa), rigid parts as solids, one TetGen volume. 111476 tets; labels: 0 cloth, 1, 2 rigid. Not behaviour-validated.

## SHA-256

- `final_mesh.mesh`: `d2253e7474cfc5801d8c0523b51f589b8396778d58ca50b705c709a76380f6c9`
- `heterogeneous_params.npz`: `6e6c7fe9e43781d0e8839fee4f39109ef8382350bfbfc7bc83fb4490a46eb117`
- `inferred_params.json`: `289cf6df41c5b18b973d58ecc487b71473a69f6e4e5b3b35bd72b6d079cd8d21`
- `asset.json`: `c9be504ae92e1a02f3f6a01a8b917e85bad3893b25f6fb7e7be51fa273690579`

Command (from `examples/10_heterogeneous`); cloth decimated to 4000 faces, rigid meshes kept at their regenerated resolution:

```
uv run --with tetgen --with fast-simplification python cloth_to_tets.py assets/handled_fabric_bag assets/cloth_bag_fem --thickness 0.003 --young 1e5 --cloth-faces 4000 --pad-box -1 -1 0.33 1 1 1 --pad-edge 0.01
```
