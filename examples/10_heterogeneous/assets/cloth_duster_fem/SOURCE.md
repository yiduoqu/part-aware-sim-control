# cloth_duster_fem

Converted by `cloth_to_tets.py` from `assets/cloth_cleaning_wand` (cloth mid-surface plus rigid meshes): cloth as a 2.5 mm solid shell (E 500000 Pa), rigid parts as solids, one TetGen volume. 28667 tets; labels: 0 cloth, 1 rigid. Not behaviour-validated.

## SHA-256

- `final_mesh.mesh`: `c7e3628965cb48036e4b5263ccfafbb96c57f06fb2f2800e3267bddb50d37056`
- `heterogeneous_params.npz`: `278fb9230a246f975bb609ed0a0b03d4d49224df1ec1fa1e557b876a54e3a35f`
- `inferred_params.json`: `12d8ca0d183319df6b36023a17c307386d02129e719bc011e3f3f56e9cf6a91a`
- `asset.json`: `79490dc7b8323b73b5f5e2df841039b6f41cc097ddefe719d466c1bde55a1a68`

Command (from `examples/10_heterogeneous`):

```
uv run --with tetgen --with fast-simplification python cloth_to_tets.py assets/cloth_cleaning_wand assets/cloth_duster_fem --thickness 0.0025 --young 5e5 --pad-box -1 -1 -1 0 1 1
```
