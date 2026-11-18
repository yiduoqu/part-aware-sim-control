# buckled_strap

Re-imported from `generated_tool_assets/assets/buckled_strap` (regenerated finer meshes; generated_tool_assets HEAD `66768d3` plus its uncommitted working-tree regeneration, 2026-11-16; schema rigid-cloth-asset: cloth mid-surface, rigid meshes, attachments). Loaded by `cloth_hybrid.py`.

35 mm-wide, 2.5 mm-thick woven strap loop with an open-slot metal buckle and deformable webbing.

Original procedural geometry generated in WSL by `../../tools/generate_hybrid_assets.py`. No third-party mesh or texture copied.

Geometry reference: https://seatosummit.com/products/accessory-straps-hook-release

Design differences: Generic two-slot ladder buckle inspired by accessory-strap hardware; wider and thicker than the cited 20 mm strap. Fixed fastened ends, no sliding adjustment or loose tail. Physical membrane has thickness/collision radius and areal mass; a closed 2.5 mm render skin follows its normals.

Only static checks performed. Material values are provisional estimates; no behavior, loading, electrical or thermal validation. Rigid groups and cloth attachments require explicit scene integration later.

## SHA-256

- `appearance/albedo.png`: `f18254316d62a68175fcede06909ff58e7d31fcbb9d9189d71a4e1616adb98bd`
- `appearance/cloth_skin.npz`: `6487a1df28cbfa3bffa0c9196d65314709849406775f7702f9d7fe73c58202a1`
- `appearance/visual_mesh.glb`: `897068ba4e38d561fd67fd7743c98853d9149b9c707e8db1a834c47be82f5a1d`
- `appearance/visual_to_physics.npz`: `26590ee698727c6654feed434670acb545754163f0f91ea19af51b53a39df153`
- `asset.json`: `d1f931fcdca0956b3d5bbf985c38600e5e3f6397a7fb9dfaa34864ebcff25de6`
- `materials.json`: `fe2e0d21822ce88b535d4baf24b77ee687e773a551a36f99cb615604e472699e`
- `physics/attachments.json`: `0bb868bb3614d7cd245145637341120427315aa3cfb14da3953dc39054c785f9`
- `physics/attachments.npz`: `b4947e90c94b08b3b8775886adedeb15fc811c141209f47232cb9eb646180d12`
- `physics/cloth.obj`: `ee253989171d98b88673294a1602a37175351468a31f7c56d3ffb7a56f313fc8`
- `physics/cloth_mesh.npz`: `d02df263993a6456256e0d6667e6db3436d98157537d044a8ff50f1647ab95b2`
- `physics/rigid_1.npz`: `e8f8085791997dc4959d78e41e108782d697490e624241ca2ea509090807caf9`
- `physics/rigid_1.obj`: `bd8eaffea6217b814e4821cc771512b6f7f63efceb5b51c6bd37c386ac32e038`
- `physics/rigid_1_tets.npz`: `8e6918c7b143c000305b717c8f5805e47b5be424090b8e42e92707642af35061`
