# cloth_cleaning_wand

Re-imported from `generated_tool_assets/assets/cloth_cleaning_wand` (regenerated finer meshes; generated_tool_assets HEAD `66768d3` plus its uncommitted working-tree regeneration, 2026-11-16; schema rigid-cloth-asset: cloth mid-surface, rigid meshes, attachments). Loaded by `cloth_hybrid.py`.

Star-shaped cloth duster: eight medium sheets attached along a slim central connector, with free outer edges and tips.

Original procedural geometry generated in WSL by `../../tools/generate_hybrid_assets.py`. No third-party mesh or texture copied.

Geometry reference: https://www.oxo.com/microfiber-delicate-duster.html

Design differences: User-specified star arrangement of eight medium cloth sheets. OXO hand/delicate dusters inform the grip and cloth-tool concept; their microfiber construction is not copied. Each continuous panel attaches along its inner root edge; outer edges and the last 27 mm of the tip are free. No fringe cuts or hairlike strips. A closed 0.8 mm visual skin follows the membrane normals. Filter attached root particles against their bound body during future integration.

Only static checks performed. Material values are provisional estimates; no behavior, loading, electrical or thermal validation. Rigid groups and cloth attachments require explicit scene integration later.

## SHA-256

- `appearance/albedo.png`: `a1005f9d19097e7101df13eab277dcc694f864137d00ea9779d05125ed4912b3`
- `appearance/cloth_skin.npz`: `e4b20241e926fea09afdd976da93364f2a71c8cb5a1758d813e2d9fc2a360e7d`
- `appearance/visual_mesh.glb`: `284648fa90630d5defdaf691a7d65ebaa7be695326b41356c6bf68a80ebe54cd`
- `appearance/visual_to_physics.npz`: `9501164c7d999a958062a5f6db887cfb186e385b7c18722a9243d122607007c3`
- `asset.json`: `4a947f6db7bf74d5d9c3b256011c23ef7464288175534a96ab09b4948e31493e`
- `manifest.json`: `02f4ff51c129b109f4e4636537e38bae509562805387676258d3c1beb6ab84b0`
- `materials.json`: `c028044ca3d97e86ef2e9c3265e52a19f7ed5e8de534d2cb8e33d110d1fa5080`
- `physics/attachments.json`: `2cbea5257469c0a4aaa4259b56cbde0a6302a6b208bb424d437f84b1324270d8`
- `physics/attachments.npz`: `52eb3ef84392d79acc38e71d6989262178137a28c0415c35001fc032ba88fb81`
- `physics/cloth.obj`: `4a7cf0a1b461b15bbfa1cb1d449956570780f0f57089c539423bddce75ae9fc0`
- `physics/cloth_mesh.npz`: `e56895e5ca67a0ac43e17c14493c93a6a34385fbe213dbc3aa05e5d083a04c64`
- `physics/rigid_1.npz`: `3e5e955d16e25c923a7bde581dbedf6cfb34ea79b3d4197d230c44d37ec51e72`
- `physics/rigid_1.obj`: `c45a175a8e57467a2d6c449be4a5c9787b61ba3f0556dfee0bdf4f1c2c4febb7`
- `physics/rigid_1_tets.npz`: `bd3f00aa75f4735787a1eec47dfbca3eaff8b2cce0e1849f2c72aa3b0607528f`
- `validation.json`: `febcbd270f4fcaec34fe73202a6dc676f67241cdb81efc6e10d0f203108510e9`
