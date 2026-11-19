"""Build the ``soap_bar`` bundle: the bar that goes in ``hinged_soap_case``.

    python examples/10_heterogeneous/assets/generate_soap_bar.py

One part, one material. The grid, the tetrahedra and the file formats are
:mod:`generate_hinged_soap_case`'s; only the shape and the numbers are this file's.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import generate_hinged_soap_case as grid
import numpy as np

HERE = Path(__file__).resolve().parent
NAME = "soap_bar"

# Millimetres; the mesh is written in metres.
HALF_X, HALF_Y, HEIGHT = 45.0, 27.5, 25.0
CORNER_R = 10.0

PARTS = (("soap", "bar soap", 5.0e7, 0.40, 1050.0, 0.35, (238, 229, 205)),)
RIGID_GROUPS = [[0]]


def axis_lines(half: float) -> np.ndarray:
    """Lines across one plan axis: five cells over each corner radius, about 6 mm between them."""
    flat = half - CORNER_R
    inner = np.linspace(-flat, flat, 2 * max(1, round(flat / 6.0)) + 1)
    corner = flat + np.linspace(0.0, CORNER_R, 5)[1:]
    return np.r_[-corner[::-1], inner, corner]


def build_volume() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, zs = axis_lines(HALF_X), axis_lines(HALF_Y), np.linspace(0.0, HEIGHT, 9)
    labels = np.zeros((len(xs) - 1, len(ys) - 1, len(zs) - 1), dtype=np.int64)
    return grid.grid_volume(xs, ys, zs, labels, lambda points: grid.round_corners(points, HALF_X, HALF_Y, CORNER_R))


def source_md(facts: dict, folder: Path) -> str:
    name, material, young, nu, rho, mu, _rgb = PARTS[0]
    lo, hi = facts["bounds_m"]
    hashes = "\n".join(
        f"- `{p.relative_to(folder)}`: `{hashlib.sha256(p.read_bytes()).hexdigest()}`"
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "SOURCE.md"
    )
    return f"""# Soap bar

The bar for [`hinged_soap_case`](../hinged_soap_case/SOURCE.md): {2 * HALF_X:.0f} x {2 * HALF_Y:.0f} x {HEIGHT:.0f} mm with a
{CORNER_R:.0f} mm plan-view corner radius, one part and one material. Built by `../generate_{NAME}.py` on the case
generator's grid. It follows no reference photo and no particular product.

Coordinates are in metres, Z up. The origin is the centre of the bar's lower face, X its long axis.

| Index | Part | Material | E in Pa | Poisson ratio | Density in kg/m^3 | Friction | Tetrahedra | Mass in g |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | {name} | {material} | {young:.1e} | {nu:.2f} | {rho:.0f} | {mu:.2f} | {facts["per_part"][0]:,} | {1e3 * facts["mass_kg"]:.1f} |

Rigid groups `{json.dumps(RIGID_GROUPS)}`: the whole bar is one rigid body, which is how the scenes run it. The
modulus is a starting estimate for a dry bar, not a measurement; a wet or worn bar is neither modelled.
No simulation was run when the bundle was written.

## Verification from the written files

| Measurement | Result |
|---|---|
| Physics vertices / tetrahedra | {facts["vertices"]:,} / {facts["tetrahedra"]:,} |
| Boundary triangles | {facts["boundary_faces"]:,} |
| Bounding box in m, min to max | ({lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}) to ({hi[0]:.4f}, {hi[1]:.4f}, {hi[2]:.4f}) |
| Smallest / largest tetrahedron | {1e9 * facts["min_volume_m3"]:.3f} / {1e9 * facts["max_volume_m3"]:.3f} mm^3 |
| Dihedral angles | {facts["dihedral_deg"][0]:.1f} to {facts["dihedral_deg"][1]:.1f} degrees |
| Summed tet volume / volume enclosed by the boundary | {1e6 * facts["volume_m3"]:.6f} / {1e6 * facts["enclosed_volume_m3"]:.6f} cm^3 |
| Boundary Euler characteristic | {facts["boundary_euler_characteristic"]} |

Every tetrahedron has positive signed volume, no face has more than two owners, and every boundary edge
has exactly two boundary triangles.

## SHA-256

{hashes}
"""


def main() -> None:
    folder = HERE / NAME
    (folder / "appearance").mkdir(parents=True, exist_ok=True)
    points, tets, tet_labels = build_volume()
    faces, owner = grid.boundary(tets)
    facts = grid.measure(points, tets, tet_labels, faces, PARTS)
    name, material, young, nu, rho, mu, rgb = PARTS[0]

    (folder / "final_mesh.mesh").write_text(grid.medit(points, tets, tet_labels))
    np.savez_compressed(
        folder / "heterogeneous_params.npz",
        part_indices=np.zeros(1, dtype=np.int64),
        part_E_nu=np.array([[young, nu]]),
        part_density=np.array([rho]),
        part_colors=np.array([rgb], dtype=np.float64),
        part_volume_fill_mode_codes=np.zeros(1, dtype=np.int64),
        tet_E_nu=np.tile([young, nu], (len(tets), 1)),
        tet_density=np.full(len(tets), rho),
        tet_part_labels=tet_labels.astype(np.int64),
    )
    prediction = {
        "part_name": name,
        "major_material_name": material,
        "part_color_rgb": list(rgb),
        "youngs_modulus_pa": young,
        "poisson_ratio": nu,
        "density_kg_m3": rho,
        "friction_coefficient": mu,
        "part_index": 0,
        "volume_fill_mode": "solid_fill",
    }
    materials = {
        "schema_version": "generated-heterogeneous-materials-v1",
        "inferred_object_name": NAME,
        "num_parts": 1,
        "predictions": [prediction],
        "notes": "Engineering starting estimates, not measured or simulation-calibrated.",
    }
    (folder / "inferred_params.json").write_text(json.dumps(materials, indent=2) + "\n")
    asset = {
        "name": NAME,
        "description": "90 x 55 x 25 mm bar of soap with rounded plan corners, to go in hinged_soap_case.",
        "units": "m",
        "up_axis": "Z",
        "generator": f"generate_{NAME}.py (procedural grid of hexahedra cut into tetrahedra, cell part labels)",
        "construction": "Tensor-product hexahedral grid, six tetrahedra per cell, plan corners rounded by moving vertices",
        "rigid_groups": RIGID_GROUPS,
        "source_assets": [],
        "behavior_validated": False,
        "metadata": {
            "origin": "Centre of the lower face at z=0",
            "goes_with": "hinged_soap_case",
            "tuning_status": "Source estimates; run as one rigid body.",
        },
    }
    (folder / "asset.json").write_text(json.dumps(asset, indent=2) + "\n")

    png = grid.palette_png(PARTS)
    binding = grid.visual_binding(points, faces, owner, tet_labels, PARTS)
    (folder / "appearance" / "albedo.png").write_bytes(png)
    (folder / "appearance" / "visual_mesh.glb").write_bytes(
        grid.glb_bytes(binding["visual_rest_vertices_m"], binding["visual_faces"], binding["visual_uv"], png, NAME)
    )
    np.savez_compressed(folder / "appearance" / "visual_to_physics.npz", **binding)
    (folder / "SOURCE.md").write_text(source_md(facts, folder))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
