"""Build the ``hinged_soap_case`` bundle: a travel soap case whose lid is fused to the tray by a thin web.

    python examples/10_heterogeneous/assets/generate_hinged_soap_case.py

Needs only numpy and Pillow. The volume is a tensor-product grid of hexahedra, each cut into six
tetrahedra along the diagonal that leaves the case's vertical axis, so the mesh is conforming without a
mesher; the plan-view corners are rounded by moving vertices, never by cutting cells. It writes the same
files as the other generated bundles, then ``SOURCE.md`` with what it measured on the way out.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
NAME = "hinged_soap_case"

# Every length below is in millimetres; the mesh is written in metres.
HALF_X, HALF_Y = 50.0, 32.0  # outer plan half-extents of tray and lid
WALL = 2.0  # walls, floor and lid top, two cells through
CORNER_R = 12.0  # outer plan-view corner radius
# A 25 mm bar stands 11 mm out of the tray, so a gripper can take it by its sides, and the lid is no
# taller than the hinge is high, so it can hang open behind the tray without lifting it.
BASE_TOP, LID_BOTTOM, LID_TOP = 16.0, 17.0, 32.0  # the rims face each other across a 1 mm gap
HINGE_HALF_X = 24.0  # the web and its knuckles run along X behind the -Y wall
HINGE_CLEAR, HINGE_THICK = 1.0, 1.0  # gap between wall and web, and the web's thickness
WEB_Z = (13.0, 20.0)  # the free length of the web, centred on the gap between the rims
KNUCKLE = 3.0  # height of the blocks that join the web to the tray and to the lid
TAB_HALF_X, TAB_DEPTH, TAB_THICK = 12.0, 8.0, 2.0  # the thumb tab on the lid's +Y rim

PARTS = (
    # name, material, E [Pa], nu, density [kg/m^3], friction, sRGB
    ("base_tray", "polypropylene", 1.5e9, 0.42, 910.0, 0.30, (104, 170, 210)),
    ("lid", "polypropylene", 1.5e9, 0.42, 910.0, 0.30, (150, 202, 232)),
    ("living_hinge", "polypropylene web (bending-equivalent)", 4.0e7, 0.42, 910.0, 0.30, (58, 112, 160)),
    ("lid_tab", "polypropylene", 1.5e9, 0.42, 910.0, 0.30, (196, 228, 244)),
)
RIGID_GROUPS = [[0], [1, 3]]


def grid_lines() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    back = -HALF_Y - HINGE_CLEAR
    xs = np.r_[-HALF_X, -HALF_X + 1.0, np.arange(-HALF_X + WALL, HALF_X - WALL + 0.5, 3.0), HALF_X - 1.0, HALF_X]
    ys = np.r_[
        back - HINGE_THICK,
        back - HINGE_THICK / 2.0,
        back,
        -HALF_Y,
        -HALF_Y + 1.0,
        np.arange(-HALF_Y + WALL, HALF_Y - WALL + 0.5, 3.0),
        HALF_Y - 1.0,
        HALF_Y,
        HALF_Y + np.arange(2.0, TAB_DEPTH + 0.5, 2.0),
    ]
    zs = np.r_[
        0.0,
        1.0,
        np.linspace(WALL, WEB_Z[0] - KNUCKLE, 3),
        np.arange(WEB_Z[0], WEB_Z[1] + 0.5, 1.0),
        np.linspace(WEB_Z[1] + KNUCKLE, LID_TOP - WALL, 5),
        LID_TOP - 1.0,
        LID_TOP,
    ]
    return xs, ys, zs


def cell_labels(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Part index of every grid cell, or -1 where the cell is empty; decided at the cell centre."""
    cx, cy, cz = np.meshgrid(*((a[:-1] + a[1:]) / 2.0 for a in (xs, ys, zs)), indexing="ij")
    plan = (np.abs(cx) < HALF_X) & (np.abs(cy) < HALF_Y)
    hollow = (np.abs(cx) < HALF_X - WALL) & (np.abs(cy) < HALF_Y - WALL)
    behind = (cy < -HALF_Y) & (np.abs(cx) < HINGE_HALF_X)
    labels = np.full(cx.shape, -1, dtype=np.int64)
    labels[plan & (cz < BASE_TOP) & ((cz < WALL) | ~hollow)] = 0
    labels[plan & (cz > LID_BOTTOM) & ((cz > LID_TOP - WALL) | ~hollow)] = 1
    labels[behind & (cz > WEB_Z[0] - KNUCKLE) & (cz < WEB_Z[0])] = 0
    labels[behind & (cz > WEB_Z[1]) & (cz < WEB_Z[1] + KNUCKLE)] = 1
    labels[behind & (cy < -HALF_Y - HINGE_CLEAR) & (cz > WEB_Z[0]) & (cz < WEB_Z[1])] = 2
    labels[(cy > HALF_Y) & (np.abs(cx) < TAB_HALF_X) & (cz > LID_BOTTOM) & (cz < LID_BOTTOM + TAB_THICK)] = 3
    return labels


def round_corners(
    points: np.ndarray, half_x: float = HALF_X, half_y: float = HALF_Y, radius: float = CORNER_R
) -> None:
    """Turn the plan's square corners into arcs, keeping the wall two cells thick around them.

    In each corner square of side ``radius`` a point moves along its own ray from the square's inner
    corner, so the lines ``max(u, v) = c`` become arcs of radius ``c``.
    """
    u = np.abs(points[:, 0]) - (half_x - radius)
    v = np.abs(points[:, 1]) - (half_y - radius)
    corner = (u > 0.0) & (v > 0.0)
    scale = np.maximum(u[corner], v[corner]) / np.hypot(u[corner], v[corner])
    points[corner, 0] = np.sign(points[corner, 0]) * (half_x - radius + u[corner] * scale)
    points[corner, 1] = np.sign(points[corner, 1]) * (half_y - radius + v[corner] * scale)


def build_volume() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertices [m], positively wound tetrahedra and their part labels."""
    xs, ys, zs = grid_lines()
    return grid_volume(xs, ys, zs, cell_labels(xs, ys, zs), round_corners)


def grid_volume(xs, ys, zs, labels: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut the labelled cells of a grid [mm] into tetrahedra; ``shape`` moves the grid's points in place."""
    ci, cj, ck = np.nonzero(labels >= 0)
    # Mirror the split across x = 0 and y = 0, so every cell's diagonal runs outward and the cells on a
    # rounded corner's bisector are cut along it instead of into slivers.
    flip_x = (xs[ci] + xs[ci + 1]) < 0.0
    flip_y = (ys[cj] + ys[cj + 1]) < 0.0
    ny, nz = len(ys), len(zs)

    def corner(a: int, b: int, c: int) -> np.ndarray:
        return ((ci + np.where(flip_x, 1 - a, a)) * ny + cj + np.where(flip_y, 1 - b, b)) * nz + ck + c

    tets = []
    for order in itertools.permutations(range(3)):
        step = [0, 0, 0]
        path = [corner(0, 0, 0)]
        for axis in order:
            step[axis] = 1
            path.append(corner(*step))
        tets.append(np.stack(path, axis=1))
    tets = np.stack(tets, axis=1).reshape(-1, 4)
    tet_labels = np.repeat(labels[ci, cj, ck], 6)

    points = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    shape(points)
    used, tets = np.unique(tets, return_inverse=True)
    tets = tets.reshape(-1, 4)
    points = points[used] * 1e-3
    negative = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]]) < 0.0
    tets[negative, :2] = tets[negative, 1::-1]
    return points, tets, tet_labels


def boundary(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outward-wound faces with exactly one owner, and the tetrahedron that owns each."""
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]])
    owner = np.tile(np.arange(len(tets)), 4)
    _, first, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if count.max() > 2:
        raise ValueError("a face has more than two owners")
    outer = np.sort(first[count == 1])
    return faces[outer], owner[outer]


def dihedral_range(points: np.ndarray, tets: np.ndarray) -> tuple[float, float]:
    """Smallest and largest dihedral angle over all tetrahedra [deg]."""
    p = points[tets]
    lo, hi = 180.0, 0.0
    for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)):
        edge = p[:, b] - p[:, a]
        edge /= np.linalg.norm(edge, axis=1, keepdims=True)
        n1 = p[:, c] - p[:, a]
        n2 = p[:, d] - p[:, a]
        n1 -= (n1 * edge).sum(axis=1, keepdims=True) * edge
        n2 -= (n2 * edge).sum(axis=1, keepdims=True) * edge
        cos = (n1 * n2).sum(axis=1) / (np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1))
        angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        lo, hi = min(lo, float(angle.min())), max(hi, float(angle.max()))
    return lo, hi


def palette_png(parts=PARTS) -> bytes:
    strip = np.repeat(np.array([p[6] for p in parts], dtype=np.uint8)[None, :, None, :], 64, axis=2)
    image = Image.fromarray(np.repeat(strip.reshape(1, -1, 3), 64, axis=0))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def glb_bytes(positions: np.ndarray, faces: np.ndarray, uvs: np.ndarray, png: bytes, name: str = NAME) -> bytes:
    """One textured mesh, identity node, positions left in the Z-up physics frame."""
    chunks = [faces.astype("<u4").tobytes(), positions.astype("<f4").tobytes(), png, uvs.astype("<f4").tobytes()]
    views, offset = [], 0
    for chunk in chunks:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(chunk)})
        offset += len(chunk) + (-len(chunk)) % 4
    binary = b"".join(chunk + b"\0" * ((-len(chunk)) % 4) for chunk in chunks)
    pos32 = positions.astype(np.float32)
    primitive = {"attributes": {"POSITION": 1, "TEXCOORD_0": 2}, "indices": 0, "mode": 4, "material": 0}
    surface = {"baseColorTexture": {"index": 0}, "roughnessFactor": 0.45, "metallicFactor": 0.0}
    index_range = {"min": [0], "max": [int(faces.max())]}
    position_range = {"min": pos32.min(axis=0).tolist(), "max": pos32.max(axis=0).tolist()}
    doc = {
        "asset": {"version": "2.0", "generator": f"generate_{name}.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [{"name": name, "primitives": [primitive]}],
        "materials": [{"pbrMetallicRoughness": surface, "doubleSided": False}],
        "textures": [{"source": 0}],
        "images": [{"bufferView": 2, "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "type": "SCALAR", "count": int(faces.size), **index_range},
            {"bufferView": 1, "componentType": 5126, "type": "VEC3", "count": len(pos32), **position_range},
            {"bufferView": 3, "componentType": 5126, "type": "VEC2", "count": len(uvs)},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * ((-len(text)) % 4)
    header = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(text) + 8 + len(binary))
    json_chunk = struct.pack("<I4s", len(text), b"JSON") + text
    return header + json_chunk + struct.pack("<I4s", len(binary), b"BIN\0") + binary


def visual_binding(
    points: np.ndarray, faces: np.ndarray, owner: np.ndarray, tet_labels: np.ndarray, parts=PARTS
) -> dict:
    """Three unshared visual vertices per boundary face, each pointing at the palette cell of its part."""
    surface = np.unique(faces)
    to_surface = np.searchsorted(surface, faces.reshape(-1))
    part = tet_labels[owner]
    uv = np.stack([(np.repeat(part, 3) + 0.5) / len(parts), np.full(3 * len(faces), 0.5)], axis=1)
    return {
        "visual_rest_vertices_m": points[faces.reshape(-1)].astype(np.float32),
        "visual_faces": np.arange(3 * len(faces), dtype=np.int32).reshape(-1, 3),
        "visual_uv": uv.astype(np.float32),
        "surface_vertex_to_physics_vertex": surface.astype(np.int64),
        "visual_vertex_to_surface_vertex": to_surface.astype(np.int64),
        "physics_vertex_indices": faces.reshape(-1).astype(np.int64),
        "boundary_face_tet_indices": owner.astype(np.int64),
        "boundary_face_part_ids": part.astype(np.int32),
    }


def medit(points: np.ndarray, tets: np.ndarray, tet_labels: np.ndarray) -> str:
    lines = ["MeshVersionFormatted 1", "Dimension 3", "Vertices", str(len(points))]
    lines += [f"{x:.9g} {y:.9g} {z:.9g} 0" for x, y, z in points]
    lines += ["Tetrahedra", str(len(tets))]
    lines += [f"{a} {b} {c} {d} {ref}" for (a, b, c, d), ref in zip(tets + 1, tet_labels + 1, strict=True)]
    return "\n".join([*lines, "End", ""])


def measure(
    points: np.ndarray, tets: np.ndarray, tet_labels: np.ndarray, faces: np.ndarray, parts=PARTS
) -> dict:
    """The static facts SOURCE.md reports; raises if the solid is not a single closed manifold."""
    volume = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]]) / 6.0
    if volume.min() <= 0.0:
        raise ValueError("non-positive tetrahedron")
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique_edges, per_edge = np.unique(edges, axis=0, return_counts=True)
    if not np.all(per_edge == 2):
        raise ValueError("the boundary is not a closed manifold")
    euler = len(np.unique(faces)) - len(unique_edges) + len(faces)
    tri = points[faces]
    enclosed = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    density = np.array([p[4] for p in parts])[tet_labels]
    lo, hi = dihedral_range(points, tets)
    return {
        "vertices": len(points),
        "tetrahedra": len(tets),
        "per_part": [int((tet_labels == i).sum()) for i in range(len(parts))],
        "part_mass_kg": [float((volume * density)[tet_labels == i].sum()) for i in range(len(parts))],
        "bounds_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "mass_kg": float((volume * density).sum()),
        "min_volume_m3": float(volume.min()),
        "max_volume_m3": float(volume.max()),
        "volume_m3": float(volume.sum()),
        "enclosed_volume_m3": enclosed,
        "dihedral_deg": [lo, hi],
        "boundary_faces": len(faces),
        "boundary_euler_characteristic": int(euler),
    }


def source_md(facts: dict, folder: Path) -> str:
    lo, hi = facts["bounds_m"]
    rows = "\n".join(
        f"| {i} | {name} | {material} | {young:.1e} | {nu:.2f} | {rho:.0f} | {mu:.2f} | {facts['per_part'][i]:,} "
        f"| {1e3 * facts['part_mass_kg'][i]:.2f} |"
        for i, (name, material, young, nu, rho, mu, _rgb) in enumerate(PARTS)
    )
    hashes = "\n".join(
        f"- `{p.relative_to(folder)}`: `{hashlib.sha256(p.read_bytes()).hexdigest()}`"
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "SOURCE.md"
    )
    inner = (2 * (HALF_X - WALL), 2 * (HALF_Y - WALL), LID_TOP - 2 * WALL)
    return f"""# Hinged soap case

A travel soap case in its closed pose: a tray and a lid of the same plan and about the same depth, joined
along the -Y side by a thin web that is part of the same mesh. Built by `../generate_{NAME}.py` (numpy only: a tensor-product
grid of hexahedra, six tetrahedra each, corners rounded by moving vertices). It follows no reference photo
and no particular product; the dimensions are ordinary ones for this kind of case.

Coordinates are in metres, Z up. The origin is the centre of the tray's footprint on the plane z = 0. X is
the long axis, the hinge is on -Y and the thumb tab on +Y. The lid opens by turning about X, its top going
toward -Y.

## Dimensions and materials

- Tray and lid: {2 * HALF_X:.0f} x {2 * HALF_Y:.0f} mm in plan, {CORNER_R:.0f} mm corner radius, {WALL:.0f} mm walls, floor and lid top.
- Tray z = 0 to {BASE_TOP:.0f} mm, lid z = {LID_BOTTOM:.0f} to {LID_TOP:.0f} mm. The rims face each other across a {LID_BOTTOM - BASE_TOP:.0f} mm gap and share no vertex.
- Cavity: {inner[0]:.0f} x {inner[1]:.0f} x {inner[2]:.0f} mm, enough for the 90 x 55 x 25 mm [`soap_bar`](../soap_bar/SOURCE.md), which stands {WALL + 25.0 - BASE_TOP:.0f} mm out of the tray.
- Web: {2 * HINGE_HALF_X:.0f} mm along X, {HINGE_THICK:.0f} mm thick, free from z = {WEB_Z[0]:.0f} to {WEB_Z[1]:.0f} mm, standing {HINGE_CLEAR:.0f} mm off the back wall. A {KNUCKLE:.0f} mm knuckle joins each end to the tray (below) and the lid (above); the knuckles carry the tray's and the lid's labels.
- Thumb tab: {2 * TAB_HALF_X:.0f} x {TAB_DEPTH:.0f} x {TAB_THICK:.0f} mm, flush with the lid's lower rim on +Y.

| Index | Part | Material | E in Pa | Poisson ratio | Density in kg/m^3 | Friction | Tetrahedra | Mass in g |
|---|---|---|---:|---:|---:|---:|---:|---:|
{rows}

Rigid groups `{json.dumps(RIGID_GROUPS)}`: the tray is one body, the lid and its tab another, and the web is
the only part left deformable. Leave them out to run every part as finite elements.

The values are engineering starting estimates, not measurements. The web's modulus is not polypropylene's:
a real living hinge is about 0.3 mm thick, this one is meshed at {HINGE_THICK:.0f} mm, and 4.0e7 Pa gives the meshed web
the bending stiffness of the thin one (1.5e9 x 0.3^3). A real hinge also yields and stays where it is put;
this elastic web springs back toward the closed pose.

## What is not modelled or tested

- No latch, no overlapping lip and no seal. Only contact between the rims and the web's stiffness hold the
  lid shut, so a scene needs lid-to-tray contact: as two rigid groups, or as soft self-contact whose
  rest-shape exclusion is under the {LID_BOTTOM - BASE_TOP:.0f} mm rim gap.
- No simulation was run when the bundle was written. What the lid, the web and the tray do when the case is
  opened is measured by the scenes that use it, and reported in `../../README.md`.
- The walls are thicker than a moulded case's and the plan is extruded straight up, with no draft or
  rounded top edge.

## Verification from the written files

| Measurement | Result |
|---|---|
| Physics vertices / tetrahedra | {facts["vertices"]:,} / {facts["tetrahedra"]:,} |
| Boundary triangles | {facts["boundary_faces"]:,} |
| Bounding box in m, min to max | ({lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}) to ({hi[0]:.4f}, {hi[1]:.4f}, {hi[2]:.4f}) |
| Mass from volume x density | {1e3 * facts["mass_kg"]:.2f} g |
| Smallest / largest tetrahedron | {1e9 * facts["min_volume_m3"]:.4f} / {1e9 * facts["max_volume_m3"]:.4f} mm^3 |
| Dihedral angles | {facts["dihedral_deg"][0]:.1f} to {facts["dihedral_deg"][1]:.1f} degrees |
| Summed tet volume / volume enclosed by the boundary | {1e6 * facts["volume_m3"]:.6f} / {1e6 * facts["enclosed_volume_m3"]:.6f} cm^3 |
| Boundary Euler characteristic | {facts["boundary_euler_characteristic"]} |

Every tetrahedron has positive signed volume, no face has more than two owners, and every boundary edge
has exactly two boundary triangles. The repository's `load_labelled_tets` accepts the volume as one
face-connected solid. Labels come from the grid cell a tetrahedron was cut from.

## SHA-256

{hashes}
"""


def main() -> None:
    folder = HERE / NAME
    (folder / "appearance").mkdir(parents=True, exist_ok=True)
    points, tets, tet_labels = build_volume()
    faces, owner = boundary(tets)
    facts = measure(points, tets, tet_labels, faces)

    (folder / "final_mesh.mesh").write_text(medit(points, tets, tet_labels))
    elastic = np.array([[p[2], p[3]] for p in PARTS])
    density = np.array([p[4] for p in PARTS])
    np.savez_compressed(
        folder / "heterogeneous_params.npz",
        part_indices=np.arange(len(PARTS), dtype=np.int64),
        part_E_nu=elastic,
        part_density=density,
        part_colors=np.array([p[6] for p in PARTS], dtype=np.float64),
        part_volume_fill_mode_codes=np.zeros(len(PARTS), dtype=np.int64),
        tet_E_nu=elastic[tet_labels],
        tet_density=density[tet_labels],
        tet_part_labels=tet_labels.astype(np.int64),
    )
    predictions = [
        {
            "part_name": name,
            "major_material_name": material,
            "part_color_rgb": list(rgb),
            "youngs_modulus_pa": young,
            "poisson_ratio": nu,
            "density_kg_m3": rho,
            "friction_coefficient": mu,
            "part_index": i,
            "volume_fill_mode": "solid_fill",
        }
        for i, (name, material, young, nu, rho, mu, rgb) in enumerate(PARTS)
    ]
    materials = {
        "schema_version": "generated-heterogeneous-materials-v1",
        "inferred_object_name": NAME,
        "num_parts": len(PARTS),
        "predictions": predictions,
        "notes": "Engineering starting estimates, not measured or simulation-calibrated. The hinge modulus "
        "is a bending-equivalent value for the meshed 1 mm web. Rigid grouping is in asset.json.",
    }
    (folder / "inferred_params.json").write_text(json.dumps(materials, indent=2) + "\n")
    asset = {
        "name": NAME,
        "description": "100 x 64 x 32 mm travel soap case, closed: a tray and a lid of about the same depth joined by a thin "
        "web on one long side, with a thumb tab on the other.",
        "units": "m",
        "up_axis": "Z",
        "generator": f"generate_{NAME}.py (procedural grid of hexahedra cut into tetrahedra, cell part labels)",
        "construction": "Tensor-product hexahedral grid, six tetrahedra per cell, plan corners rounded by moving vertices",
        "rigid_groups": RIGID_GROUPS,
        "source_assets": [],
        "behavior_validated": False,
        "metadata": {
            "origin": "Centre of the tray footprint at z=0",
            "hinge_axis": [1, 0, 0],
            "hinge_side": "-Y",
            "thumb_tab_side": "+Y",
            "rest_pose": "closed",
            "rim_gap_m": (LID_BOTTOM - BASE_TOP) * 1e-3,
            "cavity_m": [2e-3 * (HALF_X - WALL), 2e-3 * (HALF_Y - WALL), 1e-3 * (LID_TOP - 2 * WALL)],
            "tuning_status": "Source estimates; no latch or lip, no simulation run. Needs lid-to-tray contact "
            "and a per-scene hinge stiffness.",
        },
    }
    (folder / "asset.json").write_text(json.dumps(asset, indent=2) + "\n")

    png = palette_png()
    binding = visual_binding(points, faces, owner, tet_labels)
    (folder / "appearance" / "albedo.png").write_bytes(png)
    (folder / "appearance" / "visual_mesh.glb").write_bytes(
        glb_bytes(binding["visual_rest_vertices_m"], binding["visual_faces"], binding["visual_uv"], png)
    )
    np.savez_compressed(folder / "appearance" / "visual_to_physics.npz", **binding)
    (folder / "SOURCE.md").write_text(source_md(facts, folder))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
