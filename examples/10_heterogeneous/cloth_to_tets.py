# SPDX-License-Identifier: Apache-2.0
"""Convert a rigid+cloth asset (rigid-cloth-asset) to the labelled-tet package the cable scenes load.

The cloth mid-surface becomes a thin solid shell (offset ``thickness`` about it) and the rigid meshes
solids; their boolean union is tetrahedralized with TetGen, tets labelled by centroid (0: cloth, i: rigid body i).
Cloth vertices on a rigid mesh (the seams) are first sunk ``SINK`` into it, so the shell fuses with it, and a
triangle whose three corners all sit on rigid meshes is dropped (it would lie inside).

Run with the extras: ``uv run --with tetgen --with fast-simplification python cloth_to_tets.py <source> <out>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fast_simplification
import numpy as np
import scipy.spatial
import tetgen
import trimesh

SINK = 0.0015
"""How far seam vertices sink into the rigid mesh [m]."""
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
SLIVER = 1e-9
MIN_VOLUME = 1e-12
"""Smaller tets are dropped: at single precision, a bench-height pose rounds them inside out."""
"""Tets whose volume is under this fraction of the regular tet on their longest edge are dropped."""
RIGID = {  # material per rigid body id: (E [Pa], nu, density [kg/m^3]); the solver scales rigid tets down anyway
    "cloth_cleaning_wand": {1: (2.2e9, 0.37, 1100.0)},
    "handled_fabric_bag": {1: (7.0e10, 0.33, 2700.0), 2: (1.5e9, 0.4, 950.0)},
}


def split_long_edges(vertices, faces, longest: float, box) -> tuple[np.ndarray, np.ndarray]:
    """Bisect every edge over ``longest`` whose midpoint lies in ``box`` (lo, hi): the surface stays closed and conforming."""
    vertices, faces = [*vertices], faces.tolist()
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    while True:
        owner = {}
        for i, f in enumerate(faces):
            for k in range(3):
                owner.setdefault(frozenset((f[k], f[(k + 1) % 3])), []).append(i)
        v = np.asarray(vertices)
        picks = []
        for e, adj in owner.items():
            a, b = tuple(e)
            mid = 0.5 * (v[a] + v[b])
            if np.linalg.norm(v[a] - v[b]) > longest and (lo <= mid).all() and (mid <= hi).all():
                picks.append((np.linalg.norm(v[a] - v[b]), a, b, adj))
        taken, done = set(), False
        for _, a, b, adj in sorted(picks, reverse=True):
            if taken.intersection(adj):
                continue
            taken.update(adj)
            m = len(vertices)
            vertices.append(0.5 * (v[a] + v[b]))
            for i in adj:
                f = faces[i]
                k = next(k for k in range(3) if {f[k], f[(k + 1) % 3]} == {a, b})
                p, q, r = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
                faces[i] = [p, m, r]
                faces.append([m, q, r])
            done = True
        if not done:
            return np.asarray(vertices), np.asarray(faces)


def shell_solid(top, bottom, faces) -> trimesh.Trimesh:
    """The closed solid between the ``top`` and ``bottom`` copies of the surface ``faces``.

    A boundary edge's wall splits along the diagonal :func:`prism_tets` cuts its side face by, so the solid's
    walls and the prisms built beside it share their triangles.
    """
    n = len(top)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    present = {tuple(e) for e in edges}
    quads = []
    for a, b in edges:
        if (b, a) in present:
            continue
        # Edge u < v: the diagonal runs from v's bottom to u's top.
        quads += ((a, a + n, b + n), (a, b + n, b)) if a < b else ((a, a + n, b), (a + n, b + n, b))
    solid = trimesh.Trimesh(
        np.vstack((top, bottom)), np.vstack((faces, faces[:, ::-1] + n, np.array(quads).reshape(-1, 3))), process=False
    )
    solid.remove_unreferenced_vertices()
    return solid


def prism_tets(faces: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Three tets per triangle between its top and bottom nodes, the side faces cut as :func:`shell_solid` does."""
    a, b, c = np.sort(faces, axis=1).T
    # a < b < c: (a b c a'), (b c a' b'), (c a' b' c') with the bottom unprimed.
    return np.concatenate(
        [
            np.stack((bottom[a], bottom[b], bottom[c], top[a]), 1),
            np.stack((bottom[b], bottom[c], top[a], top[b]), 1),
            np.stack((bottom[c], top[a], top[b], top[c]), 1),
        ]
    )


def convert(
    source: Path,
    out: Path,
    *,
    thickness: float,
    young: float,
    cloth_faces: int,
    rigid_faces: int,
    ratio: float,
    dihedral: float,
    sliver: float,
    pad_edge: float,
    pad_bodies: tuple,
    pad_box: tuple,
):
    meta = json.loads((source / "asset.json").read_text())
    materials = json.loads((source / "materials.json").read_text())["cloth"]
    mesh = np.load(source / "physics/cloth_mesh.npz")
    vertices, faces = mesh["rest_vertices_m"], mesh["triangles"].astype(np.int32)
    if cloth_faces and len(faces) > cloth_faces:
        vertices, faces = fast_simplification.simplify(vertices, faces, target_count=cloth_faces, agg=3)
        faces = faces.astype(np.int32)
    rigids = {}
    for body in meta["rigid_bodies"]:
        v, f = trimesh.load(source / body["mesh"], force="mesh", process=False).vertices, None
        solid = trimesh.load(source / body["mesh"], force="mesh", process=False)
        if rigid_faces and len(solid.faces) > rigid_faces:
            v, f = fast_simplification.simplify(solid.vertices, solid.faces.astype(np.int32), target_count=rigid_faces)
            solid = trimesh.Trimesh(v, f, process=True)
        if pad_edge and body["body_id"] in pad_bodies:  # surface nodes for the pads to touch
            v, f = split_long_edges(solid.vertices, solid.faces, pad_edge, pad_box)
            solid = trimesh.Trimesh(v, f, process=False)
        rigids[body["body_id"]] = solid
    vertices = vertices.copy()
    seam = np.zeros(len(vertices), dtype=bool)
    for solid in rigids.values():
        closest, distance, tri = trimesh.proximity.closest_point(solid, vertices)
        near = distance < SINK
        vertices[near] = closest[near] - solid.face_normals[tri[near]] * SINK
        seam |= near
    keep = ~seam[faces].all(axis=1)
    faces = faces[keep]
    normals = trimesh.Trimesh(vertices, faces, process=False).vertex_normals
    top, bottom = vertices + normals * thickness / 2, vertices - normals * thickness / 2
    # Triangles on a seam fuse with the rigid solids and are tetrahedralized with them; the rest are prisms.
    stub = seam[faces].any(axis=1)
    shell = shell_solid(top, bottom, faces[stub])
    if shell.volume < 0:
        shell.invert()
    union = trimesh.boolean.union([*shell.split(only_watertight=False), *rigids.values()], engine="manifold")
    union.merge_vertices(digits_vertex=6)
    print("union", len(union.faces), "faces, volume", union.volume, "watertight", union.is_watertight, flush=True)
    gen = tetgen.TetGen(np.asarray(union.vertices) * 1000, np.asarray(union.faces, dtype=np.int32))
    nodes, tets, *_ = gen.tetrahedralize(
        order=1, quality=True, minratio=ratio, mindihedral=dihedral, fixedvolume=True, maxvolume=1e9,
        steinerleft=250000, nobisect=True, opt_scheme=1, opt_iterations=2,
    )  # fmt: skip
    nodes, tets = np.asarray(nodes) / 1000, np.asarray(tets, dtype=np.int32)
    # The prisms' nodes: those the stub's surface already made, found by position, and new ones.
    tree = scipy.spatial.cKDTree(nodes)
    pending = {}
    for copy, points in (("top", top), ("bottom", bottom)):
        d, i = tree.query(points)
        pending[copy] = np.where(d < 1e-7, i, -1)
    used = np.unique(faces[~stub])
    extra = {}
    for copy in ("top", "bottom"):
        for v in used[pending[copy][used] < 0]:
            extra[(copy, v)] = len(nodes) + len(extra)
    new_points = np.array([(top if c == "top" else bottom)[v] for c, v in extra])
    nodes = np.vstack((nodes, new_points.reshape(-1, 3)))
    ids = {c: pending[c].copy() for c in pending}
    for (c, v), k in extra.items():
        ids[c][v] = k
    tets = np.vstack((tets, prism_tets(faces[~stub], ids["top"], ids["bottom"]).astype(np.int32)))
    volume = np.linalg.det(nodes[tets[:, 1:]] - nodes[tets[:, :1]]) / 6
    tets[volume < 0, :2] = tets[volume < 0, 1::-1]
    volume = np.abs(volume)
    edges = np.stack([np.linalg.norm(nodes[tets[:, a]] - nodes[tets[:, b]], axis=1) for a, b in PAIRS], 1)
    # Slivers (volume against the longest edge's cube) come from the seams' near-coincident surfaces: drop them.
    good = (6 * np.sqrt(2) * volume / edges.max(axis=1) ** 3 > sliver) & (volume > MIN_VOLUME)
    print("slivers dropped", int((~good).sum()), "of", len(tets), flush=True)
    tets = tets[good]
    used = np.unique(tets)
    nodes, tets = nodes[used], np.searchsorted(used, tets).astype(np.int32)
    centroid = nodes[tets].mean(axis=1)
    labels = np.zeros(len(tets), dtype=np.int64)
    for body, solid in rigids.items():
        labels[solid.contains(centroid)] = body
    labels_all = [0, *rigids]
    rigid = RIGID[meta["name"]]
    table = {0: (young, 0.3, materials["areal_density"] / thickness), **rigid}
    elastic = np.array([table[i][:2] for i in labels_all])
    density = np.array([table[i][2] for i in labels_all])
    index = np.searchsorted(labels_all, labels)
    colors = {p["part_id"]: p["color_rgb"] for p in json.loads((source / "materials.json").read_text())["parts"]}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "heterogeneous_params.npz",
        part_indices=np.array(labels_all), part_E_nu=elastic, part_density=density,
        part_colors=np.array([colors[i] for i in labels_all], float), part_volume_fill_mode_codes=np.zeros(len(labels_all), np.int64),
        tet_E_nu=elastic[index], tet_density=density[index], tet_part_labels=labels,
    )  # fmt: skip
    with (out / "final_mesh.mesh").open("w") as f:
        f.write(f"MeshVersionFormatted 1\nDimension 3\nVertices\n{len(nodes)}\n")
        np.savetxt(f, np.column_stack((nodes, np.zeros(len(nodes)))), fmt=["%.12g"] * 3 + ["%d"])
        f.write(f"Tetrahedra\n{len(tets)}\n")
        np.savetxt(f, np.column_stack((tets + 1, labels + 1)), fmt="%d")
        f.write("End\n")
    names = {p["part_id"]: p for p in json.loads((source / "materials.json").read_text())["parts"]}
    predictions = [
        {
            "part_name": names[i]["name"], "major_material_name": names[i]["material"], "part_color_rgb": colors[i],
            "youngs_modulus_pa": float(table[i][0]), "poisson_ratio": float(table[i][1]), "density_kg_m3": float(table[i][2]),
            "friction_coefficient": 0.5, "part_index": i, "volume_fill_mode": "solid_fill",
        }
        for i in labels_all
    ]  # fmt: skip
    (out / "inferred_params.json").write_text(
        json.dumps({"schema_version": "generated-heterogeneous-materials-v1", "inferred_object_name": meta["name"],
                    "num_parts": len(labels_all), "predictions": predictions}, indent=2) + "\n"
    )  # fmt: skip
    asset = {
        "name": out.name, "description": f"{meta['description']} Cloth as a {thickness * 1000:g} mm solid shell of tets.",
        "units": "m", "up_axis": "Z", "construction": "cloth_to_tets.py: thickened mid-surface and rigid meshes, union, TetGen",
        "rigid_groups": [[i] for i in rigids], "source_assets": [source.name], "behavior_validated": False,
        "metadata": {"thickness_m": thickness, "cloth_young_pa": young, "cloth_faces": int(len(faces)), "tets": len(tets)},
    }  # fmt: skip
    (out / "asset.json").write_text(json.dumps(asset, indent=2) + "\n")
    hashes = "\n".join(
        f"- `{p}`: `{hashlib.sha256((out / p).read_bytes()).hexdigest()}`"
        for p in ("final_mesh.mesh", "heterogeneous_params.npz", "inferred_params.json", "asset.json")
    )
    (out / "SOURCE.md").write_text(
        f"# {out.name}\n\nConverted by `cloth_to_tets.py` from `assets/{source.name}` (cloth mid-surface plus rigid meshes):"
        f" cloth as a {thickness * 1000:g} mm solid shell (E {young:g} Pa), rigid parts as solids, one TetGen volume."
        f" {len(tets)} tets; labels: 0 cloth, {', '.join(map(str, rigids))} rigid. Not behaviour-validated.\n\n## SHA-256\n\n{hashes}\n"
    )
    return nodes, tets, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--thickness", type=float, default=0.0025)
    parser.add_argument("--young", type=float, default=5.0e5)
    parser.add_argument("--cloth-faces", type=int, default=0, help="decimate the cloth to this many triangles")
    parser.add_argument("--rigid-faces", type=int, default=0, help="decimate each rigid mesh to this many triangles")
    parser.add_argument("--sliver", type=float, default=SLIVER, help="drop tets under this quality")
    parser.add_argument("--pad-edge", type=float, default=0.008, help="longest surface edge of the held part [m]")
    parser.add_argument(
        "--pad-box",
        type=float,
        nargs=6,
        default=(-9, -9, -9, 9, 9, 9),
        help="where the held part is split: lo then hi [m]",
    )
    parser.add_argument("--ratio", type=float, default=3.0, help="TetGen radius-edge ratio bound")
    parser.add_argument("--dihedral", type=float, default=10.0, help="TetGen minimum dihedral [deg]")
    a = parser.parse_args()
    nodes, tets, labels = convert(a.source, a.out, thickness=a.thickness, young=a.young, cloth_faces=a.cloth_faces,
                                  rigid_faces=a.rigid_faces, ratio=a.ratio, dihedral=a.dihedral, sliver=a.sliver, pad_edge=a.pad_edge, pad_bodies=(1,), pad_box=(a.pad_box[:3], a.pad_box[3:]))  # fmt: skip
    volume = np.linalg.det(nodes[tets[:, 1:]] - nodes[tets[:, :1]]) / 6
    print({"tets": len(tets), "vertices": len(nodes), "by_label": {int(k): int((labels == k).sum()) for k in np.unique(labels)},
           "min_volume_m3": float(volume.min())})  # fmt: skip


if __name__ == "__main__":
    main()
