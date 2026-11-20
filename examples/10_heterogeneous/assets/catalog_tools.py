"""Keep ``catalog.json`` in step with the asset folders and draw their previews.

    uv run python examples/10_heterogeneous/assets/catalog_tools.py add      # catalog new generated assets
    uv run python examples/10_heterogeneous/assets/catalog_tools.py preview  # previews/<id>.png + previews.png

``add`` appends an entry for every folder with an ``asset.json`` (the generated_tool_assets formats)
that the catalog does not list yet, and refreshes the entries it made before. Entries imported from
elsewhere are left untouched. ``preview`` renders each catalogued object's ``appearance/visual_mesh.glb``
with its albedo, flat-shaded from a fixed view, and tiles them into ``previews.png``.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
REPO = EXAMPLES.parent.parent
sys.path.insert(0, str(REPO))

GENERATOR = "generated_tool_assets (procedural geometry, Manifold union + TetGen, centroid part labels)"
TILE = 320


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(folder: Path) -> dict:
    return {
        str(p.relative_to(folder)): {"sha256": _sha(p), "size_bytes": p.stat().st_size}
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "SOURCE.md" and "previews" not in p.parts
    }


def _listed_hashes_match(folder: Path, files: dict) -> bool:
    listed = re.findall(r"- `([^`]+)`: `([0-9a-f]{64})`", (folder / "SOURCE.md").read_text())
    return bool(listed) and all(files.get(rel, {}).get("sha256") == digest for rel, digest in listed)


def _bounds(points: np.ndarray) -> dict:
    lo, hi = points.min(axis=0), points.max(axis=0)
    return {"bounds_m": [lo.tolist(), hi.tolist()], "extent_m": (hi - lo).tolist()}


def _scenes(name: str) -> list[str]:
    pattern = re.compile(rf"[\"']{re.escape(name)}[\"']")
    return sorted(p.name for p in EXAMPLES.glob("*.py") if pattern.search(p.read_text()))


def _tet_entry(folder: Path) -> dict:
    from robosimstudio_materials.heterogeneous import load_labelled_tets

    vertices, tets, labels, elastic, density = load_labelled_tets(
        folder / "final_mesh.mesh", folder / "heterogeneous_params.npz"
    )
    corners = vertices[tets]
    volume = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.0
    names = {p["part_index"]: p for p in json.loads((folder / "inferred_params.json").read_text())["predictions"]}
    parts = []
    for label in np.unique(labels):
        rows = labels == label
        parts.append(
            {
                "id": int(label),
                "name": names.get(int(label), {}).get("part_name", f"part_{label}"),
                "tetrahedra": int(rows.sum()),
                "youngs_modulus_pa": float(elastic[rows, 0].max()),
                "poisson_ratio": float(elastic[rows, 1].max()),
                "density_kg_m3": float(density[rows].max()),
            }
        )
    young = elastic[:, 0]
    return {
        "vertices": int(len(vertices)),
        "tetrahedra": int(len(tets)),
        **_bounds(vertices),
        "source_mass_kg": float((volume * density).sum()),
        "source_stiffness_ratio": float(young.max() / young.min()),
        "parts": parts,
        "checks": {
            "labelled_tet_loader_passed": True,
            "no_duplicate_tetrahedra": len(np.unique(np.sort(tets, axis=1), axis=0)) == len(tets),
        },
    }


def generated_entry(folder: Path) -> dict:
    meta = json.loads((folder / "asset.json").read_text())
    kind = meta.get("schema", "labelled-tet-v1")
    files = _files(folder)
    entry = {
        "id": folder.name,
        "path": folder.name,
        "kind": kind,
        "source": meta.get("generator", GENERATOR),
        "import_status": "generated",
        "description": meta.get("description", ""),
    }
    if (folder / "final_mesh.mesh").is_file():
        entry |= _tet_entry(folder)
    else:
        entry |= _bounds(_glb(folder / "appearance" / "visual_mesh.glb")[0])
        entry["checks"] = {}
    if meta.get("rigid_groups") is not None:
        entry["rigid_groups"] = meta["rigid_groups"]
    entry["checks"]["source_hashes_match"] = _listed_hashes_match(folder, files)
    entry |= {
        "used_by": _scenes(folder.name),
        "behavior_validated_at_source": bool(meta.get("behavior_validated", False)),
        "limitations": meta.get("metadata", {}).get("tuning_status", "Source estimates; tune per scene."),
        "readme": "SOURCE.md",
        "preview": f"previews/{folder.name}.png",
        "files": files,
    }
    return entry


KINDS = {"labelled-tet-v1": "tet + rigid groups", "rigid-socket-v1": "rigid socket", "rigid-mesh-v1": "rigid mesh"}


def _table(objects: list[dict]) -> str:
    rows = ["| Object | Kind | Tetrahedra | Extent (mm) | Scenes |", "| --- | --- | ---: | --- | --- |"]
    for o in objects:
        extent = " x ".join(f"{1e3 * e:.0f}" for e in o["extent_m"])
        tets = f"{o['tetrahedra']:,}" if "tetrahedra" in o else "-"
        used = ", ".join(f"`{u[:-3]}`" for u in o["used_by"] if u.startswith(("cable_", "tool_"))) or "-"
        rows.append(
            f"| [{o['id']}]({o['path']}/SOURCE.md) | {KINDS.get(o['kind'], 'cloth + rigid')} | {tets} | {extent} | {used} |"
        )
    return "\n".join(rows)


def add() -> None:
    path = HERE / "catalog.json"
    catalog = json.loads(path.read_text())
    kept = [o for o in catalog["objects"] if o.get("import_status") != "generated"]
    known = {o["id"] for o in kept}
    folders = sorted(p.parent for p in HERE.glob("*/asset.json") if p.parent.name not in known)
    catalog["objects"] = kept + [generated_entry(f) for f in folders]
    path.write_text(json.dumps(catalog, indent=2) + "\n")
    readme = HERE / "README.md"
    start, end = "<!-- generated-assets:start -->\n", "\n<!-- generated-assets:end -->"
    text = readme.read_text()
    head, rest = text.split(start)
    readme.write_text(head + start + _table(catalog["objects"][len(kept) :]) + end + rest.split(end)[1])
    print(f"catalog: {len(kept)} imported + {len(folders)} generated objects")


def _glb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions, triangle indices and UVs of a single-mesh GLB."""
    blob = path.read_bytes()
    length = struct.unpack_from("<I", blob, 12)[0]
    doc = json.loads(blob[20 : 20 + length])
    binary = blob[20 + length + 8 :]
    sizes = {5121: (np.uint8, 1), 5123: (np.uint16, 2), 5125: (np.uint32, 4), 5126: (np.float32, 4)}
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}

    def read(index: int) -> np.ndarray:
        acc = doc["accessors"][index]
        view = doc["bufferViews"][acc["bufferView"]]
        dtype, _ = sizes[acc["componentType"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        n = acc["count"] * width[acc["type"]]
        return np.frombuffer(binary, dtype=dtype, count=n, offset=start).reshape(acc["count"], -1)

    prim = doc["meshes"][0]["primitives"][0]
    positions = read(prim["attributes"]["POSITION"]).astype(np.float64)
    faces = read(prim["indices"]).reshape(-1, 3).astype(np.int64)
    uvs = read(prim["attributes"]["TEXCOORD_0"]).astype(np.float64)
    return positions, faces, uvs


def _tet_surface(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boundary triangles of a tet bundle without a visual mesh, coloured by part."""
    from robosimstudio_materials.heterogeneous import load_labelled_tets

    vertices, tets, labels, _elastic, _density = load_labelled_tets(
        folder / "final_mesh.mesh", folder / "heterogeneous_params.npz"
    )
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]])
    owner = np.tile(labels, 4)
    _, first, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    outer = first[count == 1]
    rgb = {
        p["part_index"]: p.get("part_color_rgb", (150, 150, 150))
        for p in json.loads((folder / "inferred_params.json").read_text())["predictions"]
    }
    colour = np.array([rgb.get(int(label), (150, 150, 150)) for label in owner[outer]], dtype=np.float64)
    return vertices, faces[outer], colour


def render(folder: Path, size: int = TILE):
    """Flat-shaded, albedo-coloured view of the visual mesh from the front-left, above."""
    from PIL import Image, ImageDraw

    if (folder / "appearance" / "visual_mesh.glb").is_file():
        positions, faces, uvs = _glb(folder / "appearance" / "visual_mesh.glb")
        albedo = np.asarray(Image.open(folder / "appearance" / "albedo.png").convert("RGB"), dtype=np.float64)
        h, w, _ = albedo.shape
        uv = uvs[faces].mean(axis=1) % 1.0
        colour = albedo[(uv[:, 1] * (h - 1)).astype(int), (uv[:, 0] * (w - 1)).astype(int)]
    else:
        positions, faces, colour = _tet_surface(folder)
    az, el = np.radians(35.0), np.radians(25.0)
    right = np.array([np.cos(az), np.sin(az), 0.0])
    toward = np.array([-np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    up = np.cross(toward, right)
    tri = positions[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    light = np.abs(normal @ (0.6 * toward + 0.5 * up + 0.3 * right) / 0.85)
    shade = colour * (0.35 + 0.65 * np.clip(light, 0.0, 1.0))[:, None]
    p2 = np.stack([positions @ right, positions @ up], axis=1)
    lo, hi = p2.min(axis=0), p2.max(axis=0)
    scale = 0.9 * size / max(hi - lo)
    screen = (p2 - (lo + hi) / 2.0) * scale * np.array([1.0, -1.0]) + size / 2.0
    depth = (tri @ toward).mean(axis=1)
    image = Image.new("RGB", (size, size), (245, 245, 242))
    draw = ImageDraw.Draw(image)
    for f in np.argsort(depth):
        draw.polygon([tuple(screen[i]) for i in faces[f]], fill=tuple(int(c) for c in shade[f]))
    return image


def preview() -> None:
    from PIL import Image, ImageDraw

    catalog = json.loads((HERE / "catalog.json").read_text())
    (HERE / "previews").mkdir(exist_ok=True)
    tiles = []
    for obj in catalog["objects"]:
        image = render(HERE / obj["path"])
        image.save(HERE / "previews" / f"{obj['id']}.png")
        extent = " x ".join(f"{1e3 * e:.0f}" for e in obj["extent_m"])
        ImageDraw.Draw(image).text((8, 8), f"{obj['id']}\n{extent} mm", fill=(20, 20, 20))
        tiles.append(image)
    columns = 6
    sheet = Image.new("RGB", (columns * TILE, -(-len(tiles) // columns) * TILE), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * TILE, (i // columns) * TILE))
    sheet.save(HERE / "previews.png", optimize=True)
    print(f"previews: {len(tiles)} objects -> previews.png")


if __name__ == "__main__":
    {"add": add, "preview": preview}[sys.argv[1]]()
