"""Generate the hooked elastic cord from dimensioned solids. All CAD lengths are mm.

python generate_hooked_elastic_cord.py --output ./build/hooked_elastic_cord
Use --help for geometry and meshing overrides. Requires Python 3.11+, gmsh,
tetgen, trimesh, numpy and Pillow. Exact tested versions are in the report.
"""
import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import gmsh
import numpy as np
import tetgen
import trimesh
from PIL import Image


@dataclass
class Dimensions:
    cord_diameter_mm: float = 4
    cord_free_length_mm: float = 250
    taper_length_mm: float = 10
    taper_end_diameter_mm: float = 5.6
    grip_length_mm: float = 24
    grip_width_mm: float = 12
    grip_height_mm: float = 8
    grip_chamfer_mm: float = 0.75
    hook_section_diameter_mm: float = 6
    hook_inner_radius_mm: float = 8
    hook_tip_angle_deg: float = 33
    surface_edge_mm: float = 1.55
    max_tet_volume_mm3: float = 1.0

    @property
    def interface(self):
        return self.cord_free_length_mm / 2 + self.taper_length_mm

    @property
    def centre(self):
        return self.interface + self.grip_length_mm + self.hook_inner_radius_mm + self.hook_section_diameter_mm / 2


def joined(volumes):
    result, _ = gmsh.model.occ.fuse(volumes[:1], volumes[1:])
    if len(result) != 1:
        raise ValueError("CAD solids did not fuse into one volume")
    return result[0]


def right_grip(d):
    occ = gmsh.model.occ
    block = occ.addBox(d.interface, -d.grip_width_mm / 2, -d.grip_height_mm / 2,
                       d.grip_length_mm, d.grip_width_mm, d.grip_height_mm)
    occ.synchronize()
    # OCC needs an adjacent face for each edge to specify its chamfer.
    edges = {}
    for _, face in gmsh.model.getBoundary([(3, block)], oriented=False):
        for _, edge in gmsh.model.getBoundary([(2, face)], oriented=False):
            edges[edge] = face
    (block_volume,) = occ.chamfer([block], list(edges), list(edges.values()), [d.grip_chamfer_mm])
    r = d.hook_section_diameter_mm / 2
    radius = d.hook_inner_radius_mm + r
    angle = math.radians(d.hook_tip_angle_deg)
    arc = (3, occ.addTorus(d.centre, 0, 0, radius, r, angle=math.pi + angle))
    occ.rotate([arc], d.centre, 0, 0, 0, 0, 1, -angle)
    tip = (3, occ.addSphere(d.centre + radius * math.cos(angle), -radius * math.sin(angle), 0, r))
    root = (3, occ.addSphere(d.centre - radius, 0, 0, r))
    return joined([block_volume, root, arc, tip])


def make_surface(d):
    occ = gmsh.model.occ
    grip = right_grip(d)
    reflected = occ.copy([grip])
    occ.rotate(reflected, 0, 0, 0, 0, 1, 0, math.pi)
    half = d.cord_free_length_mm / 2
    r, end = d.cord_diameter_mm / 2, d.taper_end_diameter_mm / 2
    cord = joined([(3, occ.addCylinder(-half, 0, 0, 2 * half, 0, 0, r)),
                   (3, occ.addCone(half, 0, 0, d.taper_length_mm, 0, 0, r, end)),
                   (3, occ.addCone(-half, 0, 0, -d.taper_length_mm, 0, 0, r, end))])
    # Fragment the union so TetGen must preserve both material interfaces.
    parts, _ = occ.fragment([cord], [grip, *reflected])
    if len(parts) != 3:
        raise ValueError("Expected three conforming CAD volumes")
    occ.synchronize()
    gmsh.option.setNumber('Mesh.MeshSizeMin', d.surface_edge_mm)
    gmsh.option.setNumber('Mesh.MeshSizeMax', d.surface_edge_mm)
    gmsh.model.mesh.generate(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    _, triangles = gmsh.model.mesh.getElementsByType(2)
    order = np.argsort(tags)
    faces = np.searchsorted(tags[order], triangles).reshape(-1, 3).astype(np.int32)
    return xyz.reshape(-1, 3)[order], faces


def mesh(d):
    gmsh.initialize(['cord', '-nopopup'], readConfigFiles=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.add('cord')
        vertices, faces = make_surface(d)
        xyz, cells, *_ = tetgen.TetGen(vertices, faces).tetrahedralize(
            quality=True, fixedvolume=True, maxvolume=d.max_tet_volume_mm3,
            nobisect=True, minratio=1.5, mindihedral=12)
        cells = cells.copy()
        xyz, cells = improve_interior(xyz, cells, d)
    finally:
        gmsh.finalize()
    det = np.linalg.det(xyz[cells[:, 1:]] - xyz[cells[:, :1]])
    cells[det < 0, :2] = cells[det < 0, 1::-1]
    labels = np.digitize(xyz[cells, 0].mean(1), [-d.interface, d.interface]).astype(np.int64)
    return xyz / 1000, cells, labels


def improve_interior(xyz, cells, d):
    # Reattach TetGen cells to the original CAD model: Netgen can then
    # optimize interiors while retaining both the external and interface mesh.
    tags, surface_xyz, _ = gmsh.model.mesh.getNodes()
    known = {tuple(row): tag for row, tag in zip(surface_xyz.reshape(-1, 3), tags)}
    node_tags = np.array([known.get(tuple(row), 0) for row in xyz], dtype=np.int64)
    if np.count_nonzero(node_tags) != len(tags):
        raise ValueError("TetGen did not preserve all constrained surface nodes")
    new = node_tags == 0
    node_tags[new] = np.arange(tags.max() + 1, tags.max() + 1 + new.sum())
    volumes = sorted(gmsh.model.getEntities(3), key=lambda ent: gmsh.model.occ.getCenterOfMass(*ent)[0])
    tet_parts = np.digitize(xyz[cells, 0].mean(1), [-d.interface, d.interface])
    node_parts = np.digitize(xyz[:, 0], [-d.interface, d.interface])
    for part, (_, volume) in enumerate(volumes):
        inner = new & (node_parts == part)
        gmsh.model.mesh.addNodes(3, volume, node_tags[inner], xyz[inner].ravel())
        gmsh.model.mesh.addElementsByType(volume, 4, [], node_tags[cells[tet_parts == part]].ravel())
    gmsh.model.mesh.optimize('Netgen')
    ids, coords, _ = gmsh.model.mesh.getNodes()
    _, connectivity = gmsh.model.mesh.getElementsByType(4)
    order = np.argsort(ids)
    return coords.reshape(-1, 3)[order], np.searchsorted(ids[order], connectivity).reshape(-1, 4)


def boundary(cells):
    oriented = cells[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(oriented, axis=1), axis=0, return_index=True, return_counts=True)
    if counts.max() > 2:
        raise ValueError("Non-manifold volume")
    selected = first[counts == 1]
    return oriented[selected], selected // 4


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def write_package(out, d, xyz, cells, labels):
    appearance = out / 'appearance'
    appearance.mkdir()
    names = ['hook_a', 'elastic_cord', 'hook_b']
    materials = ['ABS', 'silicone rubber', 'ABS']
    elastic = np.array([[2.2e9, .37], [1e6, .49], [2.2e9, .37]])
    density = np.full(3, 1100.)
    colors = np.array([[200, 205, 210], [35, 40, 46], [200, 205, 210]], dtype=float)
    friction = [.35, .9, .35]
    with (out / 'final_mesh.mesh').open('w') as fh:
        fh.write(f'MeshVersionFormatted 1\nDimension 3\nVertices\n{len(xyz)}\n')
        np.savetxt(fh, xyz, fmt='%.12g %.12g %.12g 0')
        fh.write(f'Tetrahedra\n{len(cells)}\n')
        np.savetxt(fh, np.column_stack([cells + 1, labels + 1]), fmt='%d')
        fh.write('End\n')
    np.savez_compressed(out / 'heterogeneous_params.npz',
        part_indices=np.arange(3, dtype=np.int64), part_E_nu=elastic, part_density=density,
        part_colors=colors, part_volume_fill_mode_codes=np.zeros(3, np.int64),
        tet_part_labels=labels, tet_E_nu=elastic[labels], tet_density=density[labels])
    predictions = [dict(part_index=i, part_name=names[i], major_material_name=materials[i],
        part_color_rgb=colors[i].astype(int).tolist(), youngs_modulus_pa=elastic[i, 0],
        poisson_ratio=elastic[i, 1], density_kg_m3=density[i], friction_coefficient=friction[i],
        volume_fill_mode='solid_fill') for i in range(3)]
    write_json(out / 'inferred_params.json', dict(schema_version='generated-heterogeneous-materials-v1',
        inferred_object_name='hooked_elastic_cord', num_parts=3, predictions=predictions,
        notes='Engineering starting estimates, not measured or simulation-calibrated. Rigid grouping is in asset.json.'))
    faces, owners = boundary(cells)
    ids = faces.ravel().astype(np.int64)
    surface_ids, surface_map = np.unique(ids, return_inverse=True)
    rest = xyz[ids].astype(np.float32)
    visual_faces = np.arange(len(ids), dtype=np.int32).reshape(-1, 3)
    part_ids = labels[owners].astype(np.int32)
    uv = np.column_stack([(np.repeat(part_ids, 3) + .5) / 3, np.full(len(ids), .5)]).astype(np.float32)
    np.savez_compressed(appearance / 'visual_to_physics.npz', visual_faces=visual_faces,
        boundary_face_tet_indices=owners.astype(np.int64), boundary_face_part_ids=part_ids,
        surface_vertex_to_physics_vertex=surface_ids, visual_vertex_to_surface_vertex=surface_map.astype(np.int64),
        physics_vertex_indices=ids, visual_rest_vertices_m=rest, visual_uv=uv)
    atlas = Image.fromarray(np.tile(np.repeat(colors.astype(np.uint8), 64, axis=0)[None, :, :], (64, 1, 1)))
    atlas.save(appearance / 'albedo.png')
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=atlas, metallicFactor=0.,
                                                    roughnessFactor=.65, doubleSided=False)
    visual = trimesh.Trimesh(rest, visual_faces, process=False,
        visual=trimesh.visual.texture.TextureVisuals(uv=uv, material=material))
    visual.export(appearance / 'visual_mesh.glb')
    desc = (f'A {d.cord_free_length_mm:g} mm × Ø{d.cord_diameter_mm:g} mm silicone cord has '
            f'{d.taper_length_mm:g} mm tapers, {d.grip_length_mm:g} × {d.grip_width_mm:g} × '
            f'{d.grip_height_mm:g} mm ABS grips and Ø{d.hook_section_diameter_mm:g} mm hooks '
            f'with {d.hook_inner_radius_mm:g} mm nominal inner radii.')
    metadata = dict(asdict(d), bight_center_abs_x_mm=d.centre, cord_grip_interface_abs_x_mm=d.interface,
        override_levers=['hook youngs_modulus_pa -> 1e8 for <=100x effective contrast',
                        'part 1 youngs_modulus_pa', 'part 1 damping_pa_s'],
        tuning_status='No simulation retuning performed; source estimates only.')
    write_json(out / 'asset.json', dict(name='hooked_elastic_cord', description=desc, units='m',
        up_axis='Z',
        generator='generate_hooked_elastic_cord.py (OpenCASCADE boolean solids, TetGen volume with shared part '
                  'interfaces, Netgen interior optimisation, centroid part labels)',
        construction='Boolean union surface, TetGen volume, centroid part labels',
        rigid_groups=[[0], [2]], source_assets=[], behavior_validated=False, metadata=metadata))
    lines = [f'- `{p.relative_to(out).as_posix()}`: `{hashlib.sha256(p.read_bytes()).hexdigest()}`'
             for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'SOURCE.md').write_text(f'# Hooked elastic cord\n\n{desc}\n\n'
        'OpenCASCADE Boolean solids share triangulated material interfaces. TetGen filled the volume, '
        'Netgen optimized its interior, and final tetrahedron centroids selected part labels. '
        f'The mesh contains {len(xyz):,} vertices and {len(cells):,} tetrahedra. '
        'Parts 0 and 2 are ABS hooks and grips (2.2 GPa). Part 1 is the silicone cord and tapers (1 MPa). '
        'Both mouths face −Y. The bundle underwent static checks only.\n\n## SHA-256\n\n'
        + '\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('hooked_elastic_cord'))
    for key, value in asdict(Dimensions()).items():
        parser.add_argument('--' + key.replace('_', '-'), type=float, default=value)
    args = vars(parser.parse_args())
    out = args.pop('output').resolve()
    if out.exists():
        parser.error(f'Output already exists: {out}. Choose an unused --output path.')
    if any(v <= 0 or not math.isfinite(v) for v in args.values()):
        parser.error('Geometry and mesh parameters must be finite and positive.')
    d = Dimensions(**args)
    if d.grip_chamfer_mm * 2 >= min(d.grip_width_mm, d.grip_height_mm, d.grip_length_mm):
        parser.error('Chamfer must be smaller than half the shortest grip dimension.')
    if not d.hook_tip_angle_deg < 90:
        parser.error('Hook tip angle must be below 90 degrees.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        bundle = Path(temp) / 'hooked_elastic_cord'
        bundle.mkdir()
        xyz, cells, labels = mesh(d)
        write_package(bundle, d, xyz, cells, labels)
        bundle.rename(out)
    print(f'{out} — {len(xyz)} vertices, {len(cells)} tetrahedra')


if __name__ == '__main__':
    main()
