# Object assets

Reusable object bundles for the heterogeneous manipulation tasks. Each object
has a metric tetrahedral mesh, per-tetrahedron material arrays, part descriptions,
and a textured visual mesh with its matching physics binding.

Start with **kitchen tongs + cookie**, **sauce keeper**, or **toilet plunger** for
the new tool-use tasks. Clawhauser and the revised USB hub were already in this
repository. This import adds source metadata and previews to those two without
changing their existing mesh, materials, texture, or binding.

## Available objects

| Object | Suggested use | Tetrahedra | Source stiffness ratio | Preparation needed |
| --- | --- | ---: | ---: | --- |
| [Clawhauser](clawhauser/) | Pick/place, hang by the loop | 6,850 | 10x | Existing scenes apply their own material/contact settings. |
| [USB hub](usb_hub/) | Cable placement, hanging, retrieval | 9,652 | 40,000x | Existing scene uses rigid part groups and tuned cable/mass settings; the source is not that runtime configuration. |
| [Kitchen tongs](kitchen_tongs/SOURCE.md) | Squeeze the arms to pick up the cookie | 2,727 | 1x | Both labels currently use the same 5 MPa silicone; a part label alone does not imply different materials. |
| [Cookie](cookie/SOURCE.md) | Object manipulated by the tongs | 11,920 | 1x | Both labels use a 3 GPa proxy. Decide the scene's effective stiffness/rigid treatment explicitly. |
| [Sauce keeper / squeeze bottle](sauce_keeper_400ml/SOURCE.md) | Visible bottle compression | 65,623 | 1.1x | Source E is 2/2.2 GPa and needs scene tuning. No liquid is included; a usable fluid cavity is not established. |
| [Toilet plunger](toilet_plunger/SOURCE.md) | Compress the rubber cup against a surface | 17,728 | 20,000x | Source rubber/wood E is 1 MPa/20 GPa. Override the handle stiffness for the project's <=100x effective contrast. |

The ratios above are `max(E) / min(E)` in the **unchanged source arrays**, not
the settings used in existing or future scenes. Materials are inferred starting
estimates, not calibrated measurements. For new scenes, keep the effective
stiffness contrast at or below 100x and record overrides in the scene configuration.
Do not overwrite the source files to tune an individual task. Visible deformation
and useful task coverage take priority over runtime; roughly 15 minutes per run
is acceptable for the current exploration stage.

## What is in a bundle

- `final_mesh.mesh`: Medit volume mesh. Coordinates are already in metres;
  do not reapply the scale factor from the historical scaling report.
- `heterogeneous_params.npz`: `tet_part_labels`, `tet_E_nu`, and `tet_density`,
  in the same tetrahedron order as the mesh.
- `inferred_params.json`: part names, source material estimates, and construction
  assumptions. Some assumptions, such as a hollow bottle, still need checking.
- `appearance/visual_mesh.glb`, `albedo.png`, `visual_to_physics.npz`: the
  matched appearance and rest-pose binding. Keep these files together.
- `metric_mesh_scaling.json`, `volume_topology.json`, `final_export_manifest.json`,
  and `appearance/visual_manifest.json`: original export records.
- `appearance/qa/`: original appearance diagnostics and a rest-pose preview.

The bundle axes and poses are source-specific. Check `catalog.json` bounds and
the preview before choosing a scene rotation or grasp. Historical paths inside
the source reports describe the original generation run; they are not extra
runtime dependencies. `catalog.json` records the files actually shipped here.

The existing `HeterogeneousSoftSpec` in
[`robosimstudio_materials/heterogeneous.py`](../../../robosimstudio_materials/heterogeneous.py)
accepts the mesh, parameter, and material-JSON paths. Use its `part_overrides`
for scene-specific changes. Its loader fixes tetrahedron winding in memory while
preserving element order. Appearance rendering must use the supplied binding;
the FEM constructor alone does not attach the textured visual mesh. Soft-contact
friction is currently global even when the source JSON lists per-part values.

## Validation and provenance

All six objects passed static checks on 2026-11-17:

- Every canonical file matches HAG4R-MPC commit
  `3624989aea2925f26166c33f1d09b109ed63aa80` byte-for-byte.
- The current repository's `load_labelled_tets` function accepts each volume:
  valid indices/material ranges, no degenerate tets or isolated vertices, and one
  face-connected volume with no face shared by more than two tetrahedra.
- No duplicate tetrahedra; material JSON agrees with the per-tet arrays.
- GLB vertices/faces, physics indices, boundary labels and rest positions agree
  with the visual binding; maximum binding error is below 1e-6 m.
- Appearance-manifest hashes match and all PNGs decode.

Detailed measurements, file hashes and source URLs are in [catalog.json](catalog.json).
These checks do **not** establish Newton contact stability, runtime, task success,
physical calibration, suction, liquid flow, or real-robot transfer. No new
simulation was run during this import. The source previews are static appearance
checks, not successful task recordings.

Source: [HAG4R-MPC object exports](https://github.com/ericchen321/HAG4R-MPC/tree/3624989aea2925f26166c33f1d09b109ed63aa80/assets/objects),
generated through the DiagGen workflow. Original source files and attribution are
retained. That source snapshot does not declare a separate asset license; this
import does not assert that the destination's Apache-2.0 software license covers
the imported object data.

## Sources for tasks still missing assets

Rubber stamp, the soap-opening setup, and spatula still need suitable deformable
assets. Use the [project reference photos](https://drive.google.com/drive/folders/1mZNNoP9tlQjqniIwdyB6iZYi29OHYjqa)
for asset generation. The [shared DiagGen folder](https://drive.google.com/drive/folders/1FW5MmvtleQC1Hq1ijwIGCci9wzZzjmpc)
also contains the earlier selected-assets archive and reference videos.

These NVIDIA SimReady candidates are **links only**, not imported or validated
as deformable assets. Their packages use rigid USD physics and need conversion
or scene loading work. They include CC-BY-4.0 license/attribution metadata, which
must accompany any future import; copy composition dependencies as well as the
geometry payload.

| Candidate | Use and limitation |
| --- | --- |
| [Silicone spatula](https://huggingface.co/datasets/nvidia/simready-assets/tree/21b045a878e13d4ca922f0f8ebc2ff1e84bb06d4/utensil_spatula_silicone_a01) | Handle/head geometry; no elastic tetrahedral model. |
| [Bar soap](https://huggingface.co/datasets/nvidia/simready-assets/tree/21b045a878e13d4ca922f0f8ebc2ff1e84bb06d4/cleaning_barSoap_a01) | A soap bar, not an opening package or case. |
| [Sponge](https://huggingface.co/datasets/nvidia/simready-assets/tree/21b045a878e13d4ca922f0f8ebc2ff1e84bb06d4/cleaning_sponge_b01) | Geometry for deformable meshing. |
| [Plastic tray](https://huggingface.co/datasets/nvidia/simready-assets/tree/21b045a878e13d4ca922f0f8ebc2ff1e84bb06d4/dishware_plastic_tray_a01) | Rigid prop for cable placement. |
| [Pan](https://huggingface.co/datasets/nvidia/simready-assets/tree/21b045a878e13d4ca922f0f8ebc2ff1e84bb06d4/cookware_pan_a01) | Rigid prop for spatula contact. |
| [Mustard bottle](https://huggingface.co/datasets/nvidia/simready-assets/tree/21b045a878e13d4ca922f0f8ebc2ff1e84bb06d4/food_bottle_mustard_a01) | Alternative bottle geometry; no squeeze or liquid simulation. |
