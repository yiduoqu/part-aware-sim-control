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

## Generated assets

Every package from `generated_tool_assets` is here too. They are procedural, built with a Manifold
union and TetGen, with centroid part labels. They were checked statically at the source; behaviour
is checked by the scenes that use them. Where a name was already taken, the
generated copy has a suffix: `usb_hub_gen` sits next to the original `usb_hub`, and
`toilet_plunger_generated` next to the HAG4R `toilet_plunger`. Their catalog entries carry `"import_status": "generated"`.

<!-- generated-assets:start -->
| Object | Kind | Tetrahedra | Extent (mm) | Scenes |
| --- | --- | ---: | --- | --- |
| [buckled_strap](buckled_strap/SOURCE.md) | cloth + rigid | - | 282 x 47 x 122 | - |
| [charger](charger/SOURCE.md) | tet + rigid groups | 18,156 | 324 x 45 x 27 | `cable_hang_stick`, `cable_scene`, `cable_sort_plug`, `cable_tray_multi`, `cable_untangle_sort` |
| [cloth_bag_fem](cloth_bag_fem/SOURCE.md) | tet + rigid groups | 111,476 | 480 x 284 x 398 | - |
| [cloth_bag_fem_coarse](cloth_bag_fem_coarse/SOURCE.md) | tet + rigid groups | 40,904 | 480 x 284 x 398 | `tool_pack_bag` |
| [cloth_cleaning_wand](cloth_cleaning_wand/SOURCE.md) | cloth + rigid | - | 290 x 68 x 68 | - |
| [cloth_duster_fem](cloth_duster_fem/SOURCE.md) | tet + rigid groups | 28,667 | 290 x 68 x 68 | `tool_duster` |
| [coiled_usb_cable](coiled_usb_cable/SOURCE.md) | tet + rigid groups | 71,030 | 242 x 34 x 48 | `cable_scene`, `cable_untangle_sort` |
| [corded_soldering_iron](corded_soldering_iron/SOURCE.md) | tet + rigid groups | 19,117 | 508 x 100 x 26 | `tool_dock_iron`, `tool_scene` |
| [gap_sponge_wand](gap_sponge_wand/SOURCE.md) | tet + rigid groups | 14,813 | 235 x 18 x 69 | `tool_gap_channel`, `tool_scene` |
| [handled_fabric_bag](handled_fabric_bag/SOURCE.md) | cloth + rigid | - | 480 x 284 x 398 | - |
| [power_strip](power_strip/SOURCE.md) | tet + rigid groups | 35,632 | 772 x 48 x 28 | `cable_retrieve`, `cable_scene` |
| [rigid_pan](rigid_pan/SOURCE.md) | rigid mesh | - | 413 x 240 x 57 | `tool_spatula_cook` |
| [silicone_spatula](silicone_spatula/SOURCE.md) | tet + rigid groups | 29,121 | 295 x 65 x 14 | `tool_pack_bag`, `tool_scene`, `tool_spatula_cook` |
| [socket_mains](socket_mains/SOURCE.md) | rigid socket | - | 21 x 28 x 24 | - |
| [socket_mains_loose](socket_mains_loose/SOURCE.md) | rigid socket | - | 21 x 33 x 25 | - |
| [socket_mains_snug](socket_mains_snug/SOURCE.md) | rigid socket | - | 21 x 28 x 24 | - |
| [socket_mini_b](socket_mini_b/SOURCE.md) | rigid socket | - | 12 x 28 x 24 | - |
| [socket_mini_b_loose](socket_mini_b_loose/SOURCE.md) | rigid socket | - | 12 x 28 x 24 | - |
| [socket_mini_b_snug](socket_mini_b_snug/SOURCE.md) | rigid socket | - | 12 x 28 x 24 | - |
| [socket_station](socket_station/SOURCE.md) | rigid socket | - | 16 x 28 x 24 | - |
| [socket_station_loose](socket_station_loose/SOURCE.md) | rigid socket | - | 16 x 31 x 31 | `tool_dock_iron` |
| [socket_station_snug](socket_station_snug/SOURCE.md) | rigid socket | - | 16 x 28 x 25 | `tool_dock_iron` |
| [socket_usb_a](socket_usb_a/SOURCE.md) | rigid socket | - | 17 x 28 x 24 | - |
| [socket_usb_a_loose](socket_usb_a_loose/SOURCE.md) | rigid socket | - | 17 x 31 x 24 | `cable_plug_place` |
| [socket_usb_a_snug](socket_usb_a_snug/SOURCE.md) | rigid socket | - | 17 x 28 x 24 | `cable_plug_place` |
| [socket_usb_c](socket_usb_c/SOURCE.md) | rigid socket | - | 13 x 28 x 24 | - |
| [socket_usb_c_loose](socket_usb_c_loose/SOURCE.md) | rigid socket | - | 13 x 28 x 24 | - |
| [socket_usb_c_snug](socket_usb_c_snug/SOURCE.md) | rigid socket | - | 13 x 28 x 24 | - |
| [sponge_wand](sponge_wand/SOURCE.md) | tet + rigid groups | 16,874 | 255 x 50 x 91 | `tool_asset_check`, `tool_pack_bag`, `tool_scene`, `tool_sponge_wipe` |
| [toilet_plunger_generated](toilet_plunger_generated/SOURCE.md) | tet + rigid groups | 20,735 | 110 x 110 x 377 | `tool_scene` |
| [usb_hub_gen](usb_hub_gen/SOURCE.md) | tet + rigid groups | 19,025 | 496 x 35 x 15 | `cable_hang_stick`, `cable_plug_place`, `cable_scene`, `cable_sort_plug`, `cable_tray_multi`, `cable_untangle_sort`, `tool_pack_bag` |
| [whiteboard_eraser](whiteboard_eraser/SOURCE.md) | tet + rigid groups | 36,910 | 110 x 48 x 32 | `tool_scene`, `tool_stain_probe`, `tool_whiteboard_erase` |
| [y_splitter_cable](y_splitter_cable/SOURCE.md) | tet + rigid groups | 20,969 | 344 x 132 x 9 | `cable_scene`, `cable_untangle_sort` |
<!-- generated-assets:end -->

## Previews

[`previews.png`](previews.png) shows every catalogued object, and `previews/<id>.png` shows each one
on its own. Both are rendered from `appearance/visual_mesh.glb` and its albedo.

```bash
uv run python examples/10_heterogeneous/assets/catalog_tools.py add      # catalog new generated folders
uv run python examples/10_heterogeneous/assets/catalog_tools.py preview  # redraw the previews
```

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
