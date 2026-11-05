# Blender interchange for storyboard/previs round trips (status as of 2026-11-05)

*Evidence base, read first.* This session could not reach `docs.blender.org` or `developer.blender.org` (DNS blocked by the network policy), and the shared web-search quota was already used up. Almost every finding below therefore comes from **running the official Blender builds that Blender publishes on PyPI** inside this sandbox, plus the add-on source code shipped inside those builds. The builds were `bpy` 5.2.3 LTS on Python 3.13, and `bpy` 4.5.15 LTS and 4.2.23 LTS on Python 3.11. The shipped add-on source includes glTF-Blender-IO v5.2.41 and v4.2.83, the FBX add-on and the extensions CLI. Also used: `usd-core` 26.11, `opentimelineio` 0.18.1, `fake-bpy-module-*` API stubs, and PyPI metadata. A "local test" citation links the exact package that was executed. PyPI upload dates can lag the official release date by a few days. The test scripts are in the session scratchpad, which may not persist: `/tmp/claude-0/-home-user-Speaker-Diarization/e6d7b62f-0368-5f51-9db4-b4ef76e4f814/scratchpad/tests/`.

## 1. Format options and their Blender support (4.2 LTS → 5.2 LTS), plus the release landscape

### Takeaway
As of November 2026 the current Blender release is **5.2.3 LTS, which runs Python 3.13**. 4.5 LTS is still being patched. 4.2 LTS had its last build in July 2026, and 5.3 had no stable build on PyPI.

No single interchange file round-trips everything:
- **glTF** keeps names, custom properties and editable keyframes. It loses the lens value in mm (it is re-derived from FOV with a 24 mm vertical sensor), the render aspect, shift, depth of field (DoF), area lights, and active lens animation.
- **USD** keeps lens, sensor, shift, DoF, lights and custom attributes. On import it renames merged objects and turns transform animation into Transform Cache constraints instead of keyframes.
- **FBX** keeps the camera and keyframes but mangles dictionary properties and light colours.
- **Alembic** drops custom properties and lights on import.
- **COLLADA** is gone in 5.x.

The most robust bridge has two parts:
1. A generated Python build script, or a server-side `bpy` 5.2.3 run, which matches the app's Python 3.13, that writes a native .blend.
2. A Blender extension that exports a JSON sidecar of the user's edits.

glTF extras and USD `userProperties` are good secondary channels, and the server can parse both without Blender.

### Cited Findings

**Release landscape**
- `bpy.app.version_string` reports `"5.2.3 LTS"` (file version (5, 2, 45)), `"4.5.15 LTS"` and `"4.2.23 LTS"` (file version (4, 2, 68)), so all three are LTS lines — local tests of [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/), [bpy 4.5.15](https://pypi.org/project/bpy/4.5.15/), [bpy 4.2.23](https://pypi.org/project/bpy/4.2.23/).
- PyPI upload dates for `bpy` — [bpy release history](https://pypi.org/project/bpy/#history):

  | Version | Upload date |
  |---|---|
  | 4.2.0 | 2024-07-25 |
  | 4.3.0 | 2024-11-21 |
  | 4.4.0 | 2025-03-18 |
  | 4.5.0 | 2025-07-21 |
  | 5.0.0 | 2025-11-18 |
  | 5.0.1 | 2025-12-16 |
  | 5.1.0 | 2026-03-17 |
  | 5.1.2 | 2026-05-19 |
  | 5.2.0 | 2026-07-14 |
  | 5.2.1 | 2026-08-25 |
  | 5.2.2 | 2026-09-15 |
  | 5.2.3 | 2026-10-22 |
  | 4.5.15 | 2026-10-22 |
  | 4.2.23 | 2026-07-21 (last 4.2 upload) |

- No 5.3 `bpy` exists on PyPI, and `fake-bpy-module-5.3` returns 404. `fake-bpy-module-latest` (20261031) is built from the Blender main-branch daily build — [fake-bpy-module-latest](https://pypi.org/project/fake-bpy-module-latest/).
- Python version per Blender line: wheels for 4.2.x through 5.0.x are cp311; wheels for 5.1.x and 5.2.x are cp313. `bpy` 5.2.3 declares `Requires-Python: ==3.13.*` — [bpy on PyPI](https://pypi.org/project/bpy/).

**glTF 2.0 (built-in add-on `io_scene_gltf2`, developed in the [glTF-Blender-IO repo](https://github.com/KhronosGroup/glTF-Blender-IO))**
- Generator string in exported files: `Khronos glTF Blender I/O v5.2.41` from Blender 5.2.3 and `v4.2.83` from 4.2.23 — local tests ([5.2.3](https://pypi.org/project/bpy/5.2.3/), [4.2.23](https://pypi.org/project/bpy/4.2.23/)).
- Exporter `bpy.ops.export_scene.gltf`, listed as identifier → UI name, default — introspected in [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/), with 4.2.23 compared:
  - `export_extras` → "Custom Properties", **False** ("Export custom properties as glTF extras").
  - `export_cameras` → "Cameras", **False**.
  - `export_lights` → "Punctual Lights", **False** ("Uses KHR_lights_punctual").
  - `export_yup` → "+Y Up", True.
  - `export_animations`, True.
  - `export_animation_mode`, default `'ACTIONS'`; choices `ACTIONS | ACTIVE_ACTIONS | BROADCAST | NLA_TRACKS | SCENE`.
  - `export_force_sampling` → "Always Sample Animations", True.
  - `export_frame_range` → "Limit to Playback Range", False.
  - `export_frame_step`, 1.
  - `export_pointer_animation` → "Export Animation Pointer (Experimental)", False: "Export material, Light & Camera animation as Animation Pointer. Available only for baked animation mode 'NLA Tracks' and 'Scene'". It already exists in 4.2.
  - `export_import_convert_lighting_mode` → "Lighting Mode", `SPEC` (Standard) | `COMPAT` (Unitless) | `RAW`, default `SPEC`.
  - Selection filters: `use_selection`, `use_visible`, `use_renderable`, `use_active_collection` (+ `use_active_collection_with_nested`), `use_active_scene`, `collection`.
  - `export_hierarchy_full_collections` → "Full Collection Hierarchy", False.
  - `export_anim_slide_to_zero`, False.
  - `export_optimize_animation_size`, True.
  - `export_bake_animation`, False ("useful when using constraints or driver").
  - `export_current_frame`, False ("When off, frame 0 is used as rest transformations").
  - New since 4.2: `export_merge_animation` (`NLA_TRACK | ACTION | NONE`, default `ACTION`) and `export_sampling_interpolation_fallback` (`LINEAR | STEP`).
- Importer `bpy.ops.import_scene.gltf` in 5.2.3 — [bpy 5.2.3 introspection](https://pypi.org/project/bpy/5.2.3/):
  - New since 4.2: `import_scene_extras`, True: "Import scene extras as custom properties. Existing custom properties will be overwritten".
  - New since 4.2: `import_scene_as_collection` (True), `import_select_created_objects`, `import_unused_materials`, `import_merge_material_slots`, `import_point_as_pointcloud`.
  - Unchanged: `import_pack_images` (True), `merge_vertices`, `import_shading`, `bone_heuristic`, `guess_original_bind_pose` (True), `import_webp_texture`, and the shared `export_import_convert_lighting_mode`.
- Extras export code: `generate_extras()` copies every key of the element's custom properties. In 5.2 it also copies the values from `bl_system_properties_get()`. It skips a BLACK_LIST: `cycles`, `cycles_visibility`, `cycles_curves`, `glTF2ExportSettings`, `gltf2_mesh_applied`, and the `gltf2_*variant*` keys. It is called from the gatherers for nodes, mesh data, cameras, lights, materials, joints, actions/scene animation and scenes. The importer's `set_extras()` writes extras back as custom properties — [glTF-Blender-IO](https://github.com/KhronosGroup/glTF-Blender-IO), `io_scene_gltf2/blender/com/extras.py` as shipped in bpy 5.2.3.
- Round-trip test: object custom properties came back intact, including int, float, float array and nested dict. They were written as `node.extras` and restored on re-import. Camera-data properties went to `cameras[].extras` and scene properties to `scenes[].extras`, and both came back (also in 4.2.23). A collection's properties are exported only with `export_hierarchy_full_collections=True`, which emits a node `"SHOT_010"` with `extras {"p2s_shot_id": ...}` — local tests [5.2.3](https://pypi.org/project/bpy/5.2.3/), [4.2.23](https://pypi.org/project/bpy/4.2.23/).
- Axis conversion with +Y Up: Blender (x, y, z) becomes glTF (x, z, −y). For example, camera location (0, −8, 1.6) was written as `[0, 1.6, 8]`. A Blender camera with `rotation_euler = (90°, 0, 0)`, which looks horizontally along +Y, is written with no rotation, i.e. identity — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- glTF import creates objects with `rotation_mode = 'QUATERNION'` (both versions) — local tests.
- Light export: area lights are not exported (the node remains without a light). Point, spot and sun lights use KHR_lights_punctual. The unit conversions and the 5.2 temperature handling are listed under question 5 — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- The PyPI `bpy` build lacks the Draco and meshoptimizer bridge libraries. The export logs "Draco is not available because library could not be found at …/bpy/lib/libbf_intern_draco_bridge.so", so mesh compression is unavailable in headless `bpy` exports — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).

**USD (built-in C++ `wm.usd_export` / `wm.usd_import`)**
- Export options in 5.2.3 — [bpy 5.2.3 introspection](https://pypi.org/project/bpy/5.2.3/):
  - `export_custom_properties` → "Custom Properties", **True** ("Export custom properties as USD attributes").
  - `custom_properties_namespace`, `'userProperties'`.
  - `author_blender_name` → "Blender Names", True ("Author USD custom attributes containing the original Blender object and object data names").
  - `export_animation`, **False** ("Export all frames in the render frame range").
  - `export_cameras` and `export_lights`, True.
  - `convert_orientation`, False, with `export_global_forward_selection='NEGATIVE_Z'` and `export_global_up_selection='Y'` used when it is enabled.
  - `xform_op_mode`, `TRS | TOS | MAT`.
  - `root_prim_path`, `'/root'`.
  - `relative_paths`, True.
  - `allow_unicode`, True in 5.2 (False in 4.2).
  - New since 4.2: `convert_scene_units` (`METERS` default … `CUSTOM`), `meters_per_unit` (1.0), `merge_parent_xform` (False), `export_textures_mode`, `incremental_frames`.
  - Removed since 4.2: `visible_objects_only`, `export_textures`.
- Import options in 5.2.3 — [bpy 5.2.3 introspection](https://pypi.org/project/bpy/5.2.3/):
  - `import_cameras` and `import_lights`, True.
  - New since 4.2: `property_import_mode` → "Custom Properties", `NONE | USER | ALL`, default `ALL` ("Behavior when importing USD attributes as Blender custom properties"). 4.2 has no such option but still imported `userProperties`.
  - New: `merge_parent_xform`, **True** ("Allow USD primitives to merge with their Xform parent if they are the only child in the hierarchy").
  - New: `apply_unit_conversion_scale`, True ("Scale the scene objects by the USD stage's meters per unit value").
  - `light_intensity_scale`, 1.0.
  - `set_frame_range`, True.
  - `read_mesh_attributes`, True ("Read USD Primvars as mesh attributes").
  - `prim_path_mask`, `create_collection` (False), `import_visible_only` (True), `import_defined_only` (True), `scale` (1.0).
  - There is no animation toggle.
- Exported file from 5.2.3 — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/):
  - Stage metadata: `metersPerUnit = 1`, `upAxis = "Z"`, `timeCodesPerSecond = 24`, `startTimeCode = 1`, `endTimeCode = 96`, `doc = "Blender v5.2.3 LTS"`, `defaultPrim = "root"`.
  - Each object becomes an `Xform` prim with its data as a child prim, e.g. `/root/CAM_shot_010/CAMDATA_shot_010` of type Camera.
  - Custom properties are written as `custom string userProperties:p2s_kind`, `custom int userProperties:p2s_rev`, `custom double3 userProperties:p2s_vec`, and so on. **Dictionary/group properties are silently omitted.**
  - `userProperties:blender:object_name` / `userProperties:blender:data_name` are authored.
  - On animated objects even static properties are written as `.timeSamples`.
  - Scene and collection custom properties are **not** written.
- Lights in the USD file: point → `SphereLight`, spot → `SphereLight` + `inputs:shaping:cone:angle`, area → `RectLight` with width/height, sun → `DistantLight` with `inputs:angle`. Intensities and units are listed under question 5 — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- Server-side parsing without Blender works: `usd-core` 26.11 (wheels cp311–cp314, about 28.8 MB for manylinux x86_64, license `LicenseRef-TOST-1.0`) read every `userProperties:*` attribute, the per-frame `UsdGeom.Camera.GetCamera(t)` values, and the world transforms of Blender's file — [usd-core 26.11](https://pypi.org/project/usd-core/26.11/), local test. Wheels for cp313 exist since usd-core 25.8 (2025-07-31) — [usd-core history](https://pypi.org/project/usd-core/#history).

**FBX (`export_scene.fbx` / `import_scene.fbx`, Python add-on)**
- Export defaults: `use_custom_props` → "Custom Properties", **False**; `bake_anim` True; `bake_anim_step` 1.0; `bake_anim_simplify_factor` 1.0; `apply_unit_scale` True; `apply_scale_options='FBX_SCALE_NONE'`; `axis_forward='-Z'`; `axis_up='Y'`; `object_types` = all; `path_mode='AUTO'`; `embed_textures` False.
- Import defaults: `use_custom_props` True ("Import user properties as custom properties"); `use_custom_props_enum_as_string` True; `use_anim` True; `anim_offset` 1.0.
- Source for both bullets: [bpy 5.2.3 introspection](https://pypi.org/project/bpy/5.2.3/).

**Alembic (`wm.alembic_export` / `wm.alembic_import`)**
- Export: `export_custom_properties` True ("Export custom properties to Alembic .userProperties"); `start`/`end` default to the scene range; `xsamples` 1; `gsamples` 1; `flatten` False.
- Import: `set_frame_range` True; `always_add_cache_reader` False; `is_sequence` False; `scale` 1.0. There is no option to import custom properties.
- Source for both bullets: [bpy 5.2.3 introspection](https://pypi.org/project/bpy/5.2.3/).

**COLLADA**
- `bpy.ops.wm.collada_export/collada_import` exist in 4.2.23 and are absent in 5.2.3, which only has OBJ, PLY, STL, USD and Alembic operators under `wm`, and FBX and glTF under `export_scene`/`import_scene` — local tests [4.2.23](https://pypi.org/project/bpy/4.2.23/), [5.2.3](https://pypi.org/project/bpy/5.2.3/).

**Round-trip matrix.** The test scene was exported from Blender, re-imported into an empty scene in the same Blender version, and inspected. Settings used:
- glTF: `export_extras`, `export_cameras`, `export_lights`, `export_animation_mode='SCENE'`, `export_pointer_animation=True`.
- USD: `export_animation=True` with all other options at default.
- FBX: `use_custom_props=True`.
- Alembic: defaults.

| Item | glTF 5.2 (4.2 same unless noted) | USD 5.2 (4.2 differences) | FBX | Alembic |
|---|---|---|---|---|
| Object names | kept | **merged objects take the child prim (data) name**: CAM_shot_010 → "CAMDATA_shot_010", SUBJ_… → "PROXY_human_1m75". With `merge_parent_xform=False`, Empties keep the object names and the data becomes child objects, lights renamed `.001` | kept | kept |
| Object custom props | kept, incl. nested dict | kept except dict | kept, but dict became a string `'<bpy id prop: owner="OBCAM_shot_010", name="p2s_dict", …>'` | **lost** |
| Camera-data props | kept | kept | kept | lost |
| Scene props | kept | lost | lost | lost |
| Camera transform animation | 96 per-frame LINEAR keys (2 BEZIER keys if `export_force_sampling=False`) | **`TRANSFORM_CACHE` constraint** (`object_path '/root/CAM_shot_010'`), no keyframes | 47 LINEAR keys on loc/rot/scale | `TRANSFORM_CACHE` constraint |
| Lens (zoom) animation | yfov pointer imported as a camera-data action on an **NLA track, not active**: lens stayed 55.695 mm at frames 1/24/48 | 48 LINEAR `lens` keys in 5.2; **none in 4.2** | 46 keys, plus `dof.focus_distance` keys | 96 keys in 5.2; none in 4.2 |
| Lens/sensor | 35 mm → **55.695 mm, sensor_fit VERTICAL** (36×24 sensor) | 35 mm, sensor 36 × 15.082, fit HORIZONTAL | 35 mm, 36×24, AUTO | 35 mm, AUTO |
| Shift / DoF | lost / lost | kept / kept (`use_dof` True in 5.2, False in 4.2) | kept / kept | shift kept / DoF off |
| Lights | point, spot, sun kept (energies restored); **area dropped** (becomes Empty) | all four kept, including area | area → POINT; white lights came back (0.8, 0.8, 0.8) | lights → Empties |
| Frame range | not set (1–250) | set (1–96) | not set | set |

Sources for the matrix: local tests [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/) and [bpy 4.2.23](https://pypi.org/project/bpy/4.2.23/).

**Generated Python script → native .blend**
- One script built the test scene under both 4.2.23 and 5.2.3 with only two version shims: `strips` vs `sequences`, and `new_effect(length=, input1=)` vs `new_effect(frame_end=, seq1=)`. It created:
  - a per-shot collection;
  - a camera with lens, sensor, shift, clip and DoF set;
  - location and `lens` keyframes;
  - proxy meshes built with `mesh.from_pydata`;
  - sun, point, spot and area lights;
  - a timeline marker bound to the camera;
  - a second scene whose VSE holds movie, image and scene strips and a CROSS effect.

  It then saved the .blend with `wm.save_as_mainfile(relative_remap=True)` and exported glTF, USD, FBX and ABC — local tests [5.2.3](https://pypi.org/project/bpy/5.2.3/), [4.2.23](https://pypi.org/project/bpy/4.2.23/).
- A Text datablock's `use_module` property means "Run this text as a Python script on loading". However, the preference `use_scripts_auto_execute` defaults to **False**, so an embedded sync script will not run automatically on open — introspection of [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/) and [4.2.23](https://pypi.org/project/bpy/4.2.23/).
- `bpy.app.handlers` includes `save_pre`, `save_post`, `load_post`, `depsgraph_update_post`, `frame_change_post` and others, which can trigger a sidecar export on save — [bpy 5.2.3 introspection](https://pypi.org/project/bpy/5.2.3/).
- Collection Exporters exist in both 4.2 and 5.2:
  - API: `Collection.exporters` (`CollectionExport` with `filepath` and `export_properties`), plus the operators `collection.exporter_add(name=<FileHandler idname>)`, `collection.exporter_export(index=)` and `collection.export_all`.
  - In 5.2, `exporter_add` accepted `IO_FH_gltf2`, `IO_FH_usd`, `IO_FH_fbx` and `IO_FH_alembic`.
  - `exporter_export` wrote a GLB containing only that collection, with the settings persisted in the .blend.
  - Source: local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).

**`bpy` PyPI module (Blender as a Python module)**
- PyPI description — [bpy on PyPI](https://pypi.org/project/bpy/):
  - "This package provides Blender as a Python module for use in studio pipelines, web services…"
  - "Each Blender release supports one Python version, and the package is only compatible with that version."
  - "Blender versions outside the current LTS window are removed from PyPI but are available at https://download.blender.org/pypi/bpy/".
  - License field: `GPL-3.0`.
- Wheels for 5.2.3: manylinux_2_28_x86_64 402.8 MB, macosx_11_0_arm64 245.2 MB, win_amd64 339.6 MB, win_arm64 210.6 MB. 4.2.x and 4.5.x also ship macOS x86_64 wheels; **5.0+ has no macOS x86_64 wheel**, and no Linux aarch64 wheel exists for any version — [bpy files](https://pypi.org/project/bpy/5.2.3/#files).
- Installed footprint of 5.2.3 is 875 MB, and `import bpy` takes about 0.74 s. The Linux `.so` links system `libGL`, `libGLX`, `libX11`, `libXi`, `libXext`, `libXfixes`, `libXrender`, `libxkbcommon`, `libSM`, `libICE` and `libxcb`. It worked fully headless for building, saving, importing and exporting (rendering was not tested) — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).

**Extensions (4.2+ platform)**
- The manifest file is `blender_manifest.toml`.
  - Required keys: `id`, `schema_version`, `name`, `tagline`, `version`, `type`, `maintainer`, `license` (a list of strings), `blender_version_min`.
  - Optional keys: `blender_version_max`, `website`, `copyright`, `permissions`, `tags`, `platforms`, `wheels`.
  - `type` must be one of `{"add-on", "theme"}`.
  - `permissions` keys are limited to `files`, `network`, `clipboard`, `camera` and `microphone`, each with a short string explaining why. A list form is accepted only in non-strict mode, for legacy files.
  - `wheels` entries must end in `.whl`.
  - `[build]` accepts `paths` or `paths_exclude_pattern`, not both.
  - Source: [Blender `scripts/addons_core/bl_pkg/cli/blender_ext.py`](https://projects.blender.org/blender/blender), as shipped in bpy 5.2.3.
- `blender_ext.py` runs standalone under plain Python 3.13 without Blender:
  - `validate <dir>` printed "Success parsing TOML".
  - `build --source-dir … --output-dir …` produced `p2s_bridge-0.1.0.zip` containing `blender_manifest.toml` and `__init__.py`.
  - A bad key was rejected with: `key "filesystem" must be one of ('files', 'microphone', 'camera', 'network', 'clipboard')`.
  - Source: local test with the [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/) copy.

### Inferences
- **Primary export path.**
  1. The server emits a versioned `manifest.json` with shots, cameras, subjects, set elements, lights, clips and fps, plus a media folder.
  2. A build script turns it into a scene. Run the script either server-side with `bpy` 5.2.3, which matches Python 3.13 and needs about 875 MB plus X11/GL libraries in the image, or in the user's Blender through the extension's "Import P2S package" operator.
  3. A server-built .blend is only openable by Blender 4.5+ (question 5). If users may still run 4.2, ship the package and build on their machine; the script ran unchanged on 4.2 and 5.2 apart from the two shims.
- **Return path.** The extension writes a JSON sidecar on demand or in a `save_post` handler. It contains per-shot camera matrices and lens at keyed frames, sensor, shift, DoF, subject/element transforms, strip and marker timing, and a fingerprint. The server diffs fingerprints per shot. This avoids every lossy exporter conversion in the matrix.
- **Fallback channels the server can read without Blender.** glTF (`pygltflib`, pure Python) and USD (`usd-core`, which supports Python 3.13). glTF is the better "open anywhere" deliverable for editable keyframes. USD is better for camera fidelity but not for hand-editing animation.
- Avoid FBX and Alembic for metadata round trips.

### Gaps
- Whether 5.3 was in beta or released on 2026-11-05 could not be confirmed: no PyPI build, and the release notes were unreachable.
- The official release-notes and manual wording for each change was not quoted, because docs.blender.org and developer.blender.org were blocked.
- `blender` executable CLI flags such as `-b`, `--python`, `--python-exit-code`, `--factory-startup` and `--command extension build` were not verified, because the PyPI wheel has no executable. Only the Python side of `extension build`/`validate` was run.
- Support for legacy `bl_info` add-ons in 5.x was not verified.
- GPL implications were not verified: whether generated scripts or extensions that `import bpy`, or a server that links `bpy`, must be GPL. The Blender license FAQ was unreachable. PyPI lists `GPL-3.0`.
- The ratification status of KHR_animation_pointer was not checked; Blender labels the option "(Experimental)".
- The `export_format` enum is dynamic and could not be introspected. `'GLTF_SEPARATE'` was used successfully, and Collection Exporters default to `.glb`.

## 2. Stable identity for round trips (tagging objects so edits map back to shot/subject/element IDs)

### Takeaway
Use **custom properties** as the identity carrier, written onto objects, object data (camera and light), per-shot collections, the scene and VSE strips. Do not rely on names, which users can rename, which get auto-suffixed, and which are capped at 63 bytes in 4.x. Do not rely on `session_uid`, which is not persistent.

Custom properties survive .blend and glTF fully. They survive USD except dictionaries. FBX survives except dictionaries, and Alembic does not survive at all. Duplicating an object copies its custom properties, so the importer must detect ID collisions.

### Cited Findings
- Custom properties set with `obj["p2s_shot_id"] = "shot_010"` on objects, camera data, collections, scenes and VSE strips persisted in .blend files. They opened correctly in 5.2.3 from a 4.2 file, and in 4.5.15 from a 5.2 file — local tests [5.2.3](https://pypi.org/project/bpy/5.2.3/), [4.5.15](https://pypi.org/project/bpy/4.5.15/).
- `bpy.ops.object.duplicate()` (Shift+D) produced `SUBJ_shot_010_hero.001` carrying the same `p2s_subject_id="hero"` — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- `ID.session_uid` changed from 169 to 230 after re-opening the same file, so it is runtime-only — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- Name length: `ID.name` `length_max` is 64 in 4.2.23 and 4.5.15, i.e. 63 bytes. A 100-character name was truncated to 63. In 5.2.3 `length_max` is 256 and 100 characters were stored intact — local tests [4.2.23](https://pypi.org/project/bpy/4.2.23/), [4.5.15](https://pypi.org/project/bpy/4.5.15/), [5.2.3](https://pypi.org/project/bpy/5.2.3/).
- glTF behaviour, in brief:
  - Extras are emitted for nodes, meshes, cameras, lights, materials, actions and scenes.
  - In `SCENE` animation mode the animation's `extras` also carry the object's properties.
  - Collections appear as nodes with extras only with "Full Collection Hierarchy".
  - The importer restores extras as custom properties. Scene extras are controlled by `import_scene_extras` in 5.2.
  - Source: [glTF-Blender-IO](https://github.com/KhronosGroup/glTF-Blender-IO) source as shipped in bpy 5.2.3, plus local tests.
- USD behaviour, in brief:
  - Properties are written as `userProperties:<name>`, with the namespace configurable via `custom_properties_namespace`. Names that already carry a namespace prefix are written as-is (from the option description).
  - Dictionaries are dropped.
  - `userProperties:blender:object_name` is authored, but the importer did **not** use it to restore object names (with default `merge_parent_xform=True`).
  - Re-imported custom properties land on the merged object, or on the Xform Empty when merging is off.
  - Source: local tests [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/); option descriptions from introspection.
- In 5.2 the glTF exporter reads both user custom properties (`element.keys()`) and system properties (`element.bl_system_properties_get()`). So properties registered via `bpy.props` by an add-on also become extras — [glTF-Blender-IO](https://github.com/KhronosGroup/glTF-Blender-IO), `blender/com/extras.py` as shipped in bpy 5.2.3.
- FBX import in 4.2.23 added an empty `cycles` group property to lights — local test [bpy 4.2.23](https://pypi.org/project/bpy/4.2.23/).

### Inferences
- Suggested schema:
  - Scene: `p2s_project_id`, `p2s_schema`, `p2s_export_id`, `p2s_fps`.
  - Collection `SHOT_<id>`: `p2s_shot_id`, `p2s_rev`, `p2s_hash`.
  - Each object: `p2s_id` (UUID), `p2s_kind` (camera | subject | set_element | key_light | atmosphere), `p2s_shot_id`, `p2s_ref` (subject_id or element kind).
  - Duplicate `p2s_id`/`p2s_kind` on the object data (camera and light), because USD/glTF importers split or merge objects and data.
- Keep values scalar (str, int, float, flat float arrays). Encode structured data as a JSON string, because USD drops dictionaries and FBX corrupts them.
- **Change detection.** At export, store a canonical fingerprint per shot, e.g. a hash of rounded camera keys, lens, sensor, shift, subject transforms and strip timing, in `p2s_hash`. On return, recompute it; only shots whose recomputed hash differs are updated.
  - Objects without `p2s_id` are user additions.
  - IDs that are missing are deletions.
  - An ID that appears twice is a duplicate. Keep the object whose name matches the exported name as the original and treat the rest as new elements.
- Always store the shot's exported values too, either in the manifest or as properties such as `p2s_src_lens`, so the server can tell "user edited" apart from "lossy conversion". This matters most for the glTF lens change from 35 to 55.695 mm.

### Gaps
- Library linking and overrides, and asset-library workflows, were not tested.
- Blender exposes no persistent per-ID UUID; none was found in the RNA properties introspected (`session_uid` is runtime-only). A full RNA scan for one was not done.

## 3. Camera maths and conventions

### Takeaway
A Blender camera looks down its local −Z axis with +Y up. Its frame size is set by `lens` (mm) together with `sensor_width`/`sensor_height` and `sensor_fit`, and by the render aspect. `Camera.angle_y` is **sensor-height based, not the real vertical FOV**.

glTF stores only `yfov` (the real vertical FOV) and `aspectRatio`. USD stores focal length and apertures in tenths of a scene unit.

All of the following were verified numerically against `bpy_extras.object_utils.world_to_camera_view`: the forward and inverse pinhole formulas, the shift convention, the horizon-to-tilt formula, the ground-plane placement, and the scale-to-distance relation.

### Cited Findings
- Camera RNA in 5.2.3:
  - Lens: `lens` (default 50, "Perspective Camera focal length value in millimeters"), `lens_unit` (`MILLIMETERS | FOV`), `type` (`PERSP` …).
  - Sensor: `sensor_width` (36, "Horizontal size of the image sensor area in millimeters"), `sensor_height` (24), `sensor_fit` (`AUTO | HORIZONTAL | VERTICAL`).
  - Framing: `shift_x`/`shift_y` (0), `clip_start` (0.1), `clip_end` (1000), `angle`/`angle_x`/`angle_y`.
  - DoF: `dof.use_dof` (False), `dof.focus_object`, `dof.focus_distance` (10), `dof.aperture_fstop` (2.8), `aperture_blades`, `aperture_rotation`, `aperture_ratio`.
  - Background images: `background_images` and `show_background_images`.
  - New in 5.x: custom-camera fields (`custom_filepath`, `custom_shader`, `custom_mode`) and `composition_guide_color`.
  - Source: [bpy 5.2.3 introspection](https://pypi.org/project/bpy/5.2.3/).
- With 35 mm, a 36 mm sensor, AUTO fit and 2048×858 (aspect A = 2.38695):
  - `angle_x` = 54.4322°, i.e. 2·atan(18/35).
  - `angle_y` = 37.8493°, i.e. 2·atan(12/35), which is sensor-height based.
  - The real vertical FOV is 2·atan(tan(hfov/2)/A) = **24.3179°**, matching the exported glTF `yfov = 0.424427` rad.
  - Source: local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- glTF export formulas:
  - `aspectRatio = (resolution_x·pixel_aspect_x)/(resolution_y·pixel_aspect_y)`.
  - `yfov = yvof_blender_to_gltf(cam.angle, w, h, sensor_fit)`. When w ≥ h it returns `2·atan(tan(angle/2)/aspect)` unless the fit is VERTICAL, in which case it returns `angle`. When w < h it returns `angle` unless the fit is HORIZONTAL, in which case it applies the same conversion.
  - `znear = clip_start`, `zfar = clip_end`.
  - Source: [glTF-Blender-IO](https://github.com/KhronosGroup/glTF-Blender-IO), `blender/com/conversion.py` and `blender/exp/cameras.py` as shipped in bpy 5.2.3.
- glTF import code: `cam.angle_y = yfov; cam.lens_unit = "FOV"; cam.sensor_fit = "VERTICAL"`, with the comment `# TODO: fov/aspect ratio` (the aspect is ignored and the render resolution is not set). A missing `zfar` becomes `clip_end = 1e12`. Orthographic import uses `ortho_scale = max(xmag, ymag)·2` — [glTF-Blender-IO](https://github.com/KhronosGroup/glTF-Blender-IO), `blender/imp/camera.py` as shipped in bpy 5.2.3.
- Shift is measured in units of the **larger** frame dimension. `shift_y = 0.05` moved the frame-centre point from normalised y = 0.5 to 0.3807, i.e. 0.5 − 0.05·A. The USD export wrote `verticalApertureOffset = 0.018`, i.e. 0.05 × 36 mm = 1.8 mm in tenths of a metre — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- `world_to_camera_view(scene, cam_obj, coord)` returns (x, y, depth):
  - x and y are normalised with (0, 0) at the bottom-left and (1, 1) at the top-right; depth is the distance along the view axis.
  - A point 8 m ahead on the optical axis gave (0.5, 0.5, 8.0). Feet at (0.5, 0, 0) gave (0.5608, 0.0359) and the head at 1.75 m gave (0.5608, 0.5435). The camera was at (0, −8, 1.6) looking along +Y.
  - `Camera.view_frame(scene=)` returned corners (±0.5, ±0.2095, −0.9722), i.e. (±0.5, ±0.5/A, −f/Sw).
  - Source: local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- Orientation, verified:
  - `rotation_euler = (90°, 0, 0)` looks along world +Y with +Z up.
  - The exact construction is `R = Rz(pan) @ Rx(90°+tilt) @ Rz(roll)`, giving view direction (−sin pan·cos tilt, cos pan·cos tilt, sin tilt).
  - With roll = 0, `R.to_euler('XYZ')` equals (90°+tilt, 0, pan) exactly; pan 10° and tilt −8° gave (82°, 0, 10°).
  - With roll ≠ 0, the naive XYZ Euler (90°+tilt, roll, pan) is **not** a pure roll. Tilt −15° with roll 10° gave view direction (−0.0449, 0.9659, −0.2549) instead of (0, 0.9659, −0.2588).
  - Source: local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- Horizon: for a level-roll camera the horizon appears at normalised y = 0.5 − (f/Sw)·A·tan(tilt). This was measured as 0.8261 for tilt −8°, matching the formula — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- Inverse placement on the ground plane z = 0, verified:
  - Ray in camera space: d_c = ((u−0.5)·Sw/f, (v−0.5)·Sw/(f·A), −1).
  - World ray: d_w = R·d_c; t = −C_z/d_w.z; P = C + t·d_w.
  - Re-projecting P returned exactly (u, v) = (0.62, 0.12).
  - For a level camera, depth Z = f·H_s/(s·Sh_eff) with Sh_eff = Sw/A, and camera height h = H_s·(0.5 − v_feet)/s. Measured: s = 0.4061 gave Z = 10.0 and h = 1.6, as built.
  - Source: local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- USD camera units:
  - UsdGeomCamera documents `focalLength`, `horizontalAperture` and `verticalAperture` "in tenths of a scene unit", `horizontalApertureOffset` in the same units as the aperture, and `clippingRange` and `focusDistance` "in scene units". The fallback for an unauthored stage is `metersPerUnit` 0.01 with `upAxis` Y.
  - Blender (metersPerUnit = 1) wrote `focalLength 0.35` for 35 mm, `horizontalAperture 0.36`, `verticalAperture 0.15082` (36/A mm), `clippingRange (0.1, 500)`, `focusDistance 6`, `fStop 2.8`, `projection "perspective"`, and per-frame `focalLength` timeSamples for an animated lens.
  - Sources: schema docs read from [usd-core 26.11](https://pypi.org/project/usd-core/26.11/) via `Usd.SchemaRegistry` (cf. [UsdGeomCamera reference](https://openusd.org/release/api/class_usd_geom_camera.html), not fetched in-session); local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- glTF spec semantics: `yfov` is the vertical FOV in radians, `aspectRatio` is optional, and the camera looks down −Z with +Y up. These were used here as implemented by Blender's code above — [glTF 2.0 spec](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) (not fetched in-session).
- Camera background images:
  - `cam.data.background_images.new()` exposes `source` (`IMAGE | MOVIE_CLIP`), `image`, `clip`, `alpha` (0.5 when new), `frame_method` (`STRETCH | FIT | CROP`), `display_depth` (`BACK | FRONT`), `offset`, `scale` and `rotation`.
  - The default `frame_method` was FIT in 5.2 and STRETCH in 4.2.
  - Loading the shot's PNG worked in both versions.
  - Source: local tests [5.2.3](https://pypi.org/project/bpy/5.2.3/), [4.2.23](https://pypi.org/project/bpy/4.2.23/).

### Inferences
- **Storyboard → 3D, per shot.**
  1. Pick the render resolution from the aspect ratio, e.g. 2.39:1 → 2048×858. Set `sensor_fit='HORIZONTAL'` (not AUTO, so portrait or square frames do not silently refit) with `sensor_width=36`, and set `lens` from the shot.
  2. Convert the app's top-left frame coordinates to Blender's bottom-left: v = 1 − y_app.
  3. Tilt from horizon height: tilt = atan((0.5 − v_horizon)·Sw/(f·A)). Alternatively keep the camera level and use `shift_y = (v_horizon − 0.5)/A` for a keystone-free look. Roll is applied last as `Rz(roll)`.
  4. Subject with visible feet: ray-cast the feet point to z = 0, then check the predicted scale s′ = f·H_s/(Z·Sh_eff). If s′ differs from the board's s, adjust the camera height or the subject height, using h = H_s·(0.5 − v_feet)/s for level cameras.
  5. Close-ups with feet out of frame: place the subject along the ray through the head or centre at depth Z = H_s/(s·2·tan(vfov/2)).
  6. Set elements (boxes): ray-cast the bottom edge to the ground and size by angular extent.
  7. Overlay the generated keyframe image as a camera background image so users can align proxies by eye.
- **3D → storyboard on return.** Evaluate each subject's feet, head and centre at a reference frame with `world_to_camera_view`, then compute scale = (v_head − v_feet) and y_app = 1 − v. Recover the lens in the app's convention as f36 = Sw_app/2 / tan(hfov/2), where hfov is computed from Blender's lens, sensor and fit. From glTF, use **f36 = 18/(A·tan(yfov/2))** and ignore the importer's 55.695 mm value.
- Camera moves as keyframes (not tested):
  - dolly: `location` along the view vector.
  - truck: along the local X axis.
  - crane/pedestal: world Z.
  - pan: `Rz` about world Z.
  - tilt: about local X.
  - orbit: parent to an Empty at the subject and key the Empty's Z rotation.
  - zoom: `data.lens` keys.
  - handheld: a NOISE F-Curve modifier on rotation and location.

  glTF exports constraint- or modifier-driven motion only when sampled or baked (`export_bake_animation`).
- Object "front" convention: Blender's Front view looks along +Y, so proxies modelled facing −Y are facing the front view. This is a Blender convention not re-verified in-session. Map "facing" to Z rotation relative to the camera direction.

### Gaps
- The shift formula was verified for landscape and AUTO/HORIZONTAL fit only. Portrait frames and VERTICAL fit were not tested numerically.
- Orthographic and panoramic cameras were not exercised.
- Composing shift with tilt in the inverse ray was derived, not tested; it adds +shift_x and +shift_y·A to (u − 0.5) and (v − 0.5).

## 4. Timeline/editorial: VSE via Python, API changes 4.x → 5.x, camera markers, OpenTimelineIO

### Takeaway
Build a separate "EDIT" scene whose Video Sequence Editor holds three kinds of strip: per-shot movie strips (the generated clips), image strips (the keyframes), and/or scene strips pointing at the previs scene with a per-strip `scene_camera`. Inside the previs scene, bind cameras to `timeline_markers`.

The VSE Python API was renamed in stages across 4.4, 4.5, 5.0 and 5.1; 5.2 deprecates the old timing names for removal in 6.0. Blender has no EDL, FCPXML or OTIO importer in core, and OTIO has no Blender adapter, so the bridge must map OTIO ↔ VSE itself.

### Cited Findings
- **API by version.** Verified on the 4.2.23, 4.5.15 and 5.2.3 binaries; the 4.4, 5.0 and 5.1 rows come from the [fake-bpy-module-4.4](https://pypi.org/project/fake-bpy-module-4.4/), [-5.0](https://pypi.org/project/fake-bpy-module-5.0/) and [-5.1](https://pypi.org/project/fake-bpy-module-5.1/) stubs.

  | Version | Change |
  |---|---|
  | 4.2 | `scene.sequence_editor.sequences` / `sequences_all`; types `Sequence`, `MovieSequence`, `SceneSequence`, `ImageSequence`, `CrossSequence`; `new_effect(name, type, channel, frame_start, frame_end, seq1, seq2)`. The 4.2 stubs also list `seq3`; the 4.2.23 binary has only `seq1` and `seq2`. Effect types include `OVER_DROP` and `TRANSFORM`. |
  | 4.4 | Adds `strips` / `strips_all` alongside the deprecated `sequences`. Types renamed to `Strip`, `MovieStrip`, `SceneStrip`, …. `new_effect` still uses `frame_end, seq1, seq2`. |
  | 4.5 | `new_effect(…, frame_end, input1, input2)`. The `sequences` description reads "(Deprecated: Replaced by '.strips') Top-level strips only". The `OVER_DROP` effect type is gone (stubs). |
  | 5.0 | `sequences` removed. `new_effect(…, frame_start, length, input1, input2)`. Adds `WorkSpace.sequencer_scene` (UI code reads `context.sequencer_scene`). The `TRANSFORM` effect type is gone (stubs). |
  | 5.1 | Adds `Strip.left_handle`, `right_handle`, `duration`, `content_start`, `content_end`, `content_duration`, `left_handle_offset`, `right_handle_offset`, and `content_trim_start/end` on media strips. |
  | 5.2 | Reading `frame_final_start`, `frame_final_end` or `frame_final_duration` emits `DeprecationWarning: '…frame_final_end' is expected to be removed in Blender 6.0`. The `COMPOSITOR` effect type appears. `new_movie` and `new_sound` gain a `stream` parameter. `Strip.connections` and `SceneStrip.view_layer` appear in the binary; the stubs show them first in 5.2, but the stub parse may miss earlier versions. |

- Stable signatures in 5.2.3 — [bpy 5.2.3 introspection](https://pypi.org/project/bpy/5.2.3/):
  - `new_movie(name, filepath, channel, frame_start, fit_method={FIT,FILL,STRETCH,ORIGINAL}, stream)`
  - `new_image(name, filepath, channel, frame_start, fit_method)`
  - `new_scene(name, scene, channel, frame_start)`
  - `new_sound(...)`, `new_meta`, `new_clip`, `new_mask`, `remove(strip)`
  - `SceneStrip.scene_camera`
  - `SceneStrip.scene_input`: `CAMERA` ("Use the Scene's 3D camera as input") or `SEQUENCER`
  - `TimelineMarker.camera`, also present in 4.2
  - `MovieStrip.fps`, read-only
- Behaviour in local tests [5.2.3](https://pypi.org/project/bpy/5.2.3/) and [4.2.23](https://pypi.org/project/bpy/4.2.23/):
  - A 2 s, 24 fps MP4 became a 48-frame strip with `fps` 24.0.
  - A 23.976 fps clip reported `fps` 23.976 and was still laid out 1:1 as 48 frames in the 24 fps scene. The RNA function neither retimes the clip nor changes the scene fps.
  - A scene strip defaulted to the full scene range (96 frames) with `scene_input='CAMERA'`.
  - The CROSS effect length followed the overlap of its inputs. Butted, non-overlapping clips produced a zero-length 49→49 effect; an 8-frame overlap produced 41→49.
  - Custom properties on strips worked.
- UI operator `sequencer.movie_strip_add` defaults: `adjust_playback_rate=True` ("Play at normal speed regardless of scene FPS"), `use_framerate=True` ("Set frame rate of the current scene to the frame rate of the movie"), `sound=True`, `fit_method='FIT'`, `set_view_transform=True`, `relative_path=True`. The RNA `strips.new_movie()` does none of these — [bpy 5.2.3 introspection](https://pypi.org/project/bpy/5.2.3/).
- Setting `ImageStrip.duration = 36` on a strip starting at frame 10 gave `left_handle` 10 and `right_handle` 46. The same timing names are absent in 4.5.15 — local tests [5.2.3](https://pypi.org/project/bpy/5.2.3/), [4.5.15](https://pypi.org/project/bpy/4.5.15/).
- The factory startup loaded via `read_factory_settings(use_empty=True)` has no "Video Editing" workspace; it contains Animation, Compositing, Geometry Nodes, Layout, Modeling, Rendering, Scripting, Sculpting, Shading, Texture Paint and UV Editing. In 5.2, `workspace.sequencer_scene = edit_scene` could be set from a script — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- OpenTimelineIO:
  - `opentimelineio` 0.18.1 (2025-11-09, cp39–cp313 wheels, Apache-2.0) has only the built-in adapters `otio_json`, `otioz` and `otiod`.
  - Its schema types include Timeline, Stack, Track, Clip, Gap, Transition, Marker, ExternalReference, ImageSequenceReference, LinearTimeWarp and FreezeFrame.
  - Source: [opentimelineio 0.18.1](https://pypi.org/project/opentimelineio/0.18.1/), local introspection.
- Separate OTIO adapter packages on PyPI:
  - [otio-cmx3600-adapter 1.0.0](https://pypi.org/project/otio-cmx3600-adapter/) (EDL, 2023-07-07)
  - [otio-fcp-adapter 1.0.0](https://pypi.org/project/otio-fcp-adapter/) (FCP7 XML)
  - [otio-fcpx-xml-adapter 1.0.0](https://pypi.org/project/otio-fcpx-xml-adapter/)
  - [otio-aaf-adapter 2.0.0](https://pypi.org/project/otio-aaf-adapter/) (2025-11-21)
  - [otio-ale-adapter 1.1.0](https://pypi.org/project/otio-ale-adapter/) (2025-12-12)
  - [otio-kdenlive-adapter 0.0.3](https://pypi.org/project/otio-kdenlive-adapter/) (2026-01-03)
  - None of the names `blender-otio`, `otio-blender`, `vse-otio`, `opentimelineio-blender` or `bpy-otio` exist on PyPI (404).
- Blender's core add-ons contain no EDL, OTIO or FCPXML importer. A grep of the bundled scripts in 4.2.23 and 5.2.3 found nothing. The 4.2 core add-ons are: bl_pkg, copy_global_transform, cycles, hydra_storm, io_anim_bvh, io_curve_svg, io_mesh_uv_layout, io_scene_fbx, io_scene_gltf2, node_wrangler, pose_library, rigify, ui_translate, viewport_vr_preview — local tests [4.2.23](https://pypi.org/project/bpy/4.2.23/), [5.2.3](https://pypi.org/project/bpy/5.2.3/).

### Inferences
- **Two ways to structure the edit:**
  - **(a) One continuous previs scene.** Lay shot k out at frames [S_k, S_k + D_k). Call `scene.timeline_markers.new(f"shot_{id}", frame=S_k)` and set `marker.camera = cam_k`, so the scene cuts between cameras by itself. Add the EDIT scene's strips on top.
  - **(b) One scene per shot.** Use scene strips with `scene_camera` overrides.

  Option (a) keeps a single frame space for re-import. Option (b) isolates shots but multiplies scenes.
- **Version shim.** Pick the collection with `strips` if present, else `sequences`. Pick `new_effect` keywords by inspecting `bpy.types.StripsTopLevel.bl_rna.functions["new_effect"].parameters`; fall back to `SequencesTopLevel` on 4.2. Use `left_handle`/`duration` when present (5.1+), else `frame_final_*`. Make clips overlap by the transition length before adding CROSS.
- **Retime detection on return.** Read each strip's handles and `channel`, and each marker's frame, and compare them with the exported values stored on the strip, e.g. `strip["p2s_shot_id"]`.
- **OTIO mapping**, implemented in the bridge:

  | OTIO | Blender VSE |
  |---|---|
  | Track | channel |
  | Clip with ExternalReference | `new_movie`, trimmed via `content_trim_start` / `left_handle_offset` from `source_range` |
  | Gap | a frame offset |
  | Transition (SMPTE_Dissolve) | CROSS effect over the overlap |
  | Marker | `timeline_markers` |
  | `RationalTime(rate)` | `scene.render.fps` / `fps_base` |

  This lets the app import or export EDL, FCPXML and AAF through OTIO adapters without any Blender-side dependency.

### Gaps
- Community OTIO/EDL add-ons for Blender, such as studio pipeline tools, could not be searched for (no web search).
- 5.3 VSE changes are unknown.
- Behaviour of `bpy.ops.sequencer.*` operators in background mode without a sequencer area was not tested; only the RNA `strips.new_*` functions were.

## 5. Gotchas (units, frame rates, media formats, paths, packing, version compatibility)

### Takeaway
The main traps:
- 5.x .blend files cannot be opened by 4.2. They can be opened by 4.5 LTS.
- 5.2 compresses .blend files by default.
- Video cannot be packed into a .blend.
- Relative OS paths passed to `bpy` resolve against "/".
- SVG fails silently as an image.
- AVIF works only in 5.x.
- glTF import ignores the aspect ratio, refits the lens, and leaves imported lens animation inactive on the NLA.
- USD and Alembic import turn animation into cache constraints.
- glTF import re-times keys with the scene fps.
- Blender exports USD with metersPerUnit 1 and Z up, while the USD fallback is 0.01 and Y up.
- Legacy `Action.fcurves` is gone in 5.0, so read keys through `bpy_extras.anim_utils.action_get_channelbag_for_slot`.

### Cited Findings
- **.blend compatibility.**
  - An uncompressed 5.2 .blend starts with `BLENDER17-01v0502`; a 4.2 .blend starts with `BLENDER-v402`.
  - 4.2.23 refused both the compressed and uncompressed 5.2 files: "Failed to read blend file …, not a blend file".
  - 4.5.15 opened both. It reported file version (5, 2, 45) with all objects, strips, custom properties and camera-bound markers present, plus warnings such as "error: negative users in strip".
  - 5.2.3 opened the 4.2 file with strips and markers intact.
  - Source: local tests [4.2.23](https://pypi.org/project/bpy/4.2.23/), [4.5.15](https://pypi.org/project/bpy/4.5.15/), [5.2.3](https://pypi.org/project/bpy/5.2.3/).
- **Compression.** The preference `filepaths.use_file_compression` is True in 5.2.3 and False in 4.2.23. With it on, saved files begin with the zstd magic `28 B5 2F FD`; the test file was 93 KB compressed versus 579 KB with `compress=False`. Tools that parse .blend files must handle zstd. Blender ships a pure-Python header reader, `scripts/modules/_blendfile_header.py`, which supports the BHead4, SmallBHead8 and LargeBHead8 block layouts — local test and source in [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- **Paths.**
  - Passing `"media/clip.mp4"` (relative to the OS working directory) to `new_movie`/`images.load` in an unsaved session was resolved against "/". `pack_all` then failed with "source path '/media/key.avif' not found", and the path was stored as `//../../../../../../../media/...`.
  - With absolute inputs, `save_as_mainfile(relative_remap=True)` stored `//../../media/clip_shot_010.mp4`, where `//` means relative to the .blend.
  - `preferences.filepaths.use_relative_paths` defaults to True.
  - Source: local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- **Packing.** `bpy.ops.file.pack_all()` packed images (`Image.packed_file`). `MovieStrip` has no packed-file property, and `MovieClip` exposes no `packed_file` either, so generated video must ship next to the .blend. File operators available: `pack_all`, `unpack_all`, `make_paths_relative`, `make_paths_absolute`, `find_missing_files`, `report_missing_files`, `autopack_toggle` — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).
- **Image formats.**
  - `bpy.path.extensions_image` in 5.2.3: .avif .bmp .cin .dds .dpx .exr .hdr .j2c .jp2 .jpeg .jpg .pdd .png .psb .psd .rgb .rgba .sgi .tga .tif .tiff .tx .webp. There is no .svg.
  - `images.load("x.svg")` did **not** raise. It logged "IMB_load_image_from_memory: unknown file-format" and returned a 0×0 image, so check `img.size`.
  - AVIF loaded in 5.2.3 but failed the same silent way in 4.2.23. WebP loaded in both.
  - SVG can only come in as curves, through the core add-on `io_curve_svg` (FileHandler `IO_FH_svg_as_curves`).
  - Writable image formats in 5.2: AVIF, JPEG, OPEN_EXR, PNG, WEBP, BMP, CINEON, DPX, IRIS, JPEG2000, HDR, TARGA, TIFF, OPEN_EXR_MULTILAYER. 4.2 has no AVIF.
  - Source: local tests [5.2.3](https://pypi.org/project/bpy/5.2.3/), [4.2.23](https://pypi.org/project/bpy/4.2.23/).
- **Video.**
  - `bpy.path.extensions_movie` in 5.2.3 includes .mp4 .mov .mkv .webm .avi .mxf .m4v .mpg .mpeg .ts .m2ts .ogv .gif .wmv .flv .dv .r3d.
  - FFmpeg containers in 5.2: MPEG4, MKV, WEBM, AVI, DV, FLASH, MPEG1, MPEG2, OGG, QUICKTIME.
  - FFmpeg codecs in 5.2: AV1, H264, H265, PRORES, DNXHD, FFV1, PNG, QTRLE, THEORA, among others. 4.2 lacks H265 and PRORES.
  - Source: local tests [5.2.3](https://pypi.org/project/bpy/5.2.3/), [4.2.23](https://pypi.org/project/bpy/4.2.23/).
- **Frame rates.**
  - Scene defaults are `render.fps = 24` and `render.fps_base = 1.0`; NTSC rates use fps / fps_base, e.g. 24 / 1.001.
  - glTF export writes time as frame/fps, so frame 1 becomes 0.041667 s and no slide to zero is applied by default.
  - glTF import multiplies seconds by the **current** scene's fps × fps_base. Importing the 24 fps file into a 25 fps scene put its keys at frames 1.0417 → 50.0.
  - USD carries `timeCodesPerSecond = 24`, and its import sets the frame range.
  - Source: local tests and [glTF-Blender-IO](https://github.com/KhronosGroup/glTF-Blender-IO), `blender/imp/animation_node.py` as shipped in bpy 5.2.3.
- **Units.**
  - New scenes use `unit_settings.system = 'METRIC'` with `scale_length = 1.0` (1 BU = 1 m).
  - Blender's USD export writes `metersPerUnit = 1` and `upAxis = "Z"`; the 5.x options `convert_scene_units` and `meters_per_unit` control this.
  - USD's fallback when unauthored is `metersPerUnit` 0.01 and `upAxis` Y. Blender's import applies the stage's meters-per-unit scale by default (`apply_unit_conversion_scale=True`).
  - FBX exports with `apply_unit_scale=True` and `apply_scale_options='FBX_SCALE_NONE'` and imported as version "7400".
  - Source: local tests [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/), [usd-core 26.11](https://pypi.org/project/usd-core/26.11/).
- **Lighting units.**
  - glTF Standard mode: point and spot intensity in cd = W/(4π)·683, so 1000 W → 54351.41 cd. Sun intensity in lux = W/m²·683, so 3 → 2049.
  - glTF Unitless (`COMPAT`) mode: 1000 W → 79.577 and 3 → 3.
  - glTF spot cones: `outerConeAngle = spot_size/2` and `innerConeAngle = outer·(1 − spot_blend)`.
  - Constant used: `PBR_WATTS_TO_LUMENS = 683`.
  - In 5.2 the glTF exporter multiplies the light color by `temperature_color` when `use_temperature` is set. `use_temperature`, `temperature` (K) and `exposure` exist in 4.5 and 5.2 but not in 4.2.
  - USD: 1000 W point → `SphereLight inputs:intensity 318.31` (= 1000/π) with `normalize 1`; spot → `inputs:shaping:cone:angle 22.5` (half angle in degrees) and `softness 0.15`; 200 W area → `RectLight intensity 63.66`; 3 W/m² sun → `DistantLight intensity 0.75` with `angle 0.263`.
  - Sources: [glTF-Blender-IO](https://github.com/KhronosGroup/glTF-Blender-IO) (`blender/com/conversion.py`, `blender/exp/lights.py`, `blender/exp/light_spots.py`, `blender/imp/light.py` as shipped in bpy 5.2.3); local tests; spec context in [KHR_lights_punctual](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md) (not fetched in-session).
- **Animation data API.**
  - 4.2 has only legacy `Action.fcurves`, `groups` and `id_root`.
  - 4.5 has both the legacy API and slotted actions (`slots`, `layers`, `is_action_layered`, `AnimData.action_slot`).
  - 5.0+ removed `Action.fcurves` and `groups`. Use `bpy_extras.anim_utils.action_get_channelbag_for_slot(action, animdata.action_slot).fcurves`; helpers `action_ensure_channelbag_for_slot` and `animdata_get_channelbag_for_assigned_slot` also exist.
  - `keyframe_insert()` works unchanged across versions.
  - Sources: local tests [4.2.23](https://pypi.org/project/bpy/4.2.23/), [4.5.15](https://pypi.org/project/bpy/4.5.15/), [5.2.3](https://pypi.org/project/bpy/5.2.3/); [fake-bpy-module-5.0](https://pypi.org/project/fake-bpy-module-5.0/) stub shows only `layers`/`slots` on Action.
- **glTF import animation placement.** Each glTF animation becomes an action pushed to an NLA track. The camera object's action was also assigned as active, but the camera-data lens action was not, so the lens did not animate until assigned. Seen in both 4.2.23 and 5.2.3 — local tests.
- **Drag-and-drop file handlers.** Python FileHandlers in 5.2.3 are `IO_FH_gltf2`, `IO_FH_svg_as_curves`, `SEQUENCER_FH_movie_strip`, `SEQUENCER_FH_image_strip`, `SEQUENCER_FH_sound_strip`, `VIEW3D_FH_camera_background_image` and `VIEW3D_FH_empty_image`. `IO_FH_usd` and `IO_FH_alembic` also worked as collection-exporter names — local test [bpy 5.2.3](https://pypi.org/project/bpy/5.2.3/).

### Inferences
- Target 4.5 LTS and 5.2 LTS for server-built .blend files. Build 4.2 users' files client-side, or save from a `bpy` 4.5 worker, noting that 4.5 uses Python 3.11 rather than the app's 3.13.
- Always pass absolute paths into `bpy`, then save with `relative_remap=True`. Ship a zip containing the .blend, `media/`, `manifest.json` and the extension.
- Convert SVG boards to PNG or WebP server-side. Use AVIF only for 5.x users.
- Store fps in the manifest and in scene properties. Set `render.fps` and `fps_base` **before** importing glTF, and conform 23.976 clips explicitly; for example, keep the scene at 24 / 1.001, or retime the clip if the app works at 24.
- After any glTF import, apply three fixups:
  1. set the render resolution from `aspectRatio`;
  2. set `sensor_fit='HORIZONTAL'` and lens = 18/(A·tan(yfov/2));
  3. assign the NLA lens action as active (in 5.x via `animation_data.action` plus `action_slot`).
- For USD returns, sample `Usd.Stage` directly on the server, which is exact, rather than relying on Blender's re-import. Users wanting editable camera keys from USD must bake the Transform Cache constraint.
- Keep generated scripts free of `frame_final_*`, `sequences` and `Action.fcurves`, or wrap them, to stay valid into 6.0.
- Run `bpy` inside a separate worker process, not inside the FastAPI event loop: the module has a large native state and import cost. The Blender docs on this ("Blender as a Python Module") were not fetched.

### Gaps
- Official guidance on `bpy` thread-safety and running one instance per process was not retrieved (docs blocked). The recommendation above is precautionary.
- Colour management of generated clips (sRGB vs Rec.709, `set_view_transform`, `colorspace_settings`) and audio (sound strips, A/V sync drift for 23.976) were not tested.
- Rendering (EEVEE/Workbench need a GPU or OpenGL; Cycles CPU should work headless) was not tested in the `bpy` build.
- The exact Debian/Ubuntu package names for the X11 and GL libraries `bpy` needs in a slim container were not verified; only the `ldd` soname list above is.
