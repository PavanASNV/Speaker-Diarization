# Previs and storyboard-to-3D in Blender, and Blender as the control layer for photoreal AI video (state as of 5 Nov 2026)

*How this was researched (a note, not a section).* About 20 web searches were run before the shared per-turn search budget ran out. After that, more searching was not possible. Direct page fetches failed for digitalproduction.com, mixar.app, invideo.io and higgsfield.ai (network policy). arxiv.org and fal.ai were blocked as the assignment expected. Because of this, many details below come only from search-engine summaries of the cited pages. They are marked **(search summary)**. Three sets of primary texts were read in full:
(1) the Ubisoft Shot Manager repo and its docs, via git clone (last commit 2023-02-27);
(2) the MCP for Blender repo (formerly `blender-mcp`), via git clone (last commit 2026-11-04, package version 2.2.7);
(3) Higgsfield's Blender connector "skill" texts, returned by the Higgsfield MCP connector attached to this session through `bl_get_skill` and read on 2026-11-05.
Item (3) is cited as **HF skill: name**. Those links point to the Higgsfield Blender product page, because the skill text has no public URL. It is the vendor's own guidance and was not independently verified.

## 1. How studios and indie filmmakers do previs in Blender (scene structure, cameras on markers, shot managers, proxies, lenses, Grease Pencil storyboards, editorial export)

### Takeaway
Two layouts dominate:
- **Previs/layout:** one continuous-action Blender scene with many cameras, where each shot is a camera plus a frame range. This is the SPA Sequencer and Ubisoft Shot Manager model, with cameras bound to timeline markers.
- **Production:** one .blend per shot with linked assets, built from a shot database.

Storypencil-style "scene strips in the VSE" connect 2D boards to 3D. Blender has no native shot entity, so every tool adds its own. The best-documented open-source one, Ubisoft Shot Manager with OTIO edit export, has been unmaintained since 2023. An app should therefore own a minimal shot schema and treat markers, the VSE and OTIO only as views of it that other tools can read.

### Cited Findings
**Scene structure: one action scene with many cameras, vs one scene or file per shot**
- The SPA Sequencer add-on states that "in most cases during Previz a Blend file will contain one Scene where all action occurs." Shots point at that scene and use their own cameras. **(search summary)** — [Blender Extensions: SPA Sequencer](https://extensions.blender.org/add-ons/spa-sequencer/)
- Ubisoft Shot Manager's philosophy:
  - Build a continuous action "with a plausible timing" in one scene, add several cameras, then define when each camera records and in what order.
  - A **Shot** is a camera plus a start and end time.
  - A **Take** is an ordered list of shots ("a take is an edit"), independent of scene time.
  - A dedicated "Shots Play Mode" plays shot to shot, including time jumps ("ellipses") and jumps back to replay the same action from another camera.
  — [Shot Manager docs: General philosophy](https://github.com/ubisoft/shotmanager/blob/main/docs/getting-started/general-philosophy.rst)
- The Shot Manager README calls this "very convenient for previz and 3D layout" and says the add-on "introduces a true shot entity in Blender scenes." That implies Blender itself lacks one. — [Shot Manager README](https://github.com/ubisoft/shotmanager)
- A newer community add-on thread is titled "Hyper Shot — Blender has no native concept of shot states, so I built one." Its date was not checked. — [Blender Artists: Hyper Shot](https://blenderartists.org/t/hyper-shot-blender-has-no-native-concept-of-shot-states-so-i-built-one/1641870)
- An older, undated BlenderWiki user page describes an open-movie-style workflow with exactly one .blend per shot. That file passes from layout to animation to lighting to compositing, with assets linked in. **(search summary)** — [BlenderWiki User:Eibriel](https://wiki.blender.org/index.php/User:Eibriel)
- Practitioners on Blender Artists (thread date not checked) describe two patterns. **(search summary)** — [Blender Artists: File organization for animated short film](https://blenderartists.org/t/file-organization-for-animated-short-film/1190776)
  - The Blender Kitsu add-on uses tracker data to build a dedicated shot file and link the required assets into it.
  - Another user does layout as one sequence in one .blend with camera markers, then splits it into one file per shot for animation.

**Shot entities, camera binding to markers, and shot-manager add-ons**
- Binding cameras to timeline markers is the lightweight standard approach. Several add-ons build on it:
  - Shot Master (lists Blender 4.0) offers "quick bind cameras to markers" plus camera and shot lists. — [Shot Master](https://blendermarket.com/products/shot-master)
  - The free Previz Camera Tools script also bound cameras to markers, added the cameras to the VSE and showed cut timings in the 3D viewport. It is old and had not been ported to 2.80 when posted. — [Blender Artists: Previz Camera Tools](https://blenderartists.org/t/previz-camera-tools/699752)
  - Timeline Marker Groups keeps camera assignments in named marker groups. — [Timeline Marker Groups](https://blendermarket.com/products/timeline-marker-groups)
  - All three **(search summary)**.
- Camera Configurator (Superhive, written for Blender 5.x) manages shots that each have their own camera, rig, frame range and focus settings. It shows timeline markers, tracks shot status and "automatically generates [SHOT]_START and [SHOT]_END markers for every shot." **(search summary)** — [Camera Configurator docs](https://superhivemarket.com/products/camera-configurator/docs)
- Shotlist (Blender 2.80+) lets you add shots and jump between them, and can lock all shots and markers to prevent accidental edits. **(search summary)** — [Shotlist docs](https://superhivemarket.com/products/shotlist-shots-manager/docs)
- Ubisoft Shot Manager status:
  - GPLv3. First built in 2020 on Blender 2.83 for an animated TV series and used on several Ubisoft projects.
  - Now "NOT UNDER DEVELOPMENT ANYMORE", maintained only on LTS (2.93 and 3.3 up to Dec 2022).
  - Last changelog entry is 2.1.46 (2023-02-27).
  — [README](https://github.com/ubisoft/shotmanager), [CHANGELOG](https://github.com/ubisoft/shotmanager/blob/main/CHANGELOG.md)
- Shot Manager feature list:
  - shots built from scene cameras, and sequences built from shots
  - non-linear real-time playback
  - global retiming of the action and the edit
  - global camera video and sound backgrounds
  - batch rendering (EEVEE, Cycles, playblast)
  - "Edit file import and export based on OpenTimelineIO"
  - Grease Pencil sketching per shot (experimental)
  - "Edit forth and back with VSE" (experimental)

  Its stated limitation: it is not compatible with native camera binding, because its play mode replaces it. It ships tools to convert bound markers into shots. — [Shot Manager docs: Features and limitations](https://github.com/ubisoft/shotmanager/blob/main/docs/main-panel/features.rst)
- Production use of Shot Manager:
  - A "Project" settings object holds naming conventions, output resolution and file formats. A production tool can set it when a scene opens.
  - There is a stable API for pipeline integration.
  - The full shot name is `[Proj Id][sep][Seq Id][sep][Shot Id]`.
  — [Use in production](https://github.com/ubisoft/shotmanager/blob/main/docs/how-to/use-in-production.rst), [Entity namings](https://github.com/ubisoft/shotmanager/blob/main/docs/getting-started/entity-namings.rst)

**Grease Pencil storyboarding**
- Storypencil is a bundled add-on documented in the Blender manual (manual pages exist through 4.1; docs added Sept 2022). Its authors are Antonio Vazquez, Matias Mendiola, Daniel Martinez Lara and Rodrigo Blaas. It storyboards in the VSE, with each shot as a scene used as a Scene Strip. **(search summary)** — [Blender manual: Storypencil](https://www.blender.org/manual/en/addons/sequencer/storypencil.html), [Manual 4.1](https://docs.blender.org/manual/es/4.1/addons/sequencer/storypencil.html), [Add-on proposal #100665](https://projects.blender.org/blender/blender-addons/issues/100665), [docboard commit, Sept 2022](https://lists.blender.org/pipermail/bf-docboard-svn/2022-September/006165.html)
  - Setup starts from the 2D Animation template.
  - "New Window" mode opens the source scene in a new window with timelines kept in sync.
  - "Switching" mode jumps to a dedicated workspace and temporarily copies sequencer audio into the source scene.
  - Movie output renders one clip per Scene strip. Image output renders images on the keyframes of each strip.
  - A "base scene" lets new shots reuse cameras, Grease Pencil objects and backgrounds.
  - Known issue: sync can fail when a scene is pinned to a workspace.
- Shot Manager 2.0 (released Aug 2022) added in-Blender storyboarding. A shot "Type" separates Storyboard Shots (pure 2D) from Camera Shots. Separate "layouts" serve the storyboard and previz contexts. Storyboard frames can be detached into free 3D objects, and there is a continuous drawing mode and a frame grid. — [What's new in 2.0](https://github.com/ubisoft/shotmanager/blob/main/docs/what-s-new/what-s-new-in-2-0.rst); release timing from the maintainer **(search summary)** — [Shot Manager wiki](https://github.com/ubisoft/shotmanager/wiki)

**Blocking proxies and mannequins**
- Higgsfield's greybox procedure builds "role-coloured semantic proxies" with hero, support, background and occluder roles. It animates "deterministic key poses and camera motion, with readable holds and no random per-object movement." Proxies are built or updated idempotently, and existing assets in a live user scene are reused rather than reset. — [HF skill: blender-greybox](https://higgsfield.ai/plugins/blender)
- Every generated asset keeps a measured proxy footprint "so generation cannot derail the layout." The proxy "survives until the generated asset passes scale, silhouette, material, and camera-read audits." Fidelity tiers are fixed up front: hero = detailed, midground = stylized, background = blockout. — [HF skill: blender-scene-spec](https://higgsfield.ai/plugins/blender), [HF skill: blender-generation](https://higgsfield.ai/plugins/blender)
- Higgsfield's connector imports generated motion as "a 27-bone armature with Mixamo-style bone names plus a skinned body." It goes into its own collection and the scene is set to the clip's 20 fps. — [HF connector tool description `bl_import_motion` (this session)](https://higgsfield.ai/plugins/blender)
- Higgsfield's model catalog lists `sam_3_3d_body`: "One human image → body shape/pose GLB and optional keypoints/parameters." — [HF skill: blender-volatile](https://higgsfield.ai/plugins/blender)
- MCP for Blender notes two Meshy behaviours. Models arrive "sized to real-world height with the origin at the bottom." With your own key, asking for a character "in an A-pose" or "in a T-pose" gives a rig-ready pose. — [MCP for Blender README](https://github.com/ahujasid/blender-mcp/blob/main/README.md)
- Storyboarder Shot Generator's character models are open source. They ship as Blender files plus FBX, glTF and OBJ exports. **(search summary)** — [wonderunit/shot-generator-models](https://github.com/wonderunit/shot-generator-models)

**Lens and camera practice**
- From Higgsfield's camera gate:
  - The camera locks before lighting, with one explicit active camera.
  - Subject target, lens, distance, height and framing come from the scene spec.
  - "Wide lenses exaggerate depth, long ones flatten it; extreme focal length is not a patch for wrong geometry."
  - Preserve headroom, contact shadows and silhouette separation.
  - Camera motion is "built on semantic parent rigs (orbit, dolly, handheld), not an opaque stack of keyed transforms."
  - Freeze the camera before lighting, because lighting is camera-relative.
  - "Swapping a hero asset invalidates every camera set against its old shell." Placement and near clipping must be re-checked per camera.
  — [HF skill: blender-lighting-camera](https://higgsfield.ai/plugins/blender)
- From Higgsfield's shot rules:
  - One motion clip is 2–30 s. Longer sequences split into separate shots ("a two-minute four-camera sequence is four shot blocks, one clip per camera").
  - "One primary camera move per clip — never hide several unrelated camera cuts inside one clip," with at most one subtle secondary move.
  - Each shot records duration/fps, active camera, object roles, subject start and end transforms, beats, continuity anchors to neighbouring shots, and acceptance criteria (composition, path, timing, silhouette).
  — [HF skill: blender-greybox](https://higgsfield.ai/plugins/blender)

**Editorial export (VSE, EDL/OTIO)**
- Shot Manager's EDL import/export was removed in 2021 to drop the OpenTimelineIO dependency. It came back in 2.0 (Aug 2022) with OTIO wheels bundled inside the add-on. **(search summary)** — [Shot Manager wiki](https://github.com/ubisoft/shotmanager/wiki)
- OTIO's list of tools includes Shot Manager and a separate "Blender VSE IO" add-on for OTIO import/export in the VSE. **(search summary)** — [OTIO wiki: Tools and Projects Using OpenTimelineIO](https://github.com/PixarAnimationStudios/OpenTimelineIO/wiki/Tools-and-Projects-Using-OpenTimelineIO), [OTIO apps and tools](https://otio-website.vercel.app/apps-and-tools)
- Blender Studio has a post titled "OpenTimelineIO in Blender." Only the title was seen; the post was not read. — [Blender Studio blog](https://studio.blender.org/blog/opentimelineio-in-blender)

### Inferences
- **Own the shot entity.** Blender offers markers and camera binding but no shot object, and every mature tool adds one. The app should store its own shot records inside the .blend, keyed by the app's `shot_id`, either as a scene-level custom property group or a JSON text datablock. It should then mirror them into views that need no add-on:
  - timeline markers named by `shot_id` and bound to each shot camera, so plain Blender plays the cut;
  - optionally, an "EDIT" scene of VSE scene strips for order and duration, following the Storypencil pattern.

  Do not depend on Ubisoft Shot Manager. It has been frozen since Feb 2023 on 2.93/3.3 LTS, while Higgsfield already requires Blender 5.1+.
- **Pick a structure per app scene.** There are two options:
  - **Pattern A, continuous action** (SPA/Shot Manager): one Blender scene per app scene. Shots are cameras over frame ranges that do not overlap. This suits coverage of one action, such as dialogue from several angles. A blocking edit then rightly affects every shot whose range sees it.
  - **Pattern B, isolated shots** (Storypencil scene strips or Kitsu's per-shot files): one Blender scene per shot. Set and character data are shared by linking or instancing from a base scene.

  Claude plans each shot as an independent 2D tableau, and the requirement is that only changed shots regenerate. So Pattern B is the safer default and Pattern A should be offered as an option for coverage sequences. Either way, change detection must evaluate each shot, not each object (see section 4).
- **Proxy conventions.** Copy what Higgsfield and MCP for Blender do: role-coloured proxies with semantic names (`HERO_`, `SUP_`, `BG_`, `OCC_`, `SET_<label>`), real metre scale, origin at the feet and a rig-ready A/T pose. Mixamo-style bone names let generated motion such as Higgsfield's 27-bone imports retarget onto the proxies. The app's "set elements as labelled boxes" map directly to `SET_<label>` proxies with the label also stored in a custom property.
- **Camera conventions.** Use one camera per shot named `CAM_<shotId>`, parented to a semantic rig (dolly, orbit, crane, handheld) so the move type can be read back. Write focal length, sensor width and fit, shift and DOF explicitly rather than relying on defaults, so a lens word in a prompt (e.g. "35mm") matches the Blender camera. On re-import, check Higgsfield's rules (2–30 s; one primary move) and warn the user when an edit breaks them or exceeds a target model's maximum duration.
- **Show the pencil panel in the camera.** Put each shot's pencil panel in as that camera's background image. Shot Manager's global camera backgrounds are the precedent. The artist then blocks against Claude's plan, and Storypencil or Grease Pencil can be used to redraw.
- **Editorial export.** Export the app's ordered shot list as OTIO (EDL as a fallback), with clip names equal to shot ids. That is the interchange format Shot Manager and Blender VSE IO already use.

### Gaps
- These topics were not researched because the search budget ran out:
  - MPFB/MakeHuman and Rigify metarig blocking practice
  - Blender's default camera sensor and sensor-fit conventions, and Super 35 versus full-frame presets in Blender 5.x
  - whether Storypencil is still bundled in Blender 4.2+ or only on the Extensions platform
  - Grease Pencil v3 (4.3+) effects on storyboard add-ons
  - native VSE EDL import/export in 5.x
- Blender Studio's current pipeline documentation (Shot Builder/Kitsu) was not read.
- No studio articles on film or TV previs in Blender were found (for example by previs vendors).

## 2. Existing storyboard/script-to-3D previs tools and AI scene builders that work with Blender: formats, and what to copy

### Takeaway
The closest analogue to Prompt-to-Scene Creator is Higgsfield's Blender add-on and MCP bridge (public since about Aug–Sep 2026). Its pitch is "prompt the blockout, describe the camera move, adjust by hand, reblock." Its published skill texts expose a JSON scene manifest whose ids persist as custom properties, so re-running it updates rather than duplicates. Its per-shot deliverable is a greybox MP4 plus the exact first-frame PNG. MCP for Blender (the most widely used open MCP server for Blender) contributes a closed visual check loop and provenance stamping. Academic systems (SceneCraft, 3D-GPT, BlenderAlchemy, Holodeck, LayoutGPT, Holodeck 2.0, SceneAssistant) contribute the plan-to-Blender-Python and render-critique loop. Storyboarder's Shot Generator works with Blender through GLB/glTF assets. For ShotPro, Previs Pro, FrameForge, Cine Tracer, Cascadeur and Plask, no export information was found in this session.

### Cited Findings
**Storyboarder Shot Generator (Wonder Unit)**
- Custom models are imported as GLB/glTF. The maintainers recommend Blender for modelling and export and do not support importing OBJ, COLLADA or FBX. Their wiki advises keeping the exported scene simple and applying modifiers on export. **(search summary)** — [Creating custom 3D Models for Shot Generator](https://github.com/wonderunit/storyboarder/wiki/Creating-custom-3D-Models-for-Shot-Generator), [Exporting Objects from Sketchup](https://github.com/wonderunit/storyboarder/wiki/Exporting-Objects-from-Sketchup)
- Character models are open source (Blender files plus FBX/glTF/OBJ). **(search summary)** — [shot-generator-models](https://github.com/wonderunit/shot-generator-models)
- glTF output from Shot Generator was requested in issue #1807. A user reported in issue #1940 that GLB export failed silently; its status is unknown. **(search summary)** — [#1807](https://github.com/wonderunit/storyboarder/issues/1807), [#1940](https://github.com/wonderunit/storyboarder/issues/1940)

**Higgsfield for Blender (2026)**
- Launch and pitch:
  - Launch post on X: "Prompt the scene and build the blockout > Describe the camera move and get it animated > Adjust anything by hand > Reblock the whole shot in seconds. Now available via Higgsfield MCP or Supercomputer." The post ID decodes to late Aug 2026. — [Higgsfield on X](https://x.com/higgsfield/status/2092255768770920506)
  - Trade coverage is dated 2026-09-23 (title only; fetch blocked). — [Digital Production](https://digitalproduction.com/2026/09/23/higgsfield-brings-ai-generation-into-blender-oh-goodey/)
  - A third-party blog headline reads "AI Blockout in Seconds (Aug 2026)." — [explainx](https://explainx.ai/blog/higgsfield-blender-mcp-ai-blockout-august-2026)
  - A related product is titled "3D Jutsu: Build and Animate 3D Scenes from a Prompt" (title only). — [Higgsfield blog](https://higgsfield.ai/blog/higgsfield-3d-jutsu)
  - One article describes "camera control with your phone" (title only). — [Cachephoto](https://www.cachephoto.com/en/cineblog/higgsfield-blender-plugin-camara-telefono/)
- Requirements and setup **(search summary)** — [Higgsfield Blender page](https://higgsfield.ai/plugins/blender), [Higgsfield help center: external integrations](https://higgsfield.ai/creator-hub/help-center/integrations/external-integrations-higgsfield), [erikataranto.com](https://erikataranto.com/en/blog/higgsfield-blender)
  - Blender 5.1+ on Windows and macOS; inference runs in the cloud and needs a Higgsfield account.
  - The MCP bridge is added as a custom connector named "Higgsfield Bridge" at `bridge.higgsfield.ai/mcp`.
  - "Scene Builder" assembles blockouts in an open .blend file.
  - The 3D Model section places generated meshes at the 3D cursor.
- Conflicting reports about Higgsfield:
  - **Credits:** one guide says blocking by hand uses no credits; another says everything draws on plan credits. **(search summary)** — [erikataranto.com](https://erikataranto.com/en/blog/higgsfield-blender), [Higgsfield help center](https://higgsfield.ai/creator-hub/help-center/integrations/external-integrations-higgsfield)
  - **Agents:** official pages name Claude for the bridge, while a user report says Codex also works. **(search summary)** — [developersdigest](https://www.developersdigest.tech/blog/gpt-6-blender-interactive-3d)
- The connector spans several hosts: After Effects (`ae_*`), Premiere Pro (`pr_*`), Blender (`bl_*`), Cinema 4D (`c4d_*`) and a "browser-based 3D Blocking Studio" (`bs_*`). — [HF connector server instructions (this session)](https://higgsfield.ai/creator-hub/help-center/integrations/external-integrations-higgsfield)
- **Scene Manifest.** `bl_build_scene` and `bl_build_blockout` take one JSON object with these fields: `name`, `construction_mode`, `motion_mode`, `frame_start`/`frame_end`, `collections[]`, `materials[]`, `objects[]` (with `id`, `name`, `kind`, `role`, `collection`, `dimensions`, `location`, `material`), `camera` (with `frame_objects`), `lights[]` and `render` (engine, resolution, fps).
  - "Manifest ids persist as custom properties, so re-running the same manifest updates the managed datablocks instead of duplicating them."
  - Broad builds run `dry_run: true` first.
  - Names are "semantic and unique — `ENV_floor`, `PRP_table`, `HERO_robot`, `CAM_main`, `LGT_key` … describing role, never creation order."
  — [HF skill: blender-scene-spec](https://higgsfield.ai/plugins/blender)
- **Scene Passport.** A pre-build spec. Each asset records a route [EXISTING|BLOCK|GEN], fidelity, dimensions in metres, location, orientation in radians, origin, parent and role. It also has SHOT (active camera, framing/lens/target, depth layers), LOOK, LIGHTING (key direction relative to camera, motivation, contrast), MOTION (beats, holds) and ACCEPTANCE sections. — [HF skill: blender-scene-spec](https://higgsfield.ai/plugins/blender)
- **Per-shot motion deliverable.** `bl_render_motion_reference` writes a "native H.264 MP4 plus its exact first-frame PNG, default 1280x720, 2–30 s," and restores the previous render settings. Before passing, the start, middle and end frames are viewed to check camera axis, collisions, ground contact, screen direction, beat timing and subject scale. — [HF skill: blender-greybox](https://higgsfield.ai/plugins/blender)
- **Generation rules.**
  - The live model catalog and its JSON Schema are the only source of truth for parameters.
  - Estimate without spending, then submit with a stable `idempotency_key`. Reuse the key after an uncertain timeout rather than resubmitting.
  - Record job id, model id, parameters, result path and "which Blender object/material/World/shot consumed it."
  - `gpt_image_2` is the route for "composition-preserving restyle."
  — [HF skill: blender-generation](https://higgsfield.ai/plugins/blender), [HF skill: blender-volatile](https://higgsfield.ai/plugins/blender)
- **Tools in this session's connector build.** The connector attached here exposes `bl_render` (PNG; EEVEE Next, Cycles or Workbench), `bl_screenshot`, `bl_execute` (arbitrary bpy), `bl_generate_image`, `bl_generate_3d` (default `tripo_3d`), `bl_image_to_3d`, `bl_import_motion`, `bl_add_camera`, `bl_set_active_camera`, `bl_set_frame`, `bl_set_transform` and others. It does not expose the `bl_generate_video` or `bl_render_motion_reference` tools its skills mention. The skills themselves warn that "the typed tool set varies by connector build." — [HF connector tool list (this session)](https://higgsfield.ai/plugins/blender), [HF skill: blender-volatile](https://higgsfield.ai/plugins/blender)

**MCP for Blender (`ahujasid/blender-mcp`, renamed; PyPI `mcp-for-blender`)**
- MIT licence, version 2.2.7, last commit 2026-11-04. The AI writes Blender Python for modelling, layout, materials, animation and rigging. The MCP adds nine tools:
  - `execute_blender_code`
  - `look`, which views the `viewport`, `camera`, `angles` (an auto-framed multi-view sheet) or `frames` (an animation strip) in solid, material, rendered, wireframe or xray shading
  - `get_scene_info`, a compact summary with one line per object
  - `generate_3d` (Tripo, Hunyuan3D, Hyper3D Rodin, Meshy)
  - `search_assets` and `import_asset` (Poly Haven, Sketchfab, Poly Pizza)
  - `get_addon_status`
  - `disable_telemetry` and `record_trajectory_feedback`, for data collection
  — [README](https://github.com/ahujasid/blender-mcp/blob/main/README.md), [pyproject](https://github.com/ahujasid/blender-mcp/blob/main/pyproject.toml)
- Protocol: JSON over TCP. A command has `type` and `params`; a response has `status` plus `result` or `message`. The default address is localhost:9876. Remote mode "has no authentication or encryption and lets clients run Python in Blender." `execute_blender_code` carries a warning to "ALWAYS save your work before using it." — [README](https://github.com/ahujasid/blender-mcp/blob/main/README.md), [addon.py](https://github.com/ahujasid/blender-mcp/blob/main/addon.py)
- Provenance stamping: on import it writes `polyhaven_id`, `polyhaven_url`, `polyhaven_authors`, `polyhaven_resolution`, `polyhaven_licence` and `polypizza_attribution` onto objects, materials, images and worlds as custom properties, so the information "survives the session" inside the .blend. — [README](https://github.com/ahujasid/blender-mcp/blob/main/README.md)
- Its code identifies texture maps "by the custom property stamped at download time rather than by parsing the image's name." The old name parser had mangled map names (e.g. "nor_gl" became "gl"). — [addon.py](https://github.com/ahujasid/blender-mcp/blob/main/addon.py)
- One directory snapshot reported about 32.4k GitHub stars, with a caveat that this says nothing about code quality. **(search summary)** — [ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fahujasid%2Fblender-mcp)

**Research systems that generate or edit Blender scenes (mostly 2023–2024, plus 2025–26 follow-ups)**
- SceneCraft (ICML 2024, arXiv 2403.01248) turns text into Blender Python. It plans with a spatial scene graph ("blueprint"), refines iteratively, and handles scenes of up to about 100 assets. **(search summary)** — [Emergent Mind: SceneCraft](https://www.emergentmind.com/papers/2403.01248)
- 3D-GPT and SceneCraft both write Python that drives procedural frameworks (Infinigen, Blender). Holodeck builds indoor environments over several LLM rounds: floors and walls, doors and windows, object selection and placement, using retrieved assets. LayoutGPT prompts an LLM for layout parameters. **(search summary)** — [3D Scene Generation: A Survey (arXiv 2505.05474)](https://arxiv.org/pdf/2505.05474)
- BlenderAlchemy (ECCV 2024) is a vision-language-model editor. It iteratively refines Blender Python edit programs from user intent; it edits scenes rather than generating them. LayoutGPT "often neglects physical plausibility." **(search summary)** — [Holodeck 2.0 (arXiv 2508.05899, Aug 2025)](https://arxiv.org/html/2508.05899)
- Holodeck 2.0 (Aug 2025) generates worlds with vision-language guidance and supports editing, using GPT-o3 for spatial constraints. In the authors' own 450-person study its edits scored 9.06 on average, against 4.29 for BlenderAlchemy and 4.87 for BlenderGym. These are self-reported. **(search summary)** — [Holodeck 2.0](https://arxiv.org/pdf/2508.05899)
- SceneAssistant (arXiv 2603.12238, Mar 2026) is a visual-feedback agent. It notes that most prior methods "operate in an open-loop manner": generated layouts or code run without refinement based on rendered results. **(search summary)** — [SceneAssistant](https://arxiv.org/pdf/2603.12238)

**Commercial "blockout → AI video" services (secondary sources only)**
- Mixar renders beauty, clay and depth guides across a shot and hands them over as references. **(search summary; fetch blocked)** — [Mixar blog](https://www.mixar.app/blog/blender-blockout-to-video)
- fal's "H3 Max" is presented as a video-to-video model that turns a Blender render or any 3D animation into photoreal video. **(search summary; fal.ai blocked)** — [fal H3 Max](https://fal.ai/h3-max-3d-to-video)

### Inferences
- **Copy Higgsfield's manifest pattern for export.** Generate the .blend from the app's shot plan with a manifest whose stable ids are written to custom properties. Re-running it must be idempotent, so re-export updates in place. Use semantic role names, keep a proxy footprint for every asset, and run a dry run before broad changes. Use the per-shot "MP4 + exact first-frame PNG" as the standard artifact handed to video models.
- **Copy the closed visual check loop** from MCP for Blender (`look` with camera and multi-angle sheets) and SceneAssistant. After building each shot from Claude's plan, render the camera view and have Claude compare it with the pencil panel. Fix it before the user opens Blender. This removes the open-loop weakness that SceneAssistant identifies.
- **Copy provenance stamping** from MCP for Blender. Stamp `p2s_*` ids on every managed datablock and match on them, never on names.
- **Formats.** Use the .blend as the editable master. Use GLB/glTF for assets, which is all Storyboarder accepts and is what Higgsfield and MCP for Blender generators return. Use OTIO for the edit, a JSON manifest (embedded text datablock plus sidecar file) for meaning, and MP4 + PNG per shot for model inputs.
- **Where to differ.** Higgsfield's Blender loop ("adjust by hand, reblock") is essentially the same loop as this app. The documented gaps the app can fill are per-shot change detection, regenerating only dirty shots, and learning from user edits. Nothing found shows Higgsfield doing any of these.

### Gaps
- No export, import or Blender-interop details were collected for ShotPro, Previs Pro, FrameForge, Cine Tracer, Cascadeur or Plask. The search budget ran out, and the one combined query returned nothing on ShotPro, Previs Pro or FrameForge.
- BlenderGPT's 2026 status is unknown.
- Higgsfield's own product and help pages could not be fetched. It is unconfirmed whether its Blender add-on sends depth or other control passes to video models; the skill texts show only MP4 motion references plus first frames.
- Whether SceneCraft, 3D-GPT and BlenderAlchemy code is maintained in 2026 is unknown.

## 3. Blender as the control layer for AI video: passes, model inputs, which give the best control today, and keeping output photoreal

### Takeaway
Dense, per-frame structural control from Blender passes (depth, pose, canny or line) is only documented for open models: Wan VACE and the LTX-2 IC-LoRA adapters, usually through ComfyUI. Hosted models take weaker kinds of control:
- **Seedance 2.0:** reference videos and images, with roles assigned by @-tags.
- **Veo 3.1:** first and last frame plus up to three reference images.
- **Kling 3.0 Motion Control:** an appearance image plus a motion video, with the camera steered only by the prompt.
- **Runway Aleph:** video-to-video restyle.

The pattern that best keeps a photoreal look is:
1. Treat the grey blocking only as structure (depth, lines, skeleton, camera path), never as an appearance reference.
2. Turn the Blender camera view into a photoreal keyframe with an image model that preserves composition.
3. Animate with image-to-video. Add the greybox or depth video as a reference or control signal, at moderate strength, only where the camera move matters.

### Cited Findings
**Open models with dense control**
- LTX-2 official IC-LoRA control adapters:
  - Names: `LTX-2-19b-IC-LoRA-Depth-Control`, `-Pose-Control` and `-Canny-Control`.
  - Depth works from depth-map sequences and suits "architectural walkthroughs with camera movement following depth."
  - Pose takes OpenPose or MediaPipe keypoints, for character animation and motion retargeting.
  - Canny takes white-on-black edge maps, for line art and strict composition.
  - Depthcrafter and Blender are listed as depth-map sources.
  - Prompts should describe style, not motion, because motion comes from the reference video.
  - For approximate structure with more creative freedom, one guide suggests depth `attention_strength` around 0.6.
  - Run one mode at a time to avoid VRAM problems.
  **(search summary)** — [LTX docs: IC-LoRA](https://docs.ltx.video/open-source-model/usage-guides/ic-lo-ra), [LTX blog: How to use IC-LoRA in LTX-2](https://ltx.io/model/model-blog/how-to-use-ic-lora-in-ltx-2)
- Community LTX-2.3 (22B) depth, pose and canny IC-LoRAs exist under Apache 2.0, for example a depth model that conditions generation on a monocular depth video. **(search summary)** — [Hugging Face: ltx2.3 iclora depth](https://huggingface.co/SyFeee/ltx2.3-chinese-drama-iclora-depth)
- Wan VACE (2.1/2.2) evidence. There is no agreed default strength; the advice is to start from the template and sweep values. **(search summary)**
  - Wan VACE accepts depth and pose control videos. — [RunComfy: Wan 2.2 VACE pose workflow](https://www.runcomfy.com/comfyui-workflows/wan-2-2-vace-in-comfyui-pose-driven-motion-video-workflow?action=run-workflow)
  - The ComfyUI-WanVaceAdvanced extension adds per-frame strength scheduling and a `VaceStrengthTester` node. — [RunComfy: ComfyUI-WanVaceAdvanced](https://www.runcomfy.com/comfyui-nodes/ComfyUI-WanVaceAdvanced)
  - One workflow page says depth control is more stable than pose control. — [ComfyUI.org: Wan depth control](https://www.comfyui.org/en/unlock-advanced-video-depth-control-wan-model)
  - Depth and pose can be chained as separate encoders, each with its own strength; this architecture was not verified. — [Promptus: Wan 2.1 VACE motion control](https://www.promptus.ai/blog/comfyui-wan-21-vace-video-motion-control)
  - A forum comment says the 0.2 "strength" in `WanVaceToVideo` applies to the reference image, not to the control signal; this is low reliability. — [r/comfyui via mirror](https://redlib.ssps.io/r/comfyui/comments/1l7ajzc/why_is_the_reference_image_being_completely/mwv6jek/?context=3)
- A ComfyUI-based Blender add-on/workflow ("Blender to ComfyUI AI Renderer 2.0") takes structure from depth, outline or pose passes to drive Wan VACE video generation. A page in that result set also says aspect ratio and fps must match between Blender and the workflow. **(search summary)** — [RunComfy: Blender to ComfyUI AI Renderer 2.0](https://www.runcomfy.com/comfyui-workflows/blender-to-comfyui-ai-renderer-2-0-workflow-cinematic-video-output)
- NVIDIA's RTX AI video guide describes a graph that makes a depth map from the Blender scene and combines it with a text prompt. The result is a photoreal image matching the layout and perspective (FLUX.1 Depth-class, with NVFP4 acceleration). That image is the photoreal keyframe step. **(search summary)** — [NVIDIA: RTX AI Video Generation Guide](https://www.nvidia.com/en-us/geforce/news/rtx-ai-video-generation-guide/)
- One write-up calls depth an "invisible layout guide" for AI video. **(search summary)** — [MindStudio: depth maps as storyboards](https://www.mindstudio.ai/blog/depth-maps-as-storyboards-ai-video-generation)

**Hosted models (the app's targets)**
- **Seedance 2.0 reference-to-video.** Mostly third-party guides **(search summary)**; ByteDance documentation was not found. — [WaveSpeed guide](https://wavespeed.ai/blog/posts/seedance-2-0-complete-guide-multimodal-video-creation/), [Morphic guide](https://morphic.com/resources/how-to/seedance-2-guide), [BudgetPixel](https://budgetpixel.com/blog/seedance-20-the-reference-first-video-model-that-finally-feels-like-directing), [Technology.org user report, 2026-02-23](https://www.technology.org/2026/02/23/how-i-used-reference-videos-to-replicate-complex-camera-movements-in-seedance-2-0/)
  - It takes up to 9 reference images and up to 3 reference videos (about 15 s combined; one guide says 2–15 s total), plus audio and text.
  - Guides suggest assigning roles explicitly, e.g. `@Image 1` for identity, `@Video 1` for camera movement and `@Audio 1` for rhythm. Unnamed references "may be ignored or misinterpreted."
  - Guides recommend a reference video for camera work. A user report says a circling push-in reference clip was replicated in the output.
- Seedance 2.0 technical report (arXiv 2604.14148, Apr 2026): in its evaluation, advanced camera movement was the hardest sub-category, and no model scored above 3.14 on any metric. This is a snippet only; arXiv could not be fetched. — [Seedance 2.0 report](https://arxiv.org/pdf/2604.14148)
- Conflict: Wikipedia claims a Seedance 2.5 release in July 2026 (30 s native video, 50 multimodal references). Nothing else corroborated it, and the page looked unreliable. — [Wikipedia: Seedance 2.0](https://en.wikipedia.org/wiki/Seedance_2.0)
- **Kling 3.0 Motion Control** (official API) — [Kling API: Motion Control 3.0](https://kling.ai/document-api/api/video/motion-control)
  - Endpoint path `motion-control/kling-3.0`. Inputs: prompt, appearance image, motion video and an Element.
  - The reference video may be at most 10 s when orientation comes from the image, and at most 30 s when it comes from the video.
  - Elements are referenced by `element_id`. With an Element, the output follows the character orientation of the reference video.
  - There is no dedicated camera parameter; camera intent goes in the prompt.

  **(search summary)**
- Kling gateway notes **(search summary)** — [Evolink](https://evolink.ai/kling-v3-motion-control), [Replicate](https://replicate.com/kwaivgi/kling-v3-motion-control), [GPTProto guide](https://gptproto1.substack.com/p/how-to-use-kling-30-motion-control), [Picassoia pricing blog](https://blog.picassoia.com/kling-3-0-motion-control-api-pricing-and-free-alternatives)
  - Reference videos should "contain clear human motion without cuts or camera movement."
  - Only video-created Elements are supported (`reference_type=video_refer`).
  - Model ids differ by provider: `kling-v3-motion-control` and `kwaivgi/kling-v3-motion-control`.
  - One blog puts the API price at about $0.12–0.18 per second.
- **Veo 3.1** — [Google AI Developers Forum thread](https://discuss.ai.google.dev/t/how-does-veo3-1-achieve-the-using-first-and-last-frames-function-through-the-api/112947), [official docs pointer](https://ai.google.dev/gemini-api/docs/video), [dev.to guide](https://dev.to/_37bbf0c253c0b3edec531e/how-to-use-veo-31-api-4f3h), [CometAPI guide](https://www.cometapi.com/how-to-use-veo-3-1-api/), [Evolink](https://evolink.ai/de/veo-3-1), [Reddit snapshot, 2026-02-21](https://reddit.sentinel-team.org/posts/1qzmla5/snapshots/2026-02-21T04%3A30%3A30.805217Z)
  - Third-party pages say it supports first- and last-frame image-to-video (1–2 images) and a reference mode with up to 3 images.
  - The Google forum points to the Gemini API video docs' "multiple-images" section, and one reply says first/last frame is available through Vertex AI.
  - Conflicts: one guide puts `last_frame` in config, while a community REST guide says `lastFrame` must be in `instances[0]`, using `bytesBase64Encoded` and `veo-3.1-generate-preview`. Whether Veo 3.1 Fast accepts reference images is disputed: a community table says no, a commenter says Flow allows up to three.

  **(search summary)**
- **Runway Aleph** — [Layer: Gen-4 Aleph](https://www.layer.ai/docs/models/runway-gen4-aleph), [Layer: Aleph 2](https://layer.ai/docs/models/runway-aleph2), [Scenario help: Aleph essentials](https://help.scenario.com/articles/6499445829-runway-aleph-the-essentials)
  - Gen-4 Aleph is a video-to-video model that "restyles and transforms existing footage based on a text prompt."
  - The original Aleph takes up to 3 reference images.
  - "Aleph 2" is described as applying new styles and editing motion "while keeping the source video's structure."
  - No source confirmed a documented workflow for 3D or Blender render input.

  **(search summary)**
- A changelog aggregator lists Runway "Gen-4.5" on 2025-12-01 (title only). — [ToolColumn: Runway changelog](https://www.toolcolumn.com/tools/runway/changelog)

**Practitioner guidance on blockout renders**
- Invideo warns that "a grey render is not automatically a technical depth pass." It should be described as a camera or composition reference unless a depth map was actually exported and the route accepts it. **(search summary; fetch blocked)** — [Invideo FAQ: Blender blocking render as camera reference](https://invideo.io/faq/how-do-you-use-a-blender-blocking-render-as-a-camera/)
- Mixar hands beauty, clay and depth guides across the shot to video models as references. **(search summary)** — [Mixar](https://www.mixar.app/blog/blender-blockout-to-video)
- Higgsfield's greybox deliverable is a per-shot H.264 MP4 plus its exact first-frame PNG. `gpt_image_2` is its "composition-preserving restyle" route. — [HF skill: blender-greybox](https://higgsfield.ai/plugins/blender), [HF skill: blender-volatile](https://higgsfield.ai/plugins/blender)
- Blender 5.x scripting traps for writing passes and video:
  - The compositor tree moved from `Scene.node_tree` to `Scene.compositing_node_group`, and `CompositorNodeComposite` no longer exists.
  - A compositing group without `CompositorNodeRLayers` makes Blender skip rendering and write a transparent black frame ("a suspiciously fast render is this bug").
  - Blender 5.1 background sessions can omit `FFMPEG` as an output format.
  - The EEVEE enum may read `BLENDER_EEVEE` in some scenes and `BLENDER_EEVEE_NEXT` in others.
  — [HF skill: blender-volatile](https://higgsfield.ai/plugins/blender)

### Inferences
- **Control ladder, strongest first (Nov 2026).**
  1. Dense per-frame control in open models: Wan VACE or LTX-2 IC-LoRA with Blender depth, pose or canny. These give the most predictable layout and camera following.
  2. Reference-video imitation: a Seedance 2.0 `@Video` reference, or Runway Aleph video-to-video restyle.
  3. Keyframe anchoring: Veo 3.1 first/last frame plus references, or image-to-video on any model.
  4. Text-only camera description: Kling Motion Control's camera.

  For guaranteed camera moves from edited cameras, the app needs either an open-model path (self-hosted, or a host that exposes VACE or IC-LoRA) or reference-video imitation in Seedance 2.0. Hosted models never received 3D data directly in anything found.
- **Recommended per-shot recipe: structure from 3D, look from references, keyframe first.**
  1. From each edited shot camera, render conditioning at the first frame (and last frame if the model supports it): depth, line art or canny, skeleton, and a clay preview.
  2. Turn the frame into a photoreal keyframe with a composition-preserving image model. Options are a depth or line control image model (FLUX Depth-class, as in NVIDIA's guide) or an editing model given the clay render plus a strict "keep composition" instruction (Higgsfield's `gpt_image_2` route). Supply character and set identity as reference images, not the grey proxies.
  3. Generate video from the photoreal keyframe(s), choosing by model:
     - Seedance 2.0: keyframe and identity images as `@Image`, plus the greybox or depth MP4 as `@Video1`, stated in the prompt as "camera movement and blocking only."
     - Veo 3.1: first and last frame, with the camera move written in the prompt.
     - Wan 2.2: VACE depth control video at moderate or decaying strength, plus the first-frame reference.
     - Kling 3.0: image-to-video. For performance, Motion Control with a mannequin performance rendered from a locked-off camera.
  4. When motion comes from a control video, prompts describe look and lighting, not motion (the LTX guidance).
- **Photoreal guardrails.**
  - Never pass the clay or grey beauty render as an appearance or style reference, and label reference roles explicitly.
  - Prefer depth over pose for small or occluded figures, since depth is reported as more stable.
  - Keep control strength moderate, around 0.6 as LTX's suggested start for approximate structure, or decay per-frame strength in VACE so later frames gain realistic detail.
  - Match aspect ratio and fps between Blender and the model.
  - Lock the camera in performance-reference renders for Kling.
  - Keep exposure and lighting direction in the photoreal keyframe consistent with the Blender key light. Higgsfield's camera-relative lighting rules suggest the key-light azimuth should be written into the prompt.
- **Pass preparation (implementation detail; Blender pass names come from general knowledge, not re-checked this session).**
  - Depth from Mist or Z, normalised per shot. Many depth-control models were trained on estimator-style relative depth (the Wan 2.2 VACE workflow uses Depth Anything v2), so a cheap way to match their expectations is to run the same estimator on the clean clay render.
  - Line art from Freestyle or Line Art, or a canny filter on the clay render.
  - OpenPose-format skeletons drawn by projecting armature joints through the shot camera.
  - Object-index or Cryptomatte masks for per-subject segmentation.
  - A camera-path JSON with per-frame intrinsics and extrinsics, kept for future camera-conditioned models.
  - Build these with Blender 5.x compositor APIs, given the traps above.
- **Duration and move checks at hand-off.** Respect model limits before submitting: Kling Motion Control at most 10 s or 30 s depending on the orientation source, Seedance references about 15 s total, and Higgsfield's 2–30 s clip rule. Split shots with more than one primary camera move instead of hoping the model follows a compound move. The Seedance report rates advanced camera moves as the hardest category.

| App target | Best Blender-derived input (documented) | Photoreal tactic | Still unverified |
|---|---|---|---|
| Seedance 2.0 | Up to 9 images, 3 videos (about 15 s) and audio, with @-roles | Photoreal keyframes as `@Image`; greybox or depth MP4 as `@Video`, "camera/blocking only" | Official API field names; whether a depth video is read as geometry or as style |
| Veo 3.1 | First + last frame; up to 3 reference images | Restyle Blender start and end frames to photoreal; camera move in text | Exact request schema; reference images on Fast |
| Kling 3.0 | Motion Control: image + motion video + Element; camera only via prompt | Mannequin performance rendered with a locked camera; photoreal appearance image | Start/end-frame image-to-video and camera parameters on 3.0 |
| Wan 2.2 | VACE depth/pose control video (open models, ComfyUI) | Depth-first, moderate or decaying strength, photoreal first frame | Whether the app's hosted Wan 2.2 provider exposes VACE or Fun-Control |
| Sora 2 | Not researched | — | Everything |
| (Optional) LTX-2 | IC-LoRA depth, pose or canny | Around 0.6 strength; style-only prompt | — |
| (Optional) Runway Aleph | Video-to-video restyle + up to 3 references | Restyle the clay or greybox MP4 with photoreal references | Official limits and structure fidelity |

### Gaps
- Sora 2 input options (image or video references, camera control) were not researched.
- The official Veo 3.1 parameter schema was not reached; sources conflict.
- Seedance 2.0's official API (BytePlus/Volcengine) parameter names and limits were not checked.
- These were not covered (search budget): Kling 3.0's non-motion-control camera and start/end-frame options; Wan 2.2 Fun-Control and Fun-Camera; Uni3C, ReCamMaster, CameraCtrl-style camera-trajectory adapters; hosted availability of VACE.
- No quantitative benchmark compares how faithfully these models follow Blender-derived control.
- Recommended strength values are anecdotal. Wan VACE sources conflict, and LTX's 0.6 is one guide's suggestion.
- Runway's own documentation for Aleph and Aleph 2 was not reached.

## 4. Round-trip practices: object IDs, detecting edits, projecting 3D back to 2D layouts or keyframe conditioning, and versioning so only edited shots regenerate

### Takeaway
No standard exists, but the published patterns agree:
- stable app-owned ids in custom properties, never names;
- idempotent "re-run the manifest = update in place" builds;
- snapshots and labelled checkpoints before any mutation;
- provenance records that tie every generated artifact to the shot that used it;
- never silently overriding a camera or setting the user changed on purpose.

Detecting changes is up to the app. Compute a per-shot fingerprint of the evaluated camera, timing and visible-object state, classify what changed, and regenerate only the dirty shots, at the smallest scope.

### Cited Findings
- Higgsfield manifests: ids "persist as custom properties, so re-running the same manifest updates the managed datablocks instead of duplicating them." Names are semantic and unique, describing role, "never creation order." — [HF skill: blender-scene-spec](https://higgsfield.ai/plugins/blender)
- Higgsfield's checkpoint manifest is taken immediately before any mutation. — [HF skill: blender-scene-spec](https://higgsfield.ai/plugins/blender)
  - It captures a scene snapshot, the protected or edited objects, the active camera, frame range, render engine and current frame, a screenshot, and "the planned created/modified/deleted names."
  - Checkpoints carry labels that state the state reached; "copies named only by time carry no recovery information."
  - Risky edits go into "new collections/datablocks" and the replaced originals are kept until the audit passes.
- Respecting user edits: "An existing intentional camera, view transform, exposure, World, or rig is preserved and recorded unless the brief demands the change." Also: "If the existing file's active camera, resolution, render engine, or exposure looks intentional, changing it silently is forbidden." — [HF skill: blender-scene-spec](https://higgsfield.ai/plugins/blender), [HF skill: blender-lighting-camera](https://higgsfield.ai/plugins/blender)
- Generation provenance and idempotency:
  - Keep "job id, model id, parameters, result path/URL, and which Blender object/material/World/shot consumed it."
  - Submit with a stable `idempotency_key` and reuse it after an uncertain timeout.
  - A proxy stays until its replacement passes audit.
  — [HF skill: blender-generation](https://higgsfield.ai/plugins/blender)
- Asset swaps break cameras: "Swapping a hero asset invalidates every camera set against its old shell … Re-verify placement and near clipping per camera after any swap." — [HF skill: blender-lighting-camera](https://higgsfield.ai/plugins/blender)
- MCP for Blender matches datablocks by a custom property stamped at import time, not by parsing names. Its provenance properties are saved into the .blend so a later user can trace origin. — [addon.py](https://github.com/ahujasid/blender-mcp/blob/main/addon.py), [README](https://github.com/ahujasid/blender-mcp/blob/main/README.md)
- Live-link transport precedent: JSON commands over a local TCP socket (`type`/`params` in, `status`/`result` out; localhost:9876 by default). Remote mode is unauthenticated. — [MCP for Blender README](https://github.com/ahujasid/blender-mcp/blob/main/README.md)
- Shot Manager pipeline hooks: project-wide naming (`[Proj][Seq][Shot]`), settings pushed by a production tool when a file opens, and a stable API. — [Use in production](https://github.com/ubisoft/shotmanager/blob/main/docs/how-to/use-in-production.rst)
- Other shot-state precedents: Camera Configurator tracks per-shot status, Shotlist locks shots and markers against accidental edits, and the Kitsu add-on builds shot files from tracker data. **(search summary)** — [Camera Configurator docs](https://superhivemarket.com/products/camera-configurator/docs), [Shotlist docs](https://superhivemarket.com/products/shotlist-shots-manager/docs), [Blender Artists thread](https://blenderartists.org/t/file-organization-for-animated-short-film/1190776)
- Blender 5.1 animation API change that matters for any script reading edited camera or character animation: "Slotted actions (5.1+): `action.fcurves` is GONE." F-curves are reached through `action.layers[].strips[].channelbags[].fcurves`, filtered by `animation_data.action_slot.handle`. Without that filter, the walk "also returns the object-data curves and edits them by accident." `keyframe_insert` still works. — [HF skill: blender-volatile](https://higgsfield.ai/plugins/blender)
- Going from a generated image back to 3D proxies is possible. Higgsfield's catalog includes `sam_3_3d_body` (image → body shape/pose GLB plus keypoints) and several image-to-3D routes. — [HF skill: blender-volatile](https://higgsfield.ai/plugins/blender)

### Inferences
- **ID scheme.** Stamp every managed datablock (objects, cameras, lights, collections, scenes, markers, VSE strips) with custom properties: `p2s_id` (an app UUID), `p2s_kind` (subject, set, camera, light or shot), `p2s_shot_ids` (for shot-local data), `p2s_role`, `p2s_label` and `p2s_export_rev`. Embed the full export manifest as a text datablock (e.g. `P2S_MANIFEST.json`) and also write it as a sidecar file. Match on `p2s_id`, never on names, since users rename and duplicate objects (MCP for Blender's name-parsing lesson). Treat objects without `p2s_id` as user additions and import them as new set elements, labelled by the user or inferred from the name.
- **Edit detection (headless, on save or import).** For each shot, sample the evaluated scene at the shot's frames (start, middle, end and every N frames) and hash a quantised record of:
  - camera world matrix, focal length, sensor, shift, clip range and DOF focus;
  - shot frame range, order and duration;
  - for every object assigned to the shot or inside its view frustum: world matrix, dimensions, armature pose (bone matrices), and light colour, energy and direction.

  Quantise to sensible tolerances (e.g. 1 mm and 0.05°) so float noise does not mark shots dirty. Compare against the fingerprint stored at export. Read animation through the Blender 5.1 slotted-action API.
- **Classify each change to pick the smallest regeneration.**

  | Change | Action |
  |---|---|
  | Timing only, new duration within the existing clip | Re-trim or re-time; no generation |
  | Timing only, new duration exceeds the clip | Regenerate or extend |
  | Camera only | New keyframe + clip; reuse identity references |
  | Blocking or pose | New keyframe + clip |
  | Lights or atmosphere only | Restyle keyframe + clip |
  | Shared set element | Every shot whose frustum contains it |
  | Hero asset swap | Re-check every camera (Higgsfield rule) |

  Never touch clean shots or their keyframes, clips and feedback history.
- **Project 3D back to the app's 2D layout** (the exact Blender helper, e.g. `bpy_extras.object_utils.world_to_camera_view`, is from general knowledge, not re-checked). For each dirty shot, project subject bounding boxes and key joints (head, pelvis, feet) through the shot camera into normalised frame coordinates. From those, update:
  - subject position and scale (height as a fraction of frame), and facing (subject forward vector against the camera axis);
  - shot size, from subject height fraction;
  - camera angle, from pitch and roll (dutch);
  - horizon, from pitch and lens;
  - key-light azimuth and elevation relative to the camera;
  - the camera move label (dolly, pan, tilt, crane, orbit), by breaking down the camera animation.

  Then write those fields back into the shot JSON, re-render the pencil panel so 2D and 3D agree, and rewrite the prompt text. Kling and Veo receive the camera move only as text, so this rewrite matters.
- **Conditioning artifacts per dirty shot.** Render depth, line, skeleton and mask passes, the clay MP4 plus its exact first-frame PNG (Higgsfield's deliverable), and the camera-path JSON. Cache each by content hash.
- **Versioning.** Content-address every generated artifact by hash(conditioning inputs + prompt + model + parameters + seed). Keep earlier takes and mark shots stale rather than deleting them. Submit jobs with idempotency keys. Merge per shot and per field in three ways (base export, app changes, Blender changes). Mark fields the user edited as locked, so later replanning by Claude cannot silently overwrite them (Higgsfield's "changing it silently is forbidden"). Log each edit as a labelled preference signal for the app's feedback learning.
- **Transport.**
  - **File-based (recommended first):** the user saves the .blend; the app runs headless Blender with a diff script. This is simple, safe and works offline.
  - **Live add-on:** a socket like MCP for Blender's syncs instantly but must stay on localhost, given the unauthenticated remote mode.
  - Higgsfield's connector itself shows that an agent can also drive edits live through MCP.

### Gaps
- No published standard or vendor documentation was found for per-shot change detection or selective regeneration in Blender previs. The design above is inferred.
- USD as an interchange (prim paths as ids, layers for user edits) was not researched.
- Blender's built-in change hooks (depsgraph update handlers, msgbus) and library-override behaviour for linked sets in 5.x were not checked.
- How Higgsfield's "reblock the whole shot" handles hand edits (merge or overwrite) is not documented in what was read.

## 5. 2025–2026 research and products on 3D-aware or previs-conditioned video generation

### Takeaway
Research from 2025–2026 converges on one recipe: keep an explicit 3D proxy (point cloud, cache, mesh, skeleton, or a full layout-plus-lighting scene), render it along the target camera, and condition the video model on those renders. Examples are GEN3C, 3DScenePrompt (ICLR 2026), E³C, Track2View and LiVER. Edited Blender previs is naturally such a proxy. Products (Higgsfield in Blender, fal H3 Max, Mixar) so far use renders as reference or restyle inputs rather than true 3D conditioning of hosted models.

### Cited Findings
- **GEN3C (NVIDIA):** predicts depth for input images, lifts it into a 3D cache, and generates video "conditioned on the 2D renderings of the 3D cache with the new camera trajectory provided by the user." **(search summary)** — [NVIDIA Research: GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/)
- **3DScenePrompt (ICLR 2026; arXiv 2510.14945, Oct 2025):** extends an input video along a user-chosen camera path. It combines temporal conditioning from the last few frames with spatial conditioning from rendered views of a static-only point-cloud memory, built by dynamic SLAM with dynamic masking. Code availability was listed as "to be confirmed." **(search summary)** — [Project page](https://cvlab-kaist.github.io/3DScenePrompt/), [ICLR 2026 listing](https://mlanthology.org/iclr/2026/lee2026iclr-3d/), [paper](https://arxiv.org/pdf/2510.14945)
- **E³C (arXiv 2605.26316, May 2026):** rebuilds a point-cloud memory from context frames and renders it into the target future viewpoints as aligned conditioning. It also renders dynamic human skeletons into those viewpoints. This is close to "render previs skeletons and set, then condition." **(search summary)** — [E³C](https://arxiv.org/pdf/2605.26316)
- **Track2View (arXiv 2606.15534, Jun 2026):** conditions a video diffusion transformer on paired 3D point tracks projected into both the source and target camera views, instead of dense renders. **(search summary)** — [Track2View](https://arxiv.org/html/2606.15534)
- **LiVER (arXiv 2604.07966, Apr 2026), "Lighting-grounded Video Generation with Renderer-based Agent Reasoning":** conditions video synthesis on an explicit, editable 3D scene proxy with object layout, lighting and camera parameters. Of the papers found, its inputs are the closest to the app's (layout, key light, camera). **(search summary)** — [LiVER](https://arxiv.org/html/2604.07966)
- **GeoStream (arXiv 2606.15162, Jun 2026):** precise camera-controlled streaming video generation; the method was not checked. — [GeoStream](https://arxiv.org/pdf/2606.15162)
- **Products and guides that turn 3D into photoreal video (2026):**
  - Higgsfield's Blender greybox → MP4 + first-frame flow. — [HF skill: blender-greybox](https://higgsfield.ai/plugins/blender)
  - fal "H3 Max" 3D-to-video, video-to-video. **(search summary)** — [fal](https://fal.ai/h3-max-3d-to-video)
  - Mixar's beauty, clay and depth guides. **(search summary)** — [Mixar](https://www.mixar.app/blog/blender-blockout-to-video)
  - NVIDIA's Blender → depth → photoreal image → video guide. **(search summary)** — [NVIDIA](https://www.nvidia.com/en-us/geforce/news/rtx-ai-video-generation-guide/)
  - Flick's "Blender for AI Filmmaking: The 2026 Guide" (title only). — [Flick](https://flick.art/blog/blender-ai-filmmaking)
- **Seedance 2.0 report (Apr 2026):** advanced camera movement remains the hardest category even for leading models. **(search summary)** — [arXiv 2604.14148](https://arxiv.org/pdf/2604.14148)

### Inferences
- Designing the export around per-shot camera-aligned renders (depth, skeleton, masks, clay) plus a camera-path JSON fits the research direction. The same artifacts that drive Wan VACE or LTX today match the inputs that GEN3C-, E³C- and LiVER-style models consume. They should also suit any future hosted "3D or camera conditioning" endpoint.
- None of the evidence gathered shows hosted vendors (ByteDance, Google, Kuaishou, OpenAI) taking 3D or camera-path inputs directly. The practical bridge today is reference video (Seedance) or video-to-video (Aleph), plus photoreal keyframes. Camera precision for complex moves is still the weak point, per the Seedance report.
- LiVER and E³C are the two most relevant papers to read in full once arXiv is reachable. LiVER's editable layout, lighting and camera proxy mirrors the app's frame-layout schema. E³C's skeleton renders mirror mannequin blocking.

### Gaps
- No arXiv paper could be opened, so methods, benchmarks, code release and licences are unverified. Everything above comes from search snippets.
- These were not covered (search budget): VideoFrom3D, Diffusion-as-Shader, ReCamMaster, Uni3C, CameraCtrl II, AC3D and similar 2025 camera or 3D-guided video papers; NVIDIA Cosmos-style world models; and any "previs-to-video" products beyond those listed.
- No evidence was found on whether any hosted model (Seedance 2.0, Veo 3.1, Kling 3.0, Sora 2, Wan 2.2 hosts) accepts explicit camera trajectories or depth videos as geometry rather than as style references.
