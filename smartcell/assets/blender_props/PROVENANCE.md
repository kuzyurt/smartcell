# Blender prop provenance

These 49 OBJ meshes are the visual set dressing used by the zero-shot
benchmark (`smartcell/evaluate_imported.py`). They were exported from
[Poly Haven](https://polyhaven.com/) models, which are released under **CC0**.

The source models include: `Barrel_01`, `barrel_03`, `cardboard_box_01`,
`caged_hanging_light`, `concrete_road_barrier`, `fire_alarm`,
`hand_truck`, `industrial_pastic_container`, `korean_fire_extinguisher_01`,
`metal_stool_01`, `metal_tool_chest`, `mounted_fluorescent_lights`,
`old_military_crate`, `plastic_crate_02`, `portable_welding_cart`,
`steel_frame_shelves_01`, `steel_frame_shelves_03`, `tool_cart`,
`WetFloorSign_01`, `worn_metal_rack`.

Each mesh is stored in world coordinates with Z-up (the same convention as
MuJoCo). `manifest.json` records the mesh, flat linear RGBA colour and source
material name. No textures, HDRIs or other Poly Haven files are
redistributed.

`smartcell/blender_props.py` loads every mesh with `contype=0` and `conaffinity=0`, so
the props are invisible to the physics engine. They change what the wrist
camera sees, not how the cell behaves.
