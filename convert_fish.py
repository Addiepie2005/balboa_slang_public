import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath='scenes/hw3/meshes/BarramundiFish.fbx')
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
bpy.ops.object.select_all(action='DESELECT')
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.object.join()
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.wm.ply_export(filepath='scenes/hw3/meshes/BarramundiFish.ply', export_normals=True,
                      export_uv=False, export_colors='NONE',
                      export_triangulated_mesh=True,
                      forward_axis='NEGATIVE_Z', up_axis='Y')
