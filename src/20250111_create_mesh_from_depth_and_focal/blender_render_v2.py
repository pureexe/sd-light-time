# script for doing blender render
# to RUN: blender -b -P blender_render.py -- model.obj environment.exr <fov_radian> <z_offset> output.png
import bpy
import sys
import os
import math

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        bpy.data.textures.remove(block)
    for block in bpy.data.images:
        bpy.data.images.remove(block)

def import_obj(obj_path):
    bpy.ops.import_scene.obj(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    
    # Set material to diffuse with base color white
    mat = bpy.data.materials.new(name="DiffuseWhite")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.new(type="ShaderNodeBsdfDiffuse")
    bsdf.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    
    output = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    obj.data.materials.append(mat)

    # Set object rotation to (90, 0, 90)
    obj.rotation_euler = (math.radians(90), math.radians(0), math.radians(90))
    
    return obj

def setup_camera(fov, z_position):
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    
    cam.location = (z_position, 0, 0) # z_position in GL format is X in blender
    cam.rotation_euler = (math.pi/2, 0, math.pi/2)
    cam.data.angle = fov
    #cam.data.angle = math.radians(fov)
    
    return cam

def setup_environment(env_map_path):
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new(name="World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    for node in nodes:
        nodes.remove(node)
    
    bg_node = nodes.new(type='ShaderNodeBackground')
    env_tex = nodes.new(type='ShaderNodeTexEnvironment')
    env_tex.image = bpy.data.images.load(env_map_path)
    env_tex.image.filepath = env_map_path
    
    mapping_node = nodes.new(type='ShaderNodeMapping')
    mapping_node.vector_type = 'TEXTURE'
    
    # disable rotation and flip
    #mapping_node.inputs['Scale'].default_value = (1, -1, 1) # we flip Y-axis to match from outside to inside
    #mapping_node.inputs['Rotation'].default_value[2] = math.radians(180)
    
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    
    links.new(tex_coord.outputs['Generated'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], env_tex.inputs['Vector'])
    links.new(env_tex.outputs['Color'], bg_node.inputs['Color'])
    links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

def configure_cycles_for_gpu():
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = True
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 1024

def render_image(output_path):
    configure_cycles_for_gpu()
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # Ensure transparency
    bpy.ops.render.render(write_still=True)

def save_blend_file(output_path):
    blend_path = output_path.replace(".png", ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

def main():
    if len(sys.argv) < 6:
        print("Usage: blender -b -P script.py -- <obj_path> <env_map_path> <fov_radian> <z_position> <output_path>")
        return
    
    argv = sys.argv[sys.argv.index('--') + 1:]
    obj_path = argv[0]
    env_map_path = argv[1]
    fov = float(argv[2])
    z_position = float(argv[3])
    output_path = argv[4]
    
    clear_scene()
    import_obj(obj_path)
    setup_camera(fov, z_position)
    setup_environment(env_map_path)
    render_image(output_path)
    #save_blend_file(output_path)
    
if __name__ == "__main__":
    main()