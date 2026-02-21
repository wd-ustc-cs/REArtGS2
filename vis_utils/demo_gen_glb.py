import os
import json
import numpy as np
import bpy


def clear_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def reset(obj):
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (1, 1, 1)


def optimize_mesh(obj, decimate_ratio=0.1):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
    modifier.ratio = decimate_ratio
    modifier.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier="Decimate")
    

def load_mesh_with_material(src_root, meta, optimize=False):
    for entry in meta:
        mesh_path = os.path.join(src_root, entry['visuals'][0])
        bpy.ops.wm.ply_import(filepath=mesh_path)
        
        # Get the imported object (assumes it's the active object)
        obj = bpy.context.active_object
        if optimize:
            optimize_mesh(obj)
        mesh = obj.data
        
        # Create a new material
        mat = bpy.data.materials.new(name=f"Material_{obj.name}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        # Clear default nodes
        nodes.clear()
        
        # Create necessary nodes
        node_vertexcolor = nodes.new(type='ShaderNodeVertexColor')
        node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        
        # Connect nodes
        links = mat.node_tree.links
        links.new(node_vertexcolor.outputs['Color'], node_bsdf.inputs['Base Color'])
        links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
        
        # Assign material to object
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
        
        # Enable vertex color display
        if len(mesh.vertex_colors) == 0:
            mesh.vertex_colors.new()


def load_joint_info(meta):
    joint_info = []
    for i, entry in enumerate(meta):
        if entry['joint'] == 'hinge':
            info = {
                'type': 'revolute',
                'origin': np.array(entry['jointData']['axis']['origin']),
                'direction': np.array(entry['jointData']['axis']['direction'])
            }
        elif entry['joint'] == 'slider':
            info = {
                'type': 'prismatic',
                'origin': np.array(entry['jointData']['axis']['origin']),
                'direction': np.array(entry['jointData']['axis']['direction'])
            }
        elif entry['joint'] == 'heavy':
            info = {} # root part
        else:
            raise ValueError(f"Unknown joint type: {entry['joint']}")
        joint_info.append(info)
    return joint_info


def R_from_direction_angle(direction, theta):
    # Normalize direction vector
    direction = np.array(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    # Rodriguez rotation formula
    K = np.array([[0, -direction[2], direction[1]],
                  [direction[2], 0, -direction[0]], 
                  [-direction[1], direction[0], 0]])
    
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def animate_joint(joint_info, obj_name, r_range=[-0.3, 0.3], p_range=[-0.05, 0.10], n_frames=100):
    revolute_range = np.concatenate([
        np.linspace(np.pi * r_range[0], np.pi * r_range[1], n_frames // 2),
        np.linspace(np.pi * r_range[0], np.pi * r_range[1], n_frames // 2)[::-1],
    ])
    prismatic_range = np.concatenate([
        np.linspace(p_range[0]-0.05, p_range[0], n_frames // 4),
        np.linspace(p_range[0], p_range[1], n_frames // 4),
        np.linspace(p_range[0], p_range[1], n_frames // 4)[::-1],
        np.linspace(p_range[0], p_range[1], n_frames // 4),
    ])
    # Set scene frame range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = n_frames - 1
    
    # Get all parts except camera and light
    objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and obj_name in obj.name]
    
    # Animate other objects
    for i, (obj, info) in enumerate(zip(objects, joint_info)):
        if not info:  # root part
            translation = obj.matrix_world.to_translation()
            continue
        
        reset(obj)
        loc = np.array(obj.location)
        if info['type'] == 'revolute':
            direction = info['direction']
            origin = info['origin']
            # Animate rotation
            for frame in range(n_frames):
                bpy.context.scene.frame_set(frame)
                theta = revolute_range[frame]
                if 'storage_45503_start_1' in obj.name:
                    theta = -0.5 + theta / 2
                elif 'cabinet_3r_white_start_1' in obj.name:
                    theta = 0.7 + theta
                R = R_from_direction_angle(direction, theta)
                new_rotation = bpy.data.objects.new("", None).rotation_euler.to_matrix()
                for i in range(3):
                    for j in range(3):
                        new_rotation[i][j] = R[i,j]
                obj.rotation_euler = new_rotation.to_euler()
                new_loc = -R @ origin + origin + translation
                obj.location = new_loc
                obj.keyframe_insert(data_path='location')
                obj.keyframe_insert(data_path='rotation_euler')
                
        elif info['type'] == 'prismatic':
            direction = info['direction']
            # Animate translation
            for frame in range(n_frames):
                bpy.context.scene.frame_set(frame)
                theta = prismatic_range[frame]
                print(obj.name)
                if 'table_25493_start_2' in obj.name:
                    theta = -0.1 + theta
                elif 'table_31249_start_1' in obj.name:
                    theta = -theta * 1.4 - 0.16
                elif 'storage_47648_start_2' in obj.name:
                    theta = theta - 0.05
                elif 'cabinet_1r2p_transparent_start_2' in obj.name:
                    theta = theta - 0.05
                t = direction * theta
                obj.location = loc + t + translation
                obj.keyframe_insert(data_path='location')

def export_glb(export_path):
    bpy.ops.export_scene.gltf(
        filepath=export_path,
        export_format='GLB',
        use_selection=False,
        export_materials=True,
        export_textures=True, 
        export_animations=True,
        # export_extras=True,  # 导出自定义属性
        export_lights=True,  # 导出灯光
        export_apply=True,   # 应用修改器
        export_draco_mesh_compression_enable=True,  # 启用Draco压缩
        export_draco_mesh_compression_level=10       # 压缩级别(0-10)
    )

r_ranges = {
    'cabinet_3r_white': [-0.3, 0.2],
    'cabinet_1r_white': [-0.2, 0.1],
    'washingmachine_1r': [-0.2, 0.3],
    'cabinet_1r2p_transparent': [-0.3, 0.3],
    'microwave': [-0.17, 0.4],

    'oven_101908': [-0.14, 0.22],
    'table_25493': [-0.5, 0.1],
    'storage_47648': [-0.2, 0.5],
    'table_31249': [-0.2, 0.3],
    'storage_45503': [-0.2, 0.4],

    'foldchair_102255': [-0.5, 0.2],
    'washer_103776': [-0.5, 0.1],
    'fridge_10905': [-0.5, 0.5],
    'oven_101917': [-0.3, 0.5],
    'stapler_103111': [-0.9, 0.0],
    'USB_100109': [-1, 1],
    'laptop_10211': [-0.5, 0.4],
    'scissor_11100': [-0.5, 0.05],
}

p_ranges = {
    'cabinet_1r2p_transparent': [-0.05, 0.45],

    'table_25493': [-0.05, 0.10],
    'storage_47648': [-0.04, 0.04],
    'table_31249': [-0.05, 0.10],
    'storage_45503': [-0.05, 0.10],

    'storage_45135': [-0.25, 0.25],
    'blade_103706': [-0.2, 0.15],
}

#subset = 'paris'
#objects = 'foldchair_102255 washer_103776 fridge_10905 blade_103706 storage_45135 oven_101917 stapler_103111 USB_100109 laptop_10211 scissor_11100'.split(' ')

subset = 'realscan'
objects = 'cabinet_3r_white cabinet_1r_white washingmachine_1r cabinet_1r2p_transparent microwave'.split(' ')
objects = 'cabinet_3r_white'.split(' ')

#subset = 'sapien'
#objects = 'oven_101908 table_25493 storage_47648 table_31249 storage_45503'.split(' ')
#objects = 'storage_45503'.split(' ')
dst_root =  f'/home/yuliu/Projects/ArtGS/data/demo/{subset}'
for object in objects:
    print(f'Processing {object}')
#    clear_objects()
    src_root = f'/home/yuliu/Projects/ArtGS/data/demo/{subset}/{object}'
    meta = json.load(open(os.path.join(src_root, 'joint_info_cano.json'), 'r'))

    # Load mesh with material
    loaded = False
    for obj in bpy.context.scene.objects:
        if object in obj.name:
            loaded = True
            break
    if not loaded:
        load_mesh_with_material(src_root, meta, optimize=True)
    # Load axis info
    joint_info = load_joint_info(meta)
    # Animate joint
    r_range = r_ranges.get(object, [-0.5, 0.5])
    p_range = p_ranges.get(object, [-0.05, 0.10])
    animate_joint(joint_info, object, n_frames=100, r_range=r_range, p_range=p_range)
    # Export GLB
    export_path = os.path.join(dst_root, f'glbs/{object}.glb')
    bpy.ops.export_scene.gltf(
        filepath=export_path,
        export_format='GLB',  
        use_selection=False,
        export_lights=True,
        export_apply=True    
    )
