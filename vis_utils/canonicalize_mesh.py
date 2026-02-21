import os
import json
import trimesh
import numpy as np
import open3d as o3d


def save_axis_mesh(k, center, filepath):
    '''support rotate only for now'''
    axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=1.0, cone_height=0.08)
    arrow = np.array([0., 0., 1.], dtype=np.float32)
    n = np.cross(arrow, k) 
    rad = np.arccos(np.dot(arrow, k))
    R_arrow = get_rotation_axis_angle(n, rad)
    axis.rotate(R_arrow, center=(0, 0, 0))
    axis.translate(center[:3])
    o3d.io.write_triangle_mesh(filepath, axis)


def normalize(v):
    return v / np.sqrt(np.sum(v**2))


def get_rotation_axis_angle(k, theta):
    '''
    Rodrigues' rotation formula
    args:
    * k: direction vector of the axis to rotate about
    * theta: the (radian) angle to rotate with
    return:
    * 3x3 rotation matrix
    '''
    k = normalize(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R


def export_axis_mesh(arti, exp_dir, i):
    center = np.array(arti['axis']['origin'], dtype=np.float32)
    k = np.array(arti['axis']['direction'], dtype=np.float32)
    save_axis_mesh(k, center, os.path.join(exp_dir, f'axis_{i}.ply'))
    save_axis_mesh(-k, center, os.path.join(exp_dir, f'axis_oppo_{i}.ply'))


def canonicalize_mesh(meta, src_root, dst_root, obj_name):
    # rx90 = trimesh.transformations.rotation_matrix(angle=np.pi/2, direction=[1, 0, 0], point=[0, 0, 0])
    rx90 = np.eye(4)
    # rz180 = trimesh.transformations.rotation_matrix(angle=np.pi, direction=[0, 0, 1], point=[0, 0, 0])
    rz180 = np.eye(4)
    rotation_matrix = np.dot(rz180, rx90)
    meshes = []
    for i, entry in enumerate(meta):
        mesh_path = os.path.join(src_root, entry['visuals'][0])
        mesh = trimesh.load(mesh_path)
        mesh.apply_transform(rotation_matrix)
        meshes.append(mesh)
    scale = [1.0, 1.0, 1.0]
    # resize
    # scale = [4.9, 4.9, 5.1]
    # for i, mesh in enumerate(meshes):
    #     mesh.vertices *= scale
    total_mesh = trimesh.util.concatenate(meshes)
    translation = -total_mesh.centroid
    # translate
    # translation += [0.99, 2.40, 4.54]
    origin = meta[1]['jointData']['axis']['origin']
    for i, entry in enumerate(meta):
        mesh = meshes[i]
        mesh.vertices += translation
        base_name =  entry['visuals'][0].split('/')[-1]
        new_base_name = f'{obj_name}_{base_name}'
        cano_path = os.path.join(dst_root, entry['visuals'][0].replace('meshes', 'meshes_cano')).replace(base_name, new_base_name)
        if entry['joint'] == 'hinge' or entry['joint'] == 'slider':
            direction = entry['jointData']['axis']['direction']
            new_direction = np.dot(rotation_matrix[:3, :3], np.array(direction))
            origin = entry['jointData']['axis']['origin']
            new_origin = np.dot(rotation_matrix[:3, :3], np.array(origin) * scale) + translation
            entry['jointData']['axis']['direction'] = new_direction.tolist()
            entry['jointData']['axis']['origin'] = new_origin.tolist()
        mesh.export(cano_path)
        entry['visuals'][0] = entry['visuals'][0].replace('meshes', 'meshes_cano').replace(base_name, new_base_name)
        meta[i] = entry
    with open(os.path.join(dst_root, 'joint_info_cano.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    return meta


def process(src_root, dst_root, obj_name):
    meta = json.load(open(os.path.join(src_root, 'joint_info.json'), 'r'))
    os.makedirs(dst_root, exist_ok=True)
    os.makedirs(os.path.join(dst_root, 'meshes_cano'), exist_ok=True)
    meta = canonicalize_mesh(meta, src_root, dst_root, obj_name)
    for i, entry in enumerate(meta):
        if entry['joint'] == 'hinge' or entry['joint'] == 'slider':
            joint_info = entry['jointData']
            export_axis_mesh(joint_info, os.path.join(dst_root, 'meshes_cano'), i)


if __name__ == '__main__':
    subset = 'paris'
    objects = 'foldchair_102255 washer_103776 fridge_10905 blade_103706 storage_45135 oven_101917 stapler_103111 USB_100109 laptop_10211 scissor_11100'.split(' ')

    # subset = 'sapien'
    # objects = 'oven_101908 table_25493 storage_47648 table_31249 storage_45503'.split(' ')
    for object in objects:
        print(f'Processing {object}')
        src_root = f'data/demo/{subset}/{object}'
        dst_root =  f'data/demo/{subset}/{object}'
        process(src_root, dst_root, object)


    
