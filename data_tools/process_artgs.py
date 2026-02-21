import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import json

from glob import glob
from PIL import Image
from data_tools.process_utils import *


if __name__ == '__main__':
    root = os.path.expanduser(f'~/Projects/ArtGS_realease/data/')
    dataset = 'artgs'
    subset = 'sapien'
    data_dir = f'{root}/{dataset}/{subset}'
    scenes = sorted(glob(f'{data_dir}/*'))
    scene_names = [os.path.basename(s) for s in scenes]
    print(scene_names)
    for scene_name in ['storage_47648']:
        scene = f'{data_dir}/{scene_name}'
        num_slots = json.load(open('./arguments/num_slots.json', 'r'))[dataset][subset][scene_name]
        for split in ['train']:
            file_start = json.load(open(f'{scene}/transforms_train_start.json'))
            file_end = json.load(open(f'{scene}/transforms_train_end.json'))

            fovx, fovy = file_start['camera_angle_x'], file_start['camera_angle_y']
            K = np.zeros((3, 3))
            w, h = Image.open(f'{scene}/start/{split}/rgba/0000.png').size
            K[0, 0] = fov2focal(fovx, w)
            K[1, 1] = fov2focal(fovy, h)
            K[0, 2] = w // 2
            K[1, 2] = h // 2
            K[2, 2] = 1

            poses_start = [frame['transform_matrix'] for frame in file_start['frames']]
            poses_end = [frame['transform_matrix'] for frame in file_end['frames']]
        cluster = 'subset' == 'realscan' or num_slots > 5
        generate_pcd(scene, K, poses_start, poses_end, num_slots, reprocess=True, cluster=cluster, visualize=True)
                
