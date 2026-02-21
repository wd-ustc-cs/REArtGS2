import os
import json


def export_urdf(input_dir, output_dir, scene, larger_motion_state):
    meta_info = json.load(open(os.path.join(input_dir, 'joint_info.json')))
    urdf = '<?xml version="1.0"?>\n<robot name="robot">\n'
    urdf += '\t<link name="base"/>\n'
    for i, item in enumerate(meta_info):
        state = 'start' if larger_motion_state == 0 else 'end'
        item["visuals"][0] = item["visuals"][0].replace('start', state)
        link_name = f'link_{i}'
        urdf += f'\t<link name="{link_name}">\n'
        if item['parent'] != -1:
            urdf += f'\t\t<visual>\n'
            urdf += f'\t\t\t<origin xyz="{-item["jointData"]["axis"]["origin"][0]} {-item["jointData"]["axis"]["origin"][1]} {-item["jointData"]["axis"]["origin"][2]}"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</visual>\n'
            urdf += f'\t\t<collision>\n'
            urdf += f'\t\t\t<origin xyz="{-item["jointData"]["axis"]["origin"][0]} {-item["jointData"]["axis"]["origin"][1]} {-item["jointData"]["axis"]["origin"][2]}"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</collision>\n'
            urdf += f'\t</link>\n'
            joint_type = 'revolute' if item["joint"] == 'hinge' else 'prismatic'
            urdf += f'\t<joint name="{item["name"]}" type="{joint_type}" >\n'
            urdf += f'\t\t<origin xyz="{item["jointData"]["axis"]["origin"][0]} {item["jointData"]["axis"]["origin"][1]} {item["jointData"]["axis"]["origin"][2]}" rpy="0 0 0"/>\n'
            urdf += f'\t\t<axis xyz="{item["jointData"]["axis"]["direction"][0]} {item["jointData"]["axis"]["direction"][1]} {item["jointData"]["axis"]["direction"][2]}"/>\n'
            urdf += f'\t\t<parent link="link_{item["parent"]}"/>\n'
            urdf += f'\t\t<child link="{link_name}"/>\n'
            urdf += f'\t</joint>\n'
        else:
            urdf += f'\t\t<visual>\n'
            urdf += f'\t\t\t<origin xyz="0 0 0"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</visual>\n'
            urdf += f'\t\t<collision>\n'
            urdf += f'\t\t\t<origin xyz="0 0 0"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="{item["visuals"][0]}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</collision>\n'
            urdf += f'\t</link>\n'
            urdf += f'\t<joint name="joint_{i}" type="fixed" >\n'
            urdf += f'\t\t<origin rpy="1.570796326794897 0 -1.570796326794897" xyz="0 0 0"/>\n'
            urdf += f'\t\t<parent link="base"/>\n'
            urdf += f'\t\t<child link="{link_name}"/>\n'
            urdf += f'\t</joint>\n'
    urdf += '</robot>'
    with open(os.path.join(output_dir, f'{scene}.urdf'), 'w') as f:
        f.write(urdf)


dataset = 'artgs'
subset = 'sapien'
subset = 'realscan'
# scenes = 'oven_101908 table_25493 storage_45503 storage_47648 table_31249'.split(' ')
scenes = 'storage_47648'.split(' ')
scenes = 'washingmachine_1r cabinet_1r2p_transparent microwave cabinet_1r_white cabinet_3r_white'.split(' ')
output_dir = 'data/demo/realscan'
larger_motion_state = json.load(open('arguments/larger_motion_state.json'))[dataset][subset]
for s in scenes:
    output_scene = os.path.join(output_dir, s)
    # os.makedirs(output_scene, exist_ok=True)
    # input_dir = f'outputs/{dataset}/{subset}/{s}/1.0-0.5/train/ours_20000'
    # print(f'Copying {input_dir} to {output_scene}')
    # os.system(f'cp -r {input_dir}/meshes {output_scene}/')
    # os.system(f'cp {input_dir}/joint_info.json {output_scene}/')
    export_urdf(output_scene, output_scene, s, larger_motion_state[s])
print('done')