import bpy
import sys
import torch

#import pip
#pip.main(['install', 'torch', 'torchvision', 'torchaudio', '--user'])
packages_path = "C:\\Users\\eggyr\\AppData\\Roaming\\Python\\Python311\\Scripts" + "\\..\\site-packages"
sys.path.insert(0, packages_path)

from mathutils import Euler

import time
import numpy as np
import pathlib

wkdir_path = str(pathlib.Path(__file__).parent.resolve())
wkdir_path = wkdir_path[:wkdir_path.rfind('\\')]
wkdir_path = wkdir_path[:wkdir_path.rfind('\\')]
wkdir_path += "\\data_processing"

sys.path.append(wkdir_path)

print(f'working directory path: {wkdir_path}')

data_path = wkdir_path[:wkdir_path.rfind('\\')]
data_path = data_path[:data_path.rfind('\\')]
data_path += "\\motion_data\\"

print(f'data path: {data_path}')

from data_reformat import format_data, read_rep1, from_blend_coords, to_blend_coords
from data_reformat import get_vec_from_euler, representation0_injection
from data_reformat import get_bone_mapping, get_representation1_mapping
from data_reformat import representation1, representation1_backwards_partial, representation1_partial_mask

print(f'import success')

VISUALIZE = 195 # the sequence to visualize

"""
Blender Object Manipulation Functions
"""

def create_empty(position, name):
    """
    creates a new empty with specified name and position and returns it
    """
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=position)
    new_empty = bpy.context.object
    new_empty.name = f"{name}" 
    return new_empty

def rotate_about_world_origin(x, rotation):
    """
    rotates an object about world origin
    """
    rot_mat = Euler(rotation).to_matrix().to_4x4()
    x.matrix_world = rot_mat @ x.matrix_world

def move_bone(armature_name, bone_name, translation):
    """
    move armature bone, currently only used for testing IK solve speeds
    """
    # Enter pose mode so rig can be modified
    bpy.ops.object.mode_set(mode='POSE')

    armature = bpy.data.objects[armature_name]
    bone = armature.pose.bones[bone_name]

    bone.rotation_mode = 'XYZ'
    bone.location[2] += translation[0]
    bone.location[1] += translation[1]
    bone.location[0] += translation[2]

def test_movement(iterations):
    """
    time how fast it takes for blender to solve IK movements by moving a single bone back and forth
    """
    times = np.zeros((iterations,))
    for i in range(iterations):
        move_bone('hutao', 'footcontrol.l', [0, -0.5, 0.5])

        start_time = time.time()
        bpy.context.view_layer.update()
        end_time = time.time()
        ik_time = end_time - start_time
        times[i] = ik_time

        move_bone('hutao', 'footcontrol.l', [0, 0.5, -0.5])
        bpy.context.view_layer.update()
    print(f'average solve time: {np.mean(times)*1000} ms, std {np.std(times)*1000} ms')


"""
Visualization of Data
"""

def visualize_skeleton(positions):
    """
    accepts a numpy array of shape (frames, joints, 3) and keyframes empties to the location of joints

    the empties will be called joint.{JOINT_NUM}, they will be created if not existing, otherwise the keyframes
    will overwrite the existing empties.
    """
    for frame in range(positions.shape[0]):
        for joint in range(positions.shape[1]):
            # create empty if not exist, otherwise retrieve the empty
            x = bpy.context.scene.objects.get(f"joint.{joint}")
            if not x:
                x = create_empty([0,0,0], f"joint.{joint}")

            # set the position, update scene and add keyframe    
            x.location = positions[frame, joint]
            bpy.context.view_layer.update()
            x.keyframe_insert(data_path="location", frame=frame)


def test_visualization(file_path):
    """
    a testing function for all the functions to ensure everything works properly
    """
    all_global_positions, all_global_rotations = format_data(file_path)

    # visualize original ground truth
    visualize_skeleton(all_global_positions[VISUALIZE])
    
    # creating new representation
    rep1 = representation1(all_global_positions, all_global_rotations)
    print(f'CONVERTED TO CUSTOM REPRESENTATION \n\nrep1 first instance: \n{rep1[0, 0]}')
    restored_positions = representation1_backwards(rep1)
    print(f'RESTORED TO ORIGINAL FORMAT IN BLENDER COORDS\n\nfirst instance of restored: {restored_positions[0,0]}')
    all_restored_positions = from_blend_coords(restored_positions)
    print(f'RESTORED TO ORIGINAL FORMAT IN ORIGINAL COORDS\n\nfirst instance of restored: {all_restored_positions[0,0]}')
    
    # visualize the armature movement controlled by rep1 data
    visualize_armature(rep1[VISUALIZE])
    
    print(f'PARTIAL RESTORATION CHECK:')

    # overwrites elbow and knee positions with the ik pull target rotation instead into ground truth
    all_global_positions = representation0_injection(torch.from_numpy(all_global_positions)).numpy()

    partial_back = representation1_backwards_partial(rep1)
    mask = representation1_partial_mask()
    original_masked = all_global_positions * mask[:, np.newaxis] 
    total_error = np.sum(np.abs(original_masked - partial_back)) / (all_global_positions.shape[0] * all_global_positions.shape[1])
    print(f'avg error per frame: {total_error}')

    error_by_joint = np.mean(np.abs(original_masked - partial_back), axis=-1)
    error_by_joint = np.mean(error_by_joint, axis=(0, 1))
    bm = get_bone_mapping()
    print(f'avg error for each joint:')
    for bone, index in bm.items():
        print(f'  bone {bone}: {error_by_joint[index]}')

    print(f"sample left elbow back values: {partial_back[10,:,bm['left elbow']]}")
    print(f"sample left elbow original values: {original_masked[10,:,bm['left elbow']]}")
    print(f"sample left knee back values: {partial_back[10,:,bm['left knee']]}")
    print(f"sample left knee original values: {original_masked[10,:,bm['left knee']]}")

    print(f'all visualizations successful')
    
    return all_restored_positions


def representation1_backwards(rep1):
    """
    uses blender to create the armature and observe the locations of all joints. Also used to visualize neural network output
    """
    INDICES_IN_USE = 22
    rep0 = np.zeros((rep1.shape[0], rep1.shape[1], INDICES_IN_USE, 3))
    
    for sequence in range(rep1.shape[0]):
        for frame in range(rep1.shape[1]):
            # move controllers of the armature in blender
            representation1_set_locations(rep1[sequence, frame], -1)

            # observe where the solved joint positions ended up and 
            # store the information back into global position representation (rep0)
            for joint in range(rep1.shape[2]):
                rep0[sequence, frame, joint] = bpy.data.objects[f"solved_joint.{joint}"].location
    return rep0


def visualize_armature(rep1_sequence, offset=0, convert_coords=False):
    """
    keyframes all the controllers of a sequence in rep1, optional offset frames and conversion into blender coords
    """
    for frame in range(rep1_sequence.shape[0]):
        representation1_set_locations(rep1_sequence[frame], frame + offset, convert_coords)


def representation1_set_locations(rep1_frame, keyframe_frame=-1, convert_coords=False):
    """
    rep1_frame is a single frame containing all the controllers, shape should be (22, 3)

    keyframe_frame is an optional parameter, if set to anything other than -1, a keyframe in blender is created
    for all the controllers at that frame

    convert_coords will run to_blend_coords() on all the resulting locations after conversion from rep1 to cartesian,
    this is done if the data wasn't converted to blender coords before conversion into rep1.
    """
    rm1 = get_representation1_mapping()
    rm = get_representation1_mapping()
    locations = np.zeros((22, 3))
    
    # joints
    root_location = get_vec_from_euler(rep1_frame[rm1['root']])
    head = bpy.data.objects[f"ctrl_head"]
    locations[rm['head']] = get_vec_from_euler(rep1_frame[rm1['head']]) + root_location
    spine_top = bpy.data.objects[f"ctrl_spine_top"]
    locations[rm['spine top']] = get_vec_from_euler(rep1_frame[rm1['spine top']]) + root_location
    root_joint = bpy.data.objects[f"root"]
    locations[rm['root']] = root_location
    root_joint.rotation_euler = Euler(rep1_frame[rm1['root rotation']], 'XYZ')
    
    # hands
    left_hand = bpy.data.objects[f"ctrl_arm.l"]
    locations[rm['left hand']] = get_vec_from_euler(rep1_frame[rm1['left hand']]) + root_location

    right_hand = bpy.data.objects[f"ctrl_arm.r"]
    locations[rm['right hand']] = get_vec_from_euler(rep1_frame[rm1['right hand']]) + root_location
    
    # feet
    left_foot = bpy.data.objects[f"ctrl_leg.l"]
    locations[rm['left foot']] = get_vec_from_euler(rep1_frame[rm1['left foot']]) + root_location
    
    right_foot = bpy.data.objects[f"ctrl_leg.r"]
    locations[rm['right foot']] = get_vec_from_euler(rep1_frame[rm1['right foot']]) + root_location
    
    pull_target_multiplier = 1
    if convert_coords:
        pull_target_multiplier = 100

    left_elbow = bpy.data.objects["ctrl_elbow.l"]
    locations[rm['left elbow']] = get_vec_from_euler(rep1_frame[rm1['left elbow']], pull_target_multiplier) + 0.5 * (locations[rm['left hand']] + locations[rm['spine top']])
    right_elbow = bpy.data.objects["ctrl_elbow.r"]
    locations[rm['right elbow']] = get_vec_from_euler(rep1_frame[rm1['right elbow']], pull_target_multiplier) + 0.5 * (locations[rm['right hand']] + locations[rm['spine top']])
    left_knee = bpy.data.objects["ctrl_knee.l"]
    locations[rm['left knee']] = get_vec_from_euler(rep1_frame[rm1['left knee']], pull_target_multiplier) + 0.5 * (locations[rm['left foot']] + locations[rm['root']])
    right_knee = bpy.data.objects["ctrl_knee.r"]
    locations[rm['right knee']] = get_vec_from_euler(rep1_frame[rm1['right knee']], pull_target_multiplier) + 0.5 * (locations[rm['right foot']] + locations[rm['root']])

    if convert_coords:
        for i in range(len(locations)):
            locations[i] = to_blend_coords(locations[i])

    left_hand.location = locations[rm['left hand']]
    left_elbow.location = locations[rm['left elbow']]
    right_hand.location = locations[rm['right hand']]
    right_elbow.location = locations[rm['right elbow']]

    left_foot.location = locations[rm['left foot']]
    left_knee.location = locations[rm['left knee']]
    right_foot.location = locations[rm['right foot']]
    right_knee.location = locations[rm['right knee']]

    head.location = locations[rm['head']]
    spine_top.location = locations[rm['spine top']]
    root_joint.location = locations[rm['root']]
    
    bpy.context.view_layer.update()

    if keyframe_frame != -1: 
        left_hand.keyframe_insert(data_path="location", frame=keyframe_frame)
        left_elbow.keyframe_insert(data_path="location", frame=keyframe_frame)
        right_hand.keyframe_insert(data_path="location", frame=keyframe_frame)
        right_elbow.keyframe_insert(data_path="location", frame=keyframe_frame)
        left_foot.keyframe_insert(data_path="location", frame=keyframe_frame)
        left_knee.keyframe_insert(data_path="location", frame=keyframe_frame)
        right_foot.keyframe_insert(data_path="location", frame=keyframe_frame)
        right_knee.keyframe_insert(data_path="location", frame=keyframe_frame)
        head.keyframe_insert(data_path="location", frame=keyframe_frame)
        spine_top.keyframe_insert(data_path="location", frame=keyframe_frame)
        root_joint.keyframe_insert(data_path="location", frame=keyframe_frame)
        root_joint.keyframe_insert(data_path="rotation_euler", frame=keyframe_frame)


def test():
    file_name = "lafan1_detail_model_benchmark_5_0-2231.json"
    save_name = "CONVERTED_lafan1_detail_model_benchmark_5_0-2231.json"
    test_visualization(data_path + file_name)


def visualize_from_file():
    gt_file_name = "new_rep_6D_elbknee_lafan1_context_model_benchmark_30_0-2231_gt (1).json"
    file_name = "new_rep_6D_elbknee_lafan1_context_model_benchmark_30_0-2231 (1).json"
    rep1_gt = read_rep1(data_path + gt_file_name)
    rep1_gt = np.array(rep1_gt)
    rep1 = read_rep1(data_path + file_name)
    rep1 = np.array(rep1)
    print(f'rep1_gt shape: {rep1_gt.shape} rep1 shape: {rep1.shape}')

    rm = get_representation1_mapping()
    print(f"inferenced pull target data: left elbow \n{rep1_gt[200, :, rm['left elbow']]}")
    print(f"gt pull target data: left elbow \n{rep1[200, :, rm['left elbow']]}")
    '''
    visualize_armature(rep1[310], convert_coords=True)
    visualize_armature(rep1[560], offset=50, convert_coords=True)
    visualize_armature(rep1[650], offset=100, convert_coords=True)
    visualize_armature(rep1[750], offset=150, convert_coords=True)
    visualize_armature(rep1[850], offset=200, convert_coords=True)
    '''
    visualize_armature(rep1_gt[310], convert_coords=True)
    visualize_armature(rep1_gt[560], offset=50, convert_coords=True)
    visualize_armature(rep1_gt[650], offset=100, convert_coords=True)
    visualize_armature(rep1_gt[750], offset=150, convert_coords=True)
    visualize_armature(rep1_gt[850], offset=200, convert_coords=True)
    
visualize_from_file()
#test()
