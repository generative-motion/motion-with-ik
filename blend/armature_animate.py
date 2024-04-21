import bpy
from mathutils import Euler

import time
import numpy as np
import pathlib
import sys

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

from data_reformat import format_data, from_blend_coords
from data_reformat import get_vec_from_euler
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
    bm = get_bone_mapping()

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
        
        '''
        x = bpy.context.scene.objects.get(f"joint.left elbow target")
        if not x:
            x = create_empty([0,0,0], f"joint.left elbow target")
        x.location = 2 * positions[frame, bm['left elbow']] - 1.5 * positions[frame, bm['left hand']] - 0.5 * positions[frame, bm['left shoulder']] + positions[frame, bm['left hand']]
        bpy.context.view_layer.update()
        x.keyframe_insert(data_path="location", frame=frame)

        x = bpy.context.scene.objects.get(f"joint.right elbow target")
        if not x:
            x = create_empty([0,0,0], f"joint.right elbow target")
        x.location = 2 * positions[frame, bm['right elbow']] - 1.5 * positions[frame, bm['right hand']] - 0.5 * positions[frame, bm['right shoulder']] + positions[frame, bm['right hand']]
        bpy.context.view_layer.update()
        x.keyframe_insert(data_path="location", frame=frame)

        x = bpy.context.scene.objects.get(f"joint.left knee target")
        if not x:
            x = create_empty([0,0,0], f"joint.left knee target")
        x.location = 2 * positions[frame, bm['left knee']] - 1.5 * positions[frame, bm['left foot']] - 0.5 * positions[frame, bm['left hip']] + positions[frame, bm['left foot']]
        bpy.context.view_layer.update()
        x.keyframe_insert(data_path="location", frame=frame)

        x = bpy.context.scene.objects.get(f"joint.right knee target")
        if not x:
            x = create_empty([0,0,0], f"joint.right knee target")
        x.location = 2 * positions[frame, bm['right knee']] - 1.5 * positions[frame, bm['right foot']] - 0.5 * positions[frame, bm['right hip']] + positions[frame, bm['right foot']]
        bpy.context.view_layer.update()
        x.keyframe_insert(data_path="location", frame=frame)
        '''


def test_visualization(file_path):
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
    for frame in range(rep1.shape[1]):
        representation1_set_locations(rep1[VISUALIZE, frame], frame)
    
    print(f'PARTIAL RESTORATION CHECK:')

    partial_back = representation1_backwards_partial(rep1)
    mask = representation1_partial_mask()
    original_masked = all_global_positions * mask[:, np.newaxis]
    total_error = np.sum(np.abs(original_masked - partial_back))
    print(f'total error: {total_error}')
    
    #visualize_skeleton(partial_back[VISUALIZE])
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


def representation1_set_locations(rep1_frame, keyframe_frame=-1):
    rm1 = get_representation1_mapping()

    root_location = get_vec_from_euler(rep1_frame[rm1['root']])
    
    # hands
    left_hand = bpy.data.objects[f"ctrl_arm.l"]
    left_hand.location = get_vec_from_euler(rep1_frame[rm1['left hand']]) + root_location
    left_elbow = bpy.data.objects["ctrl_elbow.l"]
    left_elbow.location = get_vec_from_euler(rep1_frame[rm1['left elbow']]) + left_hand.location

    right_hand = bpy.data.objects[f"ctrl_arm.r"]
    right_hand.location = get_vec_from_euler(rep1_frame[rm1['right hand']]) + root_location
    right_elbow = bpy.data.objects["ctrl_elbow.r"]
    right_elbow.location = get_vec_from_euler(rep1_frame[rm1['right elbow']]) + right_hand.location
    
    # feet
    left_foot = bpy.data.objects[f"ctrl_leg.l"]
    left_foot.location = get_vec_from_euler(rep1_frame[rm1['left foot']]) + root_location
    left_knee = bpy.data.objects["ctrl_knee.l"]
    left_knee.location = get_vec_from_euler(rep1_frame[rm1['left knee']]) + left_foot.location
    
    right_foot = bpy.data.objects[f"ctrl_leg.r"]
    right_foot.location = get_vec_from_euler(rep1_frame[rm1['right foot']]) + root_location
    right_knee = bpy.data.objects["ctrl_knee.r"]
    right_knee.location = get_vec_from_euler(rep1_frame[rm1['right knee']]) + right_foot.location
    
    # joints
    head = bpy.data.objects[f"ctrl_head"]
    head.location = get_vec_from_euler(rep1_frame[rm1['head']]) + root_location

    spine_top = bpy.data.objects[f"ctrl_spine_top"]
    spine_top.location = get_vec_from_euler(rep1_frame[rm1['spine top']]) + root_location

    root_joint = bpy.data.objects[f"root"]
    root_joint.location = root_location
    root_joint.rotation_euler = Euler(rep1_frame[rm1['root rotation']], 'XYZ')
    
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


file_name = "lafan1_detail_model_benchmark_5_0-2231.json"
save_name = "CONVERTED_lafan1_detail_model_benchmark_5_0-2231.json"
test_visualization(data_path + file_name)
