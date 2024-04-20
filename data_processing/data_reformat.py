"""
COPY AND PASTED FROM BLENDER FILE
"""

import bpy
from bpy import context as C
from mathutils import Euler
import mathutils
from math import pi
import math

import time
import numpy as np
import os
import json

def create_object(position, id):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=position)

    # store the created Empty in a variable for further manipulation
    new_empty = bpy.context.object
    new_empty.name = f"{id}"  # Rename the Empty
    #rotate_about_world_origin(new_empty, (3,0,0))
    return new_empty
    
    
def rotate_about_world_origin(x, rotation):
    rot_mat = Euler(rotation).to_matrix().to_4x4()
    x.matrix_world = rot_mat @ x.matrix_world
    

def move_bone(bone_name, amount):
    # Enter pose mode so rig can be modified
    bpy.ops.object.mode_set(mode='POSE')

    # Specify the armature object
    armature = bpy.data.objects['rig1']

    # Specify the bone
    bone = armature.pose.bones[bone_name]

    # Set the bone's rotation mode to Euler XYZ, if not already
    bone.rotation_mode = 'XYZ'
    bone.location[2] += amount[0]
    bone.location[1] += amount[1]
    bone.location[0] += amount[2]


def test_movement(iterations):
    times = np.zeros((iterations,))
    
    for i in range(iterations):

        move_bone('footcontrol.l', [0, -0.5, 0.5])

        start_time = time.time()
        bpy.context.view_layer.update()
        end_time = time.time()
        ik_time = end_time - start_time

        move_bone('footcontrol.l', [0, 0.5, -0.5])
        bpy.context.view_layer.update()
        
        times[i] = ik_time
        
    return times


#times = test_movement(1000)
#print(f'average solve time: {np.mean(times)*1000} ms, std {np.std(times)*1000} ms')




def visualize(all_global_positions, sequence):
    for frame in range(all_global_positions.shape[1]):
        for joint in range(all_global_positions.shape[2]):
            x = bpy.data.objects[f"joint.{joint}"]
            x.location = all_global_positions[sequence, frame, joint]
            bpy.context.view_layer.update()
            #rotate_about_world_origin(x, (pi/2,0,0))
            #bpy.context.view_layer.update()
            x.keyframe_insert(data_path="location", frame=frame)


def read_input(file_path):
    with open(file_path) as json_data:
        d = json.load(json_data)

    return d['positions'], d['rotations'], d['parents'], d['foot_contact']

def init_armature():
    for i in range(22):
        create_object((0,0,0), f"joint.{i}")

def main(file_path):
    # reading all data
    init_armature()
    all_orig_positions, all_orig_rotations, parents, all_foot_contact = read_input(file_path)
    all_orig_positions = np.array(all_orig_positions)
    all_orig_rotations = np.array(all_orig_rotations)
    print(f'original positions shape: {all_orig_positions.shape}')

    all_global_positions, all_global_rotations = conv_rig(all_orig_positions, all_orig_rotations, parents)
    visualize(all_global_positions, 0)
    
    # creating new representation
    all_new_positions, all_new_rotations = representation1(all_global_positions)
    print(f'rep1 first instance: \nloc:\n{all_new_positions[0, 0]}, \n\nrot: \n{all_new_rotations[0, 0]}')
    
    
def get_bone_mapping():
    '''
    IK controller returns three joints: root joint of the bone (1 joint) and the direction vector (2 joints)
    otherwise returns the single joint
    '''
    bone_mapping = {
        'left hand': [17, 16, 17],
        'right hand': [21, 20, 21],
        'left foot': [3, 3, 4],
        'right foot': [7, 7, 8],
        'left shoulder': 15,
        'right shoulder': 19,
        'left hip': 1,
        'right hip': 5,
        'head': 13
    }
    return bone_mapping


def get_euler_from_vector(vec):
    vec = vec / np.linalg.norm(vec)
    
    yaw = np.arctan2(vec[1], vec[0])
    pitch = np.arcsin(-vec[2])
    roll = 0
    
    return [yaw, pitch, roll]
    
    
def representation1(all_global_positions):
    '''
    Representation1:
    - One loc/rot for hands and feet as IK control. (4, 4)
    - Two shoulder and two hip endpoints, location only. (4, 0)
    - One head location (1, 0)
    Final structure: (7 location, 4 rotation)
    '''
    new_locations = 9
    new_rotations = 4
    all_new_positions = np.zeros((all_global_positions.shape[0], all_global_positions.shape[1], new_locations, 3))
    all_new_rotations = np.zeros((all_global_positions.shape[0], all_global_positions.shape[1], new_rotations, 3))
    bm = get_bone_mapping()
    
    for sequence in range(all_global_positions.shape[0]):
        for frame in range(all_global_positions.shape[1]):
            # hands
            #print(f"hands: {all_global_positions[sequence, frame, (bm['left hand'][0])]}")
            all_new_positions[sequence, frame, 0] = all_global_positions[sequence, frame, (bm['left hand'][0])]
            all_new_rotations[sequence, frame, 0] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['left hand'][2])] - all_global_positions[sequence, frame, (bm['left hand'][1])])
    
            all_new_positions[sequence, frame, 1] = all_global_positions[sequence, frame, (bm['right hand'][0])]
            all_new_rotations[sequence, frame, 1] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['right hand'][2])] - all_global_positions[sequence, frame, (bm['right hand'][1])])
    
            # feet
            all_new_positions[sequence, frame, 2] = all_global_positions[sequence, frame, (bm['left foot'][0])]
            all_new_rotations[sequence, frame, 2] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['left foot'][2])] - all_global_positions[sequence, frame, (bm['left foot'][1])])
    
            all_new_positions[sequence, frame, 3] = all_global_positions[sequence, frame, (bm['right foot'][0])]
            all_new_rotations[sequence, frame, 3] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['right foot'][2])] - all_global_positions[sequence, frame, (bm['right foot'][1])])
    
            # roots and head
            all_new_positions[sequence, frame, 4] = all_global_positions[sequence, frame, (bm['left shoulder'])]
            all_new_positions[sequence, frame, 5] = all_global_positions[sequence, frame, (bm['right shoulder'])]
            all_new_positions[sequence, frame, 6] = all_global_positions[sequence, frame, (bm['left hip'])]
            all_new_positions[sequence, frame, 7] = all_global_positions[sequence, frame, (bm['right hip'])]
            all_new_positions[sequence, frame, 8] = all_global_positions[sequence, frame, (bm['head'])]
            
    return all_new_positions, all_new_rotations
            
    

def conv_rig(positions, rotations, parents):
    # takes in original positions and rotations and returns the new representation of positions and rotations
    global_rot, global_pos = fk(rotations, positions, parents)
    # print(f"all pos {global_pos.shape}: {global_pos}\nall rot {global_rot.shape}: {global_rot}")
    return global_pos, global_rot

    
def fk(lrot, lpos, parents):
    """
    Calculate forward kinematics.

    Args:
        lrot (Tensor): Local rotation of joints. Shape: (..., joints, 3, 3)
        lpos (Tensor): Local position of joints. Shape: (..., joints, 3)
        parents (list of int or 1D int Tensor): Parent indices.

    Returns:
        Tensor, Tensor: (global rotation, global position).
            Shape: (..., joints, 3, 3), (..., joints, 3)
    """
    gr = [lrot[..., :1, :, :]]
    gp = [lpos[..., :1, :]]

    #print(f'{gr[0].shape}, {gp[0].shape}')

    for i in range(1, len(parents)):
        gr_parent = gr[parents[i]]
        gp_parent = gp[parents[i]]

        gr_i = np.matmul(gr_parent, lrot[..., i:i + 1, :, :])
        gp_i = gp_parent + \
            np.matmul(gr_parent, lpos[..., i:i + 1, :, None]).squeeze(-1)

        gr.append(gr_i)
        gp.append(gp_i)

    return np.concatenate(gr, axis=-3), np.concatenate(gp, axis=-2)


#create_object((1,0,0), 'test')
file_path = "C:\\Users\\eggyr\\OneDrive\\RPI\\S10\\Projects in ML\\final\\lafan1_detail_model_benchmark_5_0-2231.json"
main(file_path)
