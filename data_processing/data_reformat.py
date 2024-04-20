import numpy as np
import os
import json
import torch
import math
import time

from scipy.spatial.transform import Rotation as R


def read_input(file_path):
    with open(file_path) as json_data:
        d = json.load(json_data)

    return d['positions'], d['rotations'], d['parents'], d['foot_contact']
        
def swap_coordinate_axes(positions, one, two):
    new_pos = positions.copy()
    new_pos[..., [one, two]] = positions[..., [two, one]]
    return new_pos

def flip_coordinate_axis(positions, axis):
    positions[..., axis] *= -1
    return positions

def to_blend_coords(positions):
    positions = swap_coordinate_axes(positions, 1, 2)
    positions = swap_coordinate_axes(positions, 0, 1)
    positions = flip_coordinate_axis(positions, 1)
    positions = positions * 0.01
    return positions
    
def from_blend_coords(positions):
    positions = flip_coordinate_axis(positions, 1)
    positions = swap_coordinate_axes(positions, 0, 1)
    positions = swap_coordinate_axes(positions, 1, 2)
    positions = positions * 100
    return positions

def get_euler_from_vector(vec, keep_length = False):
    length = np.linalg.norm(vec)
    vec = vec / length
    
    MAX_LENGTH = 2
    
    yaw = np.arctan2(vec[1], vec[0])
    pitch = np.arcsin(-vec[2])
    roll = 0

    if keep_length:
        roll = length * (360 / MAX_LENGTH)
    
    return [yaw, pitch, roll]

def get_vec_from_euler(angle):
    MAX_LENGTH = 2
    length = angle[2] / (360 / MAX_LENGTH)
    
    if length == 0:
        length = 1
        
    y = np.cos(angle[1]) * np.sin(angle[0])
    z = np.sin(angle[1])
    x = np.cos(angle[1]) * np.cos(angle[0])
    
    x *= length
    y *= length
    z *= -length
    
    return np.array([x, y, z])


def make_converted_json(file_path, save_path):
    # reading all data
    all_orig_positions, all_orig_rotations, parents, all_foot_contact = read_input(file_path)
    all_orig_positions = np.array(all_orig_positions)
    all_orig_rotations = np.array(all_orig_rotations)
    print(f'FINSIHED IMPORTING DATA \n\noriginal positions shape: {all_orig_positions.shape}')

    all_global_positions, all_global_rotations = conv_rig(all_orig_positions, all_orig_rotations, parents)
    all_global_positions = to_blend_coords(all_global_positions)
    #visualize(all_global_positions, 0)
    
    print(f'CONVERTED ORIGINAL DATA TO GLOBAL \n\nsample data: {all_orig_positions[0,0]}')
    
    # creating new representation
    rep1 = representation1(all_global_positions)
    print(f'CONVERTED TO CUSTOM REPRESENTATION \n\nrep1 first instance: \nrot: \n{rep1[0, 0]}')
    
    rep1list = rep1.tolist()
    with open(save_path, 'w') as file:
        json.dump(rep1list, file)
    
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
        'head': 13,
        'root': 0
    }
    return bone_mapping

    
def representation1(all_global_positions):
    '''
    Representation1:
    - One loc/rot for hands and feet as IK control. (4, 4)
    - Two shoulder and two hip endpoints, location only. (4, 0)
    - One head location (1, 0)
    Final structure: (9 location, 4 rotation)
    condensed into 13 rotations
    '''
    MAX_REACH = 150
    NEW_ROTATIONS = 13
    INDEXES_TO_USE = 22
    #all_new_positions = np.zeros((all_global_positions.shape[0], all_global_positions.shape[1], new_locations, 3))
    all_new_rotations = np.zeros((all_global_positions.shape[0], all_global_positions.shape[1], INDEXES_TO_USE, 3))
    bm = get_bone_mapping()
    
    for sequence in range(all_global_positions.shape[0]):
        for frame in range(all_global_positions.shape[1]):
            # hands
            #print(f"hands: {all_global_positions[sequence, frame, (bm['left hand'][0])]}")
            all_new_rotations[sequence, frame, 0] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['left hand'][0])] - all_global_positions[sequence, frame, (bm['root'])], True)
            all_new_rotations[sequence, frame, 1] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['left hand'][2])] - all_global_positions[sequence, frame, (bm['left hand'][1])])
    
            all_new_rotations[sequence, frame, 2] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['right hand'][0])] - all_global_positions[sequence, frame, (bm['root'])], True)
            all_new_rotations[sequence, frame, 3] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['right hand'][2])] - all_global_positions[sequence, frame, (bm['right hand'][1])])
    
            # feet
            all_new_rotations[sequence, frame, 4] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['left foot'][0])] - all_global_positions[sequence, frame, (bm['root'])], True)
            all_new_rotations[sequence, frame, 5] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['left foot'][2])] - all_global_positions[sequence, frame, (bm['left foot'][1])])
    
            all_new_rotations[sequence, frame, 6] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['right foot'][0])] - all_global_positions[sequence, frame, (bm['root'])], True)
            all_new_rotations[sequence, frame, 7] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['right foot'][2])] - all_global_positions[sequence, frame, (bm['right foot'][1])])
    
            # joints and root
            all_new_rotations[sequence, frame, 8] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['left shoulder'])] - all_global_positions[sequence, frame, (bm['root'])], True)
            all_new_rotations[sequence, frame, 9] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['right shoulder'])] - all_global_positions[sequence, frame, (bm['root'])], True)
            all_new_rotations[sequence, frame, 10] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['left hip'])] - all_global_positions[sequence, frame, (bm['root'])], True)
            all_new_rotations[sequence, frame, 11] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['right hip'])] - all_global_positions[sequence, frame, (bm['root'])], True)
            all_new_rotations[sequence, frame, 12] = get_euler_from_vector(all_global_positions[sequence, frame, (bm['root'])], True)
            
    return all_new_rotations
    

def conv_rig(positions, rotations, parents):
    # takes in original positions and rotations and returns the new representation of positions and rotations
    global_rot, global_pos = fk(rotations, positions, parents)
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


file_name = "lafan1_detail_model_benchmark_5_0-2231.json"
save_name = "CONVERTED_lafan1_detail_model_benchmark_5_0-2231.json"
file_path = "C:\\Users\\eggyr\\OneDrive\\RPI\\S10\\Projects in ML\\final\\"
make_converted_json(file_path + file_name, file_path + save_name)
