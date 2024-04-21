import numpy as np
import json

MAX_LENGTH = 2

"""
FILE IO
"""
def read_input(file_path):
    with open(file_path) as json_data:
        d = json.load(json_data)

    return d['positions'], d['rotations'], d['parents'], d['foot_contact']


"""
COORDINATE AND VECTOR MANIPULATION
"""
        
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
    #positions = flip_coordinate_axis(positions, 1)
    positions = positions * 0.01
    return positions
    
    
def from_blend_coords(positions):
    #positions = flip_coordinate_axis(positions, 1)
    positions = swap_coordinate_axes(positions, 0, 1)
    positions = swap_coordinate_axes(positions, 1, 2)
    positions = positions * 100
    return positions

def get_euler_from_vec_vectorized(vec, keep_length = False):
    length = np.linalg.norm(vec, axis=-1)
    vec = vec / length[..., None]

    yaw = np.arctan2(vec[..., 1], vec[..., 0])
    pitch = np.arcsin(-vec[..., 2])
    roll = np.zeros_like(yaw)

    if keep_length:
        roll = length * (6 / MAX_LENGTH)

    return np.stack([yaw, pitch, roll], axis=-1)


def get_vec_from_euler(angle):
    length = angle[2] / (6 / MAX_LENGTH)
    
    if length == 0:
        length = 1
        
    y = np.cos(angle[1]) * np.sin(angle[0])
    z = np.sin(angle[1])
    x = np.cos(angle[1]) * np.cos(angle[0])
    
    x *= length
    y *= length
    z *= -length
    
    return np.array([x, y, z])

def get_vec_from_euler_vectorized(angle):
    '''
    angle: (..., 3)
    '''
    length = angle[..., 2] / (6 / MAX_LENGTH)
    
    mask = length == 0
    length[mask] = 1
        
    y = np.cos(angle[..., 1]) * np.sin(angle[..., 0])
    z = np.sin(angle[..., 1])
    x = np.cos(angle[..., 1]) * np.cos(angle[..., 0])
    
    x *= length
    y *= length
    z *= -length
    
    return np.stack([x, y, z], axis=-1)


"""
HELPERS
"""

def to_global(positions, rotations, parents):
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

    for i in range(1, len(parents)):
        gr_parent = gr[parents[i]]
        gp_parent = gp[parents[i]]

        gr_i = np.matmul(gr_parent, lrot[..., i:i + 1, :, :])
        gp_i = gp_parent + \
            np.matmul(gr_parent, lpos[..., i:i + 1, :, None]).squeeze(-1)

        gr.append(gr_i)
        gp.append(gp_i)

    return np.concatenate(gr, axis=-3), np.concatenate(gp, axis=-2)

def get_bone_mapping():
    '''
    IK controller returns three joints: root joint of the bone (1 joint) and the direction vector (2 joints)
    otherwise returns the single joint
    '''
    bone_mapping = {
        'left hand': 17,
        'left elbow': 16,
        'right hand': 21,
        'right elbow': 20,
        'left foot': 3,
        'left knee': 2,
        'left toe': 4,
        'right foot': 7,
        'right knee': 6,
        'right toe': 8,
        'left shoulder': 15,
        'right shoulder': 19,
        'left hip': 1,
        'right hip': 5,
        'head': 13,
        'spine top': 12,
        'root': 9
    }
    return bone_mapping

def get_representation1_mapping():
    rm1 = {
        'left hand': 0,
        'left elbow': 1,
        'right hand': 2,
        'right elbow': 3,
        'left foot': 4,
        'left knee': 5,
        'right foot': 6,
        'right knee': 7,
        'head': 8,
        'spine top': 9,
        'root': 10,
        'root rotation': 11
    }
    return rm1


def make_converted_json(file_path, save_path):
    # reading all data
    all_orig_positions, all_orig_rotations, parents, all_foot_contact = read_input(file_path)
    all_orig_positions = np.array(all_orig_positions)
    all_orig_rotations = np.array(all_orig_rotations)
    print(f'FINSIHED IMPORTING DATA \n\noriginal positions shape: {all_orig_positions.shape}')

    all_global_positions, all_global_rotations = to_global(all_orig_positions, all_orig_rotations, parents)
    all_global_positions = to_blend_coords(all_global_positions)
    #visualize(all_global_positions, 0)
    
    print(f'CONVERTED ORIGINAL DATA TO GLOBAL \n\nsample data: {all_orig_positions[0,0]}')
    
    # creating new representation
    rep1 = representation1(all_global_positions, all_global_rotations)
    print(f'CONVERTED TO CUSTOM REPRESENTATION \n\nrep1 first instance: \nrot: \n{rep1[0, 0]}')
    
    rep1list = rep1.tolist()
    with open(save_path, 'w') as file:
        json.dump(rep1list, file)

    partial_back = representation1_backwards_partial(rep1)
    mask = representation1_partial_mask()
    original_masked = all_global_positions * mask[:, np.newaxis]
    total_error = np.sum(np.abs(original_masked - partial_back))
    print(f'total error: {total_error}')



"""
MAIN CONVERSION MODULES
"""

def representation1(rep0, root_angles):
    '''
    rep0 represents the original global positions of all joints

    Representation1:
    - One loc/rot for hands and feet as IK control. (4, 4)
    - Two shoulder and two hip endpoints, location only. (4, 0)
    - One head location (1, 0)
    Final structure: (9 location, 4 rotation)
    condensed into 13 rotations
    '''
    INDICES_IN_USE = 22
    rep1 = np.zeros((rep0.shape[0], rep0.shape[1], INDICES_IN_USE, 3))
    bm = get_bone_mapping()
    rm1 = get_representation1_mapping()

    root_location = rep0[:, :, bm['root']]

    # hands
    rep1[:,:, rm1['left hand']] = get_euler_from_vec_vectorized(rep0[:,:, bm['left hand']] - root_location, True)
    rep1[:,:, rm1['left elbow']] = get_euler_from_vec_vectorized(3 * rep0[:,:, bm['left elbow']] - rep0[:,:, bm['left hand']] - rep0[:,:, bm['left shoulder']] - root_location, True)

    rep1[:,:, rm1['right hand']] = get_euler_from_vec_vectorized(rep0[:,:, bm['right hand']] - root_location, True)
    rep1[:,:, rm1['right elbow']] = get_euler_from_vec_vectorized(3 * rep0[:,:, bm['right elbow']] - rep0[:,:, bm['right hand']] - rep0[:,:, bm['right shoulder']] - root_location, True)

    # feet
    rep1[:,:, rm1['left foot']] = get_euler_from_vec_vectorized(rep0[:,:, bm['left foot']] - root_location, True)
    rep1[:,:, rm1['left knee']] = get_euler_from_vec_vectorized(3 * rep0[:,:, bm['left knee']] - rep0[:,:, bm['left foot']] - rep0[:,:, bm['left hip']] - root_location, True)

    rep1[:,:, rm1['right foot']] = get_euler_from_vec_vectorized(rep0[:,:, bm['right foot']] - root_location, True)
    rep1[:,:, rm1['right knee']] = get_euler_from_vec_vectorized(3 * rep0[:,:, bm['right knee']] - rep0[:,:, bm['right foot']] - rep0[:,:, bm['right hip']] - root_location, True)

    # joints and root
    rep1[:,:, rm1['spine top']] = get_euler_from_vec_vectorized(rep0[:,:, bm['spine top']] - root_location, True)
    rep1[:,:, rm1['head']] = get_euler_from_vec_vectorized(rep0[:,:, bm['head']] - root_location, True)
    rep1[:,:, rm1['root']] = get_euler_from_vec_vectorized(root_location, True)
    rep1[:,:, rm1['root rotation']] = m9dtoeuler(root_angles)
            
    return rep1


def m9dtoeuler(m9d):
     pitch = -1*np.arcsin(m9d[:, :, 0, 2, 0])
     roll = np.arctan2( m9d[:, :, 0, 2, 1] / np.cos(pitch) , m9d[:, :, 0, 2, 2] / np.cos(pitch))
     yaw = np.arctan2( m9d[:, :, 0, 1, 0] / np.cos(pitch) , m9d[:, :, 0, 0, 0] / np.cos(pitch))
     return np.array(np.stack((yaw, pitch, roll), axis=-1))

    
def representation1_backwards_partial(rep1):
    INDICES_IN_USE = 22
    rep0 = np.zeros((rep1.shape[0], rep1.shape[1], INDICES_IN_USE, 3))
    bm = get_bone_mapping()
    rm1 = get_representation1_mapping()
    
    root_location = get_vec_from_euler_vectorized(rep1[:, :, rm1['root']])

    #hands
    rep0[:, :, bm['left hand']] = get_vec_from_euler_vectorized(rep1[:, :, rm1['left hand']]) + root_location
    rep0[:, :, bm['right hand']] = get_vec_from_euler_vectorized(rep1[:, :, rm1['right hand']]) + root_location

    rep0[:, :, bm['left foot']] = get_vec_from_euler_vectorized(rep1[:, :, rm1['left foot']]) + root_location
    rep0[:, :, bm['right foot']] = get_vec_from_euler_vectorized(rep1[:, :, rm1['right foot']]) + root_location

    rep0[:, :, bm['head']] = get_vec_from_euler_vectorized(rep1[:, :, rm1['head']]) + root_location
    rep0[:, :, bm['spine top']] = get_vec_from_euler_vectorized(rep1[:, :, rm1['spine top']]) + root_location
    rep0[:, :, bm['root']] = root_location
            
    return rep0


def representation1_partial_mask():
    bm = get_bone_mapping()
    mask = np.zeros((22,))
    mask[bm['left hand']] = 1
    mask[bm['right hand']] = 1
    mask[bm['left foot']] = 1
    mask[bm['right foot']] = 1
    mask[bm['head']] = 1
    mask[bm['spine top']] = 1
    mask[bm['root']] = 1
    return mask


"""
def representation1backwards(rep1, keyframe):
    INDICES_IN_USE = 22
    rep0 = np.zeros((rep1.shape[0], rep1.shape[1], INDICES_IN_USE, 3))
    bm = get_bone_mapping()
    rm1 = get_representation1_mapping()
    
    for sequence in range(rep1.shape[0]):
        for frame in range(rep1.shape[1]):
            root_location = get_vec_from_euler(rep1[sequence, frame, rm1['root']])
            
            # hands
            left_hand = bpy.data.objects[f"ctrl_arm.l"]
            left_hand.location = get_vec_from_euler(rep1[sequence, frame, rm1['left hand']]) + root_location
            left_elbow = bpy.data.objects["ctrl_elbow.l"]
            left_elbow.location = get_vec_from_euler(rep1[sequence, frame, rm1['left elbow']]) + root_location

            right_hand = bpy.data.objects[f"ctrl_arm.r"]
            right_hand.location = get_vec_from_euler(rep1[sequence, frame, rm1['right hand']]) + root_location
            right_elbow = bpy.data.objects["ctrl_elbow.r"]
            right_elbow.location = get_vec_from_euler(rep1[sequence, frame, rm1['right elbow']]) + root_location
            
            # feet
            left_foot = bpy.data.objects[f"ctrl_leg.l"]
            left_foot.location = get_vec_from_euler(rep1[sequence, frame, rm1['left foot']]) + root_location
            left_knee = bpy.data.objects["ctrl_knee.l"]
            left_knee.location = get_vec_from_euler(rep1[sequence, frame, rm1['left knee']]) + root_location
            
            right_foot = bpy.data.objects[f"ctrl_leg.r"]
            right_foot.location = get_vec_from_euler(rep1[sequence, frame, rm1['right foot']]) + root_location
            right_knee = bpy.data.objects["ctrl_knee.r"]
            right_knee.location = get_vec_from_euler(rep1[sequence, frame, rm1['right knee']]) + root_location
            
            # joints
            head = bpy.data.objects[f"ctrl_head"]
            head.location = get_vec_from_euler(rep1[sequence, frame, rm1['head']]) + root_location

            spine_top = bpy.data.objects[f"ctrl_spine_top"]
            spine_top.location = get_vec_from_euler(rep1[sequence, frame, rm1['spine top']]) + root_location

            root_joint = bpy.data.objects[f"root"]
            root_joint.location = root_location
            root_joint.rotation_euler = Euler(rep1[sequence, frame, rm1['root rotation']], 'XYZ')
            
            bpy.context.view_layer.update()
            
            # store the information back into global positions
            for joint in range(rep1.shape[2]):
                rep0[sequence, frame, joint] = bpy.data.objects[f"solved_joint.{joint}"].location
            
            if keyframe != -1 and sequence == keyframe:
                print(f'creating keyframes for sequence {sequence}!')
                left_hand.keyframe_insert(data_path="location", frame=frame)
                left_elbow.keyframe_insert(data_path="location", frame=frame)
                right_hand.keyframe_insert(data_path="location", frame=frame)
                right_elbow.keyframe_insert(data_path="location", frame=frame)
                left_foot.keyframe_insert(data_path="location", frame=frame)
                left_knee.keyframe_insert(data_path="location", frame=frame)
                right_foot.keyframe_insert(data_path="location", frame=frame)
                right_knee.keyframe_insert(data_path="location", frame=frame)
                head.keyframe_insert(data_path="location", frame=frame)
                spine_top.keyframe_insert(data_path="location", frame=frame)
                root_joint.keyframe_insert(data_path="location", frame=frame)
                root_joint.keyframe_insert(data_path="rotation_euler", frame=frame)
            
    return rep0
"""


file_name = "lafan1_detail_model_benchmark_5_0-2231.json"
save_name = "CONVERTED_lafan1_detail_model_benchmark_5_0-2231.json"
file_path = "..\\..\\..\\final\\"
make_converted_json(file_path + file_name, file_path + save_name)

