import numpy as np
import os
import json
import sys

def read_input(file_path):
    with open(file_path) as json_data:
        d = json.loads(json_data)

    return d['positions'], d['rotations'], d['parents'], d['foot_contact']


def conv_rig(positions):


def get_bone(positions, bone):


def (positions):
