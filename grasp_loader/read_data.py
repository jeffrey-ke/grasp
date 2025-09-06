import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from grasp_data import GraspInfo

def read_grasps_from(folder_path):
    files = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"{folder_path} does not exist!")
    else:
        folder_path = Path.cwd() / folder_path / id
    for item in os.listdir(folder_path):
        file_path = os.path.join(folder_path, item)
        if os.path.isfile(file_path):
            files.append(file_path)
    
    for f in files:
        data = torch.load(f)
        robot_name = data['robot_name']
        object_name = data['object_name']
        hand_config = data['succ_q']
        print(f"Robot: {robot_name}, Object: {object_name}")
        for i, q in enumerate(hand_config):
            print(f"Loaded example {i+1}:")
            q = q.cpu().numpy().tolist()
            yield GraspInfo(
                    object_xyz=q[:3],
                    object_rpy=q[3:6],
                    leap_qpos=q[6:],
                    object_name=object_name,
                    robot_name=robot_name,
                    grasp_id=i,
            )
