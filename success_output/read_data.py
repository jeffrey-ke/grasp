import os
import sys
import warnings
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def main(folder_path):
    files = []
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
            print(f"Example {i+1}:")
            q = q.cpu().numpy().tolist()
            print(' base xyz', q[:3])
            print(' base rpy', q[3:6])
            print(' joint angles', q[6:])
            print('---')    
    
if __name__ == "__main__":
    main(os.path.join(ROOT_DIR, 'leaphand'))
