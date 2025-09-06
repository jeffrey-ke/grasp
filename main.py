import os
import pdb
from argparse import ArgumentParser

import numpy as np
from xarm.wrapper import XArmAPI

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import grasp_loader

class XArmController:
    def __init__(self, ip, pose_dir="./xarm_poses"):
        self.arm = XArmAPI(ip)
        self.arm.connect()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.default_speed = 100
        self.pose_dir = pose_dir
        if not os.path.exists(pose_dir):
            os.makedirs(pose_dir)

    def save_current_qpos(self, qpos_file="current_qpos.npz"):
        """
        Saves the current position of the xarm to a npz file with key "qpos"
        """
        qpos = self.arm.get_position()[1]
        pose = self.arm.get_position()[1]
        r, p, y = pose[3:]
        x, y, z = pose[:3]
        ee_pose = np.array([x, y, z, r, p, y])
        np.savez(qpos_file, qpos=qpos, ee_pose=ee_pose)
        print(f"Saved current position: {qpos} to {qpos_file}")
        print(f"Saved current end effector pose: {ee_pose} to {qpos_file}")

    def load_current_qpos(self, qpos_file="current_qpos.npz", interpolate_steps=-1, delay=0.01, mode="qpos"):
        """
        Loads the current position of the xarm from a npz file with key "qpos"
        """
        data = np.load(qpos_file)
        if mode == "qpos":
            qpos = data["qpos"]
            self.move_qpos(qpos)

        elif mode == "ee_pose":
            ee_pose = data["ee_pose"]
            self.move_ee_pose(ee_pose)

    def get_forward_kinematics(self, qpos = None):
        """
        Get the forward kinematics of the robot.
        """
        if qpos is None:
            qpos = self.arm.get_joint_states()[1][0]
        pdb.set_trace()
        return self.arm.get_forward_kinematics(qpos)

    def move_qpos(self, qpos):
        """
        Move the robot to the specified joint positions.
        :param qpos: A list of joint positions.
        """
        if len(qpos) != 7:
            raise ValueError("qpos must be a list of 7 joint positions.")
        self.arm.set_joint_position(qpos)

    def move_ee_pos(self, ee_pos):
        current_pose = self.arm.get_position()[1]
        r,p,y = current_pose[3:]
        self.arm.set_position(ee_pos[0], ee_pos[1], ee_pos[2], r, p, y, wait=True)

    def move_ee_delta_pos(self, ee_delta_pos):
        current_pose = self.arm.get_position()[1]
        x, y, z = current_pose[:3]
        r, p, y = current_pose[3:]
        self.arm.set_position(x + ee_delta_pos[0], y + ee_delta_pos[1], z + ee_delta_pos[2], r, p, y, wait=True)

    def move_ee_pose(self, ee_pose):
        """
        Move the end effector to the specified pose.
        :param ee_pose: A list of 6 elements representing [x, y, z, roll, pitch, yaw].
        """
        if len(ee_pose) != 6:
            raise ValueError("ee_pose must be a list of 6 elements.")
        self.arm.set_position(ee_pose[0], ee_pose[1], ee_pose[2], ee_pose[3], ee_pose[4], ee_pose[5], wait=True) #, is_radian=True

    def disconnect(self):
        self.arm.disconnect()


class LeapNode:
    finger_mapping = {
        1: [0,1,2,3],
        2: [4,5,6,7],
        3: [8,9,10,11],
        0: [12,13,14,15]
    }
    def __init__(self, pose_dir="./leap_poses"):
        ####Some parameters
        # I recommend you keep the current limit from 350 for the lite, and 550 for the full hand
        # Increase KP if the hand is too weak, decrease if it's jittery.
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        # For example ls /dev/serial/by-id/* to find your LEAP Hand. Then use the result.
        # For example: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7W91VW-if00-port0
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 1000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 1000000)
                self.dxl_client.connect()
            except Exception:
                try:
                    self.dxl_client = DynamixelClient(motors, '/dev/tty.usbserial-FT94W5IF', 1000000)
                    self.dxl_client.connect()
                except Exception:
                    print("❌ Failed to connect to any available serial port")
                    print("Available ports: /dev/ttyUSB0, /dev/ttyUSB1, /dev/tty.usbserial-*")
                    raise RuntimeError("No available serial port for LEAP Hand connection")
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        # self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        self.pose_dir = pose_dir
        if not os.path.exists(pose_dir):
            os.makedirs(pose_dir)
    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #allegro compatibility joint angles.  It adds 180 to make the fully open position at 0 instead of 180
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility for policies, it assumes the ranges are [-1,1] and then convert to leap hand ranges.
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position of the robot
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()
    #These combined commands are faster FYI and return a list of data
    def pos_vel(self):
        return self.dxl_client.read_pos_vel()
    #These combined commands are faster FYI and return a list of data
    def pos_vel_eff_srv(self):
        return self.dxl_client.read_pos_vel_cur()

    def interpolate_move(self, target_qpos, num_steps=100, delay=0.01):
        """
        Interpolates the movement from the current position to the target position in num_steps steps.
        """
        current_qpos = self.read_pos()
        for step in range(num_steps):
            interpolated_qpos = current_qpos + (target_qpos - current_qpos) * (step / num_steps)
            self.set_leap(interpolated_qpos)
            time.sleep(delay)

    def save_current_qpos(self, qpos_file="current_qpos.npz"):
        """
        Saves the current position of the leap node to a npz file with key "qpos"
        """
        qpos = self.read_pos()
        np.savez(qpos_file, qpos=qpos)
        print(f"Saved current position: {qpos} to {qpos_file}")

    def load_current_qpos(self, qpos_file="current_qpos.npz", interpolate_steps=-1, delay=0.01):
        """
        Loads the current position of the leap node from a npz file with key "qpos"
        """
        data = np.load(qpos_file)
        qpos = data["qpos"]
        if interpolate_steps > 0:
            self.interpolate_move(qpos, num_steps=interpolate_steps, delay=delay)
        else:
            self.set_leap(qpos)

    def reset(self):
        """
        Resets the robot to a neutral position.
        """
        neutral_qpos_leap = lhu.allegro_to_LEAPhand(np.zeros(16))
        self.interpolate_move(neutral_qpos_leap, num_steps=100, delay=0.01)

    def disable_finger_torque(self, indices=[]):
        if len(indices) == 0:
            indices = list(self.finger_mapping.keys())
        for finger_index in indices:
            if finger_index in self.finger_mapping:
                self.dxl_client.set_torque_enabled(self.finger_mapping[finger_index], False)
            else:
                print(f"Invalid finger index: {finger_index}")

    def enable_finger_torque(self, indices=[]):
        if len(indices) == 0:
            indices = list(self.finger_mapping.keys())
        for finger_index in indices:
            if finger_index in self.finger_mapping:
                self.dxl_client.set_torque_enabled(self.finger_mapping[finger_index], True)
            else:
                print(f"Invalid finger index: {finger_index}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--result-folder", type=str, default="results")
    args = parser.parse_args()

    xarm_controller = XArmController("192.168.1.241")
    leap_hand = LeapNode()
    initial_leap_pose = lhu.allegro_to_LEAPhand(np.zeros(16), zeros=False)
    leap_hand.interpolate_move(initial_leap_pose)
    time.sleep(5)

    experiment_id = input("Input name of the folder with the grasps")
    results_path = Path.cwd() / "results" / f"{experiment_id}"
    for grasp in grasp_loader.read_grasps_from(Path.cwd() / "grasps" / f"{experiment_id}")
        input(f"Hit Enter when you have placed {grasp.object_name} at location [x,y,z] {grasp.object_xyz} meters from arm base")
        move the arm to align the hand frame to what's specified in object_rpy
        leap_hand.interpolate_move(grasp.leap_qpos)
