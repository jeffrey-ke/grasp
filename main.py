from xarm.wrapper import XArmAPI
import numpy as np

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import os
import pdb

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

    xarm_controller = XArmController("192.168.1.241")
    # xarm_controller.move_ee_pose(np.array([400, 6.3, 150, 180 ,0, 0]))
    # object_height = 150
    # qpos = np.array([0.1060,  0.0244, -0.0842,  1.2666, -1.0907,  0.8831,  0.7941,  0.3491,
    #      0.1341,  1.2199,  1.1833, -0.0529,  0.2620,  1.2789,  1.2669, -0.4548,
    #      0.3866,  0.8570,  2.0940,  2.4430, -1.2000,  0.2983])
    # natural_leap_pose = np.array([3.3916316, 4.948622, 3.4729326, 2.5301, -1.2483,  2.2305,
    #                          3.4867384, 3.9684083, 3.342544, 4.713923, 3.3318062, 3.7383113,
    #                          4.402525, 3.2244277, 3.3854957, 3.5511656])
    # R_object_to_robot = np.array([[-1, 0,  0],
    #         [ 0, 0,  1],
    #         [ 0, 1,  0]])
    # xarm_controller = XArmController("192.168.1.248")
    leap_hand = LeapNode()
    initial_leap_pose = lhu.allegro_to_LEAPhand(np.zeros(16), zeros=False)

    leap_hand.interpolate_move(initial_leap_pose)
    time.sleep(5)
    # for i in range(1):
    #     # move to the START position for xarm and leap
    #     print(xarm_controller.arm.get_position()[1])
    #     xarm_controller.move_ee_pose(np.array([250, 6.3, object_height, 180, 0, 0]))
    #     xarm_controller.move_ee_pose(np.array([400, 6.3, object_height+150, 180, 0, 0]))
    #     leap_hand.interpolate_move(initial_leap_pose)
    #     time.sleep(1)

    #     # move to the START orientation
    #     orientation = R_object_to_robot @ qpos[3:6] * (180 / np.pi)
    #     xarm_controller.move_ee_pose(np.array([400, 6.3, object_height+150, orientation[0], orientation[1], orientation[2]]))
    #     time.sleep(1)

    #     # orientation = R_object_to_robot @ qpos[3:6] * (180 / np.pi)
    #     # xarm_controller.move_ee_pose(np.array([400, 6.3, -70, orientation[0], orientation[1], orientation[2]]))
    #     # time.sleep(1)

    #     # move to the OBJECT position
    #     # xarm_controller.move_ee_delta_pos(np.array([100, 0, 0, 0, 0, 0]))
    #     # time.sleep(1)

    #     # move to the OBJECT position
    #     delta_pos = R_object_to_robot @ qpos[:3] * 1000
    #     xarm_controller.move_ee_delta_pos(np.array([delta_pos[0]+100, delta_pos[1]+120, delta_pos[2], 0, 0, 0]))
    #     time.sleep(1)

    #     # # LEAP hand grasp the object
    #     real_pose = lhu.LEAPsim_to_LEAPhand(qpos[6:])
    #     leap_hand.interpolate_move(real_pose)
    #     time.sleep(5)

    #     # back to the leap hand neutral position
    #     leap_hand.interpolate_move(initial_leap_pose)
    #     time.sleep(1)
    #     # back to the START position and orientation
    #     xarm_controller.move_ee_pose(np.array([400, 6.3, object_height, 180, 0, 0]))
    #     time.sleep(2)

    #     leap_hand.interpolate_move(natural_leap_pose)
    #     time.sleep(2)

    #     xarm_controller.move_ee_pose(np.array([200, 6.3, -51, 180, 0, 0]))
    #     time.sleep(2)


    # try:
    #     import pygame
    # except:
    #     print("keyboard module not found. Please install it using 'pip install keyboard'.")
    #     exit(0)

    # # Initialize pygame
    # pygame.init()

    # # Create a small window (needed for capturing events)
    # screen = pygame.display.set_mode((200, 100))
    # pygame.display.set_caption("Key Listener")

    # print("Press keys (press 'q' to quit)...")

    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.KEYDOWN:
    #             key_name = pygame.key.name(event.key)
    #             print(f"You pressed: {key_name}")
    #             if key_name == 'q':
    #                 running = False
    #         elif event.type == pygame.QUIT:
    #             running = False

    # pygame.quit()

