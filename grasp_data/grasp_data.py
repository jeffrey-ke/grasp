from dataclasses import dataclass

@dataclass
class GraspInfo:
    object_xyz: list[float]
    object_rpy: list[float]
    leap_qpos: list[float]
    object_name: str
    robot_name: str
    grasp_id: int
