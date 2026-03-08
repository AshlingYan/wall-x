import sys
sys.path.append("./")

import numpy as np

from my_robot.base_robot import Robot

from controller.Piper_controller import PiperController
from sensor.Realsense_sensor import RealsenseSensor

from data.collect_any import CollectAny

# setting your realsense serial
# 根据实际安装位置分配序列号（与你的 realsense_serial.py 输出一致）:
# - head: 头部/顶部相机，俯视整个工作区 (device 1)
# - right_wrist: 右臂腕部相机 (can0) (device 2)
# - left_wrist: 左臂腕部相机 (can1) (device 3)


# # 旧机械臂底座使用以下相机序列号
# CAMERA_SERIALS = {
#     'head': '152122072933',        # device 1: 头部相机
#     'right_wrist': '134222070539', # device 2: 右腕相机
#     'left_wrist': '134222070313',  # device 3: 左腕相机
# }

# 新机械臂底座使用以下相机序列号
CAMERA_SERIALS = {
    'head': '152122072933',        # device 1: 头部相机
    'right_wrist': '152122076290', # device 2: 右腕相机
    'left_wrist': '213722070453',  # device 3: 左腕相机
}

# Define start position (in degrees)
START_POSITION_ANGLE_LEFT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
]

# Define start position (in degrees)
START_POSITION_ANGLE_RIGHT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
]

condition = {
    "save_path": "./save/",
    "task_name": "test",
    "save_format": "hdf5",
    "save_freq": 10, 
}

class PiperDual(Robot):
    def __init__(self, condition=condition, move_check=True, start_episode=0):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.controllers = {
            "arm":{
                "left_arm": PiperController("left_arm"),
                "right_arm": PiperController("right_arm"),
            }
        }
        self.sensors = {
            "image": {
                "cam_head": RealsenseSensor("cam_head"),
                "cam_left_wrist": RealsenseSensor("cam_left_wrist"),
                "cam_right_wrist": RealsenseSensor("cam_right_wrist"),
            },
        }

    def reset(self):
        self.controllers["arm"]["left_arm"].reset(START_POSITION_ANGLE_LEFT_ARM)
        self.controllers["arm"]["right_arm"].reset(START_POSITION_ANGLE_RIGHT_ARM)

    def set_up(self):
        super().set_up()
        # can0 连接右臂，can1 连接左臂
        self.controllers["arm"]["left_arm"].set_up("can1")
        self.controllers["arm"]["right_arm"].set_up("can0")

        self.sensors["image"]["cam_head"].set_up(CAMERA_SERIALS['head'], is_depth=False)
        self.sensors["image"]["cam_left_wrist"].set_up(CAMERA_SERIALS['left_wrist'], is_depth=False)
        self.sensors["image"]["cam_right_wrist"].set_up(CAMERA_SERIALS['right_wrist'], is_depth=False)

        self.set_collect_type({"arm": ["joint","qpos","gripper"],
                               "image": ["color"]
                               })
        print("set up success!")

if __name__ == "__main__":
    import time
    
    robot = PiperDual()
    robot.set_up()

    # collection test
    data_list = []
    for i in range(100):
        print(i)
        data = robot.get()
        robot.collect(data)
        time.sleep(0.1)
    robot.finish()
    
    # moving test
    move_data = {
        "arm":{
            "left_arm":{
            "qpos":[0.057, 0.0, 0.216, 0.0, 0.085, 0.0, 0.057, 0.0, 0.216, 0.0, 0.085, 0.0],
            "gripper":0.2,
            },
        }
    }
    robot.move(move_data)
    
    move_data = {
        "arm":{
            "left_arm":{
            "qpos":[0.060, 0.0, 0.260, 0.0, 0.085, 0.0, 0.060, 0.0, 0.260, 0.0, 0.085, 0.0],
            "gripper":0.2,
            },
        }
    }
    robot.move(move_data)
