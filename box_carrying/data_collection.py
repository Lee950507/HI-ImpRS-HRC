#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import transformation as tsf
import scipy.linalg as linalg

import message_filters
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion, Pose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int8, String, Bool, Float64MultiArray
from sensor_msgs.msg import JointState
from EMGProcessor import EMGProcessor

import sys
import os
import rospy
import signal
import subprocess
import time
import queue
import threading


# 获取 ROS 工作空间的路径
workspace_path = '/home/clover/catkin_ws'

# 添加编译后的库路径
sys.path.append(os.path.join(workspace_path, 'devel', 'lib'))


def vrpn_launch_roslaunch():
    launch_file = "~/catkin_ws/src/vrpn_client_ros/launch/sample.launch"  # 替换为你的 launch 文件路径
    # 启动 roslaunch
    command = f"roslaunch {launch_file} server:=192.168.10.7"
    return subprocess.Popen(command, shell=True)


def signal_handler(sig, frame):
    print('Python shutdown signal received...')
    rospy.signal_shutdown("shutdown by manual")  # 标记节点为关闭
    # 终止 roslaunch_process
    if 'roslaunch_process' in locals():
        print('Shutdown roslaunch process.')
        roslaunch_process.terminate()
        roslaunch_process.wait()  # 等待进程终止
    print('Python shutdown.')
    sys.exit(0)


def transform_to_pose(pose_stamped):
    return np.array([
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
        pose_stamped.pose.orientation.x,
        pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z,
        pose_stamped.pose.orientation.w
    ])


def transform_to_joint(joint_state):
    joint = [joint_state.position[0], joint_state.position[1], joint_state.position[2]]
    current_time = joint_state.header.stamp.secs + 1e-9 * joint_state.header.stamp.nsecs
    return np.array(joint), np.array(current_time)


def convert_to_pose_stamped(pose, frame_id, stamp):
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    pose_stamped.header.stamp = stamp
    pose_stamped.pose.position.x = pose[0]
    pose_stamped.pose.position.y = pose[1]
    pose_stamped.pose.position.z = pose[2]
    # q = tf.transformations.quaternion_from_euler(pose[3], pose[4], pose[5])
    # pose_stamped.pose.orientation = Quaternion(*q)
    pose_stamped.pose.orientation.x = pose[3]
    pose_stamped.pose.orientation.y = pose[4]
    pose_stamped.pose.orientation.z = pose[5]
    pose_stamped.pose.orientation.w = pose[6]
    return pose_stamped


def multi_callback(object_data, muscle_coactivation, start_time):
    global sub_object_all
    global sub_object_time
    global muscle_coactivation_all
    global last_index
    sub_object = transform_to_pose(object_data)
    # print("sub_object:", sub_object)

    # # 第一次调用时设置初始时间
    # if index_counter == 0:
    #     index = 0
    # else:
    #     index = int((time_array[index_counter] - time_array[0]) * 1000)

    try:
        muscle_coactivation = np.asarray(muscle_coactivation)
        index = muscle_coactivation.shape[1]
        if index != last_index:
            sub_object_all.append(sub_object)
            sub_object_time.append(time.time() - start_time)
            print(muscle_coactivation.shape)
            muscle_coactivation_all.append(muscle_coactivation[:, -1])
            last_index = index
            # 控制循环频率
            time.sleep(0.001)  # 1kHz控制频率
    except Exception as e:
        pass
        # print(f"Error occurred: {e}")
        # print('Muscle coactivation not available')
        # print(len(muscle_coactivation[0]))
        # print(len(muscle_coactivation[1]))
        # print(len(muscle_coactivation[2]))
        # print(len(muscle_coactivation[3]))


    # print(f"Index: {index}")
    # print(f"coactivation: {muscle_coactivation[-1]}")



if __name__ == '__main__':
    global sub_object_all
    global sub_object_time
    global muscle_coactivation_all
    global last_index

    folder = '/home/clover/Chenzui/HI-ImpRS-HRC/data/emg_record/taichi/chenzui_vs_yiming/1.1'
    os.makedirs(folder, exist_ok=True)
    rospy.init_node('data_collection')
    signal.signal(signal.SIGINT, signal_handler)

    roslaunch_process = vrpn_launch_roslaunch()
    time.sleep(1)

    last_index = 0

    # # 准备控制循环所需变量
    # index_counter = 0
    # time_array = np.zeros(100000)
    #
    # # 创建躯干数据订阅器
    # torso_data = None

    start_time = time.time()
    emg_processor = EMGProcessor(channel_num=4, sample_fre=200, start_time=start_time, save=True, save_folder=folder)
    data_queue = queue.Queue()
    threads = [
        threading.Thread(
            target=emg_processor.read_emg,
            args=(data_queue,),
            name="EMG-Reader"
        ),
        threading.Thread(
            target=emg_processor.process_emg,
            args=(data_queue,),
            name="EMG-Processor"
        )
    ]
    for t in threads:
        t.daemon = True
        t.start()
    time.sleep(5.0)

    object_data = None

    def object_callback(msg):
        global object_data
        object_data = msg

    object_subscriber = rospy.Subscriber('/vrpn_client_node/object/pose', PoseStamped, object_callback)
    sub_object_all, sub_object_time, muscle_coactivation_all = [], [], []

    try:
        print("Starting data collection...")
        while not rospy.is_shutdown():
            while object_data is None:
                rospy.loginfo_throttle(1, "Waiting for object data...")
                time.sleep(0.01)
                continue
            while len(emg_processor.all_emg_data[0]) == 0:
                rospy.loginfo_throttle(1, "Waiting for EMG data...")
                time.sleep(0.01)
                continue
            # print(emg_processor.all_emg_data)

            multi_callback(
                object_data,
                emg_processor.all_emg_data,
                start_time,
            )

        # # 控制循环频率
        # time.sleep(0.001)  # 1kHz控制频率

        print("Execution finished.")
        emg_processor.read_emg_flag = False

        # 保持程序运行，等待中断信号
        while not rospy.is_shutdown():
            interrupt = False
            time.sleep(1)

            data_queue.join()
            for t in threads:
                t.join()


    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # 清理资源
        if 'roslaunch_process' in globals():
            roslaunch_process.terminate()
        emg_processor.read_emg_flag = False
        emg_processor.save_file()
        np.save(f'{folder}/sub_object_all.npy', sub_object_all)
        np.save(f'{folder}/sub_object_time.npy', sub_object_time)
        np.save(f'{folder}/muscle_coactivation_all.npy', muscle_coactivation_all)

