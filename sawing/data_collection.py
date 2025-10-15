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
    launch_file = "~/catkin_ws/src/vrpn_client_ros/launch/sample.launch"
    command = f"roslaunch {launch_file} server:=192.168.10.7"
    return subprocess.Popen(command, shell=True)


def signal_handler(sig, frame):
    print('Python shutdown signal received...')
    rospy.signal_shutdown("shutdown by manual")
    if 'roslaunch_process' in locals():
        print('Shutdown roslaunch process.')
        roslaunch_process.terminate()
        roslaunch_process.wait()
    print('Python shutdown.')
    sys.exit(0)


def transform_to_pose(pose_stamped):
    """将PoseStamped消息转换为numpy数组"""
    if pose_stamped is None:
        return None
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
    pose_stamped.pose.orientation.x = pose[3]
    pose_stamped.pose.orientation.y = pose[4]
    pose_stamped.pose.orientation.z = pose[5]
    pose_stamped.pose.orientation.w = pose[6]
    return pose_stamped


def multi_callback(optitrack_data, muscle_coactivation, start_time):
    """
    处理多个数据源的回调函数

    参数:
        optitrack_data: 包含所有6个刚体数据的字典
        muscle_coactivation: 肌肉协同激活数据
        start_time: 开始时间
    """
    global optitrack_data_all
    global optitrack_time_all
    global muscle_coactivation_all
    global last_index

    try:
        # 检查是否所有刚体数据都已接收
        all_received = all(optitrack_data[key] is not None for key in optitrack_data.keys())

        if not all_received:
            return

        # 转换所有刚体数据
        transformed_data = {}
        for key, pose_stamped in optitrack_data.items():
            transformed_data[key] = transform_to_pose(pose_stamped)

        # 处理肌肉协同激活数据
        muscle_coactivation = np.asarray(muscle_coactivation)
        index = muscle_coactivation.shape[1]

        if index != last_index:
            # 保存当前时间戳
            current_time = time.time() - start_time
            optitrack_time_all.append(current_time)

            # 保存所有刚体数据
            optitrack_data_all['shouL'].append(transformed_data['shouL'])
            optitrack_data_all['elbowL'].append(transformed_data['elbowL'])
            optitrack_data_all['wristL'].append(transformed_data['wristL'])
            optitrack_data_all['shouR'].append(transformed_data['shouR'])
            optitrack_data_all['elbowR'].append(transformed_data['elbowR'])
            optitrack_data_all['wristR'].append(transformed_data['wristR'])

            # 保存肌肉协同激活数据
            muscle_coactivation_all.append(muscle_coactivation[:, -1])

            last_index = index

            # 打印调试信息
            if len(optitrack_time_all) % 100 == 0:  # 每100个数据点打印一次
                print(f"Collected {len(optitrack_time_all)} samples, EMG shape: {muscle_coactivation.shape}")

            time.sleep(0.001)  # 1kHz控制频率

    except Exception as e:
        # print(f"Error in callback: {e}")
        pass


def save_all_data(folder):
    """保存所有收集的数据"""
    try:
        print("\nSaving data...")

        # 保存OptiTrack数据
        for key in optitrack_data_all.keys():
            data_array = np.array(optitrack_data_all[key])
            filename = os.path.join(folder, f'{key}_data.npy')
            np.save(filename, data_array)
            print(f"Saved {key} data: shape {data_array.shape}")

        # 保存时间戳
        time_array = np.array(optitrack_time_all)
        time_filename = os.path.join(folder, 'optitrack_time.npy')
        np.save(time_filename, time_array)
        print(f"Saved timestamp data: shape {time_array.shape}")

        # 保存肌肉协同激活数据
        muscle_array = np.array(muscle_coactivation_all)
        muscle_filename = os.path.join(folder, 'muscle_coactivation.npy')
        np.save(muscle_filename, muscle_array)
        print(f"Saved muscle coactivation data: shape {muscle_array.shape}")

        # 保存为单个字典文件(可选)
        all_data = {
            'shouL': np.array(optitrack_data_all['shouL']),
            'elbowL': np.array(optitrack_data_all['elbowL']),
            'wristL': np.array(optitrack_data_all['wristL']),
            'shouR': np.array(optitrack_data_all['shouR']),
            'elbowR': np.array(optitrack_data_all['elbowR']),
            'wristR': np.array(optitrack_data_all['wristR']),
            'timestamp': time_array,
            'muscle_coactivation': muscle_array
        }
        combined_filename = os.path.join(folder, 'all_data_combined.npy')
        np.save(combined_filename, all_data)
        print(f"Saved combined data to {combined_filename}")

        print(f"\nAll data saved successfully to {folder}")
        print(f"Total samples collected: {len(optitrack_time_all)}")

    except Exception as e:
        print(f"Error saving data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    global optitrack_data_all
    global optitrack_time_all
    global muscle_coactivation_all
    global last_index

    folder = '/home/clover/Chenzui/HI-ImpRS-HRC/data/emg_record/sawing/chenzui&zhuo/4'
    os.makedirs(folder, exist_ok=True)
    rospy.init_node('data_collection')
    signal.signal(signal.SIGINT, signal_handler)

    roslaunch_process = vrpn_launch_roslaunch()
    time.sleep(1)

    last_index = 0

    # 初始化数据存储结构
    optitrack_data_all = {
        'shouL': [],
        'elbowL': [],
        'wristL': [],
        'shouR': [],
        'elbowR': [],
        'wristR': []
    }
    optitrack_time_all = []
    muscle_coactivation_all = []

    # 初始化OptiTrack数据字典
    optitrack_data = {
        'shouL': None,
        'elbowL': None,
        'wristL': None,
        'shouR': None,
        'elbowR': None,
        'wristR': None
    }

    # 启动EMG处理器
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
    print("EMG processor initialized")


    # 定义回调函数(为每个topic创建独立的回调)
    def shouL_callback(msg):
        global optitrack_data
        optitrack_data['shouL'] = msg


    def elbowL_callback(msg):
        global optitrack_data
        optitrack_data['elbowL'] = msg


    def wristL_callback(msg):
        global optitrack_data
        optitrack_data['wristL'] = msg


    def shouR_callback(msg):
        global optitrack_data
        optitrack_data['shouR'] = msg


    def elbowR_callback(msg):
        global optitrack_data
        optitrack_data['elbowR'] = msg


    def wristR_callback(msg):
        global optitrack_data
        optitrack_data['wristR'] = msg


    # 订阅所有OptiTrack topics
    shouL_subscriber = rospy.Subscriber('/vrpn_client_node/shouL/pose', PoseStamped, shouL_callback)
    elbowL_subscriber = rospy.Subscriber('/vrpn_client_node/elbowL/pose', PoseStamped, elbowL_callback)
    wristL_subscriber = rospy.Subscriber('/vrpn_client_node/wristL/pose', PoseStamped, wristL_callback)
    shouR_subscriber = rospy.Subscriber('/vrpn_client_node/shouR/pose', PoseStamped, shouR_callback)
    elbowR_subscriber = rospy.Subscriber('/vrpn_client_node/elbowR/pose', PoseStamped, elbowR_callback)
    wristR_subscriber = rospy.Subscriber('/vrpn_client_node/wristR/pose', PoseStamped, wristR_callback)

    print("OptiTrack subscribers initialized")

    try:
        print("\n" + "=" * 70)
        print("Starting data collection...")
        print("Press Ctrl+C to stop and save data")
        print("=" * 70 + "\n")

        while not rospy.is_shutdown():
            # 等待OptiTrack数据
            all_optitrack_received = all(optitrack_data[key] is not None for key in optitrack_data.keys())
            if not all_optitrack_received:
                missing_topics = [key for key, val in optitrack_data.items() if val is None]
                rospy.loginfo_throttle(2, f"Waiting for OptiTrack data... Missing: {missing_topics}")
                time.sleep(0.01)
                continue

            # 等待EMG数据
            if len(emg_processor.all_emg_data[0]) == 0:
                rospy.loginfo_throttle(1, "Waiting for EMG data...")
                time.sleep(0.01)
                continue

            # 处理数据
            multi_callback(
                optitrack_data,
                emg_processor.all_emg_data,
                start_time,
            )

        print("Execution finished.")
        emg_processor.read_emg_flag = False

        # 保持程序运行,等待中断信号
        while not rospy.is_shutdown():
            time.sleep(1)

            data_queue.join()
            for t in threads:
                t.join()

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping data collection...")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nCleaning up and saving data...")

        # 清理资源
        if 'roslaunch_process' in globals():
            roslaunch_process.terminate()

        # 停止EMG采集
        emg_processor.read_emg_flag = False
        emg_processor.save_file()

        # 保存所有OptiTrack数据
        save_all_data(folder)

        print("\nData collection completed!")