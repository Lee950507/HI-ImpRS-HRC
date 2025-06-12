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

from libpython_curi_dual_arm_ic import Python_CURI_Control

# 获取 ROS 工作空间的路径
workspace_path = '/home/clover/catkin_ws'

# 添加编译后的库路径
sys.path.append(os.path.join(workspace_path, 'devel', 'lib'))


def launch_roslaunch():
    launch_file = "~/catkin_ws/src/curi_whole_body_interface/launch/python_curi_dual_arm_ic_qbhand.launch"  # 替换为你的 launch 文件路径
    # 启动 roslaunch
    command = f"roslaunch {launch_file}"
    return subprocess.Popen(command, shell=True)


def vrpn_launch_roslaunch():
    launch_file = "~/catkin_ws/src/vrpn_client_ros/launch/sample.launch"  # 替换为你的 launch 文件路径
    # 启动 roslaunch
    command = f"roslaunch {launch_file} server:=192.168.10.10"
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


def multi_callback(sub_torso, reference_traj, reference_stiff, torso_pub, time_array, index_counter, muscle_coactivation):
    sub_torso, time_array[index_counter] = transform_to_joint(sub_torso)
    print("torso_joint:", sub_torso)

    # 第一次调用时设置初始时间
    if index_counter == 0:
        start_time = time_array[0]
        index = 0
    else:
        index = int((time_array[index_counter] - time_array[0]) * 1000)

    print(f"Index: {index}")
    print(f"coactivation: {muscle_coactivation[-1]}")

    if index <= 18299:
        right_pos = reference_traj[index, :3] + robot_right_position_init
        # right_pos = robot_right_position_init
        left_pos = robot_left_position_init
        right_stiff = reference_stiff[index, :]

        robot_right_pose_matrix = np.r_[
            np.c_[robot_right_rotation_matrix_init, right_pos.T], np.array([[0, 0, 0, 1]])]
        robot_left_pose_matrix = np.r_[
            np.c_[robot_left_rotation_matrix_init, left_pos.T], np.array([[0, 0, 0, 1]])]

        p, R = tsf.transform_torso_base_to_torso_end(sub_torso)
        T_TorsoBaseToTorsoEnd = np.r_[np.c_[R, p.T], np.array([[0, 0, 0, 1]])]
        T_MobileBaseToTorsoBase = np.array([[1, 0, 0, 0.2375], [0, 1, 0, 0], [0, 0, 1, 0.53762], [0, 0, 0, 1]])

        base2torso_matrix = np.linalg.inv(T_MobileBaseToTorsoBase @ T_TorsoBaseToTorsoEnd)

        robot_left_pose_matrix = base2torso_matrix_init @ robot_left_pose_matrix
        robot_right_pose_matrix = base2torso_matrix @ robot_right_pose_matrix

        T_MobileBaseToLeftArmBase, T_MobileBaseToRightArmBase = tsf.transform_robot_base_to_arm_base(sub_torso)
        T = np.linalg.inv(T_MobileBaseToRightArmBase)

        # 阻抗变化
        stiffness_matrix = [100, 100, 100, 42, 42, 42]
        stiffness_matrix[:3] = np.abs(np.dot(T[0:3, 0:3], right_stiff)) / 2
        impe_r = Float64MultiArray(data=stiffness_matrix)

        # 获取关节速度参数
        joint1_vel_ = rospy.get_param("joint1_vel", 0.08)
        joint3_vel_ = rospy.get_param("joint3_vel", 0.07)

        # 创建躯干命令消息
        torso_cmd = JointState()

        # 根据索引调整躯干关节速度
        if index <= 4000:
            torso_joint1_vel = 0
            torso_joint3_vel = -joint3_vel_
        elif index <= 12000 and index >= 4000:
            torso_joint1_vel = joint1_vel_
            torso_joint3_vel = 0
        elif index >= 12000 and index <= 17000:
            torso_joint1_vel = 0
            torso_joint3_vel = joint3_vel_
        else:
            torso_joint1_vel = 0
            torso_joint3_vel = 0

        torso_cmd.velocity = [torso_joint1_vel, 0, torso_joint3_vel, 0, 0, 0, 0]

        # 发布命令到机器人
        # curi.set_tcp_servo(robot_left_pose_matrix, robot_right_pose_matrix)
        torso_pub.publish(torso_cmd)
        # impe_r_pub.publish(impe_r)  # 取消注释以启用阻抗控制

        return index_counter + 1, False
    else:
        print("Trajectory completed!")
        return index_counter, True


if __name__ == '__main__':
    rospy.init_node('HI_ImpRS_hrc')
    signal.signal(signal.SIGINT, signal_handler)

    # 创建发布器
    torso_pub = rospy.Publisher("/curi_torso/joint/cmd_vel", JointState, queue_size=10)

    # 启动 roslaunch
    roslaunch_process = launch_roslaunch()
    time.sleep(1)  # 等待一段时间以确保 ROS 节点启动
    # 启动控制器

    curi = Python_CURI_Control(0, [])
    curi.start()

    time.sleep(1)
    # vrpn_roslaunch_process = vrpn_launch_roslaunch()

    ## Initialization of robot end effector poses
    robot_left_position_init = np.array([0.8, 0.05, 0.7])
    robot_right_position_init = np.array([1.0, -0.15, 0.9])

    robot_left_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    robot_right_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    robot_left_pose_matrix_init = np.r_[
        np.c_[robot_left_rotation_matrix_init, robot_left_position_init.T], np.array([[0, 0, 0, 1]])]
    robot_right_pose_matrix_init = np.r_[
        np.c_[robot_right_rotation_matrix_init, robot_right_position_init.T], np.array([[0, 0, 0, 1]])]

    subscriber_torso = rospy.wait_for_message('/curi_torso/joint_states', JointState)
    sub_torso, _ = transform_to_joint(subscriber_torso)
    print(sub_torso)
    # Current Joint States of Torso
    p, R = tsf.transform_torso_base_to_torso_end(sub_torso)
    T_TorsoBaseToTorsoEnd = np.r_[np.c_[R, p.T], np.array([[0, 0, 0, 1]])]
    print(T_TorsoBaseToTorsoEnd)
    T_MobileBaseToTorsoBase = np.array([[1, 0, 0, 0.2375], [0, 1, 0, 0], [0, 0, 1, 0.53762], [0, 0, 0, 1]])

    base2torso_matrix_init = np.linalg.inv(T_MobileBaseToTorsoBase @ T_TorsoBaseToTorsoEnd)
    print(base2torso_matrix_init)
    initial_robot_left_pose_matrix = base2torso_matrix_init @ robot_left_pose_matrix_init
    initial_robot_right_pose_matrix = base2torso_matrix_init @ robot_right_pose_matrix_init
    print("left",initial_robot_left_pose_matrix)
    print("right", initial_robot_right_pose_matrix)
    curi.set_tcp_moveL(initial_robot_left_pose_matrix, initial_robot_right_pose_matrix)

    while curi.get_curi_mode(0) != 2 and curi.get_curi_mode(1) != 2:
        print("waiting robot external control")
        time.sleep(1)

    reference_traj = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/box_carrying/traj_box_carrying_18300.npy', allow_pickle=True)
    reference_stiff = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/box_carrying/stiff_box_carrying_18300.npy', allow_pickle=True)

    # 准备控制循环所需变量
    index_counter = 0
    time_array = np.zeros(100000)

    # 创建躯干数据订阅器
    torso_data = None

    emg_processor = EMGProcessor()
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


    def torso_callback(msg):
        global torso_data
        torso_data = msg


    torso_subscriber = rospy.Subscriber('/curi_torso/joint_states', JointState, torso_callback)

    try:
        print("Starting trajectory execution...")
        trajectory_completed = False

        while not rospy.is_shutdown() and not trajectory_completed:
            if torso_data is None:
                rospy.loginfo_throttle(1, "Waiting for torso data...")
                time.sleep(0.01)
                continue

            # 处理躯干数据并控制机器人
            index_counter, trajectory_completed = multi_callback(
                torso_data,
                reference_traj,
                reference_stiff,
                torso_pub,
                time_array,
                index_counter,
                emg_processor.all_emg_data
            )

            # 控制循环频率
            time.sleep(0.001)  # 1kHz控制频率

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
        if 'roslaunch_process' in globals():
            roslaunch_process.terminate()
        # if 'vrpn_roslaunch_process' in globals():
        #     vrpn_roslaunch_process.terminate()
