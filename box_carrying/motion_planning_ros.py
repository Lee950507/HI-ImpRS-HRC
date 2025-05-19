#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import utils
import transformation as tsf
import main_opt_static as mos
import message_filters

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation as R
from itertools import product
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from geometry_msgs.msg import PoseStamped

import sys
import os
import rospy
import signal
import subprocess
import time

import tkinter as tk
from tkinter import messagebox

from utils import plot_skeleton

# 获取 ROS 工作空间的路径
workspace_path = '/home/clover/catkin_ws'

# 添加编译后的库路径
sys.path.append(os.path.join(workspace_path, 'devel', 'lib'))

from libpython_curi_dual_arm_ic import Python_CURI_Control


last_relative_pose_wrists = None
last_object_pose = None
global ind
ind = 1


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


def compress_bounds(joint_angle_bounds, q, compression_factor=0.5):
    new_bounds = []
    joint_center = q

    for i, (lower, upper) in enumerate(joint_angle_bounds):
        range_half = (upper - lower) * compression_factor / 2
        center = joint_center[i]
        new_lower = center - range_half
        new_upper = center + range_half

        new_bounds.append((new_lower, new_upper))

    return new_bounds


def trans_shoulder2global(joint_pos, shoulder_pos, arm='right'):
    if arm=='left':
        joint_pos[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos[1] = -joint_pos[1]
        joint_pos = joint_pos + shoulder_pos
    if arm=='right':
        joint_pos[[0, 1]] = -joint_pos[[1, 0]]
        joint_pos = joint_pos + shoulder_pos
    return joint_pos


def trans_global2shoulder(shoulder, elbow, wrist, arm='left'):
    if arm=='left':
        elbow_new = elbow - shoulder
        elbow_new = np.array([elbow_new[1], -elbow_new[0], elbow_new[2]])
        wrist_new = wrist - shoulder
        wrist_new = np.array([wrist_new[1], -wrist_new[0], wrist_new[2]])
    if arm=='right':
        elbow_new = elbow - shoulder
        elbow_new = np.array([-elbow_new[1], -elbow_new[0], elbow_new[2]])
        wrist_new = wrist - shoulder
        wrist_new = np.array([-wrist_new[1], -wrist_new[0], wrist_new[2]])
    return elbow_new, wrist_new


def minimum_jerk_trajectory(start, end, t_start, t_end, t_sample):
    # 时间差
    T = t_end - t_start

    # Minimum jerk 的多项式系数
    c0 = start[0]
    c1 = start[1]
    c2 = start[2] / 2.0
    c3 = (20 * (end[0] - start[0]) - (8 * end[1] + 12 * start[1]) * T - (3 * start[2] - end[2]) * T**2) / (2 * T**3)
    c4 = (-30 * (end[0] - start[0]) + (14 * end[1] + 16 * start[1]) * T + (3 * start[2] - 2 * end[2]) * T**2) / (2 * T**4)
    c5 = (12 * (end[0] - start[0]) - (6 * end[1] + 6 * start[1]) * T - (start[2] - end[2]) * T**2) / (2 * T**5)

    # 时间序列
    time_steps = np.arange(t_start, t_end, t_sample)
    trajectory = []
    for t in time_steps:
        dt = t - t_start
        # 位置
        position = c0 + c1 * dt + c2 * dt**2 + c3 * dt**3 + c4 * dt**4 + c5 * dt**5
        # 速度
        velocity = c1 + 2 * c2 * dt + 3 * c3 * dt**2 + 4 * c4 * dt**3 + 5 * c5 * dt**4
        # 加速度
        acceleration = 2 * c2 + 6 * c3 * dt + 12 * c4 * dt**2 + 20 * c5 * dt**3
        trajectory.append([position, velocity, acceleration])

    return np.array(trajectory)


def generate_trajectory_with_speed_limit(waypoints, speed_limit, t_total, t_sample):
    num_waypoints = len(waypoints)
    if num_waypoints < 2:
        raise ValueError("需要至少两个 waypoints 来生成轨迹")

    # 平均分配时间
    t_waypoints = np.linspace(0, t_total, num_waypoints)

    # 存储最终轨迹
    full_trajectory = []

    for i in range(num_waypoints - 1):
        start = [waypoints[i], 0, 0]  # 假设初始速度和加速度为 0
        end = [waypoints[i + 1], 0, 0]  # 假设目标速度和加速度为 0
        t_start = t_waypoints[i]
        t_end = t_waypoints[i + 1]

        # 生成 minimum jerk 轨迹
        segment_trajectory = minimum_jerk_trajectory(start, end, t_start, t_end, t_sample)

        # 合并轨迹段
        if i > 0:
            # 避免重复第一个点
            full_trajectory = np.vstack((full_trajectory, segment_trajectory[1:]))
        else:
            full_trajectory = segment_trajectory

    return full_trajectory


def update(frame):
    global current_q, global_positions, trajectory_hand, trajectory_elbow
    ax.clear()
    ax.set_xlim((0.7, 2.2))
    ax.set_ylim((-0.7, 0.8))
    ax.set_zlim((0.0, 1.5))
    # ax.set_xlim((-0.5, 0.1))
    # ax.set_ylim((0, 0.6))
    # ax.set_zlim((0.9, 1.5))

    ax.view_init(elev=30, azim=-30)

    new_bounds = compress_bounds(joint_angle_bounds, current_q, compression_factor=comp_factor)
    joint_angle_ranges = [np.linspace(lower, upper, num_samples_per_joint) for lower, upper in new_bounds]

    q_combinations = np.array(list(product(*joint_angle_ranges)))

    scores = []
    candidate_elbows = []
    candidate_hands = []

    for q in q_combinations:
        elbow_cand, hand_cand = mos.forward_kinematics(q, d_uar, d_lar)
        hand_cand = trans_shoulder2global(hand_cand, shoulder, arm='right')
        elbow_cand = trans_shoulder2global(elbow_cand, shoulder, arm='right')

        candidate_elbows.append(elbow_cand)
        candidate_hands.append(hand_cand)
        s = utils.calculate_upper_limb_score_with_joint_angles(q)
        scores.append(s)

    scores = np.array(scores)
    candidate_elbows = np.array(candidate_elbows)
    candidate_hands = np.array(candidate_hands)

    norm_obj = Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(norm_obj(scores))
    # ax.scatter(candidate_hands[:, 0], candidate_hands[:, 1], candidate_hands[:, 2],
    #            c=colors, s=5, alpha=0.5)

    # ref_point = global_positions[8]
    ref_point = global_positions[5]

    ## Find the neighbor with the lowest ergo score

    target_idx = np.argmin(scores)
    candidate_q = q_combinations[target_idx]

    new_elbow = candidate_elbows[target_idx]
    new_hand = candidate_hands[target_idx]

    dist = np.linalg.norm(new_hand - ref_point)
    if dist > max_disp:
        ratio = max_disp / dist
    else:
        ratio = 1.0

    new_q = current_q + ratio * (candidate_q - current_q)

    ## Find the neighbor pointing to the optimal point

    # A = ref_point
    # B = optimal_position
    # expected_p = A + frame * (B - A) / 20
    #
    # candidate_expected_dists = np.linalg.norm(candidate_hands - expected_p, axis=1)
    # target_idx = np.argmin(candidate_expected_dists)
    # candidate_q = q_combinations[target_idx]
    # candidate_hand = candidate_hands[target_idx]
    # candidate_elbow = candidate_elbows[target_idx]
    #
    # # 计算从当前手腕到选择候选点的位移
    # disp = candidate_hand - ref_point
    # dist = np.linalg.norm(disp)
    # if dist > max_disp:
    #     ratio = max_disp / dist
    # else:
    #     ratio = 1.0
    #
    # # 在关节空间中按比例线性插值更新配置
    # new_q = current_q + ratio * (candidate_q - current_q)

    new_elbow, new_hand = mos.forward_kinematics(new_q, d_uar, d_lar)
    new_hand = trans_shoulder2global(new_hand, shoulder, arm='right')
    new_elbow = trans_shoulder2global(new_elbow, shoulder, arm='right')

    current_q = new_q

    s = utils.calculate_upper_limb_score_with_joint_angles(current_q)
    score_history.append(s)

    # global_positions[7] = new_elbow
    # global_positions[8] = new_hand
    global_positions[4] = new_elbow
    global_positions[5] = new_hand

    # 将新位置加入轨迹（便于绘制轨迹）
    trajectory_hand.append(new_hand.copy())
    trajectory_elbow.append(new_elbow.copy())
    joint_history.append(new_q.copy())

    # ---------------------- 绘制部分 ----------------------
    # 绘制肩部、肘部、手腕点（用不同颜色标记），以及轨迹
    ax.scatter(shoulder[0], shoulder[1], shoulder[2], c='black', s=50, label='Shoulder')
    ax.scatter(new_elbow[0], new_elbow[1], new_elbow[2], c='blue', s=50, label='Elbow')
    ax.scatter(new_hand[0], new_hand[1], new_hand[2], c='green', s=50, label='Hand')

    # 绘制轨迹（手腕轨迹）
    traj = np.array(trajectory_hand)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='green', linestyle='--')

    # 绘制从肩部到手腕的连线，模拟手臂
    ax.plot([shoulder[0], new_elbow[0], new_hand[0]],
            [shoulder[1], new_elbow[1], new_hand[1]],
            [shoulder[2], new_elbow[2], new_hand[2]], c='red', linewidth=2)

    # 绘制参考最优位置（optimal_position，供对比）
    ax.scatter(optimal_position[0], optimal_position[1], optimal_position[2],
               c='magenta', s=50, label='Optimal Position')

    utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='black')

    # 设定坐标轴标签与标题
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f'Iteration {frame + 1}')
    ax.legend()


def run_iterations(num_iterations):
    global current_q, global_positions, trajectory_hand, trajectory_elbow, score_history, joint_history

    # 执行所有迭代
    for frame in range(num_iterations):
        # 原update函数的内容，但不重置图形
        global_positions_copy = global_positions.copy()

        new_bounds = compress_bounds(joint_angle_bounds, current_q, compression_factor=comp_factor)
        joint_angle_ranges = [np.linspace(lower, upper, num_samples_per_joint) for lower, upper in new_bounds]

        q_combinations = np.array(list(product(*joint_angle_ranges)))

        scores = []
        candidate_elbows = []
        candidate_hands = []

        for q in q_combinations:
            elbow_cand, hand_cand = mos.forward_kinematics(q, d_uar, d_lar)
            hand_cand = trans_shoulder2global(hand_cand, shoulder, arm='right')
            elbow_cand = trans_shoulder2global(elbow_cand, shoulder, arm='right')

            candidate_elbows.append(elbow_cand)
            candidate_hands.append(hand_cand)
            s = utils.calculate_upper_limb_score_with_joint_angles(q)
            scores.append(s)

        scores = np.array(scores)
        candidate_elbows = np.array(candidate_elbows)
        candidate_hands = np.array(candidate_hands)

        # ref_point = global_positions[8]
        ref_point = global_positions[5]

        ## Find the neighbor with the lowest ergo score
        target_idx = np.argmin(scores)
        candidate_q = q_combinations[target_idx]

        new_elbow = candidate_elbows[target_idx]
        new_hand = candidate_hands[target_idx]

        dist = np.linalg.norm(new_hand - ref_point)
        if dist > max_disp:
            ratio = max_disp / dist
        else:
            ratio = 1.0

        new_q = current_q + ratio * (candidate_q - current_q)

        new_elbow, new_hand = mos.forward_kinematics(new_q, d_uar, d_lar)
        new_hand = trans_shoulder2global(new_hand, shoulder, arm='right')
        new_elbow = trans_shoulder2global(new_elbow, shoulder, arm='right')

        current_q = new_q

        s = utils.calculate_upper_limb_score_with_joint_angles(current_q)
        score_history.append(s)

        # global_positions[7] = new_elbow
        # global_positions[8] = new_hand
        global_positions[4] = new_elbow
        global_positions[5] = new_hand

        # 将新位置加入轨迹（便于绘制轨迹）
        trajectory_hand.append(new_hand.copy())
        trajectory_elbow.append(new_elbow.copy())
        joint_history.append(new_q.copy())

        print(f"Iteration {frame + 1}/{num_iterations} completed. Current score: {s:.4f}")

    # 迭代完成后绘制最终结果
    ax.set_xlim((0.7, 2.2))
    ax.set_ylim((-0.7, 0.8))
    ax.set_zlim((0.0, 1.5))
    ax.view_init(elev=30, azim=-30)

    # 绘制肩部、肘部、手腕点（用不同颜色标记），以及轨迹
    ax.scatter(shoulder[0], shoulder[1], shoulder[2], c='black', s=50, label='Shoulder')
    ax.scatter(new_elbow[0], new_elbow[1], new_elbow[2], c='blue', s=50, label='Elbow')
    ax.scatter(new_hand[0], new_hand[1], new_hand[2], c='green', s=50, label='Hand')

    # 绘制轨迹（手腕轨迹）
    traj = np.array(trajectory_hand)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='green', linestyle='--')

    # 绘制从肩部到手腕的连线，模拟手臂
    ax.plot([shoulder[0], new_elbow[0], new_hand[0]],
            [shoulder[1], new_elbow[1], new_hand[1]],
            [shoulder[2], new_elbow[2], new_hand[2]], c='red', linewidth=2)

    # 绘制参考最优位置（optimal_position，供对比）
    ax.scatter(optimal_position[0], optimal_position[1], optimal_position[2],
               c='magenta', s=50, label='Optimal Position')

    utils.plot_skeleton(ax, global_positions, skeleton_parent_indices, color='black')

    # 设定坐标轴标签与标题
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f'Final Result after {num_iterations} Iterations')
    ax.legend()

    plt.show()

    # 绘制分数历史趋势图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(score_history) + 1), score_history)
    plt.xlabel('Iteration')
    plt.ylabel('Ergonomic Score')
    plt.title('Ergonomic Score History')
    plt.grid(True)
    plt.show()

    return trajectory_hand, trajectory_elbow, joint_history, score_history


if __name__ == '__main__':
    rospy.init_node('vf_hrc')
    signal.signal(signal.SIGINT, signal_handler)
    # 启动 roslaunch
    roslaunch_process = launch_roslaunch()
    time.sleep(1)  # 等待一段时间以确保 ROS 节点启动
    # 启动控制器

    curi = Python_CURI_Control(0, [])
    curi.start()

    time.sleep(1)  #
    vrpn_roslaunch_process = vrpn_launch_roslaunch()

    ## Initialization of robot end effector poses
    robot_left_position_init = np.array([1.0, 0.15, 1.2])
    robot_right_position_init = np.array([0.9, -0.25, 0.7])

    robot_left_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    robot_right_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    robot_left_pose_matrix_init = np.r_[
        np.c_[robot_left_rotation_matrix_init, robot_left_position_init.T], np.array([[0, 0, 0, 1]])]
    robot_right_pose_matrix_init = np.r_[
        np.c_[robot_right_rotation_matrix_init, robot_right_position_init.T], np.array([[0, 0, 0, 1]])]

    base2torso_matrix = np.array([[1, 0, 0, -0.29], [0, 1, 0, 0], [0, 0, 1, -0.985], [0, 0, 0, 1]])
    initial_robot_left_pose_matrix = base2torso_matrix @ robot_left_pose_matrix_init
    initial_robot_right_pose_matrix = base2torso_matrix @ robot_right_pose_matrix_init

    print("init_left_arm_pub", initial_robot_left_pose_matrix)

    print("init_right_arm_pub", initial_robot_right_pose_matrix)

    curi.set_tcp_moveL(initial_robot_left_pose_matrix, initial_robot_right_pose_matrix)

    while curi.get_curi_mode(0) != 2 and curi.get_curi_mode(1) != 2:
        print("waiting robot external control")
        time.sleep(1)

    print("Start planning...")
    time.sleep(3)
    subscriber_robot = rospy.wait_for_message('/vrpn_client_node/robot/pose', PoseStamped)
    # subscriber_shouL = rospy.wait_for_message('/vrpn_client_node/shouL/pose', PoseStamped)
    subscriber_shouR = rospy.wait_for_message('/vrpn_client_node/shouR/pose', PoseStamped)
    # subscriber_elbowL = rospy.wait_for_message('/vrpn_client_node/elbowL/pose', PoseStamped)
    subscriber_elbowR = rospy.wait_for_message('/vrpn_client_node/elbowR/pose', PoseStamped)
    # subscriber_wristL = rospy.wait_for_message('/vrpn_client_node/wristL/pose', PoseStamped)
    subscriber_wristR = rospy.wait_for_message('/vrpn_client_node/wristR/pose', PoseStamped)
    print("collecting human data successfully!")

    sub_robot = transform_to_pose(subscriber_robot)
    # sub_shouL = transform_to_pose(subscriber_shouL)
    sub_shouR = transform_to_pose(subscriber_shouR)
    # sub_elbowL = transform_to_pose(subscriber_elbowL)
    sub_elbowR = transform_to_pose(subscriber_elbowR)
    # sub_wristL = transform_to_pose(subscriber_wristL)
    sub_wristR = transform_to_pose(subscriber_wristR)

    # sub_robot = np.array([-0.2195, 1.11462, 0, 0, 0, 0, 1])
    # sub_shouL = np.array([2, 1.5, 0.25, 0, 0, 0, 1])
    # sub_shouR = np.array([2, 1.5, -0.25, 0, 0, 0, 1])
    # sub_elbowL = np.array([1.9, 1.3, 0.3, 0, 0, 0, 1])
    # sub_elbowR = np.array([1.9, 1.3, -0.3, 0, 0, 0, 1])
    # sub_wristL = np.array([1.8, 1.2, 0.3, 0, 0, 0, 1])
    # sub_wristR = np.array([1.8, 1.4, -0.3, 0, 0, 0, 1])

    T_optitrack2robotbase = np.linalg.inv(
        tsf.transform_optitrack_origin_to_optitrack_robot(
            sub_robot) @ tsf.transform_optitrack_robot_to_robot_base())
    # shouL_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouL[:3] + T_optitrack2robotbase[:3, 3]
    shouR_position_init = T_optitrack2robotbase[:3, :3] @ sub_shouR[:3] + T_optitrack2robotbase[:3, 3]
    # elbowL_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowL[:3] + T_optitrack2robotbase[:3, 3]
    elbowR_position_init = T_optitrack2robotbase[:3, :3] @ sub_elbowR[:3] + T_optitrack2robotbase[:3, 3]
    # wristL_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristL[:3] + T_optitrack2robotbase[:3, 3]
    wristR_position_init = T_optitrack2robotbase[:3, :3] @ sub_wristR[:3] + T_optitrack2robotbase[:3, 3]

    joint_angle_bounds = [
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 1
        (-math.pi / 18, 17 * math.pi / 18),  # Joint 2
        (-np.pi / 3, np.pi / 2),  # Joint 3
        (-np.pi / 2, np.pi / 3)  # Joint 4
    ]
    optimal_q = [0, 0, 0, -math.pi / 6]

    skeleton_joint_name, skeleton_joints, skeleton_parent_indices, skeleton_joint_local_translation = \
        utils.read_skeleton_motion('/home/clover/Chenzui/Ergo-Manip/data/demo_2_test_chenzui_only_optitrack2hotu.npy')
    skeleton_joint = skeleton_joints[500, :]
    global_positions, global_rotations = utils.forward_kinematics(skeleton_joint_local_translation,
                                                                  skeleton_joint, skeleton_parent_indices)
    global_positions[:, 2] = global_positions[:, 2] * 1.2

    global_positions[4] = global_positions[3] + (elbowR_position_init - shouR_position_init)
    # global_positions[7] = global_positions[6] + (elbowL_position_init - shouL_position_init)
    global_positions[5] = global_positions[3] + (wristR_position_init - shouR_position_init)
    # global_positions[8] = global_positions[6] + (wristL_position_init - shouL_position_init)

    shou_center = shouR_position_init
    global_positions = global_positions + np.array([shou_center[0], shou_center[1], 0])


    initial_position = global_positions[5]

    # Body dimensions
    d_ual, d_uar, d_lal, d_lar = mos.calculate_arm_dimensions(shouR_position_init, elbowR_position_init,
                                                              wristR_position_init, shouR_position_init,
                                                              elbowR_position_init, wristR_position_init)

    # 计算初始“最优”位置（仅用于可视化对比），这里采用 optimal_q 得到的手腕位置
    _, optimal_position = mos.forward_kinematics(optimal_q, d_uar, d_lar)
    # optimal_position = trans_shoulder2global(optimal_position, global_positions[6], arm='left')
    optimal_position = trans_shoulder2global(optimal_position, global_positions[3], arm='right')

    # p_elbowL_init, p_wristL_init = trans_global2shoulder(shouL_position_init, elbowL_position_init,
    #                                                      wristL_position_init, arm='left')
    p_elbowR_init, p_wristR_init = trans_global2shoulder(global_positions[3], global_positions[4], global_positions[5],
                                                         arm='right')

    current_q = mos.inverse_kinematics(p_elbowR_init, p_wristR_init, d_uar, d_lar)
    current_score = utils.calculate_upper_limb_score_with_joint_angles(current_q)

    # hand_current = global_positions[8]
    # elbow_current = global_positions[7]
    hand_current = global_positions[5]
    elbow_current = global_positions[4]

    # shoulder = global_positions[6].copy()
    shoulder = global_positions[3].copy()

    # 为动画记录历史轨迹（可选）
    trajectory_hand = [hand_current.copy()]
    trajectory_elbow = [elbow_current.copy()]

    score_history = []
    joint_history = []

    # 设置候选离散采样数、压缩系数和迭代次数
    num_samples_per_joint = 15
    comp_factor = 0.1
    num_iterations = 30
    max_disp = 0.03  # maximum allowed displacement per iteration in global (hand) space

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ---------------------- 动画启动 ----------------------
    # anim = FuncAnimation(fig, update, frames=num_iterations, interval=800, repeat=False)
    # anim.save("/home/ubuntu/Ergo-Manip/vector_field/figs/animation_left_arm_straight.gif", writer=PillowWriter(fps=2))
    # plt.show()

    trajectory_hand, trajectory_elbow, joint_history, score_history = run_iterations(num_iterations)

    # 在这里添加你想在迭代完成后执行的代码
    print("Iterations completed. Continuing with next steps...")

    waypoints_ergo = trajectory_hand
    waypoints_straight = [trajectory_hand[0], trajectory_hand[len(trajectory_hand) - 1]]

    speed_limit = 0.05  # 最大速度限制
    t_total = 8  # 总时间
    t_sample = 0.001  # 采样时间间隔 (1000 Hz)

    # 生成轨迹
    trajectory_ergo = generate_trajectory_with_speed_limit(waypoints_ergo, speed_limit, t_total, t_sample)
    trajectory_straight = generate_trajectory_with_speed_limit(waypoints_straight, speed_limit, t_total, t_sample)

    plt.figure()
    plt.plot(trajectory_straight[:, 0])
    # plt.plot(trajectory_straight[:, 1])
    # plt.plot(trajectory_straight[:, 2])
    plt.show()

    ## Normalize position
    # position_ergo = trajectory_ergo[:, 0] - trajectory_ergo[0, 0]
    position_ergo = trajectory_straight[:, 0] - trajectory_straight[0, 0]

    print("left_arm_current", curi.get_tcp(0))

    print("right_arm_current", curi.get_tcp(1))

    root = tk.Tk()
    root.withdraw()

    # 显示带有确定和取消按钮的弹窗
    response = messagebox.askokcancel("确认", "是否继续执行程序？")

    if response:  # 如果用户点击确定
        print("用户点击了确定，程序继续执行")
        # 继续执行后续代码
    else:  # 如果用户点击取消
        print("用户点击了取消，程序终止")

    for i in range(len(position_ergo)):
        robot_left_position = robot_left_position_init + position_ergo[i]
        robot_right_position = robot_right_position_init

        robot_left_pose_matrix = np.r_[
            np.c_[robot_left_rotation_matrix_init, robot_left_position.T], np.array([[0, 0, 0, 1]])]
        robot_right_pose_matrix = np.r_[
            np.c_[robot_right_rotation_matrix_init, robot_right_position.T], np.array([[0, 0, 0, 1]])]

        robot_left_pose_matrix = base2torso_matrix @ robot_left_pose_matrix
        robot_right_pose_matrix = base2torso_matrix @ robot_right_pose_matrix

        curi.set_tcp_servo(robot_left_pose_matrix, robot_right_pose_matrix)
        time.sleep(0.001)

    while 1:
        interrupt = False
        time.sleep(1)
