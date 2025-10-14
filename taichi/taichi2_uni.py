#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import transformation as tsf
import scipy.linalg as linalg

import message_filters
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion, Pose, WrenchStamped
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
import pickle
import traceback
import argparse
from keras.models import load_model

from libpython_curi_dual_arm_ic import Python_CURI_Control

from stiffness_estimator import StiffnessDampingEKF

# 获取 ROS 工作空间的路径
workspace_path = '/home/clover/catkin_ws'

# 添加编译后的库路径
sys.path.append(os.path.join(workspace_path, 'devel', 'lib'))

# 刚度模式:
# 1: 定刚度
# 2: 参考刚度变化轨迹
# 3: 基于肌肉激活的变阻抗
# 4: HI-ImpRS (LSTM基于EMG)
# 5: HI-ImpRS-Force (LSTM基于交互力估计的刚度,同时记录EMG激活度对比)
STIFFNESS_MODE = 5

lstm_model = None
scalers = None
look_back = 10
training_max_activation = 0.2
activation_history = []
previous_stiffness = None
EMG_WINDOW_SIZE = 5
MAX_STIFFNESS_CHANGE_RATE = 4.0

# 新增:力传感器和刚度估计相关变量
force_data = None
robot_pose_data = None  # 新增:机器人末端位姿数据
stiffness_estimators = None  # 三个方向的刚度估计器
human_stiffness_profile = None  # Human stiffness profile
estimated_stiffness_history = []  # 存储估计的刚度历史

# 新增:机器人位置和速度历史(用于计算速度)
robot_position_history = []
robot_velocity_history = []
last_position = None
last_time = None

# 新增:模式5中保存两种激活度的对比数据
activation_comparison_data = {
    'timestamp': [],
    'emg_activation': [],  # 基于EMG的激活度
    'force_activation': [],  # 基于力的激活度
    'predicted_activation': [],  # LSTM预测的激活度
    'estimated_stiffness': [],  # 估计的人体刚度
    'reference_stiffness': [],  # 参考人体刚度
    'robot_stiffness': []  # 最终机器人刚度
}

# 加载LSTM模型(如果使用HI-ImpRS模式)
if STIFFNESS_MODE in [4, 5]:
    save_dir = os.path.expanduser('~/Chenzui/HI-ImpRS-HRC/LSTM/saved_multivariate_lstm_with_max_act')

    try:
        model_path = os.path.join(save_dir, 'multivariate_lstm_model.h5')
        lstm_model = load_model(model_path)
        print(f"Success to load model from: {model_path}")

        params_path = os.path.join(save_dir, 'params.pkl')
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        look_back = params.get('look_back', 10)
        training_max_activation = params.get('max_activation', 0.1)
        print(f"Success to load,look_back = {look_back}, maximum activation = {training_max_activation}")

        scaler_path = os.path.join(save_dir, 'scalers.pkl')
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        print(f"Success to load scalers: {scalers}")

    except Exception as e:
        print(f"Fail to load: {e}")
        traceback.print_exc()
        STIFFNESS_MODE = 1
        print("Falling back to default stiffness mode")

# 新增:加载human stiffness profile(仅模式5需要)
if STIFFNESS_MODE == 5:
    try:
        # 修改为你的human stiffness profile文件路径
        human_stiffness_path = '/home/clover/Chenzui/HI-ImpRS-HRC/data/taichi/human_stiffness_profile.npy'
        human_stiffness_profile = np.load(human_stiffness_path)
        print(f"Success to load human stiffness profile from: {human_stiffness_path}")
        print(f"Human stiffness profile shape: {human_stiffness_profile.shape}")

        # 初始化三个方向的刚度估计器
        stiffness_estimators = {
            'x': StiffnessDampingEKF(dt=0.001, initial_K=500.0, initial_D=10.0),
            'y': StiffnessDampingEKF(dt=0.001, initial_K=500.0, initial_D=10.0),
            'z': StiffnessDampingEKF(dt=0.001, initial_K=500.0, initial_D=10.0)
        }
        print("Stiffness estimators initialized")

    except Exception as e:
        print(f"Fail to load human stiffness profile: {e}")
        traceback.print_exc()
        STIFFNESS_MODE = 1
        print("Falling back to default stiffness mode")

# 肌肉激活到刚度的映射参数(用于模式3)
MUSCLE_TO_STIFFNESS_PARAMS = {
    'min_activation': 0.01,
    'max_activation': 0.25,
    'min_stiffness': 150,  # N/m
    'max_stiffness': 1000  # N/m
}

# 默认固定刚度值(用于模式1)
DEFAULT_STIFFNESS = np.array([400, 400, 400])

current_muscle_activation = 0.1


def launch_roslaunch():
    launch_file = "~/catkin_ws/src/curi_whole_body_interface/launch/python_curi_dual_arm_ic_qbhand.launch"
    command = f"roslaunch {launch_file}"
    return subprocess.Popen(command, shell=True)


def vrpn_launch_roslaunch():
    launch_file = "~/catkin_ws/src/vrpn_client_ros/launch/sample.launch"
    command = f"roslaunch {launch_file} server:=192.168.10.10"
    return subprocess.Popen(command, shell=True)


def signal_handler(sig, frame):
    print('Python shutdown signal received...')

    # 保存激活度对比数据
    if STIFFNESS_MODE == 5 and len(activation_comparison_data['timestamp']) > 0:
        save_activation_comparison()

    rospy.signal_shutdown("shutdown by manual")
    if 'roslaunch_process' in locals():
        print('Shutdown roslaunch process.')
        roslaunch_process.terminate()
        roslaunch_process.wait()
    print('Python shutdown.')
    sys.exit(0)


def save_activation_comparison():
    """保存激活度对比数据"""
    try:
        comparison_file = os.path.join(folder, 'activation_comparison.npy')
        np.save(comparison_file, activation_comparison_data)
        print(f"Activation comparison data saved to {comparison_file}")

        # 保存为CSV以便查看
        csv_file = os.path.join(folder, 'activation_comparison.csv')
        import pandas as pd
        df = pd.DataFrame({
            'timestamp': activation_comparison_data['timestamp'],
            'emg_activation': activation_comparison_data['emg_activation'],
            'force_activation': activation_comparison_data['force_activation'],
            'predicted_activation': activation_comparison_data['predicted_activation']
        })
        df.to_csv(csv_file, index=False)
        print(f"Activation comparison CSV saved to {csv_file}")

        # 生成对比图
        plot_activation_comparison(folder)

    except Exception as e:
        print(f"Error saving activation comparison data: {e}")
        traceback.print_exc()


def plot_activation_comparison(save_folder):
    """绘制激活度对比图"""
    try:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        timestamps = np.array(activation_comparison_data['timestamp'])

        # 子图1: EMG vs Force激活度
        ax1 = axes[0]
        ax1.plot(timestamps, activation_comparison_data['emg_activation'],
                 'b-', label='EMG-based Activation', linewidth=1.5, alpha=0.7)
        ax1.plot(timestamps, activation_comparison_data['force_activation'],
                 'r-', label='Force-based Activation', linewidth=1.5, alpha=0.7)
        ax1.set_ylabel('Activation', fontsize=11)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Activation Comparison: EMG vs Force', fontsize=12, fontweight='bold')

        # 子图2: LSTM预测激活度
        ax2 = axes[1]
        ax2.plot(timestamps, activation_comparison_data['predicted_activation'],
                 'g-', label='LSTM Predicted Activation', linewidth=1.5)
        ax2.set_ylabel('Predicted Activation', fontsize=11)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('LSTM Predicted Activation', fontsize=12, fontweight='bold')

        # 子图3: 机器人刚度
        ax3 = axes[2]
        robot_stiffness = np.array(activation_comparison_data['robot_stiffness'])
        if len(robot_stiffness) > 0:
            ax3.plot(timestamps, robot_stiffness[:, 0], label='X-axis', linewidth=1.5)
            ax3.plot(timestamps, robot_stiffness[:, 1], label='Y-axis', linewidth=1.5)
            ax3.plot(timestamps, robot_stiffness[:, 2], label='Z-axis', linewidth=1.5)
        ax3.set_ylabel('Robot Stiffness (N/m)', fontsize=11)
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Robot Stiffness', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plot_file = os.path.join(save_folder, 'activation_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Activation comparison plot saved to {plot_file}")
        plt.close()

    except Exception as e:
        print(f"Error plotting activation comparison: {e}")
        traceback.print_exc()


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
    pose_stamped.pose.orientation.x = pose[3]
    pose_stamped.pose.orientation.y = pose[4]
    pose_stamped.pose.orientation.z = pose[5]
    pose_stamped.pose.orientation.w = pose[6]
    return pose_stamped


def get_muscle_activation(emg_data):
    if emg_data is None or len(emg_data) == 0:
        return 0.1

    try:
        activation = np.mean(emg_data[-1])  # 取最新数据的平均
        activation = np.clip(activation, 0, 1)
        return activation
    except Exception as e:
        print(f"Error processing EMG data: {e}")
        return 0.1


def update_activation_history(activation, history_length=10):
    global activation_history
    activation_history.append(activation)
    if len(activation_history) > history_length:
        activation_history = activation_history[-history_length:]


def warmup_lstm_model():
    """预热LSTM模型,消除第一次预测的延迟"""
    global lstm_model, look_back, scalers, training_max_activation

    if lstm_model is None or STIFFNESS_MODE not in [4, 5]:
        return

    print("预热LSTM模型...")
    try:
        dummy_traj = np.zeros((look_back, 3))
        dummy_muscle = np.zeros((look_back, 1))
        dummy_max_act = np.ones((look_back, 1)) * (training_max_activation or 1.0)

        if scalers is not None:
            traj_scaler = scalers.get('traj_scaler')
            muscle_in_scaler = scalers.get('muscle_in_scaler')
            max_act_scaler = scalers.get('max_act_scaler')

            if traj_scaler is not None:
                dummy_traj = traj_scaler.transform(dummy_traj)

            if muscle_in_scaler is not None:
                dummy_muscle = muscle_in_scaler.transform(dummy_muscle)

            if max_act_scaler is not None and training_max_activation is not None:
                max_act_value = max_act_scaler.transform([[training_max_activation]])[0][0]
                dummy_max_act = np.ones((look_back, 1)) * max_act_value

        X_traj = dummy_traj.reshape(1, look_back, -1)
        X_muscle = dummy_muscle.reshape(1, look_back, -1)
        X_max_act = dummy_max_act.reshape(1, look_back, -1)

        for _ in range(5):
            _ = lstm_model.predict([X_traj, X_muscle, X_max_act])

        print("LSTM模型预热完成")
    except Exception as e:
        print(f"LSTM模型预热失败: {e}")
        traceback.print_exc()


def estimate_stiffness_from_force(index, force_wrench):
    """
    基于交互力和机器人位置估计人体端点刚度

    参数:
        index: 当前时间步索引
        force_wrench: 力/力矩数据 (WrenchStamped消息)

    返回:
        estimated_stiffness: 估计的三轴刚度 [Kx, Ky, Kz]
    """
    global stiffness_estimators, robot_position_history, robot_velocity_history
    global last_position, last_time, estimated_stiffness_history, robot_pose_data

    if force_wrench is None or stiffness_estimators is None or robot_pose_data is None:
        return np.array([500.0, 500.0, 500.0])

    try:
        # 从订阅的位姿数据中提取当前位置
        current_position = np.array([
            robot_pose_data.pose.position.x,
            robot_pose_data.pose.position.y,
            robot_pose_data.pose.position.z
        ])

        # 提取力数据
        force = np.array([
            force_wrench.wrench.force.x,
            force_wrench.wrench.force.y,
            force_wrench.wrench.force.z
        ])

        # 当前时间
        current_time = force_wrench.header.stamp.secs + 1e-9 * force_wrench.header.stamp.nsecs

        # 计算速度(数值微分)
        if last_position is not None and last_time is not None:
            dt = current_time - last_time
            if dt > 0:
                velocity = (current_position - last_position) / dt
            else:
                velocity = np.zeros(3)
        else:
            velocity = np.zeros(3)

        # 更新历史
        last_position = current_position.copy()
        last_time = current_time

        # 对三个方向分别进行刚度估计
        estimated_stiffness = np.zeros(3)
        axes = ['x', 'y', 'z']

        for i, axis in enumerate(axes):
            # 使用EKF估计刚度
            result = stiffness_estimators[axis].step(
                f_meas=force[i],
                x_c=current_position[i],
                x_dot_c=velocity[i]
            )
            estimated_stiffness[i] = result['stiffness']

        # 物理约束
        estimated_stiffness = np.clip(estimated_stiffness, 50.0, 2000.0)

        # 保存到历史
        estimated_stiffness_history.append(estimated_stiffness.copy())

        return estimated_stiffness

    except Exception as e:
        print(f"Error in stiffness estimation from force: {e}")
        traceback.print_exc()
        return np.array([500.0, 500.0, 500.0])


def calculate_stiffness(index, reference_stiff, reference_traj, emg_data=None, force_wrench=None):
    global STIFFNESS_MODE, lstm_model, MUSCLE_TO_STIFFNESS_PARAMS, DEFAULT_STIFFNESS
    global scalers, look_back, activation_history, training_max_activation
    global previous_stiffness, EMG_WINDOW_SIZE, MAX_STIFFNESS_CHANGE_RATE
    global human_stiffness_profile, activation_comparison_data

    actual_look_back = 10 if look_back is None else look_back

    # 模式4需要EMG数据
    if STIFFNESS_MODE == 4:
        muscle_activation = get_muscle_activation(emg_data)
        update_activation_history(muscle_activation, look_back)

    if STIFFNESS_MODE == 1:
        return DEFAULT_STIFFNESS

    elif STIFFNESS_MODE == 2:
        return reference_stiff[index, :]

    elif STIFFNESS_MODE == 3:
        if emg_data is None or len(emg_data) == 0:
            return DEFAULT_STIFFNESS
        try:
            current_emg = emg_data[-1][-1]
            activation_history.append(current_emg)
            if len(activation_history) > EMG_WINDOW_SIZE:
                activation_history = activation_history[-EMG_WINDOW_SIZE:]

            smoothed_emg = np.mean(activation_history)

            params = MUSCLE_TO_STIFFNESS_PARAMS
            normalized_activation = (smoothed_emg - params['min_activation']) / (
                    params['max_activation'] - params['min_activation'])
            normalized_activation = np.clip(normalized_activation, 0, 1)

            stiffness_range = params['max_stiffness'] - params['min_stiffness']
            target_stiffness = params['min_stiffness'] + normalized_activation * stiffness_range

            if previous_stiffness is not None:
                delta = target_stiffness - previous_stiffness
                if np.abs(delta) > MAX_STIFFNESS_CHANGE_RATE:
                    delta = np.sign(delta) * MAX_STIFFNESS_CHANGE_RATE
                current_stiffness = previous_stiffness + delta
            else:
                current_stiffness = target_stiffness

            previous_stiffness = current_stiffness

            direction_weights = np.array([1.0, 1.0, 1.0])
            return np.ones(3) * current_stiffness * direction_weights

        except Exception as e:
            print(f"Error in muscle-based stiffness calculation: {e}")
            traceback.print_exc()
            return DEFAULT_STIFFNESS

    elif STIFFNESS_MODE == 4:
        # 模式4: HI-ImpRS (LSTM基于EMG)
        if lstm_model is None or len(activation_history) < actual_look_back:
            print("Collecting EMG data...")
            return DEFAULT_STIFFNESS

        try:
            current_traj = reference_traj[max(0, index - actual_look_back + 1):index + 1, :3]

            if len(current_traj) < actual_look_back:
                padding = np.zeros((actual_look_back - len(current_traj), 3))
                current_traj = np.vstack([padding, current_traj])

            current_traj = current_traj[-actual_look_back:]

            padded_history = activation_history.copy()
            while len(padded_history) < actual_look_back:
                padded_history.insert(0, 0)

            muscle_input = np.array(padded_history[-actual_look_back:]).reshape(-1, 1)

            max_act = training_max_activation if training_max_activation is not None else 1.0
            max_act_input = np.ones((actual_look_back, 1)) * max_act

            if scalers is not None:
                traj_scaler = scalers.get('traj_scaler')
                muscle_in_scaler = scalers.get('muscle_in_scaler')
                max_act_scaler = scalers.get('max_act_scaler')

                if traj_scaler is not None:
                    current_traj = traj_scaler.transform(current_traj)

                if muscle_in_scaler is not None:
                    muscle_input = muscle_in_scaler.transform(muscle_input)

                if max_act_scaler is not None:
                    max_act_value = max_act_scaler.transform([[max_act]])[0][0]
                    max_act_input = np.ones((actual_look_back, 1)) * max_act_value

            X_traj = current_traj.reshape(1, actual_look_back, -1)
            X_muscle = muscle_input.reshape(1, actual_look_back, -1)
            X_max_act = max_act_input.reshape(1, actual_look_back, -1)

            predictions = lstm_model.predict([X_traj, X_muscle, X_max_act])

            if scalers is not None and 'muscle_out_scaler' in scalers:
                muscle_out_scaler = scalers['muscle_out_scaler']
                predictions = muscle_out_scaler.inverse_transform(predictions)

            predicted_activation = predictions[0][0]
            print("predicted_activation:", predicted_activation)

            target_stiffness = reference_stiff[index, :] * (20 * predicted_activation)
            current_stiffness = DEFAULT_STIFFNESS.copy()

            if previous_stiffness is not None:
                delta_1 = target_stiffness[0] - previous_stiffness[0]
                if np.abs(delta_1) > MAX_STIFFNESS_CHANGE_RATE:
                    delta_1 = np.sign(delta_1) * MAX_STIFFNESS_CHANGE_RATE
                current_stiffness[0] = previous_stiffness[0] + delta_1

                delta_2 = target_stiffness[1] - previous_stiffness[1]
                if np.abs(delta_2) > MAX_STIFFNESS_CHANGE_RATE:
                    delta_2 = np.sign(delta_2) * MAX_STIFFNESS_CHANGE_RATE
                current_stiffness[1] = previous_stiffness[1] + delta_2

                delta_3 = target_stiffness[2] - previous_stiffness[2]
                if np.abs(delta_3) > MAX_STIFFNESS_CHANGE_RATE:
                    delta_3 = np.sign(delta_3) * MAX_STIFFNESS_CHANGE_RATE
                current_stiffness[2] = previous_stiffness[2] + delta_3
            else:
                current_stiffness = target_stiffness

            previous_stiffness = current_stiffness

            return current_stiffness

        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            traceback.print_exc()
            return DEFAULT_STIFFNESS

    elif STIFFNESS_MODE == 5:
        # 模式5: HI-ImpRS-Force (LSTM基于交互力估计的刚度,同时记录EMG激活度对比)
        if lstm_model is None or human_stiffness_profile is None:
            print("LSTM model or human stiffness profile not loaded")
            return DEFAULT_STIFFNESS

        if force_wrench is None:
            print("Waiting for force data...")
            return DEFAULT_STIFFNESS

        try:
            current_time = rospy.Time.now().to_sec()

            # 1. 计算基于EMG的激活度(用于对比)
            emg_activation = 0.0
            if emg_data is not None and len(emg_data) > 0:
                emg_activation = get_muscle_activation(emg_data)

            # 2. 基于交互力估计人体端点刚度
            estimated_human_stiffness = estimate_stiffness_from_force(index, force_wrench)

            # 3. 从human stiffness profile获取参考刚度
            if index < len(human_stiffness_profile):
                reference_human_stiffness = human_stiffness_profile[index, :]
            else:
                reference_human_stiffness = human_stiffness_profile[-1, :]

            # 4. 计算当前激活度 = 估计刚度 / 参考刚度
            reference_human_stiffness = np.maximum(reference_human_stiffness, 50)
            current_activation = estimated_human_stiffness / reference_human_stiffness

            # 取平均作为标量激活度
            scalar_activation = np.mean(current_activation)
            scalar_activation = np.clip(scalar_activation, 0.0, 0.4)  # 限制范围

            # 5. 更新激活度历史(使用force-based activation)
            update_activation_history(scalar_activation, actual_look_back)

            # 6. 准备LSTM输入
            if len(activation_history) < actual_look_back:
                print(f"Collecting activation history... ({len(activation_history)}/{actual_look_back})")
                return DEFAULT_STIFFNESS

            # 轨迹数据
            current_traj = reference_traj[max(0, index - actual_look_back + 1):index + 1, :3]
            if len(current_traj) < actual_look_back:
                padding = np.zeros((actual_look_back - len(current_traj), 3))
                current_traj = np.vstack([padding, current_traj])
            current_traj = current_traj[-actual_look_back:]

            # 激活度数据(从刚度估计获得)
            padded_history = activation_history.copy()
            while len(padded_history) < actual_look_back:
                padded_history.insert(0, 0)
            muscle_input = np.array(padded_history[-actual_look_back:]).reshape(-1, 1)

            # 最大激活度
            max_act = training_max_activation if training_max_activation is not None else 1.0
            max_act_input = np.ones((actual_look_back, 1)) * max_act

            # 7. 数据缩放
            if scalers is not None:
                traj_scaler = scalers.get('traj_scaler')
                muscle_in_scaler = scalers.get('muscle_in_scaler')
                max_act_scaler = scalers.get('max_act_scaler')

                if traj_scaler is not None:
                    current_traj = traj_scaler.transform(current_traj)

                if muscle_in_scaler is not None:
                    muscle_input = muscle_in_scaler.transform(muscle_input)

                if max_act_scaler is not None:
                    max_act_value = max_act_scaler.transform([[max_act]])[0][0]
                    max_act_input = np.ones((actual_look_back, 1)) * max_act_value

            # 8. LSTM预测
            X_traj = current_traj.reshape(1, actual_look_back, -1)
            X_muscle = muscle_input.reshape(1, actual_look_back, -1)
            X_max_act = max_act_input.reshape(1, actual_look_back, -1)

            predictions = lstm_model.predict([X_traj, X_muscle, X_max_act])

            if scalers is not None and 'muscle_out_scaler' in scalers:
                muscle_out_scaler = scalers['muscle_out_scaler']
                predictions = muscle_out_scaler.inverse_transform(predictions)

            predicted_activation = predictions[0][0]

            # 9. 计算目标刚度
            target_stiffness = reference_stiff[index, :] * predicted_activation
            current_stiffness = DEFAULT_STIFFNESS.copy()

            # 10. 平滑处理
            if previous_stiffness is not None:
                for i in range(3):
                    delta = target_stiffness[i] - previous_stiffness[i]
                    if np.abs(delta) > MAX_STIFFNESS_CHANGE_RATE:
                        delta = np.sign(delta) * MAX_STIFFNESS_CHANGE_RATE
                    current_stiffness[i] = previous_stiffness[i] + delta
            else:
                current_stiffness = target_stiffness

            previous_stiffness = current_stiffness

            # 11. 保存对比数据
            activation_comparison_data['timestamp'].append(current_time)
            activation_comparison_data['emg_activation'].append(emg_activation)
            activation_comparison_data['force_activation'].append(scalar_activation)
            activation_comparison_data['predicted_activation'].append(predicted_activation)
            activation_comparison_data['estimated_stiffness'].append(estimated_human_stiffness.copy())
            activation_comparison_data['reference_stiffness'].append(reference_human_stiffness.copy())
            activation_comparison_data['robot_stiffness'].append(current_stiffness.copy())

            # 打印对比信息
            print(
                f"EMG activation: {emg_activation:.3f}, Force activation: {scalar_activation:.3f}, Predicted: {predicted_activation:.3f}")
            print(f"Estimated human stiffness: {estimated_human_stiffness}")
            print(f"Robot stiffness: {current_stiffness}")

            return current_stiffness

        except Exception as e:
            print(f"Error in force-based LSTM prediction: {e}")
            traceback.print_exc()
            return DEFAULT_STIFFNESS

    return DEFAULT_STIFFNESS


def multi_callback(sub_torso, reference_traj, reference_stiff, torso_pub, time_array, index_counter,
                   emg_data, force_wrench, robot_right_position_init):
    sub_torso, time_array[index_counter] = transform_to_joint(sub_torso)

    if index_counter == 0:
        start_time = time_array[0]
        index = 0
    else:
        index = int((time_array[index_counter] - time_array[0]) * 1000)

    if index <= 10799:
        right_pos = reference_traj[index, :3] + robot_right_position_init
        left_pos = robot_left_position_init

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

        # 根据选择的模式计算刚度
        right_stiff = calculate_stiffness(
            index,
            reference_stiff,
            reference_traj,
            emg_data=emg_data,  # 模式5也需要EMG数据用于对比
            force_wrench=force_wrench
        )

        joint1_vel_ = rospy.get_param("joint1_vel", 0.08)

        torso_cmd = JointState()

        if index >= 0 and index < 2700:
            torso_joint1_vel = joint1_vel_
        elif index < 5400 and index >= 2700:
            torso_joint1_vel = -joint1_vel_
        elif index < 8100 and index >= 5400:
            torso_joint1_vel = joint1_vel_
        elif index < 10700 and index >= 8100:
            torso_joint1_vel = -joint1_vel_
        else:
            torso_joint1_vel = 0

        torso_cmd.velocity = [torso_joint1_vel, 0, 0, 0, 0, 0, 0]

        curi.set_impedance(1, 0, 2 * math.sqrt(right_stiff[0]), right_stiff[0])
        curi.set_impedance(1, 1, 2 * math.sqrt(right_stiff[1]), right_stiff[1])
        curi.set_impedance(1, 2, 2 * math.sqrt(right_stiff[2]), right_stiff[2])

        curi.set_tcp_servo(robot_left_pose_matrix, robot_right_pose_matrix)
        torso_pub.publish(torso_cmd)

        # 打印调试信息
        if STIFFNESS_MODE == 5:
            if force_wrench is not None:
                force = np.array([
                    force_wrench.wrench.force.x,
                    force_wrench.wrench.force.y,
                    force_wrench.wrench.force.z
                ])
                print(f"Force data: {force}")
        elif STIFFNESS_MODE == 4:
            if emg_data is not None and len(emg_data) > 0:
                print(f"EMG data: {emg_data[-1][-1]}")

        print(f"index: {index}")
        print(f"Using stiffness mode {STIFFNESS_MODE}, current stiffness: {right_stiff}")

        return index_counter + 1, False
    else:
        print("Trajectory completed!")
        # 保存激活度对比数据
        if STIFFNESS_MODE == 5:
            save_activation_comparison()
        return index_counter, True


if __name__ == '__main__':
    rospy.init_node('HI_ImpRS_hrc')
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Control robot with variable impedance.')
    parser.add_argument('--stiffness_mode', type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help='Stiffness mode: 1=Fixed, 2=Reference, 3=Muscle-Based, 4=HI-ImpRS-EMG, 5=HI-ImpRS-Force')
    args = parser.parse_args()

    STIFFNESS_MODE = args.stiffness_mode
    print(f"Selected stiffness mode: {STIFFNESS_MODE}")

    torso_pub = rospy.Publisher("/curi_torso/joint/cmd_vel", JointState, queue_size=10)

    roslaunch_process = launch_roslaunch()
    time.sleep(1)

    curi = Python_CURI_Control(0, [])
    curi.start()

    time.sleep(1)

    ## Initialization of robot end effector poses
    robot_left_position_init = np.array([0.7, 0.3, 0.65])
    robot_right_position_init = np.array([0.95, -0.2, 1.2])

    robot_left_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    robot_right_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    robot_left_pose_matrix_init = np.r_[
        np.c_[robot_left_rotation_matrix_init, robot_left_position_init.T], np.array([[0, 0, 0, 1]])]
    robot_right_pose_matrix_init = np.r_[
        np.c_[robot_right_rotation_matrix_init, robot_right_position_init.T], np.array([[0, 0, 0, 1]])]

    subscriber_torso = rospy.wait_for_message('/curi_torso/joint_states', JointState)
    sub_torso, _ = transform_to_joint(subscriber_torso)
    print(sub_torso)

    p, R = tsf.transform_torso_base_to_torso_end(sub_torso)
    T_TorsoBaseToTorsoEnd = np.r_[np.c_[R, p.T], np.array([[0, 0, 0, 1]])]
    print(T_TorsoBaseToTorsoEnd)
    T_MobileBaseToTorsoBase = np.array([[1, 0, 0, 0.2375], [0, 1, 0, 0], [0, 0, 1, 0.53762], [0, 0, 0, 1]])

    base2torso_matrix_init = np.linalg.inv(T_MobileBaseToTorsoBase @ T_TorsoBaseToTorsoEnd)
    print(base2torso_matrix_init)
    initial_robot_left_pose_matrix = base2torso_matrix_init @ robot_left_pose_matrix_init
    initial_robot_right_pose_matrix = base2torso_matrix_init @ robot_right_pose_matrix_init
    print("left", initial_robot_left_pose_matrix)
    print("right", initial_robot_right_pose_matrix)
    curi.set_tcp_moveL(initial_robot_left_pose_matrix, initial_robot_right_pose_matrix)

    while curi.get_curi_mode(0) != 2 and curi.get_curi_mode(1) != 2:
        print("waiting robot external control")
        time.sleep(1)

    reference_traj = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/taichi/traj_taichi_uni_5400.npy',
                             allow_pickle=True)
    reference_stiff = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/taichi/stiffness_results/stiff_wuxi_5400.npy',
                              allow_pickle=True)
    reference_traj = np.tile(reference_traj, (2, 1)).reshape(-1, 7)
    reference_stiff = np.tile(reference_stiff, (2, 1)).reshape(-1, 3)

    index_counter = 0
    time_array = np.zeros(100000)

    if STIFFNESS_MODE in [4, 5] and lstm_model is not None:
        warmup_lstm_model()

    torso_data = None
    force_data = None
    robot_pose_data = None

    # 模式5也需要EMG处理器用于对比
    folder = '/home/clover/Chenzui/HI-ImpRS-HRC/taichi/data_0621/wuxi/20'
    os.makedirs(folder, exist_ok=True)

    emg_processor = EMGProcessor(channel_num=2, sample_fre=200, start_time=None, save=True, save_folder=folder)
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


    def torso_callback(msg):
        global torso_data
        torso_data = msg


    def force_callback(msg):
        global force_data
        force_data = msg


    def robot_pose_callback(msg):
        """订阅机器人末端位姿"""
        global robot_pose_data
        robot_pose_data = msg


    torso_subscriber = rospy.Subscriber('/curi_torso/joint_states', JointState, torso_callback)

    # 力传感器订阅器(模式5需要)
    if STIFFNESS_MODE == 5:
        force_subscriber = rospy.Subscriber('/wrench', WrenchStamped, force_callback)
        print("Force sensor subscriber initialized")

        robot_pose_subscriber = rospy.Subscriber('/curi_arm/right/tcp_pose', PoseStamped, robot_pose_callback)
        print("Robot pose subscriber initialized")

    try:
        print("Starting trajectory execution...")
        print(f"Using stiffness mode: {STIFFNESS_MODE}")
        trajectory_completed = False

        while not rospy.is_shutdown() and not trajectory_completed:
            if torso_data is None:
                rospy.loginfo_throttle(1, "Waiting for torso data...")
                time.sleep(0.01)
                continue

            # 模式5需要等待力数据和位姿数据
            if STIFFNESS_MODE == 5:
                if force_data is None:
                    rospy.loginfo_throttle(1, "Waiting for force data...")
                    time.sleep(0.01)
                    continue
                if robot_pose_data is None:
                    rospy.loginfo_throttle(1, "Waiting for robot pose data...")
                    time.sleep(0.01)
                    continue

            # 获取EMG数据
            current_emg_data = emg_processor.all_emg_data if emg_processor is not None else None

            index_counter, trajectory_completed = multi_callback(
                torso_data,
                reference_traj,
                reference_stiff,
                torso_pub,
                time_array,
                index_counter,
                current_emg_data,
                force_data,
                robot_right_position_init
            )

            time.sleep(0.001)

        print("Execution finished.")
        if emg_processor is not None:
            emg_processor.read_emg_flag = False

        # 保存激活度对比数据
        if STIFFNESS_MODE == 5:
            save_activation_comparison()

        while not rospy.is_shutdown():
            interrupt = False
            time.sleep(1)

            if emg_processor is not None:
                data_queue.join()
                for t in threads:
                    t.join()

    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        # 即使出错也尝试保存数据
        if STIFFNESS_MODE == 5:
            save_activation_comparison()
    finally:
        if 'roslaunch_process' in globals():
            roslaunch_process.terminate()