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

# 新增:Z方向力控制相关参数
FORCE_CONTROL_PARAMS = {
    'desired_force': -10.0,  # 期望的Z方向接触力(负值表示向下压) (N)
    'Kp_force': 0.5,  # 力控制比例增益 (m/N)
    'Kd_force': 0.05,  # 力控制微分增益 (m·s/N)
    'max_position_adjustment': 0.01,  # 最大位置调整量 (m)
    'force_deadband': 1.0,  # 力控制死区 (N)
    'min_stiffness_z': 50.0,  # Z方向最小刚度 (N/m)
    'max_stiffness_z': 500.0  # Z方向最大刚度 (N/m)
}

# Z方向力控制状态变量
force_control_state = {
    'last_force_error': 0.0,
    'z_position_adjustment': 0.0,
    'force_history': []
}

# 统一的实验数据记录(所有模式通用)
experiment_data = {
    'timestamp': [],
    'index': [],
    'mode': [],  # 刚度模式

    # 交互力数据
    'force_x': [],
    'force_y': [],
    'force_z': [],
    'torque_x': [],
    'torque_y': [],
    'torque_z': [],

    # 位置数据
    'pos_x': [],
    'pos_y': [],
    'pos_z': [],
    'pos_ref_x': [],  # 参考位置
    'pos_ref_y': [],
    'pos_ref_z': [],

    # 刚度数据
    'stiffness_x': [],
    'stiffness_y': [],
    'stiffness_z': [],
    'stiffness_ref_x': [],  # 参考刚度
    'stiffness_ref_y': [],
    'stiffness_ref_z': [],

    # EMG数据
    'emg_activation': [],

    # 模式5特有: 基于力的激活度
    'force_activation': [],  # 基于力估计的激活度
    'predicted_activation': [],  # LSTM预测的激活度
    'estimated_human_stiffness_x': [],
    'estimated_human_stiffness_y': [],

    # Z方向力控制
    'z_force_error': [],
    'z_position_adjustment': [],
    'z_stiffness_adapted': []  # Z方向自适应刚度
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
DEFAULT_STIFFNESS = np.array([400, 400, 200])  # Z方向使用较低的默认刚度

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

    # 保存实验数据
    if len(experiment_data['timestamp']) > 0:
        save_experiment_data()

    rospy.signal_shutdown("shutdown by manual")
    if 'roslaunch_process' in locals():
        print('Shutdown roslaunch process.')
        roslaunch_process.terminate()
        roslaunch_process.wait()
    print('Python shutdown.')
    sys.exit(0)


def save_experiment_data():
    """保存实验数据(所有模式通用)"""
    try:
        # 保存为numpy格式
        data_file = os.path.join(folder, f'experiment_data_mode_{STIFFNESS_MODE}.npy')
        np.save(data_file, experiment_data)
        print(f"Experiment data saved to {data_file}")

        # 保存为CSV格式
        csv_file = os.path.join(folder, f'experiment_data_mode_{STIFFNESS_MODE}.csv')
        import pandas as pd

        # 构建DataFrame
        df_dict = {
            'timestamp': experiment_data['timestamp'],
            'index': experiment_data['index'],
            'mode': experiment_data['mode'],

            # 力数据
            'force_x': experiment_data['force_x'],
            'force_y': experiment_data['force_y'],
            'force_z': experiment_data['force_z'],
            'torque_x': experiment_data['torque_x'],
            'torque_y': experiment_data['torque_y'],
            'torque_z': experiment_data['torque_z'],

            # 位置数据
            'pos_x': experiment_data['pos_x'],
            'pos_y': experiment_data['pos_y'],
            'pos_z': experiment_data['pos_z'],
            'pos_ref_x': experiment_data['pos_ref_x'],
            'pos_ref_y': experiment_data['pos_ref_y'],
            'pos_ref_z': experiment_data['pos_ref_z'],

            # 刚度数据
            'stiffness_x': experiment_data['stiffness_x'],
            'stiffness_y': experiment_data['stiffness_y'],
            'stiffness_z': experiment_data['stiffness_z'],
            'stiffness_ref_x': experiment_data['stiffness_ref_x'],
            'stiffness_ref_y': experiment_data['stiffness_ref_y'],
            'stiffness_ref_z': experiment_data['stiffness_ref_z'],

            # EMG
            'emg_activation': experiment_data['emg_activation'],

            # Z方向力控制
            'z_force_error': experiment_data['z_force_error'],
            'z_position_adjustment': experiment_data['z_position_adjustment'],
            'z_stiffness_adapted': experiment_data['z_stiffness_adapted']
        }

        # 模式5特有数据
        if STIFFNESS_MODE == 5:
            df_dict.update({
                'force_activation': experiment_data['force_activation'],
                'predicted_activation': experiment_data['predicted_activation'],
                'estimated_human_stiffness_x': experiment_data['estimated_human_stiffness_x'],
                'estimated_human_stiffness_y': experiment_data['estimated_human_stiffness_y']
            })

        df = pd.DataFrame(df_dict)
        df.to_csv(csv_file, index=False)
        print(f"Experiment data CSV saved to {csv_file}")

        # 生成可视化图表
        plot_experiment_data(folder)

    except Exception as e:
        print(f"Error saving experiment data: {e}")
        traceback.print_exc()


def plot_experiment_data(save_folder):
    """绘制实验数据可视化图表"""
    try:
        timestamps = np.array(experiment_data['timestamp'])

        # 创建多子图
        if STIFFNESS_MODE == 5:
            fig, axes = plt.subplots(5, 1, figsize=(16, 18))
        else:
            fig, axes = plt.subplots(4, 1, figsize=(16, 14))

        # 子图1: 交互力
        ax1 = axes[0]
        ax1.plot(timestamps, experiment_data['force_x'], 'r-', label='Force X', linewidth=1.2, alpha=0.8)
        ax1.plot(timestamps, experiment_data['force_y'], 'g-', label='Force Y', linewidth=1.2, alpha=0.8)
        ax1.plot(timestamps, experiment_data['force_z'], 'b-', label='Force Z', linewidth=1.2, alpha=0.8)
        ax1.axhline(y=FORCE_CONTROL_PARAMS['desired_force'], color='b',
                    linestyle='--', label='Desired Z Force', linewidth=1.5, alpha=0.6)
        ax1.set_ylabel('Force (N)', fontsize=11)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Interaction Forces - Mode {STIFFNESS_MODE}', fontsize=12, fontweight='bold')

        # 子图2: 位置追踪
        ax2 = axes[1]
        ax2.plot(timestamps, experiment_data['pos_x'], 'r-', label='Actual X', linewidth=1.2, alpha=0.7)
        ax2.plot(timestamps, experiment_data['pos_ref_x'], 'r--', label='Ref X', linewidth=1.0, alpha=0.5)
        ax2.plot(timestamps, experiment_data['pos_y'], 'g-', label='Actual Y', linewidth=1.2, alpha=0.7)
        ax2.plot(timestamps, experiment_data['pos_ref_y'], 'g--', label='Ref Y', linewidth=1.0, alpha=0.5)
        ax2.plot(timestamps, experiment_data['pos_z'], 'b-', label='Actual Z', linewidth=1.2, alpha=0.7)
        ax2.plot(timestamps, experiment_data['pos_ref_z'], 'b--', label='Ref Z', linewidth=1.0, alpha=0.5)
        ax2.set_ylabel('Position (m)', fontsize=11)
        ax2.legend(loc='upper right', fontsize=9, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Position Tracking', fontsize=12, fontweight='bold')

        # 子图3: 刚度对比
        ax3 = axes[2]
        ax3.plot(timestamps, experiment_data['stiffness_x'], 'r-', label='Stiffness X', linewidth=1.5)
        ax3.plot(timestamps, experiment_data['stiffness_y'], 'g-', label='Stiffness Y', linewidth=1.5)
        ax3.plot(timestamps, experiment_data['stiffness_z'], 'b-', label='Stiffness Z (Force Control)', linewidth=1.5)
        if STIFFNESS_MODE in [2, 5]:
            ax3.plot(timestamps, experiment_data['stiffness_ref_x'], 'r--',
                     label='Ref X', linewidth=1.0, alpha=0.5)
            ax3.plot(timestamps, experiment_data['stiffness_ref_y'], 'g--',
                     label='Ref Y', linewidth=1.0, alpha=0.5)
        ax3.set_ylabel('Stiffness (N/m)', fontsize=11)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Robot Stiffness', fontsize=12, fontweight='bold')

        # 子图4: Z方向力控制
        ax4 = axes[3]
        ax4_twin = ax4.twinx()

        line1 = ax4.plot(timestamps, experiment_data['z_force_error'],
                         'r-', label='Z Force Error', linewidth=1.5)
        line2 = ax4_twin.plot(timestamps, experiment_data['z_position_adjustment'],
                              'b-', label='Position Adjustment', linewidth=1.5)
        line3 = ax4_twin.plot(timestamps, experiment_data['z_stiffness_adapted'],
                              'g--', label='Adapted Z Stiffness', linewidth=1.5, alpha=0.7)

        ax4.set_ylabel('Force Error (N)', fontsize=11, color='r')
        ax4_twin.set_ylabel('Adjustment (m) / Stiffness (N/m)', fontsize=11, color='b')
        ax4.set_xlabel('Time (s)', fontsize=11)

        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right', fontsize=9)

        ax4.grid(True, alpha=0.3)
        ax4.set_title('Z-axis Force Control Performance', fontsize=12, fontweight='bold')

        # 子图5: 模式5特有 - 激活度对比
        if STIFFNESS_MODE == 5:
            ax5 = axes[4]
            ax5.plot(timestamps, experiment_data['emg_activation'],
                     'b-', label='EMG Activation', linewidth=1.5, alpha=0.7)
            ax5.plot(timestamps, experiment_data['force_activation'],
                     'r-', label='Force-based Activation', linewidth=1.5, alpha=0.7)
            ax5.plot(timestamps, experiment_data['predicted_activation'],
                     'g-', label='LSTM Predicted Activation', linewidth=1.5)
            ax5.set_ylabel('Activation', fontsize=11)
            ax5.set_xlabel('Time (s)', fontsize=11)
            ax5.legend(loc='upper right', fontsize=9)
            ax5.grid(True, alpha=0.3)
            ax5.set_title('Activation Comparison (Mode 5)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plot_file = os.path.join(save_folder, f'experiment_visualization_mode_{STIFFNESS_MODE}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Experiment visualization saved to {plot_file}")
        plt.close()

        # 额外生成力-位置关系图
        plot_force_position_relationship(save_folder)

    except Exception as e:
        print(f"Error plotting experiment data: {e}")
        traceback.print_exc()


def plot_force_position_relationship(save_folder):
    """绘制力-位置关系图"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # XY平面力分布
        ax1 = axes[0, 0]
        scatter = ax1.scatter(experiment_data['pos_x'], experiment_data['pos_y'],
                              c=experiment_data['force_z'], cmap='viridis',
                              s=10, alpha=0.6)
        ax1.set_xlabel('Position X (m)', fontsize=10)
        ax1.set_ylabel('Position Y (m)', fontsize=10)
        ax1.set_title('Z Force Distribution in XY Plane', fontsize=11, fontweight='bold')
        plt.colorbar(scatter, ax=ax1, label='Force Z (N)')
        ax1.grid(True, alpha=0.3)

        # Z方向力-位置关系
        ax2 = axes[0, 1]
        ax2.scatter(experiment_data['pos_z'], experiment_data['force_z'],
                    s=10, alpha=0.5, c='blue')
        ax2.axhline(y=FORCE_CONTROL_PARAMS['desired_force'], color='r',
                    linestyle='--', label='Desired Force', linewidth=2)
        ax2.set_xlabel('Position Z (m)', fontsize=10)
        ax2.set_ylabel('Force Z (N)', fontsize=10)
        ax2.set_title('Z Force vs Position', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 刚度-力误差关系
        ax3 = axes[1, 0]
        ax3.scatter(experiment_data['z_force_error'], experiment_data['z_stiffness_adapted'],
                    s=10, alpha=0.5, c='green')
        ax3.set_xlabel('Z Force Error (N)', fontsize=10)
        ax3.set_ylabel('Adapted Z Stiffness (N/m)', fontsize=10)
        ax3.set_title('Stiffness Adaptation vs Force Error', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # XY方向力大小
        ax4 = axes[1, 1]
        force_xy_magnitude = np.sqrt(np.array(experiment_data['force_x']) ** 2 +
                                     np.array(experiment_data['force_y']) ** 2)
        timestamps = np.array(experiment_data['timestamp'])
        ax4.plot(timestamps, force_xy_magnitude, 'purple', linewidth=1.2, label='XY Force Magnitude')
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Force Magnitude (N)', fontsize=10)
        ax4.set_title('XY Plane Force Magnitude', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(save_folder, f'force_position_analysis_mode_{STIFFNESS_MODE}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Force-position analysis saved to {plot_file}")
        plt.close()

    except Exception as e:
        print(f"Error plotting force-position relationship: {e}")
        traceback.print_exc()


def z_force_controller(current_z_force, dt=0.001):
    """
    Z方向PD力控制器

    参数:
        current_z_force: 当前Z方向力 (N)
        dt: 时间步长 (s)

    返回:
        z_adjustment: Z方向位置调整量 (m)
        z_stiffness: Z方向刚度 (N/m)
    """
    global force_control_state, FORCE_CONTROL_PARAMS

    # 计算力误差
    desired_force = FORCE_CONTROL_PARAMS['desired_force']
    force_error = desired_force - current_z_force

    # 力死区处理
    if abs(force_error) < FORCE_CONTROL_PARAMS['force_deadband']:
        force_error = 0.0

    # 平滑力信号(移动平均)
    force_control_state['force_history'].append(current_z_force)
    if len(force_control_state['force_history']) > 5:
        force_control_state['force_history'].pop(0)
    smoothed_force = np.mean(force_control_state['force_history'])
    smoothed_error = desired_force - smoothed_force

    # PD控制
    Kp = FORCE_CONTROL_PARAMS['Kp_force']
    Kd = FORCE_CONTROL_PARAMS['Kd_force']

    # 比例项
    p_term = Kp * smoothed_error

    # 微分项
    if force_control_state['last_force_error'] is not None:
        force_error_rate = (smoothed_error - force_control_state['last_force_error']) / dt
        d_term = Kd * force_error_rate
    else:
        d_term = 0.0

    # 总的位置调整
    z_adjustment = p_term + d_term

    # 限制调整幅度
    max_adj = FORCE_CONTROL_PARAMS['max_position_adjustment']
    z_adjustment = np.clip(z_adjustment, -max_adj, max_adj)

    # 累积调整
    force_control_state['z_position_adjustment'] = z_adjustment

    # 更新状态
    force_control_state['last_force_error'] = smoothed_error

    # 根据力误差自适应调整Z方向刚度
    error_ratio = abs(smoothed_error) / (abs(desired_force) + 1e-6)
    error_ratio = np.clip(error_ratio, 0, 1)

    min_K = FORCE_CONTROL_PARAMS['min_stiffness_z']
    max_K = FORCE_CONTROL_PARAMS['max_stiffness_z']

    # 误差大->刚度小, 误差小->刚度大
    z_stiffness = max_K - (max_K - min_K) * error_ratio

    return z_adjustment, z_stiffness


def record_experiment_data(index, current_time, force_wrench, current_position,
                           reference_position, current_stiffness, reference_stiffness,
                           emg_activation=None, force_activation=None, predicted_activation=None,
                           estimated_human_stiffness_xy=None):
    """
    记录实验数据(所有模式通用)
    """
    global experiment_data, STIFFNESS_MODE, FORCE_CONTROL_PARAMS

    try:
        # 基础数据
        experiment_data['timestamp'].append(current_time)
        experiment_data['index'].append(index)
        experiment_data['mode'].append(STIFFNESS_MODE)

        # 交互力数据
        if force_wrench is not None:
            experiment_data['force_x'].append(force_wrench.wrench.force.x)
            experiment_data['force_y'].append(force_wrench.wrench.force.y)
            experiment_data['force_z'].append(force_wrench.wrench.force.z)
            experiment_data['torque_x'].append(force_wrench.wrench.torque.x)
            experiment_data['torque_y'].append(force_wrench.wrench.torque.y)
            experiment_data['torque_z'].append(force_wrench.wrench.torque.z)
        else:
            experiment_data['force_x'].append(0.0)
            experiment_data['force_y'].append(0.0)
            experiment_data['force_z'].append(0.0)
            experiment_data['torque_x'].append(0.0)
            experiment_data['torque_y'].append(0.0)
            experiment_data['torque_z'].append(0.0)

        # 位置数据
        experiment_data['pos_x'].append(current_position[0])
        experiment_data['pos_y'].append(current_position[1])
        experiment_data['pos_z'].append(current_position[2])
        experiment_data['pos_ref_x'].append(reference_position[0])
        experiment_data['pos_ref_y'].append(reference_position[1])
        experiment_data['pos_ref_z'].append(reference_position[2])

        # 刚度数据
        experiment_data['stiffness_x'].append(current_stiffness[0])
        experiment_data['stiffness_y'].append(current_stiffness[1])
        experiment_data['stiffness_z'].append(current_stiffness[2])
        experiment_data['stiffness_ref_x'].append(reference_stiffness[0])
        experiment_data['stiffness_ref_y'].append(reference_stiffness[1])
        experiment_data['stiffness_ref_z'].append(reference_stiffness[2])

        # EMG数据
        experiment_data['emg_activation'].append(emg_activation if emg_activation is not None else 0.0)

        # Z方向力控制数据
        if force_wrench is not None:
            z_force_error = FORCE_CONTROL_PARAMS['desired_force'] - force_wrench.wrench.force.z
        else:
            z_force_error = 0.0
        experiment_data['z_force_error'].append(z_force_error)
        experiment_data['z_position_adjustment'].append(force_control_state['z_position_adjustment'])
        experiment_data['z_stiffness_adapted'].append(current_stiffness[2])

        # 模式5特有数据
        if STIFFNESS_MODE == 5:
            experiment_data['force_activation'].append(force_activation if force_activation is not None else 0.0)
            experiment_data['predicted_activation'].append(
                predicted_activation if predicted_activation is not None else 0.0)
            if estimated_human_stiffness_xy is not None:
                experiment_data['estimated_human_stiffness_x'].append(estimated_human_stiffness_xy[0])
                experiment_data['estimated_human_stiffness_y'].append(estimated_human_stiffness_xy[1])
            else:
                experiment_data['estimated_human_stiffness_x'].append(0.0)
                experiment_data['estimated_human_stiffness_y'].append(0.0)

    except Exception as e:
        print(f"Error recording experiment data: {e}")
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
        activation = np.mean(emg_data[-1])
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
    基于交互力和机器人位置估计人体端点刚度 (仅用于XY方向)
    """
    global stiffness_estimators, robot_position_history, robot_velocity_history
    global last_position, last_time, estimated_stiffness_history, robot_pose_data

    if force_wrench is None or stiffness_estimators is None or robot_pose_data is None:
        return np.array([500.0, 500.0])

    try:
        current_position = np.array([
            robot_pose_data.pose.position.x,
            robot_pose_data.pose.position.y,
            robot_pose_data.pose.position.z
        ])

        force = np.array([
            force_wrench.wrench.force.x,
            force_wrench.wrench.force.y,
            force_wrench.wrench.force.z
        ])

        current_time = force_wrench.header.stamp.secs + 1e-9 * force_wrench.header.stamp.nsecs

        if last_position is not None and last_time is not None:
            dt = current_time - last_time
            if dt > 0:
                velocity = (current_position - last_position) / dt
            else:
                velocity = np.zeros(3)
        else:
            velocity = np.zeros(3)

        last_position = current_position.copy()
        last_time = current_time

        estimated_stiffness = np.zeros(2)
        axes = ['x', 'y']

        for i, axis in enumerate(axes):
            result = stiffness_estimators[axis].step(
                f_meas=force[i],
                x_c=current_position[i],
                x_dot_c=velocity[i]
            )
            estimated_stiffness[i] = result['stiffness']

        estimated_stiffness = np.clip(estimated_stiffness, 50.0, 2000.0)

        return estimated_stiffness

    except Exception as e:
        print(f"Error in stiffness estimation from force: {e}")
        traceback.print_exc()
        return np.array([500.0, 500.0])


def calculate_stiffness(index, reference_stiff, reference_traj, emg_data=None, force_wrench=None):
    """
    计算三轴刚度并返回额外信息用于数据记录

    返回:
        tuple: (final_stiffness, emg_activation, force_activation, predicted_activation, estimated_human_stiffness_xy)
    """
    global STIFFNESS_MODE, lstm_model, MUSCLE_TO_STIFFNESS_PARAMS, DEFAULT_STIFFNESS
    global scalers, look_back, activation_history, training_max_activation
    global previous_stiffness, EMG_WINDOW_SIZE, MAX_STIFFNESS_CHANGE_RATE
    global human_stiffness_profile

    actual_look_back = 10 if look_back is None else look_back

    # 初始化返回值
    xy_stiffness = DEFAULT_STIFFNESS[:2].copy()
    z_stiffness = FORCE_CONTROL_PARAMS['min_stiffness_z']

    # 初始化额外数据
    emg_activation = None
    force_activation = None
    predicted_activation = None
    estimated_human_stiffness_xy = None

    # 计算EMG激活度(所有模式)
    if emg_data is not None and len(emg_data) > 0:
        emg_activation = get_muscle_activation(emg_data)

    # 模式4需要EMG数据
    if STIFFNESS_MODE == 4:
        if emg_activation is not None:
            update_activation_history(emg_activation, look_back)

    if STIFFNESS_MODE == 1:
        xy_stiffness = DEFAULT_STIFFNESS[:2]

    elif STIFFNESS_MODE == 2:
        xy_stiffness = reference_stiff[index, :2]

    elif STIFFNESS_MODE == 3:
        if emg_data is None or len(emg_data) == 0:
            xy_stiffness = DEFAULT_STIFFNESS[:2]
        else:
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
                    delta = target_stiffness - previous_stiffness[0]
                    if np.abs(delta) > MAX_STIFFNESS_CHANGE_RATE:
                        delta = np.sign(delta) * MAX_STIFFNESS_CHANGE_RATE
                    current_stiffness_scalar = previous_stiffness[0] + delta
                else:
                    current_stiffness_scalar = target_stiffness

                xy_stiffness = np.array([current_stiffness_scalar, current_stiffness_scalar])

            except Exception as e:
                print(f"Error in muscle-based stiffness calculation: {e}")
                traceback.print_exc()
                xy_stiffness = DEFAULT_STIFFNESS[:2]

    elif STIFFNESS_MODE == 4:
        if lstm_model is None or len(activation_history) < actual_look_back:
            print("Collecting EMG data...")
            xy_stiffness = DEFAULT_STIFFNESS[:2]
        else:
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

                target_stiffness_xy = reference_stiff[index, :2] * (20 * predicted_activation)
                current_stiffness_xy = DEFAULT_STIFFNESS[:2].copy()

                if previous_stiffness is not None:
                    for i in range(2):
                        delta = target_stiffness_xy[i] - previous_stiffness[i]
                        if np.abs(delta) > MAX_STIFFNESS_CHANGE_RATE:
                            delta = np.sign(delta) * MAX_STIFFNESS_CHANGE_RATE
                        current_stiffness_xy[i] = previous_stiffness[i] + delta
                else:
                    current_stiffness_xy = target_stiffness_xy

                xy_stiffness = current_stiffness_xy

            except Exception as e:
                print(f"Error in LSTM prediction: {e}")
                traceback.print_exc()
                xy_stiffness = DEFAULT_STIFFNESS[:2]

    elif STIFFNESS_MODE == 5:
        if lstm_model is None or human_stiffness_profile is None:
            print("LSTM model or human stiffness profile not loaded")
            xy_stiffness = DEFAULT_STIFFNESS[:2]
        elif force_wrench is None:
            print("Waiting for force data...")
            xy_stiffness = DEFAULT_STIFFNESS[:2]
        else:
            try:
                # 基于交互力估计人体端点刚度
                estimated_human_stiffness_xy = estimate_stiffness_from_force(index, force_wrench)

                # 从human stiffness profile获取参考刚度
                if index < len(human_stiffness_profile):
                    reference_human_stiffness = human_stiffness_profile[index, :2]
                else:
                    reference_human_stiffness = human_stiffness_profile[-1, :2]

                # 计算当前激活度
                reference_human_stiffness = np.maximum(reference_human_stiffness, 50)
                current_activation = estimated_human_stiffness_xy / reference_human_stiffness

                force_activation = np.mean(current_activation)
                force_activation = np.clip(force_activation, 0.0, 0.4)

                # 更新激活度历史
                update_activation_history(force_activation, actual_look_back)

                if len(activation_history) < actual_look_back:
                    print(f"Collecting activation history... ({len(activation_history)}/{actual_look_back})")
                    xy_stiffness = DEFAULT_STIFFNESS[:2]
                else:
                    # LSTM预测 (与模式4相同的流程)
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

                    target_stiffness_xy = reference_stiff[index, :2] * (20 * predicted_activation)
                    current_stiffness_xy = DEFAULT_STIFFNESS[:2].copy()

                    if previous_stiffness is not None:
                        for i in range(2):
                            delta = target_stiffness_xy[i] - previous_stiffness[i]
                            if np.abs(delta) > MAX_STIFFNESS_CHANGE_RATE:
                                delta = np.sign(delta) * MAX_STIFFNESS_CHANGE_RATE
                            current_stiffness_xy[i] = previous_stiffness[i] + delta
                    else:
                        current_stiffness_xy = target_stiffness_xy

                    xy_stiffness = current_stiffness_xy

            except Exception as e:
                print(f"Error in force-based LSTM prediction: {e}")
                traceback.print_exc()
                xy_stiffness = DEFAULT_STIFFNESS[:2]

    # Z方向力控制 (所有模式通用)
    if force_wrench is not None:
        try:
            current_z_force = force_wrench.wrench.force.z
            z_adjustment, z_stiffness = z_force_controller(current_z_force)
        except Exception as e:
            print(f"Error in Z force control: {e}")
            z_stiffness = FORCE_CONTROL_PARAMS['min_stiffness_z']
    else:
        z_stiffness = FORCE_CONTROL_PARAMS['min_stiffness_z']

    # 组合XYZ刚度
    final_stiffness = np.array([xy_stiffness[0], xy_stiffness[1], z_stiffness])

    # 更新previous_stiffness (只保存XY)
    previous_stiffness = xy_stiffness.copy()

    return final_stiffness, emg_activation, force_activation, predicted_activation, estimated_human_stiffness_xy


def multi_callback(sub_torso, reference_traj, reference_stiff, torso_pub, time_array, index_counter,
                   emg_data, force_wrench, robot_right_position_init):
    sub_torso, time_array[index_counter] = transform_to_joint(sub_torso)

    if index_counter == 0:
        start_time = time_array[0]
        index = 0
    else:
        index = int((time_array[index_counter] - time_array[0]) * 1000)

    if index <= 10799:
        # 基准参考轨迹
        right_pos_ref = reference_traj[index, :3] + robot_right_position_init

        # Z方向力控制位置调整
        z_adjustment = force_control_state['z_position_adjustment']
        right_pos = right_pos_ref.copy()
        right_pos[2] += z_adjustment

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

        # 计算刚度并获取额外数据
        right_stiff, emg_activation, force_activation, predicted_activation, estimated_human_stiffness_xy = calculate_stiffness(
            index,
            reference_stiff,
            reference_traj,
            emg_data=emg_data,
            force_wrench=force_wrench
        )

        # 记录实验数据(所有模式)
        current_time = rospy.Time.now().to_sec()
        record_experiment_data(
            index=index,
            current_time=current_time,
            force_wrench=force_wrench,
            current_position=right_pos,
            reference_position=right_pos_ref,
            current_stiffness=right_stiff,
            reference_stiffness=reference_stiff[index, :],
            emg_activation=emg_activation,
            force_activation=force_activation,
            predicted_activation=predicted_activation,
            estimated_human_stiffness_xy=estimated_human_stiffness_xy
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

        # 设置三轴阻抗
        curi.set_impedance(1, 0, 2 * math.sqrt(right_stiff[0]), right_stiff[0])
        curi.set_impedance(1, 1, 2 * math.sqrt(right_stiff[1]), right_stiff[1])
        curi.set_impedance(1, 2, 2 * math.sqrt(right_stiff[2]), right_stiff[2])

        curi.set_tcp_servo(robot_left_pose_matrix, robot_right_pose_matrix)
        torso_pub.publish(torso_cmd)

        # 打印调试信息
        if force_wrench is not None:
            force = np.array([
                force_wrench.wrench.force.x,
                force_wrench.wrench.force.y,
                force_wrench.wrench.force.z
            ])
            print(f"Force XYZ: {force}")

        print(f"index: {index}")
        print(f"Z position adjustment: {z_adjustment:.4f}m")
        print(f"Using stiffness mode {STIFFNESS_MODE}, current stiffness XYZ: {right_stiff}")

        return index_counter + 1, False
    else:
        print("Trajectory completed!")
        save_experiment_data()
        return index_counter, True


if __name__ == '__main__':
    rospy.init_node('HI_ImpRS_hrc_sawing')
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Control robot with variable impedance for sawing task.')
    parser.add_argument('--stiffness_mode', type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help='Stiffness mode: 1=Fixed, 2=Reference, 3=Muscle-Based, 4=HI-ImpRS-EMG, 5=HI-ImpRS-Force')
    parser.add_argument('--desired_force', type=float, default=-10.0,
                        help='Desired Z-axis contact force (N, negative for downward)')
    args = parser.parse_args()

    STIFFNESS_MODE = args.stiffness_mode
    FORCE_CONTROL_PARAMS['desired_force'] = args.desired_force

    print(f"Selected stiffness mode: {STIFFNESS_MODE}")
    print(f"Desired Z-axis force: {FORCE_CONTROL_PARAMS['desired_force']} N")

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

    folder = '/home/clover/Chenzui/HI-ImpRS-HRC/taichi/data_sawing/1010'
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
        global robot_pose_data
        robot_pose_data = msg


    torso_subscriber = rospy.Subscriber('/curi_torso/joint_states', JointState, torso_callback)

    # 力传感器订阅器(所有模式都需要)
    force_subscriber = rospy.Subscriber('/wrench', WrenchStamped, force_callback)
    print("Force sensor subscriber initialized (required for all modes)")

    robot_pose_subscriber = rospy.Subscriber('/curi_arm/right/tcp_pose', PoseStamped, robot_pose_callback)
    print("Robot pose subscriber initialized")

    try:
        print("Starting sawing task execution...")
        print(f"Using stiffness mode: {STIFFNESS_MODE}")
        print(f"XY: Variable impedance, Z: Force control (target={FORCE_CONTROL_PARAMS['desired_force']}N)")
        print("Force data will be recorded for all modes")
        trajectory_completed = False

        while not rospy.is_shutdown() and not trajectory_completed:
            if torso_data is None:
                rospy.loginfo_throttle(1, "Waiting for torso data...")
                time.sleep(0.01)
                continue

            if force_data is None:
                rospy.loginfo_throttle(1, "Waiting for force data...")
                time.sleep(0.01)
                continue
            if robot_pose_data is None:
                rospy.loginfo_throttle(1, "Waiting for robot pose data...")
                time.sleep(0.01)
                continue

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

        save_experiment_data()

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
        save_experiment_data()
    finally:
        if 'roslaunch_process' in globals():
            roslaunch_process.terminate()