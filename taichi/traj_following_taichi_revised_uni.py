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
import pickle
import traceback
import argparse
from keras.models import load_model

from libpython_curi_dual_arm_ic import Python_CURI_Control

# 获取 ROS 工作空间的路径
workspace_path = '/home/clover/catkin_ws'

# 添加编译后的库路径
sys.path.append(os.path.join(workspace_path, 'devel', 'lib'))
# export PYTHONPATH=$PYTHONPATH:/home/clover/catkin_ws/devel/lib


# 1: 定刚度, 2: 参考刚度变化轨迹, 3: 基于肌肉激活的变阻抗, 4: HI-ImpRS (LSTM)
STIFFNESS_MODE = 4
lstm_model = None
scalers = None
look_back = 10
training_max_activation = 0.2
activation_history = []
previous_stiffness = None
EMG_WINDOW_SIZE = 5
MAX_STIFFNESS_CHANGE_RATE = 5.0


# 加载LSTM模型（如果使用HI-ImpRS模式）
if STIFFNESS_MODE == 4:
    save_dir = os.path.expanduser('~/Chenzui//HI-ImpRS-HRC/LSTM/saved_multivariate_lstm_with_max_act')

    try:
        model_path = os.path.join(save_dir, 'multivariate_lstm_model.h5')
        lstm_model = load_model(model_path)
        print(f"Success to load model from: {model_path}")

        params_path = os.path.join(save_dir, 'params.pkl')
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        look_back = params.get('look_back', 10)
        training_max_activation = params.get('max_activation', 0.1)
        print(f"Success to load，look_back = {look_back}, maximum activation = {training_max_activation}")

        scaler_path = os.path.join(save_dir, 'scalers.pkl')
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        print(f"Success to load scalers: {scalers}")

    except Exception as e:
        print(f"Fail to load: {e}")
        traceback.print_exc()
        STIFFNESS_MODE = 1
        print("Falling back to default stiffness mode")

# 肌肉激活到刚度的映射参数（用于模式3）
MUSCLE_TO_STIFFNESS_PARAMS = {
    'min_activation': 0.01,
    'max_activation': 0.25,
    'min_stiffness': 100,  # N/m
    'max_stiffness': 1000  # N/m
}

# 默认固定刚度值（用于模式1）
DEFAULT_STIFFNESS = np.array([300, 300, 300])

current_muscle_activation = 0.1


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


def get_muscle_activation(emg_data):
    if emg_data is None or len(emg_data) == 0:
        return 0.1

    try:
        activation = np.mean(emg_data)
        activation = np.clip(activation, 0, 1)

        return activation
    except Exception as e:
        print(f"Error processing EMG data: {e}")
        return 0.1


def update_activation_history(activation, history_length=10):
    global activation_history

    # 添加新的激活度
    activation_history.append(activation)

    # 保持历史记录不超过指定长度
    if len(activation_history) > history_length:
        activation_history = activation_history[-history_length:]


def calculate_stiffness(index, reference_stiff, emg_data=None):
    global STIFFNESS_MODE, lstm_model, MUSCLE_TO_STIFFNESS_PARAMS, DEFAULT_STIFFNESS
    global scalers, look_back, activation_history, training_max_activation
    global previous_stiffness, EMG_WINDOW_SIZE, MAX_STIFFNESS_CHANGE_RATE

    actual_look_back = 10 if look_back is None else look_back

    muscle_activation = get_muscle_activation(emg_data)

    if STIFFNESS_MODE == 4:
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
            normalized_activation = np.clip(normalized_activation, 0, 1)  # 限制在[0,1]范围

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

            direction_weights = np.array([1.0, 1.0, 1.0])  # x, y, z 方向的权重
            return np.ones(3) * current_stiffness * direction_weights

        except Exception as e:
            print(f"Error in muscle-based stiffness calculation: {e}")
            traceback.print_exc()
            return DEFAULT_STIFFNESS


    elif STIFFNESS_MODE == 4:
        # 模式4: HI-ImpRS (LSTM预测)

        if lstm_model is None or len(activation_history) < actual_look_back:
            print("No LSTM model... ERROR!!!")
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
            # predicted_activation = np.clip(predicted_activation, 0, 1)
            print("predicted_activation:", predicted_activation)

            # params = MUSCLE_TO_STIFFNESS_PARAMS
            # stiffness_range = params['max_stiffness'] - params['min_stiffness']
            #
            # stiffness = params['min_stiffness'] + predicted_activation * stiffness_range
            #
            # direction_weights = np.array([1.0, 1.0, 1.0])
            return reference_stiff[index, :] * (50 * predicted_activation)

        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            traceback.print_exc()
            return DEFAULT_STIFFNESS

    return DEFAULT_STIFFNESS


def multi_callback(sub_torso, reference_traj, reference_stiff, torso_pub, time_array, index_counter, emg_data):
    sub_torso, time_array[index_counter] = transform_to_joint(sub_torso)
    # print(sub_torso)

    # 第一次调用时设置初始时间
    if index_counter == 0:
        start_time = time_array[0]
        index = 0
    else:
        index = int((time_array[index_counter] - time_array[0]) * 1000)

    # print(f"Index: {index}")

    if index <= 10799:
        right_pos = reference_traj[index, :3] + robot_right_position_init
        # right_pos = robot_right_position_init
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
        right_stiff = calculate_stiffness(index, reference_stiff, emg_data)

        # 获取关节速度参数
        joint1_vel_ = rospy.get_param("joint1_vel", 0.08)
        # joint3_vel_ = rospy.get_param("joint3_vel", 0.07)

        # 创建躯干命令消息
        torso_cmd = JointState()

        # 根据索引调整躯干关节速度
        if index < 2700:
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

        # 发布命令到机器人
        torso_pub.publish(torso_cmd)
        curi.set_tcp_servo(robot_left_pose_matrix, robot_right_pose_matrix)
        # curi.set_trans_impedance(0,trans_D, trans_K)

        curi.set_impedance(1, 0, 2 * math.sqrt(right_stiff[0]), right_stiff[0])
        curi.set_impedance(1, 1, 2 * math.sqrt(right_stiff[1]), right_stiff[1])
        curi.set_impedance(1, 2, 2 * math.sqrt(right_stiff[2]), right_stiff[2])

        print("EMG data", emg_data[-1][-1])
        print(f"Using stiffness mode {STIFFNESS_MODE}, current stiffness: {right_stiff}")
        return index_counter + 1, False
    else:
        print("Trajectory completed!")
        return index_counter, True


if __name__ == '__main__':
    rospy.init_node('HI_ImpRS_hrc')
    signal.signal(signal.SIGINT, signal_handler)

    # 命令行参数解析 - 允许用户选择刚度模式
    parser = argparse.ArgumentParser(description='Control robot with variable impedance.')
    parser.add_argument('--stiffness_mode', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Stiffness mode: 1=Fixed, 2=Reference, 3=Muscle-Based, 4=HI-ImpRS')
    args = parser.parse_args()

    STIFFNESS_MODE = args.stiffness_mode
    print(f"Selected stiffness mode: {STIFFNESS_MODE}")

    # 创建发布器
    torso_pub = rospy.Publisher("/curi_torso/joint/cmd_vel", JointState, queue_size=10)
    # impe_r_pub = rospy.Publisher("/curi_arm/right/impedance", Float64MultiArray, queue_size=10)

    # 启动 roslaunch
    roslaunch_process = launch_roslaunch()
    time.sleep(1)  # 等待一段时间以确保 ROS 节点启动
    # 启动控制器

    curi = Python_CURI_Control(0, [])
    curi.start()

    time.sleep(1)
    # vrpn_roslaunch_process = vrpn_launch_roslaunch()

    ## Initialization of robot end effector poses
    robot_left_position_init = np.array([0.7, 0.3, 0.5])
    robot_right_position_init = np.array([1.0, -0.5, 0.85])

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

    reference_traj = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/taichi/traj_taichi_uni_5400.npy', allow_pickle=True)
    reference_stiff = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/taichi/stiff_taichi_uni_5400.npy', allow_pickle=True)
    reference_traj = np.tile(reference_traj, (2, 1)).reshape(-1, 7)
    reference_stiff = np.tile(reference_stiff, (2, 1)).reshape(-1, 3)

    # 准备控制循环所需变量
    index_counter = 0
    time_array = np.zeros(100000)

    # 创建躯干数据订阅器
    torso_data = None
    folder = '/home/clover/Chenzui/HI-ImpRS-HRC/taichi/data_0620/zhuo/20'
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


    torso_subscriber = rospy.Subscriber('/curi_torso/joint_states', JointState, torso_callback)

    try:
        print("Starting trajectory execution...")
        print(f"Using stiffness mode: {STIFFNESS_MODE}")
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
        # 清理资源
        if 'roslaunch_process' in globals():
            roslaunch_process.terminate()
        # if 'vrpn_roslaunch_process' in globals():
        #     vrpn_roslaunch_process.terminate()
