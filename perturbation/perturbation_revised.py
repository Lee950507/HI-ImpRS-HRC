
import numpy as np
import matplotlib.pyplot as plt
import transformation as tsf
import math
import sys
import os
import rospy
import signal
import subprocess
import time

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int8, String, Bool, Float64MultiArray
from sensor_msgs.msg import JointState

# 获取 ROS 工作空间的路径
workspace_path = '/home/clover/catkin_ws'

# 添加编译后的库路径
sys.path.append(os.path.join(workspace_path, 'devel', 'lib'))


from libpython_curi_dual_arm_ic import Python_CURI_Control


def launch_roslaunch():
    launch_file = "~/catkin_ws/src/curi_whole_body_interface/launch/python_curi_dual_arm_ic_qbhand.launch"  # 替换为你的 launch 文件路径
    # 启动 roslaunch
    command = f"roslaunch {launch_file}"
    return subprocess.Popen(command, shell=True)


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


def minimum_jerk_trajectory(init_pos, target_pos, total_time=0.5, dt=0.001):
    xi = init_pos
    xf = target_pos
    d = total_time
    list_t = []
    list_x = []
    t = 0
    while t < d:
        x = xi + (xf-xi) * (10*(t/d)**3 - 15*(t/d)**4 + 6*(t/d)**5)
        list_t.append(t)
        list_x.append(x)
        t += dt
    return np.array(list_x)


def generate_perturb(data):
    my_traj_up = minimum_jerk_trajectory(0, 0.02)
    my_traj_mid = np.tile(0.02, 3000)
    my_traj_down = minimum_jerk_trajectory(0.02, 0)
    my_traj = np.append(my_traj_up, my_traj_mid)
    my_traj = np.append(my_traj, my_traj_down)

    perturb = np.zeros((4000, 3))
    if data == 1:
        perturb[:, 0] = my_traj
    if data == 2:
        perturb[:, 1] = my_traj
    if data == 3:
        perturb[:, 2] = my_traj
    if data == 4:
        perturb[:, 0] = math.sqrt(2) * my_traj / 2
        perturb[:, 1] = math.sqrt(2) * my_traj / 2
    if data == 5:
        perturb[:, 0] = math.sqrt(2) * my_traj / 2
        perturb[:, 2] = math.sqrt(2) * my_traj / 2
    if data == 6:
        perturb[:, 1] = math.sqrt(2) * my_traj / 2
        perturb[:, 2] = math.sqrt(2) * my_traj / 2

    # T_l, T_r =  tsf.transform_robot_base_to_arm_base(np.array([0, 0, 0]))
    # for i in range(len(perturb[: ,1])):
    #     perturb[i, :] = np.linalg.inv(T_r[:3, :3]) @ perturb[i, :]

    return perturb


def generate_trajectory(initial_pose):
    t = np.array([7, 14, 21, 28, 35, 42])
    k = np.array([1, 2, 3, 4, 5, 6])
    np.random.shuffle(k)
    trajectory = np.tile(initial_pose, 50000).reshape(-1, 3)
    for i in range(len(t)):
        n = t[i] * 1000
        trajectory[n:n+4000, :3] = trajectory[n:n+4000, :3] + generate_perturb(k[i])

    return k, trajectory


if __name__ == '__main__':
    rospy.init_node('HI_ImpRS_perturbation')
    signal.signal(signal.SIGINT, signal_handler)

    # 启动 roslaunch
    roslaunch_process = launch_roslaunch()
    time.sleep(1)  # 等待一段时间以确保 ROS 节点启动
    # 启动控制器

    curi = Python_CURI_Control(0, [])
    curi.start()
    time.sleep(1)

    # optitrack start streaming ...
    vrpn_roslaunch_process = vrpn_launch_roslaunch()


    ## Initialization of robot end effector poses
    robot_left_position_init = np.array([0.95, 0.15, 1.3])
    robot_right_position_init = np.array([0.9, -0.25, 0.7])

    robot_left_rotation_matrix_init = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # robot_left_rotation_matrix_init = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
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
    print("left", initial_robot_left_pose_matrix)
    print("right", initial_robot_right_pose_matrix)
    curi.set_tcp_moveL(initial_robot_left_pose_matrix, initial_robot_right_pose_matrix)

    while curi.get_curi_mode(0) != 2 and curi.get_curi_mode(1) != 2:
        print("waiting robot external control")
        time.sleep(1)

    # get the human upper limb joint position ...
    subscriber_shouR = rospy.wait_for_message('/vrpn_client_node/shouR/pose', PoseStamped)
    subscriber_elbowR = rospy.wait_for_message('/vrpn_client_node/elbowR/pose', PoseStamped)
    subscriber_wristR = rospy.wait_for_message('/vrpn_client_node/wristR/pose', PoseStamped)

    sub_shouR = transform_to_pose(subscriber_shouR)
    sub_elbowR = transform_to_pose(subscriber_elbowR)
    sub_wristR = transform_to_pose(subscriber_wristR)

    sub_joint_pos = np.array([sub_shouR, sub_elbowR, sub_wristR])
    np.savetxt('trajectory_revised/yuchen/joint_pos_6_h.txt', sub_joint_pos, delimiter=',', fmt='%.08f')

    # initial_pose = np.array([0.48968172, 0.38453146, 0.46961821, 0.76166407, -0.08329541, 0.56650357, 0.30332065])
    k, trajectory = generate_trajectory(robot_left_position_init)

    np.savetxt('trajectory_revised/yuchen/perturbation_6_h.txt', trajectory, delimiter=',', fmt='%.08f')
    np.savetxt('trajectory_revised/yuchen/k_6_h.txt', k, delimiter=',', fmt='%.08f')

    # plt.plot(trajectory[:, 0])
    # plt.plot(trajectory[:, 1])
    # plt.plot(trajectory[:, 2])
    # plt.show()


    init_trans_K = curi.get_K(0,0)
    init_trans_D = curi.get_D(0,0)

    init_rot_K = curi.get_K(0,3)
    init_rot_D = curi.get_D(0,3)

    print("init_trans_K ", init_trans_K)
    print("init_trans_D ", init_trans_D)
    print("init_rot_K ", init_rot_K)
    print("init_rot_D ", init_rot_D)

    desired_trans_K = 2500
    desired_trans_D = 100

    desired_rot_K = 80
    desired_rot_D = 10

    trans_K = init_trans_K
    trans_D = init_trans_D

    rot_K = init_rot_K
    rot_D = init_rot_D

    try:
        print("Starting trajectory execution...")
        for i in range(len(trajectory)):

            if trans_K < desired_trans_K:
                trans_K = trans_K + 0.4

            if trans_D < desired_trans_D:
                trans_D = trans_D + 0.04

            if rot_K < desired_rot_K:
                rot_K = rot_K + 0.04

            if rot_D < desired_rot_D:
                rot_D = rot_D + 0.004

            curi.set_trans_impedance(0,trans_D, trans_K)
            curi.set_rot_impedance(0, rot_D, rot_K)

            robot_left_position = trajectory[i, :3]
            robot_right_position = robot_right_position_init

            robot_left_pose_matrix = np.r_[
                np.c_[robot_left_rotation_matrix_init, robot_left_position.T], np.array([[0, 0, 0, 1]])]
            robot_right_pose_matrix = np.r_[
                np.c_[robot_right_rotation_matrix_init, robot_right_position.T], np.array([[0, 0, 0, 1]])]

            robot_left_pose_matrix = base2torso_matrix_init @ robot_left_pose_matrix
            robot_right_pose_matrix = base2torso_matrix_init @ robot_right_pose_matrix

            curi.set_tcp_servo(robot_left_pose_matrix, robot_right_pose_matrix)

            time.sleep(0.001)
        print("Execution finished.")

        # 保持程序运行，等待中断信号
        while not rospy.is_shutdown():
            interrupt = False
            time.sleep(1)

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # 清理资源
        if 'roslaunch_process' in globals():
            roslaunch_process.terminate()



