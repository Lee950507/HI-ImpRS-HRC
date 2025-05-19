# This is a Python script for uni-manual human-robot collaborative box carrying.
import numpy as np
import rospy
import math
import message_filters
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion, Pose
from std_msgs.msg import Int8, String, Bool, Float64MultiArray
from sensor_msgs.msg import JointState
import transformation as tsf
from time import time

import tf   
import tf2_ros
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as R
import scipy.linalg as linalg

torso_cmd = JointState()
left_pub = rospy.Publisher('/panda_left/cartesain_command_tele', PoseStamped, queue_size=1)
right_pub = rospy.Publisher('/panda_right/cartesain_command_tele', PoseStamped, queue_size=1)
torso_pub = rospy.Publisher("/curi_torso/joint/cmd_vel", JointState, queue_size=10)
impe_r_pub = rospy.Publisher('/panda_right/impedance_update', Float64MultiArray, queue_size=1)


def generate_robot_ee_cmd(human_state):
    robot_ee_cmd = human_state
    robot_ee_cmd[0] = human_state[0] - 0.85
    return robot_ee_cmd


def transform_to_pose(pose_stamped):
    pose = [pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z,
            pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, pose_stamped.pose.orientation.z,
            pose_stamped.pose.orientation.w]
    return np.array(pose)


def transform_to_joint(joint_state):
    joint = [joint_state.position[0], joint_state.position[1], joint_state.position[2]]
    time = joint_state.header.stamp.secs + 1e-9*joint_state.header.stamp.nsecs
    return np.array(joint), np.array(time)


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


def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix


def multi_callback(sub_torso):
    reference_traj = np.load('/home/curi/Chenzui/TASE/data/taichi/traj_taichi_uni_5400.npy')
    reference_stiff = np.load('/home/curi/Chenzui/TASE/data/taichi/stiff_taichi_uni_5400.npy')
    initial_pose = np.array([0.9, -0.1, 1.35, -0.45267768, 0.54952344, 0.53335966, 0.45676513])
    reference_traj = np.tile(reference_traj, (2, 1)).reshape(-1, 7)
    reference_stiff = np.tile(reference_stiff, (2, 1)).reshape(-1, 3)

    global i
    torso_joint, time[i] = transform_to_joint(sub_torso)
    # print("torso_joint",torso_joint, "time",time)
    index = int((time[i] - time[0]) * 1000)
    print(time[i])
    print(index)

    if index <= 10799:
        right_pose = reference_traj[index, :] + initial_pose
        right_stiff = reference_stiff[index, :]
        # TO arm base frame
        T_MobileBaseToLeftArmBase, T_MobileBaseToRightArmBase = tsf.transform_robot_base_to_arm_base(torso_joint)
        
        T = np.linalg.inv(T_MobileBaseToRightArmBase) 
        right_position_arm_base = T[:3, :3] @ right_pose[:3] + T[:3, 3]
        # right_orientation_arm_base = np.array([0.45206971, 0.58083515, 0.025424533, 0.67647401])

        right_rotation_temp = R.from_quat(initial_pose[3:])
        right_pose_orientation_matrix = right_rotation_temp.as_matrix()
        right_orientation_matrix_new = T[:3, :3] @ right_pose_orientation_matrix
        right_qua_temp = R.from_matrix(right_orientation_matrix_new)
        right_orientation_arm_base = right_qua_temp.as_quat()

        right_pose_arm_base =  np.append(right_position_arm_base, right_orientation_arm_base)
        print("right", right_pose_arm_base)
        right_pose_stamped = convert_to_pose_stamped(right_pose_arm_base, "panda_right_link0", rospy.Time.now())

        # impedance variation
        stiffness_matrix = [100, 100, 100, 33, 33, 33]
        stiffness_matrix[:3] = np.abs(np.dot(T[0:3, 0:3], right_stiff)) / 2
        # print("stiffness", stiffness_matrix)
        impe_r = Float64MultiArray(data=stiffness_matrix)

        joint1_vel_ = rospy.get_param("joint1_vel", 0.05)
        joint3_vel_ = rospy.get_param("joint3_vel", 0)

        if index < 2700:
            torso_joint1_vel = joint1_vel_
            torso_joint3_vel = 0
        elif index < 5400 and index >= 2700:
            torso_joint1_vel = -joint1_vel_
            torso_joint3_vel = 0
        elif index < 8100 and index >= 5400:
            torso_joint1_vel = joint1_vel_
            torso_joint3_vel = 0
        elif index < 10700 and index >= 8100:
            torso_joint1_vel = -joint1_vel_
            torso_joint3_vel = 0
        else:
            torso_joint1_vel = 0
            torso_joint3_vel = 0
        torso_cmd.velocity = [torso_joint1_vel, 0, 0, 0, 0, 0, 0]

        # !!!warning: the following code will activate the robot
        right_pub.publish(right_pose_stamped)
        # torso_pub.publish(torso_cmd)
        # impe_r_pub.publish(impe_r)
        
        i += 1
    else:
        print("Done!")

if __name__ == '__main__':
    try:
        rospy.init_node('ucb_hrc')

        subscriber_torso = message_filters.Subscriber('/curi_torso/joint_states', JointState)

        print("Here we go!")
        i = 0
        time = np.zeros(100000)
        sync = message_filters.ApproximateTimeSynchronizer([subscriber_torso], 10, 1)
        sync.registerCallback(multi_callback)

        rospy.spin()
        
    except rospy.ROSInterruptException as e:
        print(e)
