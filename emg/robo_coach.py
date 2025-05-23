#!/usr/bin/env python
import rospy
import message_filters
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion, Pose
from std_msgs.msg import Int8, String, Bool, Float64MultiArray
import numpy as np
# from transform import *
from queue import Queue
from threading import Thread
import time
import matplotlib.pyplot as plt
import pytrigno
import neurokit2 as nk
import scipy

# parameters about position
panda_flag = 'left'
d_z = 0.2  # difference between current z position and tractor point
x_fix = 0.8414423646400921
y_fix = 0.04860129822740884
z_traj_min = 1.02768197
# z_traj_min = 1.05768197
# z_traj_min = 0.97768197
z_traj_max = 1.52061786
z_start = 1.126269148
# z_start = 1.129269148
# z_start = 1.076269148
personal_para = 0

# parameters about impedance
max_emg = 300
k_p = 2.0
k_d = 0.1
k_p1 = 2.0
k_d1 = 0.1
pd_change_para = 30
pd_change_para_stf = 0.01
ref_activation = 0.3
target_activation = ref_activation  # changeable parameter

init_stiffness_flag = True
read_emg_flag = True
emg_save_flag = False
pose_save_flag = False

save_time = 60
plot_emg_num = 500
proc_emg_num = 6

emg_sample_rate = 2000
emg_samples_per_read = 200
emg_channel_num = 1
emg_filtering_rate = 4

plot_time_step = 0.1
z_stiffness_min = 150
z_stiffness_max = 350
init_impe = [200, 200, 250, 36, 36, 36]
init_stiffness = 200
emg_ip = '192.168.10.30'
pose_pub = rospy.Publisher('/panda_left/cartesain_command_tele', PoseStamped, queue_size=1)
impe_pub = rospy.Publisher('/panda_left/impedance_update', Float64MultiArray, queue_size=1)
callback_time_step = 0.02

demo_z = 0.0
emg_error_past = 0
stf_error_past = 0
stiffness_value = init_impe[2]
emg_data_smooth = np.array([])
all_stiffness_z = np.array([])
all_emg_data = np.array([])
all_pose = np.array([])
emg_data = np.array([])
all_pose_time = np.array([])
all_emg_data_time = np.array([])

T_l = np.array([[0.498566, -0.516245, 0.696364, 0.19741], [-0.353553, 0.612372, 0.707107, 0.07009],
                [-0.791475, -0.598741, 0.122788, 1.33849], [0, 0, 0, 1]])

def transform_to_pose(pose):
    """
    from position and orientation to pose
    """
    # from position and orientation to pose
    pose_update = np.array([pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    return pose_update


def convert_from_pose_stamped(pose, frame_id):
    """
    from pose to position and orientation
    """
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = pose[0]
    pose_stamped.pose.position.y = pose[1]
    pose_stamped.pose.position.z = pose[2]
    pose_stamped.pose.orientation.x = pose[3]
    pose_stamped.pose.orientation.y = pose[4]
    pose_stamped.pose.orientation.z = pose[5]
    pose_stamped.pose.orientation.w = pose[6]
    return pose_stamped


def transform_to_world_frame(pose):
    """
    Transform position and orientation in arm frame to world frame.
    """
    if panda_flag == 'left':
        # Matrix could transform position and orientation in arm frame to world frame
        T_l = np.array([[0.498566, -0.516245, 0.696364, 0.19741], [-0.353553, 0.612372, 0.707107, 0.07009],
                        [-0.791475, -0.598741, 0.122788, 1.33849], [0, 0, 0, 1]])
        
        # print(pose.shape)
        left_pos = pose[0:3]
        # left_ori = left_pose[3:7]
        data_pos_transform_l = np.dot(T_l[0:3, 0:3], left_pos) + (T_l[0:3, 3:4]).T
        data_pos_transform_l = np.reshape(data_pos_transform_l, -1)
        # data_ori_transform_l = np.array([0.8, 0.3, 0.5, -0.1])
        # left_pose_new = np.append(data_pos_transform_l, data_ori_transform_l)
        pose_new = data_pos_transform_l
    else:
        # Matrix could transform position and orientation in arm frame to world frame
        T_r = np.array([[0.498566, 0.516245, 0.696364, 0.19741], [0.353553, 0.612372, -0.707107, -0.07009],
                        [-0.791475, 0.598741, 0.122788, 1.33849], [0, 0, 0, 1]])
        right_pos = pose[0:3]
        # right_ori = right_pose[3:7]
        data_pos_transform_r = np.dot(T_r[0:3, 0:3], right_pos) + (T_r[0:3, 3:4]).T
        data_pos_transform_r = np.reshape(data_pos_transform_r, -1)
        # data_ori_transform_r = np.array([0.8, -0.3, 0.5, 0.16])
        # right_pose_new = np.append(data_pos_transform_r, data_ori_transform_r)
        pose_new = data_pos_transform_r
    return pose_new


def transform_to_arm_base(pose):
    """
    Transform position and orientation in world frame to arm frame.
    """
    if panda_flag == 'left':
        # Matrix could transform position and orientation in arm frame to world frame
        T_l = np.array([[0.498566, -0.516245, 0.696364, 0.19741], [-0.353553, 0.612372, 0.707107, 0.07009],
                        [-0.791475, -0.598741, 0.122788, 1.33849], [0, 0, 0, 1]])
        T_l_inv = np.linalg.inv(T_l)

        left_pos = pose[0:3]
        # left_ori = left_pose[3:7]
        data_pos_transform_l = np.dot(T_l_inv[0:3, 0:3], left_pos) + (T_l_inv[0:3, 3:4]).T
        data_pos_transform_l = np.reshape(data_pos_transform_l, -1)
        data_ori_transform_l = np.array([0.73745, -0.08215,  0.53335, -0.40585])
        pose_new = np.append(data_pos_transform_l, data_ori_transform_l)
    else:
        # Matrix could transform position and orientation in arm frame to world frame
        T_r = np.array([[0.498566, 0.516245, 0.696364, 0.19741], [0.353553, 0.612372, -0.707107, -0.07009],
                        [-0.791475, 0.598741, 0.122788, 1.33849], [0, 0, 0, 1]])
        T_r_inv = np.linalg.inv(T_r)

        right_pos = pose[0:3]
        # right_ori = right_pose[3:7]
        data_pos_transform_r = np.dot(T_r_inv[0:3, 0:3], right_pos) + (T_r_inv[0:3, 3:4]).T
        data_pos_transform_r = np.reshape(data_pos_transform_r, -1)
        data_ori_transform_r = np.array([0.8, -0.3, 0.5, 0.16])
        pose_new = np.append(data_pos_transform_r, data_ori_transform_r)
    return pose_new


def generate_stiffness_pd_based_emg(Kp, Kd, feedback_pressure):
    """
    Change stiffness of z-axis with PD controller.
    """
    global emg_error_past
    global stiffness_value
    global target_activation
    # current value of tactile sensor
    feedback_pressure = np.mean(feedback_pressure)
    error = target_activation - feedback_pressure
    if emg_error_past == 0:
        error_dot = 0
    else:
        error_dot = emg_error_past - error
    emg_error_past = error
    output = Kp * error + Kd * error_dot
    stiffness_value = stiffness_value + output * pd_change_para
    if stiffness_value < z_stiffness_min:
        stiffness_value = z_stiffness_min
    elif stiffness_value > z_stiffness_max:
        stiffness_value = z_stiffness_max
    return stiffness_value


def generate_stiffness_pd_based_stf(Kp, Kd, target_stiffness):
    """
    Change stiffness of z-axis with PD controller.
    """
    global stf_error_past
    global stiffness_value
    error = target_stiffness - stiffness_value
    if stf_error_past == 0:
        error_dot = 0
    else:
        error_dot = stf_error_past - error
    stf_error_past = error
    output = Kp * error + Kd * error_dot
    stiffness_value = stiffness_value + output * pd_change_para_stf
    if stiffness_value < z_stiffness_min:
        stiffness_value = z_stiffness_min
    elif stiffness_value > z_stiffness_max:
        stiffness_value = z_stiffness_max
    return stiffness_value


def read_emg(out_q):
    """
    Thread of communicating with EMG device.
    """
    dev = pytrigno.TrignoEMG(channel_range=(0, 0), samples_per_read=emg_samples_per_read, host=emg_ip)
    dev.set_channel_range((0, emg_channel_num - 1))
    dev.start()
    while read_emg_flag is True:
        start_time = time.time()
        data = dev.read() * 1e6
        assert data.shape == (emg_channel_num, emg_samples_per_read)
        # print(emg_channel_num, '-channel achieved')
        data = np.transpose(np.array(data), (1, 0))
        end_time = time.time()
        # out_q.put(data)
        # print(end_time - start_time)        
        out_q.put(data)
        # print('read emg')
    dev.stop()


def process_one_channel(data, sampling_rate):
    """
    process one channel of EMG signal
    """
    data_mean = np.mean(data[1:200])
    raw = data - data_mean * np.ones_like(data)
    t = np.arange(0, raw.size / sampling_rate, 1 / sampling_rate)
    emgfwr = abs(raw)

    # Linear Envelope
    numpasses = 3
    lowpass_rate = 6

    Wn = lowpass_rate / (sampling_rate / 2)
    [b, a] = scipy.signal.butter(numpasses, Wn, 'low')
    EMGLE = scipy.signal.filtfilt(b, a, emgfwr, padtype='odd', padlen=3*(max(len(b), len(a))-1))  # 低通滤波

    ref = max_emg
    normalized_EMG = EMGLE / ref
    return normalized_EMG


def emg_rectification(q, sample_rate):
    """
    From EMG signals to Maximum Voluntary Contraction (MVC).
    """
    emg_data = q.get()  # Get some emg data
    # data_filter, data_clean, data_mvc, data_activity, data_abs, data_mvcscale = \
    #     process_all_channels(emg_data, emg_channel_num, sample_rate, emg_filtering_rate)
    emg_data = emg_data[:, 0]
    data_mvc = process_one_channel(emg_data, sample_rate)
    # print(data_mvc)
    q.task_done()  # Indicate completion
    emg_mean = data_mvc.mean()
    if emg_mean < 0:
        emg_mean = 0
    elif emg_mean > 1:
        emg_mean = 1
    return emg_mean


def find_nearest_idx(arr, value):
    """
    Find two nearest data with value in array. Return index of these data.
    array(idx1) <= array(idx2)
    """
    arr = np.asarray(arr)
    array = np.asarray(arr) - value
    if min(array) <= 0 <= max(array):
        max_array = np.ones_like(array) * max(array)
        min_array = np.ones_like(array) * min(array)
        up_zero = np.where(array > 0, array, max_array)
        down_zero = np.where(array < 0, array, min_array)
        idx2 = up_zero.argmin()
        idx1 = down_zero.argmax()
    elif min(array) <= max(array) <= 0:
        idx2 = array.argmax()
        array[idx2] = float("-inf")
        idx1 = array.argmax()
    elif 0 <= min(array) <= max(array):
        idx1 = array.argmin()
        array[idx1] = float("inf")
        idx2 = array.argmin()
    else:
        print('something is wrong')
        idx1 = 0
        idx2 = 0
    assert arr[idx1] <= arr[idx2]
    # if arr[idx1] > arr[idx2]:
    #     print(value, idx1, idx2, arr[idx1], arr[idx2])
    return idx1, idx2


def find_xy(z):
    """
    Find position_x and position_y according current y.
    """
    global x_d
    global y_d
    global z_d
    z_near_idx1, z_near_idx2 = find_nearest_idx(z_d, z)
    if z_near_idx1==0 and z_near_idx2==1 and z < z_d[0]:
        x = x_d[0]
        y = y_d[0]
    else:
        z_near1, z_near2 = z_d[z_near_idx1], z_d[z_near_idx2]
        x_near1, x_near2 = x_d[z_near_idx1], x_d[z_near_idx2]
        y_near1, y_near2 = y_d[z_near_idx1], y_d[z_near_idx2]
        x = (z - z_near1) / (z_near2 - z_near1) * (x_near2 - x_near1) + x_near1
        y = (z - z_near1) / (z_near2 - z_near1) * (y_near2 - y_near1) + y_near1
    return x, y


def find_pose(pose):
    global all_pose
    global all_pose_time
    global init_stiffness_flag
    # pose = transform_to_pose(sub_pose)  # from position and orientation to pose
    world_pos = transform_to_world_frame(pose)  # from arm base frame to world frame
    pos = world_pos.copy()
    # delta_z = pos[2] - z0  # change of position_Z
    # demo_z = demo_z + delta_z
    demo_x, demo_y = find_xy(pos[2] + personal_para)
    pos[0] = demo_x
    pos[1] = demo_y
    # pos[2] = pos[2] - d_z
    # pos[0] = x_fix
    # pos[1] = y_fix


    if z_traj_min < pos[2] < z_traj_max:
        pos[2] = pos[2] - d_z
    elif pos[2] < z_traj_min:
        pos[2] = z_traj_min - d_z
    elif z_traj_max < pos[2]:
        pos[2] = z_traj_max - d_z

    if pos[2] < z_start - d_z:  # about stiffness
        init_stiffness_flag = True
    else:
        init_stiffness_flag = False

    pos_record_time = time.time()
    pos_record = pos[2]
    all_pose_time = np.append(all_pose_time, pos_record_time)
    all_pose = np.append(all_pose, pos_record)
    demo_pose = transform_to_arm_base(pos)  # from world frame to left arm base frame
    pose_stamped = convert_from_pose_stamped(demo_pose, 'panda_left_link0')
    return pose_stamped, pos, demo_pose


def process_emg(q):
    global emg_save_flag
    global init_stiffness_flag
    global tic_start
    global emg_data_smooth
    global all_emg_data
    global all_emg_data_time
    global all_stiffness_z
    global emg_data
    global tic
    while time.time() - tic_start < save_time:
        if time.time() - tic > callback_time_step:
            emg = emg_rectification(q, emg_sample_rate)
            emg_data_smooth = np.append(emg_data_smooth, emg)
            if len(emg_data_smooth) > 3:
                emg_data_smooth = emg_data_smooth[len(emg_data_smooth) - 3:]
            emg_data_mean = emg_data_smooth.mean()
            emg_data_record_time = time.time()
            emg_data_record = emg_data_mean
            all_emg_data = np.append(all_emg_data, emg_data_record)
            all_emg_data_time = np.append(all_emg_data_time, emg_data_record_time)
            # print(emg_data_mean)

            # if len(all_emg_data) > plot_emg_num:
            #     all_emg_data = all_emg_data[len(all_emg_data) - plot_emg_num:]

            # PD controller
            emg_data = np.append(emg_data, emg_data_mean)
            if len(emg_data) > proc_emg_num:
                emg_data = emg_data[len(emg_data) - proc_emg_num:]
            if init_stiffness_flag is True:
                print('init stiffness')
                stiffness_value = generate_stiffness_pd_based_stf(k_p, k_d, init_stiffness)
            else:
                stiffness_value = generate_stiffness_pd_based_emg(k_p1, k_d1, emg_data)

            # stiffness_value = generate_stiffness_pd_based_emg(k_p, k_d, emg_data)
            # print(stiffness_value)

            all_stiffness_z = np.append(all_stiffness_z, stiffness_value)
            stiffness_matrix = init_impe.copy()
            stiffness_matrix[2] = stiffness_value
            print(stiffness_matrix)
            stiffness_matrix[:3] = np.abs(np.dot(np.linalg.inv(T_l)[0:3, 0:3], stiffness_matrix[:3]))
            # print(stiffness_matrix, np.sum(np.square(stiffness_matrix[:3])))
            impe = Float64MultiArray(data=stiffness_matrix)

            # stiffness_matrix = init_impe.copy()
            # stiffness_matrix[:3] = np.abs(np.dot(np.linalg.inv(T_l)[0:3, 0:3], stiffness_matrix[:3]))
            # impe = Float64MultiArray(data=stiffness_matrix)
            # all_stiffness_z = np.append(all_stiffness_z, init_impe[2])
            # print(init_impe[2])

            impe_pub.publish(impe)
    else:
        np.save('muscle_activation.npy', all_emg_data)
        np.save('muscle_activation_time.npy', all_emg_data_time)
        np.save('stiffness_z.npy', all_stiffness_z)
        print('EMG data is saved.')
        emg_save_flag = True
        read_emg_flag = False


def multi_callback(subscriber_pose):
    global init_z_flag
    global tic_start
    global pose_save_flag
    global emg_data_smooth
    global emg_data
    global demo_z
    global all_pose
    global all_pose_time
    global tic
    # global plt
    # global ax
    if time.time() - tic > callback_time_step:
        tic = time.time()
        subscriber_pose = transform_to_pose(subscriber_pose)
        # print(subscriber_pose.shape)

        pose, world_pose, demo_pose = find_pose(subscriber_pose)
        pose_pub.publish(pose)
    if time.time() - tic_start > save_time and pose_save_flag is False:
        np.save('world_pose.npy', all_pose)
        np.save('world_pose_time.npy', all_pose_time)
        print('Pose data is saved.')
        pose_save_flag = True



if __name__ == "__main__":
    try:
        z_start = (z_traj_max - z_traj_min) * 0.2 + z_traj_min
        traj_l = np.load('traj_l.npy')
        x_d = -pow((traj_l[:98, 0]-traj_l[0, 0]) ** 2 + (traj_l[:98, 1]-traj_l[0, 1]) ** 2, 0.5) + x_fix
        y_d = y_fix * np.ones_like(-traj_l[:98, 1])
        z_d = traj_l[:98, 2] - traj_l[0, 2] + z_traj_min + 0.05
        rospy.init_node('resistance_training')

        # plt.ion()  # turn on interactive mode
        # plt.rcParams["font.family"] = "Times New Roman"
        # plt.style.use('classic')
        # fig, ax = plt.subplots()  # create a figure and an axis
        tic = time.time()
        tic_start = time.time()

        q = Queue()  # Create a shared queue
        thread_emg = Thread(target=read_emg, args=(q,))
        thread_emg_process = Thread(target=process_emg, args=(q,))
        thread_emg.start()
        thread_emg_process.start()
        q.join()  # Wait for all produced items to be consumed

        # subscriber_pose = message_filters.Subscriber('/panda_left/cartesian_states', Pose)
        # sync = message_filters.ApproximateTimeSynchronizer([subscriber_pose, q], 10, 1)
        # sync = message_filters.ApproximateTimeSynchronizer([subscriber_pose], 10, 1)
        # sync.registerCallback(multi_callback)

        # rospy.Subscriber('/panda_left/cartesian_states', Pose, multi_callback)
        rospy.Subscriber('/panda_left/cartesian_states', Pose, lambda data: multi_callback(data))
        rospy.spin()
        read_emg_flag = False

    except rospy.ROSInterruptException as e:
        print(e)
        read_emg_flag = False
