import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import os


def preprocess_pose(pose):
    pos_shou_zhuo = pose[:, 5:8]
    pos_elbow_zhuo = pose[:, 12:15]
    pos_wrist_zhuo = pose[:, 19:22]
    l_zhuo = pos_wrist_zhuo - pos_shou_zhuo
    r_zhuo = pos_elbow_zhuo - pos_shou_zhuo
    l_zhuo[:, [1, 2]] = l_zhuo[:, [2, 1]]
    l_zhuo[:, [0, 1]] = l_zhuo[:, [1, 0]]
    r_zhuo[:, [1, 2]] = r_zhuo[:, [2, 1]]
    r_zhuo[:, [0, 1]] = r_zhuo[:, [1, 0]]

    pos_shou_cz = pose[:, 26:29]
    pos_elbow_cz = pose[:, 33:36]
    pos_wrist_cz = pose[:, 40:43]
    l_cz = pos_wrist_cz - pos_shou_cz
    r_cz = pos_elbow_cz - pos_shou_cz
    l_cz[:, [1, 2]] = l_cz[:, [2, 1]]
    l_cz[:, [0, 1]] = l_cz[:, [1, 0]]
    l_cz[:, 0] = - l_cz[:, 0]
    r_cz[:, [1, 2]] = r_cz[:, [2, 1]]
    r_cz[:, [0, 1]] = r_cz[:, [1, 0]]
    r_cz[:, 0] = - r_cz[:, 0]

    return l_zhuo, r_zhuo, l_cz, r_cz


def calculate_endpoint_stiffness(l, r, a1, a2, b1, b2, A):
    V_1 = l / np.linalg.norm(l)
    V_2 = np.cross(np.cross(r, l), l) / np.linalg.norm(np.cross(np.cross(r, l), l))
    V_3 = np.cross(r, l) / np.linalg.norm(np.cross(r, l))

    V = np.around(np.array([V_1, V_2, V_3]).T, 5)

    d1 = np.linalg.norm(l)
    d2 = np.linalg.norm(np.dot(r, np.cross(np.cross(r, l), l) / np.linalg.norm(np.cross(np.cross(r, l), l))))
    D_s = np.around(np.diag([1, a1 / d1, a2 * d2]) / pow(a1 * a2 * d2 / d1, 1 / 3), 5)

    Acc = b1 * A + b2

    Ke = np.around(V @ (Acc * D_s) @ V.T, 3)
    return Ke


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


if __name__ == '__main__':
    # 创建输出目录
    output_dir = "stiffness_results_bi"
    os.makedirs(output_dir, exist_ok=True)

    # 加载姿态、力和肌电数据
    pose = pd.read_csv('/home/ubuntu/HI-ImpRS-HRC/data/taichi/unimanual_data/2.csv')
    emg = np.load('/home/ubuntu/HI-ImpRS-HRC/data/taichi/unimanual_data/2.npy')
    pose = np.array(pose)[8690:9020, :] / 100
    emg = abs(emg[146167:151667, 1:5])

    # 处理肌电信号
    emg_process_bic_zhuo_right = moving_average(emg[:, 0], 500) / 300
    emg_process_tri_zhuo_right = moving_average(emg[:, 1], 500) / 300
    A_zhuo_right = (emg_process_tri_zhuo_right + emg_process_bic_zhuo_right) / 2

    emg_process_bic_zhuo_left = moving_average(emg[:, 2], 500) / 300
    emg_process_tri_zhuo_left = moving_average(emg[:, 3], 500) / 300
    A_zhuo_left = (emg_process_bic_zhuo_left + emg_process_tri_zhuo_left) / 2

    # 对肌电数据进行降采样
    A_zhuo_right = A_zhuo_right[::16]
    A_zhuo_right = A_zhuo_right[7:337]
    A_zhuo_left = A_zhuo_left[::16]
    A_zhuo_left = A_zhuo_left[7:337]

    # 处理姿态数据
    l_zhuo_right, r_zhuo_right, l_zhuo_left, r_zhuo_left = preprocess_pose(pose)

    # 确保数组长度匹配
    min_len_right = min(len(l_zhuo_right), len(r_zhuo_right), len(A_zhuo_right))
    min_len_left = min(len(l_zhuo_left), len(r_zhuo_left), len(A_zhuo_left))

    # 截取数组到相同长度
    l_zhuo_right = l_zhuo_right[:min_len_right]
    r_zhuo_right = r_zhuo_right[:min_len_right]
    A_zhuo_right = A_zhuo_right[:min_len_right]

    l_zhuo_left = l_zhuo_left[:min_len_left]
    r_zhuo_left = r_zhuo_left[:min_len_left]
    A_zhuo_left = A_zhuo_left[:min_len_left]

    print(f"Data lengths after alignment: Right={min_len_right}, Left={min_len_left}")

    # 定义5组参数
    A = [
        [0.272, 1.314, 3847.141, 151.684],
        [0.107, 2.200, 2678.765, 149.597],
        [0.399, 2.926, 1819.695, 128.581],
        [0.341, 4.073, 2699.123, 112.562],
        [0.167, 4.528, 1260.290, 94.951]
    ]

    # 保存参数集
    param_sets = np.array(A)
    # np.save(os.path.join(output_dir, "parameter_sets.npy"), param_sets)
    # print(f"Saved parameter sets to {os.path.join(output_dir, 'parameter_sets.npy')}")

    # 存储所有参数集的结果
    all_Ke_right = []
    all_Ke_diag_right = []
    all_Ke_left = []
    all_Ke_diag_left = []

    # 对每个参数集计算刚度
    for param_idx, param_set in enumerate(A):
        print(f"Processing parameter set {param_idx + 1}: {param_set}")

        # 提取参数
        a1, a2, b1, b2 = param_set

        # 计算左手刚度
        Ke_left = []
        Ke_diag_left = []
        for i in range(len(l_zhuo_left)):
            Ke = calculate_endpoint_stiffness(l_zhuo_left[i, :], r_zhuo_left[i, :], a1, a2, b1, b2, A_zhuo_left[i])
            Ke_left.append(Ke)
            Ke_diag_left.append(np.diagonal(Ke))
        Ke_left = np.array(Ke_left)
        Ke_diag_left = np.array(Ke_diag_left)

        # 保存左手刚度数据
        left_file = os.path.join(output_dir, f"Ke_left_set{param_idx + 1}.npy")
        np.save(left_file, Ke_diag_left)
        print(f"Saved Left hand stiffness for parameter set {param_idx + 1} to {left_file}")

        # 计算右手刚度
        Ke_right = []
        Ke_diag_right = []
        for i in range(len(l_zhuo_right)):
            Ke2 = calculate_endpoint_stiffness(l_zhuo_right[i, :], r_zhuo_right[i, :], a1, a2, b1, b2, A_zhuo_right[i])
            Ke_right.append(Ke2)
            Ke_diag_right.append(np.diagonal(Ke2))
        Ke_right = np.array(Ke_right)
        Ke_diag_right = np.array(Ke_diag_right)

        # 保存右手刚度数据
        right_file = os.path.join(output_dir, f"Ke_right_set{param_idx + 1}.npy")
        np.save(right_file, Ke_diag_right)
        print(f"Saved Right hand stiffness for parameter set {param_idx + 1} to {right_file}")

        # 保存完整刚度矩阵
        # full_left_file = os.path.join(output_dir, f"Ke_full_left_set{param_idx + 1}.npy")
        # np.save(full_left_file, Ke_left)
        #
        # full_right_file = os.path.join(output_dir, f"Ke_full_right_set{param_idx + 1}.npy")
        # np.save(full_right_file, Ke_right)

        # 存储用于可视化
        all_Ke_left.append(Ke_left)
        all_Ke_diag_left.append(Ke_diag_left)
        all_Ke_right.append(Ke_right)
        all_Ke_diag_right.append(Ke_diag_right)

    # 如果数组不为空，创建可视化比较图
    if all_Ke_diag_right and all_Ke_diag_left:
        # 1. 比较右手的刚度
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

        for axis_idx in range(3):
            ax = axes[axis_idx]

            for param_idx, Ke_diag in enumerate(all_Ke_diag_right):
                ax.plot(Ke_diag[:, axis_idx], color=colors[param_idx],
                        label=f'S{param_idx + 1}: a1={A[param_idx][0]}, a2={A[param_idx][1]}, b1={A[param_idx][2]:.1f}, b2={A[param_idx][3]:.1f}',
                        linewidth=1.5)

            ax.set_title(f'Right Hand {axis_labels[axis_idx]} Stiffness Comparison Across Different Subjects')
            ax.set_ylabel('Stiffness (N/m)')
            ax.grid(True, alpha=0.3)

            if axis_idx == 0:
                ax.legend(loc='upper right', fontsize=8)

        axes[2].set_xlabel('Time step')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'right_hand_stiffness_comparison.png'), dpi=300)

        # 2. 比较左手的刚度
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        for axis_idx in range(3):
            ax = axes[axis_idx]

            for param_idx, Ke_diag in enumerate(all_Ke_diag_left):
                ax.plot(Ke_diag[:, axis_idx], color=colors[param_idx],
                        label=f'S{param_idx + 1}: a1={A[param_idx][0]}, a2={A[param_idx][1]}, b1={A[param_idx][2]:.1f}, b2={A[param_idx][3]:.1f}',
                        linewidth=1.5)

            ax.set_title(f'Left Hand {axis_labels[axis_idx]} Stiffness Comparison Across Different Subjects')
            ax.set_ylabel('Stiffness (N/m)')
            ax.grid(True, alpha=0.3)

            if axis_idx == 0:
                ax.legend(loc='upper right', fontsize=8)

        axes[2].set_xlabel('Time step')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'left_hand_stiffness_comparison.png'), dpi=300)

        # 3. 比较左右手同一参数集下的刚度差异 (为每个参数集创建一个单独的图)
        for param_idx in range(len(A)):
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

            for axis_idx in range(3):
                ax = axes[axis_idx]

                ax.plot(all_Ke_diag_right[param_idx][:, axis_idx], color='blue',
                        label='Right Hand', linewidth=1.5)
                ax.plot(all_Ke_diag_left[param_idx][:, axis_idx], color='red',
                        label='Left Hand', linewidth=1.5)

                ax.set_title(
                    f'Parameter Set {param_idx + 1}: {axis_labels[axis_idx]} Stiffness Comparison Between Hands')
                ax.set_ylabel('Stiffness (N/m)')
                ax.grid(True, alpha=0.3)

                if axis_idx == 0:
                    ax.legend(loc='upper right', fontsize=10)

            axes[2].set_xlabel('Time step')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'hands_comparison_set{param_idx + 1}.png'), dpi=300)

    print(f"All data has been saved to the '{output_dir}' directory")
    print(f"Generated {len(A)} stiffness datasets for each hand (Left and Right)")