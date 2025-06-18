import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt


def preprocess_force(force):
    return


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
    output_dir = "stiffness_results_uni"
    os.makedirs(output_dir, exist_ok=True)

    # 加载姿态和肌电数据
    pose = pd.read_csv('/home/ubuntu/HI-ImpRS-HRC/data/taichi/unimanual_data/2.csv')
    emg = np.load('/home/ubuntu/HI-ImpRS-HRC/data/taichi/unimanual_data/2.npy')
    pose = np.array(pose)[6680:7250, :] / 100
    emg = abs(emg[123000:132500, 1:5])

    # 处理肌电信号
    emg_process_bic_zhuo = moving_average(emg[:, 0], 500) / 300
    emg_process_tri_zhuo = moving_average(emg[:, 1], 500) / 300
    A_zhuo = (emg_process_tri_zhuo + emg_process_bic_zhuo) / 2

    emg_process_bic_cz = moving_average(emg[:, 2], 500) / 300
    emg_process_tri_cz = moving_average(emg[:, 3], 500) / 300
    A_cz = (emg_process_tri_cz + emg_process_bic_cz) / 2

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
    np.save(os.path.join(output_dir, "parameter_sets.npy"), param_sets)
    print(f"Saved parameter sets to {os.path.join(output_dir, 'parameter_sets.npy')}")

    # 处理肌电数据
    A_zhuo = A_zhuo[::16]
    A_zhuo = A_zhuo[12:582]
    A_cz = A_cz[::16]
    A_cz = A_cz[12:582]

    # 处理姿态数据
    l_zhuo, r_zhuo, l_cz, r_cz = preprocess_pose(pose)

    # 存储所有参数集的结果
    all_Ke_zhuo = []
    all_Ke_diag_zhuo = []
    all_Ke_cz = []
    all_Ke_diag_cz = []

    # 对每个参数集计算刚度
    for param_idx, param_set in enumerate(A):
        print(f"Processing parameter set {param_idx + 1}: {param_set}")

        # 提取参数
        a1, a2, b1, b2 = param_set

        # 计算陈子刚度
        Ke_cz = []
        Ke_diag_cz = []
        for i in range(len(l_cz[:, 0])):
            Ke = calculate_endpoint_stiffness(l_cz[i, :], r_cz[i, :], a1, a2, b1, b2, A_cz[i])
            Ke_cz.append(Ke)
            Ke_diag_cz.append(np.diagonal(Ke))
        Ke_cz = np.array(Ke_cz)
        Ke_diag_cz = np.array(Ke_diag_cz)

        # 保存陈子刚度数据
        # cz_file = os.path.join(output_dir, f"Ke_cz_set{param_idx + 1}.npy")
        # np.save(cz_file, Ke_diag_cz)
        # print(f"Saved CZ stiffness for parameter set {param_idx + 1} to {cz_file}")

        # 计算卓的刚度
        Ke_zhuo = []
        Ke_diag_zhuo = []
        for i in range(len(l_zhuo[:, 0])):
            Ke2 = calculate_endpoint_stiffness(l_zhuo[i, :], r_zhuo[i, :], a1, a2, b1, b2, A_zhuo[i])
            Ke_zhuo.append(Ke2)
            Ke_diag_zhuo.append(np.diagonal(Ke2))
        Ke_zhuo = np.array(Ke_zhuo)
        Ke_diag_zhuo = np.array(Ke_diag_zhuo)

        # 保存卓的刚度数据
        zhuo_file = os.path.join(output_dir, f"Ke_zhuo_set{param_idx + 1}.npy")
        np.save(zhuo_file, Ke_diag_zhuo)
        print(f"Saved ZHUO stiffness for parameter set {param_idx + 1} to {zhuo_file}")

        # 保存完整刚度矩阵
        # full_cz_file = os.path.join(output_dir, f"Ke_full_cz_set{param_idx + 1}.npy")
        # np.save(full_cz_file, Ke_cz)
        #
        # full_zhuo_file = os.path.join(output_dir, f"Ke_full_zhuo_set{param_idx + 1}.npy")
        # np.save(full_zhuo_file, Ke_zhuo)

        # 存储用于可视化
        all_Ke_cz.append(Ke_cz)
        all_Ke_diag_cz.append(Ke_diag_cz)
        all_Ke_zhuo.append(Ke_zhuo)
        all_Ke_diag_zhuo.append(Ke_diag_zhuo)

    # 创建可视化比较图
    # 1. 比较ZHUO的刚度
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

    for axis_idx in range(3):
        ax = axes[axis_idx]

        for param_idx, Ke_diag in enumerate(all_Ke_diag_zhuo):
            ax.plot(Ke_diag[:, axis_idx], color=colors[param_idx],
                    label=f'S{param_idx + 1}: a1={A[param_idx][0]}, a2={A[param_idx][1]}, b1={A[param_idx][2]:.1f}, b2={A[param_idx][3]:.1f}',
                    linewidth=1.5)

        ax.set_title(f'{axis_labels[axis_idx]} Stiffness Comparison Across Different Subjects')
        ax.set_ylabel('Stiffness (N/m)')
        ax.grid(True, alpha=0.3)

        if axis_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[2].set_xlabel('Time step')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stiffness_comparison.png'), dpi=600)

    print(f"All data has been saved to the '{output_dir}' directory")