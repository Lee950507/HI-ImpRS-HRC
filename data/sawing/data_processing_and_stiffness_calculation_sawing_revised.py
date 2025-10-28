import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def repair_discontinuous_regions(data, bad_regions, method='cubic'):
    """
    修复数据中的不连续区域

    参数:
        data: 原始数据数组 (N, 3) - N个时间步，3个维度(X,Y,Z)
        bad_regions: 不良数据区间列表，如 [[80, 130], [480, 530]]
        method: 插值方法 ('linear', 'cubic', 'quadratic')

    返回:
        repaired_data: 修复后的数据数组 (N, 3)
    """
    N = data.shape[0]
    repaired_data = data.copy()

    # 创建掩码：标记哪些数据点是好的
    good_mask = np.ones(N, dtype=bool)

    # 标记坏的区域
    for start, end in bad_regions:
        good_mask[start:end + 1] = False

    # 对每个维度(X, Y, Z)分别进行插值
    for dim in range(3):
        # 获取好的数据点的索引和值
        good_indices = np.where(good_mask)[0]
        good_values = data[good_mask, dim]

        if len(good_indices) < 2:
            print(f"警告：维度{dim}没有足够的有效数据点进行插值")
            continue

        # 创建插值函数
        interp_func = interp1d(
            good_indices,
            good_values,
            kind=method,
            bounds_error=False,
            fill_value='extrapolate'
        )

        # 对坏的区域进行插值
        for start, end in bad_regions:
            bad_indices = np.arange(start, end + 1)
            repaired_data[bad_indices, dim] = interp_func(bad_indices)

        print(f"维度 {['X', 'Y', 'Z'][dim]} 修复完成")

    return repaired_data


def interpolate_to_target_length(data, target_length=5400, method='cubic'):
    """
    将数据插值到目标长度

    参数:
        data: 原始数据数组 (N, 3) - N个时间步，3个维度(X,Y,Z)
        target_length: 目标长度
        method: 插值方法 ('linear', 'cubic', 'quadratic')

    返回:
        interpolated_data: 插值后的数据数组 (target_length, 3)
    """
    original_length = data.shape[0]
    interpolated_data = np.zeros((target_length, 3))

    # 原始索引和目标索引
    original_indices = np.linspace(0, original_length - 1, original_length)
    target_indices = np.linspace(0, original_length - 1, target_length)

    # 对每个维度分别插值
    for dim in range(3):
        interp_func = interp1d(
            original_indices,
            data[:, dim],
            kind=method,
            bounds_error=False,
            fill_value='extrapolate'
        )
        interpolated_data[:, dim] = interp_func(target_indices)

    print(f"数据已从 {original_length} 个点插值到 {target_length} 个点")
    return interpolated_data


def smooth_data(data, sigma=5):
    """
    使用高斯滤波对数据进行平滑处理

    参数:
        data: 数据数组 (N, 3)
        sigma: 高斯滤波的标准差，值越大平滑程度越高

    返回:
        smoothed_data: 平滑后的数据数组 (N, 3)
    """
    smoothed_data = np.zeros_like(data)

    for dim in range(3):
        smoothed_data[:, dim] = gaussian_filter1d(data[:, dim], sigma=sigma)

    print(f"数据平滑完成 (sigma={sigma})")
    return smoothed_data


def preprocess_force(force):
    return


def preprocess_pose(shouR, elbowR, wristR, shouL, elbowL, wristL):
    pos_shou_zhuo = shouR[:, :3]
    pos_elbow_zhuo = elbowR[:, :3]
    pos_wrist_zhuo = wristR[:, :3]
    l_zhuo = pos_wrist_zhuo - pos_shou_zhuo
    r_zhuo = pos_elbow_zhuo - pos_shou_zhuo
    l_zhuo[:, [1, 2]] = l_zhuo[:, [2, 1]]
    l_zhuo[:, [0, 1]] = l_zhuo[:, [1, 0]]
    r_zhuo[:, [1, 2]] = r_zhuo[:, [2, 1]]
    r_zhuo[:, [0, 1]] = r_zhuo[:, [1, 0]]

    pos_shou_cz = shouL[:, :3]
    pos_elbow_cz = elbowL[:, :3]
    pos_wrist_cz = wristL[:, :3]
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
    output_dir = "stiffness_results_sawing/tpgmm_data"
    os.makedirs(output_dir, exist_ok=True)

    # 加载姿态和肌电数据
    all_data = np.load("/home/ubuntu/HI-ImpRS-HRC/data/emg_record/sawing/chenzui&zhuo/4/all_data_combined.npy",
                       allow_pickle=True).item()
    shouL = all_data['shouL'][2200:2525]
    elbowL = all_data['elbowL'][2200:2525]
    wristL = all_data['wristL'][2200:2525]
    shouR = all_data['shouR'][2200:2525]
    elbowR = all_data['elbowR'][2200:2525]
    wristR = all_data['wristR'][2200:2525]
    timestamp = all_data['timestamp'][2200:2525]
    emg = all_data['muscle_coactivation'][2200:2525]

    #chenzui&yuchen/6 [950:1300]
    plt.plot(wristL)
    plt.show()

    # 处理肌电信号
    A_zhuo = (emg[:, 2] + emg[:, 3]) / 2
    A_cz = (emg[:, 0] + emg[:, 1]) / 2

    # 定义5组参数
    A = [
        [0.272, 1.314, 3847.141, 151.684],
        # [0.107, 2.200, 2678.765, 149.597],
        # [0.399, 2.926, 1819.695, 128.581],
        # [0.341, 4.073, 2699.123, 112.562],
        # [0.167, 4.528, 1260.290, 94.951]
    ]

    # 保存参数集
    param_sets = np.array(A)
    np.save(os.path.join(output_dir, "parameter_sets.npy"), param_sets)
    print(f"Saved parameter sets to {os.path.join(output_dir, 'parameter_sets.npy')}")

    # 处理姿态数据
    l_zhuo, r_zhuo, l_cz, r_cz = preprocess_pose(shouR, elbowR, wristR, shouL, elbowL, wristL)

    # 目标长度和平滑参数
    TARGET_LENGTH = 325
    SMOOTH_SIGMA = 10  # 高斯滤波标准差

    # 存储插值和平滑后的结果
    all_Ke_diag_zhuo_smooth = []
    all_Ke_diag_cz_smooth = []

    # 对每个参数集计算刚度
    for param_idx, param_set in enumerate(A):
        print(f"\n{'=' * 70}")
        print(f"Processing parameter set {param_idx + 1}: {param_set}")
        print(f"{'=' * 70}")

        # 提取参数
        a1, a2, b1, b2 = param_set

        # ========== 计算陈子刚度 ==========
        print("\n1. Computing CZ stiffness...")
        Ke_cz = []
        Ke_diag_cz = []
        for i in range(len(l_cz[:, 0])):
            Ke = calculate_endpoint_stiffness(l_cz[i, :], r_cz[i, :], a1, a2, b1, b2, 1)
            Ke_cz.append(Ke)
            Ke_diag_cz.append(np.diagonal(Ke))
        Ke_cz = np.array(Ke_cz)
        Ke_diag_cz = np.array(Ke_diag_cz)
        print(f"   Original CZ stiffness shape: {Ke_diag_cz.shape}")

        # 插值到目标长度
        print(f"\n2. Interpolating CZ data to {TARGET_LENGTH} points...")
        Ke_diag_cz_interp = interpolate_to_target_length(
            Ke_diag_cz,
            target_length=TARGET_LENGTH,
            method='cubic'
        )
        print(f"   Interpolated CZ stiffness shape: {Ke_diag_cz_interp.shape}")

        # 平滑处理
        print(f"\n3. Smoothing CZ data...")
        Ke_diag_cz_smooth = smooth_data(Ke_diag_cz_interp, sigma=SMOOTH_SIGMA)

        # 保存陈子刚度数据(插值和平滑后)
        cz_file = os.path.join(output_dir, f"Ke_cz_set{param_idx + 1}.npy")
        np.save(cz_file, Ke_diag_cz_smooth)
        print(f"   Saved smoothed CZ stiffness to {cz_file}")

        # ========== 计算卓的刚度 ==========
        print("\n4. Computing ZHUO stiffness...")
        Ke_zhuo = []
        Ke_diag_zhuo = []
        for i in range(len(l_zhuo[:, 0])):
            Ke2 = calculate_endpoint_stiffness(l_zhuo[i, :], r_zhuo[i, :], a1, a2, b1, b2, 1)
            Ke_zhuo.append(Ke2)
            Ke_diag_zhuo.append(np.diagonal(Ke2))
        Ke_zhuo = np.array(Ke_zhuo)
        Ke_diag_zhuo = np.array(Ke_diag_zhuo)
        print(f"   Original ZHUO stiffness shape: {Ke_diag_zhuo.shape}")

        # 插值到目标长度
        print(f"\n5. Interpolating ZHUO data to {TARGET_LENGTH} points...")
        Ke_diag_zhuo_interp = interpolate_to_target_length(
            Ke_diag_zhuo,
            target_length=TARGET_LENGTH,
            method='cubic'
        )
        print(f"   Interpolated ZHUO stiffness shape: {Ke_diag_zhuo_interp.shape}")

        # 平滑处理
        print(f"\n6. Smoothing ZHUO data...")
        Ke_diag_zhuo_smooth = smooth_data(Ke_diag_zhuo_interp, sigma=SMOOTH_SIGMA)

        # 保存卓的刚度数据(插值和平滑后)
        zhuo_file = os.path.join(output_dir, f"Ke_zhuo_set{param_idx + 1}.npy")
        np.save(zhuo_file, Ke_diag_zhuo_smooth)
        print(f"   Saved smoothed ZHUO stiffness to {zhuo_file}")

        # 存储用于可视化
        all_Ke_diag_cz_smooth.append(Ke_diag_cz_smooth)
        all_Ke_diag_zhuo_smooth.append(Ke_diag_zhuo_smooth)

    # ========== 创建可视化比较图 ==========
    print(f"\n{'=' * 70}")
    print("Creating visualization plots...")
    print(f"{'=' * 70}")

    # 1. 比较ZHUO的刚度 (插值和平滑后)
    print("\n1. Plotting smoothed ZHUO stiffness comparison...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

    for axis_idx in range(3):
        ax = axes[axis_idx]

        for param_idx, Ke_diag in enumerate(all_Ke_diag_zhuo_smooth):
            ax.plot(Ke_diag[:, axis_idx], color=colors[param_idx],
                    label=f'S{param_idx + 1}: a1={A[param_idx][0]}, a2={A[param_idx][1]}, b1={A[param_idx][2]:.1f}, b2={A[param_idx][3]:.1f}',
                    linewidth=1.5)

        ax.set_title(f'{axis_labels[axis_idx]} Stiffness Comparison (Smoothed, N={TARGET_LENGTH})')
        ax.set_ylabel('Stiffness (N/m)')
        ax.grid(True, alpha=0.3)

        if axis_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[2].set_xlabel('Time step')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stiffness_comparison_zhuo.png'), dpi=600)
    plt.close()

    # 2. 比较CZ的刚度 (插值和平滑后)
    print("2. Plotting smoothed CZ stiffness comparison...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for axis_idx in range(3):
        ax = axes[axis_idx]

        for param_idx, Ke_diag in enumerate(all_Ke_diag_cz_smooth):
            ax.plot(Ke_diag[:, axis_idx], color=colors[param_idx],
                    label=f'S{param_idx + 1}: a1={A[param_idx][0]}, a2={A[param_idx][1]}, b1={A[param_idx][2]:.1f}, b2={A[param_idx][3]:.1f}',
                    linewidth=1.5)

        ax.set_title(f'{axis_labels[axis_idx]} Stiffness Comparison (Smoothed, N={TARGET_LENGTH})')
        ax.set_ylabel('Stiffness (N/m)')
        ax.grid(True, alpha=0.3)

        if axis_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[2].set_xlabel('Time step')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stiffness_comparison_cz.png'), dpi=600)
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"All data has been saved to the '{output_dir}' directory")
    print(f"{'=' * 70}")
    print("\nSummary:")
    print(f"  - Target data length: {TARGET_LENGTH}")
    print(f"  - Smoothing sigma: {SMOOTH_SIGMA}")
    print(f"  - Number of parameter sets: {len(A)}")
    print(f"  - Saved files:")
    print(f"    * Ke_cz_set[1-5].npy (interpolated to {TARGET_LENGTH} and smoothed)")
    print(f"    * Ke_zhuo_set[1-5].npy (interpolated to {TARGET_LENGTH} and smoothed)")
    print(f"  - Generated plots:")
    print(f"    * stiffness_comparison_zhuo.png")
    print(f"    * stiffness_comparison_cz.png")
    print(f"\n{'=' * 70}")