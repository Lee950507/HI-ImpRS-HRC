import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


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
    output_dir = "stiffness_results_uni/revised"
    os.makedirs(output_dir, exist_ok=True)

    # 加载姿态和肌电数据
    pose = pd.read_csv('/home/clover/Chenzui/HI-ImpRS-HRC/data/taichi/unimanual_data/2.csv')
    emg = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/taichi/unimanual_data/2.npy')
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

    bad_regions_cz = [[80, 130], [480, 530]]

    # 目标长度
    TARGET_LENGTH = 5400

    # 存储所有参数集的结果
    all_Ke_zhuo = []
    all_Ke_diag_zhuo = []
    all_Ke_cz = []
    all_Ke_diag_cz = []

    # 存储插值后的结果
    all_Ke_diag_zhuo_interp = []
    all_Ke_diag_cz_interp = []

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

        # 修复不连续区域
        print("\n2. Repairing discontinuous regions...")
        Ke_diag_cz = repair_discontinuous_regions(
            Ke_diag_cz,
            bad_regions_cz,
            method='cubic'
        )

        # 插值到目标长度
        print(f"\n3. Interpolating CZ data to {TARGET_LENGTH} points...")
        Ke_diag_cz_interp = interpolate_to_target_length(
            Ke_diag_cz,
            target_length=TARGET_LENGTH,
            method='cubic'
        )
        print(f"   Interpolated CZ stiffness shape: {Ke_diag_cz_interp.shape}")

        # 保存陈子刚度数据(插值后)
        cz_file = os.path.join(output_dir, f"Ke_cz_set{param_idx + 1}.npy")
        np.save(cz_file, Ke_diag_cz_interp)
        print(f"   Saved interpolated CZ stiffness to {cz_file}")

        # 可选:也保存原始长度的数据
        cz_file_original = os.path.join(output_dir, f"Ke_cz_set{param_idx + 1}_original.npy")
        np.save(cz_file_original, Ke_diag_cz)
        print(f"   Saved original CZ stiffness to {cz_file_original}")

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

        # 保存卓的刚度数据(插值后)
        zhuo_file = os.path.join(output_dir, f"Ke_zhuo_set{param_idx + 1}.npy")
        np.save(zhuo_file, Ke_diag_zhuo_interp)
        print(f"   Saved interpolated ZHUO stiffness to {zhuo_file}")

        # 可选:也保存原始长度的数据
        zhuo_file_original = os.path.join(output_dir, f"Ke_zhuo_set{param_idx + 1}_original.npy")
        np.save(zhuo_file_original, Ke_diag_zhuo)
        print(f"   Saved original ZHUO stiffness to {zhuo_file_original}")

        # 存储用于可视化
        all_Ke_cz.append(Ke_cz)
        all_Ke_diag_cz.append(Ke_diag_cz)
        all_Ke_zhuo.append(Ke_zhuo)
        all_Ke_diag_zhuo.append(Ke_diag_zhuo)

        all_Ke_diag_cz_interp.append(Ke_diag_cz_interp)
        all_Ke_diag_zhuo_interp.append(Ke_diag_zhuo_interp)

    # ========== 创建可视化比较图 ==========
    print(f"\n{'=' * 70}")
    print("Creating visualization plots...")
    print(f"{'=' * 70}")

    # 1. 比较ZHUO的刚度 (原始长度)
    print("\n1. Plotting original ZHUO stiffness comparison...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

    for axis_idx in range(3):
        ax = axes[axis_idx]

        for param_idx, Ke_diag in enumerate(all_Ke_diag_zhuo):
            ax.plot(Ke_diag[:, axis_idx], color=colors[param_idx],
                    label=f'S{param_idx + 1}: a1={A[param_idx][0]}, a2={A[param_idx][1]}, b1={A[param_idx][2]:.1f}, b2={A[param_idx][3]:.1f}',
                    linewidth=1.5)

        ax.set_title(f'{axis_labels[axis_idx]} Stiffness Comparison (Original Length)')
        ax.set_ylabel('Stiffness (N/m)')
        ax.grid(True, alpha=0.3)

        if axis_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[2].set_xlabel('Time step')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stiffness_comparison_zhuo_original.png'), dpi=600)
    plt.close()

    # 2. 比较ZHUO的刚度 (插值后)
    print("2. Plotting interpolated ZHUO stiffness comparison...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for axis_idx in range(3):
        ax = axes[axis_idx]

        for param_idx, Ke_diag in enumerate(all_Ke_diag_zhuo_interp):
            ax.plot(Ke_diag[:, axis_idx], color=colors[param_idx],
                    label=f'S{param_idx + 1}: a1={A[param_idx][0]}, a2={A[param_idx][1]}, b1={A[param_idx][2]:.1f}, b2={A[param_idx][3]:.1f}',
                    linewidth=1.5)

        ax.set_title(f'{axis_labels[axis_idx]} Stiffness Comparison (Interpolated to {TARGET_LENGTH})')
        ax.set_ylabel('Stiffness (N/m)')
        ax.grid(True, alpha=0.3)

        if axis_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[2].set_xlabel('Time step')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stiffness_comparison_zhuo_interpolated.png'), dpi=600)
    plt.close()

    # 3. 比较CZ的刚度 (原始长度)
    print("3. Plotting original CZ stiffness comparison...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for axis_idx in range(3):
        ax = axes[axis_idx]

        for param_idx, Ke_diag in enumerate(all_Ke_diag_cz):
            ax.plot(Ke_diag[:, axis_idx], color=colors[param_idx],
                    label=f'S{param_idx + 1}: a1={A[param_idx][0]}, a2={A[param_idx][1]}, b1={A[param_idx][2]:.1f}, b2={A[param_idx][3]:.1f}',
                    linewidth=1.5)

        ax.set_title(f'{axis_labels[axis_idx]} Stiffness Comparison (Original Length)')
        ax.set_ylabel('Stiffness (N/m)')
        ax.grid(True, alpha=0.3)
        if axis_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[2].set_xlabel('Time step')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stiffness_comparison_cz_original.png'), dpi=600)
    plt.close()

    # 4. 比较CZ的刚度 (插值后)
    print("4. Plotting interpolated CZ stiffness comparison...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for axis_idx in range(3):
        ax = axes[axis_idx]

        for param_idx, Ke_diag in enumerate(all_Ke_diag_cz_interp):
            ax.plot(Ke_diag[:, axis_idx], color=colors[param_idx],
                    label=f'S{param_idx + 1}: a1={A[param_idx][0]}, a2={A[param_idx][1]}, b1={A[param_idx][2]:.1f}, b2={A[param_idx][3]:.1f}',
                    linewidth=1.5)

        ax.set_title(f'{axis_labels[axis_idx]} Stiffness Comparison (Interpolated to {TARGET_LENGTH})')
        ax.set_ylabel('Stiffness (N/m)')
        ax.grid(True, alpha=0.3)

        if axis_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[2].set_xlabel('Time step')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stiffness_comparison_cz_interpolated.png'), dpi=600)
    plt.close()

    # 5. 创建对比图 (原始 vs 插值)
    print("5. Plotting original vs interpolated comparison...")
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))

    for axis_idx in range(3):
        # 原始数据 (以第一个参数集为例)
        ax_orig = axes[axis_idx, 0]
        ax_orig.plot(all_Ke_diag_cz[0][:, axis_idx], 'b-', linewidth=1.5, label='Original')
        ax_orig.set_title(f'{axis_labels[axis_idx]} - Original (N={all_Ke_diag_cz[0].shape[0]})')
        ax_orig.set_ylabel('Stiffness (N/m)')
        ax_orig.grid(True, alpha=0.3)
        ax_orig.legend()

        # 插值数据
        ax_interp = axes[axis_idx, 1]
        ax_interp.plot(all_Ke_diag_cz_interp[0][:, axis_idx], 'r-', linewidth=1.5, label='Interpolated')
        ax_interp.set_title(f'{axis_labels[axis_idx]} - Interpolated (N={TARGET_LENGTH})')
        ax_interp.set_ylabel('Stiffness (N/m)')
        ax_interp.grid(True, alpha=0.3)
        ax_interp.legend()

    axes[2, 0].set_xlabel('Time step')
    axes[2, 1].set_xlabel('Time step')
    plt.suptitle('CZ Stiffness: Original vs Interpolated (Parameter Set 1)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stiffness_original_vs_interpolated.png'), dpi=600)
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"All data has been saved to the '{output_dir}' directory")
    print(f"{'=' * 70}")
    print("\nSummary:")
    print(f"  - Original data length: {all_Ke_diag_cz[0].shape[0]}")
    print(f"  - Interpolated data length: {TARGET_LENGTH}")
    print(f"  - Number of parameter sets: {len(A)}")
    print(f"  - Saved files:")
    print(f"    * Ke_cz_set[1-5].npy (interpolated to {TARGET_LENGTH})")
    print(f"    * Ke_zhuo_set[1-5].npy (interpolated to {TARGET_LENGTH})")
    print(f"    * Ke_cz_set[1-5]_original.npy (original length)")
    print(f"    * Ke_zhuo_set[1-5]_original.npy (original length)")
    print(f"  - Generated plots:")
    print(f"    * stiffness_comparison_zhuo_original.png")
    print(f"    * stiffness_comparison_zhuo_interpolated.png")
    print(f"    * stiffness_comparison_cz_original.png")
    print(f"    * stiffness_comparison_cz_interpolated.png")
    print(f"    * stiffness_original_vs_interpolated.png")
    print(f"\n{'=' * 70}")