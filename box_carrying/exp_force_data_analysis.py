import numpy as np
import matplotlib.pyplot as plt
import os
import re
from glob import glob
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import pandas as pd

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 20
})

# 定义数据路径和文件名
base_dir = "/home/ubuntu/TCDS_data/0620_hrc_box_zhuo/robot"  # 替换为您的数据目录

# 方法名称（用于标题和图例）
method_names = ["FIC", "learning-based VIC", "EMG-based VIC", "HI-ImpRS"]

# 读取方法顺序
# method_order = np.array([1, 3, 4, 2, 3, 4, 1, 2, 3, 1, 4, 2, 2, 4, 1, 3, 1, 2, 3, 4]) # yiming box
# method_order = np.array([3, 1, 4, 2, 4, 1, 3, 2, 1, 2, 3, 4, 4, 1, 2, 3, 2, 4, 1, 3])  # wuxi box; yuchen box
method_order = np.array([4, 2, 3, 1, 3, 4, 2, 1, 1, 4, 2, 3, 1, 3, 2, 4, 2, 1, 4, 3])  # zhuo box

manual_trajectory_segments = {
    "FIC": {

    },
    "learning-based VIC": {
# wuxi box
#         1: (28434, 39000),
    },
    "EMG-based VIC": {
# yiming box
#         0: (34478, 45101),
#         1: (34804, 46000),
#         2: (36400, 46100),
#         3: (29600, 40153),
#         4: (26500, 37314)
# wuxi box
#         0: (30500, 41005),
# zhuo box
#         0: (48872, 60000),
#         1: (30452, 41000),
#         2: (34305, 45000),
#         4: (31731, 42000)
    },
    "HI-ImpRS": {
# yiming box
#         0: (52356, 63000),
# wuxi box
#         0: (28421, 38500),
#         2: (28849, 39100),
#         4: (27714, 37500),
# zhuo box
#         0: (48145, 59500),
#         1: (30708, 42000),
#         3: (29042, 40300),
#         4: (37135, 47880)
    }
}


def apply_filter(data, filter_type='savgol', **kwargs):
    """
    对数据应用不同类型的滤波器

    参数:
        data: 输入数据数组
        filter_type: 滤波器类型 ('savgol', 'moving_avg', 'butterworth', 'gaussian', 'median')
        **kwargs: 特定滤波器的参数

    返回:
        滤波后的数据
    """
    if filter_type == 'savgol':
        # Savitzky-Golay滤波器 - 保持峰值特征
        window_length = kwargs.get('window_length', 51)  # 必须是奇数
        polyorder = kwargs.get('polyorder', 3)

        # 确保窗口长度是奇数且不超过数据长度
        if window_length >= len(data):
            window_length = min(len(data) - (1 if len(data) % 2 == 0 else 0), 11)
        if window_length % 2 == 0:
            window_length += 1

        return signal.savgol_filter(data, window_length, polyorder)

    elif filter_type == 'moving_avg':
        # 移动平均滤波
        window_size = kwargs.get('window_size', 10)
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

    elif filter_type == 'butterworth':
        # Butterworth低通滤波器
        cutoff = kwargs.get('cutoff', 0.1)  # 截止频率，0到1之间
        order = kwargs.get('order', 4)

        b, a = signal.butter(order, cutoff, 'low')
        return signal.filtfilt(b, a, data)

    elif filter_type == 'gaussian':
        # 高斯滤波
        sigma = kwargs.get('sigma', 2)  # 标准差
        return gaussian_filter1d(data, sigma)

    elif filter_type == 'median':
        # 中值滤波 - 去除脉冲噪声
        kernel_size = kwargs.get('kernel_size', 5)
        return signal.medfilt(data, kernel_size)

    else:
        print(f"Unknown filter type: {filter_type}, returning original data")
        return data


def enhanced_force_filter(force_data):
    """
    对力数据应用增强滤波，结合多种滤波方法

    参数:
        force_data: 输入的力数据

    返回:
        滤波后的力数据
    """
    # 1. 首先应用中值滤波去除突发噪声
    filtered = signal.medfilt(force_data, 5)

    # 2. 应用Butterworth低通滤波去除高频噪声
    b, a = signal.butter(4, 0.15, 'low')
    filtered = signal.filtfilt(b, a, filtered)

    # 3. 最后应用Savitzky-Golay滤波保持信号特征
    window_length = min(101, len(filtered) // 10)
    if window_length % 2 == 0:
        window_length += 1
    filtered = signal.savgol_filter(filtered, window_length, 3)

    return filtered


def enhanced_adaptive_smoothing(force_data, time_data, method_name=None, exp_idx=None):
    """
    对力数据进行增强版自适应滤波，使用超强滤波力度处理异常区域

    参数:
        force_data: 力数据数组
        time_data: 对应的时间数据
        method_name: 方法名称，用于日志
        exp_idx: 实验索引，用于日志

    返回:
        处理后的力数据和异常区域
    """
    # 复制原始数据
    processed_data = np.copy(force_data)
    anomaly_regions = []

    # 1. 初步处理 - 应用全局中值滤波以减少小噪声
    processed_data = signal.medfilt(processed_data, 5)

    # 2. 检测异常区域 - 使用一阶导数检测快速变化
    derivative = np.abs(np.diff(processed_data))

    # 计算导数的阈值 - 使用较低阈值以检测更多潜在异常
    q75, q25 = np.percentile(derivative, [75, 25])
    iqr = q75 - q25
    threshold = q75 + 3.0 * iqr  # 降低阈值因子，捕获更多异常

    # 找出超过阈值的点
    anomaly_points = np.where(derivative > threshold)[0]

    if len(anomaly_points) > 0:
        # 将相邻的异常点分组，使用更大的间隔允许捕获整个异常区域
        max_gap = 30  # 增大允许的最大点间隔
        groups = []
        current_group = [anomaly_points[0]]

        for i in range(1, len(anomaly_points)):
            if anomaly_points[i] - anomaly_points[i - 1] <= max_gap:
                current_group.append(anomaly_points[i])
            else:
                if len(current_group) > 1:  # 只需要2个点就算有效异常
                    groups.append(current_group)
                current_group = [anomaly_points[i]]

        if current_group and len(current_group) > 1:
            groups.append(current_group)

        # 3. 对每个异常区域应用超强滤波
        for group in groups:
            # 确定异常区域的扩展范围
            buffer = 100  # 大幅增加缓冲区，确保捕获完整异常
            start_idx = max(0, min(group) - buffer)
            end_idx = min(len(force_data), max(group) + buffer + 1)

            # 检查区域大小
            if end_idx - start_idx < 5:
                continue  # 区域太小，跳过

            # 记录异常区域用于可视化
            anomaly_regions.append((start_idx, end_idx))

            # print(f"Applying ultra-strong smoothing to region {start_idx}-{end_idx} "
                  # f"(time {time_data[start_idx]:.2f}s to {time_data[end_idx - 1]:.2f}s)")

            # 提取需要处理的区域
            region_data = processed_data[start_idx:end_idx]

            # 4. 应用多阶段极强滤波
            # 第一阶段: 大窗口中值滤波彻底去除尖峰
            window_size = min(31, len(region_data) // 3)
            if window_size % 2 == 0:
                window_size += 1
            filtered = signal.medfilt(region_data, window_size)

            # 第二阶段: 应用更大窗口的中值滤波进一步平滑
            window_size = min(51, len(filtered) // 2)
            if window_size % 2 == 0:
                window_size += 1
            if len(filtered) > window_size:
                filtered = signal.medfilt(filtered, window_size)

            # 第三阶段: 大窗口Savitzky-Golay滤波，低阶多项式以增强平滑效果
            if len(filtered) > 51:
                sg_window = 101
                sg_order = 1  # 使用1阶多项式，更平滑
            elif len(filtered) > 31:
                sg_window = 51
                sg_order = 1
            elif len(filtered) > 11:
                sg_window = 21
                sg_order = 1
            else:
                sg_window = min(len(filtered) - (1 if len(filtered) % 2 == 0 else 0), 5)
                sg_order = 1

            if sg_window % 2 == 0:
                sg_window -= 1

            if len(filtered) > sg_window:
                filtered = signal.savgol_filter(filtered, sg_window, sg_order)

            # 第四阶段: 应用强高斯滤波
            filtered = gaussian_filter1d(filtered, sigma=50.0)  # 大幅增加sigma值

            # 第五阶段: 再次应用Savitzky-Golay进一步平滑
            if len(filtered) > sg_window:
                filtered = signal.savgol_filter(filtered, sg_window, sg_order)

            # 5. 将处理后的数据无缝融合回原始数据
            # 创建一个线性混合比例数组用于平滑过渡
            blend_ratio = np.ones(end_idx - start_idx)

            # 在边缘处创建过渡区
            transition = min(buffer // 2, 40)
            if transition > 0 and len(blend_ratio) > 2 * transition:
                # 前部过渡区
                for i in range(transition):
                    blend_ratio[i] = i / transition

                # 后部过渡区
                for i in range(1, transition + 1):
                    if end_idx - start_idx - i >= 0:
                        blend_ratio[end_idx - start_idx - i] = (i - 1) / transition

            # 混合原始数据和滤波数据，完全移除尖峰
            blended_data = (1 - blend_ratio) * region_data + blend_ratio * filtered

            # 更新处理后的数据
            processed_data[start_idx:end_idx] = blended_data

    # 6. 最终全局滤波以确保整体平滑
    # 应用中等窗口Savitzky-Golay滤波
    window_size = min(51, len(processed_data) // 10)
    if window_size % 2 == 0:
        window_size += 1
    if len(processed_data) > window_size:
        processed_data = signal.savgol_filter(processed_data, window_size, 2)

    # 应用轻度高斯滤波以确保整体平滑
    processed_data = gaussian_filter1d(processed_data, sigma=2.0)

    return processed_data, anomaly_regions


def find_trajectory_segment(position_data, time_data, method_name=None, exp_idx=None):
    """
    根据Z轴位置变化找出有效轨迹段：先上升，然后保持水平，最后下降
    主要关注数据的后50%部分

    参数:
        position_data: 位置数据数组，形状为(n, 3)，表示XYZ三个方向
        time_data: 对应的时间数据
        method_name: 当前处理的方法名称，用于特定方法的调整
        exp_idx: 实验索引，用于查找手动设置的轨迹段

    返回:
        start_idx, end_idx: 开始和结束索引
    """
    # 首先检查是否有手动设置的轨迹段
    if method_name and exp_idx is not None and method_name in manual_trajectory_segments:
        if exp_idx in manual_trajectory_segments[method_name]:
            start_idx, end_idx = manual_trajectory_segments[method_name][exp_idx]
            print(f"Using manually set trajectory segment for {method_name}, experiment {exp_idx + 1}: "
                  f"{time_data[start_idx]:.2f}s to {time_data[end_idx]:.2f}s")
            return start_idx, end_idx

    # 只关注数据的后50%
    half_point = len(position_data) // 3
    pos_z = position_data[half_point:, 2]  # Z轴数据

    # 应用滤波以平滑数据
    window_length = min(51, len(pos_z) // 10)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 5:
        pos_z_smooth = signal.savgol_filter(pos_z, window_length, 3)
    else:
        pos_z_smooth = pos_z

    # 计算Z轴的导数（速度）
    z_vel = np.gradient(pos_z_smooth)

    # 再次平滑速度数据
    if window_length >= 5:
        z_vel_smooth = signal.savgol_filter(z_vel, window_length, 3)
    else:
        z_vel_smooth = z_vel

    # 识别轨迹的三个阶段：上升、平稳、下降
    vel_std = np.std(z_vel_smooth)

    # 上升阶段：速度明显为正
    rise_indices = np.where(z_vel_smooth > vel_std * 0.8)[0]

    # 下降阶段：速度明显为负
    fall_indices = np.where(z_vel_smooth < -vel_std * 0.8)[0]

    if len(rise_indices) > 0 and len(fall_indices) > 0:
        # 找出最长的连续上升段
        rise_segments = np.split(rise_indices, np.where(np.diff(rise_indices) != 1)[0] + 1)
        longest_rise = max(rise_segments, key=len)

        # 找出最长的连续下降段
        fall_segments = np.split(fall_indices, np.where(np.diff(fall_indices) != 1)[0] + 1)
        longest_fall = max(fall_segments, key=len)

        # 确保上升在下降之前
        if longest_rise[-1] < longest_fall[0]:
            start_idx = half_point + longest_rise[0]
            end_idx = half_point + longest_fall[-1]
        else:
            # 如果下降在上升之前，可能有多个循环，使用第一个上升和最后一个下降
            start_idx = half_point + rise_indices[0]
            end_idx = half_point + fall_indices[-1]
    else:
        # 如果没有找到明显的上升和下降，使用后半段数据
        print("Warning: Could not identify specific trajectory pattern, using latter half of data")
        start_idx = half_point
        end_idx = len(time_data) - 1

    # 打印自动识别的结果，以便用户手动调整
    print(f"Auto-identified trajectory segment for {method_name}, experiment {exp_idx + 1}: "
          f"indices ({start_idx}, {end_idx}), time {time_data[start_idx]:.2f}s to {time_data[end_idx]:.2f}s")

    return start_idx, end_idx


def calculate_smoothness(data):
    """
    计算数据的平滑度指标

    参数:
        data: 输入数据数组

    返回:
        平滑度指标（越低越平滑）
    """
    # 计算一阶导数（差分）
    first_derivative = np.diff(data)

    # 计算二阶导数（差分的差分）
    second_derivative = np.diff(first_derivative)

    # 计算平滑度指标
    # 1. 一阶导数的均方根（RMSD）
    first_derivative_rmsd = np.sqrt(np.mean(np.square(first_derivative)))

    # 2. 二阶导数的均方根（RMSD）- 衡量曲率变化
    second_derivative_rmsd = np.sqrt(np.mean(np.square(second_derivative)))

    # 3. 综合指标（归一化后的加权和）
    smoothness = 0.3 * first_derivative_rmsd + 0.7 * second_derivative_rmsd

    return {
        'first_derivative_rmsd': first_derivative_rmsd,
        'second_derivative_rmsd': second_derivative_rmsd,
        'combined_smoothness': smoothness
    }


def compare_force_smoothness_and_means(force_data, position_data, method_names):
    """
    比较不同方法力数据的平滑度和平均值

    参数:
        force_data: 力数据列表
        position_data: 位置数据列表
        method_names: 方法名称列表
    """
    # 存储各方法的平滑度指标和均值
    smoothness_metrics = {
        'x': [],
        'y': [],
        'z': []
    }

    force_means = {
        'x': [],
        'y': [],
        'z': []
    }

    force_stds = {
        'x': [],
        'y': [],
        'z': []
    }

    valid_methods = []

    # 计算每个方法的平滑度指标和均值
    for method_idx, method_name in enumerate(method_names):
        if not force_data[method_idx] or not position_data[method_idx]:
            continue

        valid_methods.append(method_name)

        # 使用第一个实验的数据
        pos_time, _, _, _, traj_start_idx, traj_end_idx = position_data[method_idx][0]
        force_time, force_x, force_y, force_z = force_data[method_idx][0]

        # 找到对应的力数据段
        force_start_idx = np.searchsorted(force_time, pos_time[traj_start_idx], side='left')
        force_end_idx = np.searchsorted(force_time, pos_time[traj_end_idx], side='right')

        # 提取有效轨迹段的力数据
        force_x_segment = force_x[force_start_idx:force_end_idx]
        force_y_segment = force_y[force_start_idx:force_end_idx]
        force_z_segment = force_z[force_start_idx:force_end_idx]

        if method_idx == 0:
            # force_x_segment = force_x_segment + 0.7
            force_y_segment = force_y_segment + 0.0
        # if method_idx == 1:
        #     force_x_segment = force_x_segment - 0.6
        # if method_idx == 2:
        #     force_x_segment = force_x_segment + 1.5

        # 计算平滑度指标
        x_smoothness = calculate_smoothness(force_x_segment)
        y_smoothness = calculate_smoothness(force_y_segment)
        z_smoothness = calculate_smoothness(force_z_segment)

        smoothness_metrics['x'].append(x_smoothness['combined_smoothness'])
        smoothness_metrics['y'].append(y_smoothness['combined_smoothness'])
        smoothness_metrics['z'].append(z_smoothness['combined_smoothness'])

        # 计算力数据均值和标准差
        force_means['x'].append(np.mean(abs(force_x_segment)))
        force_means['y'].append(np.mean(abs(force_y_segment)))
        force_means['z'].append(np.mean(abs(force_z_segment)))

        force_stds['x'].append(np.std(abs(force_x_segment)))
        force_stds['y'].append(np.std(abs(force_y_segment)))
        force_stds['z'].append(np.std(abs(force_z_segment)))

        print(f"Method: {method_name}")
        print(
            f"  X Force - Mean: {force_means['x'][-1]:.4f} N, Std: {force_stds['x'][-1]:.4f} N, Smoothness: {smoothness_metrics['x'][-1]:.6f}")
        print(
            f"  Y Force - Mean: {force_means['y'][-1]:.4f} N, Std: {force_stds['y'][-1]:.4f} N, Smoothness: {smoothness_metrics['y'][-1]:.6f}")
        print(
            f"  Z Force - Mean: {force_means['z'][-1]:.4f} N, Std: {force_stds['z'][-1]:.4f} N, Smoothness: {smoothness_metrics['z'][-1]:.6f}")

    # 创建平滑度对比图
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    directions = ['x', 'y', 'z']
    titles = ['X Force Smoothness', 'Y Force Smoothness', 'Z Force Smoothness']

    for i, direction in enumerate(directions):
        axs[i].bar(valid_methods, smoothness_metrics[direction], color='skyblue')
        axs[i].set_title(titles[i])
        axs[i].set_ylabel('Smoothness Metric')
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].grid(False)

        # 在每个柱子上标注具体值
        # for j, v in enumerate(smoothness_metrics[direction]):
        #     axs[i].text(j, v + 0.001, f"{v:.4f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'force_smoothness_comparison.png'), dpi=400, bbox_inches='tight')
    plt.close()

    # 创建X和Y方向力均值对比图
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    # X方向力均值
    axs[0].bar(valid_methods, force_means['x'], color='lightcoral')
    axs[0].set_title('X Direction Force Mean')
    axs[0].set_ylabel('Force (N)')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].grid(False)

    # 添加误差线表示标准差
    axs[0].errorbar(valid_methods, force_means['x'], yerr=force_stds['x'], fmt='o', color='black', capsize=5)

    # 在每个柱子上标注具体均值
    for j, v in enumerate(force_means['x']):
        axs[0].text(j, v + (0.01 if v >= 0 else -0.3), f"{v:.2f}", ha='center')

    # Y方向力均值
    axs[1].bar(valid_methods, force_means['y'], color='lightgreen')
    axs[1].set_title('Y Direction Force Mean')
    axs[1].set_ylabel('Force (N)')
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].grid(False)

    # 添加误差线表示标准差
    axs[1].errorbar(valid_methods, force_means['y'], yerr=force_stds['y'], fmt='o', color='black', capsize=5)

    # 在每个柱子上标注具体均值
    for j, v in enumerate(force_means['y']):
        axs[1].text(j, v + (0.01 if v >= 0 else -0.3), f"{v:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'force_means_comparison.png'), dpi=400, bbox_inches='tight')
    plt.close()

    # 创建完整的力数据统计对比表格图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    table_data = []
    for i, method in enumerate(valid_methods):
        table_data.append([
            method,
            f"{force_means['x'][i]:.3f} ± {force_stds['x'][i]:.3f}",
            f"{force_means['y'][i]:.3f} ± {force_stds['y'][i]:.3f}",
            f"{force_means['z'][i]:.3f} ± {force_stds['z'][i]:.3f}",
            f"{smoothness_metrics['x'][i]:.5f}",
            f"{smoothness_metrics['y'][i]:.5f}",
            f"{smoothness_metrics['z'][i]:.5f}"
        ])

    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=['Method', 'X Force (N)', 'Y Force (N)', 'Z Force (N)', 'X Smoothness', 'Y Smoothness',
                   'Z Smoothness'],
        loc='center',
        cellLoc='center'
    )

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 添加标题
    plt.suptitle('Force Data Statistics Comparison', fontsize=16, y=0.95)

    plt.savefig(os.path.join(base_dir, 'force_statistics_table.png'), dpi=400, bbox_inches='tight')
    plt.close()

    return valid_methods, force_means, force_stds, smoothness_metrics


def standardize_trajectory_segments(position_data, method_idx):
    """
    为同一方法的所有实验标准化轨迹段长度和起止点

    参数:
        position_data: 位置数据列表
        method_idx: 方法索引

    返回:
        修正后的轨迹段起止索引
    """
    # 如果该方法没有足够的数据，直接返回
    if len(position_data[method_idx]) < 2:
        return None

    # 提取所有实验的轨迹段信息
    segments = []
    for exp_idx, (pos_time, _, _, _, traj_start_idx, traj_end_idx) in enumerate(position_data[method_idx]):
        duration = pos_time[traj_end_idx] - pos_time[traj_start_idx]
        segments.append((exp_idx, traj_start_idx, traj_end_idx, duration))

    # 如果是EMG-based VIC方法，进行特殊处理
    if method_names[method_idx] == "EMG-based VIC":
        # 根据持续时间排序，找出中位数持续时间的实验
        sorted_segments = sorted(segments, key=lambda x: x[3])
        median_segment = sorted_segments[len(sorted_segments) // 2]
        median_duration = median_segment[3]

        # 使用中位数持续时间作为标准
        standardized_segments = []
        for exp_idx, start_idx, end_idx, duration in segments:
            # 如果持续时间相差太大，调整端点
            if abs(duration - median_duration) / median_duration > 0.3:  # 允许30%的偏差
                # 获取该实验的数据
                pos_time = position_data[method_idx][exp_idx][0]

                # 计算需要的采样点数
                target_samples = int(median_duration / (pos_time[1] - pos_time[0]))

                # 调整终点
                new_end_idx = start_idx + target_samples
                if new_end_idx >= len(pos_time):
                    # 如果超出范围，则调整起点
                    new_end_idx = len(pos_time) - 1
                    new_start_idx = max(0, new_end_idx - target_samples)
                else:
                    new_start_idx = start_idx

                standardized_segments.append((exp_idx, new_start_idx, new_end_idx))
            else:
                standardized_segments.append((exp_idx, start_idx, end_idx))

        return standardized_segments

    # 其他方法不做特殊处理
    return None


def visualize_full_trajectory(position_data, method_idx):
    """
    可视化完整轨迹，并标记自动识别的段

    参数:
        position_data: 位置数据列表
        method_idx: 方法索引
    """
    if not position_data[method_idx]:
        return

    plt.figure(figsize=(16, 8))
    for exp_idx, (pos_time, pos_x, pos_y, pos_z, traj_start_idx, traj_end_idx) in enumerate(position_data[method_idx]):
        plt.plot(pos_time, pos_z, label=f'Experiment {exp_idx + 1}')

        # 标记轨迹段
        plt.axvline(x=pos_time[traj_start_idx], color=f'C{exp_idx}', linestyle='--')
        plt.axvline(x=pos_time[traj_end_idx], color=f'C{exp_idx}', linestyle='--')

        # 添加索引标注
        plt.text(pos_time[traj_start_idx], np.min(pos_z), f'{traj_start_idx}',
                 color=f'C{exp_idx}', fontsize=10)
        plt.text(pos_time[traj_end_idx], np.min(pos_z), f'{traj_end_idx}',
                 color=f'C{exp_idx}', fontsize=10)

    plt.title(f'{method_names[method_idx]} - Full Z Position Trajectories with Auto-identified Segments')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'{method_names[method_idx]}_full_trajectories.png'), dpi=400)
    plt.close()


def align_time_series(source_time, source_data, target_time):
    """
    将一个时间序列对齐到另一个时间序列

    参数:
        source_time: 源时间数组
        source_data: 源数据数组
        target_time: 目标时间数组

    返回:
        对齐后的数据数组
    """
    # 使用线性插值
    from scipy.interpolate import interp1d

    # 确保时间序列是严格递增的
    valid_indices = np.where(np.diff(source_time) > 0)[0]
    if len(valid_indices) < len(source_time) - 1:
        # 如果有相同的时间点，选择第一个
        unique_times, unique_indices = np.unique(source_time, return_index=True)
        source_time = source_time[unique_indices]
        source_data = source_data[unique_indices]

    # 创建插值函数
    f = interp1d(source_time, source_data, kind='linear', bounds_error=False, fill_value='extrapolate')

    # 对目标时间序列进行插值
    aligned_data = f(target_time)

    return aligned_data


# 查找所有实验文件夹
experiment_folders = sorted(glob(os.path.join(base_dir, "*")),
                            key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1))
                            if re.search(r'(\d+)', os.path.basename(x)) else float('inf'))

print(f"Found {len(experiment_folders)} experiment folders")

# 创建数据结构以存储每种方法的力数据和位置数据
# force_data[method_index][experiment_index] = (time_data, force_x, force_y, force_z, traj_start_idx, traj_end_idx)
force_data = [[] for _ in range(4)]
# position_data[method_index][experiment_index] = (time_data, pos_x, pos_y, pos_z)
position_data = [[] for _ in range(4)]

# 加载并组织数据
for folder_idx, folder in enumerate(experiment_folders):
    folder_name = os.path.basename(folder)
    print(f"Processing folder {folder_name} ({folder_idx + 1}/{len(experiment_folders)})")

    # 从文件夹名称中提取实验编号
    match = re.search(r'(\d+)', folder_name)
    if match:
        exp_num = int(match.group(1))
        print(f"  - Extracted experiment number: {exp_num}")
    else:
        print(f"  - Could not extract experiment number from {folder_name}, skipping")
        continue

    # 确定此文件夹使用的方法
    if folder_idx < len(method_order):
        method_idx = method_order[folder_idx] - 1  # 转换为0-based索引
    else:
        print(f"Warning: No method specified for folder {folder_name}, skipping")
        continue

    # 加载位置数据
    try:
        position_file = os.path.join(folder, "tcp_actual_position_rpy_r.txt")

        if os.path.exists(position_file):
            print(f"  - Found position file: tcp_actual_position_rpy_r.txt")

            # 读取位置数据
            try:
                position_data_raw = pd.read_csv(position_file, sep=' ', header=None)
            except:
                # 尝试不同的分隔符
                try:
                    position_data_raw = pd.read_csv(position_file, sep='\t', header=None)
                except:
                    try:
                        position_data_raw = pd.read_csv(position_file, sep=',', header=None)
                    except Exception as e:
                        print(f"  - Failed to read position file with various delimiters: {e}")
                        continue

            # 检查数据列数
            if position_data_raw.shape[1] >= 4:
                # 提取时间和位置数据
                pos_time = position_data_raw.iloc[:, 0].values / 1000000
                pos_x = position_data_raw.iloc[:, 2].values
                pos_y = position_data_raw.iloc[:, 4].values
                pos_z = position_data_raw.iloc[:, 6].values

                print(f"  - Position data shape: {position_data_raw.shape}")

                # 应用滤波处理
                pos_x_filtered = apply_filter(pos_x, filter_type='savgol', window_length=31, polyorder=3)
                pos_y_filtered = apply_filter(pos_y, filter_type='savgol', window_length=31, polyorder=3)
                pos_z_filtered = apply_filter(pos_z, filter_type='savgol', window_length=31, polyorder=3)

                # 找出有效轨迹段
                pos_data_combined = np.column_stack((pos_x_filtered, pos_y_filtered, pos_z_filtered))
                traj_start_idx, traj_end_idx = find_trajectory_segment(pos_data_combined, pos_time,
                                                                       method_name=method_names[method_idx],
                                                                       exp_idx=len(position_data[method_idx]))

                # 存储位置数据
                position_data[method_idx].append(
                    (pos_time, pos_x_filtered, pos_y_filtered, pos_z_filtered, traj_start_idx, traj_end_idx))
                print(f"  - Added position data to Method {method_idx + 1} ({method_names[method_idx]})")
            else:
                print(f"  - Position data has insufficient columns: {position_data_raw.shape[1]}, expected >=4")
                continue  # 如果没有位置数据，则跳过此文件夹
        else:
            print(f"  - No position data file found in folder {folder_name}")
            continue  # 如果没有位置数据，则跳过此文件夹

    except Exception as e:
        print(f"Error processing position data from folder {folder_name}: {e}")
        import traceback

        traceback.print_exc()
        continue  # 如果位置数据处理出错，则跳过此文件夹

    # 加载交互力数据
    try:
        # 查找力数据文件
        force_file = os.path.join(folder, "tcp_actual_force_r.txt")

        if os.path.exists(force_file):
            print(f"  - Found force file: tcp_actual_force_r.txt")

            # 读取力数据
            try:
                force_data_raw = pd.read_csv(force_file, sep=' ', header=None)
            except:
                # 尝试不同的分隔符
                try:
                    force_data_raw = pd.read_csv(force_file, sep='\t', header=None)
                except:
                    try:
                        force_data_raw = pd.read_csv(force_file, sep=',', header=None)
                    except Exception as e:
                        print(f"  - Failed to read force file with various delimiters: {e}")
                        continue

            # 检查数据列数
            if force_data_raw.shape[1] >= 4:
                # 提取时间和力数据
                force_time = force_data_raw.iloc[:, 0].values / 1000000
                force_x = force_data_raw.iloc[:, 2].values
                force_y = force_data_raw.iloc[:, 4].values
                force_z = force_data_raw.iloc[:, 6].values

                print(f"  - Force data shape: {force_data_raw.shape}")

                # 应用滤波处理
                force_x_filtered = enhanced_force_filter(force_x)
                force_y_filtered = enhanced_force_filter(force_y)
                force_z_filtered = enhanced_force_filter(force_z)

                force_x_filtered, x_regions = enhanced_adaptive_smoothing(force_x_filtered, force_time,
                                                                          method_names[method_idx],
                                                                          len(force_data[method_idx]))
                force_y_filtered, y_regions = enhanced_adaptive_smoothing(force_y_filtered, force_time,
                                                                          method_names[method_idx],
                                                                          len(force_data[method_idx]))
                force_z_filtered, z_regions = enhanced_adaptive_smoothing(force_z_filtered, force_time,
                                                                          method_names[method_idx],
                                                                          len(force_data[method_idx]))

                # 存储力数据，并包含轨迹段索引
                force_data[method_idx].append((force_time, force_x_filtered, force_y_filtered, force_z_filtered))
                print(f"  - Added force data to Method {method_idx + 1} ({method_names[method_idx]})")
            else:
                print(f"  - Force data has insufficient columns: {force_data_raw.shape[1]}, expected >=4")
        else:
            print(f"  - No force data file found in folder {folder_name}")

    except Exception as e:
        print(f"Error processing force data from folder {folder_name}: {e}")
        import traceback

        traceback.print_exc()

# 定义颜色和线型
colors = ['blue', 'red', 'green', 'purple', 'orange']
line_styles = ['-', '--', '-.', ':', '-']

# 在处理完所有数据后
for method_idx in range(4):
    if position_data[method_idx]:
        visualize_full_trajectory(position_data, method_idx)

# 分别绘制每个实验的有效轨迹段数据
# 为每种方法创建位置和力的组合图表
for method_idx in range(4):
    # 检查是否有足够的数据
    if not force_data[method_idx] or not position_data[method_idx]:
        print(f"Insufficient data for Method {method_idx + 1} ({method_names[method_idx]}), skipping")
        continue

    if len(force_data[method_idx]) != len(position_data[method_idx]):
        print(f"Mismatch between force and position data for Method {method_idx + 1}, skipping")
        continue

    # 确定此方法有多少组数据
    num_experiments = min(len(force_data[method_idx]), len(position_data[method_idx]))
    print(f"Creating plots for {method_names[method_idx]} with {num_experiments} sets")

    # 创建一个图表，包含六个子图（位置和力的X, Y, Z方向）
    fig, axes = plt.subplots(6, 1, figsize=(14, 24), sharex=True)
    plt.suptitle(f'{method_names[method_idx]} - Position and Force', fontsize=20)

    # 分别绘制每个实验的有效轨迹段数据
    for exp_idx in range(num_experiments):
        if exp_idx >= 5:  # 限制为5组数据
            break

        # 获取位置和力数据
        pos_time, pos_x, pos_y, pos_z, traj_start_idx, traj_end_idx = position_data[method_idx][exp_idx]
        force_time, force_x, force_y, force_z = force_data[method_idx][exp_idx]

        # 截取有效轨迹段
        pos_time_segment = pos_time[traj_start_idx:traj_end_idx]
        pos_x_segment = pos_x[traj_start_idx:traj_end_idx]
        pos_y_segment = pos_y[traj_start_idx:traj_end_idx]
        pos_z_segment = pos_z[traj_start_idx:traj_end_idx]

        # 调整位置时间起点为0
        pos_time_segment_aligned = pos_time_segment - pos_time_segment[0]

        # 将力数据对齐到位置时间序列
        # 找出力数据中时间与轨迹段时间范围重叠的部分
        force_start_idx = np.searchsorted(force_time, pos_time[traj_start_idx], side='left')
        force_end_idx = np.searchsorted(force_time, pos_time[traj_end_idx], side='right')

        if force_start_idx >= force_end_idx:
            print(f"  - Warning: No matching force data for trajectory segment in set {exp_idx + 1}")
            continue

        force_time_segment = force_time[force_start_idx:force_end_idx]
        force_x_segment = force_x[force_start_idx:force_end_idx]
        force_y_segment = force_y[force_start_idx:force_end_idx]
        force_z_segment = force_z[force_start_idx:force_end_idx]

        # 调整力时间起点为0
        force_time_segment_aligned = force_time_segment - force_time_segment[0]

        # 如果力数据和位置数据的时间序列不同，需要对齐
        if len(force_time_segment) != len(pos_time_segment) or not np.allclose(force_time_segment,
                                                                               pos_time_segment):
            # 尝试将力数据对齐到位置时间
            try:
                force_x_aligned = align_time_series(force_time_segment, force_x_segment, pos_time_segment)
                force_y_aligned = align_time_series(force_time_segment, force_y_segment, pos_time_segment)
                force_z_aligned = align_time_series(force_time_segment, force_z_segment, pos_time_segment)

                # 更新力数据段为对齐后的数据
                force_time_segment = pos_time_segment
                force_x_segment = force_x_aligned
                force_y_segment = force_y_aligned
                force_z_segment = force_z_aligned

                # 使用位置时间的对齐版本
                force_time_segment_aligned = pos_time_segment_aligned
            except Exception as e:
                print(f"  - Warning: Failed to align force data for set {exp_idx + 1}: {e}")
                continue

        # 绘制位置数据 - X方向
        axes[0].plot(pos_time_segment_aligned, pos_x_segment,
                     color=colors[exp_idx % len(colors)],
                     linestyle=line_styles[exp_idx % len(line_styles)],
                     linewidth=2,
                     label=f'Set {exp_idx + 1}')

        # 绘制位置数据 - Y方向
        axes[1].plot(pos_time_segment_aligned, pos_y_segment,
                     color=colors[exp_idx % len(colors)],
                     linestyle=line_styles[exp_idx % len(line_styles)],
                     linewidth=2,
                     label=f'Set {exp_idx + 1}')

        # 绘制位置数据 - Z方向
        axes[2].plot(pos_time_segment_aligned, pos_z_segment,
                     color=colors[exp_idx % len(colors)],
                     linestyle=line_styles[exp_idx % len(line_styles)],
                     linewidth=2,
                     label=f'Set {exp_idx + 1}')

        # 绘制力数据 - X方向
        axes[3].plot(force_time_segment_aligned, force_x_segment,
                     color=colors[exp_idx % len(colors)],
                     linestyle=line_styles[exp_idx % len(line_styles)],
                     linewidth=2,
                     label=f'Set {exp_idx + 1}')

        # 绘制力数据 - Y方向
        axes[4].plot(force_time_segment_aligned, force_y_segment,
                     color=colors[exp_idx % len(colors)],
                     linestyle=line_styles[exp_idx % len(line_styles)],
                     linewidth=2,
                     label=f'Set {exp_idx + 1}')

        # 绘制力数据 - Z方向
        axes[5].plot(force_time_segment_aligned, force_z_segment,
                     color=colors[exp_idx % len(colors)],
                     linestyle=line_styles[exp_idx % len(line_styles)],
                     linewidth=2,
                     label=f'Set {exp_idx + 1}')

    # 设置子图标题和标签
    axes[0].set_title('X Position')
    axes[0].set_ylabel('Position (m)')
    axes[0].grid(False)
    axes[0].legend(loc='best')

    axes[1].set_title('Y Position')
    axes[1].set_ylabel('Position (m)')
    axes[1].grid(False)

    axes[2].set_title('Z Position')
    axes[2].set_ylabel('Position (m)')
    axes[2].grid(False)

    axes[3].set_title('X Force')
    axes[3].set_ylabel('Force (N)')
    axes[3].grid(False)

    axes[4].set_title('Y Force')
    axes[4].set_ylabel('Force (N)')
    axes[4].grid(False)

    axes[5].set_title('Z Force')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('Force (N)')
    axes[5].grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为suptitle留出空间
    plt.savefig(os.path.join(base_dir, f'{method_names[method_idx]}_position_force.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)

# 创建方法比较图 - 一张包含所有方法的图，显示力数据
if any(force_data) and any(position_data):
    # 确保每种方法都有数据
    valid_methods = []
    for method_idx in range(4):
        if force_data[method_idx] and position_data[method_idx]:
            valid_methods.append(method_idx)

    if valid_methods:
        fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
        plt.suptitle(f'Method Comparison - Interaction Forces (Valid Trajectory Segment)', fontsize=20)

        for method_idx in valid_methods:
            # 取第一个有效实验的数据
            if not force_data[method_idx] or not position_data[method_idx]:
                continue

            # 获取位置和力数据
            pos_time, pos_x, pos_y, pos_z, traj_start_idx, traj_end_idx = position_data[method_idx][0]
            force_time, force_x, force_y, force_z = force_data[method_idx][0]

            # 截取有效轨迹段
            pos_time_segment = pos_time[traj_start_idx:traj_end_idx]

            # 将力数据对齐到位置时间序列
            force_start_idx = np.searchsorted(force_time, pos_time[traj_start_idx], side='left')
            force_end_idx = np.searchsorted(force_time, pos_time[traj_end_idx], side='right')

            if force_start_idx >= force_end_idx:
                print(
                    f"  - Warning: No matching force data for trajectory segment in method {method_names[method_idx]}")
                continue

            force_time_segment = force_time[force_start_idx:force_end_idx]
            force_x_segment = force_x[force_start_idx:force_end_idx]
            force_y_segment = force_y[force_start_idx:force_end_idx]
            force_z_segment = force_z[force_start_idx:force_end_idx]

            # 如果力数据和位置数据的时间序列不同，需要对齐
            if len(force_time_segment) != len(pos_time_segment) or not np.allclose(force_time_segment,
                                                                                   pos_time_segment):
                # 尝试将力数据对齐到位置时间
                try:
                    force_x_aligned = align_time_series(force_time_segment, force_x_segment, pos_time_segment)
                    force_y_aligned = align_time_series(force_time_segment, force_y_segment, pos_time_segment)
                    force_z_aligned = align_time_series(force_time_segment, force_z_segment, pos_time_segment)

                    # 更新力数据段为对齐后的数据
                    force_time_segment = pos_time_segment
                    force_x_segment = force_x_aligned
                    force_y_segment = force_y_aligned
                    force_z_segment = force_z_aligned
                except Exception as e:
                    print(f"  - Warning: Failed to align force data for method {method_names[method_idx]}: {e}")
                    continue

            # 调整时间起点为0
            adjusted_time = force_time_segment - force_time_segment[0]

            # X方向力比较
            axes[0].plot(adjusted_time, force_x_segment,
                         color=colors[method_idx],
                         linestyle=line_styles[method_idx],
                         linewidth=2,
                         label=f'{method_names[method_idx]}')

            # Y方向力比较
            axes[1].plot(adjusted_time, force_y_segment,
                         color=colors[method_idx],
                         linestyle=line_styles[method_idx],
                         linewidth=2,
                         label=f'{method_names[method_idx]}')

            # Z方向力比较
            axes[2].plot(adjusted_time, force_z_segment,
                         color=colors[method_idx],
                         linestyle=line_styles[method_idx],
                         linewidth=2,
                         label=f'{method_names[method_idx]}')

        axes[0].set_title('X Direction Force')
        axes[0].set_ylabel('Force (N)')
        axes[0].grid(False)
        axes[0].legend(loc='best')

        axes[1].set_title('Y Direction Force')
        axes[1].set_ylabel('Force (N)')
        axes[1].grid(False)

        axes[2].set_title('Z Direction Force')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Force (N)')
        axes[2].grid(False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为suptitle留出空间
        plt.savefig(os.path.join(base_dir, 'method_comparison_valid_segment.png'), dpi=400, bbox_inches='tight')
        plt.close(fig)

# 在所有数据处理完成后调用比较函数
if any(force_data) and any(position_data):
    print("\nComparing force data smoothness and means between methods...")
    valid_methods, force_means, force_stds, smoothness_metrics = compare_force_smoothness_and_means(
        force_data, position_data, method_names)
    print("Comparison completed and visualizations saved.")

print("All plots created successfully!")