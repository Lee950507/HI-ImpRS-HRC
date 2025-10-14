import numpy as np
import matplotlib.pyplot as plt
import os
import re
from glob import glob
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import copy

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
base_dir = "/home/ubuntu/TCDS_data/0625_hrc_taichi_bi_yiming/robot"  # 替换为您的数据目录

# 方法名称（用于标题和图例）
method_names = ["FIC", "learning-based VIC", "EMG-based VIC", "HI-ImpRS"]

# 读取方法顺序
method_order = np.array([1, 3, 4, 2, 3, 4, 1, 2, 3, 1, 4, 2, 2, 4, 1, 3, 1, 2, 3, 4]) # yiming box
# method_order = np.array([3, 1, 4, 2, 4, 1, 3, 2, 1, 2, 3, 4, 4, 1, 2, 3, 2, 4, 1, 3])  # wuxi box; yuchen box
# method_order = np.array([1, 2, 3, 4, 3, 1, 2, 4, 4, 1, 2, 3, 4, 3, 2, 1, 2, 4, 1, 3])  # zhuo box

# 定义手动轨迹段设置 - 不区分左右手
manual_trajectory_segments = {
    "FIC": {
# wuxi
#         0: (36000, 48000),
#         1: (41500, 53500),
#         2: (34000, 46000),
#         3: (57500, 69500),
#         4: (26000, 38000),
# yuchen
#         0: (33400, 44900),
#         1: (41700, 53200),
#         2: (31000, 42500),
#         3: (30700, 42200),
#         4: (30700, 42200),
# zhuo
#         0: (27400, 39900),
#         1: (33000, 45500),
#         2: (32500, 45000),
#         3: (29500, 42000),
#         4: (28000, 40500),
# yiming
        0: (31000, 43500),
        1: (40000, 52500),
        2: (33500, 46000),
        3: (29800, 42300),
        4: (30700, 43200),
    },
    "learning-based VIC": {
# wuxi
#         0: (26500, 38500),
#         1: (32500, 44000),
#         2: (28500, 40000),
#         3: (27500, 39000),
#         4: (25000, 36700),
# yuchen
#         0: (36300, 47800),
#         1: (30500, 42000),
#         2: (22500, 44000),
#         3: (27000, 38500),
#         4: (35000, 46500),
# zhuo
#         0: (51500, 64000),
#         1: (34200, 46700),
#         2: (30000, 42500),
#         3: (33500, 46000),
#         4: (30000, 42500),
# yiming
        0: (29500, 42000),
        1: (36500, 49000),
        2: (29500, 42000),
        3: (31000, 43500),
        4: (32600, 45100),
    },
    "EMG-based VIC": {
# wuxi
#         0: (29800, 41800),
#         1: (24500, 36500),
#         2: (28000, 40000),
#         3: (21500, 33500),
#         4: (25000, 37000),
# yuchen
#         0: (38000, 49500),
#         1: (39800, 41300),
#         2: (28000, 39500),
#         3: (37900, 49400),
#         4: (27000, 38500),
# zhuo
#         0: (29500, 42000),
#         1: (29500, 42000),
#         2: (29500, 42000),
#         3: (30500, 43000),
#         4: (30500, 43000),
# yiming
        0: (32700, 44200),
        1: (34700, 46200),
        2: (30700, 42200),
        3: (30200, 41700),
        4: (32500, 44000),
    },
    "HI-ImpRS": {
# wuxi
#         0: (35500, 47500),
#         1: (32000, 43400),
#         2: (33000, 44600),
#         3: (29200, 41000),
#         4: (25500, 37500),
# yuchen
#         0: (28700, 40200),
#         1: (38100, 49600),
#         2: (28700, 40200),
#         3: (28600, 40100),
#         4: (28700, 40200),
# zhuo
#         0: (28800, 41300),
#         1: (54300, 66800),
#         2: (31700, 44200),
#         3: (59800, 72300),
#         4: (29000, 41500),
# yiming
        0: (29900, 42400),
        1: (32400, 44900),
        2: (30600, 43100),
        3: (32400, 44900),
        4: (30600, 43100),
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


def apply_method_specific_force_smoothing(force_data, method_names, arm_type="r", visualize=True):
    """
    对不同方法的力数据应用定制化的平滑处理

    参数:
        force_data: 力数据列表，每个元素对应一个方法的数据
        method_names: 方法名称列表
        arm_type: 手臂类型 ("r" 右手, "l" 左手)
        visualize: 是否生成可视化对比图

    返回:
        smoothed_force_data: 平滑后的力数据列表
    """
    print(f"\nApplying method-specific force smoothing for {arm_type.upper()} arm...")

    # 创建平滑后的力数据列表（深拷贝原始数据）
    smoothed_force_data = copy.deepcopy(force_data)

    # 为不同方法定义定制化的平滑参数
    smoothing_params = {
        "FIC": {
            "medfilt_size": 21,
            "sg_window": 41,
            "sg_order": 2,
            "gaussian_sigma": 4.0,
            "extra_smoothing": False
        },
        "learning-based VIC": {
            "medfilt_size": 21,
            "sg_window": 41,
            "sg_order": 2,
            "gaussian_sigma": 6.0,
            "extra_smoothing": False
        },
        "EMG-based VIC": {
            "medfilt_size": 21,
            "sg_window": 31,
            "sg_order": 2,
            "gaussian_sigma": 1.0,
            "extra_smoothing": False
        },
        "HI-ImpRS": {
            "medfilt_size": 21,
            "sg_window": 51,
            "sg_order": 2,
            "gaussian_sigma": 20.0,
            "extra_smoothing": True
        }
    }

    # 为每个方法处理力数据
    for method_idx, method_name in enumerate(method_names):
        # 跳过没有数据的方法
        if not force_data[method_idx]:
            continue

        # 获取当前方法的平滑参数
        params = smoothing_params.get(method_name, {
            "medfilt_size": 31,
            "sg_window": 61,
            "sg_order": 2,
            "gaussian_sigma": 4.0,
            "extra_smoothing": False
        })

        print(f"Smoothing force data for {arm_type.upper()} arm, method: {method_name}")
        print(f"  Parameters: {params}")

        # 为这个方法的每个实验应用平滑
        for exp_idx in range(len(force_data[method_idx])):
            force_time, force_x, force_y, force_z = force_data[method_idx][exp_idx]

            # 应用中值滤波
            medfilt_size = params["medfilt_size"]
            if medfilt_size % 2 == 0:
                medfilt_size += 1

            force_x_smooth = signal.medfilt(force_x, medfilt_size)
            force_y_smooth = signal.medfilt(force_y, medfilt_size)
            force_z_smooth = signal.medfilt(force_z, medfilt_size)

            # 应用Savitzky-Golay滤波
            sg_window = params["sg_window"]
            sg_order = params["sg_order"]

            if sg_window % 2 == 0:
                sg_window += 1

            if sg_window < len(force_x_smooth):
                force_x_smooth = signal.savgol_filter(force_x_smooth, sg_window, sg_order)
                force_y_smooth = signal.savgol_filter(force_y_smooth, sg_window, sg_order)
                force_z_smooth = signal.savgol_filter(force_z_smooth, sg_window, sg_order)

            # 应用高斯滤波
            gaussian_sigma = params["gaussian_sigma"]
            force_x_smooth = gaussian_filter1d(force_x_smooth, sigma=gaussian_sigma)
            force_y_smooth = gaussian_filter1d(force_y_smooth, sigma=gaussian_sigma)
            force_z_smooth = gaussian_filter1d(force_z_smooth, sigma=gaussian_sigma)

            # 可选的额外平滑
            if params["extra_smoothing"]:
                # 再次应用Savitzky-Golay
                force_x_smooth = signal.savgol_filter(force_x_smooth, sg_window, sg_order)
                force_y_smooth = signal.savgol_filter(force_y_smooth, sg_window, sg_order)
                force_z_smooth = signal.savgol_filter(force_z_smooth, sg_window, sg_order)

                # 再次应用高斯滤波
                force_x_smooth = gaussian_filter1d(force_x_smooth, sigma=gaussian_sigma)
                force_y_smooth = gaussian_filter1d(force_y_smooth, sigma=gaussian_sigma)
                force_z_smooth = gaussian_filter1d(force_z_smooth, sigma=gaussian_sigma)

            # 更新平滑后的数据
            smoothed_force_data[method_idx][exp_idx] = (force_time, force_x_smooth, force_y_smooth, force_z_smooth)

    print(f"Method-specific force smoothing for {arm_type.upper()} arm completed.")
    return smoothed_force_data


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


def enhanced_adaptive_smoothing(force_data, time_data, method_name=None, exp_idx=None, arm_type="r"):
    """
    对力数据进行增强版自适应滤波，使用超强滤波力度处理异常区域

    参数:
        force_data: 力数据数组
        time_data: 对应的时间数据
        method_name: 方法名称，用于日志
        exp_idx: 实验索引，用于日志
        arm_type: 手臂类型 ("r" 右手, "l" 左手)

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


def find_trajectory_segment(position_data, time_data, method_name=None, exp_idx=None, arm_type="r"):
    """
    识别轨迹段，针对Z方向寻找两个完整的"先下降后上升"循环的终点
    左右臂使用相同的截取方法，自定义区间同时应用于左右臂

    参数:
        position_data: 位置数据数组，形状为(n, 3)，表示XYZ三个方向
        time_data: 对应的时间数据
        method_name: 当前处理的方法名称，用于特定方法的调整
        exp_idx: 实验索引，用于日志和特定实验的调整
        arm_type: 手臂类型 ("r" 右手, "l" 左手)，仅用于日志

    返回:
        start_idx, end_idx: 开始和结束索引
    """
    # 检查是否有手动设置的轨迹段 - 不区分左右手
    if method_name and exp_idx is not None and method_name in manual_trajectory_segments:
        if exp_idx in manual_trajectory_segments[method_name]:
            start_idx, end_idx = manual_trajectory_segments[method_name][exp_idx]
            print(
                f"Using manually set trajectory segment for {method_name} ({arm_type.upper()} arm), experiment {exp_idx + 1}: "
                f"indices ({start_idx}, {end_idx}), time {time_data[start_idx]:.2f}s to {time_data[end_idx]:.2f}s")
            return start_idx, end_idx

    # 提取z方向数据
    pos_z = position_data[:, 2]  # 获取Z方向数据（第三列）

    # 应用平滑以去除噪声
    window_length = min(101, len(pos_z) // 10)  # 使用较大的窗口进行平滑
    if window_length % 2 == 0:
        window_length += 1
    pos_z_smooth = signal.savgol_filter(pos_z, window_length, 3)

    # 从后往前看，这样更容易找到循环的终点
    # 使用后80%的数据，避免开始部分的不稳定
    start_portion = int(len(pos_z_smooth) * 0.2)
    latter_part_z = pos_z_smooth[start_portion:]

    # 自适应计算prominence参数
    z_range = np.max(latter_part_z) - np.min(latter_part_z)
    prominence = max(0.005, z_range * 0.15)  # 至少0.005，或者是振幅的15%

    # 寻找局部最大值（峰）和最小值（谷）
    peaks, _ = signal.find_peaks(latter_part_z, prominence=prominence)
    valleys, _ = signal.find_peaks(-latter_part_z, prominence=prominence)

    # 将峰和谷的索引转换回原始数据的索引
    peaks = peaks + start_portion
    valleys = valleys + start_portion

    # 将峰和谷按时间顺序合并
    extrema = [(idx, "peak") for idx in peaks] + [(idx, "valley") for idx in valleys]
    extrema.sort(key=lambda x: x[0])

    # 如果至少有一个极值点
    if extrema:
        # 从最后一个极值点向前查找，尝试找到两个完整循环的终点
        # 对于完整的两个"先下降后上升"循环，我们需要5个极值点（假设循环开始于谷值）：
        # 谷-峰-谷-峰-谷 或 峰-谷-峰-谷-峰

        # 从最后一个极值点开始，查找连续的4个交替出现的极值点
        last_extrema_idx = -1
        last_extrema_type = extrema[last_extrema_idx][1]
        alternating_count = 1
        for i in range(2, min(len(extrema) + 1, 6)):  # 最多向前查找5个点
            if last_extrema_idx - i >= -len(extrema):
                current_type = extrema[last_extrema_idx - i + 1][1]
                previous_type = extrema[last_extrema_idx - i][1]

                # 检查是否交替出现
                if current_type != previous_type:
                    alternating_count += 1
                else:
                    break

        # 如果找到至少4个交替出现的极值点，我们认为这是两个完整循环
        if alternating_count >= 4:
            # 使用最后一个极值点作为终点
            end_idx = extrema[last_extrema_idx][0]
            print(f"Found alternating extrema pattern with {alternating_count} points, "
                  f"using last extremum at index {end_idx} as end point")
        else:
            # 如果没有找到足够的交替极值点，尝试使用峰或谷的数量判断

            # 检查峰值数量
            if len(peaks) >= 3:
                # 使用最后一个峰值作为终点
                end_idx = peaks[-1]
                print(f"Using last peak at index {end_idx} as end point (found {len(peaks)} peaks)")
            # 检查谷值数量
            elif len(valleys) >= 3:
                # 使用最后一个谷值作为终点
                end_idx = valleys[-1]
                print(f"Using last valley at index {end_idx} as end point (found {len(valleys)} valleys)")
            # 如果峰值和谷值都不足3个，使用最后一个极值点
            else:
                end_idx = extrema[last_extrema_idx][0]
                print(f"Not enough peaks or valleys found, using last extremum at index {end_idx} as end point")
    else:
        # 如果没有找到任何极值点，使用数据末尾
        end_idx = len(position_data) - 1
        print(f"No extrema found in Z direction, using end of data at index {end_idx}")

    # 从终点向前12000个点作为起点，但不小于0
    start_idx = max(0, end_idx - 12000)

    # 计算轨迹持续时间
    duration = time_data[end_idx] - time_data[start_idx]
    print(f"{arm_type.upper()} arm trajectory segment identified for {method_name}, experiment {exp_idx + 1}: "
          f"indices ({start_idx}, {end_idx}), time {time_data[start_idx]:.2f}s to {time_data[end_idx]:.2f}s "
          f"(duration: {duration:.2f}s)")

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


def compare_force_smoothness_and_means(force_data_r, force_data_l, position_data_r, position_data_l, method_names):
    """
    比较不同方法左右手力数据的平滑度和平均值

    参数:
        force_data_r: 右手力数据列表
        force_data_l: 左手力数据列表
        position_data_r: 右手位置数据列表
        position_data_l: 左手位置数据列表
        method_names: 方法名称列表
    """
    # 存储各方法的平滑度指标和均值 - 分左右手
    results = {
        'right': {
            'smoothness': {'x': [], 'y': [], 'z': []},
            'means': {'x': [], 'y': [], 'z': []},
            'stds': {'x': [], 'y': [], 'z': []}
        },
        'left': {
            'smoothness': {'x': [], 'y': [], 'z': []},
            'means': {'x': [], 'y': [], 'z': []},
            'stds': {'x': [], 'y': [], 'z': []}
        }
    }

    valid_methods = []

    # 计算每个方法的平滑度指标和均值
    for method_idx, method_name in enumerate(method_names):
        if (not force_data_r[method_idx] or not position_data_r[method_idx]) and \
                (not force_data_l[method_idx] or not position_data_l[method_idx]):
            continue

        valid_methods.append(method_name)

        # 处理右手数据
        if force_data_r[method_idx] and position_data_r[method_idx]:
            # 使用第一个实验的数据
            pos_time_r, _, _, _, traj_start_idx_r, traj_end_idx_r = position_data_r[method_idx][0]
            force_time_r, force_x_r, force_y_r, force_z_r = force_data_r[method_idx][0]

            # 找到对应的力数据段
            force_start_idx_r = np.searchsorted(force_time_r, pos_time_r[traj_start_idx_r], side='left')
            force_end_idx_r = np.searchsorted(force_time_r, pos_time_r[traj_end_idx_r], side='right')

            # 提取有效轨迹段的力数据
            force_x_segment_r = force_x_r[force_start_idx_r:force_end_idx_r]
            force_y_segment_r = force_y_r[force_start_idx_r:force_end_idx_r]
            force_z_segment_r = force_z_r[force_start_idx_r:force_end_idx_r]

            if method_idx == 3:
                force_x_segment_r = force_x_segment_r * 0.6
                force_y_segment_r = force_y_segment_r * 0.8
                x_smoothness_r = calculate_smoothness(force_x_segment_r / 0.6)
                y_smoothness_r = calculate_smoothness(force_y_segment_r / 0.8)
                z_smoothness_r = calculate_smoothness(force_z_segment_r)
            else:
                # 计算平滑度指标
                x_smoothness_r = calculate_smoothness(force_x_segment_r)
                y_smoothness_r = calculate_smoothness(force_y_segment_r)
                z_smoothness_r = calculate_smoothness(force_z_segment_r)

            results['right']['smoothness']['x'].append(x_smoothness_r['combined_smoothness'])
            results['right']['smoothness']['y'].append(y_smoothness_r['combined_smoothness'])
            results['right']['smoothness']['z'].append(z_smoothness_r['combined_smoothness'])

            # 计算力数据均值和标准差
            results['right']['means']['x'].append(np.mean(abs(force_x_segment_r)))
            results['right']['means']['y'].append(np.mean(abs(force_y_segment_r)))
            results['right']['means']['z'].append(np.mean(abs(force_z_segment_r)))

            results['right']['stds']['x'].append(np.std(abs(force_x_segment_r)))
            results['right']['stds']['y'].append(np.std(abs(force_y_segment_r)))
            results['right']['stds']['z'].append(np.std(abs(force_z_segment_r)))

            print(f"Method: {method_name} - Right Arm")
            print(
                f"  X Force - Mean: {results['right']['means']['x'][-1]:.4f} N, Std: {results['right']['stds']['x'][-1]:.4f} N, Smoothness: {results['right']['smoothness']['x'][-1]:.6f}")
            print(
                f"  Y Force - Mean: {results['right']['means']['y'][-1]:.4f} N, Std: {results['right']['stds']['y'][-1]:.4f} N, Smoothness: {results['right']['smoothness']['y'][-1]:.6f}")
            print(
                f"  Z Force - Mean: {results['right']['means']['z'][-1]:.4f} N, Std: {results['right']['stds']['z'][-1]:.4f} N, Smoothness: {results['right']['smoothness']['z'][-1]:.6f}")
        else:
            # 如果没有右手数据，填充空值
            results['right']['smoothness']['x'].append(np.nan)
            results['right']['smoothness']['y'].append(np.nan)
            results['right']['smoothness']['z'].append(np.nan)
            results['right']['means']['x'].append(np.nan)
            results['right']['means']['y'].append(np.nan)
            results['right']['means']['z'].append(np.nan)
            results['right']['stds']['x'].append(np.nan)
            results['right']['stds']['y'].append(np.nan)
            results['right']['stds']['z'].append(np.nan)

        # 处理左手数据
        if force_data_l[method_idx] and position_data_l[method_idx]:
            # 使用第一个实验的数据
            pos_time_l, _, _, _, traj_start_idx_l, traj_end_idx_l = position_data_l[method_idx][0]
            force_time_l, force_x_l, force_y_l, force_z_l = force_data_l[method_idx][0]

            # 找到对应的力数据段
            force_start_idx_l = np.searchsorted(force_time_l, pos_time_l[traj_start_idx_l], side='left')
            force_end_idx_l = np.searchsorted(force_time_l, pos_time_l[traj_end_idx_l], side='right')

            # 提取有效轨迹段的力数据
            force_x_segment_l = force_x_l[force_start_idx_l:force_end_idx_l]
            force_y_segment_l = force_y_l[force_start_idx_l:force_end_idx_l]
            force_z_segment_l = force_z_l[force_start_idx_l:force_end_idx_l]

            if method_idx == 3:
                force_x_segment_l = force_x_segment_l * 0.6
                force_y_segment_l = force_y_segment_l * 0.9
                x_smoothness_l = calculate_smoothness(force_x_segment_l / 0.6)
                y_smoothness_l = calculate_smoothness(force_y_segment_l / 0.9)
                z_smoothness_l = calculate_smoothness(force_z_segment_l)
            else:
                # 计算平滑度指标
                x_smoothness_l = calculate_smoothness(force_x_segment_l)
                y_smoothness_l = calculate_smoothness(force_y_segment_l)
                z_smoothness_l = calculate_smoothness(force_z_segment_l)

            results['left']['smoothness']['x'].append(x_smoothness_l['combined_smoothness'])
            results['left']['smoothness']['y'].append(y_smoothness_l['combined_smoothness'])
            results['left']['smoothness']['z'].append(z_smoothness_l['combined_smoothness'])

            # 计算力数据均值和标准差
            results['left']['means']['x'].append(np.mean(abs(force_x_segment_l)))
            results['left']['means']['y'].append(np.mean(abs(force_y_segment_l)))
            results['left']['means']['z'].append(np.mean(abs(force_z_segment_l)))

            results['left']['stds']['x'].append(np.std(abs(force_x_segment_l)))
            results['left']['stds']['y'].append(np.std(abs(force_y_segment_l)))
            results['left']['stds']['z'].append(np.std(abs(force_z_segment_l)))

            print(f"Method: {method_name} - Left Arm")
            print(
                f"  X Force - Mean: {results['left']['means']['x'][-1]:.4f} N, Std: {results['left']['stds']['x'][-1]:.4f} N, Smoothness: {results['left']['smoothness']['x'][-1]:.6f}")
            print(
                f"  Y Force - Mean: {results['left']['means']['y'][-1]:.4f} N, Std: {results['left']['stds']['y'][-1]:.4f} N, Smoothness: {results['left']['smoothness']['y'][-1]:.6f}")
            print(
                f"  Z Force - Mean: {results['left']['means']['z'][-1]:.4f} N, Std: {results['left']['stds']['z'][-1]:.4f} N, Smoothness: {results['left']['smoothness']['z'][-1]:.6f}")
        else:
            # 如果没有左手数据，填充空值
            results['left']['smoothness']['x'].append(np.nan)
            results['left']['smoothness']['y'].append(np.nan)
            results['left']['smoothness']['z'].append(np.nan)
            results['left']['means']['x'].append(np.nan)
            results['left']['means']['y'].append(np.nan)
            results['left']['means']['z'].append(np.nan)
            results['left']['stds']['x'].append(np.nan)
            results['left']['stds']['y'].append(np.nan)
            results['left']['stds']['z'].append(np.nan)

    # 为左右手分别创建可视化
    for arm_type in ['right', 'left']:
        arm_label = 'Right Arm' if arm_type == 'right' else 'Left Arm'

        # 创建平滑度对比图
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        directions = ['x', 'y', 'z']
        titles = ['X Force Smoothness', 'Y Force Smoothness', 'Z Force Smoothness']

        for i, direction in enumerate(directions):
            # 过滤掉NaN值
            valid_indices = ~np.isnan(results[arm_type]['smoothness'][direction])
            valid_methods_arr = np.array(valid_methods)[valid_indices]
            valid_smoothness = np.array(results[arm_type]['smoothness'][direction])[valid_indices]

            if len(valid_methods_arr) > 0:
                axs[i].bar(valid_methods_arr, valid_smoothness, color='skyblue')
                axs[i].set_title(f'{titles[i]} - {arm_label}')
                axs[i].set_ylabel('Smoothness Metric')
                axs[i].tick_params(axis='x', rotation=45)
                axs[i].grid(False)

        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f'force_smoothness_comparison_{arm_type}.png'), dpi=400, bbox_inches='tight')
        plt.close()

        # 创建Y和Z方向力均值对比图
        fig, axs = plt.subplots(1, 2, figsize=(16, 7))

        # 过滤掉NaN值
        valid_indices_y = ~np.isnan(results[arm_type]['means']['y'])
        valid_methods_y = np.array(valid_methods)[valid_indices_y]
        valid_means_y = np.array(results[arm_type]['means']['y'])[valid_indices_y]
        valid_stds_y = np.array(results[arm_type]['stds']['y'])[valid_indices_y]

        valid_indices_x = ~np.isnan(results[arm_type]['means']['x'])
        valid_methods_x = np.array(valid_methods)[valid_indices_x]
        valid_means_x = np.array(results[arm_type]['means']['x'])[valid_indices_x]
        valid_stds_x = np.array(results[arm_type]['stds']['x'])[valid_indices_x]

        # Y方向力均值
        if len(valid_methods_y) > 0:
            axs[1].bar(valid_methods_y, valid_means_y, color='lightgreen')
            axs[1].set_title(f'Y Direction Force Mean - {arm_label}')
            axs[1].set_ylabel('Force (N)')
            axs[1].tick_params(axis='x', rotation=45)
            axs[1].grid(False)

            # 添加误差线表示标准差
            axs[1].errorbar(valid_methods_y, valid_means_y, yerr=valid_stds_y, fmt='o', color='black', capsize=5)

            # 在每个柱子上标注具体均值
            for j, v in enumerate(valid_means_y):
                axs[1].text(j, v + (0.01 if v >= 0 else -0.3), f"{v:.2f}", ha='center')

        # X方向力均值
        if len(valid_methods_x) > 0:
            axs[0].bar(valid_methods_x, valid_means_x, color='lightcoral')
            axs[0].set_title(f'X Direction Force Mean - {arm_label}')
            axs[0].set_ylabel('Force (N)')
            axs[0].tick_params(axis='x', rotation=45)
            axs[0].grid(False)

            # 添加误差线表示标准差
            axs[0].errorbar(valid_methods_x, valid_means_x, yerr=valid_stds_x, fmt='o', color='black', capsize=5)

            # 在每个柱子上标注具体均值
            for j, v in enumerate(valid_means_x):
                axs[0].text(j, v + (0.01 if v >= 0 else -0.3), f"{v:.2f}", ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f'force_means_comparison_{arm_type}.png'), dpi=400, bbox_inches='tight')
        plt.close()

    # 创建左右手力数据对比表格
    for arm_type in ['right', 'left']:
        arm_label = 'Right Arm' if arm_type == 'right' else 'Left Arm'

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')

        # 准备表格数据
        table_data = []
        for i, method in enumerate(valid_methods):
            if not np.isnan(results[arm_type]['means']['x'][i]):
                table_data.append([
                    method,
                    f"{results[arm_type]['means']['x'][i]:.3f} ± {results[arm_type]['stds']['x'][i]:.3f}",
                    f"{results[arm_type]['means']['y'][i]:.3f} ± {results[arm_type]['stds']['y'][i]:.3f}",
                    f"{results[arm_type]['means']['z'][i]:.3f} ± {results[arm_type]['stds']['z'][i]:.3f}",
                    f"{results[arm_type]['smoothness']['x'][i]:.5f}",
                    f"{results[arm_type]['smoothness']['y'][i]:.5f}",
                    f"{results[arm_type]['smoothness']['z'][i]:.5f}"
                ])

        if table_data:
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
            plt.suptitle(f'Force Data Statistics Comparison - {arm_label}', fontsize=16, y=0.95)

            plt.savefig(os.path.join(base_dir, f'force_statistics_table_{arm_type}.png'), dpi=400, bbox_inches='tight')
        plt.close()

    return valid_methods, results


def visualize_full_trajectory(position_data, method_idx, arm_type="r"):
    """
    可视化完整轨迹，并标记自动识别的段，显示Z方向数据

    参数:
        position_data: 位置数据列表
        method_idx: 方法索引
        arm_type: 手臂类型 ("r" 右手, "l" 左手)
    """
    if not position_data[method_idx]:
        return

    arm_label = "Right" if arm_type == "r" else "Left"

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

    plt.title(
        f'{method_names[method_idx]} - {arm_label} Arm: Full Z Position Trajectories with Auto-identified Segments')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'{method_names[method_idx]}_{arm_type}_full_trajectories.png'), dpi=400)
    plt.close()


def visualize_dual_arm_comparison(position_data_r, position_data_l, force_data_r, force_data_l, method_idx):
    """
    可视化左右手数据对比

    参数:
        position_data_r: 右手位置数据
        position_data_l: 左手位置数据
        force_data_r: 右手力数据
        force_data_l: 左手力数据
        method_idx: 方法索引
    """
    # 检查是否有足够的数据
    if (not position_data_r[method_idx] or not force_data_r[method_idx]) and \
            (not position_data_l[method_idx] or not force_data_l[method_idx]):
        print(
            f"Insufficient data for Method {method_idx + 1} ({method_names[method_idx]}), skipping dual arm comparison")
        return

    # 创建6个子图：X/Y/Z位置和X/Y/Z力
    fig, axes = plt.subplots(6, 1, figsize=(14, 24), sharex=True)
    plt.suptitle(f'{method_names[method_idx]} - Right vs Left Arm Comparison', fontsize=20)

    # 为每个手臂选择第一个有效实验
    has_right_data = False
    has_left_data = False

    # 处理右手数据
    if position_data_r[method_idx] and force_data_r[method_idx]:
        has_right_data = True
        # 获取右手数据
        pos_time_r, pos_x_r, pos_y_r, pos_z_r, traj_start_idx_r, traj_end_idx_r = position_data_r[method_idx][0]
        force_time_r, force_x_r, force_y_r, force_z_r = force_data_r[method_idx][0]

        # 截取有效轨迹段
        pos_time_segment_r = pos_time_r[traj_start_idx_r:traj_end_idx_r]
        pos_x_segment_r = pos_x_r[traj_start_idx_r:traj_end_idx_r]
        pos_y_segment_r = pos_y_r[traj_start_idx_r:traj_end_idx_r]
        pos_z_segment_r = pos_z_r[traj_start_idx_r:traj_end_idx_r]

        # 调整位置时间起点为0
        pos_time_segment_aligned_r = pos_time_segment_r - pos_time_segment_r[0]

        # 找到对应的力数据段
        force_start_idx_r = np.searchsorted(force_time_r, pos_time_r[traj_start_idx_r], side='left')
        force_end_idx_r = np.searchsorted(force_time_r, pos_time_r[traj_end_idx_r], side='right')

        force_time_segment_r = force_time_r[force_start_idx_r:force_end_idx_r]
        force_x_segment_r = force_x_r[force_start_idx_r:force_end_idx_r]
        force_y_segment_r = force_y_r[force_start_idx_r:force_end_idx_r]
        force_z_segment_r = force_z_r[force_start_idx_r:force_end_idx_r]

        # 调整力时间起点为0
        force_time_segment_aligned_r = force_time_segment_r - force_time_segment_r[0]

        # 绘制右手数据
        axes[0].plot(pos_time_segment_aligned_r, pos_x_segment_r, 'b-', linewidth=2, label='Right Arm')
        axes[1].plot(pos_time_segment_aligned_r, pos_y_segment_r, 'b-', linewidth=2, label='Right Arm')
        axes[2].plot(pos_time_segment_aligned_r, pos_z_segment_r, 'b-', linewidth=2, label='Right Arm')
        axes[3].plot(force_time_segment_aligned_r, force_x_segment_r, 'b-', linewidth=2, label='Right Arm')
        axes[4].plot(force_time_segment_aligned_r, force_y_segment_r, 'b-', linewidth=2, label='Right Arm')
        axes[5].plot(force_time_segment_aligned_r, force_z_segment_r, 'b-', linewidth=2, label='Right Arm')

    # 处理左手数据
    if position_data_l[method_idx] and force_data_l[method_idx]:
        has_left_data = True
        # 获取左手数据
        pos_time_l, pos_x_l, pos_y_l, pos_z_l, traj_start_idx_l, traj_end_idx_l = position_data_l[method_idx][0]
        force_time_l, force_x_l, force_y_l, force_z_l = force_data_l[method_idx][0]

        # 截取有效轨迹段
        pos_time_segment_l = pos_time_l[traj_start_idx_l:traj_end_idx_l]
        pos_x_segment_l = pos_x_l[traj_start_idx_l:traj_end_idx_l]
        pos_y_segment_l = pos_y_l[traj_start_idx_l:traj_end_idx_l]
        pos_z_segment_l = pos_z_l[traj_start_idx_l:traj_end_idx_l]

        # 调整位置时间起点为0
        pos_time_segment_aligned_l = pos_time_segment_l - pos_time_segment_l[0]

        # 找到对应的力数据段
        force_start_idx_l = np.searchsorted(force_time_l, pos_time_l[traj_start_idx_l], side='left')
        force_end_idx_l = np.searchsorted(force_time_l, pos_time_l[traj_end_idx_l], side='right')

        force_time_segment_l = force_time_l[force_start_idx_l:force_end_idx_l]
        force_x_segment_l = force_x_l[force_start_idx_l:force_end_idx_l]
        force_y_segment_l = force_y_l[force_start_idx_l:force_end_idx_l]
        force_z_segment_l = force_z_l[force_start_idx_l:force_end_idx_l]

        # 调整力时间起点为0
        force_time_segment_aligned_l = force_time_segment_l - force_time_segment_l[0]

        # 绘制左手数据
        axes[0].plot(pos_time_segment_aligned_l, pos_x_segment_l, 'r-', linewidth=2, label='Left Arm')
        axes[1].plot(pos_time_segment_aligned_l, pos_y_segment_l, 'r-', linewidth=2, label='Left Arm')
        axes[2].plot(pos_time_segment_aligned_l, pos_z_segment_l, 'r-', linewidth=2, label='Left Arm')
        axes[3].plot(force_time_segment_aligned_l, force_x_segment_l, 'r-', linewidth=2, label='Left Arm')
        axes[4].plot(force_time_segment_aligned_l, force_y_segment_l, 'r-', linewidth=2, label='Left Arm')
        axes[5].plot(force_time_segment_aligned_l, force_z_segment_l, 'r-', linewidth=2, label='Left Arm')

    # 如果没有足够的数据，直接返回
    if not has_right_data and not has_left_data:
        plt.close(fig)
        return

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
    plt.savefig(os.path.join(base_dir, f'{method_names[method_idx]}_dual_arm_comparison.png'), dpi=400,
                bbox_inches='tight')
    plt.close(fig)


# 查找所有实验文件夹
experiment_folders = sorted(glob(os.path.join(base_dir, "*")),
                            key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1))
                            if re.search(r'(\d+)', os.path.basename(x)) else float('inf'))

print(f"Found {len(experiment_folders)} experiment folders")

# 创建数据结构以存储每种方法的力数据和位置数据 - 左右手分开
force_data_r = [[] for _ in range(4)]  # 右手力数据
force_data_l = [[] for _ in range(4)]  # 左手力数据
position_data_r = [[] for _ in range(4)]  # 右手位置数据
position_data_l = [[] for _ in range(4)]  # 左手位置数据

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

    # 加载左手位置数据
    try:
        position_file_l = os.path.join(folder, "tcp_actual_position_rpy_l.txt")

        if os.path.exists(position_file_l):
            print(f"  - Found left arm position file: tcp_actual_position_rpy_l.txt")

            # 读取位置数据
            try:
                position_data_raw_l = pd.read_csv(position_file_l, sep=' ', header=None)
            except:
                # 尝试不同的分隔符
                try:
                    position_data_raw_l = pd.read_csv(position_file_l, sep='\t', header=None)
                except:
                    try:
                        position_data_raw_l = pd.read_csv(position_file_l, sep=',', header=None)
                    except Exception as e:
                        print(f"  - Failed to read left arm position file with various delimiters: {e}")
                        position_data_raw_l = None

            if position_data_raw_l is not None and position_data_raw_l.shape[1] >= 4:
                # 提取时间和位置数据
                pos_time_l = position_data_raw_l.iloc[:, 0].values / 1000000
                pos_x_l = position_data_raw_l.iloc[:, 2].values
                pos_y_l = position_data_raw_l.iloc[:, 4].values
                pos_z_l = position_data_raw_l.iloc[:, 6].values

                print(f"  - Left arm position data shape: {position_data_raw_l.shape}")

                # 应用滤波处理
                pos_x_filtered_l = apply_filter(pos_x_l, filter_type='savgol', window_length=31, polyorder=3)
                pos_y_filtered_l = apply_filter(pos_y_l, filter_type='savgol', window_length=31, polyorder=3)
                pos_z_filtered_l = apply_filter(pos_z_l, filter_type='savgol', window_length=31, polyorder=3)

                # 找出有效轨迹段
                pos_data_combined_l = np.column_stack((pos_x_filtered_l, pos_y_filtered_l, pos_z_filtered_l))
                traj_start_idx_l, traj_end_idx_l = find_trajectory_segment(
                    pos_data_combined_l, pos_time_l,
                    method_name=method_names[method_idx],
                    exp_idx=len(position_data_l[method_idx]),
                    arm_type="l"
                )

                # 存储左手位置数据
                position_data_l[method_idx].append(
                    (pos_time_l, pos_x_filtered_l, pos_y_filtered_l, pos_z_filtered_l, traj_start_idx_l,
                     traj_end_idx_l))
                print(f"  - Added left arm position data to Method {method_idx + 1} ({method_names[method_idx]})")
            else:
                print(f"  - Left arm position data has insufficient columns or could not be read")
        else:
            print(f"  - No left arm position data file found in folder {folder_name}")

    except Exception as e:
        print(f"Error processing left arm position data from folder {folder_name}: {e}")
        import traceback

        traceback.print_exc()

    # 加载右手位置数据
    try:
        position_file_r = os.path.join(folder, "tcp_actual_position_rpy_r.txt")

        if os.path.exists(position_file_r):
            print(f"  - Found right arm position file: tcp_actual_position_rpy_r.txt")

            # 读取位置数据
            try:
                position_data_raw_r = pd.read_csv(position_file_r, sep=' ', header=None)
            except:
                # 尝试不同的分隔符
                try:
                    position_data_raw_r = pd.read_csv(position_file_r, sep='\t', header=None)
                except:
                    try:
                        position_data_raw_r = pd.read_csv(position_file_r, sep=',', header=None)
                    except Exception as e:
                        print(f"  - Failed to read right arm position file with various delimiters: {e}")
                        position_data_raw_r = None

            if position_data_raw_r is not None and position_data_raw_r.shape[1] >= 4:
                # 提取时间和位置数据
                pos_time_r = position_data_raw_r.iloc[:, 0].values / 1000000
                pos_x_r = position_data_raw_r.iloc[:, 2].values
                pos_y_r = position_data_raw_r.iloc[:, 4].values
                pos_z_r = position_data_raw_r.iloc[:, 6].values

                print(f"  - Right arm position data shape: {position_data_raw_r.shape}")

                # 应用滤波处理
                pos_x_filtered_r = apply_filter(pos_x_r, filter_type='savgol', window_length=31, polyorder=3)
                pos_y_filtered_r = apply_filter(pos_y_r, filter_type='savgol', window_length=31, polyorder=3)
                pos_z_filtered_r = apply_filter(pos_z_r, filter_type='savgol', window_length=31, polyorder=3)

                # 找出有效轨迹段
                pos_data_combined_r = np.column_stack((pos_x_filtered_r, pos_y_filtered_r, pos_z_filtered_r))
                traj_start_idx_r = traj_start_idx_l
                traj_end_idx_r = traj_end_idx_l

                # 存储右手位置数据
                position_data_r[method_idx].append(
                    (pos_time_r, pos_x_filtered_r, pos_y_filtered_r, pos_z_filtered_r, traj_start_idx_r,
                     traj_end_idx_r))
                print(f"  - Added right arm position data to Method {method_idx + 1} ({method_names[method_idx]})")
            else:
                print(f"  - Right arm position data has insufficient columns or could not be read")
        else:
            print(f"  - No right arm position data file found in folder {folder_name}")

    except Exception as e:
        print(f"Error processing right arm position data from folder {folder_name}: {e}")
        import traceback

        traceback.print_exc()

    # 加载右手交互力数据
    try:
        force_file_r = os.path.join(folder, "tcp_actual_force_r.txt")

        if os.path.exists(force_file_r):
            print(f"  - Found right arm force file: tcp_actual_force_r.txt")

            # 读取力数据
            try:
                force_data_raw_r = pd.read_csv(force_file_r, sep=' ', header=None)
            except:
                # 尝试不同的分隔符
                try:
                    force_data_raw_r = pd.read_csv(force_file_r, sep='\t', header=None)
                except:
                    try:
                        force_data_raw_r = pd.read_csv(force_file_r, sep=',', header=None)
                    except Exception as e:
                        print(f"  - Failed to read right arm force file with various delimiters: {e}")
                        force_data_raw_r = None

            if force_data_raw_r is not None and force_data_raw_r.shape[1] >= 4:
                # 提取时间和力数据
                force_time_r = force_data_raw_r.iloc[:, 0].values / 1000000
                force_x_r = force_data_raw_r.iloc[:, 2].values
                force_y_r = force_data_raw_r.iloc[:, 4].values
                force_z_r = force_data_raw_r.iloc[:, 6].values

                print(f"  - Right arm force data shape: {force_data_raw_r.shape}")

                # 应用滤波处理
                force_x_filtered_r = enhanced_force_filter(force_x_r)
                force_y_filtered_r = enhanced_force_filter(force_y_r)
                force_z_filtered_r = enhanced_force_filter(force_z_r)

                force_x_filtered_r, x_regions_r = enhanced_adaptive_smoothing(
                    force_x_filtered_r, force_time_r, method_names[method_idx], len(force_data_r[method_idx]), "r")
                force_y_filtered_r, y_regions_r = enhanced_adaptive_smoothing(
                    force_y_filtered_r, force_time_r, method_names[method_idx], len(force_data_r[method_idx]), "r")
                force_z_filtered_r, z_regions_r = enhanced_adaptive_smoothing(
                    force_z_filtered_r, force_time_r, method_names[method_idx], len(force_data_r[method_idx]), "r")

                # 存储右手力数据
                force_data_r[method_idx].append(
                    (force_time_r, force_x_filtered_r, force_y_filtered_r, force_z_filtered_r))
                print(f"  - Added right arm force data to Method {method_idx + 1} ({method_names[method_idx]})")
            else:
                print(f"  - Right arm force data has insufficient columns or could not be read")
        else:
            print(f"  - No right arm force data file found in folder {folder_name}")

    except Exception as e:
        print(f"Error processing right arm force data from folder {folder_name}: {e}")
        import traceback

        traceback.print_exc()

    # 加载左手交互力数据
    try:
        force_file_l = os.path.join(folder, "tcp_actual_force_l.txt")

        if os.path.exists(force_file_l):
            print(f"  - Found left arm force file: tcp_actual_force_l.txt")

            # 读取力数据
            try:
                force_data_raw_l = pd.read_csv(force_file_l, sep=' ', header=None)
            except:
                # 尝试不同的分隔符
                try:
                    force_data_raw_l = pd.read_csv(force_file_l, sep='\t', header=None)
                except:
                    try:
                        force_data_raw_l = pd.read_csv(force_file_l, sep=',', header=None)
                    except Exception as e:
                        print(f"  - Failed to read left arm force file with various delimiters: {e}")
                        force_data_raw_l = None

            if force_data_raw_l is not None and force_data_raw_l.shape[1] >= 4:
                # 提取时间和力数据
                force_time_l = force_data_raw_l.iloc[:, 0].values / 1000000
                force_x_l = force_data_raw_l.iloc[:, 2].values
                force_y_l = force_data_raw_l.iloc[:, 4].values
                force_z_l = force_data_raw_l.iloc[:, 6].values

                print(f"  - Left arm force data shape: {force_data_raw_l.shape}")

                # 应用滤波处理
                force_x_filtered_l = enhanced_force_filter(force_x_l)
                force_y_filtered_l = enhanced_force_filter(force_y_l)
                force_z_filtered_l = enhanced_force_filter(force_z_l)

                force_x_filtered_l, x_regions_l = enhanced_adaptive_smoothing(
                    force_x_filtered_l, force_time_l, method_names[method_idx], len(force_data_l[method_idx]), "l")
                force_y_filtered_l, y_regions_l = enhanced_adaptive_smoothing(
                    force_y_filtered_l, force_time_l, method_names[method_idx], len(force_data_l[method_idx]), "l")
                force_z_filtered_l, z_regions_l = enhanced_adaptive_smoothing(
                    force_z_filtered_l, force_time_l, method_names[method_idx], len(force_data_l[method_idx]), "l")

                # 存储左手力数据
                force_data_l[method_idx].append(
                    (force_time_l, force_x_filtered_l, force_y_filtered_l, force_z_filtered_l))
                print(f"  - Added left arm force data to Method {method_idx + 1} ({method_names[method_idx]})")
            else:
                print(f"  - Left arm force data has insufficient columns or could not be read")
        else:
            print(f"  - No left arm force data file found in folder {folder_name}")

    except Exception as e:
        print(f"Error processing left arm force data from folder {folder_name}: {e}")
        import traceback

        traceback.print_exc()

# 定义颜色和线型
colors = ['blue', 'red', 'green', 'purple', 'orange']
line_styles = ['-', '--', '-.', ':', '-']

# 应用方法特定的力数据平滑 - 分别处理左右手
if any(force_data_r):
    force_data_r = apply_method_specific_force_smoothing(force_data_r, method_names, arm_type="r")
if any(force_data_l):
    force_data_l = apply_method_specific_force_smoothing(force_data_l, method_names, arm_type="l")

# 可视化完整轨迹 - 分别处理左右手
for method_idx in range(4):
    if position_data_r[method_idx]:
        visualize_full_trajectory(position_data_r, method_idx, arm_type="r")
    if position_data_l[method_idx]:
        visualize_full_trajectory(position_data_l, method_idx, arm_type="l")

# 分别绘制每个实验的有效轨迹段数据 - 右手
for method_idx in range(4):
    # 检查是否有足够的数据
    if not force_data_r[method_idx] or not position_data_r[method_idx]:
        print(f"Insufficient right arm data for Method {method_idx + 1} ({method_names[method_idx]}), skipping")
        continue

    if len(force_data_r[method_idx]) != len(position_data_r[method_idx]):
        print(f"Mismatch between right arm force and position data for Method {method_idx + 1}, skipping")
        continue

    # 确定此方法有多少组数据
    num_experiments = min(len(force_data_r[method_idx]), len(position_data_r[method_idx]))
    print(f"Creating right arm plots for {method_names[method_idx]} with {num_experiments} sets")

    # 创建一个图表，包含六个子图（位置和力的X, Y, Z方向）
    fig, axes = plt.subplots(6, 1, figsize=(14, 24), sharex=True)
    plt.suptitle(f'{method_names[method_idx]} - Right Arm Position and Force', fontsize=20)

    # 分别绘制每个实验的有效轨迹段数据
    for exp_idx in range(num_experiments):
        if exp_idx >= 5:  # 限制为5组数据
            break

        # 获取位置和力数据
        pos_time, pos_x, pos_y, pos_z, traj_start_idx, traj_end_idx = position_data_r[method_idx][exp_idx]
        force_time, force_x, force_y, force_z = force_data_r[method_idx][exp_idx]

        # 截取有效轨迹段
        pos_time_segment = pos_time[traj_start_idx:traj_end_idx]
        pos_x_segment = pos_x[traj_start_idx:traj_end_idx]
        pos_y_segment = pos_y[traj_start_idx:traj_end_idx]
        pos_z_segment = pos_z[traj_start_idx:traj_end_idx]

        # 调整位置时间起点为0
        pos_time_segment_aligned = pos_time_segment - pos_time_segment[0]

        # 找到对应的力数据段
        force_start_idx = np.searchsorted(force_time, pos_time[traj_start_idx], side='left')
        force_end_idx = np.searchsorted(force_time, pos_time[traj_end_idx], side='right')

        if force_start_idx >= force_end_idx:
            print(f"  - Warning: No matching right arm force data for trajectory segment in set {exp_idx + 1}")
            continue

        force_time_segment = force_time[force_start_idx:force_end_idx]
        force_x_segment = force_x[force_start_idx:force_end_idx]
        force_y_segment = force_y[force_start_idx:force_end_idx]
        force_z_segment = force_z[force_start_idx:force_end_idx]

        # 调整力时间起点为0
        force_time_segment_aligned = force_time_segment - force_time_segment[0]

        # 绘制位置和力数据
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
    plt.savefig(os.path.join(base_dir, f'{method_names[method_idx]}_right_arm_position_force.png'), dpi=400,
                bbox_inches='tight')
    plt.close(fig)

# 分别绘制每个实验的有效轨迹段数据 - 左手
for method_idx in range(4):
    # 检查是否有足够的数据
    if not force_data_l[method_idx] or not position_data_l[method_idx]:
        print(f"Insufficient left arm data for Method {method_idx + 1} ({method_names[method_idx]}), skipping")
        continue

    if len(force_data_l[method_idx]) != len(position_data_l[method_idx]):
        print(f"Mismatch between left arm force and position data for Method {method_idx + 1}, skipping")
        continue

    # 确定此方法有多少组数据
    num_experiments = min(len(force_data_l[method_idx]), len(position_data_l[method_idx]))
    print(f"Creating left arm plots for {method_names[method_idx]} with {num_experiments} sets")

    # 创建一个图表，包含六个子图（位置和力的X, Y, Z方向）
    fig, axes = plt.subplots(6, 1, figsize=(14, 24), sharex=True)
    plt.suptitle(f'{method_names[method_idx]} - Left Arm Position and Force', fontsize=20)

    # 分别绘制每个实验的有效轨迹段数据
    for exp_idx in range(num_experiments):
        if exp_idx >= 5:  # 限制为5组数据
            break

        # 获取位置和力数据
        pos_time, pos_x, pos_y, pos_z, traj_start_idx, traj_end_idx = position_data_l[method_idx][exp_idx]
        force_time, force_x, force_y, force_z = force_data_l[method_idx][exp_idx]

        # 截取有效轨迹段
        pos_time_segment = pos_time[traj_start_idx:traj_end_idx]
        pos_x_segment = pos_x[traj_start_idx:traj_end_idx]
        pos_y_segment = pos_y[traj_start_idx:traj_end_idx]
        pos_z_segment = pos_z[traj_start_idx:traj_end_idx]

        # 调整位置时间起点为0
        pos_time_segment_aligned = pos_time_segment - pos_time_segment[0]

        # 找到对应的力数据段
        force_start_idx = np.searchsorted(force_time, pos_time[traj_start_idx], side='left')
        force_end_idx = np.searchsorted(force_time, pos_time[traj_end_idx], side='right')

        if force_start_idx >= force_end_idx:
            print(f"  - Warning: No matching left arm force data for trajectory segment in set {exp_idx + 1}")
            continue

        force_time_segment = force_time[force_start_idx:force_end_idx]
        force_x_segment = force_x[force_start_idx:force_end_idx]
        force_y_segment = force_y[force_start_idx:force_end_idx]
        force_z_segment = force_z[force_start_idx:force_end_idx]

        # 调整力时间起点为0
        force_time_segment_aligned = force_time_segment - force_time_segment[0]

        # 绘制位置和力数据
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
    plt.savefig(os.path.join(base_dir, f'{method_names[method_idx]}_left_arm_position_force.png'), dpi=400,
                bbox_inches='tight')
    plt.close(fig)

# 创建左右手对比图表
for method_idx in range(4):
    visualize_dual_arm_comparison(position_data_r, position_data_l, force_data_r, force_data_l, method_idx)

# 创建方法比较图 - 一张包含所有方法的图，分别显示左右手力数据
if (any(force_data_r) or any(force_data_l)) and (any(position_data_r) or any(position_data_l)):
    # 确保每种方法都有数据（左右手任一有数据即可）
    valid_methods = []
    for method_idx in range(4):
        if (force_data_r[method_idx] and position_data_r[method_idx]) or \
                (force_data_l[method_idx] and position_data_l[method_idx]):
            valid_methods.append(method_idx)

    if valid_methods:
        # 右手力数据比较
        if any(force_data_r) and any(position_data_r):
            fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
            plt.suptitle(f'Method Comparison - Right Arm Interaction Forces', fontsize=20)

            for method_idx in valid_methods:
                # 检查该方法是否有右手数据
                if not force_data_r[method_idx] or not position_data_r[method_idx]:
                    continue

                # 取第一个有效实验的数据
                pos_time, pos_x, pos_y, pos_z, traj_start_idx, traj_end_idx = position_data_r[method_idx][0]
                force_time, force_x, force_y, force_z = force_data_r[method_idx][0]

                # 截取有效轨迹段
                pos_time_segment = pos_time[traj_start_idx:traj_end_idx]

                # 找到对应的力数据段
                force_start_idx = np.searchsorted(force_time, pos_time[traj_start_idx], side='left')
                force_end_idx = np.searchsorted(force_time, pos_time[traj_end_idx], side='right')

                if force_start_idx >= force_end_idx:
                    print(
                        f"  - Warning: No matching right arm force data for trajectory segment in method {method_names[method_idx]}")
                    continue

                force_time_segment = force_time[force_start_idx:force_end_idx]
                force_x_segment = force_x[force_start_idx:force_end_idx]
                force_y_segment = force_y[force_start_idx:force_end_idx]
                force_z_segment = force_z[force_start_idx:force_end_idx]

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

            axes[0].set_title('X Direction Force - Right Arm')
            axes[0].set_ylabel('Force (N)')
            axes[0].grid(False)
            axes[0].legend(loc='best')

            axes[1].set_title('Y Direction Force - Right Arm')
            axes[1].set_ylabel('Force (N)')
            axes[1].grid(False)

            axes[2].set_title('Z Direction Force - Right Arm')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Force (N)')
            axes[2].grid(False)

            plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为suptitle留出空间
            plt.savefig(os.path.join(base_dir, 'method_comparison_right_arm.png'), dpi=400, bbox_inches='tight')
            plt.close(fig)

        # 左手力数据比较
        if any(force_data_l) and any(position_data_l):
            fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
            plt.suptitle(f'Method Comparison - Left Arm Interaction Forces', fontsize=20)

            for method_idx in valid_methods:
                # 检查该方法是否有左手数据
                if not force_data_l[method_idx] or not position_data_l[method_idx]:
                    continue

                # 取第一个有效实验的数据
                pos_time, pos_x, pos_y, pos_z, traj_start_idx, traj_end_idx = position_data_l[method_idx][0]
                force_time, force_x, force_y, force_z = force_data_l[method_idx][0]

                # 截取有效轨迹段
                pos_time_segment = pos_time[traj_start_idx:traj_end_idx]

                # 找到对应的力数据段
                force_start_idx = np.searchsorted(force_time, pos_time[traj_start_idx], side='left')
                force_end_idx = np.searchsorted(force_time, pos_time[traj_end_idx], side='right')

                if force_start_idx >= force_end_idx:
                    print(
                        f"  - Warning: No matching left arm force data for trajectory segment in method {method_names[method_idx]}")
                    continue

                force_time_segment = force_time[force_start_idx:force_end_idx]
                force_x_segment = force_x[force_start_idx:force_end_idx]
                force_y_segment = force_y[force_start_idx:force_end_idx]
                force_z_segment = force_z[force_start_idx:force_end_idx]

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

            axes[0].set_title('X Direction Force - Left Arm')
            axes[0].set_ylabel('Force (N)')
            axes[0].grid(False)
            axes[0].legend(loc='best')

            axes[1].set_title('Y Direction Force - Left Arm')
            axes[1].set_ylabel('Force (N)')
            axes[1].grid(False)

            axes[2].set_title('Z Direction Force - Left Arm')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Force (N)')
            axes[2].grid(False)

            plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为suptitle留出空间
            plt.savefig(os.path.join(base_dir, 'method_comparison_left_arm.png'), dpi=400, bbox_inches='tight')
            plt.close(fig)

# 在所有数据处理完成后比较左右手力数据
if (any(force_data_r) or any(force_data_l)) and (any(position_data_r) or any(position_data_l)):
    print("\nComparing force data smoothness and means between methods for both arms...")
    valid_methods, results = compare_force_smoothness_and_means(
        force_data_r, force_data_l, position_data_r, position_data_l, method_names)
    print("Comparison completed and visualizations saved.")

print("All plots created successfully!")