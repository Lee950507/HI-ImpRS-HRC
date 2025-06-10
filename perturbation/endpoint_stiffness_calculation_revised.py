import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, signal


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


def resample_to_50hz(data_array, output_filename=None):
    """
    将不规则时间间隔的数据重采样为固定50Hz的频率

    参数:
    data_array: numpy数组，第一列是时间戳（微秒），后续列是要重采样的数据
    output_filename: 可选，如果提供，将结果写入该文件

    返回:
    numpy数组，重采样后的数据
    """
    # 确保输入是numpy数组
    data = np.array(data_array)

    # 提取时间戳和其他数据
    timestamps = data[:, 0]
    float_values = data[:, 1:7]  # 浮点值（列2-7）
    int_values = data[:, 7:9]  # 整数值（列8-9）

    # 确定开始和结束时间戳
    start_time = timestamps[0]
    end_time = timestamps[-1]

    # 将起始时间对齐到50Hz网格（20,000微秒的倍数）
    aligned_start = ((start_time + 19999) // 20000) * 20000  # 向上取整到最接近的20,000的倍数

    # 创建50Hz的新时间戳（每20,000微秒一个采样点）
    new_timestamps = np.arange(aligned_start, end_time, 20000)

    # 为浮点值创建线性插值函数
    float_interpolator = interpolate.interp1d(timestamps, float_values, axis=0, bounds_error=False,
                                              fill_value="extrapolate")

    # 为整数值创建最近邻插值函数
    int_interpolator = interpolate.interp1d(timestamps, int_values, axis=0, kind='nearest', bounds_error=False,
                                            fill_value="extrapolate")

    # 在新时间戳处插值
    new_float_values = float_interpolator(new_timestamps)
    new_int_values = int_interpolator(new_timestamps)

    # 将整数值四舍五入确保它们是整数
    new_int_values = np.round(new_int_values).astype(int)

    # 合并新时间戳和插值后的值
    new_data = np.column_stack((new_timestamps, new_float_values, new_int_values))

    # 如果提供了输出文件名，则将结果写入文件
    if output_filename:
        with open(output_filename, "w") as f:
            for row in new_data:
                f.write(" ".join(map(str, row)) + "\n")

    return new_data


def read_inconsistent_data(filename, expected_columns=9, strategy='skip', verbose=True):
    """
    读取可能存在列数不一致问题的数据文件

    参数:
    filename: str, 数据文件路径
    expected_columns: int, 预期的列数
    strategy: str, 处理列数不一致的策略
        - 'skip': 跳过列数不符合预期的行
        - 'fill': 用NaN填充缺失的列
        - 'trim': 截断多余的列
    verbose: bool, 是否输出详细信息

    返回:
    numpy.ndarray: 处理后的数据数组
    """
    if verbose:
        print(f"开始读取文件: {filename}")

    # 尝试使用pandas读取数据
    try:
        # 读取所有行，不强制列数一致
        all_rows = []
        line_num = 0

        with open(filename, 'r') as f:
            for line in f:
                line_num += 1
                values = line.strip().split()

                # 检查列数
                if len(values) == expected_columns:
                    # 列数正确，直接添加
                    try:
                        all_rows.append([float(x) for x in values])
                    except ValueError:
                        if verbose:
                            print(f"行 {line_num}: 包含无法转换为数字的值，已跳过")

                elif len(values) < expected_columns and strategy == 'fill':
                    # 列数不足，填充NaN
                    try:
                        row_data = [float(x) for x in values]
                        row_data.extend([float('nan')] * (expected_columns - len(values)))
                        all_rows.append(row_data)
                        if verbose and (line_num < 10 or line_num % 1000 == 0):
                            print(f"行 {line_num}: 列数不足({len(values)}), 已填充至{expected_columns}列")
                    except ValueError:
                        if verbose:
                            print(f"行 {line_num}: 包含无法转换为数字的值，已跳过")

                elif len(values) > expected_columns and strategy == 'trim':
                    # 列数过多，截断
                    try:
                        all_rows.append([float(x) for x in values[:expected_columns]])
                        if verbose and (line_num < 10 or line_num % 1000 == 0):
                            print(f"行 {line_num}: 列数过多({len(values)}), 已截断至{expected_columns}列")
                    except ValueError:
                        if verbose:
                            print(f"行 {line_num}: 包含无法转换为数字的值，已跳过")

                else:
                    # 策略为'skip'或者其他不匹配情况
                    if verbose and (line_num < 10 or line_num % 1000 == 0):
                        print(f"行 {line_num}: 列数不符({len(values)}vs{expected_columns})，已跳过")

        # 转换为numpy数组
        if len(all_rows) == 0:
            raise ValueError("没有有效数据行")

        data_array = np.array(all_rows)

        if verbose:
            print(f"成功读取了{len(data_array)}行数据")
            print(f"数据形状: {data_array.shape}")

        return data_array

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


'''
0125
'''
k = np.loadtxt('/home/ubuntu/Downloads/20250520_TCDS_revise/perturbation/optitrack/yuchen/k_6.txt')
force = np.loadtxt('/home/ubuntu/Downloads/20250520_TCDS_revise/perturbation/robot/yc12/tcp_actual_force_l.txt')
position = np.loadtxt('/home/ubuntu/Downloads/20250520_TCDS_revise/perturbation/robot/yc12/tcp_actual_position_rpy_l.txt')
# position_ref = np.loadtxt('/home/ubuntu/Downloads/20250520_TCDS_revise/perturbation/robot/ym8/tcp_desired_position_rpy_l.txt')

# force = read_inconsistent_data("/home/ubuntu/Downloads/20250520_TCDS_revise/perturbation/robot/yc11/tcp_actual_force_l.txt", expected_columns=9, strategy='skip')
# position = read_inconsistent_data("/home/ubuntu/Downloads/20250520_TCDS_revise/perturbation/robot/yc11/tcp_actual_position_rpy_l.txt", expected_columns=9, strategy='skip')

'''
切片
'''

# force = np.array(force)[1326:, 1:]
# position = np.array(position)[1326:, 1:]

# force = np.array(force)[1536:, 1:]
# position = np.array(position)[1536:, 1:]

# force = np.array(force)[1333:, 1:]
# position = np.array(position)[1333:, 1:]

'''
yiming
'''
# position = resample_to_50hz(position)[:, 1:4]
# force = resample_to_50hz(force)[:, 1:4]
# position_ref = resample_to_50hz(position_ref)[:, 1:4]

force = np.array(force)[::20, 1:4]
position = np.array(position)[::20, 1:4]
# position_ref = np.array(position_ref)[::20, 1:4]

force = force[1000:]
position = position[1000:]
# position_ref = position_ref[1500:]


index = 46

## 平滑处理
# for i in range(3):
#     force[:, i] = moving_average(force[:, i], 20)

for i in range(len(k)):
    if k[i] == 6:
        ## direction_yz
        position_y = position[i * 480:i * 480 + index, 1] - position[i * 480, 1]
        position_z = position[i * 480:i * 480 + index, 2] - position[i * 480, 2]
        force_y = force[i * 480:i * 480 + index, 1] - force[i * 480, 1]
        force_z = force[i * 480:i * 480 + index, 2] - force[i * 480, 2]
        # K_y = math.sqrt(np.mean([x**2 for x in force_y])) / math.sqrt(np.mean([x**2 for x in position_y]))
        # K_z = math.sqrt(np.mean([x ** 2 for x in force_z])) / math.sqrt(np.mean([x ** 2 for x in position_z]))
        # K_yz = math.sqrt(K_y ** 2 + K_z ** 2)
        # K_yz = math.sqrt(np.mean([x ** 2 for x in force_y]) + np.mean([x ** 2 for x in force_z])) / math.sqrt(np.mean([x ** 2 for x in position_y]) + np.mean([x ** 2 for x in position_z]))
        K_yz = math.sqrt((force_y[-1] - force_y[0]) ** 2 + (force_z[-1] - force_z[0]) ** 2) / math.sqrt(
            (position_y[-1] - position_y[0]) ** 2 + (position_z[-1] - position_z[0]) ** 2)
        print('K_yz:', K_yz)
    if k[i] == 5:
        ## direction_xz
        position_x = position[i * 480:i * 480 + index, 0] - position[i * 480, 0]
        position_z = position[i * 480:i * 480 + index, 2] - position[i * 480, 2]
        force_x = force[i * 480:i * 480 + index, 0] - force[i * 480, 0]
        force_z = force[i * 480:i * 480 + index, 2] - force[i * 480, 2]
        # K_x = math.sqrt(np.mean([x**2 for x in force_x])) / math.sqrt(np.mean([x**2 for x in position_x]))
        # K_z = math.sqrt(np.mean([x ** 2 for x in force_z])) / math.sqrt(np.mean([x ** 2 for x in position_z]))
        # K_xz = math.sqrt(K_x ** 2 + K_z ** 2)
        # K_xz = math.sqrt(np.mean([x ** 2 for x in force_x]) + np.mean([x ** 2 for x in force_z])) / math.sqrt(
        #     np.mean([x ** 2 for x in position_x]) + np.mean([x ** 2 for x in position_z]))
        K_xz = math.sqrt((force_x[-1] - force_x[0]) ** 2 +  (force_z[-1] - force_z[0]) ** 2) / math.sqrt((position_x[-1] - position_x[0]) ** 2 +  (position_z[-1] - position_z[0]) ** 2)
        print('K_xz:', K_xz)
    if k[i] == 4:
        ## direction_xy
        position_x = position[i * 480:i * 480 + index, 0] - position[i * 480, 0]
        position_y = position[i * 480:i * 480 + index, 1] - position[i * 480, 1]
        force_x = force[i * 480:i * 480 + index, 0] - force[i * 480, 0]
        force_y = force[i * 480:i * 480 + index, 1] - force[i * 480, 1]
        # K_x = math.sqrt(np.mean([x**2 for x in force_x])) / math.sqrt(np.mean([x**2 for x in position_x]))
        # K_y = math.sqrt(np.mean([x ** 2 for x in force_y])) / math.sqrt(np.mean([x ** 2 for x in position_y]))
        # K_xy = math.sqrt(K_x ** 2 + K_y ** 2)
        # K_xy = math.sqrt(np.mean([x ** 2 for x in force_x]) + np.mean([x ** 2 for x in force_y])) / math.sqrt(
        #     np.mean([x ** 2 for x in position_x]) + np.mean([x ** 2 for x in position_y]))
        K_xy = math.sqrt((force_x[-1] - force_x[0]) ** 2 + (force_y[-1] - force_y[0]) ** 2) / math.sqrt(
            (position_x[-1] - position_x[0]) ** 2 + (position_y[-1] - position_y[0]) ** 2)
        # print('K_xy:', K_xy)
    if k[i] == 3:
        ## direction_z
        position_z = position[i * 480:i * 480 + index, 2] - position[i * 480, 2]
        force_z = force[i * 480:i * 480 + index, 2] - force[i * 480, 2]
        # K_z = math.sqrt(np.mean([x ** 2 for x in force_z])) / math.sqrt(np.mean([x ** 2 for x in position_z]))
        K_z = (force_z[-1] - force_z[0]) / (position_z[-1] - position_z[0])
        K_zz = abs(K_z)
        print('K_zz:', K_zz)
    if k[i] == 2:
        ## direction_y
        position_y = position[i * 480:i * 480 + index, 1] - position[i * 480, 1]
        force_y = force[i * 480:i * 480 + index, 1] - force[i * 480, 1]
        # K_y = math.sqrt(np.mean([x**2 for x in force_y])) / math.sqrt(np.mean([x**2 for x in position_y]))
        K_y = (force_y[-1] - force_y[0]) / (position_y[-1] - position_y[0])
        K_yy = abs(K_y)
        print('K_yy:', K_yy)
    if k[i] == 1:
        ## direction_x
        position_x = position[i * 480:i * 480 + index, 0] - position[i * 480, 0]
        force_x = force[i * 480:i * 480 + index, 0] - force[i * 480, 0]
        # K_x = math.sqrt(np.mean([x**2 for x in force_x])) / math.sqrt(np.mean([x**2 for x in position_x]))
        K_x = (force_x[-1] - force_x[0]) / (position_x[-1] - position_x[0])
        K_xx = abs(K_x)
        # print('K_xx:', K_xx)

K_stiffness = np.array([K_xx, K_yy, K_zz, K_xy, K_xz, K_yz])
print(K_stiffness)

fig1, axs1 = plt.subplots(3)
fig1.suptitle('Positions')
axs1[0].plot(position[:, 0])
axs1[1].plot(position[:, 1])
axs1[2].plot(position[:, 2])

fig2, axs2 = plt.subplots(3)
fig2.suptitle('Forces')
axs2[0].plot(force[:, 0])
axs2[1].plot(force[:, 1])
axs2[2].plot(force[:, 2])

# fig3, axs3 = plt.subplots(3)
# fig3.suptitle('Positions_reference')
# axs1[0].plot(position_ref[:, 0])
# axs1[1].plot(position_ref[:, 1])
# axs1[2].plot(position_ref[:, 2])
plt.show()

