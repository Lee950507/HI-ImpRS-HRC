import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy import signal
from scipy.ndimage import gaussian_filter1d


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


# 设置全局字体大小
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# 定义数据路径和文件名
base_dir = "/home/ubuntu/TCDS_data/0618_hrc_box_yiming/emg"  # 替换为您的数据目录 # 替换为方法顺序文件路径

# 方法名称（用于标题和图例）
method_names = ["baseline", "learning-based VIC", "EMG-based VIC", "HI-ImpRS"]

# 读取方法顺序
method_order = np.array([1,3,4,2,3,4,1,2,3,1,4,2,2,4,1,3,1,2,3,4])

# 查找所有实验文件夹
experiment_folders = sorted(glob(os.path.join(base_dir, "*")),
                            key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else float('inf'))

# 过滤掉非数字命名的文件夹
experiment_folders = [folder for folder in experiment_folders if os.path.basename(folder).isdigit()]

# 创建数据结构以存储每种方法的数据
# method_data[method_index][experiment_index] = (time_data, muscle_activation_smooth)
method_data = [[] for _ in range(4)]

# 加载并组织数据
for folder_idx, folder in enumerate(experiment_folders):
    folder_name = os.path.basename(folder)
    print(f"Processing folder {folder_name} ({folder_idx + 1}/{len(experiment_folders)})")

    # 确定此文件夹使用的方法
    if folder_idx < len(method_order):
        method_idx = int(method_order[folder_idx]) - 1  # 转换为0-based索引
    else:
        print(f"Warning: No method specified for folder {folder_name}, skipping")
        continue

    # 加载数据
    try:
        time_file = os.path.join(folder, "emg_time.npy")
        muscle_smooth_file = os.path.join(folder, "muscle_activation_smooth.npy")

        # 检查文件是否存在
        if not os.path.exists(time_file) or not os.path.exists(muscle_smooth_file):
            # 尝试使用通配符查找文件
            time_files = glob(os.path.join(folder, "*time*.npy"))
            muscle_smooth_files = glob(os.path.join(folder, "*raw*.npy"))

            if time_files:
                time_file = time_files[0]
            if muscle_smooth_files:
                muscle_smooth_file = muscle_smooth_files[0]

        # 加载数据
        time_data = np.load(time_file)
        muscle_smooth_data = np.load(muscle_smooth_file)
        muscle_smooth_data = (muscle_smooth_data[0, :] + muscle_smooth_data[1, :]) / 2

        muscle_smooth_data = apply_filter(muscle_smooth_data,
                                     filter_type='savgol',
                                     # 'savgol', 'moving_avg', 'butterworth', 'gaussian', 'median'
                                     window_length=51,  # Savgol滤波器窗口大小
                                     polyorder=3)

        # 将数据添加到相应方法
        method_data[method_idx].append((time_data, muscle_smooth_data))
        print(f"  - Added to Method {method_idx + 1}, data shape: {muscle_smooth_data.shape}")

    except Exception as e:
        print(f"Error processing folder {folder_name}: {e}")

# 为每种方法创建图表
colors = ['blue', 'red', 'green', 'purple', 'orange']
line_styles = ['-', '--', '-.', ':', '-']

for method_idx in range(4):
    # 跳过没有数据的方法
    if not method_data[method_idx]:
        print(f"No data for Method {method_idx + 1}, skipping")
        continue

    # 确定此方法有多少组数据
    num_experiments = len(method_data[method_idx])
    print(f"Creating plot for Method {method_idx + 1} with {num_experiments} experiments")

    # 创建图表
    fig = plt.figure(figsize=(14, 8))
    method_data_1 = method_data[0]
    for exp_idx, (time, muscle_data) in enumerate(method_data[method_idx]):
        if exp_idx < 5:  # 限制为5组数据
            # 打印形状以确认数据正确
            print(f"  - Plotting exp {exp_idx + 1}, time shape: {time.shape}, muscle shape: {muscle_data.shape}")

            plt.plot(time, muscle_data,
                     color=colors[exp_idx % len(colors)],
                     linestyle=line_styles[exp_idx % len(line_styles)],
                     linewidth=2,
                     label=f'Experiment {exp_idx + 1}')

    plt.title(f'{method_names[method_idx]} - Muscle Activation')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation Level')
    plt.grid(False)
    plt.legend(loc='best')

    # 保存图表
    plt.savefig(os.path.join(base_dir, f'method_{method_idx + 1}_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)  # 明确关闭图形以释放内存

# 创建方法比较图 - 每种方法的第一个实验
if any(method_data):
    fig = plt.figure(figsize=(14, 8))

    for method_idx in range(4):
        if method_data[method_idx]:
            # 简单起见，只取第一个实验的数据
            time, muscle_data = method_data[method_idx][0]

            # 确认数据形状
            print(
                f"  - Method comparison: method {method_idx + 1}, time shape: {time.shape}, muscle shape: {muscle_data.shape}")

            plt.plot(time, muscle_data,
                     color=colors[method_idx],
                     linestyle=line_styles[method_idx],
                     linewidth=2,
                     label=f'{method_names[method_idx]}')

    plt.title('Method Comparison - Muscle Activation')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation Level')
    plt.grid(False)
    plt.legend(loc='best')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(os.path.join(base_dir, 'methods_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

print("All plots created successfully!")