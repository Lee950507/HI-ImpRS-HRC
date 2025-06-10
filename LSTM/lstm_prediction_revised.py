# 使用添加了最大激活值因子的模型进行预测
import numpy as np
import os
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math
import traceback
from scipy import signal


def create_multivariate_dataset(trajectory, muscle_input, max_activation, look_back=1):
    """
    为预测创建数据集，包含最大激活值

    参数:
    trajectory - 轨迹数据
    muscle_input - 输入肌肉激活数据
    max_activation - 最大激活值
    look_back - 历史窗口大小

    返回:
    适合模型输入的数据
    """
    X_traj, X_muscle, X_max_act = [], [], []

    for i in range(len(trajectory) - look_back):
        # 创建轨迹窗口
        traj_window = trajectory[i:(i + look_back)]
        X_traj.append(traj_window)

        # 创建肌肉输入窗口
        muscle_window = muscle_input[i:(i + look_back)]
        X_muscle.append(muscle_window)

        # 添加最大激活值
        X_max_act.append(np.ones(look_back) * max_activation)

    return np.array(X_traj), np.array(X_muscle), np.array(X_max_act)


def filter_muscle_data(data, method='butterworth', **kwargs):
    """
    对肌肉激活数据进行滤波

    参数:
    data - 输入的肌肉激活数据，形状为(n_samples, n_features)
    method - 滤波方法，可选值为:
             'ma' (移动平均)
             'butterworth' (巴特沃斯低通滤波器)
             'savgol' (Savitzky-Golay滤波器)
             'median' (中值滤波器)

    kwargs - 各滤波方法的特定参数:
             ma: window_size (窗口大小)
             butterworth: cutoff (截止频率), fs (采样频率), order (滤波器阶数)
             savgol: window_length (窗口长度), polyorder (多项式阶数)
             median: kernel_size (核大小)

    返回:
    滤波后的数据，与输入数据形状相同
    """
    # 保存原始数据形状
    original_shape = data.shape

    # 将数据转为一维数组，便于处理
    if len(original_shape) > 1:
        data_flat = data.flatten()
    else:
        data_flat = data

    # 根据选择的方法应用相应的滤波器
    if method == 'ma':
        # 移动平均滤波
        window_size = kwargs.get('window_size', 5)
        filtered_data = moving_average_filter(data_flat, window_size)

    elif method == 'butterworth':
        # 巴特沃斯低通滤波器
        cutoff = kwargs.get('cutoff', 0.1)  # 截止频率，标准化到[0, 1]
        fs = kwargs.get('fs', 2.0)  # 采样频率
        order = kwargs.get('order', 4)  # 滤波器阶数
        filtered_data = butterworth_filter(data_flat, cutoff, fs, order)

    elif method == 'savgol':
        # Savitzky-Golay滤波器
        window_length = kwargs.get('window_length', 11)  # 窗口长度，必须是奇数
        polyorder = kwargs.get('polyorder', 3)  # 多项式阶数
        filtered_data = savgol_filter(data_flat, window_length, polyorder)

    elif method == 'median':
        # 中值滤波器
        kernel_size = kwargs.get('kernel_size', 5)  # 核大小
        filtered_data = median_filter(data_flat, kernel_size)

    else:
        raise ValueError(f"不支持的滤波方法: {method}")

    # 将滤波后的数据恢复到原始形状
    filtered_data = filtered_data.reshape(original_shape)

    return filtered_data


def moving_average_filter(data, window_size):
    """
    应用移动平均滤波

    参数:
    data - 一维输入数据
    window_size - 滑动窗口大小

    返回:
    滤波后的数据
    """
    # 使用卷积实现移动平均
    window = np.ones(window_size) / window_size
    filtered_data = np.convolve(data, window, mode='same')

    # 处理边缘效应
    # 对于信号开始和结束的部分，窗口会超出信号范围，导致边缘效应
    # 这里我们使用原始数据填充这些区域
    half_window = window_size // 2
    filtered_data[:half_window] = data[:half_window]
    filtered_data[-half_window:] = data[-half_window:]

    return filtered_data


def butterworth_filter(data, cutoff, fs, order):
    """
    应用巴特沃斯低通滤波器

    参数:
    data - 一维输入数据
    cutoff - 截止频率 (Hz)
    fs - 采样频率 (Hz)
    order - 滤波器阶数

    返回:
    滤波后的数据
    """
    # 计算归一化截止频率
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # 设计滤波器
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # 应用滤波器
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def savgol_filter(data, window_length, polyorder):
    """
    应用Savitzky-Golay滤波器

    参数:
    data - 一维输入数据
    window_length - 窗口长度 (必须是奇数)
    polyorder - 多项式阶数

    返回:
    滤波后的数据
    """
    # 确保窗口长度是奇数
    if window_length % 2 == 0:
        window_length += 1

    # 应用Savitzky-Golay滤波器
    filtered_data = signal.savgol_filter(data, window_length, polyorder)

    return filtered_data


def median_filter(data, kernel_size):
    """
    应用中值滤波器

    参数:
    data - 一维输入数据
    kernel_size - 滤波器核大小

    返回:
    滤波后的数据
    """
    # 应用中值滤波器
    filtered_data = signal.medfilt(data, kernel_size=kernel_size)

    return filtered_data


def visualize_filtering(original_data, filtered_data, title="滤波效果比较"):
    """
    可视化原始数据和滤波后的数据

    参数:
    original_data - 原始数据
    filtered_data - 滤波后的数据
    title - 图表标题
    """
    plt.figure(figsize=(12, 6))

    # 绘制原始数据
    plt.plot(original_data, 'b-', alpha=0.5, label='原始数据')

    # 绘制滤波后的数据
    plt.plot(filtered_data, 'r-', label='滤波后的数据')

    plt.title(title)
    plt.xlabel('样本')
    plt.ylabel('振幅')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 主程序开始
print("\n开始对新样本进行预测:")

# 设置保存目录
save_dir = 'saved_multivariate_lstm_with_max_act'

# 加载模型、缩放器和参数
try:
    # 加载模型
    model_path = os.path.join(save_dir, 'multivariate_lstm_model.h5')
    model = load_model(model_path)
    print(f"成功加载模型: {model_path}")

    # 加载参数
    params_path = os.path.join(save_dir, 'params.pkl')
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    look_back = params['look_back']
    training_max_activation = params['max_activation']
    print(f"成功加载参数，look_back = {look_back}, 训练数据最大激活值 = {training_max_activation}")

    # 加载缩放器
    scaler_path = os.path.join(save_dir, 'scalers.pkl')
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    print(f"成功加载缩放器")

except Exception as e:
    print(f"加载模型或参数失败: {e}")
    traceback.print_exc()
    exit(1)

# 加载新样本数据
try:
    print("加载新样本数据...")

    # 加载不同个体的数据
    # time_data = np.load('/home/ubuntu/HI-ImpRS-HRC/data/emg_record/chenzui&yiming/sub_object_time.npy')
    trajectory_data = np.load('/home/ubuntu/HI-ImpRS-HRC/data/emg_record/chenzui&yuchen2/sub_object_all.npy')
    muscle_data = np.load('/home/ubuntu/HI-ImpRS-HRC/data/emg_record/chenzui&yuchen2/muscle_coactivation_all.npy')

    trajectory_data = trajectory_data[::10, :3] - trajectory_data[0, :3]
    muscle_input_data = (muscle_data[:, 2] + muscle_data[:, 3]) / 2
    muscle_output_data = (muscle_data[:, 0] + muscle_data[:, 1]) / 2
    muscle_input_data_raw = muscle_input_data[::10].reshape(-1, 1)
    muscle_output_data_raw = muscle_output_data[::10].reshape(-1, 1)
    # time_data = time_data[20:, :]

    trajectory_data = trajectory_data[:, :]

    muscle_input_data = filter_muscle_data(muscle_input_data_raw[:], method='butterworth',
                                           cutoff=10, fs=100, order=2)
    muscle_output_data = filter_muscle_data(muscle_output_data_raw[:], method='butterworth',
                                            cutoff=10, fs=100, order=2)

    visualize_filtering(muscle_input_data_raw, muscle_input_data, "肌肉输入数据滤波效果")
    visualize_filtering(muscle_output_data_raw, muscle_output_data, "肌肉输出数据滤波效果")

    # 计算测试数据的最大激活值 - 这是新个体的特定值
    test_max_input_activation = np.max(muscle_input_data)
    test_max_output_activation = np.max(muscle_output_data)
    test_max_activation = 0.2
    print(f"测试数据最大激活值: {test_max_activation:.4f} (训练数据: {training_max_activation:.4f})")

    # 从数据后30%开始作为新样本
    total_len = len(trajectory_data)
    start_idx = int(total_len * 0)

    new_time_data = trajectory_data[start_idx:, :]
    new_trajectory_data = trajectory_data[start_idx:, :]
    new_muscle_input_data = muscle_input_data[start_idx:, :]
    new_muscle_output_data = muscle_output_data[start_idx:, :]

    print(
        f"新样本数据形状 - 轨迹: {new_trajectory_data.shape}, 肌肉输入: {new_muscle_input_data.shape}, 肌肉输出: {new_muscle_output_data.shape}")

except Exception as e:
    print(f"加载新样本数据失败: {e}")
    traceback.print_exc()
    exit(1)

# 数据预处理 - 使用保存的缩放器
try:
    traj_scaler = scalers['traj_scaler']
    muscle_in_scaler = scalers['muscle_in_scaler']
    muscle_out_scaler = scalers['muscle_out_scaler']
    max_act_scaler = scalers['max_act_scaler']

    # 标准化新样本数据
    new_trajectory_data_scaled = traj_scaler.transform(new_trajectory_data)
    new_muscle_input_data_scaled = muscle_in_scaler.transform(new_muscle_input_data)
    new_muscle_output_data_scaled = muscle_out_scaler.transform(new_muscle_output_data)

    # 归一化最大激活值
    test_max_activation_scaled = max_act_scaler.transform([[test_max_activation]])[0][0]
    print(f"归一化后的测试最大激活值: {test_max_activation_scaled:.4f}")

    print("数据预处理完成")

except Exception as e:
    print(f"数据预处理失败: {e}")
    traceback.print_exc()
    exit(1)

# 创建模型输入数据
try:
    print("创建模型输入数据...")
    X_traj, X_muscle, X_max_act = create_multivariate_dataset(
        new_trajectory_data_scaled, new_muscle_input_data_scaled,
        test_max_activation_scaled, look_back)

    # 获取对应的真实输出值
    Y_true = new_muscle_output_data_scaled[look_back:]

    print(f"模型输入形状 - 轨迹: {X_traj.shape}, 肌肉: {X_muscle.shape}, 最大激活值: {X_max_act.shape}")
    print(f"真实输出形状: {Y_true.shape}")

except Exception as e:
    print(f"创建模型输入数据失败: {e}")
    traceback.print_exc()
    exit(1)

# 使用模型进行预测
try:
    print("开始预测...")
    predictions = model.predict([X_traj, X_muscle, X_max_act])
    print(f"预测完成，预测结果形状: {predictions.shape}")

    # 反缩放预测结果和真实值
    predictions_unscaled = muscle_out_scaler.inverse_transform(predictions)
    Y_true_unscaled = muscle_out_scaler.inverse_transform(Y_true)

    # 确保预测结果非负 (虽然使用了ReLU激活函数，但以防万一)
    predictions_unscaled = np.maximum(0, predictions_unscaled)

    # 计算RMSE
    rmse = math.sqrt(mean_squared_error(Y_true_unscaled, predictions_unscaled))
    print(f"新样本预测RMSE: {rmse:.4f}")

    # 计算相关系数
    correlation = np.corrcoef(Y_true_unscaled.flatten(), predictions_unscaled.flatten())[0, 1]
    print(f"新样本预测相关系数: {correlation:.4f}")

    # 查看最小值和最大值
    print(f"预测结果最小值: {np.min(predictions_unscaled):.4f}")
    print(f"预测结果最大值: {np.max(predictions_unscaled):.4f}")
    print(f"实际值最小值: {np.min(Y_true_unscaled):.4f}")
    print(f"实际值最大值: {np.max(Y_true_unscaled):.4f}")

except Exception as e:
    print(f"预测失败: {e}")
    traceback.print_exc()
    exit(1)

# 可视化预测结果
try:
    print("可视化预测结果...")

    # 设置图形大小
    plt.figure(figsize=(15, 10))

    # 绘制时间序列预测
    plt.subplot(2, 1, 1)
    plt.plot(Y_true_unscaled, 'b-', label='ground truth')
    plt.plot(predictions_unscaled, 'r--', label='prediction')
    plt.title(f'results (RMSE: {rmse:.4f}, correlation coefficient: {correlation:.4f})')
    plt.xlabel('timestep')
    plt.ylabel('muscle activation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制散点图
    plt.subplot(2, 1, 2)
    plt.scatter(Y_true_unscaled, predictions_unscaled, alpha=0.5)
    plt.plot([0, np.max(Y_true_unscaled)], [0, np.max(Y_true_unscaled)], 'k--', lw=2)
    plt.title('prediction vs ground truth')
    plt.xlabel('ground truth')
    plt.ylabel('prediction')
    plt.grid(True, alpha=0.3)

    # 添加测试信息
    info_text = f'max activation_test: {test_max_activation:.2f}\n max activation_train: {training_max_activation:.2f}\n ratio: {test_max_activation / training_max_activation:.2f}x'
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('new_subject_prediction_results.png', dpi=300)
    plt.show()

    # 额外绘制前100个时间步的详细比较
    plt.figure(figsize=(12, 6))
    max_points = min(100, len(Y_true_unscaled))
    plt.plot(Y_true_unscaled[:max_points], 'b-', label='ground truth')
    plt.plot(predictions_unscaled[:max_points], 'r--', label='prediction')
    plt.title(f'first {max_points} timestep comparison results')
    plt.xlabel('timestep')
    plt.ylabel('muscle activation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('detailed_prediction_comparison.png', dpi=300)
    plt.show()

except Exception as e:
    print(f"可视化结果失败: {e}")
    traceback.print_exc()

print("预测和评估完成")