from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Input, Concatenate, RepeatVector, Reshape
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import os
import pickle
from scipy import signal


def create_multivariate_dataset(trajectory, muscle_input, muscle_output, max_activation, look_back=1):
    """
    创建多输入的时间序列数据集，包含最大激活因子

    参数:
    trajectory - 轨迹数据
    muscle_input - 输入肌肉激活数据
    muscle_output - 输出肌肉激活数据
    max_activation - 最大激活值
    look_back - 历史窗口大小

    返回:
    多个输入和一个输出
    """
    X_traj, X_muscle, X_max_act, Y = [], [], [], []

    for i in range(len(trajectory) - look_back):
        # 创建轨迹窗口
        traj_window = trajectory[i:(i + look_back)]
        X_traj.append(traj_window)

        # 创建肌肉输入窗口
        muscle_window = muscle_input[i:(i + look_back)]
        X_muscle.append(muscle_window)

        # 添加最大激活值因子 (重复look_back次)
        X_max_act.append(np.ones(look_back) * max_activation)

        # 下一个时间步的输出肌肉激活
        Y.append(muscle_output[i + look_back])

    return np.array(X_traj), np.array(X_muscle), np.array(X_max_act), np.array(Y)


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


# 设置随机种子以确保可重复性
np.random.seed(7)

# 加载数据
print("加载数据...")
# time_data = np.load('/home/ubuntu/HI-ImpRS-HRC/data/emg_record/chenzui&yuchen/sub_object_time.npy')
trajectory_data = np.load('/home/ubuntu/HI-ImpRS-HRC/data/emg_record/box_carrying/chenzui_vs_wuxi/sub_object_all.npy')
muscle_data = np.load('/home/ubuntu/HI-ImpRS-HRC/data/emg_record/box_carrying/chenzui_vs_wuxi/muscle_coactivation_all.npy')

# 数据预处理
print("预处理数据...")
# time_data = time_data[::10].reshape(-1, 1)
trajectory_data = trajectory_data[::10, :3] - trajectory_data[0, :3]
muscle_input_data = (muscle_data[:, 2] + muscle_data[:, 3]) / 2
muscle_output_data = (muscle_data[:, 0] + muscle_data[:, 1]) / 2
muscle_input_data_raw = muscle_input_data[::10].reshape(-1, 1)
muscle_output_data_raw = muscle_output_data[::10].reshape(-1, 1)
# time_data = time_data[20:, :]

# trajectory_data = trajectory_data[80:575, :]
# muscle_input_data = muscle_input_data_raw[80:575]
# muscle_output_data = muscle_output_data_raw[80:575]

muscle_input_data = filter_muscle_data(muscle_input_data_raw[:], method='butterworth',
                                    cutoff=4, fs=100, order=2)
muscle_output_data = filter_muscle_data(muscle_output_data_raw[:], method='butterworth',
                                     cutoff=4, fs=100, order=2)

visualize_filtering(muscle_input_data_raw[:], muscle_input_data, "肌肉输入数据滤波效果")
visualize_filtering(muscle_output_data_raw[:], muscle_output_data, "肌肉输出数据滤波效果")

# 计算训练数据中的最大激活值
max_input_activation = np.max(muscle_input_data)
max_output_activation = np.max(muscle_output_data)
max_activation = 0.1
print(f"训练数据最大激活值: {max_activation:.4f}")

# 创建并拟合缩放器
traj_scaler = MinMaxScaler(feature_range=(0, 1))
muscle_in_scaler = MinMaxScaler(feature_range=(0, 1))
muscle_out_scaler = MinMaxScaler(feature_range=(0, 1))
max_act_scaler = MinMaxScaler(feature_range=(0, 1))

# 使用fit_transform方法同时拟合和转换数据
trajectory_data = traj_scaler.fit_transform(trajectory_data)
muscle_input_data = muscle_in_scaler.fit_transform(muscle_input_data)
muscle_output_data = muscle_out_scaler.fit_transform(muscle_output_data)

# 对最大激活值进行归一化 (用于模型输入)
max_activation_scaled = max_act_scaler.fit_transform([[max_activation]])[0][0]

# 保存所有缩放器和最大激活值
scalers = {
    'traj_scaler': traj_scaler,
    'muscle_in_scaler': muscle_in_scaler,
    'muscle_out_scaler': muscle_out_scaler,
    'max_act_scaler': max_act_scaler
}

# 划分训练集和测试集
train_size = int(len(trajectory_data) * 0.7)
test_size = len(trajectory_data) - train_size

train_traj = trajectory_data[0:train_size]
test_traj = trajectory_data[train_size:len(trajectory_data)]

train_muscle_in = muscle_input_data[0:train_size]
test_muscle_in = muscle_input_data[train_size:len(trajectory_data)]

train_muscle_out = muscle_output_data[0:train_size]
test_muscle_out = muscle_output_data[train_size:len(trajectory_data)]

# 设置历史窗口大小
look_back = 5

# 创建训练和测试数据集，包含最大激活值因子
trainX_traj, trainX_muscle, trainX_max_act, trainY = create_multivariate_dataset(
    train_traj, train_muscle_in, train_muscle_out, max_activation_scaled, look_back)
testX_traj, testX_muscle, testX_max_act, testY = create_multivariate_dataset(
    test_traj, test_muscle_in, test_muscle_out, max_activation_scaled, look_back)

print(
    f"训练数据形状 - 轨迹: {trainX_traj.shape}, 肌肉输入: {trainX_muscle.shape}, 最大激活: {trainX_max_act.shape}, 输出: {trainY.shape}")

# 创建多输入的LSTM模型
# 轨迹输入分支
traj_input = Input(shape=(look_back, trajectory_data.shape[1]), name='trajectory_input')
traj_lstm = LSTM(4, return_sequences=False)(traj_input)

# 肌肉激活输入分支
muscle_input = Input(shape=(look_back, muscle_input_data.shape[1]), name='muscle_input')
muscle_lstm = LSTM(4, return_sequences=False)(muscle_input)

# 最大激活值输入分支 - 重塑为序列以匹配LSTM
max_act_input = Input(shape=(look_back,), name='max_activation_input')
max_act_reshaped = Reshape((look_back, 1))(max_act_input)
max_act_lstm = LSTM(2, return_sequences=False)(max_act_reshaped)

# 合并所有特征
merged = Concatenate()([traj_lstm, muscle_lstm, max_act_lstm])

# 输出层 - 肌肉激活预测 (使用ReLU确保非负输出)
output = Dense(muscle_output_data.shape[1], activation='relu')(merged)

# 创建模型
model = Model(inputs=[traj_input, muscle_input, max_act_input], outputs=output)
model.compile(loss='mean_squared_error', optimizer='adam')

# 显示模型结构
model.summary()

# 训练模型
print("开始训练模型...")
history = model.fit(
    [trainX_traj, trainX_muscle, trainX_max_act], trainY,
    epochs=20, batch_size=1, verbose=2,
    validation_data=([testX_traj, testX_muscle, testX_max_act], testY)
)

# 创建保存目录
save_dir = 'saved_multivariate_lstm_with_max_act_box_carrying'
os.makedirs(save_dir, exist_ok=True)

# 保存模型
model_path = os.path.join(save_dir, 'multivariate_lstm_model.h5')
model.save(model_path)
print(f'模型已保存到: {model_path}')

# 保存缩放器和最大激活值
scaler_path = os.path.join(save_dir, 'scalers.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scalers, f)
print(f'缩放器已保存到: {scaler_path}')

# 保存训练参数和最大激活值
params = {
    'look_back': look_back,
    'trajectory_dim': trajectory_data.shape[1],
    'muscle_input_dim': muscle_input_data.shape[1],
    'muscle_output_dim': muscle_output_data.shape[1],
    'max_activation': max_activation  # 保存原始的最大激活值
}

params_path = os.path.join(save_dir, 'params.pkl')
with open(params_path, 'wb') as f:
    pickle.dump(params, f)
print(f'训练参数已保存到: {params_path}')

# 进行预测
print("进行模型评估...")
trainPredict = model.predict([trainX_traj, trainX_muscle, trainX_max_act])
testPredict = model.predict([testX_traj, testX_muscle, testX_max_act])

# 反归一化预测结果
trainPredict = muscle_out_scaler.inverse_transform(trainPredict)
trainY_inv = muscle_out_scaler.inverse_transform(trainY)
testPredict = muscle_out_scaler.inverse_transform(testPredict)
testY_inv = muscle_out_scaler.inverse_transform(testY)

# 计算RMSE
train_rmse = math.sqrt(mean_squared_error(trainY_inv, trainPredict))
test_rmse = math.sqrt(mean_squared_error(testY_inv, testPredict))
print(f'训练RMSE: {train_rmse:.4f}')
print(f'测试RMSE: {test_rmse:.4f}')

# 计算相关系数
train_corr = np.corrcoef(trainY_inv.flatten(), trainPredict.flatten())[0, 1]
test_corr = np.corrcoef(testY_inv.flatten(), testPredict.flatten())[0, 1]
print(f'训练相关系数: {train_corr:.4f}')
print(f'测试相关系数: {test_corr:.4f}')

# 绘制学习曲线
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('curve')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(alpha=0.3)

# 绘制训练集预测结果
plt.subplot(2, 2, 2)
plt.plot(muscle_output_data_raw[:train_size], 'b-', alpha=0.5, label='raw data')
plt.plot(trainY_inv, 'b-', label='ground truth')
plt.plot(trainPredict, 'r--', label='prediction')
plt.title(f'train set prediction (RMSE: {train_rmse:.4f}, correlation coefficient: {train_corr:.4f})')
plt.xlabel('timestep')
plt.ylabel('muscle activation')
plt.legend()
plt.grid(alpha=0.3)

# 绘制测试集预测结果
plt.subplot(2, 2, 3)
plt.plot(muscle_output_data_raw[train_size:], 'b-', alpha=0.5, label='raw data')
plt.plot(testY_inv, 'b-', label='ground truth')
plt.plot(testPredict, 'r--', label='prediction')
plt.title(f'test set prediction (RMSE: {test_rmse:.4f}, correlation coefficient: {test_corr:.4f})')
plt.xlabel('timestep')
plt.ylabel('muscle activation')
plt.legend()
plt.grid(alpha=0.3)

# 绘制预测值与实际值的散点图
plt.subplot(2, 2, 4)
plt.scatter(trainY_inv, trainPredict, alpha=0.5, label='train')
plt.scatter(testY_inv, testPredict, alpha=0.5, label='test')
plt.plot([0, max(np.max(trainY_inv), np.max(testY_inv))],
         [0, max(np.max(trainY_inv), np.max(testY_inv))], 'k--', lw=2)
plt.title('prediction vs ground truth')
plt.xlabel('ground truth')
plt.ylabel('prediction')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300)
plt.show()

print("训练完成")