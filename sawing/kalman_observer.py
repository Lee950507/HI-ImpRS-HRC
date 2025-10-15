import numpy as np
import matplotlib.pyplot as plt


class StiffnessDampingEKF:
    """
    正确的环境刚度阻尼估计 - 纯参数估计EKF

    物理模型: f = K_e * x_c + D_e * ẋ_c

    状态向量: θ = [K_e, D_e]^T (只有参数！)

    状态方程: θ_{k+1} = θ_k + w_k (随机游走模型)
    观测方程: f_k = K_e * x_c,k + D_e * ẋ_c,k + v_k
    """

    def __init__(self, dt=0.001, initial_K=1000.0, initial_D=10.0):
        self.dt = dt

        # 状态: θ = [K_e, D_e]
        self.theta = np.array([initial_K, initial_D])

        # 协方差矩阵 P
        self.P = np.diag([1e5, 1e2])

        # 过程噪声协方差 Q - 参数缓慢变化
        # 这个值需要根据实际情况调整
        self.Q = np.diag([1e0, 1e-2])  # 很小的过程噪声

        # 测量噪声协方差 R - 力传感器噪声
        self.R = 1.0  # 标准差1N的力噪声

    def predict(self):
        """
        预测步骤
        θ_{k+1|k} = θ_{k|k} (常量模型)
        P_{k+1|k} = P_{k|k} + Q
        """
        # 状态预测 - 参数不变
        # self.theta 不变

        # 协方差预测
        self.P = self.P + self.Q

    def update(self, f_meas, x_c, x_dot_c):
        """
        更新步骤

        观测方程: f = K_e * x_c + D_e * ẋ_c

        参数:
            f_meas: 测量的力 (N)
            x_c: 机器人位置 (m)
            x_dot_c: 机器人速度 (m/s)
        """
        K_e, D_e = self.theta

        # 观测预测 h(θ)
        f_pred = K_e * x_c + D_e * x_dot_c

        # 观测雅可比 H = ∂h/∂θ = [x_c, ẋ_c]
        H = np.array([[x_c, x_dot_c]])

        # 新息
        innovation = f_meas - f_pred

        # 新息协方差
        S = H @ self.P @ H.T + self.R
        S = S[0, 0]  # 转为标量

        # 卡尔曼增益
        K_gain = self.P @ H.T / S
        K_gain = K_gain.flatten()

        # 状态更新
        self.theta = self.theta + K_gain * innovation

        # 参数物理约束
        self.theta[0] = max(self.theta[0], 1.0)  # K_e > 0
        self.theta[1] = max(self.theta[1], 0.01)  # D_e > 0

        # 协方差更新 (Joseph形式)
        I = np.eye(2)
        IKH = I - np.outer(K_gain, H)
        self.P = IKH @ self.P @ IKH.T + np.outer(K_gain, K_gain) * self.R

        # 对称化
        self.P = (self.P + self.P.T) / 2

    def step(self, f_meas, x_c, x_dot_c):
        """完整滤波步骤"""
        self.predict()
        self.update(f_meas, x_c, x_dot_c)

        return {
            'stiffness': self.theta[0],
            'damping': self.theta[1],
            'stiffness_std': np.sqrt(self.P[0, 0]),
            'damping_std': np.sqrt(self.P[1, 1]),
            'state': self.theta.copy(),
            'covariance': self.P.copy()
        }


class RecursiveLeastSquares:
    """
    递推最小二乘 (RLS) - 对比算法
    模型: f = K * x + D * ẋ
    """

    def __init__(self, forgetting_factor=0.998):
        self.lambda_f = forgetting_factor
        self.theta = np.array([1000.0, 10.0])
        self.P = np.eye(2) * 1e5

    def update(self, f_meas, x_c, x_dot_c):
        """RLS更新"""
        phi = np.array([x_c, x_dot_c])

        # 增益
        Pphi = self.P @ phi
        K = Pphi / (self.lambda_f + phi @ Pphi)

        # 预测误差
        error = f_meas - self.theta @ phi

        # 参数更新
        self.theta = self.theta + K * error

        # 协方差更新
        self.P = (self.P - np.outer(K, Pphi)) / self.lambda_f

        # 约束
        self.theta[0] = max(self.theta[0], 1.0)
        self.theta[1] = max(self.theta[1], 0.01)

        return {
            'stiffness': self.theta[0],
            'damping': self.theta[1]
        }


def test_algorithms():
    """测试算法 - 时变参数版本"""

    # 定义时变参数函数
    def get_time_varying_params(t, scenario='step'):
        """
        根据不同场景返回时变的刚度和阻尼

        参数:
            t: 当前时间 (s)
            scenario: 场景类型
                'constant': 常量
                'step': 阶跃变化
                'ramp': 斜坡变化
                'sinusoidal': 正弦变化
                'multiple_step': 多阶跃
        """
        if scenario == 'constant':
            K_true = 1000.0
            D_true = 50.0

        elif scenario == 'step':
            # 单次阶跃
            K_true = 1000.0 if t < 3.0 else 2000.0
            D_true = 50.0 if t < 5.0 else 100.0

        elif scenario == 'ramp':
            # 线性增长
            K_true = 1000.0 + 300.0 * min(t / 5.0, 1.0)
            D_true = 50.0 + 50.0 * min(t / 5.0, 1.0)

        elif scenario == 'sinusoidal':
            # 正弦波动
            K_true = 1500.0 + 500.0 * np.sin(2 * np.pi * 0.3 * t)
            D_true = 75.0 + 25.0 * np.sin(2 * np.pi * 0.2 * t)

        elif scenario == 'multiple_step':
            # 多次阶跃
            if t < 2.0:
                K_true, D_true = 1000.0, 50.0
            elif t < 4.0:
                K_true, D_true = 1500.0, 80.0
            elif t < 6.0:
                K_true, D_true = 2000.0, 60.0
            elif t < 8.0:
                K_true, D_true = 1200.0, 90.0
            else:
                K_true, D_true = 1800.0, 70.0
        else:
            K_true, D_true = 1000.0, 50.0

        return K_true, D_true

    # 测试不同场景
    scenarios = ['constant', 'step', 'ramp', 'sinusoidal', 'multiple_step']
    noise_std = 5.0  # 固定噪声水平

    for scenario in scenarios:
        print(f"\n{'=' * 70}")
        print(f"测试场景: {scenario.upper()} - 力噪声标准差: {noise_std} N")
        print(f"{'=' * 70}")

        # 初始化估计器
        ekf = StiffnessDampingEKF(dt=0.001)
        ekf.R = noise_std ** 2
        # 增加过程噪声以跟踪时变参数
        if scenario != 'constant':
            ekf.Q = np.diag([1e1, 1e-1])  # 提高过程噪声

        rls = RecursiveLeastSquares(forgetting_factor=0.998)  # **修改：提高遗忘因子以避免发散**

        # 模拟参数
        np.random.seed(42)
        n_steps = 10000
        dt = 0.001

        # 存储结果
        ekf_K_history = []
        ekf_D_history = []
        rls_K_history = []
        rls_D_history = []
        K_true_history = []
        D_true_history = []
        time_history = []
        force_history = []
        position_history = []
        velocity_history = []

        for i in range(n_steps):
            t = i * dt

            # 激励信号（多频以提高可观测性）
            x_c = 0.01 * np.sin(2 * np.pi * 1.0 * t) + \
                  0.005 * np.sin(2 * np.pi * 3.0 * t)

            x_dot_c = 0.01 * 2 * np.pi * 1.0 * np.cos(2 * np.pi * 1.0 * t) + \
                      0.005 * 2 * np.pi * 3.0 * np.cos(2 * np.pi * 3.0 * t)

            # 获取当前时刻的真实参数（时变）
            K_true, D_true = get_time_varying_params(t, scenario)

            # 真实力
            f_true = K_true * x_c + D_true * x_dot_c

            # 测量（带噪声）
            f_meas = f_true + np.random.randn() * noise_std

            # EKF估计
            ekf_result = ekf.step(f_meas, x_c, x_dot_c)
            ekf_K_history.append(ekf_result['stiffness'])
            ekf_D_history.append(ekf_result['damping'])

            # RLS估计 - **添加异常检测**
            try:
                rls_result = rls.update(f_meas, x_c, x_dot_c)
                # 检查结果是否有效
                if np.isnan(rls_result['stiffness']) or np.isnan(rls_result['damping']) or \
                   np.isinf(rls_result['stiffness']) or np.isinf(rls_result['damping']):
                    # 如果出现nan或inf，使用上一个有效值
                    if i > 0:
                        rls_K_history.append(rls_K_history[-1])
                        rls_D_history.append(rls_D_history[-1])
                    else:
                        rls_K_history.append(1000.0)
                        rls_D_history.append(50.0)
                else:
                    rls_K_history.append(rls_result['stiffness'])
                    rls_D_history.append(rls_result['damping'])
            except:
                # 如果出错，使用上一个有效值
                if i > 0:
                    rls_K_history.append(rls_K_history[-1])
                    rls_D_history.append(rls_D_history[-1])
                else:
                    rls_K_history.append(1000.0)
                    rls_D_history.append(50.0)

            # 记录真实值
            K_true_history.append(K_true)
            D_true_history.append(D_true)

            time_history.append(t)
            force_history.append(f_meas)
            position_history.append(x_c)
            velocity_history.append(x_dot_c)

        # **修改：添加nan处理**
        # 计算误差统计（忽略nan值）
        ekf_K_error_abs = np.abs(np.array(ekf_K_history) - np.array(K_true_history))
        ekf_D_error_abs = np.abs(np.array(ekf_D_history) - np.array(D_true_history))
        rls_K_error_abs = np.abs(np.array(rls_K_history) - np.array(K_true_history))
        rls_D_error_abs = np.abs(np.array(rls_D_history) - np.array(D_true_history))

        # 统计最后2000个样本
        ekf_K_final = np.nanmean(ekf_K_history[-2000:])
        ekf_D_final = np.nanmean(ekf_D_history[-2000:])
        ekf_K_std = np.nanstd(ekf_K_history[-2000:])
        ekf_D_std = np.nanstd(ekf_D_history[-2000:])

        rls_K_final = np.nanmean(rls_K_history[-2000:])
        rls_D_final = np.nanmean(rls_D_history[-2000:])
        rls_K_std = np.nanstd(rls_K_history[-2000:])
        rls_D_std = np.nanstd(rls_D_history[-2000:])

        K_true_final = np.mean(K_true_history[-2000:])
        D_true_final = np.mean(D_true_history[-2000:])

        print(f"\n【EKF结果】")
        print(f"  刚度: {ekf_K_final:.1f} ± {ekf_K_std:.1f} N/m "
              f"(真值: {K_true_final:.1f}, MAE: {np.nanmean(ekf_K_error_abs):.1f})")
        print(f"  阻尼: {ekf_D_final:.2f} ± {ekf_D_std:.2f} N·s/m "
              f"(真值: {D_true_final:.1f}, MAE: {np.nanmean(ekf_D_error_abs):.2f})")

        print(f"\n【RLS结果】")
        print(f"  刚度: {rls_K_final:.1f} ± {rls_K_std:.1f} N/m "
              f"(真值: {K_true_final:.1f}, MAE: {np.nanmean(rls_K_error_abs):.1f})")
        print(f"  阻尼: {rls_D_final:.2f} ± {rls_D_std:.2f} N·s/m "
              f"(真值: {D_true_final:.1f}, MAE: {np.nanmean(rls_D_error_abs):.2f})")

        # 绘图
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)

        # 刚度估计
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_history, K_true_history, 'r-', linewidth=2.5,
                 label='True', alpha=0.9, zorder=3)
        ax1.plot(time_history, ekf_K_history, 'b-', linewidth=1.0,
                 label=f'EKF (MAE={np.nanmean(ekf_K_error_abs):.1f})', alpha=0.7)
        ax1.plot(time_history, rls_K_history, 'g--', linewidth=1.0,
                 label=f'RLS (MAE={np.nanmean(rls_K_error_abs):.1f})', alpha=0.7)
        ax1.set_ylabel('Stiffness (N/m)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Time-Varying Stiffness - {scenario.upper()}',
                      fontsize=13, fontweight='bold')
        # **添加ylim限制避免绘图错误**
        K_min = min(np.nanmin(K_true_history), np.nanmin(ekf_K_history), np.nanmin(rls_K_history))
        K_max = max(np.nanmax(K_true_history), np.nanmax(ekf_K_history), np.nanmax(rls_K_history))
        if np.isfinite(K_min) and np.isfinite(K_max):
            ax1.set_ylim([K_min * 0.9, K_max * 1.1])

        # 阻尼估计
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(time_history, D_true_history, 'r-', linewidth=2.5,
                 label='True', alpha=0.9, zorder=3)
        ax2.plot(time_history, ekf_D_history, 'b-', linewidth=1.0,
                 label=f'EKF (MAE={np.nanmean(ekf_D_error_abs):.2f})', alpha=0.7)
        ax2.plot(time_history, rls_D_history, 'g--', linewidth=1.0,
                 label=f'RLS (MAE={np.nanmean(rls_D_error_abs):.2f})', alpha=0.7)
        ax2.set_ylabel('Damping (N·s/m)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Time-Varying Damping', fontsize=13, fontweight='bold')
        # **添加ylim限制**
        D_min = min(np.nanmin(D_true_history), np.nanmin(ekf_D_history), np.nanmin(rls_D_history))
        D_max = max(np.nanmax(D_true_history), np.nanmax(ekf_D_history), np.nanmax(rls_D_history))
        if np.isfinite(D_min) and np.isfinite(D_max):
            ax2.set_ylim([D_min * 0.9, D_max * 1.1])

        # 刚度误差
        ax3 = fig.add_subplot(gs[2, :])
        ax3.plot(time_history, ekf_K_error_abs, 'b-', linewidth=0.8,
                 label=f'EKF (mean={np.nanmean(ekf_K_error_abs):.1f})', alpha=0.7)
        ax3.plot(time_history, rls_K_error_abs, 'g--', linewidth=0.8,
                 label=f'RLS (mean={np.nanmean(rls_K_error_abs):.1f})', alpha=0.7)
        ax3.set_ylabel('Stiffness Error (N/m)', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Stiffness Absolute Error', fontsize=12)

        # 阻尼误差
        ax4 = fig.add_subplot(gs[3, :])
        ax4.plot(time_history, ekf_D_error_abs, 'b-', linewidth=0.8,
                 label=f'EKF (mean={np.nanmean(ekf_D_error_abs):.2f})', alpha=0.7)
        ax4.plot(time_history, rls_D_error_abs, 'g--', linewidth=0.8,
                 label=f'RLS (mean={np.nanmean(rls_D_error_abs):.2f})', alpha=0.7)
        ax4.set_ylabel('Damping Error (N·s/m)', fontsize=11)
        ax4.set_xlabel('Time (s)', fontsize=11)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Damping Absolute Error', fontsize=12)

        # 位置和速度
        ax5 = fig.add_subplot(gs[4, 0])
        ax5.plot(time_history, position_history, 'b-', linewidth=0.8)
        ax5.set_ylabel('Position (m)', fontsize=10)
        ax5.set_xlabel('Time (s)', fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.set_title('Input Position', fontsize=11)

        ax6 = fig.add_subplot(gs[4, 1])
        ax6.plot(time_history, velocity_history, 'g-', linewidth=0.8)
        ax6.set_ylabel('Velocity (m/s)', fontsize=10)
        ax6.set_xlabel('Time (s)', fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_title('Input Velocity', fontsize=11)

        plt.suptitle(f'Time-Varying Parameter Estimation - {scenario.upper()}',
                     fontsize=14, fontweight='bold', y=0.995)

        plt.savefig(f'time_varying_{scenario}.png', dpi=300, bbox_inches='tight')
        plt.show()  # 直接显示图片

        print(f"\n图像已保存并显示: time_varying_{scenario}.png")

if __name__ == "__main__":
    test_algorithms()

    print(f"\n{'=' * 70}")
    print("所有测试完成！")
    print(f"{'=' * 70}")