import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StiffnessIdentification:
    def __init__(self, subject_name, pose_file_path, impedance_file_path):
        """初始化类，加载数据"""
        self.subject = subject_name
        self.pose = self._load_pose_data(pose_file_path)
        self.K = self._load_impedance_data(impedance_file_path)
        self.plot_data = []

        # 肌肉激活水平数据 - 应从外部获取或计算
        self.A = self._get_muscle_activation_data()

        # 计算几何参数
        self.V, self.d1, self.d2 = self._calculate_geometric_parameters()

    def _load_pose_data(self, file_path):
        """加载姿态数据"""
        try:
            return np.loadtxt(file_path, delimiter=',')
        except Exception as e:
            logger.error(f"加载姿态数据失败: {e}")
            raise

    def _load_impedance_data(self, file_path):
        """加载阻抗数据"""
        try:
            df = pd.read_excel(file_path)
            return df.to_numpy()
        except Exception as e:
            logger.error(f"加载阻抗数据失败: {e}")
            raise

    def _get_muscle_activation_data(self):
        """获取肌肉激活水平数据"""
        # 原始代码中的A是一维数组，但使用时似乎需要每个索引对应一个值
        # 修改为正确的二维结构
        # raw_data = np.array([0.09542, 0.25224, 0.09617, 0.17838, 0.09179, 0.17792,
        #                      0.09819, 0.1907, 0.09013, 0.18529, 0.1023, 0.18885])
        # raw_data = np.array([0.11519, 0.20092, 0.1052, 0.19076, 0.09908, 0.2097,
        #                      0.10043, 0.19755, 0.10366, 0.18373, 0.08938, 0.17988])
        raw_data = np.array([0.0938, 0.18283, 0.08153, 0.1677, 0.09728, 0.20265,
                             0.08245, 0.16499, 0.08146, 0.15267, 0.07081, 0.1401])

        # 重塑为更合理的形式 - 假设每个激活水平是一个标量
        return raw_data.reshape(12)

    def calculate_endpoint_stiffness(self, pose_data):
        """计算端点刚度的几何参数"""
        pos_shoulder = pose_data[0, :3] / 100
        pos_elbow = pose_data[1, :3] / 100
        pos_wrist = pose_data[2, :3] / 100

        l = np.zeros(3)
        r = np.zeros(3)

        # 计算上臂和前臂向量
        l_init = pos_wrist - pos_shoulder
        l[0] = -l_init[0]
        l[1] = -l_init[2]
        l[2] = l_init[1]

        r_init = pos_elbow - pos_shoulder
        r[0] = -r_init[0]
        r[1] = -r_init[2]
        r[2] = r_init[1]

        # 计算标准基向量
        V_1 = l / np.linalg.norm(l)

        # 检查是否有可能发生除零错误
        cross_product = np.cross(l, r)
        norm_cross = np.linalg.norm(cross_product)
        if norm_cross < 1e-10:
            logger.warning("计算交叉积时发现近似零值，可能导致不稳定结果")
            # 添加小的扰动以避免数值问题
            cross_product += np.array([1e-10, 1e-10, 1e-10])
            norm_cross = np.linalg.norm(cross_product)

        V_3 = cross_product / norm_cross
        V_2 = np.cross(V_3, V_1)  # 更简洁的计算方式

        V = np.array([V_1, V_2, V_3]).T

        d1 = np.linalg.norm(l)
        d2 = np.linalg.norm(np.dot(r, V_2))  # 更简洁的计算方式

        return V, d1, d2

    def _calculate_geometric_parameters(self):
        """为所有姿态计算几何参数"""
        n_poses = 12  # 假设有12个姿态

        # 预分配数组以提高效率
        V_all = np.zeros((n_poses, 3, 3))
        d1_all = np.zeros(n_poses)
        d2_all = np.zeros(n_poses)

        for i in range(n_poses):
            pose_slice = self.pose[3 * i:3 * i + 3, :]
            V, d1, d2 = self.calculate_endpoint_stiffness(pose_slice)
            V_all[i] = V
            d1_all[i] = d1
            d2_all[i] = d2

        return V_all, d1_all, d2_all

    def objective_function(self, x):
        """优化目标函数"""
        obj = 0
        ki = [1] * 12  # 每个姿态的刚度矩阵数量

        # 预计算刚度矩阵
        Ksh = [np.array([[[self.K[j][i * 6], self.K[j][i * 6 + 3], self.K[j][i * 6 + 4]],
                          [self.K[j][i * 6 + 3], self.K[j][i * 6 + 1], self.K[j][i * 6 + 5]],
                          [self.K[j][i * 6 + 4], self.K[j][i * 6 + 5], self.K[j][i * 6 + 2]]]
                         for i in range(ki[j])])
               for j in range(len(ki))]

        for j in range(len(ki)):
            # 构造矩阵D
            ds = np.array([[1, 0, 0],
                           [0, x[0] / self.d1[j], 0],
                           [0, 0, x[1] * self.d2[j]]])

            # 计算模型预测的刚度矩阵
            vdvt = self.V[j] @ ds @ self.V[j].T

            # 确保不取绝对值，除非有物理意义要求
            model_K = vdvt * (x[2] * self.A[j] + x[3])

            for i in range(ki[j]):
                # 使用矩阵对数度量计算差异
                # 确保矩阵是正定的，避免对负值或零取对数
                if np.any(model_K <= 0) or np.any(Ksh[j][i] <= 0):
                    penalty = 1e6  # 大惩罚值
                    obj += penalty
                    continue

                log1 = np.log(model_K)
                log2 = np.log(Ksh[j][i])
                obj += np.linalg.norm(log1 - log2, 'fro')  # Frobenius范数

        self.plot_data.append(obj)
        return obj

    def optimize(self, initial_guess=None, method='L-BFGS-B'):
        """执行优化"""
        if initial_guess is None:
            initial_guess = np.array([0.2, 2, 1500, 150])

        bounds = [(0.1, 0.4), (0.5, 5), (500, 4500), (50, 500)]

        # 约束函数 - 确保所有参数都是正的
        def constraint(x):
            return x  # 简单地返回参数，因为bounds已经限制了范围

        constraint_definition = {'type': 'ineq', 'fun': constraint}

        logger.info(f"开始优化，使用初始猜测值: {initial_guess}")

        if method == 'L-BFGS-B':
            result = minimize(self.objective_function, initial_guess,
                              bounds=bounds, method='L-BFGS-B')
        elif method == 'nelder-mead':
            result = minimize(self.objective_function, initial_guess,
                              method='nelder-mead')
        elif method == 'SLSQP':
            result = minimize(self.objective_function, initial_guess,
                              constraints=constraint_definition,
                              bounds=bounds, method='SLSQP')
        elif method == 'differential_evolution':
            result = differential_evolution(self.objective_function, bounds)
        else:
            raise ValueError(f"未知的优化方法: {method}")

        min_value = result.fun
        optimal_point = result.x

        logger.info(f"优化完成。最终目标函数值: {min_value}")
        logger.info(f"最优参数: c1={optimal_point[0]}, c2={optimal_point[1]}, "
                    f"alpha1={optimal_point[2]}, alpha2={optimal_point[3]}")

        return min_value, optimal_point

    def plot_optimization_progress(self):
        """绘制优化进度"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.plot_data, label=f'Subject: {self.subject}')
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.title('优化进度')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # 对数刻度更好地显示收敛行为
        plt.tight_layout()

        # 保存图像
        output_dir = "optimization_results"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{self.subject}_optimization_progress.png")
        plt.close()

        return self.plot_data[-1]


def main():
    """主函数"""
    subject = 'yiming'
    pose_file = '/home/ubuntu/Downloads/20250520_TCDS_revise/perturbation/optitrack/yuchen/overall_joint_pos.txt'
    impedance_file = "/home/ubuntu/Downloads/20250520_TCDS_revise/perturbation/impedance_yc.xlsx"

    # 创建识别对象
    identifier = StiffnessIdentification(subject, pose_file, impedance_file)

    # 尝试不同的优化方法
    methods = ['L-BFGS-B', 'SLSQP', 'differential_evolution']
    best_result = (float('inf'), None)

    for method in methods:
        try:
            logger.info(f"使用 {method} 方法进行优化")
            min_value, optimal_point = identifier.optimize(method=method)

            # 记录最佳结果
            if min_value < best_result[0]:
                best_result = (min_value, optimal_point)

            # 绘制优化进度
            identifier.plot_optimization_progress()

        except Exception as e:
            logger.error(f"方法 {method} 优化失败: {e}")

    # 报告最佳结果
    logger.info(f"最佳优化结果: 目标函数值 = {best_result[0]}")
    logger.info(f"最优参数: c1={best_result[1][0]}, c2={best_result[1][1]}, "
                f"alpha1={best_result[1][2]}, alpha2={best_result[1][3]}")

    return best_result[0]


if __name__ == '__main__':
    main()