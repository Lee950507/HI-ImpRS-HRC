import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import os


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


def calculate_endpoint_stiffness(l, r, a1, a2, b1, b2):
    V_1 = l / np.linalg.norm(l)
    V_2 = np.cross(np.cross(r, l), l) / np.linalg.norm(np.cross(np.cross(r, l), l))
    V_3 = np.cross(r, l) / np.linalg.norm(np.cross(r, l))

    V = np.around(np.array([V_1, V_2, V_3]).T, 5)
    # print(V @ V.T)

    d1 = np.linalg.norm(l)
    d2 = np.linalg.norm(np.dot(r, np.cross(np.cross(r, l), l) / np.linalg.norm(np.cross(np.cross(r, l), l))))
    # D_s = np.around(np.diag([1, a1 / d1, a2 * d2]), 3)
    D_s = np.around(np.diag([1, a1 / d1, a2 * d2]) / pow(a1 * a2 * d2 / d1, 1 / 3), 5)
    # print('D_s:', D_s)

    A = 0.01
    Acc = b1 * A + b2

    Ke = np.around(V @ (Acc * D_s) @ V.T, 3)
    return Ke


if __name__ == '__main__':
    output_dir = "stiffness_results"
    os.makedirs(output_dir, exist_ok=True)

    pose = pd.read_csv('/home/ubuntu/HI-ImpRS-HRC/data/box_carrying/preprocess/1/processing_051602.csv')
    pose = np.array(pose)[2539:4380, :] / 100

    l_zhuo, r_zhuo, l_cz, r_cz = preprocess_pose(pose)

    # Fix the typo in the third parameter set
    A = [
        [0.272, 1.314, 3847.141, 151.684],
        [0.107, 2.200, 2678.765, 149.597],
        [0.399, 2.926, 1819.695, 128.581],  # Fixed value
        [0.341, 4.073, 2699.123, 112.562],
        [0.167, 4.528, 1260.290, 94.951]
    ]

    # Store results for each parameter set
    all_Ke_zhuo = []
    all_Ke_diag_zhuo = []

    # Calculate stiffness for each parameter set
    for param_idx, param_set in enumerate(A):
        Ke_zhuo = []
        Ke_diag_zhuo = []

        for i in range(len(l_zhuo[:, 0])):
            Ke2 = calculate_endpoint_stiffness(
                l_zhuo[i, :], r_zhuo[i, :],
                param_set[0], param_set[1], param_set[2], param_set[3]
            )
            Ke_zhuo.append(Ke2)
            Ke_diag_zhuo.append(np.diagonal(Ke2))

        # Convert to numpy arrays
        Ke_zhuo_array = np.array(Ke_zhuo)
        Ke_diag_zhuo_array = np.array(Ke_diag_zhuo)

        # Save stiffness matrices and diagonal values for this parameter set
        set_num = param_idx + 1

        # Save diagonal stiffness values
        diag_file = os.path.join(output_dir, f"stiffness_diag_set{set_num}.npy")
        np.save(diag_file, Ke_diag_zhuo_array)
        print(f"Saved diagonal stiffness values for Set {set_num} to {diag_file}")

        for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
            axis_file = os.path.join(output_dir, f"stiffness_{axis_name}_set{set_num}.npy")
            np.save(axis_file, Ke_diag_zhuo_array[:, axis_idx])
            print(f"Saved {axis_name}-axis stiffness for Set {set_num} to {axis_file}")

            # Store in lists for visualization
        all_Ke_zhuo.append(Ke_zhuo_array)
        all_Ke_diag_zhuo.append(Ke_diag_zhuo_array)

    # 1. Plot stiffness values over time for each parameter set
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    axes = axes.flatten()

    # 用于存储轴标签和标题
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

    # 为每个轴创建一个图
    for axis_idx in range(3):  # 0=X, 1=Y, 2=Z
        ax = axes[axis_idx]

        # 绘制所有参数集在当前轴上的数据
        for param_idx, Ke_diag in enumerate(all_Ke_diag_zhuo):
            ax.plot(Ke_diag[:, axis_idx], label=f'S{param_idx + 1}')

        ax.set_title(f'{axis_labels[axis_idx]} Stiffness Comparison Across Different Subjects')
        ax.set_ylabel('Stiffness (N/m)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel('Time step')
    plt.tight_layout()
    plt.savefig('stiffness_time_series_by_axis.png', dpi=300)
    plt.show()

    # 2. Compare average stiffness for each axis across parameter sets
    mean_stiffness_x = [np.mean(Ke_diag[:, 0]) for Ke_diag in all_Ke_diag_zhuo]
    mean_stiffness_y = [np.mean(Ke_diag[:, 1]) for Ke_diag in all_Ke_diag_zhuo]
    mean_stiffness_z = [np.mean(Ke_diag[:, 2]) for Ke_diag in all_Ke_diag_zhuo]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(A))
    width = 0.25

    ax.bar(x - width, mean_stiffness_x, width, color='r', label='X-axis')
    ax.bar(x, mean_stiffness_y, width, color='g', label='Y-axis')
    ax.bar(x + width, mean_stiffness_z, width, color='b', label='Z-axis')

    ax.set_xlabel('Subjects')
    ax.set_ylabel('Average Stiffness (N/m)')
    ax.set_title('Comparison of Average Stiffness Values')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i + 1}' for i in range(len(A))])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add parameter values as text on the bars
    for i, param_set in enumerate(A):
        ax.text(i - 0.3, mean_stiffness_x[i] + 5, f'{mean_stiffness_x[i]:.1f}',
                color='black', fontsize=8, ha='center')
        ax.text(i, mean_stiffness_y[i] + 5, f'{mean_stiffness_y[i]:.1f}',
                color='black', fontsize=8, ha='center')
        ax.text(i + 0.3, mean_stiffness_z[i] + 5, f'{mean_stiffness_z[i]:.1f}',
                color='black', fontsize=8, ha='center')

    plt.tight_layout()
    plt.savefig('average_stiffness_comparison.png', dpi=300)
    plt.show()

    # 3. Plot stiffness distribution using boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Prepare data for boxplots
    x_stiffness = [Ke_diag[:, 0] for Ke_diag in all_Ke_diag_zhuo]
    y_stiffness = [Ke_diag[:, 1] for Ke_diag in all_Ke_diag_zhuo]
    z_stiffness = [Ke_diag[:, 2] for Ke_diag in all_Ke_diag_zhuo]

    # Plot boxplots for each axis
    axes[0].boxplot(x_stiffness)
    axes[0].set_title('X-axis Stiffness Distribution')
    axes[0].set_xticklabels([f'S{i + 1}' for i in range(len(A))])
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(y_stiffness)
    axes[1].set_title('Y-axis Stiffness Distribution')
    axes[1].set_xticklabels([f'S{i + 1}' for i in range(len(A))])
    axes[1].grid(True, alpha=0.3)

    axes[2].boxplot(z_stiffness)
    axes[2].set_title('Z-axis Stiffness Distribution')
    axes[2].set_xticklabels([f'S{i + 1}' for i in range(len(A))])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stiffness_distribution.png', dpi=300)
    plt.show()

    # 4. Plot parameter influence on average stiffness
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate overall average stiffness for each parameter set
    avg_stiffness = [(np.mean(Ke_diag[:, 0]) + np.mean(Ke_diag[:, 1]) + np.mean(Ke_diag[:, 2])) / 3
                     for Ke_diag in all_Ke_diag_zhuo]

    # Create parameter labels
    param_labels = [f'S{i + 1}\n'
                    for i, p in enumerate(A)]

    # Plot
    ax.bar(np.arange(len(A)), avg_stiffness, color='purple', alpha=0.7)
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Average Overall Stiffness (N/m)')
    ax.set_title('Influence of Parameters on Average Stiffness')
    ax.set_xticks(np.arange(len(A)))
    ax.set_xticklabels(param_labels)
    ax.grid(True, alpha=0.3)

    # Add values on top of bars
    for i, v in enumerate(avg_stiffness):
        ax.text(i, v + 5, f'{v:.1f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('parameter_influence.png', dpi=300)
    plt.show()