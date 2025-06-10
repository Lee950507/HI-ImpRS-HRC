import os
import numpy as np
import argparse
from natsort import natsorted
import re


def extract_group_number(folder_name):
    """从文件夹名称中提取组号，用于排序"""
    # 匹配 "X.Y" 格式的文件夹名
    match = re.match(r'(\d+)\.(\d+)', folder_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (float('inf'), float('inf'))  # 对于不匹配的文件夹，放到最后


def merge_data_files(base_folder, output_folder):
    """
    合并指定文件夹内所有子文件夹中的数据文件

    参数:
    base_folder - 包含所有数据子文件夹的基础文件夹路径
    output_folder - 输出合并后数据的文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    # 按组号排序文件夹（如1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3）
    subfolders = sorted(subfolders, key=extract_group_number)

    print(f"排序后的子文件夹顺序: {subfolders}")

    # 初始化空列表来存储每个文件的数据
    sub_object_data = []
    muscle_coactivation_data = []

    # 遍历每个子文件夹
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)

        # 构建文件路径
        sub_object_file = os.path.join(subfolder_path, "sub_object_all.npy")
        muscle_file = os.path.join(subfolder_path, "muscle_coactivation_all.npy")

        # 检查文件是否存在
        if not os.path.exists(sub_object_file) or not os.path.exists(muscle_file):
            print(f"警告: 在 {subfolder} 中未找到所需文件，跳过此文件夹")
            continue

        # 加载数据
        try:
            sub_obj = np.load(sub_object_file)
            muscle = np.load(muscle_file)

            print(f"从 {subfolder} 加载的数据形状: sub_object={sub_obj.shape}, muscle={muscle.shape}")

            # 添加到列表
            sub_object_data.append(sub_obj)
            muscle_coactivation_data.append(muscle)

        except Exception as e:
            print(f"加载 {subfolder} 中的数据时出错: {e}")
            continue

    # 检查是否有数据被加载
    if not sub_object_data or not muscle_coactivation_data:
        print("错误: 没有找到有效数据")
        return

    # 检查所有数据的形状是否兼容（除了第一维外）
    sub_obj_shapes = [data.shape[1:] for data in sub_object_data]
    muscle_shapes = [data.shape[1:] for data in muscle_coactivation_data]

    if len(set(str(shape) for shape in sub_obj_shapes)) > 1:
        print(f"警告: sub_object数据形状不一致: {sub_obj_shapes}")

    if len(set(str(shape) for shape in muscle_shapes)) > 1:
        print(f"警告: muscle数据形状不一致: {muscle_shapes}")

    # 沿第一个维度（通常是时间维度）连接数据
    try:
        merged_sub_object = np.concatenate(sub_object_data, axis=0)
        merged_muscle = np.concatenate(muscle_coactivation_data, axis=0)

        print(f"合并后的数据形状: sub_object={merged_sub_object.shape}, muscle={merged_muscle.shape}")

        # 保存合并后的数据
        output_sub_obj_path = os.path.join(output_folder, "sub_object_all.npy")
        output_muscle_path = os.path.join(output_folder, "muscle_coactivation_all.npy")

        np.save(output_sub_obj_path, merged_sub_object)
        np.save(output_muscle_path, merged_muscle)

        print(f"数据已合并并保存到: {output_folder}")
        print(f"- {output_sub_obj_path}")
        print(f"- {output_muscle_path}")

    except Exception as e:
        print(f"合并或保存数据时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并多个文件夹中的numpy数据文件')
    parser.add_argument('--base_folder', type=str, required=True,
                        help='包含所有数据子文件夹的基础文件夹路径')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='输出合并后数据的文件夹路径')

    args = parser.parse_args()

    merge_data_files(args.base_folder, args.output_folder)