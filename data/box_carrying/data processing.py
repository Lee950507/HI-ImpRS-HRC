import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


stiff = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/taichi/stiffness_results_bi/Ke_right_set5.npy')
stiff_ref = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/taichi/bi_6900/stiff_taichi_bi_r_6900.npy')

print(f"Original stiff shape: {stiff.shape}")
print(f"Reference stiff_ref shape: {stiff_ref.shape}")

original_time = np.arange(stiff.shape[0])
target_time = np.linspace(0, stiff.shape[0] - 1, stiff_ref.shape[0])

# 为每个轴创建插值函数
stiff_interp = np.zeros((stiff_ref.shape[0], stiff.shape[1]))
for axis in range(stiff.shape[1]):
    # 创建三次样条插值函数
    f = interpolate.interp1d(original_time, stiff[:, axis], kind='cubic', bounds_error=False, fill_value='extrapolate')

    # 应用插值
    stiff_interp[:, axis] = f(target_time)

np.save('/home/clover/Chenzui/HI-ImpRS-HRC/data/taichi/stiffness_results_bi/stiff_wuxi_r_6900.npy', stiff_interp)

plt.figure(figsize=(14, 8))

# 绘制X轴刚度比较
plt.subplot(3, 1, 1)
plt.plot(stiff_ref[:, 0], 'b-', label='Reference X-axis', alpha=0.7)
plt.plot(stiff_interp[:, 0], 'r-', label='Interpolated X-axis')
plt.legend()
plt.title('X-axis Stiffness Comparison')
plt.grid(True, alpha=0.3)

# 绘制Y轴刚度比较
plt.subplot(3, 1, 2)
plt.plot(stiff_ref[:, 1], 'b-', label='Reference Y-axis', alpha=0.7)
plt.plot(stiff_interp[:, 1], 'g-', label='Interpolated Y-axis')
plt.legend()
plt.title('Y-axis Stiffness Comparison')
plt.grid(True, alpha=0.3)

# 绘制Z轴刚度比较
plt.subplot(3, 1, 3)
plt.plot(stiff_ref[:, 2], 'b-', label='Reference Z-axis', alpha=0.7)
plt.plot(stiff_interp[:, 2], 'c-', label='Interpolated Z-axis')
plt.legend()
plt.title('Z-axis Stiffness Comparison')
plt.xlabel('Time steps')
plt.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('/home/ubuntu/HI-ImpRS-HRC/data/box_carrying/stiffness_results_uni/stiffness_interpolation_comparison.png', dpi=300)
plt.show()