import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


'''
读取扰动顺序文件
'''
# <editor-fold>
# k = np.loadtxt('/home/lee/PycharmProjects/TASE/data/20240124/k_0124.txt')

'''
0125
'''

# </editor-fold>

'''
读取位置和力数据
'''
# <editor-fold>
# force = pd.read_csv('/home/lee/PycharmProjects/TASE/data/20240124/0124_4/force.csv')
# position = pd.read_csv('/home/lee/PycharmProjects/TASE/data/20240124/0124_4/position.csv')

# force = pd.read_csv('/home/lee/PycharmProjects/TASE/data/20240124/0124_3/force.csv')
# position = pd.read_csv('/home/lee/PycharmProjects/TASE/data/20240124/0124_3/position.csv')

# force = pd.read_csv('/home/lee/PycharmProjects/TASE/data/20240124/0124_2/force.csv')
# position = pd.read_csv('/home/lee/PycharmProjects/TASE/data/20240124/0124_2/position.csv')

'''
0125
'''
k = np.loadtxt('/home/lee/PycharmProjects/TASE/data/20240126/trajectory/k_0126_63.txt')
force = pd.read_csv('/home/lee/PycharmProjects/TASE/data/20240126/rosbag/0126_63/force.csv')
position = pd.read_csv('/home/lee/PycharmProjects/TASE/data/20240126/rosbag/0126_63/position.csv')
# </editor-fold>

'''
切片
'''

# force = np.array(force)[1326:, 1:]
# position = np.array(position)[1326:, 1:]

# force = np.array(force)[1536:, 1:]
# position = np.array(position)[1536:, 1:]

# force = np.array(force)[1333:, 1:]
# position = np.array(position)[1333:, 1:]

# <editor-fold>
'''
0125_1
'''
# force = np.array(force)[1838:, 1:]
# position = np.array(position)[1838:, 1:]
# force = np.array(force)[1298:, 1:]
# position = np.array(position)[1298:, 1:]
# force = np.array(force)[1173:, 1:]
# position = np.array(position)[1173:, 1:]
'''
0125_2
'''
# force = np.array(force)[842:, 1:]
# position = np.array(position)[842:, 1:]
# force = np.array(force)[876:, 1:]
# position = np.array(position)[876:, 1:]
# force = np.array(force)[934:, 1:]
# position = np.array(position)[934:, 1:]
'''
0125_3
'''
# force = np.array(force)[844:, 1:]
# position = np.array(position)[844:, 1:]
# force = np.array(force)[:, 1:]
# position = np.array(position)[:, 1:]
# force = np.array(force)[893:, 1:]
# position = np.array(position)[893:, 1:]
'''
0125_4
'''
# force = np.array(force)[2289:, 1:]
# position = np.array(position)[2289:, 1:]
# force = np.array(force)[1077:, 1:]
# position = np.array(position)[1077:, 1:]
# force = np.array(force)[902:, 1:]
# position = np.array(position)[902:, 1:]
'''
0125_5
'''
# force = np.array(force)[1215:, 1:]
# position = np.array(position)[1215:, 1:]
# force = np.array(force)[828:, 1:]
# position = np.array(position)[828:, 1:]
# force = np.array(force)[889:, 1:]
# position = np.array(position)[889:, 1:]
'''
0125_6
'''
# force = np.array(force)[785:, 1:]
# position = np.array(position)[785:, 1:]
# force = np.array(force)[892:, 1:]
# position = np.array(position)[892:, 1:]
# force = np.array(force)[1142:, 1:]
# position = np.array(position)[1142:, 1:]
'''
0125_7
'''
# force = np.array(force)[1930:, 1:]
# position = np.array(position)[1930:, 1:]
# force = np.array(force)[892:, 1:]
# position = np.array(position)[892:, 1:]
# force = np.array(force)[:, 1:]
# position = np.array(position)[:, 1:]
# </editor-fold>

# <editor-fold>
'''
0126_1
'''
# force = np.array(force)[1326:, 1:]
# position = np.array(position)[1326:, 1:]
# force = np.array(force)[1513:, 1:]
# position = np.array(position)[1513:, 1:]
# force = np.array(force)[1960:, 1:]
# position = np.array(position)[1960:, 1:]
'''
0126_2
'''
# force = np.array(force)[2775:, 1:]
# position = np.array(position)[2775:, 1:]
# force = np.array(force)[1682:, 1:]
# position = np.array(position)[1682:, 1:]
# force = np.array(force)[1198:, 1:]
# position = np.array(position)[1198:, 1:]
'''
0126_3
'''
# force = np.array(force)[1212:, 1:]
# position = np.array(position)[1212:, 1:]
# force = np.array(force)[1495:, 1:]
# position = np.array(position)[1495:, 1:]
# force = np.array(force)[895:, 1:]
# position = np.array(position)[895:, 1:]
'''
0126_4
'''
# force = np.array(force)[1563:, 1:]
# position = np.array(position)[1563:, 1:]
# force = np.array(force)[2240:, 1:]
# position = np.array(position)[2240:, 1:]
# force = np.array(force)[1370:, 1:]
# position = np.array(position)[1370:, 1:]
'''
0126_5
'''
# force = np.array(force)[1055:, 1:]
# position = np.array(position)[1055:, 1:]
# force = np.array(force)[972:, 1:]
# position = np.array(position)[972:, 1:]
# force = np.array(force)[790:, 1:]
# position = np.array(position)[790:, 1:]
'''
0126_6
'''
# force = np.array(force)[978:, 1:]
# position = np.array(position)[978:, 1:]
# force = np.array(force)[1125:, 1:]
# position = np.array(position)[1125:, 1:]
force = np.array(force)[1410:, 1:]
position = np.array(position)[1410:, 1:]
# </editor-fold>

index = 25

## 平滑处理
for i in range(3):
    force[:, i] = moving_average(force[:, i], 20)

for i in range(len(k)):
    if k[i] == 6:
        ## direction_yz
        position_y = position[i * 350:i * 350 + index, 1] - position[i * 350, 1]
        position_z = position[i * 350:i * 350 + index, 2] - position[i * 350, 2]
        force_y = force[i:(i + 1) * index, 1] - force[i * 350, 1]
        force_z = force[i:(i + 1) * index, 2] - force[i * 350, 2]
        # K_y = math.sqrt(np.mean([x**2 for x in force_y])) / math.sqrt(np.mean([x**2 for x in position_y]))
        # K_z = math.sqrt(np.mean([x ** 2 for x in force_z])) / math.sqrt(np.mean([x ** 2 for x in position_z]))
        # K_yz = math.sqrt(K_y ** 2 + K_z ** 2)
        K_yz = math.sqrt(np.mean([x ** 2 for x in force_y]) + np.mean([x ** 2 for x in force_z])) / math.sqrt(np.mean([x ** 2 for x in position_y]) + np.mean([x ** 2 for x in position_z]))
        # print('K_yz:', K_yz)
    if k[i] == 5:
        ## direction_xz
        position_x = position[i * 350:i * 350 + index, 0] - position[i * 350, 0]
        position_z = position[i * 350:i * 350 + index, 2] - position[i * 350, 2]
        force_x = force[i * 350:i * 350 + index, 0] - force[i * 350, 0]
        force_z = force[i * 350:i * 350 + index, 2] - force[i * 350, 2]
        # K_x = math.sqrt(np.mean([x**2 for x in force_x])) / math.sqrt(np.mean([x**2 for x in position_x]))
        # K_z = math.sqrt(np.mean([x ** 2 for x in force_z])) / math.sqrt(np.mean([x ** 2 for x in position_z]))
        # K_xz = math.sqrt(K_x ** 2 + K_z ** 2)
        K_xz = math.sqrt(np.mean([x ** 2 for x in force_x]) + np.mean([x ** 2 for x in force_z])) / math.sqrt(
            np.mean([x ** 2 for x in position_x]) + np.mean([x ** 2 for x in position_z]))
        print('K_xz:', K_xz)
    if k[i] == 4:
        ## direction_xy
        position_x = position[i * 350:i * 350 + index, 0] - position[i * 350, 0]
        position_y = position[i * 350:i * 350 + index, 1] - position[i * 350, 1]
        force_x = force[i * 350:i * 350 + index, 0] - force[i * 350, 0]
        force_y = force[i * 350:i * 350 + index, 1] - force[i * 350, 1]
        # K_x = math.sqrt(np.mean([x**2 for x in force_x])) / math.sqrt(np.mean([x**2 for x in position_x]))
        # K_y = math.sqrt(np.mean([x ** 2 for x in force_y])) / math.sqrt(np.mean([x ** 2 for x in position_y]))
        # K_xy = math.sqrt(K_x ** 2 + K_y ** 2)
        K_xy = math.sqrt(np.mean([x ** 2 for x in force_x]) + np.mean([x ** 2 for x in force_y])) / math.sqrt(
            np.mean([x ** 2 for x in position_x]) + np.mean([x ** 2 for x in position_y]))
        # print('K_xy:', K_xy)
    if k[i] == 3:
        ## direction_z
        position_z = position[i * 350:i * 350 + index, 2] - position[i * 350, 2]
        force_z = force[i * 350:i * 350 + index, 2] - force[i * 350, 2]
        K_z = math.sqrt(np.mean([x ** 2 for x in force_z])) / math.sqrt(np.mean([x ** 2 for x in position_z]))
        K_zz = K_z
        # print('K_zz:', K_zz)
    if k[i] == 2:
        ## direction_y
        position_y = position[i * 350:i * 350 + index, 1] - position[i * 350, 1]
        force_y = force[i * 350:i * 350 + index, 1] - force[i * 350, 1]
        K_y = math.sqrt(np.mean([x**2 for x in force_y])) / math.sqrt(np.mean([x**2 for x in position_y]))
        K_yy = K_y
        # print('K_yy:', K_yy)
    if k[i] == 1:
        ## direction_x
        position_x = position[i * 350:i * 350 + index, 0] - position[i * 350, 0]
        force_x = force[i * 350:i * 350 + index, 0] - force[i * 350, 0]
        K_x = math.sqrt(np.mean([x**2 for x in force_x])) / math.sqrt(np.mean([x**2 for x in position_x]))
        K_xx = K_x
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
plt.show()

