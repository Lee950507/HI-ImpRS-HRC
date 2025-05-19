import random
import numpy as np
import matplotlib.pyplot as plt
import transformation as tsf
import math


def minimum_jerk_trajectory(init_pos, target_pos, total_time=0.5, dt=0.001):
    xi = init_pos
    xf = target_pos
    d = total_time
    list_t = []
    list_x = []
    t = 0
    while t < d:
        x = xi + (xf-xi) * (10*(t/d)**3 - 15*(t/d)**4 + 6*(t/d)**5)
        list_t.append(t)
        list_x.append(x)
        t += dt
    return np.array(list_x)


def generate_perturb(data):
    my_traj_up = minimum_jerk_trajectory(0, 0.02)
    my_traj_mid = np.tile(0.02, 3000)
    my_traj_down = minimum_jerk_trajectory(0.02, 0)
    my_traj = np.append(my_traj_up, my_traj_mid)
    my_traj = np.append(my_traj, my_traj_down)

    perturb = np.zeros((4000, 3))
    if data == 1:
        perturb[:, 0] = my_traj
    if data == 2:
        perturb[:, 1] = my_traj
    if data == 3:
        perturb[:, 2] = my_traj
    if data == 4:
        perturb[:, 0] = math.sqrt(2) * my_traj / 2
        perturb[:, 1] = math.sqrt(2) * my_traj / 2
    if data == 5:
        perturb[:, 0] = math.sqrt(2) * my_traj / 2
        perturb[:, 2] = math.sqrt(2) * my_traj / 2
    if data == 6:
        perturb[:, 1] = math.sqrt(2) * my_traj / 2
        perturb[:, 2] = math.sqrt(2) * my_traj / 2

    T_l, T_r =  tsf.transform_robot_base_to_arm_base(np.array([0, 0, 0]))
    for i in range(len(perturb[: ,1])):
        perturb[i, :] = np.linalg.inv(T_r[:3, :3]) @ perturb[i, :]

    return perturb


def generate_trajectory(initial_pose):
    t = np.array([7, 14, 21, 28, 35, 42])
    k = np.array([1, 2, 3, 4, 5, 6])
    np.random.shuffle(k)
    trajectory = np.tile(initial_pose, 50000).reshape(-1, 7)
    for i in range(len(t)):
        n = t[i] * 1000
        trajectory[n:n+4000, :3] = trajectory[n:n+4000, :3] + generate_perturb(k[i])

    return k, trajectory


if __name__ == '__main__':
    # initial_pose = np.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
    # initial_pose = np.array([0.26523754, 0.42030248, 0.50026908, 0.76166407, -0.08329541, 0.56650357, 0.30332065])
    initial_pose = np.array([0.48968172, 0.38453146, 0.46961821, 0.76166407, -0.08329541, 0.56650357, 0.30332065])
    k, trajectory = generate_trajectory(initial_pose)

    np.savetxt('trajectory_txt/0126/perturbation_0126_61.txt', trajectory, delimiter=',', fmt='%.08f')
    np.savetxt('trajectory_txt/0126/k_0126_61.txt', k, delimiter=',', fmt='%.08f')

    plt.plot(trajectory[:, 0])
    plt.plot(trajectory[:, 1])
    plt.plot(trajectory[:, 2])
    # plt.show()

