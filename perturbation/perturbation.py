import random
import numpy as np
import matplotlib.pyplot as plt
import transformation as tsf

def random_frequency():
    frequency = []
    time = 0.0
    while time < 30:
        random_number = random.randint(1,10)
        frequency = np.append(np.array(frequency), random_number)
        time = time + 1.0 / random_number
    return frequency

def random_displacement(data):
    displacement = []
    for i in range(len(data)):
        random_dis = np.array([random.randint(-20,20), random.randint(-20,20), random.randint(-20,20)])
        displacement = np.append(np.array(displacement), random_dis / 1000)
    return displacement

def generate_trajectory(rf, rd, initial_pose):
    n = int()
    trajectory = np.tile(initial_pose, 30000).reshape(-1, 7)
    for i in range(len(rf) - 1):
        n = n + int(1000 / int(rf[i]))
        for j in range(20):
            trajectory[n + j, :3] = trajectory[n + j, :3] + rd[i, :]
    return trajectory


if __name__ == '__main__':
    rf = random_frequency()
    # print(rf, len(rf))
    rd = random_displacement(rf).reshape((len(rf)), -1)
    robot_to_left, robot_to_right = tsf.transform_robot_base_to_arm_base(np.array([0, 0, 0]))
    for i in range(len(rf)):
        rd[i, :] = robot_to_right[:3, :3] @ rd[i, :]
    # print(rd, len(rd))
    # initial_pose = np.array([0.48968171, 0.38453146, 0.46961821, 0.76166407, -0.08329541, 0.56650357, 0.30332065])
    initial_pose = np.array([0.26523754, 0.42030248, 0.50026908, 0.76166407, -0.08329541, 0.56650357, 0.30332065])
    # trajectory =  np.tile(initial_pose, (30000, 1)).reshape((7, 1, 30000))

    trajectory = generate_trajectory(rf, rd, initial_pose)

    np.savetxt('perturbation_high_position.txt', trajectory, delimiter=',', fmt='%.08f')

    plt.plot(trajectory[:, 0])
    plt.plot(trajectory[:, 1])
    plt.plot(trajectory[:, 2])
    plt.show()