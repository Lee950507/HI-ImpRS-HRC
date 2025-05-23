import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution


def calculate_endpoint_stiffness(pose):
    pos_shoulder = pose[4:7] / 100
    pos_elbow = pose[11:14] / 100
    pos_wrist = pose[18:21] / 100

    l = np.zeros(3)
    r = np.zeros(3)
    l_init = pos_wrist - pos_shoulder
    l[0] = - l_init[0]
    l[1] = - l_init[2]
    l[2] = l_init[1]
    r_init = pos_elbow - pos_shoulder
    r[0] = - r_init[0]
    r[1] = - r_init[2]
    r[2] = r_init[1]

    V_1 = l / np.linalg.norm(l)
    V_2 = np.cross(np.cross(l, r), l) / np.linalg.norm(np.cross(np.cross(l, r), l))
    V_3 = np.cross(l, r) / np.linalg.norm(np.cross(l, r))

    V = np.array([V_1, V_2, V_3]).T
    # print('V1:', V_1, 'V2:', V_2, 'V3:', V_3, V)

    a1 = 1000
    a2 = 500
    d1 = np.linalg.norm(l)
    d2 = np.linalg.norm(np.dot(r, np.cross(np.cross(l, r), l) / np.linalg.norm(np.cross(np.cross(l, r), l))))
    D_s = np.diag([1, a1 / d1, a2 * d2])
    # print('D_s:', D_s)

    b1 = 1
    b2 = 0.5
    A = 0.5
    Acc = b1 * A + b2

    K_e = Acc * V @ D_s @ V.T
    return V, d1, d2, K_e


def optimization(subject, a=0, b=0, c=0.0, d=0.0):
    if subject == 'chenzui':
        pose = pd.read_csv('/home/ubuntu/HI-ImpRS-HRC/perturbation/previous_data/20240125/optitrack/preprocessing.csv')
        K1 = [22.37072355, 261.61388677, 174.3714698, 73.58384043, 41.07329286, 161.36242025,
              31.97132417, 430.58941431, 291.27157732, 143.80840144, 143.26952798, 372.05486255,
              56.24579194, 750.3514475, 544.96926576, 207.74414058, 219.75448379, 644.17839718]
        K2 = [85.17666143, 54.909586, 78.75859291, 74.56925995, 82.73426264, 63.09849035,
              279.76237223, 222.40761673, 182.3224916, 241.30466251, 224.11974617, 205.80661002,
              596.86806685, 437.3084341, 448.69244951, 525.79855854, 534.8045152, 416.95339368]
        K3 = [174.38130339, 10.54757784, 169.09048966, 40.69256083, 170.01057602, 50.05351053,
              551.11788482, 36.03965158, 441.38537654, 114.6478351, 633.67583106, 111.40418666]
        K4 = [29.01251098, 228.87347439, 75.05260845, 97.42470852, 52.6003956, 163.72295357,
              65.72308391, 678.27958607, 279.25142207, 172.35167274, 200.0, 417.12779789,
              233.08477223, 1185.38142, 467.33900796, 381.10891086, 451.49427104, 789.33609579]
        K5 = [225.13815383, 55.81895256, 0., 0., 115.41787363, 226.43922311,
              0., 134.52048633, 0., 651.47299651, 1111.76998654, 468.52968654,
              205.4380525, 993.73660265, 476.08048206, 458.23220674, 592.03620721, 875.12075197]
        K6 = [149.0827872, 89.56239073, 61.87284189, 100.07963389, 86.1933544, 72.40773097,
              314.63060926, 178.0, 140.7372492, 280.0, 163.43421027, 158.45533302,
              501.74322885,  391.35996303,  500.30525341, 475.40135934, 494.71679052, 427.11043628]
        K7 = [342.84229248, 33.24536854, 168.39798055, 180.04056678, 252.19884777, 64.61184098]
        a7 = [0.00462, 0.00702, 0.01389]
        a3 = [0.01343, 0.04518, 0.01714]
        a1 = [0.01756, 0.01251, 0.05716]
        a5 = [0.0057, 0.08436, 0.16725]
        a2 = [0.01243, 0.01488, 0.0294]
        a6 = [0.00532, 0.01294, 0.03716]
        a4 = [0.02193, 0.01873, 0.13736]
        # a1 = a2 = a3 = a4 = a5 = a6 = [0.02, 0.1, 0.2]

        pose = np.array(pose)[:, :21]
        Va = [[] for i in range(7)]
        d1a = [[] for i in range(7)]
        d2a = [[] for i in range(7)]
        for i in range(7):
            V, d1, d2, K_e = calculate_endpoint_stiffness(pose[i, :])
            # print("V:", V, '\n', "d1:", d1, "d2:", d2)
            Va[i] = V
            d1a[i] = d1
            d2a[i] = d2
        v0125 = [7, 3, 1, 5, 2, 6, 4]
        V = np.asarray([Va[2], Va[4], Va[1], Va[6], Va[5]])
        d1 = np.asarray([d1a[2], d1a[4], d1a[1], d1a[6], d1a[5]])
        d2 = np.asarray([d2a[2], d2a[4], d2a[1], d2a[6], d2a[5]])
        A = np.asarray([a1, a2, a3, a4, a6])
        K = [K1, K2, K3, K4, K6]
        ki = [3, 3, 2, 3, 3]
        Ksh = [np.asarray([[[K[j][i * 6], K[j][i * 6 + 3], K[j][i * 6 + 4]],
                            [K[j][i * 6 + 3], K[j][i * 6 + 1], K[j][i * 6 + 5]],
                            [K[j][i * 6 + 4], K[j][i * 6 + 5], K[j][i * 6 + 2]]] for i in range(ki[j])]) for j in
               range(len(ki))]
    else:
        pose = pd.read_csv('/home/ubuntu/HI-ImpRS-HRC/perturbation/previous_data/20240126/optitrack/preprocessing.csv')
        K1 = [87.47932976, 15.69633818, 108.44843445, 62.33777144, 91.411043, 79.70420181,
              281.24388925, 30.24204249, 416.27283076,  88.13323483, 303.07308419, 170.0811301,
              355.23908327, 106.24818433, 608.59204032, 253.94362759, 532.26161735, 206.04344883]
        K2 = [155.01034935, 63.09601177, 79.77450085, 83.9870611, 107.75203188, 61.18903773,
              300.09039393, 78.54903971, 198.71003156, 143.22922903, 228.30216386, 96.11567507,
              698.02436228, 171.87584924, 280.21287478, 206.51981597, 484.24122781, 232.83205402]
        K3 = [18.21098986, 117.32074604, 122.00063622, 80.3329412, 77.70840043, 196.66469945,
              34.98574087, 248.07397099, 275.86122405, 154.83559084, 136.67550923, 274.41395977,
              84.09753902, 498.4060566, 499.20845245, 206.20713831, 195.84167944, 573.17565578]
        K4 = [115.80226206, 106.4903708, 160.16872646, 106.83539038, 151.69815877, 162.38041356,
              208.77556759, 214.52709943, 448.35576876, 236.10333358, 381.27967243, 271.74654383,
              243.38451531, 261.09759238, 638.06352437, 211.18867416, 336.77427678, 494.74974675]
        K5 = [33.92809251, 1.83665002, 113.13313281, 21.77617671, 90.53355258, 78.76139629,
              166.58277863, 25.14844205, 305.9264234, 104.34726407, 170.33802069, 150.34873622,
              210.22067127, 59.87090166, 1300.80001921, 130.27429936, 622.32501323, 559.521362]
        K6 = [129.97538255, 7.80877673, 174.69153621, 40.74791927, 136.15390556, 33.49862018,
              369.61073238, 36.11075783, 298.6266495, 152.70007062, 222.91283516, 114.07885143,
              539.86878636, 79.40854803, 374.2289232, 205.01316145, 361.62253307, 188.30276121]
        a1 = [0.02375, 0.05776, 0.08785]
        a2 = [0.04548, 0.04314, 0.07682]
        a3 = [0.02772, 0.05061, 0.08463]
        a4 = [0.0292, 0.0367, 0.0702]
        a5 = [0.02231, 0.06368, 0.06995]
        a6 = [0.03497, 0.04379, 0.07196]
        # a1 = a2 = a3 = a4 = a5 = a6 = [0.02, 0.1, 0.2]

        pose = np.array(pose)[:, :21]
        Va = [[] for i in range(6)]
        d1a = [[] for i in range(6)]
        d2a = [[] for i in range(6)]
        for i in range(6):
            V, d1, d2, K_e = calculate_endpoint_stiffness(pose[i, :])
            # print("V:", V, '\n', "d1:", d1, "d2:", d2)
            Va[i] = V
            d1a[i] = d1
            d2a[i] = d2
        v0126 = [1, 3, 5, 2, 4, 6]
        V = np.asarray([Va[0], Va[3], Va[1], Va[4], Va[2], Va[5]])
        d1 = np.asarray([d1a[0], d1a[3], d1a[1], d1a[4], d1a[2], d1a[5]])
        d2 = np.asarray([d2a[0], d2a[3], d2a[1], d2a[4], d2a[2], d2a[5]])
        A = np.asarray([a1, a2, a3, a4, a5, a6])
        K = [K1, K2, K3, K4, K5, K6]
        ki = [3, 3, 3, 3, 3, 3]
        Ksh = [np.asarray([[[K[j][i * 6], K[j][i * 6 + 3], K[j][i * 6 + 4]],
                            [K[j][i * 6 + 3], K[j][i * 6 + 1], K[j][i * 6 + 5]],
                            [K[j][i * 6 + 4], K[j][i * 6 + 5], K[j][i * 6 + 2]]] for i in range(ki[j])]) for j in
               range(len(ki))]

    plot_data = []

    def objective_function(x):
        obj = 0
        for j in range(len(ki)):
            # ds = np.power(abs(x[0] * x[1] * d2[j] / d1[j]), -1 / 3) * np.asarray([[1, 0, 0],
            #                                                                       [0, x[0] / d1[j], 0],
            #                                                                       [0, 0, x[1] * d2[j]]])
            ds =  np.asarray([[1, 0, 0],
                              [0, x[0] / d1[j], 0],
                              [0, 0, x[1] * d2[j]]])

            vdvt = abs(V[j, :, :] @ ds @ V[j, :, :].T)
            for i in range(ki[j]):
                log1 = np.log(vdvt * (x[2] * A[j, i] + x[3]))
                log2 = np.log(Ksh[j][i, :, :])
                # print(np.linalg.norm(log1 - log2))
                obj = obj + np.linalg.norm(log1 - log2)
        # print(obj)
        plot_data.append(obj)
        return obj

    def constraint(x):
        return [x[0], x[1], x[2], x[3]]
        # if x[2] >= 0 and x[3] >= 0:
        #     return 0
        # if x[2] >= 0 >= x[3]:
        #     return -x[3]
        # if x[3] >= 0 >= x[2]:
        #     return -x[2]
        # if x[2] <= 0 and x[3] <= 0:
        #     return -x[2] - x[3]

    constraint_definition = {'type': 'ineq', 'fun': constraint}
    # initial_guess = np.asarray([a, b, c, d])
    # initial_guess = np.asarray([2000, 150, 3, 5])
    # initial_guess = np.asarray([2000, 150, 3, 6])
    # initial_guess = np.asarray([2000, 150, 0.01, 0.5])
    # initial_guess = np.asarray([2000, 150, 0.2, 0.3])
    initial_guess = np.asarray([0.5, 2, 1500, 100])
    # initial_guess = np.asarray([[0.2, 0.3, 2000, 150], [1000, 100, 0.2, 3], [4000, 150, 0.2, 1.8]])
    # init = [2000, 150, 0.2, 0.3]
    bounds = [(0.1, 5), (0.1, 5), (0.1, 4000), (0.1, 500)]
    result = minimize(objective_function, initial_guess, bounds=bounds, method='SLSQP')
    # result = minimize(objective_function, initial_guess, method='nelder-mead')
    # result = minimize(objective_function, initial_guess, constraints=constraint_definition)
    # result = differential_evolution(objective_function, bounds)
    # result = differential_evolution(objective_function, bounds, init=init)

    min_value = result.fun
    optimal_point = result.x

    print('-' * 25, subject, '-' * 25)
    # print("最小值：", min_value)
    print("c1：", optimal_point[0])
    print("c2：", optimal_point[1])
    print("alpha1：", optimal_point[2])
    print("alpha2：", optimal_point[3])
    plt.figure()
    plt.plot(plot_data, label=subject)
    plt.legend()
    return plot_data[-1]


if __name__ == '__main__':
    # pose = pd.read_csv('/home/lee/PycharmProjects/TASE/data/20240124/processing.csv')
    # pose = pd.read_csv('/home/lee/PycharmProjects/TASE/data/20240126/optitrack/preprocessing.csv')
    # pose = np.array(pose)[:, :21]
    # for i in range(7):
    #     V, d1, d2, K_e = calculate_endpoint_stiffness(pose[i, :])
    #     print("V:", V, '\n', "d1:", d1, "d2:", d2)

    # minr = 100000
    # mina = 0
    # minb = 0
    # minc = 0
    # mind = 0
    # for d in range(100, 600, 2):
    #     for c in range(1, 400, 2):
    #         for b in range(50, 500, 2):
    #             for a in range(1000, 4000, 10):
    #                 subject = 'chenzui'
    #                 target1 = optimization(subject, a, b, c*0.01, d*0.01)
    #
    #                 subject = 'lizhuo'
    #                 target2 = optimization(subject, a, b, c*0.01, d*0.01)
    #
    #                 if target1 + target2 <= minr:
    #                     minr = target1 + target2
    #                     mina, minb, minc, mind = a, b, c, d
    # print(minr, mina, minb, minc, mind)

    subject = 'chenzui'
    target1 = optimization(subject)

    subject = 'lizhuo'
    target2 = optimization(subject)
    plt.show()


