"""
TP-GMM for box-carrying
"""
import numpy as np
import matplotlib.pyplot as plt
import RofuncML
import pandas as pd


save_params = {'save_dir': '/home/clover/Chenzui/HI-ImpRS-HRC/data/box_carrying/fig', 'format': ['png']}

raw_demo1 = np.array(pd.read_csv('/home/clover/Chenzui/HI-ImpRS-HRC/data/box_carrying/preprocess/1/processing_051602.csv')) / 100
raw_demo2 = np.array(pd.read_csv('/home/clover/Chenzui/HI-ImpRS-HRC/data/box_carrying/preprocess/2/processing_051748.csv')) / 100
raw_demo3 = np.array(pd.read_csv('/home/clover/Chenzui/HI-ImpRS-HRC/data/box_carrying/preprocess/3/processing_051907.csv')) / 100
stiffness_demo1 = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/box_carrying/preprocess/1/Ke_zhuo.npy') / 1000
stiffness_demo2 = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/box_carrying/preprocess/2/Ke_zhuo.npy') / 1000
stiffness_demo3 = np.load('/home/clover/Chenzui/HI-ImpRS-HRC/data/box_carrying/preprocess/3/Ke_zhuo.npy') / 1000
demos_x = [np.hstack((np.hstack((raw_demo1[2539:4380:10, 19:22], raw_demo1[2539:4380:10, 15:19])), stiffness_demo1[::10])), np.hstack((np.hstack((raw_demo2[1400:2900:10, 19:22], raw_demo2[1400:2900:10, 15:19])), stiffness_demo2[::10])), np.hstack((np.hstack((raw_demo3[1350:2850:10, 19:22], raw_demo3[1350:2850:10, 15:19])), stiffness_demo3[::10]))]

# demos_x = [np.hstack((np.hstack((raw_demo1[2539:4380:10, 40:43], raw_demo1[2539:4380:10, 36:40])), stiffness_demo1[::10])), np.hstack((np.hstack((raw_demo2[1400:2900:10, 40:43], raw_demo2[1400:2900:10, 36:40])), stiffness_demo2[::10])), np.hstack((np.hstack((raw_demo3[1350:2850:10, 40:43], raw_demo3[1350:2850:10, 36:40])), stiffness_demo3[::10]))]

# demos_x = [raw_demo1[2539:4380, 19:26], raw_demo2[1400:2900, 19:26], raw_demo3[1350:2850, 19:26]]


# --- TP-GMM ---
# Define the task parameters
start_xdx = [demos_x[i][0] for i in range(len(demos_x))]  # TODO: change to xdx
end_xdx = [demos_x[i][-1] for i in range(len(demos_x))]
task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
# Fit the model
Repr = RofuncML.TPGMM(demos_x, task_params, plot=True, save=True, save_params=save_params)
model = Repr.fit()

# Reproductions for the same situations
# traj, _ = Repr.reproduce(model, show_demo_idx=2)

# Reproductions for new situations: set the endpoint as the start point to make a cycled motion
ref_demo_idx = 2
start_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
end_xdx = [Repr.demos_xdx[ref_demo_idx][-1]]
end_xdx[0][0] += 0.1
Repr.task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
traj, _ = Repr.generate(model, ref_demo_idx)

np.save('/home/clover/Chenzui/HI-ImpRS-HRC/data/box_carrying/fig/traj_w_stiff_zhuo.npy', traj)
