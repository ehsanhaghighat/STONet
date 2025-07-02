"""
STONet: A Neural Operator for Modeling Solute Transport in Micro-Cracked Reservoirs

This code is part of the STONet repository: https://github.com/ehsanhaghighat/STONet

Citation:
@article{haghighat2024stonet,
  title={STONet: A neural operator for modeling solute transport in micro-cracked reservoirs},
  author={Haghighat, Ehsan and Adeli, Mohammad Hesan and Mousavi, S Mohammad and Juanes, Ruben},
  journal={arXiv preprint arXiv:2412.05576},
  year={2024}
}

Paper: https://arxiv.org/abs/2412.05576
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
import time
import imageio

import torch
from src.networks import MLP, EnrichedDeepONet, Fourier, STONet
from src.optimizers import Optimizer
from src.losses import Loss
from src.data_models import DeepONetDataModel
from src.utils import *
import logging
FORMAT = '[%(asctime)s %(levelname)s %(process)d %(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)



df_data = pd.read_csv('data/data500_train.csv', delimiter=',')
df_data['prob'] = 1.
print(df_data.keys())
print(df_data)


sample_ids = df_data['sample'].unique().astype(int)
sample_sizes = [len(df_data[df_data['sample'] == i]) for i in df_data['sample'].unique()]
assert len(set(sample_sizes)) == 1, 'The number of samples must be the same for all samples.'
sample_size_per_sim = sample_sizes[0]
print('sample_size_per_sim:', sample_size_per_sim)

time_steps = list(sorted(df_data['t'].unique()))
print('time_steps:', time_steps)

batches = {}
sample_per_time_step = []
for sid in sample_ids:
    print('sid:', sid)
    sample_data = df_data[df_data['sample'] == sid].reset_index(drop=True)
    batches[sid] = {}
    for t in time_steps:
        batch = sample_data[sample_data['t'] == t]
        if len(batch) == 0: continue
        batches[sid][t] = batch.reset_index(drop=True)
        sample_per_time_step.append(len(batches[sid][t]))

assert len(set(sample_per_time_step)) == 1, 'The number of samples must be the same for all time steps.'
print('sample_per_time_step:', set(sample_per_time_step))
sample_per_time_step = sample_per_time_step[0]

# ['sample', 'x', 'y', 'dp', 'k11', 'k12', 'k22', 'p', 'c', 'deltaC', 'vx', 'vy', 'orient', 'number', 't', 'cdot'],
output_vars = ['dp', 'k11', 'k12', 'k22', 'orient', 'p', 'vx', 'vy', 'c', 'cdot', 'deltaC']
paper_fig_vars = ['k11', 'k22', 'c', 'cdot']
paper_fig_symbols = ['$k_{11}$', '$k_{22}$', '$c$', '$\dot{c}$']
paper_fig_cmap = ['gray', 'gray', 'seismic', 'seismic']

batch_paths = {}
for batch in batches:
    batch_paths[batch] = {}
    batch_path = f'data/plots/sample-{batch}'
    for t in batches[batch]:
        batch_paths[batch][t] = f'{batch_path}/time-{t}'
        os.makedirs(batch_paths[batch][t], exist_ok=True)
        batches[batch][t].to_csv(f'{batch_paths[batch][t]}/data.csv', index=False)
        for var in output_vars:
            export_path = f'{batch_paths[batch][t]}/{var}.png'
            xs = batches[batch][t]['x']
            ys = batches[batch][t]['y']
            zs = batches[batch][t][var]
            # interpolate data on grid
            xi = np.linspace(xs.min(), xs.max(), 100)
            yi = np.linspace(ys.min(), ys.max(), 100)
            zi = griddata((xs, ys), zs, (xi[None,:], yi[:,None]), method='linear')
            # plot
            plt.figure()
            plt.pcolor(xi, yi, zi, cmap='jet')
            plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f"{batch}: t={t}   --   {var}")
            plt.savefig(export_path)
            plt.close()
        fig, ax = plt.subplots(1, 4, figsize=(18, 3))
        export_path = f'{batch_paths[batch][t]}/fig_vars.png'
        for i, (var, cmap, symbol) in enumerate(zip(paper_fig_vars, paper_fig_cmap, paper_fig_symbols)):
            xs = batches[batch][t]['x']
            ys = batches[batch][t]['y']
            zs = batches[batch][t][var]
            # interpolate data on grid
            xi = np.linspace(xs.min(), xs.max(), 100)
            yi = np.linspace(ys.min(), ys.max(), 100)
            zi = griddata((xs, ys), zs, (xi[None,:], yi[:,None]), method='linear')
            # plot
            if var == 'c':
                vmin, vmax = 0, 1
            elif var == 'cdot':
                vlim = max(abs(zi.min()), abs(zi.max()))
                vmin, vmax = -vlim, vlim
            else:
                vmin, vmax = zi.min(), zi.max()
            img = ax[i].pcolor(xi, yi, zi, cmap=cmap, vmin=vmin, vmax=vmax)
            ticks = np.linspace(vmin, vmax, 5)
            plt.colorbar(img, ax=ax[i], ticks=ticks)
            ax[i].set_title(f"{symbol}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        # ax[0].set_ylabel(f'Sample id: {batches[batch][t]["sample"][0]}')
        ax[0].set_ylabel(f'$p_{l}-p_{r}$ = {batches[batch][t]["dp"][0]:.2f}')
        plt.tight_layout()
        plt.savefig(export_path)
        print(f'saved: {export_path}')
        plt.close()
        # exit()
    # create animations
    for var in output_vars:
        images = []
        for t in batches[batch]:
            images.append(imageio.imread(os.path.join(batch_paths[batch][t], f'{var}.png')))
        imageio.mimsave(f'{batch_path}/{var}.gif', images, duration=2.0)

