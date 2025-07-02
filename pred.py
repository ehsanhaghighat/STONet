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
from matplotlib.ticker import PercentFormatter
import shutil
import imageio

import torch
from src.networks import (
    MLP,
    EnrichedDeepONet,
    Fourier,
    STONet,
    STONet_Attention
)
from src.optimizers import Optimizer
from src.data_models import DeepONetDataModel
from src.utils import *
from params import net_choices

from scipy.interpolate import LinearNDInterpolator

import logging
FORMAT = '[%(asctime)s %(levelname)s %(process)d %(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

column_feature = ['sample', 'x', 'y', 'dp', 'vx', 'vy', 'k11', 'k12', 'k22', 'c', 't', 'cdot', 'deltaC']

df_data = pd.read_csv('data/data25_test.csv', delimiter=',')
df_data['prob'] = 1.
sample_ids = df_data['sample'].unique().astype(int)



device = torch.device('cpu')



x_grid = np.linspace(df_data['x'].min(), df_data['x'].max(), 101)
y_grid = np.linspace(df_data['y'].min(), df_data['y'].max(), 101)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)


CHECKPOINT_DIR = 'checkpoints'
RESULT_DIR = 'results'


old_batches = [int(x.split('batch-')[-1]) for x in os.listdir(CHECKPOINT_DIR) if x.startswith("batch")]
batch_name = f"batch-{max(old_batches):03d}"

shutil.copyfile("pred.py", os.path.join(CHECKPOINT_DIR, batch_name, "pred.py"))






# network setup 
trunk_input_dim = dict(x=1, y=1, t=1, c=1)
branch_input_dim_1 = dict(x=1, y=1, k11=1, k12=1, k22=1, dp=1)
branch_input_dim_2 = dict(dp=1, c=1, t=1)
branch_input_dim = {**branch_input_dim_1, **branch_input_dim_2}
embedding_dim = dict(e=50)
root_input_dim = {k: 2*3*d for k, d in embedding_dim.items()}

output_feature = "cdot"
output_dim = {output_feature: 1}



for net_i in net_choices:
    embedding_dim = net_i['embedding_dim']
    model_type = net_i.get('model_type', 'STONet')
    
    trunk_actf = net_i.get('trunk_actf', 'Tanh')
    num_trunk_layers = net_i.get('num_trunk_layers', 8)
    trunk_width = net_i.get('trunk_width', 100)
    
    branch_actf = net_i.get('branch_actf', 'Tanh')
    num_branch_layers = net_i.get('num_branch_layers', 8)
    branch_width = net_i.get('branch_width', 100)
    
    root_actf = net_i.get('root_actf', 'Tanh')
    num_root_layers = net_i.get('num_root_layers', 2)
    root_width = net_i.get('root_width', 100)
    
    stonet_attention_actf = net_i.get('stonet_attention_actf', 'Tanh')
    num_stonet_attention_blocks = net_i.get('num_stonet_attention_blocks', 4)
    
    use_fourier = net_i.get('fourier', False)

    # training parameters
    learning_rate = net_i.get('learning_rate', 0.0005)
    num_epochs = net_i.get('num_epochs', 5000)
    patience_epochs = net_i.get('patience_epochs', int(num_epochs/20))

    output_path = "-".join([
        f"{model_type.lower()}",
        f"{output_feature}",
        f"e{embedding_dim}",
        f"B{num_branch_layers}x{branch_width}x{branch_actf.lower()}",
        f"T{num_trunk_layers}x{trunk_width}x{trunk_actf.lower()}",
        f"R{num_root_layers}x{root_width}x{root_actf.lower()}",
        f"A{num_stonet_attention_blocks}x{stonet_attention_actf.lower()}",
        f"lr{learning_rate:.03e}",
        f"e{num_epochs}",
    ])
    output_path = os.path.join(batch_name, output_path)
    print('output_path:', output_path)
    
    if model_type == "EnrichedDeepONet":
        trunk_net = MLP(trunk_input_dim, dict(e=embedding_dim), num_trunk_layers*[trunk_width], trunk_actf)
        branch_net_1 = MLP(branch_input_dim_1, dict(e=embedding_dim), num_branch_layers*[branch_width], branch_actf)
        branch_net_2 = MLP(branch_input_dim_2, dict(e=embedding_dim), num_branch_layers*[branch_width], branch_actf)
        root_net = MLP(root_input_dim, output_dim, num_root_layers*[root_width], root_actf)
        model = EnrichedDeepONet(trunk_net, [branch_net_1, branch_net_2], root_net)
    elif model_type == "STONet":
        trunk_net = MLP(trunk_input_dim, dict(e=embedding_dim), num_trunk_layers*[trunk_width], trunk_actf)
        branch_net = MLP(branch_input_dim_1, dict(e=embedding_dim), num_branch_layers*[branch_width], branch_actf)
        atten_nets = num_stonet_attention_blocks*[STONet_Attention(dict(e=embedding_dim), stonet_attention_actf)]
        root_net = MLP(dict(e=embedding_dim), output_dim, num_root_layers*[root_width], root_actf)
        model = STONet(trunk_net, branch_net, atten_nets, root_net)
    else:
        raise ValueError('Not recognized')

    model.to(device)




    checkpoint_path = os.path.join(CHECKPOINT_DIR, output_path)
    results_path = os.path.join(RESULT_DIR, output_path)


    try:
        scaler = MinMaxScaler.load(os.path.join(checkpoint_path, 'scaler.pkl'))
    except:
        continue


    optimizer = Optimizer(model, torch.optim.Adam)

    list_checkpoints = os.listdir(checkpoint_path)
    if 'checkpoint-end.pkl' in list_checkpoints:
        checkpoint_file_id = 'end'
    else:
        list_checkpoint_ids = [x[:-4].split('-')[-1] for x in list_checkpoints]
        checkpoint_file_id = sorted([x for x in list_checkpoint_ids if x.isnumeric()])[-1]
    checkpoint_file_name = f'checkpoint-{checkpoint_file_id}.pkl'


    try:
        optimizer.load(os.path.join(checkpoint_path, checkpoint_file_name))
    except:
        continue



    def Predict_data(input_data, t0, t1):
        dataset_t1 = input_data[input_data['t'] == float(t1)].reset_index(drop=True)
        dataset = input_data[input_data['t'] == float(t0)].reset_index(drop=True)
        true = {"true_c": dataset_t1["c"], "true_cdot": dataset["cdot"]}
        dataset = {k: np.asanyarray(v) for k, v in dataset.items()}
        dataset_transformed = scaler.transform(dataset)
        df = DeepONetDataModel.prepare_dataframe(
            dataset_transformed,
            trunk_input_dim,
            branch_input_dim,
            output_dim
        )
        data_model = DeepONetDataModel(
            df, trunk_input_dim, branch_input_dim, output_dim)
        data_model.to(device)

        pred = model.forward(data_model.inputs)
        pred_original = scaler.inverse_transform(
            {k: v.cpu().detach().numpy().flatten() for k, v in pred.items()})
        pred_original = {f"pred_{output_feature}": pred_original[output_feature]}
        return {**dataset, **pred_original, **true}


    def cdot_plot(time, x_t, y_t, cdot_t, true_cdot_t, path, cmap='viridis'):
        fig, axs = plt.subplots(1, 2, figsize=(9, 3), dpi=200)

        vlim = max(abs(true_cdot_t.min()), abs(true_cdot_t.max()))
        levels = np.linspace(-vlim, vlim, 20)
        ticks = np.linspace(-vlim, vlim, 5)
        
        ax = axs[0]
        interp = LinearNDInterpolator(list(zip(x_t, y_t)), cdot_t)
        cdot_grid = interp(X_grid, Y_grid)
        contour_cdot = ax.pcolor(X_grid, Y_grid, cdot_grid, cmap = cmap, vmin=-vlim, vmax=vlim)
        plt.colorbar(contour_cdot, ax=ax, ticks=ticks)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('cdot')

        interp = LinearNDInterpolator(list(zip(x_t, y_t)), true_cdot_t)
        true_cdot_grid = interp(X_grid, Y_grid)
        ax = axs[1]
        contour_cdot = ax.pcolor(X_grid, Y_grid, true_cdot_grid, cmap = cmap, vmin=-vlim, vmax=vlim)
        plt.colorbar(contour_cdot, ax=ax, ticks=ticks)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('$cdot^*$')
        plt.tight_layout()
        fig_path = os.path.join(path, f'cdot_{time}.png')
        plt.savefig(fig_path, dpi=200)
        plt.close()
        return fig_path


    def c_plot(time, x_t, y_t, c_t, true_c_t, path, cmap='viridis'):
        fig, axs = plt.subplots(1, 2, figsize=(9, 3), dpi=200)
        levels = np.round(np.linspace(0., 1.0, 20), decimals=6)
        ticks = np.round(np.linspace(0., 1.0, 5), decimals=6)
        
        interp = LinearNDInterpolator(list(zip(x_t, y_t)), c_t)
        c_grid = interp(X_grid, Y_grid)
        
        ax = axs[0]
        contour_cdot = ax.pcolor(X_grid, Y_grid, c_grid, cmap = cmap, vmin=0., vmax=1.0)
        plt.colorbar(contour_cdot, ax=ax, ticks=ticks)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('c')

        interp = LinearNDInterpolator(list(zip(x_t, y_t)), true_c_t)
        true_c_grid = interp(X_grid, Y_grid)
        ax = axs[1]
        contour_cdot = ax.pcolor(X_grid, Y_grid, true_c_grid, cmap = cmap, vmin=0., vmax=1.0)
        plt.colorbar(contour_cdot, ax=ax, ticks=ticks)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('$c^*$')
        plt.tight_layout()
        fig_path = os.path.join(path, f'c_{time}.png')
        plt.savefig(fig_path, dpi=200)
        plt.close()
        return fig_path

    def plot_data(time, x_t, y_t, c_t, true_c_t, path, label="c", cmap='viridis', vmin=0., vmax=1.0):
        fig, axs = plt.subplots(1, 2, figsize=(9, 3))

        ticks = np.linspace(vmin, vmax, 5)
        interp = LinearNDInterpolator(list(zip(x_t, y_t)), c_t)
        c_grid = interp(X_grid, Y_grid)

        ax = axs[0]
        contour_cdot = ax.pcolor(X_grid, Y_grid, c_grid, cmap = cmap, vmin=vmin, vmax=vmax)
        # plt.colorbar(contour_cdot, ax=ax, ticks=ticks)
        ax.label_outer()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{label}')

        interp = LinearNDInterpolator(list(zip(x_t, y_t)), true_c_t)
        true_c_grid = interp(X_grid, Y_grid)
        ax = axs[1]
        contour_cdot = ax.pcolor(X_grid, Y_grid, true_c_grid, cmap = cmap, vmin=vmin, vmax=vmax)
        ax.label_outer()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'${label}^*$')

        # plt.tight_layout()
        plt.subplots_adjust(bottom=0.18, top=0.88, left=0.1, right=0.9, wspace=0.2)
        plt.colorbar(contour_cdot, ax=ax, ticks=ticks, orientation='vertical', fraction=0.046, pad=0.04)

        fig_path = os.path.join(path, f'{label}_{time}.png')
        plt.savefig(fig_path, dpi=200)
        plt.close()
        return fig_path

    def plot_multi_time(prediction_data, col='pred_cdot', path=None, cmap='viridis', vmin=0., vmax=1.0):
        assert len(prediction_data) == 4
        fig, axs = plt.subplots(1, 4, figsize=(18, 3))
        ticks = np.linspace(vmin, vmax, 5)
        for ax, data in zip(axs, prediction_data):
            x_t = data['x']
            y_t = data['y']
            c_t = data[col]
            t_hr = np.round(data['t'] / 3600, 1)
            interp = LinearNDInterpolator(list(zip(x_t, y_t)), c_t)
            c_grid = interp(X_grid, Y_grid)
            contour_cdot = ax.pcolor(X_grid, Y_grid, c_grid, cmap = cmap, vmin=vmin, vmax=vmax)
            ax.label_outer()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('t = %.1f hr' % t_hr)
        # add colorbar only to last axis
        # plt.tight_layout()
        plt.subplots_adjust(bottom=0.18, top=0.88, left=0.07, right=0.93, wspace=0.15)
        plt.colorbar(contour_cdot, ax=ax, ticks=ticks, fraction=0.046, pad=0.04)
        img_path = os.path.join(path, f'{col}_multi.png')
        plt.savefig(img_path, dpi=200)
        plt.close()
        return img_path

    time_steps = df_data['t'].unique().astype(float)
    pred_c_all = {t: [] for t in time_steps[1:]}
    pred_cdot_all = {t: [] for t in time_steps[1:]}
    true_c_all = {t: [] for t in time_steps[1:]}
    true_cdot_all = {t: [] for t in time_steps[1:]}
    
    for c_idx, idx in enumerate(sample_ids):
        dataset_idx = df_data[df_data['sample'] == idx][column_feature].reset_index(drop=True)
        time_steps = dataset_idx['t'].unique().astype(float)
        assert time_steps[0] == 0, 'The first time step must be 0.'

        # output path
        path = os.path.join(results_path, f'test__{idx}')
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

        output_time_steps = [14400., 43200., 86400., 129600.]
        prediction_data = []
        c_plots, cdot_plots = [], []
        for t0, t1 in zip(time_steps[:-1], time_steps[1:]):
            logging.info('Predicting at time %f', t1)
            dtime = t1 - t0
            c0 = dataset_idx[dataset_idx['t'] == t0]['c'].values
            pred_c_t = c0.flatten()
            
            dataset = Predict_data(dataset_idx, t0, t1)
            true_c_t = dataset.get('true_c', None)
            true_cdot = dataset.get('true_cdot', None)

            assert f"pred_{output_feature}" == "pred_cdot"
            pred_cdot = dataset[f"pred_{output_feature}"]
            pred_c_t += pred_cdot*dtime
            pred_c_t = np.clip(pred_c_t, 0, 1)

            x_t = np.asanyarray(dataset['x'])
            y_t = np.asanyarray(dataset['y'])
            pred_c_t = np.asanyarray(pred_c_t.flatten())
            pred_cdot_t = np.asanyarray(pred_cdot.flatten())
            true_cdot_t = np.asanyarray(true_cdot) if true_cdot is not None else None

            pred_c_all[t1].append(pred_c_t)
            pred_cdot_all[t1].append(pred_cdot_t)
            true_c_all[t1].append(true_c_t)
            true_cdot_all[t1].append(true_cdot_t)

            c_plots.append(
                plot_data(t1, x_t, y_t, pred_c_t, true_c_t, path, "c", "seismic", 0., 1.0)
            )
            vlim = max(abs(true_cdot_t.min()), abs(true_cdot_t.max()))
            cdot_plots.append(
                plot_data(t1, x_t, y_t, pred_cdot_t, true_cdot_t, path, "cdot", "seismic", -vlim, vlim)
            )
            if t1 in output_time_steps:
                prediction_data.append(
                    dict(t=t1,
                         x=x_t,
                         y=y_t,
                         pred_c=pred_c_t,
                         pred_cdot=pred_cdot_t,
                         true_c=true_c_t,
                         true_cdot=true_cdot_t)
                )

        # create images
        plot_multi_time(prediction_data, 'pred_c', path, cmap='seismic', vmin=0., vmax=1.0)
        plot_multi_time(prediction_data, 'true_c', path, cmap='seismic', vmin=0., vmax=1.0)
        vlim = max([max(abs(x['true_cdot'].min()), abs(x['true_cdot'].max())) for x in prediction_data])
        plot_multi_time(prediction_data, 'pred_cdot', path, cmap='seismic', vmin=-vlim, vmax=vlim)
        plot_multi_time(prediction_data, 'true_cdot', path, cmap='seismic', vmin=-vlim, vmax=vlim)

        # create animations
        images = [imageio.imread(x) for x in c_plots]
        imageio.mimsave(os.path.join(path, 'c.gif'), images, duration=2.0)
        images = [imageio.imread(x) for x in cdot_plots]
        imageio.mimsave(os.path.join(path, 'cdot.gif'), images, duration=2.0)
        

    c_error_all = {r: {} for r in ["abs", "rel", "sym-rel"]}
    cdot_error_all = {r: {} for r in ["abs", "rel", "sym-rel"]}
    for t in pred_c_all:
        pred_c_all_t = np.concatenate(pred_c_all[t])
        true_c_all_t = np.concatenate(true_c_all[t])
        pred_cdot_all_t = np.concatenate(pred_cdot_all[t])
        true_cdot_all_t = np.concatenate(true_cdot_all[t])
        c_abs_error_t = np.abs(pred_c_all_t - true_c_all_t)
        c_rel_error_t = c_abs_error_t / (true_c_all_t + 1e-6)
        c_sym_rel_error_t = c_abs_error_t / (np.abs(pred_c_all_t) + np.abs(true_c_all_t) + 1e-6)
        cdot_abs_error_t = np.abs(pred_cdot_all_t - true_cdot_all_t)
        cdot_rel_error_t = cdot_abs_error_t / (np.abs(true_cdot_all_t) + 1e-12)
        cdot_sym_rel_error_t = cdot_abs_error_t / (np.abs(pred_cdot_all_t) + np.abs(true_cdot_all_t) + 1e-12)
        c_error_all["abs"][t] = c_abs_error_t
        c_error_all["rel"][t] = c_rel_error_t
        c_error_all["sym-rel"][t] = c_sym_rel_error_t
        cdot_error_all["abs"][t] = cdot_abs_error_t
        cdot_error_all["rel"][t] = cdot_rel_error_t
        cdot_error_all["sym-rel"][t] = cdot_sym_rel_error_t


    def plot_histogram(ax, data, bins=50, alpha=1.0, label=None):
        counts, bin_edges = np.histogram(data, bins=bins)
        counts_percentage = 100 * counts / len(data)
        ax.hist(bin_edges[:-1], bin_edges, weights=counts_percentage, alpha=alpha, label=label)
        ax.set_yscale('log')
        ax.set_ylabel(r'Percentage (%)')
        ax.set_ylim([0.001, 100])

    # plot error distribution
    err_type = "abs"
    error_c = c_error_all[err_type]
    error_cdot = cdot_error_all[err_type]

    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    error_c_all = np.concatenate(list(error_c.values()))
    plot_histogram(axs[0], error_c_all, 100)
    axs[0].set_xlabel('$|c - c^*|$')
    error_cdot_all = np.concatenate(list(error_cdot.values()))
    plot_histogram(axs[1], error_cdot_all, 100)
    axs[1].set_xlabel('$|cdot - cdot^*|$')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'test_{err_type}-err_hist.png'), dpi=200)
    plt.close()
    # unroll error
    untoll_time_step = [t for t in error_c]
    untoll_time_step_hr = [np.round(t/3600, 1) for t in untoll_time_step]
    unroll_mean_error_c = [np.mean(np.abs(error_c[t])) for t in error_c]
    unroll_std_error_c = [np.std(np.abs(error_c[t])) for t in error_c]
    unroll_mean_error_cdot = [np.mean(np.abs(error_cdot[t])) for t in error_cdot]
    unroll_std_error_cdot = [np.std(np.abs(error_cdot[t])) for t in error_cdot]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    axs[0].plot([0.]+untoll_time_step_hr, [0.]+unroll_mean_error_c, label='mean')
    # axs[0].errorbar(untoll_time_step, unroll_mean_error_c, yerr=unroll_std_error_c, fmt='o')
    axs[0].set_xlabel('time (hr)')
    axs[0].set_ylabel('$|c - c^*|$')
    axs[0].set_title('Unroll mean error')
    axs[1].plot([0.]+untoll_time_step_hr, [0.]+unroll_mean_error_cdot, label='mean')
    # axs[1].errorbar(untoll_time_step, unroll_mean_error_cdot, yerr=unroll_std_error_cdot, fmt='o')
    axs[1].set_xlabel('time (hr)')
    axs[1].set_ylabel('$|cdot - cdot^*|$')
    axs[1].set_title('Unroll mean error')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'test_unroll_{err_type}-err.png'), dpi=200)
    plt.close()
    # plot unroll error dist
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    for t in reversed(output_time_steps):
        t_hr = np.round(t/3600, 2)
        t_hr_label = f"t = {t_hr} hr"
        plot_histogram(axs[0], error_c[t], 100, alpha=0.5, label=t_hr_label)
        axs[0].set_xlabel('$|c - c^*|$')
        axs[0].legend()
        plot_histogram(axs[1], error_cdot[t], 100, alpha=0.5, label=t_hr_label)
        axs[1].set_xlabel('$|cdot - cdot^*|$')
        axs[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'test_unroll_{err_type}-err_hist.png'), dpi=200)
    plt.close()


