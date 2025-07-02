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
import time
import shutil
import torch
from src.networks import (
    MLP,
    EnrichedDeepONet,
    Fourier,
    STONet,
    STONet_Attention
)
from src.optimizers import Optimizer
from src.losses import Loss
from src.data_models import DeepONetDataModel
from src.utils import *

from params import net_choices

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

batches = []
sample_per_time_step = []
for x in df_data.groupby(['sample', 't']):
    batches.append(x[1].reset_index(drop=True))
    sample_per_time_step.append(len(x[1]))

assert len(set(sample_per_time_step)) == 1, 'The number of samples must be the same for all time steps.'
print('sample_per_time_step:', set(sample_per_time_step))
sample_per_time_step = sample_per_time_step[0]

column_feature = ['sample', 'x', 'y', 'dp', 'vx', 'vy', 'k11', 'k12', 'k22', 'c', 't', 'cdot', 'deltaC']
import_data_df = []
for batch in batches:
    batch['prob'] = batch['cdot'].abs() / batch['cdot'].abs().sum()

    selected_data_1000 = batch.sample(n=1000, weights='prob', replace=False, random_state=42)
    selected_data_500 = batch[~batch.isin(selected_data_1000.to_dict(orient='list')).all(axis=1)].sample(n=500, random_state=42)

    merged_df = pd.concat([selected_data_1000[column_feature], selected_data_500[column_feature]], ignore_index=True)
    import_data_df.append(merged_df) ### = pd.concat([import_data_df, merged_df], ignore_index=True)

size_selected_sample_per_time_step = len(merged_df)
import_data_df = pd.concat(import_data_df, ignore_index=True)
print(import_data_df)
print(import_data_df.describe())

# scale data
# import_data_dict = {k: vimport_data_df.to_dict(orient='list')}
import_data_dict = {k: import_data_df[k].values for k in import_data_df.keys() if k != 'sample'}
print('import_data_dict:', {k: v.shape for k, v in import_data_dict.items()})


scaler = MinMaxScaler().fit(import_data_dict)


import_data_norm = scaler.transform(import_data_dict)
import_data_norm['sample'] = import_data_df['sample'].values


np.random.seed(42)
np.random.shuffle(sample_ids)
ids_sample_train, ids_sample_test = np.array_split(sample_ids, [int(len(sample_ids)*0.9)])
print('sample_ids_train:', ids_sample_train.shape, ids_sample_train)
print('sample_ids_test:', ids_sample_test.shape, ids_sample_test)

sample_ids_train = np.in1d(import_data_df['sample'].values, ids_sample_train)
sample_ids_test = np.in1d(import_data_df['sample'].values, ids_sample_test)
print('num train samples:', np.sum(sample_ids_train))
print('num test samples:', np.sum(sample_ids_test))

train_data = {k: v[sample_ids_train] for k, v in import_data_norm.items()}
test_data = {k: v[sample_ids_test] for k, v in import_data_norm.items()}
print('train_data:', {k: v.shape for k, v in train_data.items()})
print('test_data:', {k: v.shape for k, v in test_data.items()})

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


CHECKPOINT_DIR = 'checkpoints'
RESULT_DIR = 'results'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)



old_batches = [x for x in os.listdir(CHECKPOINT_DIR) if x.startswith("batch")]
batch_name = f"batch-{len(old_batches)+1:03d}"
os.makedirs(os.path.join(CHECKPOINT_DIR, batch_name), exist_ok=True)

# backup code and params files
shutil.copyfile('train.py', os.path.join(CHECKPOINT_DIR, batch_name, 'train.py'))
shutil.copyfile('params.py', os.path.join(CHECKPOINT_DIR, batch_name, 'params.py'))


# network setup 
trunk_input_dim = dict(x=1, y=1, t=1, c=1)
branch_input_dim_1 = dict(x=1, y=1, k11=1, k12=1, k22=1, dp=1)
branch_input_dim_2 = dict(dp=1, c=1, t=1)
branch_input_dim = {**branch_input_dim_1, **branch_input_dim_2}
embedding_dim = dict(e=50)
root_input_dim = {k: 2*3*d for k, d in embedding_dim.items()}

output_feature = "cdot"
output_dim = {output_feature: 1}


# check params.py for net_choices
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






    class MSELoss(Loss):
        def eval(self, inputs, outputs, targets, sample_weights={}):
            squared_loss = (outputs[output_feature] - targets[output_feature])**2
            return torch.mean(squared_loss * sample_weights.get(output_feature, 1.0))

    loss_terms = {output_feature: MSELoss()}


    dataset = DeepONetDataModel.prepare_dataframe(
        train_data,
        trunk_input_dim,
        branch_input_dim,
        output_dim,
        batch_by='sample'
    )
    print('dataset:', {k: v.shape for k, v in dataset.items()})

    data_model = DeepONetDataModel(
        dataset, trunk_input_dim, branch_input_dim, output_dim)
    data_model.set_batch_size(10)
    data_model.to(device)

    optimizer = Optimizer(model, torch.optim.Adam)


    checkpoint_path = os.path.join(CHECKPOINT_DIR, output_path)
    os.makedirs(checkpoint_path, exist_ok=True)
    scaler.save(os.path.join(checkpoint_path, 'scaler.pkl'))


    if os.path.exists(os.path.join(checkpoint_path, 'checkpoint-end.pkl')):
        logging.info(f"Skipping existing item ... {checkpoint_path}")
        time.sleep(1)
        continue

    training_history = optimizer.train(loss_terms, data_model, num_epochs=num_epochs, learning_rate=learning_rate, checkpoint_path=checkpoint_path, patience_epochs=patience_epochs)

    optimizer.load(os.path.join(checkpoint_path, 'checkpoint-end.pkl'))

    results_path = os.path.join(RESULT_DIR, output_path)
    os.makedirs(results_path, exist_ok=True)


    training_history_df = pd.DataFrame(training_history)
    training_history_df.to_csv(os.path.join(results_path, 'training_history.csv'), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    ax[0].semilogy(training_history_df['loss_total'])
    ax[0].set_ylabel('total loss')
    ax[0].set_xlabel('epoch')
    ax[1].semilogy(training_history_df['loss_total'] / training_history_df['loss_total'].values[0])
    ax[1].set_ylabel('normalized loss')
    ax[1].set_xlabel('epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'training_history.png'), dpi=200)
    plt.close()


