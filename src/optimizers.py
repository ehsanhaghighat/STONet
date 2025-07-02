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
from typing import Dict, List, Callable
import time
from datetime import datetime
from tqdm import tqdm
import pandas
import torch
import logging
from src.data_models import BaseDataModel


class Optimizer:
    def __init__(self,
                 model: torch.nn.Module,
                 optim: torch.optim.Optimizer) -> None:
        self.model = model
        self.optim = optim(self.model.parameters())
        self.training_history = []

    def save(self, path: str):
        checkpoint = {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'loss': self.training_history
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.optim.load_state_dict(checkpoint['optim'])
        self.training_history = checkpoint['loss']
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def set_learning_rate(self, lr=0.001):
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        logging.info('Learning rate is set to {:f}.'.format(lr))

    def prepare_lr_scheduler(self, scheduler: Dict, num_epochs: int, patience_epochs: int = None):
        if (scheduler is None) or (len(scheduler) == 0):
            if patience_epochs is None:
                patience_epochs = num_epochs / 20
            scheduler = {
                'ReduceLROnPlateau': dict(mode='min', factor=0.75, patience=int(patience_epochs), verbose=True)
            }
        # setup scheduler
        assert isinstance(scheduler, dict) and len(scheduler) == 1, \
            'The scheduler must have one key with its params: e.g. `{"StepLR": {"step_size": 100}}`'
        name = list(scheduler.keys())[0]
        sched = getattr(torch.optim.lr_scheduler, name)(self.optim, **scheduler[name])
        if name == 'ReduceLROnPlateau':
            def scheduler_step(loss_val=0.):
                sched.step(loss_val)
                return self.optim.param_groups[0]['lr']  # ReduceLROnPlateau does not have get_lr method
        else:
            def scheduler_step(loss_val=0.):  # pylint: disable=W0613
                sched.step()
                return sched.get_lr()[0]
        return scheduler_step

    def train_step(self,
                   loss_terms: Dict[str, Callable],
                   inputs: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor],
                   weights: Dict[str, torch.Tensor]):
        """ Train the model for one step.
        """
        self.optim.zero_grad()
        outputs = self.model(inputs)
        loss_vals = {}
        for key in loss_terms:
            loss_vals[key] = loss_terms[key].forward(inputs, outputs, targets, weights)
        total_loss = sum(loss_vals.values())
        total_loss.backward(retain_graph=True)
        self.optim.step()
        loss_vals['total'] = total_loss
        return loss_vals

    def train_epoch(self,
                    loss_terms: Dict[str, Callable],
                    dataset: BaseDataModel):
        """ Train the model for one batch.
        """
        batch_losses = []
        tqdm_batch_iterator = tqdm(range(0, len(dataset)), leave=False)
        for batch in tqdm_batch_iterator:
            inputs, targets, weights = dataset[batch]
            batch_time_start = time.time()
            losses = self.train_step(loss_terms, inputs, targets, weights)
            current_time = time.time()
            loss_vals = {f"loss_{k}": v.cpu().detach().numpy() for k, v in losses.items()}
            # update progress bar
            if batch % 10 == 0:
                tqdm_batch_iterator.set_description(
                    Optimizer.batch_training_txt(
                        batch + 1, len(dataset),
                        loss_vals['loss_total'],
                        batch_time_start,
                        current_time)
                )
            batch_losses.append(loss_vals)
        return pandas.DataFrame(batch_losses).mean(axis=0).to_dict()

    def train(self,
              loss_terms: List[Callable],
              dataset: BaseDataModel,
              num_epochs: int = 1000,
              learning_rate: float = 0.001,
              lr_scheduler: Dict = dict(),
              stop_lr: float = 1e-8,
              checkpoint_path: str = None,
              patience_epochs: int = None):
        # save model/optimizer state
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_freq = int(num_epochs / 20)
        self.save(os.path.join(checkpoint_path, 'checkpoint-start.pkl'))
        # setting up the optimizer
        self.set_learning_rate(learning_rate)
        scheduler_step = self.prepare_lr_scheduler(lr_scheduler, num_epochs, patience_epochs)
        # training loop
        total_time_start = time.time()
        num_old_epochs = len(self.training_history)
        tqdm_epoch_iterator = tqdm(range(num_old_epochs, num_old_epochs+num_epochs))
        for epoch in tqdm_epoch_iterator:
            epoch_time_start = time.time()
            dataset.shuffle()
            loss_vals = self.train_epoch(loss_terms, dataset)
            current_time = time.time()
            # update learning rates
            lr = scheduler_step(loss_vals['loss_total'])
            # epoch loss
            time_s = current_time - total_time_start
            training_history = {'datetime': str(datetime.now()), 'time_s': time_s, 'lr': lr}
            training_history.update(loss_vals)
            self.training_history.append(training_history)
            # update progress bar
            tqdm_epoch_iterator.set_description(
                Optimizer.epoch_training_txt(
                    epoch + 1, num_epochs,
                    training_history['loss_total'],
                    epoch_time_start,
                    total_time_start,
                    current_time,
                    lr)
            )
            # save model/optimizer state
            if (epoch + 1) % checkpoint_freq == 0:
                self.save(os.path.join(checkpoint_path, f'checkpoint-{epoch + 1}.pkl'))
            # early stopping
            if lr < stop_lr:
                break
        # make sure to save final checkpoint
        self.save(os.path.join(checkpoint_path, 'checkpoint-end.pkl'))
        # usually, we only need training on GPU and not inference.
        # additionally, not all machines have GPU, so better to put back on cpu after training.
        # once training is done, default to cpu.
        # set trained flag and loss history
        return self.training_history

    @staticmethod
    def batch_training_txt(batch: int,
                           num_batches: int,
                           total_loss: float,
                           batch_time: float,
                           current_time: float):
        txt = []
        txt.append(
            ('batch: {:0' + str(len(str(num_batches))) + 'd}  ').format(batch, num_batches)
        )
        txt.append(
            ('time(e,t): {:0.1f}ms  ').format(1000 * (current_time - batch_time))
        )
        txt.append(
            ('loss: {:0.3e} ').format(total_loss)
        )
        return ''.join(txt)

    @staticmethod
    def epoch_training_txt(epoch: int,
                           num_epochs: int,
                           total_loss: float,
                           epoch_time: float,
                           total_time: float,
                           current_time: float,
                           learning_rate: float):
        txt = []
        txt.append(
            ('epoch: {:0' + str(len(str(num_epochs))) + 'd}  ').format(epoch, num_epochs)
        )
        txt.append(
            ('time(e,t): {:0.1f}ms/{:0.1f}s  ').format(
                1000 * (current_time - epoch_time),
                current_time - total_time)
        )
        if learning_rate:
            txt.append(
                ('lr: {:0.3e}  ').format(learning_rate)
            )
        txt.append(
            ('loss: {:0.3e} ').format(total_loss)
        )
        return ''.join(txt)

