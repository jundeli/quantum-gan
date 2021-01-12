import pennylane as qml
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import argparse
import os
import math
import datetime
import time
from frechetdist import frdist

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch

from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *
from models import Generator, Discriminator
from data.sparse_molecular_dataset import SparseMolecularDataset
from rdkit import Chem


def str2bool(v):
    return v.lower() in ('true')

dev = qml.device('default.qubit', wires=4)
@qml.qnode(dev, interface='torch')
def gen_circuit_1(w):
    # random noise as generator input
    z1 = random.uniform(-1, 1)
    z2 = random.uniform(-1, 1)
    layers = 1
    qubits = 4
    
    # construct generator circuit for both atom vector and node matrix
    for i in range(qubits):
        qml.RY(np.arcsin(z1), wires=i)
        qml.RZ(np.arcsin(z2), wires=i)
    for l in range(layers):
        for i in range(qubits):
            qml.RY(w[i], wires=i)
        for i in range(qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(w[i+qubits], wires=i+1)
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

@qml.qnode(dev, interface='torch')
def gen_circuit_2(w):
    # random noise as generator input
    z1 = random.uniform(-1, 1)
    z2 = random.uniform(-1, 1)
    layers = 1
    qubits = 4
    
    # construct generator circuit for both atom vector and node matrix
    for i in range(qubits):
        qml.RY(np.arcsin(z1), wires=i)
        qml.RZ(np.arcsin(z2), wires=i)
    for l in range(layers):
        for i in range(qubits):
            qml.RY(w[i+7], wires=i)
        for i in range(qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(w[i+qubits+7], wires=i+1)
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Solver for training and testing StarGAN.
    self = Solver(config)
    from logger import Logger
    self.logger = Logger(self.log_dir)

    # Learning rate cache for decaying.
    g_lr = self.g_lr
    d_lr = self.d_lr
    gen_weights = torch.tensor(list(np.random.rand(14)*2-1), requires_grad=True)
    self.g_optimizer = torch.optim.Adam(list(self.G.parameters())+list(self.V.parameters())+[gen_weights],
                                self.g_lr, [self.beta1, self.beta2])

    # Start training from scratch or resume training.
    start_iters = 0
    if self.resume_iters:
        start_iters = self.resume_iters
        self.restore_model(self.resume_iters)

    # Start training.
    print('Start training...')
    start_time = time.time()
    for i in range(start_iters, self.num_iters):
#         if (i+1) % self.log_step == 0:
#             mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
#             sample_list = [gen_circuit(gen_weights) for i in range(a.shape[0])]
# #             z = self.sample_z(a.shape[0])
#             print('[Valid]', '')
#         else:
        mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
#         sample_list = [gen_circuit(gen_weights) for i in range(self.batch_size)]
        sample_list = [torch.cat([gen_circuit_1(gen_weights), gen_circuit_2(gen_weights)]) for i in range(self.batch_size)]
#             z = self.sample_z(self.batch_size)

        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #

        a = torch.from_numpy(a).to(self.device).long()            # Adjacency.
        x = torch.from_numpy(x).to(self.device).long()            # Nodes.
        a_tensor = self.label2onehot(a, self.b_dim)
        x_tensor = self.label2onehot(x, self.m_dim)
        z = torch.stack(tuple(sample_list)).to(self.device).float()
#         z = torch.from_numpy(z).to(self.device).float()

        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #

        # Compute loss with real images.
        logits_real, features_real = self.D(a_tensor, None, x_tensor)
        d_loss_real = - torch.mean(logits_real)

        # Compute loss with fake images.
        edges_logits, nodes_logits = self.G(z)
        # Postprocess with Gumbel softmax
        (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
        logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
        d_loss_fake = torch.mean(logits_fake)

        # Compute loss for gradient penalty.
        eps = torch.rand(logits_real.size(0),1,1,1).to(self.device)
        x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
        x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
        grad0, grad1 = self.D(x_int0, None, x_int1)
        d_loss_gp = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)


        # Backward and optimize.
        d_loss = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
        self.reset_grad()
        d_loss.backward(retain_graph=True)
        self.d_optimizer.step()

        # Logging.
        loss = {}
        loss['D/loss_real'] = d_loss_real.item()
        loss['D/loss_fake'] = d_loss_fake.item()
        loss['D/loss_gp'] = d_loss_gp.item()

        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #

        if (i+1) % self.n_critic == 0:
            # Z-to-target
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
            g_loss_fake = - torch.mean(logits_fake)

            # Real Reward
            rewardR = torch.from_numpy(self.reward(mols)).to(self.device)
            # Fake Reward
            (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
            edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
            mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                    for e_, n_ in zip(edges_hard, nodes_hard)]
            rewardF = torch.from_numpy(self.reward(mols)).to(self.device)

            # Value loss
            value_logit_real,_ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
            value_logit_fake,_ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)
            g_loss_value = torch.mean((value_logit_real - rewardR) ** 2 + (
                                       value_logit_fake - rewardF) ** 2)
            #rl_loss= -value_logit_fake
            #f_loss = (torch.mean(features_real, 0) - torch.mean(features_fake, 0)) ** 2

            # Backward and optimize.
            g_loss = g_loss_fake + g_loss_value
            self.reset_grad()
            g_loss.backward(retain_graph=True)
            self.g_optimizer.step()
            
            R=[list(a[i].reshape(-1))  for i in range(self.batch_size)] #list(x[i]) + 
            F=[list(edges_hard[i].reshape(-1))  for i in range(self.batch_size)] #list(nodes_hard[i]) + 
            fd_bond_only = frdist(R, F)
            
            R=[list(x[i]) + list(a[i].reshape(-1))  for i in range(self.batch_size)]
            F=[list(nodes_hard[i]) + list(edges_hard[i].reshape(-1))  for i in range(self.batch_size)]
            fd_bond_atom = frdist(R, F)
            
            # Saving model checkpoint with lowest FD score
            if "fd_bond_atom_min" not in locals():
                fd_bond_atom_min = 30
            if fd_bond_atom_min > fd_bond_atom:
                if "lowest_ind" not in locals():
                    lowest_ind = 0

                if os.path.exists(os.path.join(self.model_save_dir, '{}-G.ckpt'.format(lowest_ind))):
                    os.remove(os.path.join(self.model_save_dir, '{}-G.ckpt'.format(lowest_ind)))
                    os.remove(os.path.join(self.model_save_dir, '{}-D.ckpt'.format(lowest_ind)))
                    os.remove(os.path.join(self.model_save_dir, '{}-V.ckpt'.format(lowest_ind)))

                lowest_ind = i+1
                fd_bond_atom_min = fd_bond_atom

                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.V.state_dict(), V_path)

                with open('p_qgan_hg_15p/models/p_qgan_hg_15p_red_weights.csv', 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([i+1] + list(gen_weights.detach().numpy()))
                with open('p_qgan_hg_15p/models/lowest_indices.csv', 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([i+1] + [fd_bond_atom])

            # Logging.
            loss['G/loss_fake'] = g_loss_fake.item()
            loss['G/loss_value'] = g_loss_value.item()
            loss['FD/fd_bond_only'] = fd_bond_only
            loss['FD/fd_bond_atom'] = fd_bond_atom
            print('g_loss:{}, d_loss:{}, fd_bond_only:{}, fd_bond_atom:{}'.format(g_loss.item(), \
                                                                                  d_loss.item(), fd_bond_only, fd_bond_atom))

        # =================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #

        # Print out training information.
        if (i+1) % self.log_step == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)

            # Log update
            m0, m1 = all_scores(mols, self.data, norm=True)     # 'mols' is output of Fake Reward
            m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
            m0.update(m1)
            loss.update(m0)
            for tag, value in loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)

            with open('p_qgan_hg_15p/results/q8_metric_scores_log.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow([i+1, et]+[torch.mean(rewardR).item(), torch.mean(rewardF).item()]+\
                               [value for tag, value in loss.items()])

            if self.use_tensorboard or True:
                for tag, value in loss.items():
                    self.logger.scalar_summary(tag, value, i+1)


        # Save model checkpoints.
        if (i+1) % self.model_save_step == 0:
            G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
            D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
            V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(i+1))
            torch.save(self.G.state_dict(), G_path)
            torch.save(self.D.state_dict(), D_path)
            torch.save(self.V.state_dict(), V_path)
            with open('p_qgan_hg_15p/models/q8_weights.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow([i+1] + list(gen_weights.detach().numpy()))
            print('Saved model checkpoints into {}...'.format(self.model_save_dir))

        # Decay learning rates.
        if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
            g_lr -= (self.g_lr / float(self.num_iters_decay))
            d_lr -= (self.d_lr / float(self.num_iters_decay))
            self.update_lr(g_lr, d_lr)
            print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=8, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[128], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]], help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=5000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=2500, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=5000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset')
    parser.add_argument('--log_dir', type=str, default='p_qgan_hg_15p/logs')
    parser.add_argument('--model_save_dir', type=str, default='p_qgan_hg_15p/models')
    parser.add_argument('--sample_dir', type=str, default='p_qgan_hg_15p/samples')
    parser.add_argument('--result_dir', type=str, default='p_qgan_hg_15p/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=500)

    config = parser.parse_args()
    print(config)
    main(config)
