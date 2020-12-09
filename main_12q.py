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

dev = qml.device('default.qubit', wires=12)
@qml.qnode(dev, interface='torch')
def gen_circuit(w):
    # random noise as generator input
    z = random.uniform(-1, 1)
    layers = 1
    qubits = 12
    
    # construct generator circuit for both atom vector and node matrix
    for i in range(qubits):
        qml.RY(np.arcsin(z), wires=i)
        qml.RZ(np.arccos(z), wires=i)
    for l in range(layers):
        for i in range(qubits):
            qml.RY(w[i], wires=i)
            qml.Hadamard(wires=i)
        for i in range(qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(w[i+qubits], wires=i+1)
            qml.CNOT(wires=[i, i+1])
    for i in range(qubits):
        qml.Hadamard(wires=i)
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

    if config.mode == 'train':
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        gen_weights = torch.tensor(list(np.random.rand(23)*2-1), requires_grad=True)
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
            mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
#                 z = self.sample_z(self.batch_size)
            sample_list = [gen_circuit(gen_weights) for i in range(self.batch_size)]
            z = torch.stack(tuple(sample_list)).to(self.device).float()
            print(z)

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            a = torch.from_numpy(a).to(self.device).long()            # Adjacency.
            x = torch.from_numpy(x).to(self.device).long()            # Nodes.
            a_tensor = self.label2onehot(a, self.b_dim)
            x_tensor = self.label2onehot(x, self.m_dim)
#             z = torch.from_numpy(z).to(self.device).float()

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
          
            if i == 0:
                g_loss = d_loss
                g_loss_fake = g_loss
                g_loss_value = g_loss
            print(gen_weights.detach())
            print(
                "%s\tEpoch %d/%d \t[D loss: %f]\t[G loss: %f]"
                % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i+1, self.num_iters, d_loss.item(), g_loss.item())
            )

            try:
                n_p = np.mean([i for i in MolecularMetrics.natural_product_scores(mols, norm=True) if i != 0 ])
            except:
                n_p = 0
            try:
                qed = np.mean([i for i in MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True) \
                               if i != 0 ])
            except:
                qed = 0
            try:
                logp = np.mean([i for i in MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True) \
                                if i != 0 ])
            except:
                logp = 0
            try:
                sa = np.mean([i for i in MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True) if i != 0 ])
            except:
                sa = 0
            try:
                valid_score = np.mean([i for i in MolecularMetrics.valid_scores(mols) if i != 0 ])
            except:
                valid_score = 0
            try:
                unique_score = np.mean([i for i in MolecularMetrics.unique_scores(mols) if i != 0 ])
            except:
                unique_core = 0        

            et = time.time() - start_time
            with open('metric_scores_12q.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow([i+1, et, d_loss.item(), d_loss_fake.item(), d_loss_real.item(), g_loss.item(),\
                                 g_loss_fake.item(), g_loss_value.item()] + [n_p, qed, logp, sa, valid_score, unique_score])

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.V.state_dict(), V_path)
                with open('molgan/models/weights_12q.csv', 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([i+1] + list(gen_weights.detach().numpy()))
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
        
    elif config.mode == 'test':
        self.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=12, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[128,256,512], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]], \
                        help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=10000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=5000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=10000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset')
    parser.add_argument('--log_dir', type=str, default='molgan/logs')
    parser.add_argument('--model_save_dir', type=str, default='molgan/models')
    parser.add_argument('--sample_dir', type=str, default='molgan/samples')
    parser.add_argument('--result_dir', type=str, default='molgan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=5)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=200)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)