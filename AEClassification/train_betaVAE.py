# Basic python and data processing imports
import math
import pickle

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from AEClassification.BetaVAE import BetaVAE_EP
from AEClassification.EPIDataset import EPIDataset
from AEClassification.BetaVAESolver import Solver, kl_divergence

np.set_printoptions(suppress=True)  # Suppress scientific notation when printing small
import h5py


# import matplotlib.pyplot as plt
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

model = 'H'
cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
# cell_lines = ['HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
# cell_lines = ['HUVEC', 'IMR90', 'K562', 'NHEK']
# cell_lines = ['IMR90', 'K562', 'NHEK']

# Model training parameters
num_epochs = 60
batch_size = 64
training_frac = 0.9  # fraction of data to use for training

t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
# opt = Adam(lr=1e-5)  # opt = RMSprop(lr = 1e-6)

data_path = 'data/all_sequence_data.h5'
use_cuda = True
device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
epochs = 100
z_dim = 64
beta = 2
lr=1e-3
train_ratio = 0.9  # fraction of data to use for training
l1_weight = 1e-5
l2_weight = 1e-5
recon_criterion = torch.nn.MSELoss()

training_histories = {}

for cell_line in cell_lines:
    dataset = EPIDataset(data_path, cell_line, use_cuda=use_cuda)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    x_p_length, x_e_length = train_data_loader.dataset[0][0].shape[-1], train_data_loader.dataset[0][1].shape[-1]

    net: nn.Module = BetaVAE_EP(promoter_input_length=x_p_length, enhancer_input_length=x_e_length, z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    recon_losses_train = []
    total_klds_train = []
    recon_losses_val = []
    total_klds_val = []
    best_loss = np.inf

    for epoch in range(epochs):
        mini_batch_i_train = 0
        mini_batch_i_val = 0
        pbar = tqdm(total=math.ceil(len(train_data_loader.dataset) / train_data_loader.batch_size),
                    desc='Training BetaVAE Net')
        pbar.update(mini_batch_i_train)
        batch_recon_losses = []
        batch_total_klds = []

        net.train()
        for x_p, x_e, y in train_data_loader:
            mini_batch_i_train += 1
            pbar.update(1)

            # enhancer VAE
            l2_penalty = l2_weight * sum([(p ** 2).sum() for p in net.parameters()])

            x_recon_p, x_recon_e, mu, logvar = net(x_p, x_e)
            recon_loss = recon_criterion(torch.concat([x_p, x_e], dim=2), torch.concat([x_recon_p, x_recon_e], dim=2))
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            beta_vae_loss = recon_loss + beta * total_kld + l2_penalty

            optimizer.zero_grad()
            beta_vae_loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # gradient clipping to avoid exploding gradient
            optimizer.step()

            pbar.set_description( 'Training [{}]: recon_loss:{:.8f} total_kld:{:.8f}'.format( mini_batch_i_train, recon_loss.item(), total_kld.item()))
            batch_recon_losses.append(recon_loss.item())
            batch_total_klds.append(total_kld.item())
        recon_losses_train.append(np.mean(batch_recon_losses))
        total_klds_train.append(np.mean(batch_total_klds))
        pbar.close()

        with torch.no_grad():
            pbar = tqdm(total=math.ceil(len(val_data_loader.dataset) / val_data_loader.batch_size),
                        desc='Validating AE Net')
            pbar.update(mini_batch_i_val)
            batch_recon_losses = []
            batch_total_klds = []
            for x_p, x_e, y in val_data_loader:
                mini_batch_i_val += 1
                pbar.update(1)

                x_recon_p, x_recon_e, mu, logvar = net(x_p, x_e)
                recon_loss = recon_criterion(torch.concat([x_p, x_e], dim=2),
                                             torch.concat([x_recon_p, x_recon_e], dim=2))
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                beta_vae_loss = recon_loss + beta * total_kld

                batch_recon_losses.append(recon_loss.item())
                batch_total_klds.append(total_kld.item())
                pbar.set_description(
                    'Validating [{}]: recon_loss:{:.8f} total_kld:{:.8f}'.format(mini_batch_i_train, recon_loss.item(),total_kld.item()))

            recon_losses_val.append(np.mean(batch_recon_losses))
            total_klds_val.append(np.mean(batch_total_klds))
            pbar.close()


        print("Epoch {}: train recon loss={:.8f}, train KLD={:.8f} , val recon loss={:.8f}, "
              "val KLD={:.8f}".format(epoch, recon_losses_train[-1], total_klds_train[-1],
                                        recon_losses_val[-1], total_klds_val[-1]))

        if recon_losses_val[-1] < best_loss:
            torch.save(net.state_dict(), 'AEClassification/models/net_BetaVAE_{}'.format(cell_line))
            print(
                'Best BetaVAE loss improved from {} to {}, saved best model to {}'.format(best_loss, recon_losses_val[-1],
                                                                                  'AEClassification/models/net_BetaVAE_{}'.format(
                                                                                      cell_line)))
            best_loss = recon_losses_val[-1]

        # Save training histories after every epoch
        training_histories[cell_line] = {'recon_loss_train': recon_losses_train, 'KLD_train': total_klds_train,
                                         'recon_loss_val': recon_losses_val, 'KLD_val': total_klds_val}
        pickle.dump(training_histories, open('AEClassification/models/BetaVAE_training_histories.pickle', 'wb'))

    print('Training completed for cell line {}, training history saved'.format(cell_line))





