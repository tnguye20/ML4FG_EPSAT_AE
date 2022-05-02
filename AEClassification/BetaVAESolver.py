"""solver.py"""
import math
import warnings

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from AEClassification.AE import AE
from AEClassification.BetaVAE import BetaVAE_EP, BetaVAE_B, InteractionBinaryClassifier

warnings.filterwarnings("ignore")

import os
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def reconstruction_loss(x, x_recon, distribution, criterion):
    if distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = criterion(x_recon, x)
    else:
        raise NotImplementedError

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, **kwargs):
        self.device = 'cuda' if kwargs['use_cuda'] and torch.cuda.is_available() else 'cpu'
        self.max_iter = kwargs['max_iter']

        self.z_dim = kwargs['z_dim']
        self.beta = kwargs['beta']
        self.objective = kwargs['objective']
        self.model = kwargs['model']
        self.lr = kwargs['lr']
        self.data_loader: DataLoader = kwargs['data_loader']

        self.decoder_dist = 'gaussian'

        if kwargs['model'] == 'H':
            net = BetaVAE_EP
        else:
            raise NotImplementedError('only support model H or B')

        self.x_0_length = self.data_loader.dataset[0][0].shape[-1]
        self.x_1_length = self.data_loader.dataset[0][1].shape[-1]

        self.net_0: nn.Module = net(z_dim=self.z_dim, input_length=self.x_0_length).to(self.device)
        self.optim_0 = optim.Adam(self.net_0.parameters(), lr=self.lr)

        self.net_1: nn.Module = net(z_dim=self.z_dim, input_length=self.x_1_length).to(self.device)
        self.optim_1 = optim.Adam(self.net_1.parameters(), lr=self.lr)
        self.recon_criterion = torch.nn.MSELoss()

        # self.classifier = InteractionBinaryClassifier(2 * self.z_dim)

    def train(self):
        recon_losses_0 = []
        total_klds_0 = []
        recon_losses_1 = []
        total_klds_1 = []
        self.net_mode(train=True)
        out = False

        for epoch in range(self.max_iter):
            mini_batch_i = 0
            pbar = tqdm(total=math.ceil(len(self.data_loader.dataset) / self.data_loader.batch_size),
                        desc='Training')
            pbar.update(mini_batch_i)
            batch_recon_losses_0 = []
            batch_total_klds_0 = []
            batch_recon_losses_1 = []
            batch_total_klds_1 = []

            for x_0, x_1, y in self.data_loader:
                mini_batch_i += 1
                pbar.update(1)

                # enhancer VAE
                x_0_recon, mu_0, logvar_0 = self.net_0(x_0)
                recon_loss_0 = reconstruction_loss(x_0, x_0_recon, self.decoder_dist, self.recon_criterion)
                total_kld_0, dim_wise_kld_0, mean_kld_0 = kl_divergence(mu_0, logvar_0)
                beta_vae_loss_0 = recon_loss_0 + self.beta * total_kld_0

                self.optim_0.zero_grad()
                beta_vae_loss_0.backward()
                self.optim_0.step()

                # promotor VAE
                x_1_recon, mu_1, logvar_1 = self.net_1(x_1)
                recon_loss_1 = reconstruction_loss(x_1, x_1_recon, self.decoder_dist, self.recon_criterion)
                total_kld_1, dim_wise_kld_1, mean_kld_1 = kl_divergence(mu_1, logvar_1)
                beta_vae_loss_1 = recon_loss_1 + self.beta * total_kld_1

                self.optim_1.zero_grad()
                beta_vae_loss_1.backward()
                self.optim_1.step()

                pbar.set_description('[{}] Enhancer: recon_loss:{:.5f} total_kld:{:.5f} , Promoter: recon_loss:{:.5f} total_kld:{:.5f}'.format(
                    mini_batch_i, recon_loss_0.item(), total_kld_0.item(), recon_loss_1.item(), total_kld_1.item()))
                batch_recon_losses_0.append(recon_loss_0.item())
                batch_total_klds_0.append(total_kld_0.item())
                batch_recon_losses_1.append(recon_loss_1.item())
                batch_total_klds_1.append(total_kld_1.item())

            recon_losses_0.append(np.mean(batch_recon_losses_0))
            total_klds_0.append(np.mean(batch_total_klds_0))
            recon_losses_1.append(np.mean(batch_recon_losses_1))
            total_klds_1.append(np.mean(batch_total_klds_1))

            print("Epoch {} - Enhancer: recon loss={:.3f}, total KLD={:.5f} , Promoter: recon loss={:.3f}, "
                  "total KLD={:.5f}".format(epoch, np.mean(batch_recon_losses_0), np.mean(batch_total_klds_0),
                                            np.mean(batch_recon_losses_1), np.mean(batch_total_klds_1)))
            pbar.close()
        return {'recon_loss_0': recon_losses_0, 'total_klds_0': total_klds_0, 'recon_losses_1': recon_losses_1, 'total_klds_1': total_klds_1}


    def viz_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(cuda(fixed_img1, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(cuda(fixed_img2, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.mini_batch_i)

            if self.viz_on:
                self.viz.images(samples, env=self.viz_name+'_traverse',
                                opts=dict(title=title), nrow=len(interpolation))

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.mini_batch_i))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net_0.train()
            self.net_1.train()
        else:
            self.net_0.eval()
            self.net_1.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.mini_batch_i,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.mini_batch_i))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.mini_batch_i = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.mini_batch_i))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))