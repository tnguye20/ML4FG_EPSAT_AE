import math

import torch
from torch import nn

from AEClassification.BetaVAE import View


class AE(nn.Module):
    def __init__(self, input_length, z_dim=128, kernel_size = 13, n_filters=128, stride=2):
        super(AE,self).__init__()

        self.z_dim = z_dim
        self.kernel_size = kernel_size
        self.input_length = input_length
        self.n_filters = n_filters
        self.stride = stride
        encoder_padding = int((kernel_size - 1) / 2)  # to exactly reduce the sequence length by half after conv
        self.conv1d1_length = math.floor(((input_length + 2 * encoder_padding - (kernel_size-1) - 1) / stride) + 1)
        self.conv1d2_length = math.floor(((self.conv1d1_length + 2 * encoder_padding - (kernel_size-1) - 1) / stride) + 1)
        self.conv1d3_length = math.floor(((self.conv1d2_length + 2 * encoder_padding - (kernel_size-1) - 1) / stride) + 1)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=n_filters, kernel_size=(self.kernel_size, ), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 1500
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 750
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 375
            nn.LeakyReLU(),
            View((-1, self.conv1d3_length * n_filters)),  # B, 48000
            nn.Linear(self.conv1d3_length * n_filters, z_dim),  # B, z_dim
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, self.conv1d3_length * n_filters),  # B, 48000
            View((-1, n_filters, self.conv1d3_length)),  # B, 128, 375
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 128, 750
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 128, 1500
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=4, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 4, 3000
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed