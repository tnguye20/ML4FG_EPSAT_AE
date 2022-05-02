"""model.py"""
import math

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init

from AEClassification.TimsDistributed import TimeDistributed


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def calculate_conv_length(input_length, padding, kernel_size, stride):
    return math.floor(((input_length + 2 * padding - (kernel_size - 1) - 1) / stride) + 1)


class AE_CRNN(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, promoter_input_length, enhancer_input_length, z_dim=128, kernel_size = 13, n_filters=256, stride=2, linear_units=512, lstm_cells=256, n_rnn_layers=2):
        super(AE_CRNN, self).__init__()
        self.z_dim = z_dim
        self.kernel_size = kernel_size
        self.enhancer_input_length = enhancer_input_length
        self.promoter_input_length = promoter_input_length
        self.n_conv_filters = n_filters
        self.stride = stride
        encoder_padding = int((kernel_size - 1) / 2)  # to exactly reduce the sequence length by half after conv
        self.conv1d1_length_enahncer = calculate_conv_length(enhancer_input_length, encoder_padding, kernel_size, stride)
        self.conv1d2_length_enhancer = calculate_conv_length(self.conv1d1_length_enahncer, encoder_padding, kernel_size, stride)
        self.conv1d3_length_enhancer = calculate_conv_length(self.conv1d2_length_enhancer, encoder_padding, kernel_size, stride)

        self.conv1d1_length_promoter = calculate_conv_length(promoter_input_length, encoder_padding, kernel_size, stride)
        self.conv1d2_length_promoter = calculate_conv_length(self.conv1d1_length_promoter, encoder_padding, kernel_size, stride)
        self.conv1d3_length_promoter = calculate_conv_length(self.conv1d2_length_promoter, encoder_padding, kernel_size, stride)

        self.encoder_conved_length = self.conv1d3_length_enhancer + self.conv1d3_length_promoter
        self.encoder_LSTMed_size = self.encoder_conved_length * lstm_cells * 2
        self.lstm_cells = lstm_cells

        self.encoder_conv_promoter = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=n_filters, kernel_size=(self.kernel_size, ), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 1000
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 500
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 150
            nn.LeakyReLU(),
        )

        self.encoder_conv_enhancer = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=n_filters, kernel_size=(self.kernel_size, ), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 1500
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 750
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 375
            nn.LeakyReLU(),
        )

        self.encoder_LSTM = nn.Sequential(
                                nn.LSTM(input_size=self.n_conv_filters, hidden_size=lstm_cells, num_layers=n_rnn_layers, bidirectional=True)
                            )

        self.encoder_linear = nn.Sequential(
            nn.Linear(self.encoder_LSTMed_size, linear_units),  # B, z_dim * 2
            nn.LeakyReLU(),
            nn.Linear(linear_units, z_dim),  # B, z_dim * 2
            # nn.LeakyReLU(),
            # nn.Linear(int(linear_units / 2), int(linear_units / 4)),  # B, z_dim * 2
            # nn.LeakyReLU(),
            # nn.Linear(int(linear_units / 4), z_dim * 2),  # B, z_dim * 2
        )

        self.decoder_linear = nn.ModuleList([nn.Sequential(
            # nn.Linear(z_dim, int(linear_units / 4)),  # B, 48000
            # nn.LeakyReLU(),
            # nn.Linear(int(linear_units / 4), int(linear_units / 2)),
            # nn.LeakyReLU(),
            nn.Linear(z_dim, linear_units),  # B, z_dim * 2
            nn.LeakyReLU(),
            nn.Linear(linear_units, 2 * self.lstm_cells),  # B, z_dim * 2
        ) for i in range(self.encoder_conved_length)])

        self.decoder_LSTM = nn.Sequential(
                                nn.LSTM(input_size=self.lstm_cells * 2, hidden_size=int(self.n_conv_filters / 2), num_layers=n_rnn_layers, bidirectional=True)
                            )

        self.decoder_conv_promoter = nn.Sequential(
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 128, 750
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 128, 1500
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=4, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 4, 3000
        )

        self.decoder_conv_enhancer = nn.Sequential(
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 128, 750
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 128, 1500
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=4, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 4, 3000
        )


    def forward(self, x_p, x_e):
        z = self._encode(x_p, x_e)
        recon_x_p, recon_x_e = self._decode(z)

        return recon_x_p, recon_x_e

    def _encode(self, x_p, x_e):
        encode_conv_x_p = self.encoder_conv_promoter(x_p)
        encode_conv_x_e = self.encoder_conv_enhancer(x_e)
        encode_conv_x = torch.cat((encode_conv_x_p, encode_conv_x_e), 2)
        encode_conv_x = encode_conv_x.permute(0, 2, 1)  # LSTM expects the first dim to be time/seq length
        encode_LSTM_x, _ = self.encoder_LSTM(encode_conv_x)
        distribution = torch.reshape(encode_LSTM_x, (-1, self.encoder_LSTMed_size))  # B, 48000
        distribution = self.encoder_linear(distribution)
        return distribution

    def _decode(self, z):
        decode_linear_z_repeated = torch.concat([l(z)[:, None, :] for i, l in enumerate(self.decoder_linear)], dim=1)
        decode_LSTM_z = self.decoder_LSTM(decode_linear_z_repeated)[0]
        decode_LSTM_z = decode_LSTM_z.permute(0, 2, 1)  # LSTM expects the first dim to be time/seq length
        decode_deconv_x_p, decode_deconv_x_e = decode_LSTM_z[:, :, :self.conv1d3_length_promoter], decode_LSTM_z[:, :, self.conv1d3_length_promoter:]
        recon_x_p = self.decoder_conv_promoter(decode_deconv_x_p)
        recon_x_e = self.decoder_conv_enhancer(decode_deconv_x_e)
        return recon_x_p, recon_x_e