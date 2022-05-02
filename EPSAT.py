import pickle

import h5py
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from AEClassification.EPIDataset import EPIDataset

torch.manual_seed(42)
np.random.seed(42)

class AttentionNet(nn.Module):
    def __init__(self, device):
        super(AttentionNet,self).__init__()

        # model parameters
        self.device = device
        self.input_channels = 4
        self.enhancer_length = 3000 #
        self.promoter_length = 2000 #
        self.n_kernels = 200 # Number of kernels; used to be 1024
        self.filter_length = 13 #SPEID says 40 # Length of each kernel
        self.cnn_pool_size = 6 # from SATORI
        self.LSTM_out_dim = 50 # Output direction of ONE DIRECTION of LSTM; used to be 512
        self.dense_layer_size = 800
        self.drop_out = 0.2

        self.RNN_hiddenSize = 100
        self.n_rnn_layers = 2
        self.lstm_dropout_p = 0.4

        self.SingleHeadSize = 32
        self.numMultiHeads = 8
        self.MultiHeadSize = 100
        self.genPAttn = True

        self.readout_strategy = 'normalize'

        self.batch_size=256 #batch size
        self.num_epochs=50 #number of epochs


        # Convolutional/maxpooling layers to extract prominent motifs
        # Separate identically initialized convolutional layers are trained for
        # enhancers and promoters
        # Define enhancer layers

        self.enhancer_conv_layer = nn.Sequential(
                        nn.Conv1d(in_channels=self.input_channels,out_channels=self.n_kernels, kernel_size = self.filter_length, padding = 0, bias= False),
                        nn.BatchNorm1d(num_features=self.n_kernels), # tends to help give faster convergence: https://arxiv.org/abs/1502.03167
                        # nn.Dropout2d(), # popular form of regularization: https://jmlr.org/papers/v15/srivastava14a.html
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=self.cnn_pool_size),
                        nn.Dropout(p=self.drop_out)
                    )

        self.promoter_conv_layer = nn.Sequential(
                        nn.Conv1d(in_channels=self.input_channels,out_channels=self.n_kernels, kernel_size=self.filter_length, padding = 0, bias= False),
                        nn.BatchNorm1d(num_features=self.n_kernels), # tends to help give faster convergence: https://arxiv.org/abs/1502.03167
                        # nn.Dropout2d(), # popular form of regularization: https://jmlr.org/papers/v15/srivastava14a.html
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=self.cnn_pool_size),
                        nn.Dropout(p=self.drop_out)
                    )
        self.lstm_layer = nn.Sequential(
                                nn.LSTM(self.n_kernels, self.RNN_hiddenSize, num_layers=self.n_rnn_layers, bidirectional=True)
                            )
        self.lstm_dropout = nn.Dropout(p=self.lstm_dropout_p)

        self.Q = nn.ModuleList(
            [nn.Linear(in_features=2 * self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in
             range(0, self.numMultiHeads)])
        self.K = nn.ModuleList(
            [nn.Linear(in_features=2 * self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in
             range(0, self.numMultiHeads)])
        self.V = nn.ModuleList(
            [nn.Linear(in_features=2 * self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in
             range(0, self.numMultiHeads)])

        self.RELU = nn.ModuleList([nn.ReLU() for i in range(0, self.numMultiHeads)])
        self.MultiHeadLinear = nn.Linear(in_features=self.SingleHeadSize * self.numMultiHeads,
                                         out_features=self.MultiHeadSize)  # 50
        self.MHReLU = nn.ReLU()

        self.fc3 = nn.Linear(in_features=self.MultiHeadSize, out_features=2)

    def attention(self, query, key, value, mask=None, dropout=0.0):
        #based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim = -1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn


    def forward(self, input_p, input_e):
        batch_size = input_p.shape[0]
        #First we need to do CNN for each promoter and enhancer sequences
        p_output = self.enhancer_conv_layer(input_p)
        e_output = self.promoter_conv_layer(input_e)
        # Now Merge the two layers

        output = torch.cat([p_output, e_output], dim=2)
        output = output.permute(0, 2, 1)
        #not sure why this, but the SATORI authors

        output, _ = self.lstm_layer(output)
        F_RNN = output[:,:,:self.RNN_hiddenSize]
        R_RNN = output[:,:,self.RNN_hiddenSize:]
        output = torch.cat((F_RNN,R_RNN),2)
        output = self.lstm_dropout(output)

        pAttn_concat = torch.Tensor([]).to(self.device)
        attn_concat = torch.Tensor([]).to(self.device)
        for i in range(0,self.numMultiHeads):
            query, key, value = self.Q[i](output), self.K[i](output), self.V[i](output)
            attnOut,p_attn = self.attention(query, key, value, dropout=0.2)
            attnOut = self.RELU[i](attnOut)
            # if self.usepooling:
            #     attnOut = self.MAXPOOL[i](attnOut.permute(0,2,1)).permute(0,2,1)
            attn_concat = torch.cat((attn_concat,attnOut),dim=2)
            if self.genPAttn:
                pAttn_concat = torch.cat((pAttn_concat, p_attn), dim=2)

        output = self.MultiHeadLinear(attn_concat)
        output = self.MHReLU(output)

        if self.readout_strategy == 'normalize':
            output = output.sum(axis=1)
            output = (output-output.mean())/output.std()

        output = self.fc3(output)
        output = nn.Softmax(dim=1)(output)
        assert not torch.isnan(output).any()
        if self.genPAttn:
            return output,pAttn_concat
        else:
            return output

if __name__ == '__main__':
    use_cuda = True
    batch_size = 64
    epochs = 50
    lr = 1e-3
    train_ratio = 0.9

    cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
    data_path = 'data/all_sequence_data.h5'

    device= 'cuda' if use_cuda else 'cpu'

    training_histories = {}
    for cell_line in cell_lines:
        best_f1_so_far = 0.
        dataset = EPIDataset(data_path, cell_line, use_cuda=use_cuda, is_onehot_labels=True)
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        train_losses = []
        train_f1s = []
        val_losses = []
        val_f1s = []

        net = AttentionNet(device).to(device)
        print('Create AttentionNet with {} parameters'.format(sum(p.numel() for p in net.parameters())))
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        criteria = torch.nn.CrossEntropyLoss()
        print('Training on {} samples, validating on {} samples'.format(len(train_data_loader.dataset), len(val_data_loader.dataset)))
        try:
            for epoch in range(epochs):
                mini_batch_i = 0
                mini_batch_i_val = 0

                pbar = tqdm(total=math.ceil(len(train_data_loader.dataset) / train_data_loader.batch_size),
                            desc='Training EPI Net')
                pbar.update(mini_batch_i)
                batch_losses_train = []
                batch_f1_train = []
                net.train()
                for input_p, input_e, y in train_data_loader:
                    mini_batch_i += 1
                    pbar.update(1)

                    y_pred = net(input_p, input_e)[0]
                    loss = criteria(y, y_pred)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    predictions = torch.argmax(y_pred, dim=1)
                    batch_f1_train.append(f1_score(torch.argmax(y, dim=1).detach().cpu().numpy(), predictions.detach().cpu().numpy(), average='macro', zero_division=1))

                    pbar.set_description('Training [{}] loss:{:.5f}'.format(mini_batch_i, loss.item()))
                    batch_losses_train.append(loss.item())
                train_f1s.append(np.mean(batch_f1_train))
                train_losses.append(np.mean(batch_losses_train))
                pbar.close()

                net.eval()
                with torch.no_grad():
                    pbar = tqdm(total=math.ceil(len(val_data_loader.dataset) / val_data_loader.batch_size),
                                desc='Validating EPI Net')
                    pbar.update(mini_batch_i_val)
                    batch_losses_val = []
                    batch_f1_val = []
                    for input_p, input_e, y in val_data_loader:
                        mini_batch_i_val += 1
                        pbar.update(1)

                        y_pred = net(input_p, input_e)[0]
                        loss = criteria(y, y_pred)
                        predictions = torch.argmax(y_pred, dim=1)
                        batch_losses_val.append(loss.item())
                        batch_f1_val.append(
                            f1_score(torch.argmax(y, dim=1).detach().cpu().numpy(), predictions.detach().cpu().numpy(),
                                     average='macro', zero_division=1))
                        pbar.set_description('Validating [{}] loss:{:.5f}'.format(mini_batch_i, loss.item()))

                    val_f1s.append(np.mean(batch_f1_val))
                    val_losses.append(np.mean(batch_losses_val))
                    pbar.close()
                print("Epoch {} - train loss:{:.5f}, train f1:{:.3f}, val loss:{:.5f}, val f1:{:.3f}".format(epoch, np.mean(batch_losses_train), train_f1s[-1], np.mean(batch_losses_val), val_f1s[-1]))

                if val_f1s[-1] > best_f1_so_far:
                    torch.save(net.state_dict(), 'models/satori_net_{}'.format(cell_line))
                    print('Best f1 improved from {} to {}, saved best model to {}'.format(best_f1_so_far, val_f1s[-1], 'models/net_{}'.format(cell_line)))
                    best_f1_so_far = val_f1s[-1]

                # Save training histories after every epoch
                training_histories[cell_line] = {'train_losss': train_losses, 'train_f1': train_f1s,
                                                 'val_losses': val_losses, 'val_f1': val_f1s}
                pickle.dump(training_histories, open('models/EPI_training_histories.pickle', 'wb'))

        except:
            print('Training terminated for cell line {} because of exploding gradient'.format(cell_line))
        print('Training completed for cell line {}, training history saved'.format(cell_line))

