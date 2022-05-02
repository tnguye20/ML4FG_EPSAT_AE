import math
import pickle
import random

import numpy as np
import torch
import torch as torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from AEClassification.AECNN import AE_CNN
from AEClassification.BetaVAE import BetaVAE_EP
from AEClassification.EPIDataset import EPIDataset
from AEClassification.LogisticRegression import LogisticRegression

data_path = 'data/all_sequence_data.h5'
device = 'cuda'
use_cuda = True
model_path = 'AEClassification/models/net_AE_GM12878'
cell_line = 'GM12878'
z_dim = 512
train_ratio = 0.9  # fraction of data to use for training
epochs = 1000
lr = 1e-3
batch_size = 512

dataset = EPIDataset(data_path, cell_line, use_cuda=use_cuda, is_onehot_labels=True)
data_loader = DataLoader(dataset, batch_size=512, shuffle=True)
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

x_p_length, x_e_length = dataset[0][0].shape[-1], dataset[0][1].shape[-1]

ae_model = AE_CNN(promoter_input_length=x_p_length, enhancer_input_length=x_e_length, z_dim=10).to(device)
ae_model.load_state_dict(torch.load(model_path))
ae_model.eval() # set the model to eval mode


# test signal reconstruction
# rand_idx = random.randint(1, len(dataset) - 1)
# random_sample = data_loader.dataset.__getitem__(rand_idx)[:2]
# random_sample = [x.unsqueeze(0) for x in random_sample]# add the batch dimension
# random_sample_z = vae_model._encode(*random_sample)[:, :z_dim]
# random_sample_recon = [torch.sigmoid(x) for x in vae_model._decode(random_sample_z)]
#
# plt.imshow(random_sample[0].squeeze(0).detach().cpu().numpy()[:, :10])
# plt.show()
# plt.imshow(random_sample_recon[0].squeeze(0).detach().cpu().numpy()[:, :10], vmin=0, vmax=1)
# plt.show()

# create latent representations

training_histories = {}

# TODO need to iterate through the cell lines
best_f1_so_far = 0.
train_losses = []
train_f1s = []
val_losses = []
val_f1s = []

net = LogisticRegression(input_dim=z_dim * 2, output_dim=2).to(device)
print('Create Logstic regression with {} parameters'.format(sum(p.numel() for p in net.parameters())))
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criteria = torch.nn.CrossEntropyLoss()
print('Training on {} samples, validating on {} samples'.format(len(train_data_loader.dataset),
                                                                len(val_data_loader.dataset)))
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

        with torch.no_grad():
            z = ae_model._encode(input_p, input_e)
        y_pred = net(z)

        loss = criteria(y, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(y_pred, dim=1)
        batch_f1_train.append(
            f1_score(torch.argmax(y, dim=1).detach().cpu().numpy(), predictions.detach().cpu().numpy(),
                     average='macro', zero_division=1))

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

            with torch.no_grad():
                z = ae_model._encode(input_p, input_e)
            y_pred = net(z)

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
    print("Epoch {} - train loss:{:.5f}, train f1:{:.3f}, val loss:{:.5f}, val f1:{:.3f}".format(epoch, np.mean(
        batch_losses_train), train_f1s[-1], np.mean(batch_losses_val), val_f1s[-1]))

    if val_f1s[-1] > best_f1_so_far:
        torch.save(net.state_dict(), 'models/net_BetaVAELogistic_{}'.format(cell_line))
        print('Best f1 improved from {} to {}, saved best model to {}'.format(best_f1_so_far, val_f1s[-1],
                                                                              'models/net_BetaVAELogistic_{}'.format(cell_line)))
        best_f1_so_far = val_f1s[-1]

    # Save training histories after every epoch
    training_histories[cell_line] = {'train_losss': train_losses, 'train_f1': train_f1s,
                                     'val_losses': val_losses, 'val_f1': val_f1s}
    pickle.dump(training_histories, open('models/BetaVAELogistic_training_histories.pickle', 'wb'))
