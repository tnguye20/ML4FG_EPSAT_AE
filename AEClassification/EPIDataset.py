import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EPIDataset(Dataset):
    def __init__(self, data_path, cell_line, use_cuda, is_onehot_labels=False):
        with h5py.File(data_path, 'r') as hf:
            self.X_enhancers = np.array(hf.get(cell_line + '_X_enhancers')).transpose((0, 2, 1))
            self.X_promoters = np.array(hf.get(cell_line + '_X_promoters')).transpose((0, 2, 1))
            self.labels = np.array(hf.get(cell_line + 'labels'))
            print(
                "EPIDataset: Cell line {0} has {1} EP-pairs, number of positive samples is {2}, negative is {3}, percentage postive is {4}".format(
                    cell_line, len(self.X_enhancers), np.sum(self.labels == 1), np.sum(self.labels == 0), np.sum(self.labels == 1) / len(self.labels)), end='. ')
            self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

            # change the data to channel first
            self.X_enhancers = np.moveaxis(self.X_enhancers, -1, 1)
            self.X_promoters = np.moveaxis(self.X_promoters, -1, 1)

            # augment the number of positive samples
            num_positives = np.sum(self.labels == 1)  # original number of positive samples
            num_positive_subsamples = round(num_positives * 20)  # number of subsampled positive samples
            np.random.seed(42)
            X_enhancers_positive = self.X_enhancers[self.labels == 1][np.random.choice(num_positives, num_positive_subsamples, replace=True), :, :]
            X_promoters_positive = self.X_promoters[self.labels == 1][np.random.choice(num_positives, num_positive_subsamples, replace=True), :, :]

            self.X_enhancers = np.concatenate([X_enhancers_positive, self.X_enhancers[self.labels == 0]])
            self.X_promoters = np.concatenate([X_promoters_positive, self.X_promoters[self.labels == 0]])
            self.labels = np.concatenate([np.ones(num_positive_subsamples), np.zeros(np.sum(self.labels == 0))])

            # convert to torch tensors
            self.X_enhancers = torch.Tensor(self.X_enhancers).to(self.device)
            self.X_promoters = torch.Tensor(self.X_promoters).to(self.device)
            self.labels = torch.Tensor(self.labels).to(self.device)
            print("after augmentation, percentage positive is {}".format(torch.sum(self.labels == 1).item() / len(self.labels)))

            if is_onehot_labels:
                self.labels = torch.nn.functional.one_hot(self.labels.type(torch.LongTensor), num_classes = 2).type(torch.FloatTensor).to(self.device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X_enhancers[idx],  self.X_promoters[idx], self.labels[idx]

    def get_input_length(self):
        return self.X_enhancers.shape[2]