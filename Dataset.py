import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    '''Custom data loader'''

    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        data = self.input[index]
        label = self.output[index]
        label = torch.tensor(float(label)).view(-1)
        return data, label




