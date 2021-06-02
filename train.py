import os
import pandas as pd
import numpy as np
import torch

from configuration import ini_parser
from Dataset import CustomDataset
torch.manual_seed(20)

def get_data_loader(config):
    "Create Dataloaders"

    train_generator = CustomDataset(ini_config['paths']['dataset'])
    train_loader = torch.utils.data.DataLoader(train_generator, batch_size= 4, num_workers = 4, shuffle=True)
    print('training dataloader created')
    return train_loader 


if __name__ == '__main__':
    configpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration.ini')
    ini_config = ini_parser(configpath)
    train_loader = get_data_loader(ini_config)

    for i, (data, targets) in enumerate(train_loader):
        print(data, targets)
        exit()





