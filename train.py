import os
import pandas as pd
import numpy as np
import torch
from statistics import mean

from configuration import ini_parser
from Dataset import CustomDataset
from model import EmbeddingNet

torch.manual_seed(20)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_data_loader(config):
    "Create Dataloaders"

    train_generator = CustomDataset(config['paths']['dataset'])
    train_loader = torch.utils.data.DataLoader(train_generator, batch_size= 2000, num_workers = 4, shuffle=True)
    print('training dataloader created')
    return train_generator, train_loader 


def train(net, train_loader, epoch, criterion, optimizer):
    
    train_loss = AverageMeter()
    for i, (data, targets) in enumerate(train_loader):
        data = data.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        output = net.forward(data[:, 0], data[:, 1])
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), data.size(0))
        print(train_loss.val)
    return mean(train_loss.history)

def main():
    configpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration.ini')
    ini_config = ini_parser(configpath)
    generator, train_loader = get_data_loader(ini_config)
    (n_users, n_movies), (user_to_index, movie_to_index) = generator.get_dataset_encoding()
    net = EmbeddingNet(n_users, n_movies, n_factors=150, hidden = 100, dropouts = 0.5)
    if torch.cuda.is_available():
        net.cuda()
    epochs = 100
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001, weight_decay=1e-5)
    for epoch in range(epochs):
        net.train()
        loss = train(net, train_loader, epoch, criterion, optimizer)
        print(f'Epoch:{epoch} ,loss:{loss} ')

if __name__ == '__main__':
    print(f'Number of available GPUs : {torch.cuda.device_count()}')
    main()






