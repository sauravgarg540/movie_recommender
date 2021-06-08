import os
import pandas as pd
import numpy as np
import torch
import math
from statistics import mean
from sklearn.model_selection import train_test_split

from model import EmbeddingNet
from dataset import CustomDataset
from utils import AverageMeter, prepare_data, ini_parser, check_paths

# set manual seed for easy judgement while hyper parameters tuning
torch.manual_seed(20)

def get_data_loader(X_train, X_test, y_train, y_test, config):
    "Create Dataloaders"
    train_generator = CustomDataset(X_train, y_train)
    val_generator = CustomDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_generator, batch_size= int(config['batchsize']), num_workers = 4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_generator,batch_size =int(config['batchsize']), num_workers = 4, shuffle=True)
    print('Dataloader created successfully')
    return train_loader, val_loader


def train(net, train_loader, epoch, criterion, optimizer):
    '''Training procedure for EmbbedingNet'''

    train_loss = AverageMeter()
    net.train()
    for i, (input_, target) in enumerate(train_loader):
        input_ = input_.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = net.forward(input_[:, 0], input_[:, 1])
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), input_.size(0))
    return mean(train_loss.history)

def validate(net, val_loader, epoch, criterion):
    '''Validation procedure for EmbbedingNet'''

    val_loss = AverageMeter()
    net.eval()
    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            input_ = input_.cuda()
            target = target.cuda()
            output = net.forward(input_[:, 0], input_[:, 1])
            loss = torch.sqrt(criterion(output, target))
            val_loss.update(loss.item(), input_.size(0))
    return mean(val_loss.history)

def main():

    configpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration.ini')
    config = ini_parser(configpath)
    print(config)
    (x,y),(n_users,n_movies),_,_ = prepare_data(config)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=20)
    print('dataframe loaded')

    train_loader, val_loader = get_data_loader( X_train, X_test, y_train, y_test, config)
    net = EmbeddingNet(n_users, n_movies, n_factors=config['n_factors'])
    # net.load_state_dict("checkpoints/checkpoint_4.pt")
    if torch.cuda.is_available():
        net.cuda()
    epochs = config['epochs']

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = config['lr'], weight_decay = config['wd'])
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(epochs):
            train_loss = train(net, train_loader, epoch, criterion, optimizer)
            val_loss = validate(net, val_loader, epoch, criterion)
            print(f'Epoch:{epoch}------> train loss: {train_loss}....Validation loss(RMSE): {val_loss}')
            check_paths(destination = config['checkpoint_path'])
            torch.save({'model_state_dict': net.state_dict(),}, f"checkpoints/checkpoint_{epoch}.pt")

if __name__ == '__main__':
    main()








