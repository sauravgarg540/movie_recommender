import torch
import torch.nn as nn
from itertools import zip_longest


def get_list(n):
    if isinstance(n, (int, float)):
        return [n]
    elif hasattr(n, '__iter__'):
        return list(n)
    raise TypeError('layers configuraiton should be a single number or a list of numbers')


class EmbeddingNet(nn.Module):
            
    def __init__(self, n_users, n_movies,n_factors=50, embedding_dropout=0.02, hidden=10, dropouts=0.2):
        
        super().__init__()
            
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.fc1 = nn.Linear(300, 100)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(100, 200)
        self.drop2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(200, 300)
        self.drop3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(300, 10)

        self.u.weight.data.uniform_(-0.05, 0.05)
        self.m.weight.data.uniform_(-0.05, 0.05)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

        
    def forward(self, users, movies, minmax=(0.5,1.0)):
        x = torch.cat([self.u(users), self.m(movies)], dim=1)
        x = self.drop(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop1(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.drop2(x)
        x = nn.functional.relu(self.fc3(x))
        x = self.drop3(x)
        x = self.fc4(x)
        return x