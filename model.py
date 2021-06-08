import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):
            
    def __init__(self, n_users, n_movies,n_factors=50):
        
        super().__init__()
            
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.drop = nn.Dropout(0.02)
        self.fc1 = nn.Linear(2*n_factors, 500)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, 100)
        self.fc5 = nn.Linear(100, 500)
        self.fc6 = nn.Linear(500, 1)

        self.u.weight.data.uniform_(-0.05, 0.05)
        self.m.weight.data.uniform_(-0.05, 0.05)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.kaiming_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
        nn.init.kaiming_uniform_(self.fc5.weight)
        nn.init.zeros_(self.fc5.bias)
        nn.init.xavier_normal_(self.fc6.weight)
        nn.init.zeros_(self.fc6.bias)

        
    def forward(self, users, movies, minmax=[0,5.5]):
        x = torch.cat([self.u(users), self.m(movies)], dim=1)
        x = self.drop(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop1(x)
        x = nn.functional.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = self.drop1(x)
        x = nn.functional.relu(self.fc5(x))
        x = self.drop1(x)
        x = torch.sigmoid(self.fc6(x))
        x = x*(minmax[1] - minmax[0]) + minmax[0]
        return x