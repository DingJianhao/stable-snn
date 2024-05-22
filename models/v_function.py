import torch
import torch.nn as nn
import torch.nn.functional as F

class VFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim=100) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1, bias=False)
        # torch.nn.init.orthogonal_(self.fc.weight,gain=1)
        # torch.nn.init.orthogonal_(self.W.weight,gain=1)
    
    def forward(self, x):# T, batch, ...
        # T, batch = x.shape[0], x.shape[1]
        # x = x.reshape(T*batch, -1)
        # y = self.bn(F.relu(self.fc(x)))
        # # print(y.shape)
        # return (y * self.W(y)).sum()#.sum(dim=1).reshape(T, batch)

        T, batch = x.shape[0], x.shape[1]
        x = x.reshape(T*batch, -1)
        h_1 = torch.tanh(self.layer1(x))
        out = torch.tanh(self.layer2(self.bn(h_1)))
        # out = torch.tanh(self.layer2(h_1))
        return out
    
    @staticmethod
    def dtanh(s):
        # Derivative of activation
        return 1.0 - s**2