import torch
import torch.nn as nn
from models.layers import *
import torch.nn.functional as F

class StableClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=10, eps=0.):
        super().__init__()
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.f3 = nn.Linear(hidden_dim, output_dim, bias=False)

        self.f4 = nn.Linear(hidden_dim, output_dim)
        # self.merge = MergeTemporalDim(int(T/dt))
        # self.expand = ExpandTemporalDim(int(T/dt))
        self.eps = eps

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # def g(self, input):
    #     z1 = F.leaky_relu(self.bn1(self.f1(input)))
    #     return self.bn2(self.f2(input)) + F.linear(z1, F.softplus(self.f3.weight))

    def forward(self, input):
        T = input.shape[0]
        input = input.flatten(0, 1).contiguous()
        x = F.relu(self.bn1(self.f1(input)))
        x = self.f4(x)
        return x
        
        
        # print(input.shape)
        # T = input.shape[0]
        # input = input.flatten(0, 1).contiguous()
        # z1 = F.tanh(self.bn1(self.f1(input)))
        # z2 = self.bn2(self.f2(input)) + F.linear(z1, F.softplus(self.f3.weight))

        
        # # V = F.softplus(z2).sum() #+ self.eps * torch.norm(input)
        # # return z1, z2, V # hidden_output, network_output, V
        # V = z2.sum()
        # return V