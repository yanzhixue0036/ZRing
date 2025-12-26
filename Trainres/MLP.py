import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim):
        input = input_dim
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, input*8),  # 1
            nn.BatchNorm1d(input * 8),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.Linear(input*8, input*8),  # 2
            nn.BatchNorm1d(input * 8),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.Linear(input*8, input*8),  # 3
            nn.BatchNorm1d(input * 8),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.Linear(input*8, input*8),  # 4
            nn.BatchNorm1d(input * 8),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.Linear(input*8, input*8),  # 5
            nn.BatchNorm1d(input * 8),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.Linear(input*8, input*2),  # 1
            nn.BatchNorm1d(input * 2),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.Linear(input*2, input//2),  # 2
            nn.BatchNorm1d(input//2),
            nn.LeakyReLU(negative_slope=0.01),
            
            nn.Linear(input//2, input//8),  # 3
            nn.BatchNorm1d(input//8),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(input//8, 1)
        )

    def forward(self, x):
        return self.model(x)
    