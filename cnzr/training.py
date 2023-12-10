import torch
from torch import nn



class ModelV1(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, out_features),
        )


    def forward(self, x):
        x = self.network(x)
        x = torch.softmax(x, dim=1)
        return x

