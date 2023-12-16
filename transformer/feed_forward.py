import torch
from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ffn):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, input):
        return self.linear2(F.relu(self.linear1(input)))