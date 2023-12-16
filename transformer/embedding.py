import torch
from torch import nn

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, num_positions=1024):
        super(PositionalEmbedding, self).__init__()

    def forward(self, src):
        None  # TODO

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super(TransformerEmbedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        return self.dropout(self.embedding(src) + self.positional_embedding(src))