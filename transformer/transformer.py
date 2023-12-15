import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, d_ffn, dropout, layer_norm_eps=1e-05):
        super(Transformer, self).__init__()
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        None

    def generate_square_subsequent_mask(sz, device='cpu', dtype=torch.float32):
        None