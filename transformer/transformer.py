import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, d_ffn, dropout):
        super(Transformer, self).__init__()

        self.encoder = Encoder(d_model, nhead, num_encoder_layers, d_ffn)
        self.decoder = Decoder(d_model, nhead, num_decoder_layers, d_ffn)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, mem_mask=None):
        # src: [batch_size, src_sequence_length, d_model]
        # tgt: [batch_size, tgt_sequence_length, d_model]
        # src_mask: [src_sequence_length, src_sequence_length]
        # tgt_mask: [tgt_sequence_length, tgt_sequence_length]
        # mem_mask: [tgt_sequence_length, src_sequence_length]

        encoder_outputs = self.encoder(src, src_mask)
        outputs = self.decoder(encoder_outputs, tgt, tgt_mask, mem_mask)
        return outputs

    def generate_square_subsequent_mask(self, sz, device='cpu', dtype=torch.float32):
        return torch.tril(torch.ones(size=(sz, sz), device=device, dtype=dtype))