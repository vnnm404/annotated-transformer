import torch
from torch import nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, nhead)
        self.self_attention_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ffn)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        # src: [batch_size, src_sequence_length, d_model]
        # src_mask: [src_sequence_length, src_sequence_length]

        residual = src

        src = self.self_attention(query=src, key=src, value=src, mask=src_mask)
        src = self.self_attention_layer_norm(src + residual)

        residual = src

        src = self.feed_forward(src)
        src = self.feed_forward_layer_norm(src + residual)

        return src

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, d_ffn):
        super(Encoder, self).__init__()

        self.encoders = nn.ModuleList([EncoderLayer(d_model, nhead, d_ffn) for _ in range(num_encoder_layers)])

    def forward(self, src, src_mask=None):
        # src: [batch_size, src_sequence_length, d_model]
        # src_mask: [src_sequence_length, src_sequence_length]

        for encoder in self.encoders:
            src = encoder(src, src_mask)

        return src