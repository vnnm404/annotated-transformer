import math
import torch
from torch import nn
from torch.nn import functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        # query: [batch_size, output_sequence_length, d_query]
        # key: [batch_size, input_sequence_length, d_key]
        # value: [batch_size, input_sequence_length, d_value]
        # mask: [output_sequence_length, input_sequence_length]

        d_key = key.size(2)
        batch_size = query.size(0)

        # query: [batch_size, output_sequence_length, d_query]
        # key: [batch_size, input_sequence_length, d_key]
        score = torch.einsum('bij,bkj->bik', query, key)  # [batch_size, output_sequence_length, input_sequence_length]
        score = score * (1 / math.sqrt(d_key))

        if mask is not None:
            mask = mask.unsqueeze(dim=0).repeat(batch_size, 1, 1)
            score = score.masked_fill(mask == 0, -10000)
        
        score = F.softmax(score, dim=2)

        # score: [batch_size, output_sequence_length, input_sequence_length]
        # value: [batch_size, input_sequence_length, d_value]
        output = torch.einsum('bij,bjk->bik', score, value)  # [batch_size, output_sequence_length, d_value]
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()

        # This is a bad implementation, all nheads can be done in parallel with a clever trick.
        # TODO implement parallel multi-head-attention

        if d_model % nhead != 0:
            raise ValueError('nhead % d_model != 0')

        self.nhead = nhead
        self.h_dim = d_model // nhead
        self.scaled_dot_product = ScaledDotProductAttention()
        self.W_query = nn.ModuleList([nn.Linear(d_model, self.h_dim) for _ in range(nhead)])
        self.W_key = nn.ModuleList([nn.Linear(d_model, self.h_dim) for _ in range(nhead)])
        self.W_value = nn.ModuleList([nn.Linear(d_model, self.h_dim) for _ in range(nhead)])
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query: [batch_size, output_sequence_length, d_model]
        # key: [batch_size, input_sequence_length, d_model]
        # value: [batch_size, input_sequence_length, d_model]
        # mask: [output_sequence_length, input_sequence_length]

        outputs = []

        for i in range(self.nhead):
            query_i = self.W_query[i](query)  # [batch_size, output_sequence_length, h_dim]
            key_i = self.W_key[i](key)  # [batch_size, input_sequence_length, h_dim]
            value_i = self.W_value[i](value)  # [batch_size, input_sequence_length, h_dim]

            output = self.scaled_dot_product(query=query_i, key=key_i, value=value_i, mask=mask)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=2)  # [batch_size, output_sequence_length, d_model]
        outputs = self.projection(outputs)  # [batch_size, output_sequence_length, d_model]
        return outputs