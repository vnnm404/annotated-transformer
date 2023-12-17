import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttentionSlow(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttentionSlow, self).__init__()

        '''This is an inefficient implementation, all nheads can be done in parallel with a clever trick.'''

        if d_model % nhead != 0:
            raise ValueError("nhead % d_model != 0")

        self.nhead = nhead
        self.h_dim = d_model // nhead
        self.W_query = nn.ModuleList(
            [nn.Linear(d_model, self.h_dim) for _ in range(nhead)]
        )
        self.W_key = nn.ModuleList(
            [nn.Linear(d_model, self.h_dim) for _ in range(nhead)]
        )
        self.W_value = nn.ModuleList(
            [nn.Linear(d_model, self.h_dim) for _ in range(nhead)]
        )
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query: [batch_size, output_sequence_length, d_model]
        # key: [batch_size, input_sequence_length, d_model]
        # value: [batch_size, input_sequence_length, d_model]
        # mask: [output_sequence_length, input_sequence_length]

        outputs = []

        for i in range(self.nhead):
            query_i = self.W_query[i](
                query
            )  # [batch_size, output_sequence_length, h_dim]
            key_i = self.W_key[i](key)  # [batch_size, input_sequence_length, h_dim]
            value_i = self.W_value[i](
                value
            )  # [batch_size, input_sequence_length, h_dim]

            output = self.scaled_dot_product(
                query=query_i, key=key_i, value=value_i, mask=mask
            )
            outputs.append(output)

        outputs = torch.cat(
            outputs, dim=2
        )  # [batch_size, output_sequence_length, d_model]
        outputs = self.projection(
            outputs
        )  # [batch_size, output_sequence_length, d_model]
        return outputs

    def scaled_dot_product(self, query, key, value, mask=None):
        # query: [batch_size, output_sequence_length, d_query]
        # key: [batch_size, input_sequence_length, d_key]
        # value: [batch_size, input_sequence_length, d_value]
        # mask: [output_sequence_length, input_sequence_length]

        d_key = torch.tensor(key.size(2))
        batch_size = query.size(0)

        # query: [batch_size, output_sequence_length, d_query]
        # key: [batch_size, input_sequence_length, d_key]
        score = torch.einsum(
            "bij,bkj->bik", query, key
        )  # [batch_size, output_sequence_length, input_sequence_length]
        score = score * (1 / torch.sqrt(d_key))

        if mask is not None:
            mask = mask.unsqueeze(dim=0).repeat(batch_size, 1, 1)
            score = score.masked_fill(mask == 0, -10000)

        score = F.softmax(score, dim=2)

        # score: [batch_size, output_sequence_length, input_sequence_length]
        # value: [batch_size, input_sequence_length, d_value]
        output = torch.einsum(
            "bij,bjk->bik", score, value
        )  # [batch_size, output_sequence_length, d_value]
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()

        if d_model % nhead != 0:
            raise ValueError("nhead % d_model != 0")

        self.nhead = nhead
        self.h_dim = d_model // nhead
        self.W_query = nn.Linear(d_model, d_model)
        self.W_key = nn.Linear(d_model, d_model)
        self.W_value = nn.Linear(d_model, d_model)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query: [batch_size, output_sequence_length, d_model]
        # key: [batch_size, input_sequence_length, d_model]
        # value: [batch_size, input_sequence_length, d_model]
        # mask: [output_sequence_length, input_sequence_length]

        query = self.split(
            self.W_query(query)
        )  # [batch_size, output_sequence_length, nhead, h_dim]
        key = self.split(
            self.W_key(key)
        )  # [batch_size, input_sequence_length, nhead, h_dim]
        value = self.split(
            self.W_value(value)
        )  # [batch_size, input_sequence_length, nhead, h_dim]

        # outputs: [batch_size, output_sequence_length, nhead, h_dim]
        outputs = self.scaled_dot_product(query=query, key=key, value=value, mask=mask)
        outputs = self.combine(outputs)  # [batch_size, output_sequence_length, d_model]

        outputs = self.projection(outputs)
        return outputs

    def scaled_dot_product(self, query, key, value, mask=None):
        # query: [batch_size, output_sequence_length, nhead, h_dim]
        # key: [batch_size, output_sequence_length, nhead, h_dim]
        # value: [batch_size, output_sequence_length, nhead, h_dim]
        # mask: [output_sequence_length, input_sequence_length]

        d_key = torch.tensor(key.size(2))
        batch_size = query.size(0)

        # query: [batch_size, output_sequence_length, nhead, h_dim]
        # key: [batch_size, output_sequence_length, nhead, h_dim]
        score = torch.einsum(
            "bihd,bkhd->bhik", query, key
        )  # [batch_size, nhead, output_sequence_length, input_sequence_length]
        score = score * (1 / torch.sqrt(d_key))

        if mask is not None:
            mask = (
                mask.unsqueeze(dim=0)
                .unsqueeze(dim=0)
                .repeat(batch_size, self.nhead, 1, 1)
            )
            score = score.masked_fill(mask == 0, -10000)

        score = F.softmax(score, dim=2)

        # score: [batch_size, nhead, output_sequence_length, input_sequence_length]
        # value: [batch_size, input_sequence_length, nhead, h_dim]
        output = torch.einsum(
            "bhij,bjhk->bihk", score, value
        )  # [batch_size, output_sequence_length, nhead, h_dim]
        return output

    def split(self, tensor):
        batch_size, sequence_length, d_tensor = tensor.size()
        tensor = tensor.reshape(batch_size, sequence_length, self.nhead, self.h_dim)
        return tensor

    def combine(self, tensor):
        batch_size, sequence_length, nhead, h_dim = tensor.size()
        d_model = nhead * h_dim
        tensor = tensor.reshape(batch_size, sequence_length, d_model)
        return tensor
