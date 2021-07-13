from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

from torch import Tensor
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TransformerModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_length: int,
                 num_classes: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 device=None,
                 dtype=None
                 ) -> None:
        super(TransformerModel, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                   activation, layer_norm_eps, batch_first,
                                                   **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.classifizer = nn.Linear(max_length * d_model, num_classes)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src_seq_len = src.size(1)
        # atten_mask seq_len * seq_len
        src_mask = torch.zeros((src_seq_len, src_seq_len), device="cpu").type(torch.bool)
        # padding_mask batch_size * seq_len
        src_key_padding_mask = (src == 0)

        output = self.embedding(src)
        output = self.positional_encoding(output)

        output = self.encoder(output, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = output.view(output.size(0), -1)
        output = self.classifizer(output)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
