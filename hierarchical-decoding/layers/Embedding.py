import torch
import torch.nn as nn
from utils.util import count_parameters


class Embedding(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, input_size, emb_size, dropout=0.0, norm=False):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(input_size, emb_size)

        print(f'{count_parameters(self.embedding)}')

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(emb_size)
        self.norm = norm


    def forward(self, x):
        x = self.dropout(self.embedding(x))
        if self.norm:
            x = self.layer_norm(x)

        return x
