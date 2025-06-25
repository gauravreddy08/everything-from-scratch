from SelfAttention import SelfAttention
from torch import nn
import torch

class NaiveMultiHeadAttention(nn.Module):
  def __init__(self, in_dim, out_dim, num_heads):
    super().__init__()

    self.heads = nn.ModuleList([SelfAttention(in_dim, out_dim//num_heads) for _ in range(num_heads)])
    self.linear_layer = nn.Linear(num_heads * (out_dim//num_heads), out_dim)

  def forward(self, x):
    out = torch.concat([head(x) for head in self.heads], dim=-1)
    return self.linear_layer(out)
