from torch import nn
import torch

class MultiHeadAttention(nn.Module):
  def __init__(self, in_dim, out_dim, num_heads, dropout=0.2, bias=False):
    super().__init__()
    self.K = nn.Linear(in_dim, out_dim, bias=bias)
    self.Q = nn.Linear(in_dim, out_dim, bias=bias)
    self.V = nn.Linear(in_dim, out_dim, bias=bias)

    self.num_heads = num_heads
    self.in_dim = in_dim
    self.out_dim = out_dim

  def forward(self, x):
    b, input_length, embed_dim = x.shape

    key = self.K(x)
    query = self.Q(x)
    value = self.V(x)

    context_length = x.shape[1]

    key = key.view(b, input_length, self.num_heads, self.out_dim//self.num_heads).transpose(1, 2)
    query = query.view(b, input_length, self.num_heads, self.out_dim//self.num_heads).transpose(1, 2)
    value = value.view(b, input_length, self.num_heads, self.out_dim//self.num_heads).transpose(1, 2)

    out = (query @ key.transpose(2, 3))/ key.shape[-1]**0.5
    mask = torch.tril(torch.ones(context_length, context_length))
    out_masked = out.masked_fill(mask == 0, -torch.inf)
    softmax_kq = torch.nn.functional.softmax(out_masked, dim=-1)
    out = softmax_kq @ value
    out = out.transpose(1, 2).contiguous()

    return out.view(x.shape[0], x.shape[1], self.out_dim)
