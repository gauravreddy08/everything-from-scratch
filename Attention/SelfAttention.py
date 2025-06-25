import torch
from torch import nn

class SelfAttention(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()

    self.K = nn.Linear(in_dim, out_dim, bias=False)
    self.Q = nn.Linear(in_dim, out_dim, bias=False)
    self.V = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    key = self.K(x)
    query = self.Q(x)
    value = self.V(x)

    d_k = key.shape[1]
    kq = (key @ query.transpose(1, 2)) / d_k**0.5

    context_length = x.shape[1]
    mask = torch.tril(torch.ones(context_length, context_length))
    kq_masked = kq.masked_fill(mask == 0, -torch.inf)
    softmax_kq = torch.nn.functional.softmax(kq_masked, dim=-1)
    out = softmax_kq @ value

    return out

if __name__ == "__main__":
    b = 32
    input_length = 64
    embed_dim = 128

    x = torch.randn(b, input_length, embed_dim)

    self_attention = SelfAttention(embed_dim, embed_dim)
    out = self_attention(x)

    print(out.shape)