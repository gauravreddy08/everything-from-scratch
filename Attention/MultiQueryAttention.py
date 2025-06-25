from torch import nn
import torch

class MultiQueryAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    num_heads = config['num_heads']
    emb_dim = config['emb_dim']

    self.num_heads = num_heads
    self.head_dim = emb_dim // num_heads

    self.K = nn.Linear(emb_dim, self.head_dim, bias=False)
    self.Q = nn.Linear(emb_dim, emb_dim, bias=False)
    self.V = nn.Linear(emb_dim, self.head_dim, bias=False)

  def forward(self, X):
    batch_size, seq_len, emb_dim = X.shape

    key = self.K(X)
    query = self.Q(X)
    value = self.V(X)

    query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

    key = key.transpose(1, 2)
    KQ = query @ key.unsqueeze(1) / (self.head_dim ** 0.5)

    mask = torch.tril(torch.ones((seq_len, seq_len), device = X.device)).bool()
    maskedKQ = KQ.masked_fill(~mask, -torch.inf)
    softmaxKQ = torch.softmax(maskedKQ, dim=-1)

    out = softmaxKQ @ value.unsqueeze(1)
    out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    return out