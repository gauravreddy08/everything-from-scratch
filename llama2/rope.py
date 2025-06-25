import torch
from config import llama2_7b

def get_rope_params(config=llama2_7b):
  head_dim = config['emb_dim'] // config['n_heads']
  assert head_dim % 2 == 0

  inv_freq = 1 / (10_000 ** (torch.arange(0, head_dim, 2, dtype=config['dtype'], device=config['device'])[:head_dim//2] / head_dim))
  positions = torch.arange(config['context_length'], dtype=config['dtype'], device=config['device'])
  out = positions[:, None] * inv_freq[None, :]

  return torch.cos(out).to(config['device']), torch.sin(out).to(config['device'])

def apply_rope(X, cos, sin):

  batch_size, num_heads, seq_len, head_dim = X.shape

  assert head_dim % 2 == 0

  cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
  sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

  x_even = X[..., 0::2]
  x_odd = X[..., 1::2]

  x_even.shape, x_odd.shape

  x_rotated_even = x_even * cos - x_odd * sin
  x_rotated_odd  = x_even * sin + x_odd * cos
  x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)

  return x_rotated.view(batch_size, num_heads, seq_len, head_dim)