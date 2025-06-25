import torch
from torch import nn
from config import llama2_7b, llama2_13b, llama2_70b
from rope import get_rope_params, apply_rope

class RMSNorm(nn.Module):
  def __init__(self, config = llama2_7b, eps = 1e-7):
    super().__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(config['emb_dim'], dtype =config['dtype'],
                              device=config['device']))

  def forward(self, X):
    print(X.shape)
    assert X.shape[-1] == self.gamma.shape[0]

    RMS = X / torch.sqrt(self.eps + torch.mean(X ** 2, dim=-1, keepdim=True))
    out = RMS * self.gamma
    return out.to(dtype = X.dtype)

from torch import sigmoid

class SiLU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, X):
    return X * sigmoid(X)

class FeedForward(nn.Module):
  def __init__(self, config = llama2_7b):
    super().__init__()

    self.fc1 = nn.Linear(config['emb_dim'], config['hidden_dim'], dtype=config["dtype"], bias = False, device=config['device'])
    self.fc2 = nn.Linear(config['emb_dim'], config['hidden_dim'], dtype=config["dtype"], bias = False, device=config['device'])
    self.fc3 = nn.Linear(config['hidden_dim'], config['emb_dim'], dtype=config["dtype"], bias = False, device=config['device'])

    self.silu = SiLU()

  def forward(self, X):
    x1 = self.fc1(X)
    x2 = self.fc2(X)
    x1_2 = self.silu(x1) * x2

    return self.fc3(x1_2)
  
from torch.nn.functional import softmax

class MultiHeadAttention(nn.Module):
  def __init__(self, config = llama2_7b):
    super().__init__()
    self.config = config
    self.K = nn.Linear(config['emb_dim'], config['emb_dim'], dtype=config['dtype'], bias = False)
    self.Q = nn.Linear(config['emb_dim'], config['emb_dim'], dtype=config['dtype'], bias = False)
    self.V = nn.Linear(config['emb_dim'], config['emb_dim'], dtype=config['dtype'], bias = False)

    cos, sin = get_rope_params(config)

    self.register_buffer('cos', cos)
    self.register_buffer('sin', sin)

    self.out_proj = nn.Linear(config['emb_dim'], config['emb_dim'], dtype=config['dtype'])

  def forward(self, X):
    batch_size, seq_len, d = X.shape

    key = self.K(X)
    query = self.Q(X)
    value = self.V(X)

    key = key.view(batch_size, seq_len, self.config['n_heads'], -1).transpose(1, 2)
    query = query.view(batch_size, seq_len, self.config['n_heads'], -1).transpose(1, 2)
    value = value.view(batch_size, seq_len, self.config['n_heads'], -1).transpose(1, 2)

    key = apply_rope(key, self.cos, self.sin)
    query = apply_rope(query, self.cos, self.sin)

    head_dim = X.shape[-1] // self.config['n_heads']
    QK = (query @ key.transpose(2, 3)) / (head_dim ** 0.5)

    mask = torch.tril(torch.ones((seq_len, seq_len), device=X.device)).bool()
    QK_masked = QK.masked_fill(~mask, -torch.inf)
    QK_masked = softmax(QK_masked, dim=-1)

    out = QK_masked @ value
    out = out.transpose(1, 2).contiguous()
    out = out.view(batch_size, seq_len, -1)
    out = self.out_proj(out)

    return out

class TransformerBlock(nn.Module):
  def __init__(self, config=llama2_7b):
    super().__init__()

    self.norm1 = RMSNorm(config)
    self.norm2 = RMSNorm(config)

    self.mha = MultiHeadAttention(config)

    self.ff = FeedForward(config)

  def forward(self, X):

    shortcut = X
    X = self.norm1(X)
    X = self.mha(X)
    X = X + shortcut

    shortcut = X
    X = self.norm2(X)

class Llama2(nn.Module):
  def __init__(self, config = llama2_7b):
    super().__init__()

    self.config = config

    self.embedding = nn.Embedding(config['vocab_size'], config['emb_dim'])

    self.tf_blocks = nn.Sequential(
        *[TransformerBlock(config) for _ in range(config['n_layers'])]
    )

    self.fnorm = RMSNorm(config)
    self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)

  def forward(self, X):
    X = self.embedding(X)

    X = self.tf_blocks(X)

    X = self.fnorm(X)
    X = self.out_head(X)
    return X
  
if __name__ == "__main__":
  model = Llama2(llama2_7b)
  
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total number of parameters: {total_params:,}") # 6,738,546,688 --> 7B parameters