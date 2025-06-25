import torch
from torch import nn
from config import llama3_1_8b, llama3_1_13b
from rope import get_rope_params, apply_rope

class RMSNorm(nn.Module):
  def __init__(self, config = llama3_1_8b, eps = 1e-7):
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
  def __init__(self, config = llama3_1_8b):
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

class GroupedQueryAttention(nn.Module):
  def __init__(self, config = llama3_1_8b):
    super().__init__()
    assert config['emb_dim'] % config['n_heads'] == 0
    assert config['emb_dim'] % config['n_groups'] == 0

    self.head_dim = config['emb_dim'] // config['n_heads']
    self.num_heads = config['n_heads']
    self.num_groups = config['n_groups']
    self.config = config

    assert self.num_heads % self.num_groups == 0

    self.K = nn.Linear(config['emb_dim'], self.head_dim * self.num_groups, bias=False, dtype=config['dtype'])
    self.Q = nn.Linear(config['emb_dim'], config['emb_dim'], bias=False, dtype=config['dtype'])
    self.V = nn.Linear(config['emb_dim'], self.head_dim * self.num_groups, bias=False, dtype=config['dtype'])

    self.out_proj = nn.Linear(config['emb_dim'], config['emb_dim'], bias=False, dtype=config['dtype'])

    cos, sin = get_rope_params(config)
    self.register_buffer('cos', cos)
    self.register_buffer('sin', sin)
  
  def forward(self, X):
    batch_size, seq_length, emb_dim = X.shape

    X = X.to(dtype = self.config['dtype'])

    key = self.K(X).view(batch_size, seq_length, self.num_groups, -1).transpose(1, 2)
    query = self.Q(X).view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
    value = self.V(X).view(batch_size, seq_length, self.num_groups, -1).transpose(1, 2)

    key = apply_rope(key, self.cos, self.sin)
    query = apply_rope(query, self.cos, self.sin)

    query = query.view(batch_size, self.num_heads // self.num_groups, self.num_groups, seq_length, -1)
    
    KQ = query @ key.transpose(-1, -2).unsqueeze(1)
    KQ = KQ / (self.head_dim ** 0.5)
    mask = torch.tril(torch.ones(seq_length, seq_length, device=X.device, dtype=torch.bool))
    KQ_masked = KQ.masked_fill(~mask, -torch.inf)
    KQ_softmax = torch.softmax(KQ_masked, dim=-1)

    out = KQ_softmax @ value.unsqueeze(1)
    out = out.view(batch_size, self.num_heads, seq_length, -1).transpose(1, 2).contiguous()
    return self.out_proj(out.view(batch_size, seq_length, emb_dim))

class TransformerBlock(nn.Module):
  def __init__(self, config=llama3_1_8b):
    super().__init__()

    self.norm1 = RMSNorm(config)
    self.norm2 = RMSNorm(config)

    self.mha = GroupedQueryAttention(config)

    self.ff = FeedForward(config)

  def forward(self, X):

    shortcut = X
    X = self.norm1(X)
    X = self.mha(X)
    X = X + shortcut

    shortcut = X
    X = self.norm2(X)

class Llama3(nn.Module):
  def __init__(self, config = llama3_1_8b):
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
  model = Llama3(llama3_1_8b)
  
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total number of parameters: {total_params:,}") 