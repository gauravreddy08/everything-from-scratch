from torch import nn
import torch

class GroupedQueryAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config['emb_dim'] % config['n_heads'] == 0
    assert config['emb_dim'] % config['n_groups'] == 0

    self.head_dim = config['emb_dim'] // config['n_heads']
    self.num_heads = config['n_heads']
    self.num_groups = config['n_groups']

    assert self.num_heads % self.num_groups == 0

    self.K = nn.Linear(config['emb_dim'], self.head_dim * self.num_groups, bias=False, dtype=config['dtype'])
    self.Q = nn.Linear(config['emb_dim'], config['emb_dim'], bias=False, dtype=config['dtype'])
    self.V = nn.Linear(config['emb_dim'], self.head_dim * self.num_groups, bias=False, dtype=config['dtype'])

    self.out_proj = nn.Linear(config['emb_dim'], config['emb_dim'], bias=False, dtype=config['dtype'])
    
    #### RoPE is not used in this implementation ####
    # cos, sin = get_rope_params(config)
    # self.register_buffer('cos', cos)
    # self.register_buffer('sin', sin)
  
  def forward(self, X):
    batch_size, seq_length, emb_dim = X.shape

    X = X.to(dtype = self.config['dtype'])

    key = self.K(X).view(batch_size, seq_length, self.num_groups, -1).transpose(1, 2)
    query = self.Q(X).view(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
    value = self.V(X).view(batch_size, seq_length, self.num_groups, -1).transpose(1, 2)

    #### RoPE is not used in this implementation ####
    # key = apply_rope(key, self.cos, self.sin)
    # query = apply_rope(query, self.cos, self.sin)

    query = query.view(batch_size, self.num_heads // self.num_groups, self.num_groups, seq_length, -1)
    
    KQ = query @ key.transpose(-1, -2).unsqueeze(1)
    KQ = KQ / (self.head_dim ** 0.5)
    mask = torch.tril(torch.ones(seq_length, seq_length, device=X.device, dtype=torch.bool))
    KQ_masked = KQ.masked_fill(~mask, -torch.inf)
    KQ_softmax = torch.softmax(KQ_masked, dim=-1)

    out = KQ_softmax @ value.unsqueeze(1)
    out = out.view(batch_size, self.num_heads, seq_length, -1).transpose(1, 2).contiguous()
    return self.out_proj(out.view(batch_size, seq_length, emb_dim))