from torch import nn
import torch
from config import gpt2_base, gpt2_small

class LayerNorm(nn.Module):
  def __init__(self, config = gpt2_base):
    super().__init__()

    self.scale = nn.Parameter(torch.ones(config['emb_dim']))
    self.shift = nn.Parameter(torch.zeros(config['emb_dim']))

  def forward(self, x, epsilon = 1e-7):
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    x_var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    x_normalized = (x - x_mean) / torch.sqrt(x_var + epsilon)

    return (x_normalized + self.shift)*self.scale

class MultiHeadAttention(nn.Module):
  def __init__(self, config = gpt2_base):
    super().__init__()
    self.K = nn.Linear(config['emb_dim'], config['emb_dim'], bias=config['qkv_bias'])
    self.Q = nn.Linear(config['emb_dim'], config['emb_dim'], bias=config['qkv_bias'])
    self.V = nn.Linear(config['emb_dim'], config['emb_dim'], bias=config['qkv_bias'])

    self.num_heads = config['n_heads']
    self.out_dim = config['emb_dim']

    self.linear_layer = nn.Linear(config['n_heads'] * (config['emb_dim']//config['n_heads']), config['emb_dim'])
    self.dropout = nn.Dropout(p=config['drop_rate'])

    self.config = config

  def forward(self, x):
    key = self.K(x)
    query = self.Q(x)
    value = self.V(x)

    b, in_tokens, emb_size = x.shape

    key = key.view(b, in_tokens, self.num_heads, self.out_dim//self.num_heads).transpose(1, 2)
    query = query.view(b, in_tokens, self.num_heads, self.out_dim//self.num_heads).transpose(1, 2)
    value = value.view(b, in_tokens, self.num_heads, self.out_dim//self.num_heads).transpose(1, 2)

    out = (query @ key.transpose(2, 3))/ key.shape[-1]**0.5
    mask = torch.tril(torch.ones(in_tokens, in_tokens)).to(self.config['device'])
    out_masked = out.masked_fill(mask == 0, -torch.inf)
    softmax_kq = torch.nn.functional.softmax(out_masked, dim=-1)

    out = self.dropout(softmax_kq)

    out = softmax_kq @ value
    out = out.transpose(1, 2).contiguous()

    return self.linear_layer(out.view(x.shape[0], x.shape[1], self.out_dim))

class GeLU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
              torch.sqrt(torch.tensor(2.0 / torch.pi)) *
              (x + 0.044715 * torch.pow(x, 3))
          ))

class MLP(nn.Module):
  def __init__(self, config = gpt2_base):
    super().__init__()

    self.emb_dim = config['emb_dim']

    self.layers = nn.Sequential(
        nn.Linear(self.emb_dim, self.emb_dim*4),
        GeLU(),
        nn.Linear(self.emb_dim*4, self.emb_dim),
    )

  def forward(self, x):
    return self.layers(x)

class TransformerBlock(nn.Module):
  def __init__(self, config = gpt2_base):
    super().__init__()

    self.attention = MultiHeadAttention(config)
    self.mlp = MLP(config)

    self.norm1 = LayerNorm(config)
    self.norm2 = LayerNorm(config)

    self.dropout = nn.Dropout(config['drop_rate'])

  def forward(self, x):

    attention_out = self.norm1(x)
    attention_out = self.attention(attention_out)
    attention_out = self.dropout(attention_out)

    attention_out += x

    ff_out = self.norm2(attention_out)
    ff_out = self.mlp(ff_out)
    ff_out = self.dropout(ff_out)

    ff_out += attention_out

    return ff_out

class GPT2(nn.Module):
  def __init__(self, config = gpt2_base):
    super().__init__()

    self.token_emb = nn.Embedding(config['vocab_size'], config['emb_dim'])
    self.pos_emb = nn.Embedding(config['context_length'], config['emb_dim'])

    self.dropout = nn.Dropout(config['drop_rate'])
    self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config['n_layers'])])

    self.final_norm = LayerNorm(config)
    self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'])

    self.config = config

  def forward(self, x):
    b, l = x.shape
    tokens = self.token_emb(x)
    positions = self.pos_emb(torch.arange(l, device=self.config['device']))

    x = tokens + positions

    x = self.dropout(x)
    x = self.transformer_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits
  
model = GPT2(gpt2_base)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")