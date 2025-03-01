from torch import nn
import torch
import tiktoken
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import wandb

wandb.login()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.25,
    "qkv_bias": False,
    "batch_size": 128,
    "device": device,
    'lr': 1e-5,
    'epochs': 10
  }

run = wandb.init(
    project="gpt2-chat",
    config = config
)

tokenizer = tiktoken.get_encoding('gpt2')

def text_to_tokens(text, tokenizer):
  encoded = tokenizer.encode(text)
  encoded_tensor = torch.tensor(encoded)
  return encoded_tensor

def tokens_to_text(tokens, tokenizer):
  flat = tokens.squeeze(0)
  return tokenizer.decode(flat.tolist())

with open('chat.txt', 'r') as file:
  raw_data = file.read()

all_tokens = text_to_tokens(raw_data, tokenizer)

train_tokens = all_tokens[:int(0.9*len(all_tokens))]
test_tokens = all_tokens[int(0.9*len(all_tokens)):]

class CustomDataset(Dataset):
  def __init__(self, data, config):
    self.data = data
    self.context_length = config['context_length']
  def __getitem__(self, index):
    return self.data[index:index+self.context_length], self.data[index+1:index+self.context_length+1]
  def __len__(self):
    return len(self.data)-self.context_length

train_data = CustomDataset(train_tokens, config)
test_data = CustomDataset(test_tokens, config)

train_dataloader = DataLoader(train_data,
                              batch_size=config['batch_size'],
                              shuffle=True)

test_dataloader = DataLoader(test_data,
                             batch_size=config['batch_size'],
                             shuffle=False)

class LayerNorm(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.scale = nn.Parameter(torch.ones(config['emb_dim']))
    self.shift = nn.Parameter(torch.zeros(config['emb_dim']))

  def forward(self, x, epsilon = 1e-7):
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    x_var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    x_normalized = (x - x_mean) / torch.sqrt(x_var + epsilon)

    return (x_normalized + self.shift)*self.scale

class MultiHeadAttention(nn.Module):
  def __init__(self, config):
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
  def __init__(self, config):
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
  def __init__(self, config):
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

class GPTModel(nn.Module):
  def __init__(self, config):
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

# model = GPTModel(config)

# def generate(model, idx, max_new_tokens, context_size):
#   for _ in range(max_new_tokens):
#     idx = idx[-context_size:]
#     with torch.no_grad():
#         logits = model(idx.to(device))
#     probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
#     idx_next = torch.argmax(probs, dim=-1, keepdim=True)
#     idx = torch.cat([idx, idx_next], dim=1)
#   return idx

# out = generate(model=model,
#                idx=text_to_tokens('Hello world!', tokenizer).unsqueeze(0),
#                max_new_tokens=10,
#                context_size=config['context_length'])

# tokens_to_text(out, tokenizer)

from torch.nn.functional import cross_entropy

def get_loss(logits, targets):
  logits_flat = logits.flatten(0, 1)
  targets_flat = targets.flatten()

  loss = cross_entropy(logits_flat, targets_flat)
  return loss

EPOCHS = config['epochs']
model = GPTModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

for epoch in range(EPOCHS):
  model.train()
  for X, y in train_dataloader:
    X, y = X.to(config['device']), y.to(config['device'])
    logits = model(X)
    loss = get_loss(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  for X, y in test_dataloader:
    loss = 0
    model.eval()
    with torch.no_grad():
      X, y = X.to(config['device']), y.to(config['device'])
      logits = model(X)
      loss += get_loss(logits, y)
  print(f'EPOCH {epoch}/{EPOCHS} | test_val: {loss/len(test_dataloader)}')
  wandb.log({"epoch": epoch+1, "loss": loss/len(test_dataloader)})
  torch.save(model.state_dict(), f'chat-model{epoch}.pt')
