from torch import nn
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import wandb

from main import GPT2
from config import gpt2_base, gpt2_small

model = GPT2(gpt2_base)

wandb.login()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

run = wandb.init(
    project="gpt2-chat",
    config = gpt2_base
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

train_data = CustomDataset(train_tokens, gpt2_base)
test_data = CustomDataset(test_tokens, gpt2_base)

train_dataloader = DataLoader(train_data,
                              batch_size=gpt2_base['batch_size'],
                              shuffle=True)

test_dataloader = DataLoader(test_data,
                             batch_size=gpt2_base['batch_size'],
                             shuffle=False)

from torch.nn.functional import cross_entropy

def get_loss(logits, targets):
  logits_flat = logits.flatten(0, 1)
  targets_flat = targets.flatten()

  loss = cross_entropy(logits_flat, targets_flat)
  return loss

EPOCHS = gpt2_base['epochs']
model = GPT2(gpt2_base).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=gpt2_base['lr'])

for epoch in range(EPOCHS):
  model.train()
  for X, y in train_dataloader:
    X, y = X.to(gpt2_base['device']), y.to(gpt2_base['device'])
    logits = model(X)
    loss = get_loss(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  for X, y in test_dataloader:
    loss = 0
    model.eval()
    with torch.no_grad():
      X, y = X.to(gpt2_base['device']), y.to(gpt2_base['device'])
      logits = model(X)
      loss += get_loss(logits, y)
  print(f'EPOCH {epoch}/{EPOCHS} | test_val: {loss/len(test_dataloader)}')
  wandb.log({"epoch": epoch+1, "loss": loss/len(test_dataloader)})
  torch.save(model.state_dict(), f'chat-model{epoch}.pt')