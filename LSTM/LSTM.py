import torch
from torch import nn

class ForgetGate(nn.Module):
  def __init__(self, emb_dim = 256):
    super().__init__()
    self.Wf = nn.Parameter(torch.randn(emb_dim * 2, emb_dim))
    self.bf = nn.Parameter(torch.zeros(1, emb_dim))
  
  def forward(self, X, hidden_state):
    X = torch.cat((X, hidden_state), dim=-1)
    
    X = torch.sigmoid(X @ self.Wf + self.bf)
    return X

class InputGate(nn.Module):
  def __init__(self, emb_dim = 256):
    super().__init__()
    self.Wi = nn.Parameter(torch.randn(emb_dim * 2, emb_dim))
    self.Wc = nn.Parameter(torch.randn(emb_dim * 2, emb_dim))

    self.bi = nn.Parameter(torch.zeros(1, emb_dim))
    self.bc = nn.Parameter(torch.zeros(1, emb_dim))
  
  def forward(self, X, hidden_state):
    X = torch.cat((X, hidden_state), dim=-1)

    i_t = torch.sigmoid(X @ self.Wi + self.bi)
    C_t_ = torch.tanh(X @ self.Wc + self.bc)

    return i_t * C_t_

class OutputGate(nn.Module):
  def __init__(self, emb_dim = 256):
    super().__init__()
    self.Wo = nn.Parameter(torch.randn(emb_dim * 2, emb_dim))
    self.bo = nn.Parameter(torch.zeros(1, emb_dim))

  def forward(self, X, hidden_state):
    X = torch.cat((X, hidden_state), dim=-1)
    o_t = torch.sigmoid(X @ self.Wo + self.bo)

    return o_t

class LSTM(nn.Module):
  def __init__(self, emb_dim = 256):
    super().__init__()

    self.forget_gate = ForgetGate(emb_dim)
    self.input_gate = InputGate(emb_dim)
    self.output_gate = OutputGate(emb_dim)
  
  def forward(self, X):
    batch_size, _, emb_dim = X.shape
    self.cell_state = torch.zeros(batch_size, emb_dim, dtype=torch.float)
    self.hidden_state = torch.zeros(batch_size, emb_dim, dtype=torch.float)
    
    for i in range(X.shape[1]):
      out = self.step(X[:, i, :])
    
    return out

  def step(self, X):
    ft = self.forget_gate(X, self.hidden_state)
    self.cell_state *= ft

    self.cell_state += self.input_gate(X, self.hidden_state)

    o_t = self.output_gate(X, self.hidden_state)
    self.hidden_state = o_t * torch.tanh(self.cell_state)

    return self.hidden_state