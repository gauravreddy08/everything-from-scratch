import torch
from torch import nn

class BatchNorm(nn.Module):
  def __init__(self, emb_size):
    super().__init__()

    self.scale = nn.Parameter(torch.ones(emb_size))
    self.shift = nn.Parameter(torch.zeros(emb_size))

  def forward(self, x, epsilon = 1e-7):
    x_mean = torch.mean(x, dim=0, keepdim=True)
    x_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
    x_normalized = (x - x_mean) / torch.sqrt(x_var + epsilon)

    return (x_normalized + self.shift)*self.scale
  
if __name__ == "__main__":
    x = torch.rand(32, 128)

    print(f"Mean of Tensor: {torch.mean(x[0], dim=0, keepdim=True)}")
    print(f"Variance of Tensor: {torch.var(x[0], dim=0, keepdim=True)}")
    
    batchnorm  = BatchNorm(128)
    x_normalized = batchnorm(x)

    print(f"Mean of Normalized Tensor: {torch.mean(x_normalized, dim=0, keepdim=True)[0][0]}")
    print(f"Variance of Normalized Tensor: {torch.var(x_normalized, dim=0, keepdim=True)[0][0]}")