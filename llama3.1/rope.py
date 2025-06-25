import torch

def get_rope_params(config):
  rope_base = config['rope_base']
  emb_dim = config['emb_dim']
  context_length = config['context_length']
  
  inv_freqs = 1.0 / (rope_base ** (torch.arange(0, emb_dim, 2).float() / emb_dim))

  ######################################### LLAMA 3.2 Exclusive #########################################

  factor = config['rope_base']
  orginal_context_length = config['original_context_length']
  low_freq_factor = config['low_freq_factor']
  high_freq_factor = config['high_freq_factor']

  wavelengths = 2 * torch.pi / inv_freqs

  lower_threshold = orginal_context_length / low_freq_factor
  higher_threshold = orginal_context_length / high_freq_factor

  # If freq is too low, i.e. wavelength too high
  inv_freqs = torch.where(wavelengths > lower_threshold, inv_freqs / factor, inv_freqs)
  
  # For freq in between, i.e. for wavelength between lowfreq and highfreg, adding a smoothing factor
  smoothing_factor = (wavelengths - higher_threshold) / (lower_threshold - higher_threshold)
  smoothing_factor = smoothing_factor.clamp(0, 1)
  inv_freqs = torch.where( 
      (wavelengths <= lower_threshold) & (wavelengths >= higher_threshold),
      (1 - smoothing_factor) * (inv_freqs / factor) + smoothing_factor * inv_freqs,
      inv_freqs)
  
  #######################################################################################################

  positions = torch.arange(0, context_length)
  angles = positions[:, None] * inv_freqs[None, :]

  return torch.cos(angles), torch.sin(angles)

def apply_rope(X, cos, sin):
  batch_size, num_head, seq_len, head_dim = X.shape

  cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
  sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
  
  X_odd = X[..., 1::2]
  X_even = X[..., 0::2]

  X_odd_rotated = X_odd * cos + X_even * sin
  X_even_rotated = X_even * cos - X_odd * sin

  X_rotated = torch.empty_like(X)
  X_rotated[..., 0::2] = X_even_rotated
  X_rotated[..., 1::2] = X_odd_rotated
  
  return X_rotated