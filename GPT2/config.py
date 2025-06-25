import torch

gpt2_small = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

gpt2_medium = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "batch_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

gpt2_large = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1280,
    "n_heads": 20,
    "n_layers": 36,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "batch_size": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

gpt2_xl = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1600,
    "n_heads": 25,
    "n_layers": 48,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "batch_size": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

gpt2_base = gpt2_small.copy()