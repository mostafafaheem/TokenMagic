# import torch
# from dataclasses import dataclass
# from tiktoken import get_encoding

# @dataclass
# class ModelConfig:
#     num_hidden_layers: int = 36
#     vocab_size: int = 201088
#     hidden_size: int = 2880
#     intermediate_size: int = 2880
#     swiglu_limit: float = 7.0
#     head_dim: int = 64
#     num_attention_heads: int = 64
#     num_key_value_heads: int = 8

# class RMSNorm(torch.nn.Module):
#     def __init__(self, hidden_size: int, eps: float = 1e-5, device: torch.device | None = None):
#         super(RMSNorm, self).__init__()
#         self.hidden_size = hidden_size
#         self.eps = eps
#         self.scale = torch.nn.Parameter(torch.ones(hidden_size))

#     def forward(self, x):
#         norm_x = x.norm(2, dim=-1, keepdim=True)
#         rms = norm_x * (self.hidden_size ** -0.5)
#         return x / (rms + self.eps) * self.scale

# class Model(torch.nn.Module):
#     def __init__(self, config: ModelConfig):
#         super(Model, self).__init__()
#         self.config = config
#         # Initialize model layers here based on config

#     def forward(self, x):
#         # Define forward pass here
