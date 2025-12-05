import math

import torch
import torch.nn as nn
from einops import rearrange

class SpatialGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N)) 
        x1, x2 = x.chunk(2, dim=-1)
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()
        return x1 * x2

class SGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        dim (int): Number of input channels.
        hidden_dim (int | None): Number of hidden channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x)  
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x
