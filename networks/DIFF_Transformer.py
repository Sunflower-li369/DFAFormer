import torch
import torch.nn as nn
import math

from networks.segformer import *

def lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))

class DIFFAttention(nn.Module):
    def __init__(self, dim, num_heads, layer_idx, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_size = dim // num_heads
        self.lambda_init = lambda_init(layer_idx)  

        self.q1_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q2_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k1_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)  

        self.c_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(proj_drop)

        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)

    def forward(self, x, H, W):
        B, T, C = x.shape

        q1 = self.q1_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q2 = self.q2_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k1 = self.k1_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k2 = self.k2_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_size)

        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)

        # 计算 λ
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init


        att = att1 - lambda_full * att2
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)  # [B, n_head, T, head_size]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  
        y = self.resid_dropout(self.c_proj(y))

        
        return y

class DIFFTransformer(nn.Module):
    def __init__(self, dim, num_heads, layer_idx, mlp_ratio=4, qkv_bias=False, norm_layer=nn.LayerNorm,
                     attention_class=DIFFAttention):
        super().__init__()
        # 层归一化
        self.ln_1 = norm_layer(dim)
        # 差分注意力机制
        self.attn = attention_class(
            dim=dim,
            num_heads=num_heads,
            layer_idx=layer_idx,
            qkv_bias=qkv_bias
        )

        # 层归一化
        self.ln_2 = norm_layer(dim)

        # 前馈网络
        hidden_dim = int(dim * mlp_ratio)
        # 确保MLP的输入和输出维度与transformer块的维度一致
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),  
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1)  
        )

    def forward(self, x, H, W):
        B, N, C = x.shape 
    
        if len(x.shape) == 4:  
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C) 

        attn_output = self.attn(self.ln_1(x), H, W)

        # 残差连接
        x = x + attn_output  # [B, H*W, C]

        if len(x.shape) == 4: 
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C) 
        mlp_output = self.mlp(self.ln_2(x))

        # 残差连接
        x = x + mlp_output  # [B, H*W, C]

        return x
