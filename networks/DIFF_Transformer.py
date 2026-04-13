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
        self.lambda_init = lambda_init(layer_idx)  # 初始化λ值

        # 定义 Q1, Q2, K1, K2, V 的线性投影层
        self.q1_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q2_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k1_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)  # V 投影到 dim

        self.c_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(proj_drop)

        # 初始化 λ 参数
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.head_size) * 0.1)

    def forward(self, x, H, W):
        B, T, C = x.shape
        # print(f"Input shape to DIFFAttention: {x.shape}")  # 调试信息

        # 生成 Q1, Q2, K1, K2, V
        q1 = self.q1_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q2 = self.q2_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k1 = self.k1_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k2 = self.k2_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_size)

        # 计算两个注意力矩阵
        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        # 软化注意力权重
        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)

        # 计算 λ
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 应用差异化的注意力权重
        att = att1 - lambda_full * att2
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)  # [B, n_head, T, head_size]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 重新排列为 [B, T, C]
        y = self.resid_dropout(self.c_proj(y))

        # print(f"Output shape from DIFFAttention: {y.shape}")  # 调试信息
        return y


# class DIFFTransformer(nn.Module):
#     def __init__(self, dim, num_heads, layer_idx, mlp_ratio=4, qkv_bias=False, norm_layer=nn.LayerNorm,
#                  attention_class=DIFFAttention):
#         super().__init__()
#
#         # 层归一化
#         self.ln_1 = norm_layer(dim)
#         # 差分注意力机制
#         self.attn = attention_class(
#             dim=dim,
#             num_heads=num_heads,
#             layer_idx=layer_idx,
#             qkv_bias=qkv_bias
#         )
#
#         # 层归一化
#         self.ln_2 = norm_layer(dim)
#
#         # 前馈网络
#         hidden_dim = int(dim * mlp_ratio)
#         # self.mlp = MLP(dim, embed_dim=hidden_dim, output_dim=dim)  # 确保 MLP 的输出维度与输入维度一致
#         self.mlp = MLP(dim, embed_dim=hidden_dim)
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape  # [B, H*W, C]
#         # print(f"Input shape to DIFF_Transformer: {x.shape}")  # 调试信息
#
#         # 应用差分注意力机制
#         attn_output = self.attn(self.ln_1(x), H, W)
#         # print(f"Shape after attention: {attn_output.shape}")  # 调试信息
#
#         # 残差连接
#         x = x + attn_output  # [B, H*W, C]
#
#         # 应用前馈网络
#         mlp_output = self.mlp(self.ln_2(x))
#         # print(f"Shape after MLP: {mlp_output.shape}")  # 调试信息
#
#         # 残差连接
#         x = x + mlp_output  # [B, H*W, C]
#
#         # print(f"Output shape from DIFF_Transformer: {x.shape}")  # 调试信息
#         return x

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
            nn.Dropout(0.1),  # 根据需要调整dropout率
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1)  # 根据需要调整dropout率
        )

    def forward(self, x, H, W):
        B, N, C = x.shape  # [B, H*W, C]
        # print(f"Input shape to DIFF_Transformer: {x.shape}")  # 调试信息

        # 应用差分注意力机制
        # gai 1.7: 确保输入到 LayerNorm 的张量形状为 [B, H*W, C]
        if len(x.shape) == 4:  # 如果 x 的形状为 [B, C, H, W]
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # 转换为 [B, H*W, C]

        attn_output = self.attn(self.ln_1(x), H, W)
        # print(f"Shape after attention: {attn_output.shape}")  # 调试信息

        # 残差连接
        x = x + attn_output  # [B, H*W, C]

        # 应用前馈网络
        # gai 1.7: 确保输入到 LayerNorm 的张量形状为 [B, H*W, C]
        if len(x.shape) == 4:  # 如果 x 的形状为 [B, C, H, W]
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # 转换为 [B, H*W, C]

        mlp_output = self.mlp(self.ln_2(x))
        # print(f"Shape after MLP: {mlp_output.shape}")  # 调试信息

        # 残差连接
        x = x + mlp_output  # [B, H*W, C]

        # print(f"Output shape from DIFF_Transformer: {x.shape}")  # 调试信息
        return x