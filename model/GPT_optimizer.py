import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer

from dataclasses import dataclass
import math
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 384
    block_size = 128
    batch_size = 8
    max_iters = 10000
    eval_interval = 1000
    eval_iters = 50
    dropout = 0.2
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    split: Optional[str] = None
    learning_rate = 3e-4
    best_loss = float('inf')

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta=10000):
    assert head_dim % 2 == 0, "d_k 必须是偶数"
    # (head_dim / 2)
    # 计算指数部分
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # (head_dim / 2)
    # 计算 theta
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # (seq_len)
    # 计算 m
    m = torch.arange(seq_len, device=device)
    # m 外积 theta
    # (seq_len) 外积 (head_dim) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # 变为复数形式
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device):
    # 将连续两个 dim 的值作为一个复数
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[: -1], -1, 2))
    # 维度匹配
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotary = x_complex * freqs_complex
    # 变成二维张量
    x_out = torch.view_as_real(x_rotary)
    # 铺平 flatten
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs, split):
        super().__init__()
        # 有多少个 kv heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 有多少个 q heads
        self.n_q_heads = args.n_heads
        # kv 需要复制几次，也就是几个 q heads 为一组
        self.n_rep = self.n_q_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.split = split
        self.device = args.device

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        if self.split == 'inference':
            self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
            self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        else:
            self.register_buffer('tril', torch.tril(torch.ones(args.block_size, args.block_size)))

    def forward(self, x: torch.Tensor, start_pos, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # (batch, seq_len, dim) -> (batch, seq_len, n_q_heads * head_dim)
        xq = self.wq(x)
        # (batch, seq_len, dim) -> (batch, seq_len, n_kv_heads * head_dim)
        xk = self.wk(x)
        # (batch, seq_len, dim) -> (batch, seq_len, n_kv_heads * head_dim)
        xv = self.wv(x)

        # (batch, seq_len, n_q_heads * head_dim) -> (batch, seq_len, n_q_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        # (batch, seq_len, n_kv_heads * head_dim) -> (batch, seq_len, n_kv_heads, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (batch, seq_len, n_kv_heads * head_dim) -> (batch, seq_len, n_kv_heads, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # RoPE 应用于 Q 和 K
        xq = apply_rotary_embedding(xq, freqs_complex, self.device)
        xk = apply_rotary_embedding(xk, freqs_complex, self.device)

        if self.split == 'inference':
            self.cache_k[: batch_size, start_pos : start_pos + seq_len] = xk
            self.cache_v[: batch_size, start_pos : start_pos + seq_len] = xv

            keys = self.cache_k[: batch_size, : start_pos + seq_len]
            values = self.cache_v[: batch_size, : start_pos + seq_len]
        else:
            keys = xk
            values = xv

        # 将 K 和 V 复制 n_rep 次
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (batch, seq_len, n_q_heads, head_dim) -> (batch_size, n_q_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        # (batch, seq_len, n_kv_heads, head_dim) -> (batch, n_kv_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        # (batch, seq_len, n_kv_heads, head_dim) -> (batch, n_kv_heads, seq_len, head_dim)
        values = values.transpose(1, 2)

        # (batch_size, n_q_heads, seq_len, head_dim) @ (batch, n_kv_heads, head_dim, seq_len) -> (batch_size, n_q_heads, seq_len, seq_len)
        attention = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if self.split != 'inference':
            attention = attention.masked_fill(self.tril == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1).type_as(xq)

        # (batch_size, n_q_heads, seq_len, seq_len) @ (batch, n_kv_heads, seq_len, head_dim) -> (batch_size, n_q_heads, seq_len, head_dim)
        output = torch.matmul(attention, values)
        # (batch_size, n_q_heads, seq_len, head_dim) -> (batch_size, seq_len, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        output = self.dropout(self.wo(output))
        return output

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)

class Feedforward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        x = self.dropout(x)
        return x

class Dencoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.MHA = SelfAttention(args, args.split)
        self.FFN = Feedforward(args)
        self.RMSNorm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos, freqs_complex: torch.Tensor):
        h = x + self.MHA(self.RMSNorm(x), start_pos, freqs_complex)
        output = h + self.FFN(self.RMSNorm(h))
        return output

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        assert args.vocab_size > 0, "词汇表大小必须大于 0"

        self.token_embed = nn.Embedding(args.vocab_size, args.dim)

        self.freqs_complex = precompute_theta_pos_frequencies(args.dim // args.n_heads, args.max_seq_len * 2, device=args.device)
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Dencoder(args))

        self.projection = nn.Linear(args.dim, args.vocab_size)

    def forward(self, tokens: torch.Tensor, start_pos):
        batch_size, seq_len = tokens.shape

        output = self.token_embed(tokens)

        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        for layer in self.layers:
            output = layer(output, start_pos, freqs_complex)
        output = self.projection(output)
        return output

