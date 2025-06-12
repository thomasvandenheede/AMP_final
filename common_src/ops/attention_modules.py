"""
Attention modules for CenterPoint improvements.
Provides `WindowSelfAttention` – a lightweight Swin‑style windowed
multi‑head self‑attention block for (B, C, H, W) feature maps.

Key design points for stability:
* BatchNorm2d + residual preserves low‑level cues.
* Optional cyclic shift implements Swin W‑MHA/SW‑MHA.
* Relative‑position bias supports small objects.
* proj_ratio controls Q/K/V channel reduction.

Input : (B, C, H, W)
Output: (B, C, H, W)

Example:
```python
from common_src.ops.attention_modules import WindowSelfAttention
attn = WindowSelfAttention(dim=128, window_size=8, num_heads=4, proj_ratio=0.5, shifted=True)
y = attn(x)
```
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["window_partition", "window_reverse", "WindowSelfAttention"]


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, C, H, W = x.shape
    if H % window_size != 0 or W % window_size != 0:
        raise ValueError(f"H,W must be divisible by window_size={window_size}")
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return windows.view(-1, C, window_size, window_size)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, -1, H, W)


class WindowSelfAttention(nn.Module):
    """Window‑based multi‑head self‑attention with optional cyclic shift."""

    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        num_heads: int = 4,
        shifted: bool = False,
        proj_ratio: float = 1.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        out_drop: float = 0.1,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        assert 0 < proj_ratio <= 1.0
        self.dim = dim
        self.window_size = window_size
        self.shift_size = window_size // 2 if shifted else 0
        self.num_heads = num_heads

        qk_dim = int(dim * proj_ratio)
        assert qk_dim % num_heads == 0
        self.head_dim = qk_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Norm + projections
        self.norm1 = nn.BatchNorm2d(dim)
        self.qk = nn.Conv2d(dim, qk_dim * 2, kernel_size=1, bias=qkv_bias)
        self.v  = nn.Conv2d(dim, qk_dim,     kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(qk_dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(out_drop)
        self.norm2 = nn.BatchNorm2d(dim)

        # Relative position bias
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))  # (2, w, w)
        coords_flat = coords.flatten(1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        index = rel.sum(-1)
        self.register_buffer("relative_position_index", index)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        nn.init.trunc_normal_(self.qk.weight, std=0.02)
        nn.init.trunc_normal_(self.v.weight,  std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H0, W0 = x.shape
        shortcut = x
        x = self.norm1(x)

        # pad
        pad_b = (self.window_size - H0 % self.window_size) % self.window_size
        pad_r = (self.window_size - W0 % self.window_size) % self.window_size
        if pad_b or pad_r:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        H, W = x.shape[2:]

        # shift
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        # windows
        nw = window_partition(x, self.window_size)
        nW = nw.shape[0]
        qk = self.qk(nw).flatten(2).transpose(1, 2)
        v  = self.v(nw).flatten(2).transpose(1, 2)
        q, k = qk.chunk(2, dim=-1)
                # reshape to (nW, tokens, heads, head_dim) then permute to (nW, heads, tokens, head_dim)
        q = q.reshape(nW, self.window_size*self.window_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(nW, self.window_size*self.window_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(nW, self.window_size*self.window_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size**2, self.window_size**2, -1).permute(2, 0, 1)
        attn = attn + bias.to(attn.device)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        xa = (attn @ v).permute(0, 1, 3, 2).reshape(
            nW, self.num_heads * self.head_dim, self.window_size, self.window_size
        )
        x = window_reverse(xa, self.window_size, H, W)
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        if pad_b or pad_r:
            x = x[:, :, :H0, :W0]

        x = self.proj_drop(self.proj(x))
        x = self.norm2(shortcut + x)
        return x
