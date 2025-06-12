"""CenterHead with optional Windowed Self‑Attention.

This file subclasses the original CenterHead and inserts a Swin‑style
window attention block right after the shared BEV convolution.  All
other logic (loss, bbox coder, post‑processing) is inherited unchanged.

Usage (in your model YAML):
  head:
    type: "CenterHeadAttn"
    use_window_attn: True
    window_size: 8
    shifted_window: True
    num_heads: 4
    ...   # rest of the CenterHead args

If ``use_window_attn`` is False, this behaves exactly like the original
CenterHead.
"""
from typing import List, Tuple

import torch
from torch import nn, Tensor

# Import everything the parent CenterHead expects.
from .centerpoint_head import CenterHead  # noqa: E402
from ...ops.attention_modules import WindowSelfAttention  # noqa: E402


class CenterHeadAttn(CenterHead):
    """CenterHead variant that supports an optional shared attention block."""

    def __init__(self, *args,
                 use_window_attn: bool = True,
                 window_size: int = 4,
                 num_heads: int = 2,
                 proj_ratio: float = 0.25,
                 out_drop: float = 0.2,
                 use_gate: bool = True,
                 shifted_window: bool = True,
                 **kwargs):
        kwargs.pop("type", None)
        """Extend ``CenterHead`` with attention‑specific hyper‑parameters.

        Extra Args:
            use_window_attn (bool): Enable / disable attention layer.
            window_size (int): Spatial window size (pixels) for W‑MHA.
            num_heads (int): Number of attention heads.
            shifted_window (bool): Whether to apply the Swin shift trick.
        """
        super().__init__(*args, **kwargs)

        self.use_window_attn = use_window_attn
        if self.use_window_attn:
            # First layer: plain windowed MHA
            layer1 = WindowSelfAttention(
                dim=self.shared_conv.out_channels,
                window_size=window_size,
                num_heads=num_heads,
                proj_ratio=proj_ratio,
                out_drop=out_drop,
                use_gate=use_gate,
                shifted=False,            # no shift in the first block
            )
            # Second layer: shifted windowed MHA
            layer2 = WindowSelfAttention(
                dim=self.shared_conv.out_channels,
                window_size=window_size,
                num_heads=num_heads,
                proj_ratio=proj_ratio,
                out_drop=out_drop,
                use_gate=use_gate,
                shifted=True,             # apply the cyclic shift here
            )
            self.attn = nn.Sequential(layer1, layer2)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward_single(self, x: Tensor) -> List[dict]:
        """Forward with optional attention."""
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        # Shared 3×3 conv from parent class
        x = self.shared_conv(x)

        if self.use_window_attn:
            x = self.attn(x)

        # Feed into task‑specific prediction heads
        ret_dicts = [task(x) for task in self.task_heads]
        return ret_dicts

    # NOTE: The loss and post‑processing methods are inherited unchanged.
