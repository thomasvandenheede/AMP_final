import torch
from torch import nn
from torchvision.ops import DeformConv2d

class DeformConvModule(nn.Module):
    """
    A deformable convolution block:
      1) offset_conv: predicts (2 * kernel_size * kernel_size) offsets
      2) deform_conv: DeformConv2d(using the same kernel_size, padding, stride)
      3) (optional) BatchNorm + ReLU if with_norm=True
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        with_norm: bool = True
    ):
        super().__init__()
        # 1) Offset conv: output 2 offsets per location × (kernel_size²)
        offset_channels = 2 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            in_channels,
            offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        # Zero-initialize so it starts as a regular conv
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias,   0.0)

        # 2) The actual deformable convolution
        #    Must match the same kernel_size, stride, padding
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        nn.init.kaiming_normal_(self.deform_conv.weight, mode='fan_out', nonlinearity='relu')

        # 3) BatchNorm + ReLU (if requested)
        if with_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Predict offsets
        offset = self.offset_conv(x)
        # 2) Apply deformable conv using those offsets
        x = self.deform_conv(x, offset)
        # 3) BN + ReLU (if applicable)
        if self.bn is not None:
            x = self.bn(x)
        return self.relu(x)
