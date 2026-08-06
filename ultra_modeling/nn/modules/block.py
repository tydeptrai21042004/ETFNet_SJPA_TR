# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock
# Removed unused CEM1 import to avoid undeclared timm dependency.
from ultralytics.utils.torch_utils import fuse_conv_and_bn

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost', 'C2fCIB', 'SCDown', 'PSA', 'C3k2', 'CAFEM', 'C2PSA',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'ResNetLayer', 'IN', 'Multiin', 'MF', 'Add', 'Add2', 'A2C2f', 'TGF', 'GOCI', 'SJPA', 'RTPF', 'DCSPF', 'MPCRF')


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MF(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super(MF, self).__init__()
        self.mask_map_r = nn.Conv2d(c1//2, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(c1//2, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.bottleneck1 = nn.Conv2d(c1//2, c2//2, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(c1//2, c2//2, 3, 1, 1, bias=False)
        self.se = SE_Block(c2, reduction)

    def forward(self, x):
        x_left_ori,x_right_ori = x[:, :3, :, :], x[:, 3:, :, :]
        x_left = x_left_ori * 0.5
        x_right = x_right_ori * 0.5

        x_mask_left = torch.mul(self.mask_map_r(x_left), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # RGB
        out = self.se(torch.cat([out_RGB, out_IR], 1))

        return out

class IN(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class Multiin(nn.Module):  # stereo attention block
    def __init__(self, out=1):
        super().__init__()
        self.out = out

    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        if self.out == 1:
            x = x1
        else:
            x = x2
        return x
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))

class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """
        Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1

class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """
        Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """
        Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """
        Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
    
# Lan- cai tien C2PSA

# ---------------------------------------------------------------------
# DropPath (stochastic depth)
# ---------------------------------------------------------------------
class DropPath1(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep)
        return x * rand / keep

class ConvFFN1(nn.Module):
    def __init__(self, c, expand=2.0, drop_path=0.0, act=nn.SiLU):
        super().__init__()
        hidden = int(c * expand)
        self.pw1 = Conv(c, hidden, 1, 1)
        self.dw  = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False)
        self.bn  = nn.BatchNorm2d(hidden)
        self.act = act(inplace=True)
        self.pw2 = Conv(hidden, c, 1, 1, act=False)
        self.drop_path = DropPath(drop_path)
    def forward(self, x):
        idn = x
        x = self.pw1(x)
        x = self.act(self.bn(self.dw(x)))
        x = self.pw2(x)
        return idn + self.drop_path(x)

# ==========================
# AttentionMS+ : cross-scale (×2, ×4) + local, gating động cực nhẹ
# ==========================
class AttentionMS1(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5, use_local_dilated=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.key_dim   = int(self.head_dim * attn_ratio)
        self.scale     = self.key_dim ** -0.5

        # Q,K,V
        self.q_proj = Conv(dim, num_heads * self.key_dim, 1, act=False)
        self.k_proj = Conv(dim, num_heads * self.key_dim, 1, act=False)
        self.v_proj = Conv(dim, num_heads * self.head_dim, 1, act=False)

        # Local detail: DWConv3x3 + (tuỳ chọn) DWConv3x3 dilated d=2, p=2
        self.local_dw3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.local_bn3 = nn.BatchNorm2d(dim)
        self.use_local_dilated = use_local_dilated
        if use_local_dilated:
            self.local_dw3_d2 = nn.Conv2d(dim, dim, 3, 1, 2, dilation=2, groups=dim, bias=False)
            self.local_bn3_d2 = nn.BatchNorm2d(dim)
        self.local_pw = Conv(dim, dim, 1, 1)

        # Positional (DWConv 3x3)
        self.pe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

        # Gating động theo ảnh cho out2, out4: GAP(x) -> MLP nhỏ -> 2 weights
        self.beta = nn.Parameter(torch.tensor([0.0, -0.5]))  # bias học được (khởi tạo nghiêng về ×2)
        self.gate_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # [B,C,1,1]
            nn.Flatten(1),                      # [B,C]
            nn.Linear(dim, 32, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(32, 2, bias=True)         # [B,2] ~ (w2, w4)
        )

        self.proj = Conv(dim, dim, 1, act=False)

    def _shape(self, x, B, H, W, Ckv):
        return x.view(B, self.num_heads, Ckv, H * W)

    def _attend_to_scale(self, q, k, v, Hq, Wq):
        attn = (q.transpose(-2, -1) @ k) * self.scale            # [B,h,Nq,Nk]
        attn = attn.softmax(dim=-1)
        out  = (v @ attn.transpose(-2, -1))                      # [B,h,dv,Nq]
        out  = out.view(q.shape[0], self.num_heads, v.shape[2], Hq, Wq)
        return out.flatten(1, 2)                                  # [B,C,Hq,Wq]

    def forward(self, x):
        B, C, H, W = x.shape

        # Q full-res
        q = self._shape(self.q_proj(x), B, H, W, self.key_dim)

        # K,V @ ×2
        x2 = F.avg_pool2d(x, 2, 2)
        k2 = self._shape(self.k_proj(x2), B, x2.shape[2], x2.shape[3], self.key_dim)
        v2 = self._shape(self.v_proj(x2), B, x2.shape[2], x2.shape[3], self.head_dim)
        out2 = self._attend_to_scale(q, k2, v2, H, W)

        # K,V @ ×4
        x4 = F.avg_pool2d(x2, 2, 2)
        k4 = self._shape(self.k_proj(x4), B, x4.shape[2], x4.shape[3], self.key_dim)
        v4 = self._shape(self.v_proj(x4), B, x4.shape[2], x4.shape[3], self.head_dim)
        out4 = self._attend_to_scale(q, k4, v4, H, W)

        # Local detail
        loc = self.local_bn3(self.local_dw3(x))
        if self.use_local_dilated:
            loc = loc + self.local_bn3_d2(self.local_dw3_d2(x))
        loc = F.silu(loc, inplace=True)
        loc = self.local_pw(loc)

        # Positional bias
        pos = self.pe(x)

        # Gating động: w = sigmoid(MLP(x) + beta)
        w2w4 = torch.sigmoid(self.gate_mlp(x) + self.beta)       # [B,2] in [0,1]
        w2 = w2w4[:, 0].view(B, 1, 1, 1)
        w4 = w2w4[:, 1].view(B, 1, 1, 1)

        y = loc + w2 * out2 + w4 * out4 + pos
        return self.proj(y)

class PSABlock1(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, drop_path=0.0):
        super().__init__()
        self.add  = shortcut
        self.attn = AttentionMS(c, num_heads=num_heads, attn_ratio=attn_ratio, use_local_dilated=True)
        self.ffn  = ConvFFN(c, expand=2.0, drop_path=drop_path)
    def forward(self, x):
        if self.add:
            x = x + self.attn(x)
            x = x + self.ffn(x)
        else:
            x = self.ffn(self.attn(x))
        return x

class PSA1(nn.Module):
    def __init__(self, c1, c2, e=0.5, drop_path=0.0):
        super().__init__()
        assert c1 == c2
        self.c   = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        heads = max(1, self.c // 64)
        self.attn = PSABlock(self.c, attn_ratio=0.5, num_heads=heads, drop_path=drop_path)
        self.ffn  = ConvFFN(self.c, expand=2.0, drop_path=drop_path)
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

class C2PSA1(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, drop_path=0.0):
        super().__init__()
        assert c1 == c2
        self.c   = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        dplist = [0.0] if n <= 1 else [drop_path * i / (n - 1) for i in range(n)]
        heads = max(1, self.c // 64)
        self.m = nn.Sequential(*(
            PSABlock(self.c, attn_ratio=0.5, num_heads=heads, drop_path=dplist[i])
            for i in range(n)
        ))
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

# ==========================

class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """
        Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """
        Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))
class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided, default is 1.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x):
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """
        Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        """
        Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """
        Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y

class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(Conv(c1, c2, k=7, s=2, p=3, act=True),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)
    
class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        final = torch.add(x[0], x[1])
        return final
    
class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            # final = torch.add(x[0], x[1][0])
            # final1 = torch.add(x[0], x[1][1])
            # final2 = torch.add(final, final1)
            # map_rgb = torch.unsqueeze(torch.mean(final2, 1), 1)
            # score2 = F.interpolate(map_rgb, size=(40, 40), mode="bilinear", align_corners=True)
            # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
            # depth = (score2 - score2.min()) / (score2.max() - score2.min())
            # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
            # plt.imshow(feature_img)
            # plt.show()
            # plt.savefig("29.png")
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            #print("x[0].shape:", x[0].shape)
            #print("x[1][1].shape:", x[1][1].shape)
            x1 = list(x[1])

            # Resize x1[1] về cùng spatial size với x[0] nếu cần
            if x[0].shape[2:] != x1[1].shape[2:]:
                x1[1] = F.interpolate(x1[1], size=x[0].shape[2:], mode='bilinear', align_corners=False)

            # Cộng sau khi đã khớp shape
            out = x[0] + x1[1]  # hoặc torch.add(x[0], x1[1]) cũng được
            #out = torch.add(x[0], x[1][1])
            # map_rgb = torch.unsqueeze(torch.mean(out, 1), 1)
            # score2 = F.interpolate(map_rgb, size=(40, 40), mode="bilinear", align_corners=True)
            # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
            # depth = (score2 - score2.min()) / (score2.max() - score2.min())
            # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
            # plt.imshow(feature_img)
            # plt.show()
            # plt.savefig("30.png")
            return out

        # return torch.add(x[0], x[1])
# TGF cai tien (Ket hop FRM)
class ECAAttention(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                                 # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))       # (B,1,C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B,C,1,1)
        return x * y.expand_as(x)

# ---------- MHSA + SRA (KV) ----------
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, Ha: int, Wa: int, sr_ratio: int = 1,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.C = d_model
        self.Ha, self.Wa = Ha, Wa
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.scale = self.dk ** -0.5
        self.sr = max(1, int(sr_ratio))

        self.q_proj  = nn.Linear(d_model, d_model, bias=True)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=True)

        if self.sr > 1:
            self.sr_conv = nn.Conv2d(d_model, d_model, kernel_size=self.sr, stride=self.sr, padding=0, bias=False)
            self.sr_bn   = nn.BatchNorm2d(d_model)
        else:
            self.sr_conv = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(d_model, d_model, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C), N = 2*Ha*Wa -> (B, C, 2Ha, Wa)
        B, N, C = x.shape
        assert N == 2 * self.Ha * self.Wa
        return x.view(B, 2 * self.Ha, self.Wa, C).permute(0, 3, 1, 2).contiguous()

    def _map_to_tokens(self, x_map: torch.Tensor) -> torch.Tensor:
        # (B, C, Hs, Ws) -> (B, Ns, C)
        B, C, Hs, Ws = x_map.shape
        return x_map.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x)  # (B,N,C)

        if self.sr_conv is not None:
            x_map = self._tokens_to_map(x)               # (B,C,2Ha,Wa)
            x_sr  = self.sr_bn(self.sr_conv(x_map))      # (B,C,2Ha/s,Wa/s)
            kv_in = self._map_to_tokens(x_sr)            # (B,Ns,C)
        else:
            kv_in = x                                    # (B,N,C)

        kv = self.kv_proj(kv_in)
        k, v = kv.chunk(2, dim=-1)  # (B,Ns,C)

        def split_heads(t):
            B_, Nt, Ct = t.shape
            return t.view(B_, Nt, self.num_heads, self.dk).permute(0, 2, 1, 3).contiguous()

        q = split_heads(q)  # (B,h,N, dk)
        k = split_heads(k)  # (B,h,Ns,dk)
        v = split_heads(v)  # (B,h,Ns,dk)

        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B,h,N,Ns)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out  = attn @ v                                   # (B,h,N,dk)

        out = out.transpose(1, 2).reshape(B, N, C)        # (B,N,C)
        out = self.proj_drop(self.proj(out))
        return out

# ---------- DW-MLP ----------
class DW_MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, Ha: int, Wa: int, drop: float = 0.0):
        super().__init__()
        self.Ha, self.Wa = Ha, Wa
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        self.dw    = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=True)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_features, in_features, 1, bias=True)
        self.drop  = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        assert N == 2 * self.Ha * self.Wa
        x = x.permute(0, 2, 1).reshape(B, C, 2 * self.Ha, self.Wa)
        x = self.act(self.dw(self.conv1(x)))
        x = self.conv2(x)
        x = x.flatten(2).transpose(1, 2)
        return self.drop(x)

# ---------- Transformer Block ----------
class myTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, Ha: int, Wa: int,
                 block_exp: int = 2, attn_drop: float = 0.0, resid_drop: float = 0.0, sr_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = SelfAttention(d_model, num_heads, Ha, Wa, sr_ratio, attn_drop, resid_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = DW_MLP(d_model, block_exp * d_model, Ha, Wa, drop=resid_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ---------- Cross-Modal Gating (FRM-like, anchor resolution) ----------
class CrossModalGating(nn.Module):
    def __init__(self, dim: int, reduction: int = 4, lambda_c: float = 0.3, lambda_s: float = 0.3):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        mid = max(1, dim // reduction)

        # Channel weights: MLP trên vector pooled (avg + max) của concat [rgb, ir]
        self.ch_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim * 2), nn.Sigmoid()
        )
        # Spatial weights: 1x1 -> mid -> 1x1 -> 2 channels (sigmoid)
        self.sp_mlp = nn.Sequential(
            nn.Conv2d(dim * 2, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        # xr, xi: (B, C, Ha, Wa)
        B, C, Ha, Wa = xr.shape
        x_cat = torch.cat([xr, xi], dim=1)      # (B, 2C, Ha, Wa)

        # Channel weights
        avg = torch.mean(x_cat, dim=[2, 3])     # (B, 2C)
        maxp = F.adaptive_max_pool2d(x_cat, 1).view(B, 2 * C)  # (B, 2C)
        ch = self.ch_mlp(torch.cat([avg, maxp], dim=1))        # (B, 2C)
        ch = ch.view(B, 2, C, 1, 1)                            # (B,2,C,1,1)
        w_c_rgb = ch[:, 0]; w_c_ir = ch[:, 1]                 # (B,C,1,1) mỗi cái

        # Spatial weights
        sp = self.sp_mlp(x_cat)                                # (B,2,Ha,Wa)
        w_s_rgb = sp[:, 0:1, :, :]                             # (B,1,Ha,Wa)
        w_s_ir  = sp[:, 1:1+1, :, :]

        # Gated residual exchange
        xr_out = xr + self.lambda_c * w_c_ir * xi + self.lambda_s * w_s_ir * xi
        xi_out = xi + self.lambda_c * w_c_rgb * xr + self.lambda_s * w_s_rgb * xr
        return xr_out, xi_out

class TGF(nn.Module):
    def __init__(self, d_model: int, vert_anchors: int, horz_anchors: int, n_layer: int,
                 num_heads: int = 4, block_exp: int = 2, embd_drop: float = 0.0,
                 attn_drop: float = 0.0, resid_drop: float = 0.0, sr_ratio: int = 2,
                 gate_reduction: int = 4, lambda_c: float = 0.3, lambda_s: float = 0.3):
        super().__init__()
        self.C = d_model
        self.Ha, self.Wa = vert_anchors, horz_anchors
        self.N = 2 * self.Ha * self.Wa

        self.pos_emb = nn.Parameter(torch.zeros(1, self.N, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        blocks = []
        for _ in range(n_layer):
            blocks.append(myTransformerBlock(d_model, num_heads, self.Ha, self.Wa,
                                             block_exp=block_exp, attn_drop=attn_drop,
                                             resid_drop=resid_drop, sr_ratio=sr_ratio))
        self.trans_blocks = nn.Sequential(*blocks)
        self.ln_f  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(embd_drop)
        self.pool  = nn.AdaptiveAvgPool2d((self.Ha, self.Wa))

        # Cross-Modal Gating (FRM-like, nhẹ @ (Ha,Wa))
        self.xgate = CrossModalGating(dim=d_model, reduction=gate_reduction,
                                      lambda_c=lambda_c, lambda_s=lambda_s)

        # ECA trên ghép kênh 2C @ (Ha,Wa)
        self.eca = ECAAttention(d_model * 2)
        # Mapping groups=2 để tách lại 2 modality (giảm 1/2 tham số so với 2 conv độc lập)
        self.map_pair = nn.Conv2d(d_model * 2, d_model * 2, kernel_size=1, groups=2, bias=True)

    def forward(self, x):
        rgb, ir = x  # (B,C,H,W)
        B, C, H, W = rgb.shape
        assert C == self.C

        # 1) Anchor pooling
        rgb_p = self.pool(rgb)   # (B,C,Ha,Wa)
        ir_p  = self.pool(ir)

        # 2) Token hóa
        rgb_tok = rgb_p.flatten(2).transpose(1, 2)  # (B,Ha*Wa,C)
        ir_tok  = ir_p.flatten(2).transpose(1, 2)
        tokens  = torch.cat([rgb_tok, ir_tok], dim=1)  # (B,N,C)

        # 3) Transformer
        t = self.drop(tokens + self.pos_emb)     # (B,N,C)
        t = self.trans_blocks(t)                 # (B,N,C)
        t = self.ln_f(t)

        # 4) Reshape -> 2 map @ (Ha,Wa)
        t = t.view(B, 2, self.Ha * self.Wa, C).permute(0, 1, 3, 2).contiguous()
        rgb_map = t[:, 0].view(B, C, self.Ha, self.Wa)
        ir_map  = t[:, 1].view(B, C, self.Ha, self.Wa)

        # 5) Cross-Modal Gating (FRM-like, nhẹ)
        rgb_map, ir_map = self.xgate(rgb_map, ir_map)

        # 6) ECA + mapping groups=2
        all_map = torch.cat([rgb_map, ir_map], dim=1)  # (B,2C,Ha,Wa)
        all_map = self.eca(all_map)
        all_map = self.map_pair(all_map)               # (B,2C,Ha,Wa)

        rgb_out, ir_out = torch.chunk(all_map, 2, dim=1)  # (B,C,Ha,Wa) x2

        # 7) Upsample về (H,W)
        rgb_out = F.interpolate(rgb_out, size=(H, W), mode='bilinear', align_corners=False)
        ir_out  = F.interpolate(ir_out,  size=(H, W), mode='bilinear', align_corners=False)
        return rgb_out, ir_out
class EMA(nn.Module):
   
    def __init__(self,
                 c1: int,
                 c2: int = None,
                 groups: int = 4,
                 dw_kernel: int = 3,
                 use_shuffle: bool = True,
                 scales: tuple = (1, 2, 4)):
        super().__init__()
        # --- chuẩn hóa kênh & tham số ---
        channels = c2 if c2 is not None else c1
        assert channels > 0, "EMA: channels must be > 0"
        if isinstance(scales, list):  # YAML có thể cho list
            scales = tuple(scales)
        assert len(scales) >= 1, "EMA: scales must have at least one element"
        self.c = channels
        self.g = max(1, int(groups))
        self.gc = channels // self.g
        self.use_shuffle = bool(use_shuffle)
        self.scales = scales

        # --- depthwise conv cho mỗi nhánh scale ---
        p = (dw_kernel - 1) // 2
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=dw_kernel, padding=p, groups=channels, bias=False)
            for _ in self.scales
        ])
        # fuse len(scales)*C -> C
        self.fuse = nn.Conv2d(channels * len(self.scales), channels, kernel_size=1, bias=False)

        # khối tạo attention HxW (lightweight)
        self.attn_proj1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.attn_dw    = nn.Conv2d(channels, channels, kernel_size=dw_kernel, padding=p, groups=channels, bias=False)
        self.attn_proj2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.Sigmoid()

        # trộn nhóm sau cùng (tùy chọn)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, groups=self.g, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @staticmethod
    def _channel_shuffle(x, groups: int):
        if groups <= 1:
            return x
        b, c, h, w = x.size()
        x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feats = []
        for i, s in enumerate(self.scales):
            xi = x
            if s > 1:
                xi = F.avg_pool2d(xi, kernel_size=s, stride=s, ceil_mode=False)
            xi = self.dw_convs[i](xi)
            if s > 1:
                xi = F.interpolate(xi, size=(h, w), mode='bilinear', align_corners=False)
            feats.append(xi)

        ms = torch.cat(feats, dim=1)   # (B, len(scales)*C, H, W)
        ms = self.fuse(ms)             # (B, C, H, W)

        if self.use_shuffle:
            ms = self._channel_shuffle(ms, self.g)

        attn = self.attn_proj1(ms)
        attn = self.attn_dw(attn)
        attn = self.attn_proj2(attn)
        attn = self.act(attn)

        y = x * attn + x
        y = self.mix(y)
        return y
    
from torch import Tensor

def _make_mult(x, m=8):
    return max(m, int(x + m - 1) // m * m)  # làm tròn lên bội số m

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1, bias=False, d=1, act='silu'):
        super().__init__()
        if p is None: p = d * (k - 1) // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=bias, dilation=d)
        self.bn   = nn.BatchNorm2d(c_out)
        if act == 'relu':   self.act = nn.ReLU(inplace=True)
        elif act == 'lrelu':self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act == 'hswish': self.act = nn.Hardswish()
        else:               self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class DWConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, d=1, act='silu'):
        super().__init__()
        self.dw = ConvBNAct(c_in, c_in, k=k, s=s, g=c_in, d=d, act=act)
        self.pw = ConvBNAct(c_in, c_out, k=1, s=1, act=act)
    def forward(self, x): return self.pw(self.dw(x))

class MSFEMFastUnit(nn.Module):
    
    def __init__(self, C: int, Cm: int, act='silu'):
        super().__init__()
        # chia đều cho 3 nhánh, làm tròn từng nhánh theo bội số 8
        c_each = max(1, Cm // 3)
        # phân phối dư cho nhánh A
        cA = _make_mult(c_each + (Cm - 3*c_each), 8)
        cB = _make_mult(c_each, 8)
        cC = _make_mult(c_each, 8)
        Cm_eff = cA + cB + cC  # tổng chính xác dùng để fuse

        self.stem = ConvBNAct(C, Cm_eff, k=1, act=act)
        self.a = ConvBNAct(Cm_eff, cA, k=1, act=act)
        self.b = DWConv(Cm_eff, cB, k=3, d=1, act=act)
        self.c = DWConv(Cm_eff, cC, k=5, d=1, act=act)  # nhanh hơn dilation
        self.fuse = ConvBNAct(Cm_eff, C, k=1, act=act)
        self.res  = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        z = self.stem(x)
        m = torch.cat([self.a(z), self.b(z), self.c(z)], dim=1)
        y = self.fuse(m)
        return y + self.res(x)

class CAFEM(nn.Module):
    
    def __init__(self, c1:int, c2:int, n:int=1, c3k:bool=False, e:float=0.5, g:int=1,
                 shortcut:bool=True, act:str='silu'):
        super().__init__()
        self.align = ConvBNAct(c1, c2, k=1, act=act) if c1 != c2 else nn.Identity()
        Cm = _make_mult(int(round(c2 * float(e))), 8)  # ép bội số 8
        blocks = [MSFEMFastUnit(c2, Cm, act=act) for _ in range(max(1, int(n)))]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.align(x)
        return self.blocks(x)



class GOCI(nn.Module):
    """Global Orthogonal Procrustes alignment with bounded reliability scaling.

    The module contains no trainable parameters. Statistics are estimated on a
    compact anchor grid, accumulated as buffers, and then applied to the full
    feature map. Running transforms can be frozen for TorchScript/ONNX export.
    """

    def __init__(self, channels: int, groups: int = 32, anchors: int = 8,
                 momentum: float = 0.03, eps: float = 1e-3,
                 reliability: bool = True, reliability_gamma: float = 1.5,
                 trigger_tau: float = 0.6, trigger_k: float = 12.0):
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups != 0:
            raise ValueError(f'channels={channels} must be positive and divisible by groups={groups}')
        if anchors < 2:
            raise ValueError(f'anchors={anchors} must be at least 2 for non-degenerate covariance estimates')
        self.channels = int(channels)
        self.groups = int(groups)
        self.group_width = self.channels // self.groups
        self.anchors = int(anchors)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.reliability = bool(reliability)
        self.reliability_gamma = float(reliability_gamma)
        self.trigger_tau = float(trigger_tau)
        self.trigger_k = float(trigger_k)
        d, g = self.group_width, self.groups
        eye = torch.eye(d).repeat(g, 1, 1)
        self.register_buffer('running_mu_r', torch.zeros(g, 1, d))
        self.register_buffer('running_mu_i', torch.zeros(g, 1, d))
        self.register_buffer('running_cov_r', eye.clone())
        self.register_buffer('running_cov_i', eye.clone())
        self.register_buffer('running_cross', eye.clone())
        self.register_buffer('num_updates', torch.zeros((), dtype=torch.long))
        # Export buffers contain the already-solved transform, avoiding linalg
        # operators that are not uniformly supported by deployment runtimes.
        self.register_buffer('export_mu_r', torch.zeros(g, 1, d), persistent=False)
        self.register_buffer('export_mu_i', torch.zeros(g, 1, d), persistent=False)
        self.register_buffer('export_wr', eye.clone(), persistent=False)
        self.register_buffer('export_wi', eye.clone(), persistent=False)
        self.register_buffer('export_q', eye.clone(), persistent=False)
        self.pool = nn.AdaptiveAvgPool2d((self.anchors, self.anchors))
        self.export_mode = False
        self._cached_version = None
        self._cached_eval = None

    def _apply(self, fn):
        """Apply a device/dtype transform and invalidate derived transform caches.

        ``_cached_eval`` is intentionally not a registered buffer because it is
        derived from the running statistics.  PyTorch therefore does not move it
        during ``module.to(...)``/``half()``/``float()``.  A checkpoint may also
        contain a stale CPU cache.  Clearing it here guarantees that the next
        evaluation recomputes the transforms from buffers on the target device.
        """
        result = super()._apply(fn)
        self._cached_eval = None
        self._cached_version = None
        return result

    def __getstate__(self):
        """Exclude derived evaluation transforms from Python checkpoints.

        The cache is entirely determined by registered running-statistic buffers.
        Serializing it increases checkpoint size and can retain tensors from the
        device on which evaluation last ran.  Dropping it makes raw ``torch.save``
        checkpoints as safe as the repository's stripped training checkpoints.
        """
        state = super().__getstate__()
        state['_cached_eval'] = None
        state['_cached_version'] = None
        return state

    def __setstate__(self, state):
        """Restore a module while guaranteeing that derived caches are fresh."""
        super().__setstate__(state)
        self._cached_eval = None
        self._cached_version = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Invalidate derived transforms after loading authoritative buffers.

        ``load_state_dict`` can replace running statistics while leaving the old
        update counter unchanged.  A version-only cache key would then reuse stale
        transforms.  Clearing the cache here guarantees recomputation from the newly
        loaded buffers.  Non-persistent export transforms are also rebuilt on demand.
        """
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self._cached_eval = None
        self._cached_version = None
        self.export_mode = False

    def _group_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return x.reshape(b, self.groups, self.group_width, h * w).permute(1, 0, 3, 2).reshape(
            self.groups, b * h * w, self.group_width)

    def _batch_stats(self, r: torch.Tensor, i: torch.Tensor):
        rp, ip = self.pool(r), self.pool(i)
        # Statistics define a coordinate transform and need not carry gradients.
        rt, it = self._group_tokens(rp.detach().float()), self._group_tokens(ip.detach().float())
        mu_r, mu_i = rt.mean(1, keepdim=True), it.mean(1, keepdim=True)
        rc, ic = rt - mu_r, it - mu_i
        n = max(1, rt.shape[1] - 1)
        cov_r = rc.transpose(-1, -2) @ rc / n
        cov_i = ic.transpose(-1, -2) @ ic / n
        cross = rc.transpose(-1, -2) @ ic / n
        return mu_r, mu_i, cov_r, cov_i, cross

    @staticmethod
    def _autocast_disabled(device: torch.device):
        # CUDA autocast can cast FP32 matrix products back to FP16 before SVD.
        # Explicitly disabling autocast around the full linear-algebra block keeps
        # eigh/SVD and their input products in FP32 on every supported backend.
        if device.type in {'cpu', 'cuda', 'xpu'}:
            return torch.autocast(device_type=device.type, enabled=False)
        return contextlib.nullcontext()

    @torch.no_grad()
    def _update_running(self, stats):
        buffers = (self.running_mu_r, self.running_mu_i, self.running_cov_r,
                   self.running_cov_i, self.running_cross)
        safe_stats = tuple(value.detach().to(device=buffer.device, dtype=buffer.dtype)
                           for value, buffer in zip(stats, buffers))
        if int(self.num_updates.item()) == 0:
            for buffer, value in zip(buffers, safe_stats):
                buffer.copy_(value)
        else:
            for buffer, value in zip(buffers, safe_stats):
                buffer.lerp_(value, self.momentum)
        self.num_updates.add_(1)

    def _transforms(self, mu_r, mu_i, cov_r, cov_i, cross):
        d = self.group_width
        with self._autocast_disabled(cov_r.device):
            mu_r, mu_i = mu_r.float(), mu_i.float()
            cov_r, cov_i, cross = cov_r.float(), cov_i.float(), cross.float()
            eye = torch.eye(d, device=cov_r.device, dtype=torch.float32).expand(self.groups, -1, -1)
            er, vr = torch.linalg.eigh(cov_r + self.eps * eye)
            ei, vi = torch.linalg.eigh(cov_i + self.eps * eye)
            wr = vr @ torch.diag_embed(er.clamp_min(self.eps).rsqrt()) @ vr.transpose(-1, -2)
            wi = vi @ torch.diag_embed(ei.clamp_min(self.eps).rsqrt()) @ vi.transpose(-1, -2)
            u, _, vh = torch.linalg.svd(wr @ cross @ wi, full_matrices=False)
            q = u @ vh
        return mu_r, mu_i, wr, wi, q

    @torch.no_grad()
    def prepare_for_export(self):
        """Freeze running transforms into deployment-safe buffers."""
        stats = (self.running_mu_r, self.running_mu_i, self.running_cov_r,
                 self.running_cov_i, self.running_cross)
        mu_r, mu_i, wr, wi, q = self._transforms(*stats)
        self.export_mu_r.copy_(mu_r)
        self.export_mu_i.copy_(mu_i)
        self.export_wr.copy_(wr)
        self.export_wi.copy_(wi)
        self.export_q.copy_(q)
        self.export_mode = True
        self._cached_eval = None
        return self

    def _get_transforms(self, r, i):
        if self.export_mode or torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            return self.export_mu_r, self.export_mu_i, self.export_wr, self.export_wi, self.export_q
        if self.training:
            stats = self._batch_stats(r, i)
            self._update_running(stats)
            self._cached_eval = None
            self._cached_version = None
            return self._transforms(*stats)
        stats = (self.running_mu_r, self.running_mu_i, self.running_cov_r,
                 self.running_cov_i, self.running_cross)
        version = int(self.num_updates.item())
        cache_device_ok = (self._cached_eval is not None and
                           all(t.device == r.device for t in self._cached_eval))
        if self._cached_eval is None or self._cached_version != version or not cache_device_ok:
            # A serialized model can carry a derived cache created on CPU even
            # after registered buffers have moved to CUDA.  Recompute from the
            # authoritative running buffers on the current input device.
            device_stats = tuple(t.to(device=r.device, dtype=torch.float32) for t in stats)
            self._cached_eval = tuple(t.detach() for t in self._transforms(*device_stats))
            self._cached_version = version
        return self._cached_eval

    def _apply_whitening(self, x, mu, matrix):
        b, c, h, width = x.shape
        dtype = x.dtype
        # Defensive device normalization also protects old checkpoints that may
        # contain a non-buffer derived cache on CPU.
        mu = mu.to(device=x.device, dtype=torch.float32)
        matrix = matrix.to(device=x.device, dtype=torch.float32)
        xt = x.float().reshape(b, self.groups, self.group_width, h * width).permute(0, 1, 3, 2)
        y = (xt - mu.unsqueeze(0)) @ matrix.unsqueeze(0)
        return y.permute(0, 1, 3, 2).reshape(b, c, h, width).to(dtype)

    def _align(self, r, i):
        mu_r, mu_i, wr, wi, q = self._get_transforms(r, i)
        a = self._apply_whitening(r, mu_r, wr)
        b = self._apply_whitening(i, mu_i, wi)
        bs, c, h, width = a.shape
        dtype = a.dtype
        at = a.float().reshape(bs, self.groups, self.group_width, h * width).permute(0, 1, 3, 2)
        q = q.to(device=at.device, dtype=torch.float32)
        at = at @ q.unsqueeze(0)
        a = at.permute(0, 1, 3, 2).reshape(bs, c, h, width).to(dtype)
        return a, b

    def _reliability_terms(self, a, b):
        """Return modality probabilities and the bounded correction trigger.

        The statistics are computed in FP32 so AMP/BF16 execution cannot turn
        a small energy estimate into an underflowing logarithm.  The returned
        probabilities lie on the two-simplex and are invariant to orthogonal
        channel rotations because they depend only on squared feature norms.
        """
        # The whitening transform is estimated on the anchor grid. Reliability
        # must therefore be measured on the same statistical domain; measuring
        # it on the unpooled map systematically inflates energy whenever the
        # anchor grid averages multiple feature cells.
        a_stat = self.pool(a.float())
        b_stat = self.pool(b.float())
        energy_r = a_stat.square().mean((1, 2, 3), keepdim=True)
        energy_i = b_stat.square().mean((1, 2, 3), keepdim=True)
        dev_r = torch.log(energy_r + 1e-4).abs()
        dev_i = torch.log(energy_i + 1e-4).abs()
        probability = torch.cat((-self.reliability_gamma * dev_r,
                                 -self.reliability_gamma * dev_i), 1).softmax(1)
        trigger = torch.sigmoid(self.trigger_k * (torch.maximum(dev_r, dev_i) - self.trigger_tau))
        return probability, trigger, energy_r, energy_i

    def _reliability_output(self, a, b):
        if not self.reliability:
            return torch.cat((a, b), dim=1)
        probability, trigger, _, _ = self._reliability_terms(a, b)
        scale_r = (2.0 * probability[:, :1]).sqrt().to(a.dtype)
        scale_i = (2.0 * probability[:, 1:]).sqrt().to(b.dtype)
        trigger = trigger.to(a.dtype)
        out_r = (1.0 - trigger) * a + trigger * scale_r * a
        out_i = (1.0 - trigger) * b + trigger * scale_i * b
        return torch.cat((out_r, out_i), dim=1)

    def forward(self, x):
        a, b = self._align(x[0], x[1])
        return self._reliability_output(a, b)


class SJPA(GOCI):
    """Selective Spatial Procrustes Alignment with trust-region reliability.

    The production algorithm is deliberately sequential rather than claiming a
    single joint non-convex optimum. It first estimates a groupwise whitening
    and orthogonal channel map from running RGB--IR statistics. It then scores a
    finite set of zero-padded translations by the Procrustes-optimal correlation
    (the nuclear norm of each cross-covariance) and applies a shift only when the
    pair passes the reliability and score gates. Finally, a simplex-valued,
    norm-bounded reliability correction rescales the aligned modalities.

    Native PyTorch uses the nuclear norm for candidate scoring. Export mode uses
    the Frobenius norm, which is still orthogonally invariant and avoids SVD
    operators that are unsupported by several ONNX/TensorRT toolchains. The two
    scores are not mathematically identical, so deployment equivalence must be
    reported and tested rather than assumed.
    """

    def __init__(self, channels: int, groups: int = 32, anchors: int = 8,
                 momentum: float = 0.03, eps: float = 1e-3,
                 reliability: bool = True, reliability_gamma: float = 1.5,
                 trigger_tau: float = 0.6, trigger_k: float = 12.0,
                 max_shift: int = 1, shift_penalty: float = 0.1,
                 score_threshold: float = 0.3, shift_margin: float = 0.0):
        super().__init__(channels, groups, anchors, momentum, eps, reliability,
                         reliability_gamma, trigger_tau, trigger_k)
        if max_shift < 0:
            raise ValueError('max_shift must be non-negative')
        self.max_shift = int(max_shift)
        self.shift_penalty = float(shift_penalty)
        self.score_threshold = float(score_threshold)
        self.shift_margin = float(shift_margin)
        if self.shift_margin < 0:
            raise ValueError('shift_margin must be non-negative')

    def _score_cross(self, cross):
        # Shift selection is discrete, so gradients through the score cannot
        # influence the chosen branch. Detaching avoids retaining one SVD graph
        # per candidate. The matrix product can still be autocast by an outer
        # AMP context, therefore both conversion and SVD run with autocast off.
        with self._autocast_disabled(cross.device):
            cross = cross.detach().float()
            if self.export_mode or torch.onnx.is_in_onnx_export():
                return cross.square().sum(dim=(-2, -1)).clamp_min(0).sqrt().sum(-1)
            return torch.linalg.svdvals(cross).sum((-1, -2))

    @staticmethod
    def _shift_no_wrap(x, dy: int, dx: int):
        """Translate a feature map with zero padding instead of circular wraparound."""
        if dy == 0 and dx == 0:
            return x
        h, w = x.shape[-2:]
        top, bottom = max(dy, 0), max(-dy, 0)
        left, right = max(dx, 0), max(-dx, 0)
        padded = F.pad(x, (left, right, top, bottom), mode='constant', value=0.0)
        start_y, start_x = bottom, right
        return padded[..., start_y:start_y + h, start_x:start_x + w]

    def _select_shift(self, a, b, eligible, return_diagnostics: bool = False):
        bs = a.shape[0]
        ap, bp0 = self.pool(a.float()), self.pool(b.float())
        at = ap.reshape(bs, self.groups, self.group_width, -1).permute(0, 1, 3, 2)
        bt0 = bp0.reshape(bs, self.groups, self.group_width, -1).permute(0, 1, 3, 2)
        at = at - at.mean(-2, keepdim=True)
        bt0 = bt0 - bt0.mean(-2, keepdim=True)
        n = at.shape[-2] - 1
        zero_cross = at.transpose(-1, -2) @ bt0 / n
        zero_score = self._score_cross(zero_cross) / (self.groups * self.group_width)
        eligible = eligible & (zero_score < self.score_threshold)

        scores, shifted = [], []
        shifts = [(dy, dx) for dy in range(-self.max_shift, self.max_shift + 1)
                  for dx in range(-self.max_shift, self.max_shift + 1)]
        for dy, dx in shifts:
            candidate = self._shift_no_wrap(b, dy, dx)
            bp = self.pool(candidate.float())
            bt = bp.reshape(bs, self.groups, self.group_width, -1).permute(0, 1, 3, 2)
            bt = bt - bt.mean(-2, keepdim=True)
            cross = at.transpose(-1, -2) @ bt / n
            score = (
                self._score_cross(cross) / (self.groups * self.group_width)
                - self.shift_penalty * float(dy * dy + dx * dx)
            )
            scores.append(score)
            shifted.append(candidate)
        score_matrix = torch.stack(scores, dim=1)
        best_score, selected = score_matrix.max(1)
        zero_index = shifts.index((0, 0))
        improvement = best_score - score_matrix[:, zero_index]
        accept_shift = eligible & (improvement > self.shift_margin)
        selected = torch.where(accept_shift, selected, torch.full_like(selected, zero_index))
        stack = torch.stack(shifted, dim=1)  # B, S, C, H, W
        gather_index = selected.view(bs, 1, 1, 1, 1).expand(-1, 1, b.shape[1], b.shape[2], b.shape[3])
        output = stack.gather(1, gather_index).squeeze(1)
        if not return_diagnostics:
            return output
        shift_tensor = torch.tensor(shifts, device=selected.device, dtype=torch.long)
        return output, {
            'selected_index': selected,
            'selected_shift': shift_tensor[selected],
            'eligible': eligible,
            'accept_shift': accept_shift,
            'zero_score': zero_score,
            'improvement': improvement,
            'score_matrix': score_matrix,
        }

    def forward_with_diagnostics(self, x):
        """Run the module and expose non-persistent diagnostics for tests/analysis."""
        a, b = self._align(x[0], x[1])
        probability_pre, trigger_pre, energy_r, energy_i = self._reliability_terms(a, b)
        dev_r = torch.log(energy_r + 1e-4).abs()
        dev_i = torch.log(energy_i + 1e-4).abs()
        anomaly = torch.maximum(dev_r, dev_i)
        b, shift = self._select_shift(a, b, anomaly.flatten() < self.trigger_tau, return_diagnostics=True)
        probability, trigger, energy_r_post, energy_i_post = self._reliability_terms(a, b)
        output = self._reliability_output(a, b)
        diagnostics = {
            **shift,
            'probability_pre_shift': probability_pre,
            'trigger_pre_shift': trigger_pre,
            'probability': probability,
            'trigger': trigger,
            'energy_r': energy_r_post,
            'energy_i': energy_i_post,
            'aligned_rgb': a,
            'aligned_ir': b,
        }
        return output, diagnostics

    def forward(self, x):
        a, b = self._align(x[0], x[1])
        probability, trigger, energy_r, energy_i = self._reliability_terms(a, b)
        del probability, trigger  # values are recomputed after the selected shift
        dev_r = torch.log(energy_r + 1e-4).abs()
        dev_i = torch.log(energy_i + 1e-4).abs()
        anomaly = torch.maximum(dev_r, dev_i)
        b = self._select_shift(a, b, anomaly.flatten() < self.trigger_tau)
        return self._reliability_output(a, b)


class RTPF(SJPA):
    """Residual Trust-region Procrustes Fusion.

    RTPF preserves the direct RGB--IR concatenation path and injects the
    canonical SJPA representation only through a norm-projected residual.  The
    scalar residual step is initialized at zero, so the module is exactly equal
    to direct concatenation at initialization.  For every sample ``j`` the
    correction obeys

        ||RTPF_j - concat_j||_F <= trust_radius * ||concat_j||_F.

    A deterministic fallback gate suppresses the canonical correction when both
    modalities are statistically atypical in the same pooled whitening domain.
    """

    def __init__(self, channels: int, groups: int = 32, anchors: int = 8,
                 momentum: float = 0.03, eps: float = 1e-3,
                 reliability: bool = True, reliability_gamma: float = 1.5,
                 trigger_tau: float = 0.6, trigger_k: float = 12.0,
                 max_shift: int = 1, shift_penalty: float = 0.1,
                 score_threshold: float = 0.3, shift_margin: float = 0.02,
                 trust_radius: float = 1.0, fallback_tau: float = 1.0,
                 fallback_k: float = 8.0):
        super().__init__(channels, groups, anchors, momentum, eps, reliability,
                         reliability_gamma, trigger_tau, trigger_k, max_shift,
                         shift_penalty, score_threshold, shift_margin)
        if trust_radius <= 0:
            raise ValueError('trust_radius must be positive')
        if fallback_k <= 0:
            raise ValueError('fallback_k must be positive')
        self.trust_radius = float(trust_radius)
        self.fallback_tau = float(fallback_tau)
        self.fallback_k = float(fallback_k)
        # softsign(step_parameter) is exactly zero at initialization, bounded in
        # (-1, 1), and has derivative one at zero.
        self.step_parameter = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _sample_frobenius(x: torch.Tensor) -> torch.Tensor:
        return x.float().flatten(1).norm(dim=1).view(-1, 1, 1, 1)

    def _trust_project(self, raw: torch.Tensor, candidate: torch.Tensor):
        correction = candidate - raw
        raw_norm = self._sample_frobenius(raw)
        correction_norm = self._sample_frobenius(correction)
        radius = self.trust_radius * raw_norm
        scale = torch.minimum(
            torch.ones_like(correction_norm),
            radius / (correction_norm + self.eps),
        ).to(correction.dtype)
        return correction * scale, correction_norm, radius

    def forward_with_diagnostics(self, x):
        raw = torch.cat((x[0], x[1]), dim=1)
        a, b = self._align(x[0], x[1])
        _, _, energy_r_pre, energy_i_pre = self._reliability_terms(a, b)
        dev_r_pre = torch.log(energy_r_pre + 1e-4).abs()
        dev_i_pre = torch.log(energy_i_pre + 1e-4).abs()
        anomaly = torch.maximum(dev_r_pre, dev_i_pre)
        b, shift = self._select_shift(
            a, b, anomaly.flatten() < self.trigger_tau, return_diagnostics=True
        )
        candidate = self._reliability_output(a, b)
        probability, trigger, energy_r, energy_i = self._reliability_terms(a, b)
        dev_r = torch.log(energy_r + 1e-4).abs()
        dev_i = torch.log(energy_i + 1e-4).abs()
        # At least one plausible stream is enough to permit the canonical branch;
        # simultaneous atypicality causes a smooth fallback to raw concatenation.
        fallback_gate = torch.sigmoid(
            self.fallback_k * (self.fallback_tau - torch.minimum(dev_r, dev_i))
        ).to(raw.dtype)
        correction, correction_norm, radius = self._trust_project(raw, candidate)
        step = self.step_parameter / (1.0 + self.step_parameter.abs())
        output = raw + step.to(raw.dtype) * fallback_gate * correction
        diagnostics = {
            **shift,
            'probability': probability,
            'trigger': trigger,
            'energy_r': energy_r,
            'energy_i': energy_i,
            'fallback_gate': fallback_gate,
            'step': step.detach(),
            'correction_norm': correction_norm,
            'trust_radius_norm': radius,
            'raw': raw,
            'candidate': candidate,
        }
        return output, diagnostics

    def forward(self, x):
        raw = torch.cat((x[0], x[1]), dim=1)
        a, b = self._align(x[0], x[1])
        _, _, energy_r, energy_i = self._reliability_terms(a, b)
        dev_r = torch.log(energy_r + 1e-4).abs()
        dev_i = torch.log(energy_i + 1e-4).abs()
        anomaly = torch.maximum(dev_r, dev_i)
        b = self._select_shift(a, b, anomaly.flatten() < self.trigger_tau)
        candidate = self._reliability_output(a, b)
        _, _, energy_r, energy_i = self._reliability_terms(a, b)
        dev_r = torch.log(energy_r + 1e-4).abs()
        dev_i = torch.log(energy_i + 1e-4).abs()
        fallback_gate = torch.sigmoid(
            self.fallback_k * (self.fallback_tau - torch.minimum(dev_r, dev_i))
        ).to(raw.dtype)
        correction, _, _ = self._trust_project(raw, candidate)
        step = self.step_parameter / (1.0 + self.step_parameter.abs())
        return raw + step.to(raw.dtype) * fallback_gate * correction


class DCSPF(SJPA):
    """Dual-Coordinate Safe Procrustes Fusion with a reliability--coherence guard.

    DCSPF retains two coordinate experts:

    1. the raw RGB--IR concatenation, and
    2. the statistically canonicalized SJPA representation.

    A deterministic sample-level guard selects the canonical expert when either
    the pair is jointly coherent and statistically typical, or one modality has
    a clear reliability advantage. Otherwise, the module uses the raw expert.
    Each expert has an identity-initialized grouped 1x1 adapter, so the module is
    exactly branch preserving at initialization while allowing the detector loss
    to calibrate the two coordinate systems independently.
    """

    def __init__(self, channels: int, groups: int = 32, anchors: int = 8,
                 momentum: float = 0.03, eps: float = 1e-3,
                 reliability: bool = True, reliability_gamma: float = 1.5,
                 trigger_tau: float = 0.6, trigger_k: float = 12.0,
                 max_shift: int = 1, shift_penalty: float = 0.1,
                 score_threshold: float = 0.3, shift_margin: float = 0.02,
                 coherence_tau: float = 0.35, typicality_tau: float = 0.6,
                 dominance_tau: float = 0.8):
        super().__init__(channels, groups, anchors, momentum, eps, reliability,
                         reliability_gamma, trigger_tau, trigger_k, max_shift,
                         shift_penalty, score_threshold, shift_margin)
        if not 0.0 <= coherence_tau <= 1.0:
            raise ValueError('coherence_tau must lie in [0, 1]')
        if typicality_tau < 0.0:
            raise ValueError('typicality_tau must be non-negative')
        if not 0.0 <= dominance_tau <= 1.0:
            raise ValueError('dominance_tau must lie in [0, 1]')
        self.coherence_tau = float(coherence_tau)
        self.typicality_tau = float(typicality_tau)
        self.dominance_tau = float(dominance_tau)
        fused_channels = 2 * self.channels
        # groups=2 prevents the adapter from mixing RGB and IR halves before the
        # detector's normal post-fusion convolution. Both adapters are exact
        # identities at initialization.
        self.raw_adapter = nn.Conv2d(fused_channels, fused_channels, 1, groups=2, bias=True)
        self.canonical_adapter = nn.Conv2d(fused_channels, fused_channels, 1, groups=2, bias=True)
        self._init_identity_adapter(self.raw_adapter)
        self._init_identity_adapter(self.canonical_adapter)

    @staticmethod
    def _init_identity_adapter(adapter: nn.Conv2d):
        with torch.no_grad():
            adapter.weight.zero_()
            channels_per_group = adapter.in_channels // adapter.groups
            outputs_per_group = adapter.out_channels // adapter.groups
            if channels_per_group != outputs_per_group:
                raise ValueError('identity adapter requires equal input/output width per group')
            for group in range(adapter.groups):
                start = group * outputs_per_group
                for channel in range(outputs_per_group):
                    adapter.weight[start + channel, channel, 0, 0] = 1.0
            if adapter.bias is not None:
                adapter.bias.zero_()

    def _normalized_coherence(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return a bounded groupwise RV-style cross-modal coherence in [0, 1]."""
        ap, bp = self.pool(a.float()), self.pool(b.float())
        batch = ap.shape[0]
        tokens = self.anchors * self.anchors
        x = ap.reshape(batch, self.groups, self.group_width, tokens).transpose(-1, -2)
        y = bp.reshape(batch, self.groups, self.group_width, tokens).transpose(-1, -2)
        x = x - x.mean(-2, keepdim=True)
        y = y - y.mean(-2, keepdim=True)
        normalizer = max(1, tokens - 1)
        c_xy = x.transpose(-1, -2) @ y / normalizer
        c_xx = x.transpose(-1, -2) @ x / normalizer
        c_yy = y.transpose(-1, -2) @ y / normalizer
        numerator = torch.linalg.matrix_norm(c_xy, ord='fro').sum(-1)
        energy_x = torch.linalg.matrix_norm(c_xx, ord='fro').sum(-1)
        energy_y = torch.linalg.matrix_norm(c_yy, ord='fro').sum(-1)
        denominator = (energy_x * energy_y + self.eps).sqrt()
        return (numerator / denominator).clamp(0.0, 1.0).view(batch, 1, 1, 1)

    def _guard_terms(self, a: torch.Tensor, b: torch.Tensor):
        probability, trigger, energy_r, energy_i = self._reliability_terms(a, b)
        del trigger
        dev_r = torch.log(energy_r + 1e-4).abs()
        dev_i = torch.log(energy_i + 1e-4).abs()
        coherence = self._normalized_coherence(a, b)
        typicality = torch.minimum(dev_r, dev_i)
        dominance = (probability[:, :1] - probability[:, 1:]).abs()
        coherent_pair = (coherence >= self.coherence_tau) & (typicality <= self.typicality_tau)
        reliable_single = dominance >= self.dominance_tau
        gate = (coherent_pair | reliable_single).to(a.dtype)
        return gate, coherence, typicality, dominance, probability, energy_r, energy_i

    def _canonical_branch(self, rgb: torch.Tensor, ir: torch.Tensor):
        a, b = self._align(rgb, ir)
        _, _, energy_r, energy_i = self._reliability_terms(a, b)
        dev_r = torch.log(energy_r + 1e-4).abs()
        dev_i = torch.log(energy_i + 1e-4).abs()
        anomaly = torch.maximum(dev_r, dev_i)
        b, shift = self._select_shift(
            a, b, anomaly.flatten() < self.trigger_tau, return_diagnostics=True
        )
        canonical = self._reliability_output(a, b)
        return a, b, canonical, shift

    def forward_with_diagnostics(self, x):
        raw = torch.cat((x[0], x[1]), dim=1)
        a, b, canonical, shift = self._canonical_branch(x[0], x[1])
        gate, coherence, typicality, dominance, probability, energy_r, energy_i = self._guard_terms(a, b)
        raw_adapted = self.raw_adapter(raw)
        canonical_adapted = self.canonical_adapter(canonical)
        output = gate * canonical_adapted + (1.0 - gate) * raw_adapted
        return output, {
            **shift,
            'gate': gate,
            'coherence': coherence,
            'typicality': typicality,
            'dominance': dominance,
            'probability': probability,
            'energy_r': energy_r,
            'energy_i': energy_i,
            'raw': raw,
            'canonical': canonical,
            'raw_adapted': raw_adapted,
            'canonical_adapted': canonical_adapted,
        }

    def forward(self, x):
        raw = torch.cat((x[0], x[1]), dim=1)
        a, b, canonical, _ = self._canonical_branch(x[0], x[1])
        gate, *_ = self._guard_terms(a, b)
        return gate * self.canonical_adapter(canonical) + (1.0 - gate) * self.raw_adapter(raw)


class MPCRF(nn.Module):
    """Moment-Preserving Canonical Residual Fusion.

    The module fixes three mathematical defects of direct whitening fusion:

    1. statistics are estimated from deterministic point samples rather than
       averaged anchor cells, so the estimated covariance has the same physical
       scale as the full-resolution feature map;
    2. statistics are computed independently for every sample, eliminating
       batch-composition dependence and train/evaluation transform mismatch;
    3. canonicalization is never used as a replacement representation. It only
       drives a zero-initialized, norm-bounded residual around raw concatenation.

    For each sample and channel group, regularized cross-covariance provides
    the sign-safe orthogonal Procrustes factor Q=UV^T. The bidirectional
    cycle-consistent candidates

        R_c = (R-mu_r) Sigma_r^{-1/2} Q Sigma_r^{1/2} + mu_r,
        I_c = (I-mu_i) Sigma_i^{-1/2} Q^T Sigma_i^{1/2} + mu_i

    preserve each modality's regularized first and second moments. Learned
    grouped 1x1 residual maps are initialized to zero, so the complete module is
    exactly direct concatenation at initialization. A samplewise trust projection
    guarantees

        ||MPCRF(R,I) - concat(R,I)||_F
            <= trust_radius * ||concat(R,I)||_F.
    """

    def __init__(self, channels: int, groups: int = 32, stat_grid: int = 16,
                 eps: float = 1e-3, coherence_tau: float = 0.30,
                 gate_k: float = 10.0, trust_radius: float = 0.25):
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups != 0:
            raise ValueError(
                f'channels={channels} must be positive and divisible by groups={groups}'
            )
        if stat_grid < 2:
            raise ValueError('stat_grid must be at least 2')
        if eps <= 0:
            raise ValueError('eps must be positive')
        if not 0.0 <= coherence_tau <= 1.0:
            raise ValueError('coherence_tau must lie in [0, 1]')
        if gate_k <= 0:
            raise ValueError('gate_k must be positive')
        if trust_radius <= 0:
            raise ValueError('trust_radius must be positive')

        self.channels = int(channels)
        self.groups = int(groups)
        self.group_width = self.channels // self.groups
        self.stat_grid = int(stat_grid)
        self.eps = float(eps)
        self.coherence_tau = float(coherence_tau)
        self.gate_k = float(gate_k)
        self.trust_radius = float(trust_radius)

        # The adapters act only on canonical residuals and are zero initialized.
        # Consequently the complete layer is exactly raw concatenation before
        # the detector has learned that a correction is useful.
        # Direct zero Parameters avoid consuming the global RNG. Constructing
        # Conv2d and zeroing it afterwards changes every downstream detector
        # initialization, making a same-seed comparison with Concat unfair.
        kernel_shape = (self.channels, self.group_width, 1, 1)
        self.rgb_residual_weight = nn.Parameter(torch.zeros(kernel_shape))
        self.ir_residual_weight = nn.Parameter(torch.zeros(kernel_shape))

    @staticmethod
    def _autocast_disabled(device: torch.device):
        if device.type in {'cpu', 'cuda', 'xpu'}:
            return torch.autocast(device_type=device.type, enabled=False)
        return contextlib.nullcontext()

    def _point_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Select an approximately uniform grid without averaging feature cells."""
        h, w = x.shape[-2:]
        gh, gw = min(self.stat_grid, h), min(self.stat_grid, w)
        # Cell-centre indices. Unlike average pooling, this operation does not
        # divide feature variance by the number of cells inside an anchor bin.
        iy = ((torch.arange(gh, device=x.device) * h + h // 2) // gh).clamp_max(h - 1)
        ix = ((torch.arange(gw, device=x.device) * w + w // 2) // gw).clamp_max(w - 1)
        return x.index_select(-2, iy).index_select(-1, ix)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return x.reshape(b, self.groups, self.group_width, h * w).permute(0, 1, 3, 2)

    def _canonical_candidates(self, rgb: torch.Tensor, ir: torch.Tensor):
        """Return moment-preserving candidates and bounded group coherence."""
        b, _, h, w = rgb.shape
        with self._autocast_disabled(rgb.device):
            rs = self._tokens(self._point_sample(rgb.detach().float()))
            is_ = self._tokens(self._point_sample(ir.detach().float()))
            mu_r = rs.mean(-2, keepdim=True)
            mu_i = is_.mean(-2, keepdim=True)
            rc = rs - mu_r
            ic = is_ - mu_i
            n = max(1, rs.shape[-2] - 1)
            cov_r = rc.transpose(-1, -2) @ rc / n
            cov_i = ic.transpose(-1, -2) @ ic / n

            er, vr = torch.linalg.eigh(cov_r)
            ei, vi = torch.linalg.eigh(cov_i)
            er = er.clamp_min(self.eps)
            ei = ei.clamp_min(self.eps)
            sqrt_r = vr @ torch.diag_embed(er.sqrt()) @ vr.transpose(-1, -2)
            sqrt_i = vi @ torch.diag_embed(ei.sqrt()) @ vi.transpose(-1, -2)
            inv_r = vr @ torch.diag_embed(er.rsqrt()) @ vr.transpose(-1, -2)
            inv_i = vi @ torch.diag_embed(ei.rsqrt()) @ vi.transpose(-1, -2)

            rw = rc @ inv_r
            iw = ic @ inv_i
            cross = rw.transpose(-1, -2) @ iw / n
            u, singular, vh = torch.linalg.svd(cross, full_matrices=False)
            # Use only the orthogonal polar/Procrustes factor Q=UV^T. Unlike
            # the individual singular bases U and V, Q is invariant to matched
            # SVD sign flips and rotations inside repeated-singular subspaces.
            # Q solves min_{Q^T Q=I} ||rw Q - iw||_F^2; Q^T is the inverse
            # Procrustes map. This gives a stable, cycle-consistent pair.
            q = u @ vh
            coherence = singular.clamp(0.0, 1.0).mean(-1)

            rgb_live = self._tokens(rgb.float())
            ir_live = self._tokens(ir.float())
            rgb_candidate = ((rgb_live - mu_r) @ inv_r @ q @ sqrt_r) + mu_r
            ir_candidate = (
                (ir_live - mu_i) @ inv_i @ q.transpose(-1, -2) @ sqrt_i
            ) + mu_i

            rgb_candidate = rgb_candidate.permute(0, 1, 3, 2).reshape(
                b, self.channels, h, w
            )
            ir_candidate = ir_candidate.permute(0, 1, 3, 2).reshape(
                b, self.channels, h, w
            )
        return (
            rgb_candidate.to(rgb.dtype),
            ir_candidate.to(ir.dtype),
            coherence,
            singular,
        )

    @staticmethod
    def _sample_norm(x: torch.Tensor) -> torch.Tensor:
        return x.float().flatten(1).norm(dim=1).view(-1, 1, 1, 1)

    def _forward_impl(self, x):
        rgb, ir = x
        raw = torch.cat((rgb, ir), dim=1)
        rgb_candidate, ir_candidate, coherence, singular = self._canonical_candidates(rgb, ir)

        # Smooth, bounded group confidence. The statistical decision is detached
        # from detector gradients, while the residual adapters remain fully
        # trainable through the live candidate differences.
        gate = torch.sigmoid(
            self.gate_k * (coherence - self.coherence_tau)
        ).detach()
        gate_map = gate.unsqueeze(-1).unsqueeze(-1).repeat_interleave(
            self.group_width, dim=1
        ).to(rgb.dtype)

        delta_rgb = F.conv2d(
            gate_map * (rgb_candidate - rgb),
            self.rgb_residual_weight,
            groups=self.groups,
        )
        delta_ir = F.conv2d(
            gate_map * (ir_candidate - ir),
            self.ir_residual_weight,
            groups=self.groups,
        )
        correction = torch.cat((delta_rgb, delta_ir), dim=1)

        raw_norm = self._sample_norm(raw)
        correction_norm = self._sample_norm(correction)
        radius = self.trust_radius * raw_norm
        scale = torch.minimum(
            torch.ones_like(correction_norm),
            radius / (correction_norm + self.eps),
        ).to(correction.dtype)
        output = raw + scale * correction
        diagnostics = {
            'raw': raw,
            'rgb_candidate': rgb_candidate,
            'ir_candidate': ir_candidate,
            'coherence': coherence,
            'singular_values': singular,
            'gate': gate,
            'correction': correction,
            'correction_norm': correction_norm,
            'trust_radius_norm': radius,
            'trust_scale': scale,
        }
        return output, diagnostics

    def forward_with_diagnostics(self, x):
        return self._forward_impl(x)

    def forward(self, x):
        return self._forward_impl(x)[0]
