from typing import Optional

from einops import rearrange
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor


# should normalization be affine?
class ConvNeXtBlock(nn.Module):
    def __init__(self, num_channels, scale_down=1):
        super().__init__()

        assert isinstance(num_channels, int)
        assert isinstance(scale_down, int)

        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels,
                               kernel_size=7,
                               padding=3,
                               groups=num_channels,
                               bias=False)
        
        # instance norm?
        self.norm = nn.GroupNorm(num_groups=num_channels,
                                 num_channels=num_channels,
                                 affine=True)
        
        self.conv2 = nn.Conv2d(in_channels=num_channels,
                               out_channels=4*num_channels,
                               kernel_size=1,
                               bias=False)
        
        # GeGLU?
        self.activation = nn.GELU()

        self.conv3 = nn.Conv2d(in_channels=4*num_channels,
                               out_channels=num_channels // scale_down,
                               kernel_size=1,
                               bias=False)
        
        if scale_down > 1:
            self.skip_conv = nn.Conv2d(in_channels=num_channels,
                                       out_channels=num_channels // scale_down,
                                       kernel_size=1,
                                       bias=False)
        else:
            self.skip_conv = ReturnInput()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip_conv(x)

        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)

        # 1x1 convolution for different input and output channels?
        x += identity
        return x


# make this more similar to a transformer stem?
class InputStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # should this be depthwise?
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=7,
                              padding=3,
                              bias=False)

        # is norm needed since no spatial resolution change?
        self.norm = nn.GroupNorm(num_groups=out_channels,
                                 num_channels=out_channels,
                                 affine=True)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


# make this more similar to a transformer stem?
class OutputStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        
        # is norm needed since no spatial resolution change?
        self.norm = nn.GroupNorm(num_groups=in_channels,
                                 num_channels=in_channels,
                                 affine=True)
        
        # should this be depthwise?
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=7,
                              padding=3,
                              bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)

        self.norm = nn.GroupNorm(num_groups=in_channels,
                                 num_channels=in_channels,
                                 affine=True)
        
        # should this be depthwise?
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)

        # upsample vs. interpolate?
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='nearest')
        
        # should norm be before or after upsample?
        self.norm = nn.GroupNorm(num_groups=in_channels,
                                 num_channels=in_channels,
                                 affine=True)
        
        # should this be depthwise?
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=7,
                              padding=3,
                              bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.norm(x)
        x = self.conv(x)
        return x


class Addition(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y


class Concatenation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.cat((x, y), dim=1)


class ReturnInput(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        return x
