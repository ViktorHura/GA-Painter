"""
Generalized MBConv class for both MBConv and MBInvertedConv. As well as MBConv[N] and MBInvertedConv[N].
"""

import torch
import torch.nn as nn
from ENetLayer import ENetLayer
from SEBlock import SEBlock
from util import *


class MBConvN(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor,
                 kernel_size=3, stride=1, ratio=24, probability=0):
        super(MBConvN, self).__init__()

        # Making sure that the dimensions don't change
        _padding = (kernel_size - 1) // 2
        _expanded_channels = expansion_factor * in_channels

        # A check that's used to determine whether to add the residual
        self.skip_connection = in_channels == out_channels and stride == 1

        self.expand_pointwise = nn.Identity()
        if expansion_factor != 1:
            self.expand_pointwise = ENetLayer(in_channels, _expanded_channels, kernel_size=1)

        # depthwise seperable convolution; achieved by providing the groups argument as the number of input channels
        self.depthwise = ENetLayer(_expanded_channels, _expanded_channels, kernel_size=kernel_size,
                                   stride=stride, padding=_padding, groups=_expanded_channels)

        # squeeze-and-excitation layer
        self.se = SEBlock(_expanded_channels, ratio=ratio)

        # pointwise convolution
        self.pw = ENetLayer(_expanded_channels, out_channels, kernel_size=1, act=False)

        # Unclear function; supposedly to prevent overfitting (like dropout (mentioned in the paper))
        self.dropsample = DropSample(probability)

    def forward(self, x):
        res = x

        x = self.expand_pointwise(x)
        x = self.depthwise(x)

        x = self.se(x)
        x = self.pw(x)

        if self.skip_connection:
            x = self.dropsample(x)
            x = x + res

        return x


class MBConv1(MBConvN):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, ratio=24, probability=0):
        # Expansion factor of 1
        super(MBConv1, self).__init__(in_channels, out_channels, 1, kernel_size, stride, ratio, probability)


class MBConv6(MBConvN):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, ratio=24, probability=0):
        # Expansion factor of 6
        super(MBConv6, self).__init__(in_channels, out_channels, 6, kernel_size, stride, ratio, probability)
