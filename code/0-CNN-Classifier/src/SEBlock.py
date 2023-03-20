import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, ratio=24):
        super(SEBlock, self).__init__()

        # 2d avg pooling over input, 1x1 output, #features out = #planes in
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation layer
        self.excite = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1),
                                    nn.SiLU(),
                                    nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1),
                                    nn.Sigmoid())

    def forward(self, x):
        # Squeeze
        squeezed = self.squeeze(x)
        # Excitation
        excited = self.excite(squeezed)
        # Scale
        scaled = x * excited
        return scaled
        # y = self.squeeze(x)
        # y = self.excite(y)
        # return x * y


