import torch
from torch import nn


class ENetLayer(nn.Module):
    """
    This class describes the efficient net layer
    including the convolution, batch normalization, relu and max pooling
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0,
                 groups=1, bias=False, bn=True, act=True):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the kernel
        :param stride: stride of the convolution
        :param padding: padding of the convolution
        :param groups: number of groups for grouped convolution
        :param bias: whether to use bias
        :param bn: whether to use batch normalization
        :param act: whether to use activation function
        """
        super(ENetLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        # Using SiLU as activation function because it is more stable than ReLU, it's shown to match or exceed ReLU
        self.act = nn.SiLU() if act else nn.Identity()

        # self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        # self.pool = nn.MaxPool2d(2, 2) if act else None
        self.pool = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x
