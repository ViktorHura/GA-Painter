"""
EfficientNet implementation.
"""

import math
from MBConv import *
from util import *


class EfficientNet(nn.Module):
    def __init__(self, width_scale=1, depth_scale=1, out_size=1000):
        super(EfficientNet, self).__init__()

        SCALED_DEPTH_ARRAY = [math.ceil(depth_scale * depth) for depth in BASE_DEPTH_ARRAY]
        SCALED_WIDTH_ARRAY = []
        for width in BASE_WIDTH_ARRAY:
            SCALED_WIDTH_ARRAY.append((scale_width(width[0], width_scale), scale_width(width[1], width_scale)))

        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        probabilities = [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171]

        # 3x3 convolution
        self.stem = ENetLayer(3, SCALED_WIDTH_ARRAY[0][0], stride=2, padding=1)

        s_layers = []
        for i in range(len(SCALED_WIDTH_ARRAY) - 1):
            if i == 0:
                layer_type = MBConv1
                ratio = 4
            else:
                layer_type = MBConv6
                ratio = 24
            s_layer = create_sequential_layer(*SCALED_WIDTH_ARRAY[i], SCALED_DEPTH_ARRAY[i],
                                              layer_type, kernel_size=kernel_sizes[i], stride=strides[i],
                                              ratio=ratio, probability=probabilities[i])
            s_layers.append(s_layer)

        self.s_layer = nn.Sequential(*s_layers)

        # Pointwise convolution -> pooling -> fully connected layer at last
        self.pre_head = ENetLayer(*SCALED_WIDTH_ARRAY[-1], kernel_size=1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(SCALED_WIDTH_ARRAY[-1][1], out_size))

    def forward(self, x):
        x = self.stem(x)
        x = self.s_layer(x)
        x = self.pre_head(x)
        x = self.head(x)
        return x


class EfficientNetB0(EfficientNet):
    def __init__(self, out_size=1000):
        width_scale, depth_scale = 1, 1
        super(EfficientNetB0, self).__init__(width_scale, depth_scale, out_size)


class EfficientNetB1(EfficientNet):
    def __init__(self, out_size=1000):
        width_scale, depth_scale = 1, 1.1
        super(EfficientNetB1, self).__init__(width_scale, depth_scale, out_size)


class EfficientNetB2(EfficientNet):
    def __init__(self, out_size=1000):
        width_scale, depth_scale = 1.1, 1.2
        super(EfficientNetB2, self).__init__(width_scale, depth_scale, out_size)


class EfficientNetB3(EfficientNet):
    def __init__(self, out_size=1000):
        width_scale, depth_scale = 1.2, 1.4
        super(EfficientNetB3, self).__init__(width_scale, depth_scale, out_size)


class EfficientNetB4(EfficientNet):
    def __init__(self, out_size=1000):
        width_scale, depth_scale = 1.4, 1.8
        super(EfficientNetB4, self).__init__(width_scale, depth_scale, out_size)


class EfficientNetB5(EfficientNet):
    def __init__(self, out_size=1000):
        width_scale, depth_scale = 1.6, 2.2
        super(EfficientNetB5, self).__init__(width_scale, depth_scale, out_size)


class EfficientNetB6(EfficientNet):
    def __init__(self, out_size=1000):
        width_scale, depth_scale = 1.8, 2.6
        super(EfficientNetB6, self).__init__(width_scale, depth_scale, out_size)


class EfficientNetB7(EfficientNet):
    def __init__(self, out_size=1000):
        width_scale, depth_scale = 2.0, 3.1
        super(EfficientNetB7, self).__init__(width_scale, depth_scale, out_size)
