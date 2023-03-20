"""
Utility modules
"""

import torch
from torch import nn

BASE_WIDTH_ARRAY = [(32, 16), (16, 24), (24, 40), (40, 80), (80, 112), (112, 192), (192, 320), (320, 1280)]
BASE_DEPTH_ARRAY = [1, 2, 2, 3, 3, 4, 1]


def scale_width(width, width_scale):
    width = width * width_scale
    new_width = int(width + 4)
    new_width = max(8, new_width - (new_width % 8))
    if new_width < 0.9 * width:
        new_width += 8
    return new_width

# def scale_width(w, w_factor):
#     """Scales width given a scale factor"""
#     w *= w_factor
#     new_w = (int(w+4) // 8) * 8
#     new_w = max(8, new_w)
#     if new_w < 0.9*w:
#      new_w += 8
#     return int(new_w)

# This function creates a sequential layer which is called a stage in the research paper
def create_sequential_layer(in_channels, out_channels, layer_cnt, layer_type,
                            kernel_size=3, stride=1, ratio=24, probability=0):

    layers = [layer_type(in_channels, out_channels, kernel_size=kernel_size, stride=stride, ratio=ratio,
                         probability=probability)]
    layers += [layer_type(out_channels, out_channels, kernel_size=kernel_size, ratio=ratio, probability=probability) for
               _ in range(layer_cnt - 1)]
    return nn.Sequential(*layers)


class DropSample(nn.Module):
    """
    Drop samples in x with probability p
    """

    def __init__(self, prob=0):
        super(DropSample, self).__init__()
        self.prob = prob

    def forward(self, x):
        if not (self.prob and self.training):
            return x

        batch_size = len(x)

        if torch.cuda.is_available():
            random = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
        else:
            random = torch.FloatTensor(batch_size, 1, 1, 1).uniform_()

        mask = self.prob < random

        # Scale x
        x = x.div(1 - self.prob)
        # Apply the mask, dropping some samples
        x = x * mask
        return x
