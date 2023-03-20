import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


class MNistNet(nn.Module):
    def __init__(self, grad_cam_layer="conv1"):
        super(MNistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 11, kernel_size=5)
        self.conv2 = nn.Conv2d(11, 22, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(352, 50)
        self.fc2 = nn.Linear(50, 11)

        # Gradcam hooks
        self.grad_cam_layer = grad_cam_layer
        self.gradients = None

    def get_heatmap(self, input, output, target):
        # Get gradient of the desired class
        output[:, target].backward()
        gradients = self.get_activation_gradient()
        pooled_grads = torch.mean(gradients, dim=[0, 2, 3])

        # Get the activations
        activations = self.get_activations(input).detach()

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap, 0)

        heatmap /= torch.max(heatmap)
        heatmap = heatmap.squeeze()

        #plt.matshow(heatmap)
        #plt.show()

        return heatmap

    def activation_hook(self, grad):
        self.gradients = grad

    def get_activation_gradient(self):
        return self.gradients

    def get_activations(self, x):
        activations = self.conv1(x)
        if self.grad_cam_layer == "conv1":
            return activations
        activations = F.max_pool2d(activations, 2)
        activations = F.relu(activations)

        activations = self.conv2(activations)
        return activations

    def forward(self, x):
        # Sequentially pass input x through each layer, making it easier to dissect
        x = self.conv1(x)
        if self.grad_cam_layer == "conv1":
            h = x.register_hook(self.activation_hook)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = self.conv2(x)
        if self.grad_cam_layer == "conv2":
            h = x.register_hook(self.activation_hook)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(-1, 352)
        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
