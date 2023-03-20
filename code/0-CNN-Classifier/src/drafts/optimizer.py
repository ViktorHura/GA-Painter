"""
Defining a loss function and optimizer
"""
import torch.optim as optim
import torch.nn as nn
from net import net

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.001)

