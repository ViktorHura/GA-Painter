import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils
import torch
import torch.nn.functional as F
from main import SiameseNetworkDataset, imshow, SiameseNetwork


def run():

    net = SiameseNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load("models/siamese.pth"))
    net.eval()
    net = net.to(device)

    img1 = Image.open("data/mnist/testing/3/18.png")
    img1 = img1.convert("L")
    img1 = img1.resize((100, 100))
    img1 = torch.tensor(data=np.array(img1, dtype=np.float32), )

    img1.unsqueeze_(0)
    img1 = img1.repeat(1, 1, 1)

    img1 = img1.to(device)

    img2 = Image.open("data/mnist/testing/polygon/85.png")
    img2 = img2.convert("L")
    img2 = img2.resize((100, 100))
    img2 = torch.tensor(data=np.array(img2, dtype=np.float32), )

    img2.unsqueeze_(0)
    img2 = img2.repeat(1, 1, 1)

    img2 = img2.to(device)



    with torch.no_grad():
        output1, output2 = net(img1, img2)
        euclidean_distance = F.pairwise_distance(output1, output2)
        fit = float(euclidean_distance.item())
        print(fit)


if __name__ == '__main__':
    run()