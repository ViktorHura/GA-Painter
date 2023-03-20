import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from EfficientNet import EfficientNet, EfficientNetB7, EfficientNetB0
from drafts.net import Net
from torch import nn

# Thread count, will use half of it to load the data
THREADS = 4
ROOT = '../data/imagenette2'
# PATH = '../models/b0c_bs2_epoch24_car_data_224.pth'
PATH = '../models/b0c_bs_2_epoch_256_car_data_224.pth'
TRAIN = False
EPOCHS = 256


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 2

classes = [_ for _ in os.listdir(ROOT + '/train')]
net = EfficientNetB0(out_size=len(classes))


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(trainloader, testloader, epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    start_time = time.perf_counter()
    print('Training start:', time.asctime(time.localtime()))
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print('Epoch:', epoch)
        torch.save(net.state_dict(), PATH)
        if epoch % 64 == 0:
            torch.save(net.state_dict(), PATH[:-4] + '_' + str(epoch) + PATH[-4:])

    print('Finished Training:', time.asctime(time.localtime()))
    end_time = time.perf_counter()
    print('Training time:', end_time - start_time)


def test(testloader):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    all_correct = 0
    all_preds = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs.data, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                    all_correct += 1
                total_pred[classes[label]] += 1
                all_preds += 1

    print(f"General accuracy for this classifier: {(100 * float(all_correct) / float(all_preds)):.1f}%")
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname} is {accuracy:.1f}%")


def main():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.ImageFolder(root=ROOT + '/train', transform=transform)
    _trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=THREADS//2)

    testset = torchvision.datasets.ImageFolder(root=ROOT + '/test', transform=transform)
    _testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=THREADS//2)

    return _trainloader, _testloader


def train_and_save(trainloader, testloader, PATH, epochs=2):
    # Training
    train(trainloader, testloader, epochs)
    # Saving the model
    torch.save(net.state_dict(), PATH)


if __name__ == "__main__":
    print('Program starting:', time.asctime(time.localtime()))
    net.to(device)
    trainloader, testloader = main()

    if TRAIN:
        train_and_save(trainloader, testloader, PATH, epochs=EPOCHS)

    # Loading the model
    net.load_state_dict(torch.load(PATH))

    # Test over 10000 images
    test(testloader)
