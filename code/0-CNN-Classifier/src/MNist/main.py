import time
import torch
import cv2
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageShow
from MNistNet import MNistNet


EPOCHS = 8
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10
RANDOM_SEED = 1
PATH = "./models/mnist_net.pth"
TRAIN = True

# Disable non deterministic cuDNN algorithms
torch.backends.cudnn.enabled = False
torch.manual_seed(RANDOM_SEED)  # Set random seed for reproducibility

# Init device obj
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loss criterion
loss_criterion = F.cross_entropy


# Show images
def show_images(data, targets, cols=3, rows=3) -> plt.Figure:
    fig = plt.figure()
    for i in range(cols * rows):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title("Label: {}".format(targets[i]))
        plt.xticks([])
        plt.yticks([])
    return fig


# Preparing data
def load_data() -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    # MNIST dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        # MNist mean and std to normalize, easier than calculating it ourselves
    ])

    train_set = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    expanded_train_set = torchvision.datasets.ImageFolder(root='data/expanded/train', transform=transform)

    train_set = torch.utils.data.ConcatDataset([train_set, expanded_train_set])

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE_TRAIN)

    test_set = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )

    expanded_test_set = torchvision.datasets.ImageFolder(root='data/expanded/test', transform=transform)

    test_set = torch.utils.data.ConcatDataset([test_set, expanded_test_set])

    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=BATCH_SIZE_TEST)

    return train_loader, test_loader


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, cur_epoch: int) -> None:
    model.train()    # Set model to training mode (self.train = True)
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(f"Epoch: {cur_epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}")
            print(f"Progress: {batch_idx * len(data)}/{len(train_loader.dataset)}")


def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> None:
    model.eval()    # Set model to evaluation mode (self.train = False)

    all_correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            # Sum batch loss
            test_loss += loss_criterion(outputs, target, reduction="sum").item()

            predictions = outputs.argmax(1, keepdim=True)
            all_correct += predictions.eq(target.view_as(predictions)).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    print("General accuracy: {:.2f}%\tTest Loss: {:.4f}\tAccuracy: {}/{}".format(
        100. * all_correct / len(test_loader.dataset), test_loss, all_correct, len(test_loader.dataset)))


def main():
    train_loader, test_loader = load_data()
    model = MNistNet().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.7, step_size=1)

    if TRAIN:
        start_time = time.perf_counter()
        print('Training start:', time.asctime(time.localtime()))

        for cur_epoch in range(EPOCHS):
            train(model, train_loader, optimizer, cur_epoch)
            test(model, test_loader)
            torch.save(model.state_dict(), PATH)
            if cur_epoch % 64 == 0:
                torch.save(model.state_dict(), PATH[:-4] + '_' + str(cur_epoch) + PATH[-4:])
            scheduler.step()

        print('Finished Training:', time.asctime(time.localtime()))
        end_time = time.perf_counter()
        print('Training time:', end_time - start_time)
    else:
        model.load_state_dict(torch.load(PATH))
        test(model, test_loader)


def predict_image(image_path: str, target=3) -> None:
    # Load model
    model = MNistNet(grad_cam_layer="conv2")
    PATH = "./models/mnist_net.pth"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # Load image
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize((28, 28))
    img = torch.tensor(data=np.array(img))
    img = img.reshape(1, 1, 28, 28)
    img = img.float()

    # Predict with confidence score, model output is softmax
    output = model(img)
    heatmap = model.get_heatmap(img, output, target)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

    heatmap = cv2.resize(heatmap.numpy(), (512, 512), interpolation=cv2.INTER_LINEAR)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    gcam3 = cv2.addWeighted(heatmap.astype(np.float), 0.6, img.astype(np.float), 0.4, 0)
    cv2.imshow('image', np.uint8(gcam3))

    output = F.log_softmax(output, dim=1)
    # Predict from output
    _, predicted = torch.max(output.data, 1)
    print(f"Predicted: {predicted.item()}")


if __name__ == "__main__":
    main()




