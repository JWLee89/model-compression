import argparse

import torch
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from torchvision.models.vgg import vgg16
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import transforms


def get_train_dataset(composed_transform=transforms.Compose([
    transforms.ToTensor(),
])):
    """
    For now, we will be working with the Cifar-10 Dataset.
    For more complex datasets, feel free to write your own custom
    implementation.
    """
    return CIFAR10(train=True, root='./', download=True, transform=composed_transform)


def get_eval_dataset(composed_transform=transforms.Compose([
    transforms.ToTensor()
])):
    return CIFAR10(train=False, root='./', download=True, transform=composed_transform)


def get_args() -> argparse.Namespace:
    """
    Get basic arguments for training the dataset.
    """
    parser = argparse.ArgumentParser()

    # Learning-related hyper-parameter
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum hyperparameter')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay hyperparameter')
    parser.add_argument('--gpu', type=int, default=0, help='Default GPU id')

    # Iteration-related hyperparameters
    parser.add_argument('--epochs', default=5, help='The number of epochs to train for')
    parser.add_argument('--batch_size', default=8, help='The size of the mini-batch')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    available_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    gpu = args.gpu

    print(f'available devices: {available_devices}')

    if gpu != -1:
        if len(available_devices) < gpu - 1:
            raise RuntimeError(f'Cuda device: {gpu} not available')
        else:
            print(f'Using gpu device: {gpu}')
            device = available_devices[gpu]
    else:
        device = torch.device('cpu')
    # Prepare model
    model = vgg16(pretrained=True).to(device)
    input_lastLayer = model.classifier[6].in_features

    # Change the last layer into nn.Linear
    model.classifier[6] = nn.Linear(input_lastLayer, 10)

    # Prepare loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train_dataset = get_train_dataset()
    train_dataset.train_data.to(device)
    train_dataset.train_labels.to(device)

    eval_dataset = get_eval_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Train the model
    for epoch in range(args.epochs):
        for i, (x_train, y_train) in enumerate(train_dataloader):
            y_hat = model(x_train)
            loss = criterion(y_hat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if i % 200 == 0:
            print(f'Loss: {loss}')

    # Evaluate the model

    # Prune the model,

    # Fine-tune the pruned model.

    # Re-wro