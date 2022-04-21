import argparse

import torch
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from torchvision.models.vgg import vgg16
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import transforms
from tqdm import tqdm


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

    # model_checkpoint
    parser.add_argument('--save_path', default='./', help='Location to save model')

    # Inference model

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    for i, (x_test, y_test) in enumerate(tqdm(dataloader)):
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        # Calculate accuracy
        output = model(x_test)
        _, y_pred = torch.max(output, dim=1)
        correct += (y_pred == y_test).float().sum()
    model.train()
    return 100 * correct / len(dataloader.dataset)


def train(args, device):
    # Prepare model
    model = vgg16(pretrained=True)
    input_lastLayer = model.classifier[6].in_features
    # Change the last layer into nn.Linear
    model.classifier[6] = nn.Linear(input_lastLayer, 10)
    model = model.to(device)

    # Prepare loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Prepare dataloader
    train_dataset = get_train_dataset()
    eval_dataset = get_eval_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)

    accuracy = evaluate(model, eval_dataloader, device)
    print(f'Initial accuracy on test: {accuracy}')

    # Train the model
    for epoch in tqdm(range(args.epochs)):
        for i, (x_train, y_train) in enumerate(tqdm(train_dataloader)):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_hat = model(x_train)
            loss = criterion(y_hat, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 200 == 0:
                print(f'Loss: {loss}')
        # Evaluate model after each epoch
        test_acc = evaluate(model, eval_dataloader, device)
        print(f'Test acc: {test_acc}')

    # Save model
    torch.save(model.state_dict(), f'{args.save_path}model.pt')


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

    # Train the model
    train(args, device)

    # Evaluate the model


    # Prune the model,

    # Fine-tune the pruned model.

    # Re-wro