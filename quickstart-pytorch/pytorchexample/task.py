"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# =========================================================================================
# MÔ HÌNH VÀ CÁC HÀM CHO Du lieu MNIST 
# =========================================================================================

class Net(nn.Module):
    """Mo hinh CNN gồm: 2 Conv, 2 FC layers."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([
    ToTensor(), 
    Normalize((0.1307,), (0.3081,))
])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("mnist", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

# =========================================================================================
# MÔ HÌNH VÀ CÁC HÀM CHO Du lieu CIFAR-10
# =========================================================================================

# class Net(nn.Module):
#     """Mô hình CNN cho CIFAR-10 gồm: 2 Conv, 2 Pool, và 2 FC layers."""

#     def __init__(self):
#         super(Net, self).__init__()
#         # Ảnh CIFAR-10 có 3 kênh màu (RGB) thay vì 1 kênh như MNIST, kích thước 32x32
#         self.conv1 = nn.Conv2d(3, 6, 5)       # Output: 6 x 28 x 28
#         self.pool = nn.MaxPool2d(2, 2)        # Output: 6 x 14 x 14
#         self.conv2 = nn.Conv2d(6, 16, 5)      # Output: 16 x 10 x 10
#         # self.pool lần 2 sẽ cho Output: 16 x 5 x 5
        
#         # Vì kích thước ảnh đầu vào là 32x32, sau khi qua 2 lớp pool (giảm một nửa mỗi lần)
#         # kích thước Tensor trước khi duỗi (flatten) sẽ là 16 channels * 5 * 5
#         self.fc1 = nn.Linear(16 * 5 * 5, 120) 
#         self.fc2 = nn.Linear(120, 10)         # Giữ nguyên 2 lớp FC

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# fds = None  # Cache FederatedDataset

# # Chuẩn hóa (Normalize) cho 3 kênh màu RGB của CIFAR-10
# pytorch_transforms = Compose([
#     ToTensor(), 
#     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])


# def apply_transforms(batch):
#     """Apply transforms to the partition from FederatedDataset for CIFAR-10."""
#     # Lưu ý: Dataset CIFAR-10 trên HuggingFace lưu ảnh ở key "img" thay vì "image"
#     batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#     return batch


# def load_data(partition_id: int, num_partitions: int, batch_size: int):
#     """Load partition CIFAR-10 data."""
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds = FederatedDataset(
#             dataset="cifar10",  # Đổi sang CIFAR-10
#             partitioners={"train": partitioner},
#         )
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     # Construct dataloaders
#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(
#         partition_train_test["train"], batch_size=batch_size, shuffle=True
#     )
#     testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
#     return trainloader, testloader


# def load_centralized_dataset():
#     """Load test set and return dataloader."""
#     test_dataset = load_dataset("cifar10", split="test")
#     dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
#     return DataLoader(dataset, batch_size=128)


# def train(net, trainloader, epochs, lr, device):
#     """Train the model on the training set."""
#     net.to(device)
#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
#     net.train()
#     running_loss = 0.0
#     for _ in range(epochs):
#         for batch in trainloader:
#             # Truy cập key "img" thay vì "image"
#             images = batch["img"].to(device)
#             labels = batch["label"].to(device)
#             optimizer.zero_grad()
#             loss = criterion(net(images), labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#     avg_trainloss = running_loss / len(trainloader)
#     return avg_trainloss


# def test(net, testloader, device):
#     """Validate the model on the test set."""
#     net.to(device)
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in testloader:
#             # Truy cập key "img" thay vì "image"
#             images = batch["img"].to(device)
#             labels = batch["label"].to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     loss = loss / len(testloader)
#     return loss, accuracy