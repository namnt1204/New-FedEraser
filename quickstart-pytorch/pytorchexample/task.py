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

# =========================================================================================
# MÔ HÌNH VÀ CÁC HÀM CHO Du lieu Adult 
# =========================================================================================

# class Net(nn.Module):
#     """Mô hình Neural Network cho tập dữ liệu Adult gồm: 2 FC layers."""
# 
#     def __init__(self, input_dim=104): 
#         # input_dim phụ thuộc vào số lượng feature sau khi One-hot encoding.
#         # Thường bộ dữ liệu Adult sau khi xử lý sẽ có khoảng 104-108 features.
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 2) # Phân loại nhị phân (>50K hoặc <=50K)
# 
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
# 
# fds = None  
# 
# def apply_transforms(batch):
#     """Chuyển đổi dữ liệu bảng thành PyTorch Tensors."""
#     # Giả định dữ liệu đã được số hóa và lưu ở key 'features'
#     batch["features"] = [torch.tensor(feat, dtype=torch.float32) for feat in batch["features"]]
#     return batch
# 
# def load_data(partition_id: int, num_partitions: int, batch_size: int):
#     """Load partition Adult data."""
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds= FederatedDataset(
#             dataset="adult", # Cần đảm bảo dataset name khớp với HuggingFace hoặc local
#             partitioners={"train": partitioner},
#         )
#     partition = fds.load_partition(partition_id)
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     
#     trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
#     return trainloader, testloader
# 
# def train(net, trainloader, epochs, lr, device):
#     net.to(device)
#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
#     net.train()
#     running_loss = 0.0
#     for _ in range(epochs):
#         for batch in trainloader:
#             features = batch["features"].to(device)
#             labels = batch["label"].to(device)
#             optimizer.zero_grad()
#             loss = criterion(net(features), labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#     return running_loss / len(trainloader)
# 
# def test(net, testloader, device):
#     net.to(device)
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in testloader:
#             features = batch["features"].to(device)
#             labels = batch["label"].to(device)
#             outputs = net(features)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     loss = loss / len(testloader)
#     return loss, accuracy


# =========================================================================================
# MÔ HÌNH VÀ CÁC HÀM CHO Dữ liệu Purchase
# =========================================================================================

# class Net(nn.Module):
#     """Mô hình Neural Network cho tập dữ liệu Purchase gồm: 3 FC layers."""
# 
#     def __init__(self, input_dim=600):
#         # input_dim thường là 600 đối với Purchase dataset (dựa trên 600 loại mặt hàng)
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 2) # Gom thành 2 cụm (2 classes) theo paper
# 
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
# 
# fds = None  
# 
# def apply_transforms(batch):
#     """Chuyển đổi dữ liệu Purchase thành PyTorch Tensors."""
#     batch["features"] = [torch.tensor(feat, dtype=torch.float32) for feat in batch["features"]]
#     return batch
# 
# def load_data(partition_id: int, num_partitions: int, batch_size: int):
#     """Load partition Purchase data."""
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds= FederatedDataset(
#             dataset="purchase", # Tùy thuộc vào source chứa Purchase dataset
#             partitioners={"train": partitioner},
#         )
#     partition = fds.load_partition(partition_id)
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     
#     trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
#     return trainloader, testloader
# 
# def train(net, trainloader, epochs, lr, device):
#     net.to(device)
#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
#     net.train()
#     running_loss = 0.0
#     for _ in range(epochs):
#         for batch in trainloader:
#             features = batch["features"].to(device)
#             labels = batch["label"].to(device)
#             optimizer.zero_grad()
#             loss = criterion(net(features), labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#     return running_loss / len(trainloader)
# 
# def test(net, testloader, device):
#     net.to(device)
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in testloader:
#             features = batch["features"].to(device)
#             labels = batch["label"].to(device)
#             outputs = net(features)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     loss = loss / len(testloader)
#     return loss, accuracy