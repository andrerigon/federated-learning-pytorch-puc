import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np

class DataLoaderWrapper:
    @staticmethod
    def download_dataset():
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        return trainset, testset

    @staticmethod
    def split_dataset(dataset, num_clients):
        data_indices = list(range(len(dataset)))
        split_indices = np.array_split(data_indices, num_clients)
        client_datasets = [Subset(dataset, indices) for indices in split_indices]
        return client_datasets
