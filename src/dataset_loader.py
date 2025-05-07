from typing import Tuple, List
from torch.utils.data import Subset, DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import torch
import logging
from torchvision.datasets import EuroSAT
from torch.utils.data import random_split

class DatasetLoader:
    """
    Manages dataset operations for federated learning clients.
    
    This class handles dataset downloading, preprocessing, and splitting for federated
    learning scenarios. Instead of being tied to a specific client, it can create
    loaders for any client ID on demand, making it more flexible and reusable.
    
    Attributes:
        num_clients (int): Total number of clients in federated learning setup
        testset (Dataset): Complete test dataset shared by all clients
        client_datasets (List[Subset]): List of dataset portions for each client
    """

    def __init__(self, num_clients: int):
        """
        Initialize the dataset loader with the total number of clients.
        
        Args:
            num_clients: Total number of clients participating in federated learning
            
        Raises:
            ValueError: If num_clients < 1
            RuntimeError: If dataset download or preparation fails
        """
        if num_clients < 1:
            raise ValueError("Number of clients must be at least 1")
            
        self.num_clients = num_clients
        
        try:
            # Download and prepare datasets
            trainset, self.testset = self._download_dataset()
            # Split training set into portions for each client
            self.client_datasets = self._split_dataset(trainset)
            logging.info(f"Initialized DatasetLoader for {num_clients} clients")
        except Exception as e:
            logging.error(f"Failed to initialize DatasetLoader: {str(e)}")
            raise RuntimeError(f"Dataset initialization failed: {str(e)}")

    def loader(self, 
               client_id: int,
               batch_size: int = 32,
               shuffle: bool = False,
               num_workers: int = 0,
               persistent_workers: bool = True,
               pin_memory: bool = False) -> DataLoader:
        """
        Create a DataLoader for a specific client's portion of the dataset.
        
        Args:
            client_id: Identifier for the client (0 to num_clients-1)
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data at each epoch
            num_workers: Number of subprocesses for data loading
            persistent_workers: Whether to maintain worker processes between iterations
            pin_memory: Whether to pin memory in GPU training
            
        Returns:
            DataLoader configured with the specified parameters for the client's dataset
            
        Raises:
            ValueError: If client_id is invalid
        """
        if not 0 <= client_id < self.num_clients:
            raise ValueError(f"Client ID must be between 0 and {self.num_clients-1}")
        if torch.cuda.is_available() and not pin_memory:
            logging.info("CUDA detected, enabling pin_memory for improved performance")
            pin_memory = True
            
        return DataLoader(
            dataset=self.client_datasets[client_id],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=pin_memory
        )

    def test_loader(self, 
                   batch_size: int = 32,
                   shuffle: bool = False,
                   num_workers: int = 0) -> DataLoader:
        """
        Create a DataLoader for the test dataset.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of subprocesses for data loading
            
        Returns:
            DataLoader for the complete test dataset
        """
        return DataLoader(
            dataset=self.testset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=torch.cuda.is_available()
        )

    def _download_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        Download and prepare the EuroSAT dataset with appropriate transforms.
        
        Returns:
            Tuple containing (training_dataset, test_dataset)
        """
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # match your autoencoder input
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        try:
            full = EuroSAT(
                root='./data/eurosat',
                download=True,
                transform=transform
            )

            # 80/20 train/test split
            n = len(full)
            n_train = int(0.8 * n)
            trainset, testset = random_split(full, [n_train, n - n_train])
            
            logging.info("Successfully downloaded and prepared EuroSAT dataset")
            return trainset, testset
            
        except Exception as e:
            logging.error(f"Failed to download dataset: {str(e)}")
            raise

    def _split_dataset(self, dataset: Dataset, labels_per_client: int = 2) -> List[Subset]:
        """
        Assign each client an IID shard of the dataset, ignoring labels_per_client.
        Works on Subset or any Dataset.
        """
        total = len(dataset)
        indices = np.arange(total)
        np.random.shuffle(indices)
        shards = np.array_split(indices, self.num_clients)
        return [Subset(dataset, shard.tolist()) for shard in shards]

    def get_client_data_size(self, client_id: int) -> int:
        """
        Get the number of samples assigned to a specific client.
        
        Args:
            client_id: The client's identifier
            
        Returns:
            Number of samples in the client's dataset portion
            
        Raises:
            ValueError: If client_id is invalid
        """
        if not 0 <= client_id < self.num_clients:
            raise ValueError(f"Client ID must be between 0 and {self.num_clients-1}")
        return len(self.client_datasets[client_id])

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset (10 for EuroSAT)."""
        return 10