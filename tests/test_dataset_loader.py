"""
Test suite for DatasetLoader class with optimized performance.
Tests the flexible client-based data loading functionality while maintaining
efficient resource usage and proper cleanup.
"""

import unittest
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from unittest.mock import Mock, patch
import tempfile
import shutil
import os
import gc

from dataset_loader import DatasetLoader

class TestDatasetLoader(unittest.TestCase):
    """
    Tests for the DatasetLoader class with performance optimizations.
    Verifies the functionality of multi-client dataset management and loading.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up class-level resources once for all tests.
        Creates a shared environment for testing with minimal dataset downloads.
        """
        # Set up temporary directory for dataset storage
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_data_dir = './data'
        torchvision.datasets.CIFAR10.root = cls.temp_dir

        # Initialize shared test parameters
        cls.num_clients = 4
        
        # Create a shared DatasetLoader instance that will be used across tests
        cls.shared_loader = DatasetLoader(cls.num_clients)
        
        # Create a shared DataLoader for the first client with minimal configuration
        cls.shared_dataloader = cls.shared_loader.loader(
            client_id=0,
            batch_size=4,
            shuffle=False,  # Deterministic for testing
            num_workers=0,
            persistent_workers=False,
            pin_memory=False
        )
        
        # Pre-fetch one batch for testing data properties
        cls.sample_images, cls.sample_labels = next(iter(cls.shared_dataloader))

    @classmethod
    def tearDownClass(cls):
        """
        Clean up all class-level resources and ensure proper memory management.
        """
        # Clean up shared DataLoader
        if hasattr(cls.shared_dataloader, '_iterator'):
            cls.shared_dataloader._iterator = None
        
        # Clear references to shared resources
        del cls.shared_dataloader
        del cls.shared_loader
        del cls.sample_images
        del cls.sample_labels
        
        # Remove temporary directory and restore original data path
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        torchvision.datasets.CIFAR10.root = cls.original_data_dir
        
        # Force memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def setUp(self):
        """Initialize test-specific resources."""
        self.test_specific_loaders = []

    def tearDown(self):
        """Clean up test-specific resources after each test."""
        for loader in self.test_specific_loaders:
            if hasattr(loader, '_iterator'):
                loader._iterator = None
        self.test_specific_loaders.clear()

    def test_initialization(self):
        """Verify proper initialization and data splitting."""
        loader = DatasetLoader(self.num_clients)
        # Test data size for the first client
        client_size = loader.get_client_data_size(0)
        expected_size = 50000 // self.num_clients
        self.assertEqual(client_size, expected_size)
        
        # Test test set initialization
        test_loader = loader.test_loader()
        self.assertEqual(len(test_loader.dataset), 10000)

    def test_data_loader_creation(self):
        """Test creation of DataLoaders for different clients."""
        # Create loaders for different clients with custom parameters
        for client_id in range(self.num_clients):
            custom_loader = self.shared_loader.loader(
                client_id=client_id,
                batch_size=32,
                shuffle=False,
                num_workers=0,
                persistent_workers=False,
                pin_memory=False
            )
            self.test_specific_loaders.append(custom_loader)
            
            # Verify loader configuration
            self.assertEqual(custom_loader.batch_size, 32)
            self.assertFalse(custom_loader.pin_memory)
            self.assertEqual(custom_loader.num_workers, 0)

    def test_data_loading_functionality(self):
        """Verify data format and properties across different clients."""
        # Test data loading for each client
        for client_id in range(self.num_clients):
            loader = self.shared_loader.loader(
                client_id=client_id,
                batch_size=4,
                num_workers=0
            )
            self.test_specific_loaders.append(loader)
            
            # Get a batch from each client
            images, labels = next(iter(loader))
            
            # Verify data properties
            self.assertEqual(images.shape, (4, 3, 32, 32))
            self.assertEqual(labels.shape, (4,))
            self.assertTrue(torch.is_tensor(images))
            self.assertTrue(torch.is_tensor(labels))

    def test_test_loader_functionality(self):
        """Verify test set loader configuration and functionality."""
        test_loader = self.shared_loader.test_loader(
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        self.test_specific_loaders.append(test_loader)
        
        # Verify test loader properties
        self.assertEqual(test_loader.batch_size, 16)
        self.assertEqual(len(test_loader.dataset), 10000)
        
        # Verify test data format
        images, labels = next(iter(test_loader))
        self.assertEqual(images.shape[1:], (3, 32, 32))

    def test_invalid_client_id(self):
        """Verify proper error handling for invalid client IDs."""
        with self.assertRaises(ValueError) as context:
            self.shared_loader.loader(client_id=self.num_clients)
        self.assertTrue("Client ID must be between" in str(context.exception))

    def test_data_distribution(self):
        """Verify even distribution of data across clients."""
        sizes = [self.shared_loader.get_client_data_size(i) 
                for i in range(self.num_clients)]
        
        # Verify total size
        self.assertEqual(sum(sizes), 50000)
        
        # Verify approximately equal distribution
        size_difference = max(sizes) - min(sizes)
        self.assertLessEqual(size_difference, 1)

if __name__ == '__main__':
    unittest.main()