import unittest
import pytest
from unittest.mock import Mock, patch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from federated_learning_trainer import FederatedLearningTrainer

class MockDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = [(torch.randn(3, 32, 32), torch.randint(0, 10, ())) for _ in range(size)]
    
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.data[idx]

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(32 * 32 * 32, 10)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classified = self.classifier(encoded.flatten(1))
        return decoded, classified

class TestFederatedLearningTrainer(unittest.TestCase):
    def setUp(self):
        # Create model and dataset
        self.model = MockModel()
        self.dataset = MockDataset()
        self.dataloader = DataLoader(self.dataset, batch_size=32)
        
        # Setup mocks
        self.model_manager = Mock()
        self.model_manager.create_model.return_value = MockModel()
        self.model_manager.load_model.return_value = self.model
        
        self.dataset_loader = Mock()
        self.dataset_loader.loader.return_value = self.dataloader
        self.dataset_loader.testset = self.dataset  # Assuming testset is needed
        
        # Initialize trainer without starting the thread
        self.trainer = FederatedLearningTrainer(
            id=0,
            model_manager=self.model_manager,
            dataset_loader=self.dataset_loader,
            start_thread=False  # Prevent the thread from starting
        )

        # Set required attributes
        self.trainer.global_model = self.model
        self.trainer.loader = self.dataloader
        self.trainer.thread = Mock()
        self.trainer.thread.is_alive.return_value = True

    def tearDown(self):
        self.trainer.stop()
        torch.cuda.empty_cache()

    def test_initialization(self):
        self.assertEqual(self.trainer.training_cycles, 0)
        self.assertEqual(self.trainer.model_updates, 0) 
        self.assertEqual(self.trainer.global_model_version, 0)

    def test_update_model(self):
        state = self.model.state_dict()
        self.trainer.update_model(state, new_version=1)
        self.assertEqual(self.trainer.global_model_version, 1)

    @patch('torch.quantization.prepare_qat')
    @patch('torch.quantization.convert')
    def test_train_cycle(self, mock_convert, mock_prepare):
        mock_prepare.return_value = self.model
        mock_convert.return_value = self.model
        
        # Reset training cycles
        self.trainer.training_cycles = 0
        
        # Set loader and global model
        self.trainer.loader = self.dataloader
        self.trainer.global_model = self.model

        self.trainer.train(epochs=1)
        self.assertEqual(self.trainer.training_cycles, 1)

    def test_resource_cleanup(self):
        self.trainer.thread.is_alive.return_value = False
        
        # Mock stop() method
        original_stop = self.trainer.stop
        def mock_stop():
            self.trainer.finished = True
            self.trainer.thread.join()
        self.trainer.stop = mock_stop
        
        try:
            self.trainer.stop()
            self.assertTrue(self.trainer.finished)
        finally:
            # Restore original method
            self.trainer.stop = original_stop

if __name__ == '__main__':
    unittest.main()