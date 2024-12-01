import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from federated_learning_trainer import FederatedLearningTrainer
from metrics_logger import MetricsLogger

class MockDataset(Dataset):
    def __init__(self, size=2):  # Minimum size needed
        torch.manual_seed(42)  # For reproducibility
        self.data = [(torch.randn(3, 4, 4), torch.tensor(0)) for _ in range(size)]
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 2, 1)
        self.decoder = nn.Conv2d(2, 3, 1)
        self.classifier = nn.Linear(32, 10)  # 4x4x2=32
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classified = self.classifier(encoded.flatten(1))
        return decoded, classified

class TestFederatedLearningTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create shared resources
        cls.model = MockModel()
        cls.dataset = MockDataset()
        cls.dataloader = DataLoader(cls.dataset, batch_size=1)
    
    def setUp(self):
        # Mock time.time globally
        self.time_patcher = patch('time.time')
        self.mock_time = self.time_patcher.start()
        self.mock_time.return_value = 100  # Default time value

        self.metrics_logger = Mock(spec=MetricsLogger)
        self.model_manager = Mock()
        self.model.train()  # Important: set model to training mode
        self.model_manager.create_model.return_value = self.model
        self.model_manager.load_model.return_value = self.model
        
        self.dataset_loader = Mock()
        self.dataset_loader.loader.return_value = self.dataloader
        self.dataset_loader.testset = self.dataset

        # Initialize trainer
        self.trainer = FederatedLearningTrainer(
            id=0,
            model_manager=self.model_manager,
            dataset_loader=self.dataset_loader,
            metrics_logger=self.metrics_logger,
            start_thread=False
        )

        # Setup trainer attributes
        self.trainer.global_model = self.model
        self.trainer.loader = self.dataloader
        self.trainer.local_model = self.model

    def tearDown(self):
        self.time_patcher.stop()
        if hasattr(self, 'trainer'):
            self.trainer.finished = True  # Ensure training stops
            if hasattr(self.trainer, 'global_model'):
                self.trainer.global_model = None
            if hasattr(self.trainer, 'local_model'):
                self.trainer.local_model = None
        torch.cuda.empty_cache()

    def test_initialization(self):
        """Test initial state of trainer."""
        self.assertEqual(self.trainer.training_cycles, 0)
        self.assertEqual(self.trainer.model_updates, 0)
        self.assertEqual(self.trainer.global_model_version, 0)
        self.assertFalse(self.trainer.finished)

    def test_update_model(self):
        """Test model update process."""
        state = self.model.state_dict()
        self.trainer.update_model(state, new_version=1)
        
        # Verify that the global model is updated
        self.assertEqual(self.trainer.global_model_version, 1)
        self.assertTrue(self.trainer.global_model_changed)
        self.assertEqual(self.trainer.model_updates, 1)
        # Since register_staleness is not called in update_model, we do not assert it here

    @patch('federated_learning_trainer.BytesIO')
    @patch('federated_learning_trainer.torch.save')
    def test_model_size_calculation(self, mock_torch_save, mock_bytesio):
        """Test model size calculation."""
        test_data = b'test' * 1000
        mock_buffer = MagicMock()
        mock_buffer.getvalue.return_value = test_data
        mock_bytesio.return_value = mock_buffer
        
        size = self.trainer.get_model_size(self.model)
        mock_torch_save.assert_called_once()
        self.assertEqual(size, len(test_data))

    @patch('torch.quantization.prepare_qat')
    @patch('torch.quantization.convert')
    def test_train_cycle(self, mock_convert, mock_prepare):
        """Test training cycle execution."""
        self.mock_time.side_effect = [100, 200] + [300] * 10
        mock_prepare.return_value = self.model
        mock_convert.return_value = self.model
        
        with patch.object(self.trainer, 'train_one_epoch', return_value=(0.1, 0.1)):
            self.trainer.train(epochs=1)
            self.assertEqual(self.trainer.training_cycles, 1)
            self.metrics_logger.register_loss.assert_called()

    def test_evaluate_model(self):
        """Test model evaluation."""
        with torch.no_grad():
            self.trainer.evaluate_model(self.model)
            self.metrics_logger.register_accuracy.assert_called_once()

    @patch('torch.quantization.convert')
    def test_log_model_sizes(self, mock_convert):
        """Test model size logging."""
        mock_convert.return_value = self.model
        test_data = b'test' * 1000
        mock_buffer = MagicMock()
        mock_buffer.getvalue.return_value = test_data

        with patch('federated_learning_trainer.BytesIO') as mock_bytesio:
            mock_bytesio.return_value = mock_buffer
            self.trainer.log_model_sizes(self.model)
            self.assertEqual(self.metrics_logger.register_model_size.call_count, 2)

    @patch('torch.quantization.prepare_qat')
    @patch('torch.quantization.convert')
    def test_training_time_registration(self, mock_convert, mock_prepare):
        """Test training time tracking."""
        mock_prepare.return_value = self.model
        mock_convert.return_value = self.model
        self.mock_time.side_effect = [100, 200] + [300] * 10

        with patch.object(self.trainer, 'train_one_epoch', return_value=(0.1, 0.1)):
            self.trainer.train(epochs=1)
            self.metrics_logger.register_training_time.assert_called()

    @patch('torch.quantization.prepare_qat')
    @patch('torch.quantization.convert')
    def test_metrics_during_training(self, mock_convert, mock_prepare):
        """Test metric registration during training."""
        mock_prepare.return_value = self.model
        mock_convert.return_value = self.model
        self.mock_time.side_effect = [100, 200] + [300] * 10

        with patch.object(self.trainer, 'train_one_epoch', return_value=(0.1, 0.1)):
            self.trainer.train(epochs=1)
            self.assertGreater(self.metrics_logger.register_loss.call_count, 0)
            self.metrics_logger.register_training_time.assert_called()
            self.metrics_logger.register_global_step.assert_called_once()

    @patch('torch.cuda.empty_cache')
    def test_resource_cleanup(self, mock_empty_cache):
        """Test proper resource cleanup."""
        self.trainer.global_model = self.model
        self.trainer.stop()
        self.assertTrue(self.trainer.finished)
        self.metrics_logger.flush.assert_called_once()
        mock_empty_cache.assert_called_once()

    def test_error_handling_during_training(self):
        """Test error handling in training process."""
        error_msg = "Test error"
        with patch.object(self.trainer, 'prepare_local_model', side_effect=Exception(error_msg)), \
             self.assertLogs(level='ERROR') as log:
            self.trainer.train(epochs=1)
            self.assertIn(error_msg, log.output[0])

    def test_prepare_local_model(self):
        """Test local model preparation."""
        with patch('torch.quantization.prepare_qat', return_value=self.model):
            local_model = self.trainer.prepare_local_model()
            self.assertIsNotNone(local_model)
            self.assertTrue(hasattr(local_model, 'qconfig'))

    def test_get_loss_functions_and_optimizer(self):
        """Test loss functions and optimizer setup."""
        criterion_reconstruction, criterion_classification, optimizer, scheduler = \
            self.trainer.get_loss_functions_and_optimizer(self.model)
        
        self.assertIsInstance(criterion_reconstruction, nn.MSELoss)
        self.assertIsInstance(criterion_classification, nn.CrossEntropyLoss)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_train_one_batch(self):
        """Test single batch training."""
        inputs, labels = next(iter(self.dataloader))
        optimizer = torch.optim.AdamW(self.model.parameters())
        criterion_reconstruction = nn.MSELoss()
        criterion_classification = nn.CrossEntropyLoss()

        # Do not use torch.no_grad() here
        loss, rec_loss, class_loss, decoded, classified = self.trainer.train_one_batch(
            self.model, inputs, labels, 
            criterion_reconstruction, criterion_classification, optimizer
        )

        self.assertIsNotNone(loss)
        self.assertIsNotNone(rec_loss)
        self.assertIsNotNone(class_loss)
        self.assertEqual(decoded.shape, inputs.shape)
        self.assertEqual(classified.shape[0], inputs.shape[0])

if __name__ == '__main__':
    unittest.main()