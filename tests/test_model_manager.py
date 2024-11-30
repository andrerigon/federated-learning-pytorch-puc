"""
Test suite for ModelManager class.
Tests model creation, saving, and loading functionalities with proper resource cleanup.
"""

import unittest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
import time
from unittest.mock import Mock, patch
import gc
from model_manager import ModelManager

class SimpleTestModel(nn.Module):
    """A simple model for testing purposes."""
    def __init__(self, input_size: int = 10, hidden_size: int = 5):
        super().__init__()
        self.layer = nn.Linear(input_size, hidden_size)
        
    def forward(self, x):
        return self.layer(x)

class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager functionality with proper resource management."""
    
    def setUp(self):
        """Set up test environment and resources before each test."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create test model parameters
        self.input_size = 10
        self.hidden_size = 5
        
        # Define factory function for testing
        self.model_factory = lambda input_size=10, hidden_size=5: SimpleTestModel(input_size, hidden_size)
        
        # List to track created models for cleanup
        self.models_to_cleanup = []

    def tearDown(self):
        """Clean up all resources after each test."""
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Move models to CPU and delete them
        for model in self.models_to_cleanup:
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        
        # Clear the models list
        self.models_to_cleanup.clear()
        
        # Remove temporary directory and its contents
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
        # Force garbage collection
        gc.collect()

    def _track_model(self, model):
        """Helper method to track models for cleanup."""
        self.models_to_cleanup.append(model)
        return model

    def test_initialization_with_constructor(self):
        """Test ModelManager initialization using a model class constructor."""
        manager = ModelManager(
            SimpleTestModel,
            base_dir=self.test_dir,
            input_size=self.input_size,
            hidden_size=self.hidden_size
        )
        
        self.assertEqual(manager.base_dir, Path(self.test_dir))
        self.assertFalse(manager.from_scratch)
        self.assertEqual(manager.model_kwargs['input_size'], self.input_size)
        self.assertEqual(manager.model_kwargs['hidden_size'], self.hidden_size)

    def test_initialization_with_factory(self):
        """Test ModelManager initialization using a factory function."""
        manager = ModelManager(
            self.model_factory,
            base_dir=self.test_dir,
            input_size=self.input_size,
            hidden_size=self.hidden_size
        )
        
        model = self._track_model(manager.create_model())
        self.assertIsInstance(model, SimpleTestModel)
        self.assertEqual(model.layer.in_features, self.input_size)
        self.assertEqual(model.layer.out_features, self.hidden_size)

    def test_model_creation_with_different_approaches(self):
        """Test model creation using various approaches."""
        # Test with constructor
        manager1 = ModelManager(SimpleTestModel, base_dir=self.test_dir)
        model1 = self._track_model(manager1.create_model())
        self.assertIsInstance(model1, SimpleTestModel)
        
        # Test with factory function
        manager2 = ModelManager(self.model_factory, base_dir=self.test_dir)
        model2 = self._track_model(manager2.create_model())
        self.assertIsInstance(model2, SimpleTestModel)
        
        # Test with lambda
        manager3 = ModelManager(
            lambda: SimpleTestModel(15, 7),
            base_dir=self.test_dir
        )
        model3 = self._track_model(manager3.create_model())
        self.assertIsInstance(model3, SimpleTestModel)
        self.assertEqual(model3.layer.in_features, 15)
        self.assertEqual(model3.layer.out_features, 7)

    def test_save_and_load_model(self):
        """Test model saving and loading functionality."""
        manager = ModelManager(SimpleTestModel, base_dir=self.test_dir)
        
        # Create and save a model
        original_model = self._track_model(manager.create_model())
        original_state = original_model.state_dict()
        manager.save_model(original_model, version=1)
        
        # Load the model and verify states match
        loaded_model = self._track_model(manager.load_model())
        loaded_state = loaded_model.state_dict()
        
        # Compare state dictionaries
        for key in original_state:
            self.assertTrue(torch.equal(original_state[key], loaded_state[key]))
            
        # Clean up state dictionaries explicitly
        del original_state
        del loaded_state

    def test_get_last_model_path(self):
        """Test finding the most recent model file."""
        manager = ModelManager(SimpleTestModel, base_dir=self.test_dir)
        model = self._track_model(manager.create_model())
        
        # Save multiple versions with delays
        for version in range(1, 4):
            manager.save_model(model, version=version)
            time.sleep(0.1)  # Ensure different timestamps
        
        last_path = manager.get_last_model_path()
        self.assertTrue(str(last_path).endswith('3/model.pth'))

    def test_from_scratch_behavior(self):
        """Test that from_scratch flag prevents loading saved weights."""
        # Create and save initial model
        manager = ModelManager(SimpleTestModel, base_dir=self.test_dir)
        original_model = self._track_model(manager.create_model())
        manager.save_model(original_model, version=1)
        original_state = original_model.state_dict()
        
        # Create new manager with from_scratch=True
        new_manager = ModelManager(
            SimpleTestModel,
            base_dir=self.test_dir,
            from_scratch=True
        )
        
        new_model = self._track_model(new_manager.load_model())
        new_state = new_model.state_dict()
        
        # States should be different
        for key in original_state:
            self.assertFalse(torch.equal(original_state[key], new_state[key]))
            
        # Clean up state dictionaries
        del original_state
        del new_state

    def test_error_handling(self):
        """Test error handling for various failure scenarios."""
        # Test with non-existent directory
        manager = ModelManager(SimpleTestModel, base_dir="/nonexistent/path")
        self.assertIsNone(manager.get_last_model_path())
        
        # Test with invalid model state dict
        with patch('torch.load', side_effect=Exception("Mock error")):
            manager = ModelManager(SimpleTestModel, base_dir=self.test_dir)
            model = self._track_model(manager.load_model())
            self.assertIsInstance(model, SimpleTestModel)

    def test_version_directory_structure(self):
        """Test that version directories are created correctly."""
        manager = ModelManager(SimpleTestModel, base_dir=self.test_dir)
        model = self._track_model(manager.create_model())
        
        for version in [1, 5, 10]:
            manager.save_model(model, version=version)
            version_dir = Path(self.test_dir) / str(version)
            model_path = version_dir / 'model.pth'
            self.assertTrue(version_dir.exists())
            self.assertTrue(model_path.exists())

if __name__ == '__main__':
    unittest.main()