"""
Test suite for PyTorch state dictionary serialization utilities.
These tests verify the correct functioning of serialization and deserialization
operations on various types of PyTorch models and tensors.
"""

import unittest
import torch
import torch.nn as nn
from io import BytesIO
import json
import base64
import gzip
from serialization import (
    serialize_state_dict,
    decompress_and_deserialize_state_dict
)

class TestSerialization(unittest.TestCase):
    """
    Test cases for state dictionary serialization utilities.
    Tests various scenarios including different model architectures
    and tensor types.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        Creates various neural network models for testing.
        """
        # Simple linear model
        self.linear_model = nn.Linear(10, 5)
        
        # More complex model with different layer types
        self.complex_model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16)
        )
        
        # Create some test tensors
        self.test_tensors = {
            'float': torch.randn(5, 5),
            'long': torch.randint(0, 10, (5, 5)),
            'bool': torch.randint(0, 2, (5, 5)).bool()
        }

    def test_simple_linear_model(self):
        """
        Test serialization/deserialization with a simple linear model.
        Verifies that all parameters are preserved accurately.
        """
        original_state = self.linear_model.state_dict()
        serialized = serialize_state_dict(original_state)
        recovered_state = decompress_and_deserialize_state_dict(serialized)
        
        # Check that all keys are preserved
        self.assertEqual(original_state.keys(), recovered_state.keys())
        
        # Check that all tensors are equal
        for key in original_state:
            self.assertTrue(torch.equal(original_state[key], recovered_state[key]))

    def test_complex_model(self):
        """
        Test with a more complex model containing different layer types.
        Verifies that diverse parameter types are handled correctly.
        """
        original_state = self.complex_model.state_dict()
        serialized = serialize_state_dict(original_state)
        recovered_state = decompress_and_deserialize_state_dict(serialized)
        
        # Verify all parameters match
        for key in original_state:
            self.assertTrue(torch.equal(original_state[key], recovered_state[key]))
            self.assertEqual(original_state[key].dtype, recovered_state[key].dtype)
            self.assertEqual(original_state[key].shape, recovered_state[key].shape)

    def test_different_tensor_types(self):
        """
        Test handling of different tensor types (float, long, bool).
        Ensures that data types are preserved during serialization.
        """
        serialized = serialize_state_dict(self.test_tensors)
        recovered = decompress_and_deserialize_state_dict(serialized)
        
        for key in self.test_tensors:
            self.assertTrue(torch.equal(self.test_tensors[key], recovered[key]))
            self.assertEqual(self.test_tensors[key].dtype, recovered[key].dtype)

    def test_empty_state_dict(self):
        """
        Test handling of empty state dictionaries.
        Verifies that edge cases are handled properly.
        """
        empty_dict = {}
        serialized = serialize_state_dict(empty_dict)
        recovered = decompress_and_deserialize_state_dict(serialized)
        self.assertEqual(recovered, {})

    def test_compression_effectiveness(self):
        """
        Test that compression actually reduces data size.
        Verifies that the compression step is working as expected.
        """
        # Create a large tensor with repeated values
        large_tensor = torch.ones(1000, 1000)
        state_dict = {'large_tensor': large_tensor}
        
        # Get uncompressed size
        uncompressed_buffer = BytesIO()
        torch.save(state_dict, uncompressed_buffer)
        uncompressed_size = len(uncompressed_buffer.getvalue())
        
        # Get compressed size
        serialized = serialize_state_dict(state_dict)
        compressed_size = len(base64.b64decode(json.loads(serialized)))
        
        # Compression should reduce size
        self.assertLess(compressed_size, uncompressed_size)

    def test_invalid_input(self):
        """
        Test handling of invalid inputs.
        Verifies that appropriate errors are raised for invalid data.
        """
        # Test with invalid JSON
        with self.assertRaises(json.JSONDecodeError):
            decompress_and_deserialize_state_dict("invalid json")
            
        # Test with invalid base64
        with self.assertRaises(base64.binascii.Error):
            decompress_and_deserialize_state_dict(json.dumps("invalid base64"))

if __name__ == '__main__':
    unittest.main()