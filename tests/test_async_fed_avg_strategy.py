import unittest
import torch
import torch.nn as nn
import time
from collections import OrderedDict
from aggregation_strategy import AsyncFedAvgStrategy

class TestAsyncFedAvgStrategy(unittest.TestCase):
    def setUp(self):
        """
        Create a simple neural network and initialize the strategy.
        We use a small network to make tests manageable and reproducible.
        """
        # Create a simple model for testing
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Single linear layer for simplicity
                self.fc = nn.Linear(2, 1)
                
                # Initialize weights deterministically for testing
                with torch.no_grad():
                    self.fc.weight.fill_(1.0)
                    self.fc.bias.fill_(0.0)
                    
            def forward(self, x):
                return self.fc(x)
        
        # Initialize our test environment
        self.model = SimpleNet()
        self.strategy = AsyncFedAvgStrategy(
            staleness_threshold=10,  # 10 seconds threshold
            base_mixing_weight=0.1   # 10% base mixing weight
        )
        
        # Create a client update that's different from the global model
        self.client_model = SimpleNet()
        with torch.no_grad():
            self.client_model.fc.weight.fill_(2.0)  # Different weight
            
    def test_first_update_weight(self):
        """
        Test that the first update from a client uses the base mixing weight.
        This ensures new clients don't overwhelm the global model.
        """
        client_id = "client1"
        current_time = time.time()
        
        weight = self.strategy._calculate_staleness_weight(client_id, current_time)
        self.assertEqual(weight, self.strategy.base_mixing_weight)
        
    def test_staleness_decay(self):
        """
        Test that update weights decay appropriately with staleness.
        We simulate updates at different time intervals to verify the decay.
        """
        client_id = "client2"
        current_time = time.time()
        
        # Record an initial update
        self.strategy.last_update_time[client_id] = current_time - 20  # 20 seconds ago
        
        # Calculate weight for a stale update
        weight = self.strategy._calculate_staleness_weight(client_id, current_time)
        
        # With staleness_threshold=10 and 20 seconds passed:
        # staleness_factor = 1 / (1 + 20/10) = 1/3
        # weight should be base_mixing_weight * max(0.1, 1/3)
        expected_weight = self.strategy.base_mixing_weight * (1.0 / 3.0)
        self.assertAlmostEqual(weight, expected_weight, places=5)
        
    def test_model_update_mixing(self):
        """
        Test that model parameters are properly mixed during updates.
        We verify that the new parameters are a weighted combination of
        global and client parameters.
        """
        client_id = "client3"
        extra_info = {'client_id': client_id}
        
        # Initial global model has weights of 1.0
        # Client model has weights of 2.0
        
        # Perform update
        new_state_dict = self.strategy.aggregate(
            self.model,
            self.client_model.state_dict(),
            extra_info=extra_info
        )
        
        # With base_mixing_weight=0.1:
        # new_weight = (1 - 0.1) * 1.0 + 0.1 * 2.0
        # = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        expected_weight = 1.1
        
        actual_weight = new_state_dict['fc.weight'].item()
        self.assertAlmostEqual(actual_weight, expected_weight, places=5)
        
    def test_missing_client_id(self):
        """
        Test that the strategy handles missing client IDs gracefully.
        Should return the original model state when client ID is missing.
        """
        # Try to update without client_id
        new_state_dict = self.strategy.aggregate(
            self.model,
            self.client_model.state_dict(),
            extra_info={}  # Empty extra_info
        )
        
        # Should return original model state
        self.assertTrue(
            torch.equal(
                new_state_dict['fc.weight'],
                self.model.state_dict()['fc.weight']
            )
        )
        
    def test_consecutive_updates(self):
        """
        Test that consecutive updates from the same client are handled properly.
        Verifies that staleness calculations work over multiple updates.
        """
        client_id = "client4"
        extra_info = {'client_id': client_id}
        
        # First update
        state_dict_1 = self.strategy.aggregate(
            self.model,
            self.client_model.state_dict(),
            extra_info=extra_info
        )
        
        # Wait a moment
        time.sleep(0.1)
        
        # Second update should have slightly different weight due to staleness
        state_dict_2 = self.strategy.aggregate(
            self.model,
            self.client_model.state_dict(),
            extra_info=extra_info
        )
        
        # Weights should be different due to staleness
        self.assertNotEqual(
            state_dict_1['fc.weight'].item(),
            state_dict_2['fc.weight'].item()
        )
        
    def test_extreme_staleness(self):
        """
        Test that extremely stale updates have minimal but non-zero effect.
        Verifies that the minimum weight is properly applied.
        """
        client_id = "client5"
        current_time = time.time()
        
        # Simulate a very old update (1 hour old)
        self.strategy.last_update_time[client_id] = current_time - 3600
        
        weight = self.strategy._calculate_staleness_weight(client_id, current_time)
        
        # Weight should be at least 0.1 * base_mixing_weight
        min_weight = 0.1 * self.strategy.base_mixing_weight
        self.assertGreaterEqual(weight, min_weight)
        
        # Weight should be less than base_mixing_weight
        self.assertLess(weight, self.strategy.base_mixing_weight)

if __name__ == '__main__':
    unittest.main()