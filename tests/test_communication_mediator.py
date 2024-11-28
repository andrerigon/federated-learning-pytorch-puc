"""
test_communication_mediator.py
Unit tests for the CommunicationMediator class.

These tests verify the functionality of the CommunicationMediator including:
- Initialization with various success rates
- Message delivery simulation
- Metric tracking and reporting
- Error handling
"""

import unittest
from unittest.mock import Mock, patch
import logging
from communication import CommunicationMediator

class TestCommunicationMediator(unittest.TestCase):
    """
    Test suite for CommunicationMediator class.
    Tests both normal operation and edge cases.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        Creates a mock provider that will be used across tests.
        """
        self.mock_provider = Mock()
        self.mock_provider.send_communication_command = Mock()

    def test_initialization(self):
        """
        Test proper initialization of the mediator.
        Verifies that:
        - Valid success rates are accepted
        - Invalid success rates raise ValueError
        - Initial metrics are zero
        """
        # Test valid success rate
        mediator = CommunicationMediator(0.5)
        self.assertEqual(mediator.success_rate, 0.5)
        self.assertEqual(mediator.total_attempts, 0)
        self.assertEqual(mediator.successful_attempts, 0)

        # Test boundary values
        CommunicationMediator(0.0)  # Should not raise
        CommunicationMediator(1.0)  # Should not raise

        # Test invalid success rates
        with self.assertRaises(ValueError):
            CommunicationMediator(-0.1)
        with self.assertRaises(ValueError):
            CommunicationMediator(1.1)

    def test_guaranteed_delivery(self):
        """
        Test message delivery with 100% success rate.
        Verifies that:
        - All messages are delivered
        - Metrics are tracked correctly
        - Provider is called for each message
        """
        mediator = CommunicationMediator(1.0)
        test_message = "test"

        # Send multiple messages
        for _ in range(5):
            success = mediator.send_message(test_message, self.mock_provider)
            self.assertTrue(success)

        # Verify metrics
        self.assertEqual(mediator.total_attempts, 5)
        self.assertEqual(mediator.successful_attempts, 5)
        self.assertEqual(
            self.mock_provider.send_communication_command.call_count, 
            5
        )

    def test_guaranteed_failure(self):
        """
        Test message delivery with 0% success rate.
        Verifies that:
        - No messages are delivered
        - Metrics are tracked correctly
        - Provider is never called
        """
        mediator = CommunicationMediator(0.0)
        test_message = "test"

        # Send multiple messages
        for _ in range(5):
            success = mediator.send_message(test_message, self.mock_provider)
            self.assertFalse(success)

        # Verify metrics
        self.assertEqual(mediator.total_attempts, 5)
        self.assertEqual(mediator.successful_attempts, 0)
        self.mock_provider.send_communication_command.assert_not_called()

    @patch('random.random')
    def test_probabilistic_delivery(self, mock_random):
        """
        Test message delivery with 50% success rate.
        Uses mocked random numbers to ensure deterministic testing.
        Verifies that:
        - Messages are delivered according to probability
        - Metrics match expected outcomes
        """
        mediator = CommunicationMediator(0.5)
        test_message = "test"

        # Simulate alternating success/failure
        mock_random.side_effect = [0.4, 0.6, 0.4, 0.6]  # < 0.5 succeeds

        # Send messages and verify each outcome
        self.assertTrue(mediator.send_message(test_message, self.mock_provider))
        self.assertFalse(mediator.send_message(test_message, self.mock_provider))
        self.assertTrue(mediator.send_message(test_message, self.mock_provider))
        self.assertFalse(mediator.send_message(test_message, self.mock_provider))

        # Verify final metrics
        self.assertEqual(mediator.total_attempts, 4)
        self.assertEqual(mediator.successful_attempts, 2)

    def test_log_metrics(self):
        """
        Test metric logging functionality.
        Verifies that:
        - Metrics are calculated correctly
        - Returned dictionary contains all expected fields
        - Success rate is calculated properly
        """
        mediator = CommunicationMediator(0.5)
        test_message = "test"

        # Send some messages
        for _ in range(4):
            mediator.send_message(test_message, self.mock_provider)

        # Get metrics
        metrics = mediator.log_metrics()

        # Verify metric dictionary
        self.assertIn('success_rate', metrics)
        self.assertIn('total_attempts', metrics)
        self.assertIn('successful_attempts', metrics)
        
        # Verify metric calculations
        self.assertEqual(metrics['total_attempts'], 4)
        self.assertEqual(
            metrics['success_rate'],
            metrics['successful_attempts'] / metrics['total_attempts']
        )

    def test_reset_metrics(self):
        """
        Test metric reset functionality.
        Verifies that:
        - Metrics are properly reset to zero
        - Success rate is maintained
        - New messages are tracked correctly after reset
        """
        mediator = CommunicationMediator(1.0)
        test_message = "test"

        # Send some messages
        mediator.send_message(test_message, self.mock_provider)
        mediator.send_message(test_message, self.mock_provider)
        
        # Verify initial state
        self.assertEqual(mediator.total_attempts, 2)
        self.assertEqual(mediator.successful_attempts, 2)

        # Reset metrics
        mediator.reset_metrics()
        
        # Verify reset state
        self.assertEqual(mediator.total_attempts, 0)
        self.assertEqual(mediator.successful_attempts, 0)
        self.assertEqual(mediator.success_rate, 1.0)  # Should maintain success rate

        # Verify can still send messages
        mediator.send_message(test_message, self.mock_provider)
        self.assertEqual(mediator.total_attempts, 1)
        self.assertEqual(mediator.successful_attempts, 1)

if __name__ == '__main__':
    unittest.main()