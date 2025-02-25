import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import tempfile
import threading
from metrics_logger import MetricsLogger
import numpy as np
import torch
import logging

class TestMetricsLogger(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()
        self.client_id = 1
        self.metrics_logger = MetricsLogger(
            client_id=self.client_id,
            output_dir=self.test_dir,
            generate_visualizations=True  # Enable for testing
        )
        # Set logging level to capture warnings and errors
        logging.basicConfig(level=logging.DEBUG)
    
    def tearDown(self):
        # Clean up the temporary directory after tests
        shutil.rmtree(self.test_dir)
        self.metrics_logger.close()
    
    def test_register_loss(self):
        """Test registering losses."""
        step = 1
        value = 0.5
        self.metrics_logger.register_loss('reconstruction', step, value)
        self.metrics_logger.register_loss('classification', step, value)
        self.assertIn((step, value), self.metrics_logger.metrics['losses']['reconstruction'])
        self.assertIn((step, value), self.metrics_logger.metrics['losses']['classification'])
    
    def test_register_accuracy(self):
        """Test registering accuracy."""
        step = 1
        value = 0.8
        self.metrics_logger.register_accuracy(step, value)
        self.assertIn((step, value), self.metrics_logger.metrics['accuracy'])
    
    def test_register_model_size(self):
        """Test registering model sizes."""
        step = 1
        value = 1024
        self.metrics_logger.register_model_size('non_quantized', step, value)
        self.metrics_logger.register_model_size('quantized', step, value)
        self.assertIn((step, value), self.metrics_logger.metrics['model_sizes']['non_quantized'])
        self.assertIn((step, value), self.metrics_logger.metrics['model_sizes']['quantized'])
    
    def test_register_staleness(self):
        """Test registering staleness."""
        update_number = 1
        value = 0.1
        timestamp = 1000
        self.metrics_logger.register_staleness(update_number, value, timestamp)
        self.assertIn((update_number, value, timestamp), self.metrics_logger.metrics['staleness'])
    
    def test_register_training_time(self):
        """Test registering training time."""
        step = 1
        duration = 120
        self.metrics_logger.register_training_time(step, duration)
        self.assertIn((step, duration), self.metrics_logger.metrics['training_times'])
    
    def test_register_mse_per_sample(self):
        """Test registering MSE per sample."""
        step = 1
        outputs = torch.randn(10, 3, 32, 32)
        targets = torch.randn(10, 3, 32, 32)
        self.metrics_logger.register_mse_per_sample(step, outputs, targets)
        # Check that raw_data['mse_values'] is updated
        self.assertEqual(len(self.metrics_logger.raw_data['mse_values']), 1)
        registered_step, mse_values = self.metrics_logger.raw_data['mse_values'][0]
        self.assertEqual(registered_step, step)
        self.assertEqual(mse_values.shape[0], 10)  # Should have 10 MSE values
    
    def test_register_predictions(self):
        """Test registering predictions and embeddings."""
        step = 1
        predictions = torch.tensor([0, 1, 2, 3])
        true_labels = torch.tensor([0, 1, 2, 3])
        embeddings = torch.randn(4, 128)
        self.metrics_logger.register_predictions(step, predictions, true_labels, embeddings)
        # Check embeddings data
        self.assertEqual(len(self.metrics_logger.raw_data['embeddings_data']), 1)
        registered_step, registered_embeddings, labels_np = self.metrics_logger.raw_data['embeddings_data'][0]
        self.assertEqual(registered_step, step)
        np.testing.assert_array_equal(labels_np, true_labels.numpy())
        self.assertEqual(registered_embeddings.shape, (4, 128))
        # Check confusion matrix
        self.assertEqual(len(self.metrics_logger.raw_data['confusion_matrices']), 1)
        registered_step_cm, cm = self.metrics_logger.raw_data['confusion_matrices'][0]
        self.assertEqual(registered_step_cm, step)
        self.assertEqual(cm.shape, (4, 4))
    
    def test_generate_visualizations(self):
        """Test generating visualizations."""
        with patch('threading.current_thread') as mock_current_thread:
            mock_current_thread.return_value = threading.main_thread()
            # Add sample data
            step = 1
            mse_values = np.random.rand(10)
            self.metrics_logger.raw_data['mse_values'].append((step, mse_values))
            cm = np.array([[5, 0], [0, 5]])
            self.metrics_logger.raw_data['confusion_matrices'].append((step, cm))
            embeddings = np.random.rand(10, 128)
            labels = np.random.randint(0, 2, size=10)
            self.metrics_logger.raw_data['embeddings_data'].append((step, embeddings, labels))
            # Run the method
            self.metrics_logger.generate_visualizations()
            # Since actual file outputs are not checked here, we ensure no exceptions occur
    
    def test_write_metrics_to_tensorboard(self):
        """Test writing metrics to TensorBoard."""
        with patch.object(self.metrics_logger.writer, 'add_scalar') as mock_add_scalar:
            # Add sample data
            self.metrics_logger.metrics['losses']['reconstruction'].append((1, 0.5))
            self.metrics_logger.metrics['losses']['classification'].append((1, 0.6))
            self.metrics_logger.metrics['accuracy'].append((1, 0.8))
            self.metrics_logger.metrics['model_sizes']['non_quantized'].append((1, 1000))
            self.metrics_logger.metrics['model_sizes']['quantized'].append((1, 800))
            self.metrics_logger.metrics['staleness'].append((1, 0.1, 1000))
            self.metrics_logger.metrics['training_times'].append((1, 120))
            # Run the method
            self.metrics_logger.write_metrics_to_tensorboard()
            # Ensure add_scalar was called
            self.assertTrue(mock_add_scalar.called)
    
    def test_generate_plots(self):
        """Test generating plots."""
        with patch('threading.current_thread') as mock_current_thread:
            mock_current_thread.return_value = threading.main_thread()
            # Add sample data
            self.metrics_logger.metrics['losses']['reconstruction'].append((1, 0.5))
            self.metrics_logger.metrics['losses']['classification'].append((1, 0.6))
            self.metrics_logger.metrics['accuracy'].append((1, 0.8))
            self.metrics_logger.metrics['staleness'].append((1, 0.1, 1000))
            self.metrics_logger.metrics['training_times'].append((1, 120))
            # Run the method
            self.metrics_logger.generate_plots()
            # Check that expected plot files exist
            expected_files = [
                'reconstruction_loss_plot.png',
                'classification_loss_plot.png',
                'accuracy_plot.png',
                f'staleness_over_time_{self.client_id}.png',
                'training_times_plot.png'
            ]
            for filename in expected_files:
                filepath = os.path.join(self.test_dir, f'client_{self.client_id}', filename)
                self.assertTrue(os.path.exists(filepath))
    
    def test_flush(self):
        """Test the flush method."""
        with patch.object(self.metrics_logger, 'write_metrics_to_tensorboard') as mock_write_metrics, \
             patch.object(self.metrics_logger, 'generate_visualizations') as mock_generate_visualizations, \
             patch.object(self.metrics_logger, 'generate_plots') as mock_generate_plots, \
             patch.object(self.metrics_logger, 'close') as mock_close:
            self.metrics_logger.flush()
            mock_write_metrics.assert_called_once()
            mock_generate_visualizations.assert_called_once()
            mock_generate_plots.assert_called_once()
            mock_close.assert_called_once()
    
    def test_close(self):
        """Test closing the metrics logger."""
        with patch.object(self.metrics_logger.writer, 'close') as mock_writer_close:
            self.metrics_logger.close()
            mock_writer_close.assert_called_once()
    
    def test_add_custom_scalars_failure(self):
        """Test handling failure in adding custom scalars."""
        with patch.object(self.metrics_logger.writer, 'add_custom_scalars', side_effect=Exception('Test exception')), \
            self.assertLogs(level='WARNING') as cm:
            # Re-initialize to trigger add_custom_scalars
            self.metrics_logger.__init__(client_id=self.client_id, output_dir=self.test_dir)
        # Check the log records' messages
        self.assertTrue(any('Failed to add custom scalar layout' in record.getMessage() for record in cm.records))

    def test_plot_methods_handle_exceptions(self):
        """Test that plot methods handle exceptions gracefully."""
        with patch('matplotlib.pyplot.savefig', side_effect=Exception('Test exception')), \
            self.assertLogs(level='ERROR') as cm:
            # Attempt to plot with invalid data
            self.metrics_logger.plot_loss([], [], self.test_dir)
        # Check that an error was logged
        self.assertTrue(any('Error generating plot loss_plot.png' in record.getMessage() for record in cm.records))
        
    def test_register_loss_unknown_type(self):
        """Test registering a loss with an unknown type."""
        with self.assertLogs(level='WARNING') as cm:
            self.metrics_logger.register_loss('unknown_loss', 1, 0.5)
            self.assertTrue(any('Unknown loss type' in message for message in cm.output))
    
    def test_register_model_size_unknown_type(self):
        """Test registering a model size with an unknown type."""
        with self.assertLogs(level='WARNING') as cm:
            self.metrics_logger.register_model_size('unknown_size', 1, 1024)
            self.assertTrue(any('Unknown model size type' in message for message in cm.output))
    
    def test_generate_visualizations_not_main_thread(self):
        """Test generating visualizations in a non-main thread."""
        with patch('threading.current_thread') as mock_current_thread, \
             self.assertLogs(level='WARNING') as cm:
            mock_current_thread.return_value = threading.Thread(name='Thread-1')
            self.metrics_logger.generate_visualizations()
            self.assertTrue(any('Attempting to generate visualizations outside main thread' in message for message in cm.output))
    
    def test_generate_plots_not_main_thread(self):
        """Test generating plots in a non-main thread."""
        with patch('threading.current_thread') as mock_current_thread, \
             self.assertLogs(level='WARNING') as cm:
            mock_current_thread.return_value = threading.Thread(name='Thread-1')
            self.metrics_logger.generate_plots()
            self.assertTrue(any('Skipping plot generation in non-main thread' in message for message in cm.output))
    
    def test_register_staleness_without_timestamp(self):
        """Test registering staleness without providing a timestamp."""
        update_number = 1
        value = 0.1
        with patch('time.time', return_value=1000):
            self.metrics_logger.register_staleness(update_number, value)
        self.assertIn((update_number, value, 1000), self.metrics_logger.metrics['staleness'])
    
    def test_visualization_disabled(self):
        """Test that visualizations aren't generated when disabled."""
        # Create a new metrics logger with visualizations disabled
        disabled_logger = MetricsLogger(
            client_id=self.client_id + 1,
            output_dir=self.test_dir,
            generate_visualizations=False
        )
        
        # Add sample data
        disabled_logger.metrics['losses']['reconstruction'].append((1, 0.5))
        disabled_logger.metrics['accuracy'].append((1, 0.8))
        
        # Run the method
        with patch('threading.current_thread') as mock_current_thread:
            mock_current_thread.return_value = threading.main_thread()
            disabled_logger.generate_plots()
        
        # Check that no plot files were created
        plot_files = [
            'reconstruction_loss_plot.png',
            'accuracy_plot.png'
        ]
        for filename in plot_files:
            filepath = os.path.join(self.test_dir, f'client_{self.client_id + 1}', filename)
            self.assertFalse(os.path.exists(filepath))

if __name__ == '__main__':
    unittest.main()