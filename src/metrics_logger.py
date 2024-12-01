import os
import logging
import matplotlib.pyplot as plt
import time 
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class MetricsLogger:
    def __init__(self, client_id, output_dir='runs'):
        self.client_id = client_id
        self.output_dir = os.path.join(output_dir, f"client_{client_id}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self._log = logging.getLogger(__name__)

        # Initialize metrics storage
        self.metrics = {
            'staleness': [],
            'losses': {
                'reconstruction': [],
                'classification': []
            },
            'model_sizes': {
                'non_quantized': [],
                'quantized': []
            },
            'accuracy': [],
            'global_steps': []
        }

    # Methods to register metrics
    def register_loss(self, loss_type, step, value):
        if loss_type in self.metrics['losses']:
            self.metrics['losses'][loss_type].append((step, value))
        else:
            self._log.warning(f"Unknown loss type: {loss_type}")

    def register_accuracy(self, step, value):
        self.metrics['accuracy'].append((step, value))

    def register_model_size(self, size_type, step, value):
        if size_type in self.metrics['model_sizes']:
            self.metrics['model_sizes'][size_type].append((step, value))
        else:
            self._log.warning(f"Unknown model size type: {size_type}")

    def register_staleness(self, update_number, value, timestamp=None):
        """
        Registers staleness along with update number and timestamp.

        Args:
            update_number (int): The update number.
            value (float): The staleness value.
            timestamp (float, optional): The time when staleness was recorded. If None, uses current time.
        """
        if timestamp is None:
            timestamp = time.time()  # Get current time as a UNIX timestamp
        self.metrics['staleness'].append((update_number, value, timestamp))

    def register_global_step(self, step):
        self.metrics['global_steps'].append(step)

    # Flush method to write metrics and generate plots
    def flush(self):
        self.write_metrics_to_tensorboard()
        self.generate_plots()
        self.close()

    # Methods to write metrics to TensorBoard
    def write_metrics_to_tensorboard(self):
        # Write scalar metrics
        for step, loss in self.metrics['losses']['reconstruction']:
            self.writer.add_scalar('Loss/Reconstruction', loss, step)

        for step, loss in self.metrics['losses']['classification']:
            self.writer.add_scalar('Loss/Classification', loss, step)

        for step, accuracy in self.metrics['accuracy']:
            self.writer.add_scalar('Accuracy', accuracy, step)

        for step, size in self.metrics['model_sizes']['non_quantized']:
            self.writer.add_scalar('Model/Size_NonQuantized', size, step)

        for step, size in self.metrics['model_sizes']['quantized']:
            self.writer.add_scalar('Model/Size_Quantized', size, step)

        # Unpack three values: update_number, staleness, timestamp
        for update_number, staleness, _ in self.metrics['staleness']:
            self.writer.add_scalar('Model/Staleness', staleness, update_number)

        # Optionally, write staleness over timestamp as well
        for _, staleness, timestamp in self.metrics['staleness']:
            self.writer.add_scalar('Model/Staleness_Timestamp', staleness, timestamp)

            self._log.info(f"Client {self.client_id}: Metrics have been written to TensorBoard logs.")

    # Method to generate plots
    def generate_plots(self):
        """
        Generates and saves plots for losses, accuracy, and staleness.
        """
        output_dir = self.output_dir
        client_id = self.client_id

        # Plot Reconstruction Loss
        if self.metrics['losses']['reconstruction']:
            steps, losses = zip(*self.metrics['losses']['reconstruction'])
            self.plot_loss(
                steps=steps,
                loss_values=losses,
                output_dir=output_dir,
                title=f'Client {client_id} - Reconstruction Loss over Training Cycles',
                filename='reconstruction_loss_plot.png'
            )

        # Plot Classification Loss
        if self.metrics['losses']['classification']:
            steps, losses = zip(*self.metrics['losses']['classification'])
            self.plot_loss(
                steps=steps,
                loss_values=losses,
                output_dir=output_dir,
                title=f'Client {client_id} - Classification Loss over Training Cycles',
                filename='classification_loss_plot.png'
            )

        # Plot Accuracy
        if self.metrics['accuracy']:
            steps, accuracies = zip(*self.metrics['accuracy'])
            self.plot_accuracy(
                steps=steps,
                accuracy_values=accuracies,
                output_dir=output_dir,
                title=f'Client {client_id} - Accuracy over Training Cycles',
                filename='accuracy_plot.png'
            )

        # Plot Staleness over Time
        if self.metrics['staleness']:
            updates, staleness_values, timestamps = zip(*self.metrics['staleness'])
            self.plot_staleness_over_time(
                timestamps=timestamps,
                staleness_values=staleness_values,
                output_dir=output_dir,
                client_id=client_id
            )

    def close(self):
        self.writer.close()

    # Plotting methods
    def plot_loss(self, steps, loss_values, output_dir, title='Loss over Training Cycles', filename='loss_plot.png'):
        plt.figure()
        plt.plot(steps, loss_values, marker='o')
        plt.title(title, fontsize=14)
        plt.xlabel('Training Cycle', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def plot_accuracy(self, steps, accuracy_values, output_dir, title='Accuracy over Training Cycles', filename='accuracy_plot.png'):
        plt.figure()
        plt.plot(steps, accuracy_values, marker='o')
        plt.title(title, fontsize=14)
        plt.xlabel('Training Cycle', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def plot_staleness_over_time(self, timestamps, staleness_values, output_dir, client_id):
        """
        Plots staleness over elapsed time since the simulation started.

        Args:
            timestamps (list of float): The elapsed times when staleness was recorded.
            staleness_values (list of float): The staleness values.
            output_dir (str): Directory to save the plot.
            client_id (int): The client ID.
        """
        # Convert timestamps (elapsed time in seconds) to a numpy array for plotting
        times = np.array(timestamps)

        plt.figure()
        plt.plot(times, staleness_values, marker='o')
        plt.title(f'Staleness Over Time for Client {client_id}', fontsize=14)
        plt.xlabel('Elapsed Time (seconds)', fontsize=14)
        plt.ylabel('Staleness', fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'staleness_over_time_{client_id}.png'))
        plt.close()