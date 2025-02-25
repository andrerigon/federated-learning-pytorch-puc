import os
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time 
import datetime
import numpy as np
import torch
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import torch.nn.functional as F
from io import BytesIO
import threading

class MetricsLogger:

    def __init__(self, client_id, output_dir='runs', generate_visualizations=False):
        """
        Initialize the metrics logger.

        Args:
            client_id (int): The identifier for the client.
            output_dir (str, optional): Directory to save logs and outputs. Defaults to 'runs'.
            generate_visualizations (bool, optional): Whether to generate visualizations and graphs. Defaults to False.
        """
        self.client_id = client_id
        self.output_dir = os.path.join(output_dir, f"client_{client_id}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self._log = logging.getLogger(__name__)
        self.generate_visualizations = generate_visualizations

        # Store raw data for delayed visualization
        self.raw_data = {
            'mse_values': [],  # Store (step, values) tuples
            'confusion_matrices': [],  # Store (step, matrix) tuples
            'embeddings_data': []  # Store (step, embeddings, labels) tuples
        }

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
            'global_steps': [],
            'training_times': [],  # Track training cycle times
            'predictions': [],     # Store predictions for confusion matrix
            'embeddings': [],      # Store embeddings for TSNE
            'true_labels': [],     # Store true labels
            'mse_per_sample': []   # Store MSE values per sample
        }

        # Define custom layout for TensorBoard
        layout = {
            "Model Stats": {
                "sizes": ["Multiline", ["Model/Size_NonQuantized", 
                                   "Model/Size_Quantized", 
                                   "Model/Staleness"]],
            },
            "Training Stats": {
                "losses": ["Multiline", ["Loss/Reconstruction",
                                    "Loss/Classification",
                                    "Accuracy"]],
                "performance": ["Margin", ["Performance/Training_Time",
                                     "AI_Metrics/MSE_Mean",
                                     "AI_Metrics/Batch_Accuracy"]]
            },
            "Visualizations": {
                "plots": ["Images", ["AI_Metrics/Confusion_Matrix",
                               "AI_Metrics/TSNE",
                               "AI_Metrics/MSE_Distribution"]]
            }
        }

        try:
            self.writer.add_custom_scalars(layout)
        except Exception as e:
            self._log.warning(f"Failed to add custom scalar layout: {str(e)}")
     

    def register_loss(self, loss_type, step, value):
        """Register a loss value for the specified type."""
        if loss_type in self.metrics['losses']:
            self.metrics['losses'][loss_type].append((step, value))
        else:
            self._log.warning(f"Unknown loss type: {loss_type}")

    def register_accuracy(self, step, value):
        """Register an accuracy value."""
        self.metrics['accuracy'].append((step, value))

    def register_model_size(self, size_type, step, value):
        """Register a model size value."""
        if size_type in self.metrics['model_sizes']:
            self.metrics['model_sizes'][size_type].append((step, value))
        else:
            self._log.warning(f"Unknown model size type: {size_type}")

    def register_staleness(self, update_number, value, timestamp=None):
        """
        Register staleness with timestamp.

        Args:
            update_number (int): The update number.
            value (float): The staleness value.
            timestamp (float, optional): The timestamp. If None, uses current time.
        """
        if timestamp is None:
            timestamp = time.time()
        self.metrics['staleness'].append((update_number, value, timestamp))

    def log_tsne_visualization(self, step):
        """Create and log TSNE visualization for the current step."""
        try:
            if len(self.raw_data['embeddings_data']) == 0:
                self._log.warning("No embeddings data available for TSNE visualization")
                return

            # Get the latest embeddings and labels
            _, embeddings, labels = self.raw_data['embeddings_data'][-1]
            
            # Perform TSNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create and use the figure
            plt.ioff()  # Turn off interactive mode
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab10')
            fig.colorbar(scatter)
            ax.set_title(f'TSNE Visualization - Step {step}')
            
            # Save to TensorBoard using BytesIO to avoid GUI
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            self.writer.add_image('AI_Metrics/TSNE', 
                                plt.imread(buf), 
                                step, 
                                dataformats='HWC')
            plt.close(fig)  # Explicitly close the figure
            buf.close()

        except Exception as e:
            self._log.error(f"Error in TSNE visualization: {str(e)}", exc_info=True)
            
    def register_global_step(self, step):
        """Register a global step."""
        self.metrics['global_steps'].append(step)

    def register_training_time(self, step, duration):
        """Register the time taken for a training cycle."""
        self.metrics['training_times'].append((step, duration))
        self.writer.add_scalar('Performance/Training_Time', duration, step)

    def register_mse_per_sample(self, step, outputs, targets):
        """Register MSE values per sample."""
        try:
            mse = F.mse_loss(outputs, targets, reduction='none').mean(dim=[1,2,3])
            
            # Always log the scalar summary
            self.writer.add_scalar('AI_Metrics/MSE_Mean', mse.mean().item(), step)
            
            # Only store raw data if visualizations are enabled
            if self.generate_visualizations:
                self.raw_data['mse_values'].append((step, mse.detach().cpu().numpy()))
                
        except Exception as e:
            self._log.error(f"Error in register_mse_per_sample: {str(e)}", exc_info=True)

    def register_predictions(self, step, predictions, true_labels, embeddings=None):
        """Register predictions for final state visualization."""
        try:
            # Store predictions and labels for confusion matrix
            predictions_np = predictions.cpu().numpy()
            labels_np = true_labels.cpu().numpy()
            
            # Store only latest embeddings if provided
            if embeddings is not None:
                self.raw_data['embeddings_data'] = [(
                    step, 
                    embeddings.cpu().numpy(),
                    labels_np
                )]
                
            # Update confusion matrix
            cm = confusion_matrix(labels_np, predictions_np)
            self.raw_data['confusion_matrices'] = [(step, cm)]

            # Add scalar metrics
            accuracy = (predictions == true_labels).float().mean().item()
            self.writer.add_scalar('AI_Metrics/Batch_Accuracy', accuracy, step)
        except Exception as e:
            self._log.error(f"Error in register_predictions: {str(e)}", exc_info=True)

    def generate_visualizations(self):
        """Generate all visualizations in the main thread during flush."""
        if not self.generate_visualizations:
            self._log.info("Visualizations generation disabled - skipping")
            return
            
        try:
            if not threading.current_thread() is threading.main_thread():
                self._log.warning("Attempting to generate visualizations outside main thread - deferring to flush()")
                return

            # Generate MSE distribution plots
            for step, mse_values in self.raw_data['mse_values']:
                fig = plt.figure(figsize=(10, 6))
                plt.hist(mse_values, bins=50)
                plt.title(f'MSE Distribution - Step {step}')
                plt.xlabel('MSE Value')
                plt.ylabel('Count')
                
                # Save to TensorBoard using BytesIO to avoid GUI
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                self.writer.add_image(f'AI_Metrics/MSE_Distribution/{step}', 
                                    plt.imread(buf), 
                                    step, 
                                    dataformats='HWC')
                plt.close()
                buf.close()

            # Generate confusion matrix plots
            for step, cm in self.raw_data['confusion_matrices']:
                fig = plt.figure(figsize=(10, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - Step {step}')
                
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                self.writer.add_image(f'AI_Metrics/Confusion_Matrix/{step}', 
                                    plt.imread(buf), 
                                    step, 
                                    dataformats='HWC')
                plt.close()
                buf.close()

            # Generate TSNE visualizations
            for step, embeddings, labels in self.raw_data['embeddings_data']:
                tsne = TSNE(n_components=2, random_state=42)
                embeddings_2d = tsne.fit_transform(embeddings)
                
                fig = plt.figure(figsize=(10, 10))
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=labels, cmap='tab10')
                plt.colorbar(scatter)
                plt.title(f'TSNE Visualization - Step {step}')
                
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                self.writer.add_image(f'AI_Metrics/TSNE/{step}', 
                                    plt.imread(buf), 
                                    step, 
                                    dataformats='HWC')
                plt.close()
                buf.close()

        except Exception as e:
            self._log.error(f"Error generating visualizations: {str(e)}", exc_info=True)

    def write_metrics_to_tensorboard(self):
        """Write all metrics to TensorBoard."""
        try:
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

            for update_number, staleness, _ in self.metrics['staleness']:
                self.writer.add_scalar('Model/Staleness', staleness, update_number)

            for _, staleness, timestamp in self.metrics['staleness']:
                self.writer.add_scalar('Model/Staleness_Timestamp', staleness, timestamp)

            for step, duration in self.metrics['training_times']:
                self.writer.add_scalar('Performance/Training_Time', duration, step)

            self._log.info(f"Client {self.client_id}: Metrics have been written to TensorBoard logs.")
        except Exception as e:
            self._log.error(f"Error writing metrics to TensorBoard: {str(e)}", exc_info=True)

    def generate_plots(self):
        """Generate all visualization plots."""
        if not self.generate_visualizations:
            self._log.info("Plot generation disabled - skipping")
            return
            
        if not threading.current_thread() is threading.main_thread():
            self._log.warning("Skipping plot generation in non-main thread")
            return
            
        try:
            output_dir = self.output_dir
            client_id = self.client_id

             # Plot final MSE distribution
            if self.raw_data['mse_values']:
                step, mse_values = self.raw_data['mse_values'][-1]  # Get the latest one
                self.plot_mse_distribution(step, mse_values, output_dir)


            # Plot final confusion matrix
            if self.raw_data['confusion_matrices']:
                step, cm = self.raw_data['confusion_matrices'][-1]  # Get the latest one
                self.plot_confusion_matrix(step, cm, output_dir)

             # Plot final TSNE visualization
            if self.raw_data['embeddings_data']:
                step, embeddings, labels = self.raw_data['embeddings_data'][-1]
                self.plot_tsne(step, embeddings, labels, output_dir)


            # Plot losses
            if self.metrics['losses']['reconstruction']:
                steps, losses = zip(*self.metrics['losses']['reconstruction'])
                self.plot_loss(
                    steps=steps,
                    loss_values=losses,
                    output_dir=output_dir,
                    title=f'Client {client_id} - Reconstruction Loss over Training Cycles',
                    filename='reconstruction_loss_plot.png'
                )

            if self.metrics['losses']['classification']:
                steps, losses = zip(*self.metrics['losses']['classification'])
                self.plot_loss(
                    steps=steps,
                    loss_values=losses,
                    output_dir=output_dir,
                    title=f'Client {client_id} - Classification Loss over Training Cycles',
                    filename='classification_loss_plot.png'
                )

            # Plot accuracy
            if self.metrics['accuracy']:
                steps, accuracies = zip(*self.metrics['accuracy'])
                self.plot_accuracy(
                    steps=steps,
                    accuracy_values=accuracies,
                    output_dir=output_dir,
                    title=f'Client {client_id} - Accuracy over Training Cycles',
                    filename='accuracy_plot.png'
                )

            # Plot staleness
            if self.metrics['staleness']:
                updates, staleness_values, timestamps = zip(*self.metrics['staleness'])
                self.plot_staleness_over_time(
                    timestamps=timestamps,
                    staleness_values=staleness_values,
                    output_dir=output_dir,
                    client_id=client_id
                )

            # Plot training times
            if self.metrics['training_times']:
                steps, times = zip(*self.metrics['training_times'])
                self.plot_training_times(
                    steps=steps,
                    times=times,
                    output_dir=output_dir,
                    title=f'Client {client_id} - Training Time per Cycle',
                    filename='training_times_plot.png'
                )

            self._log.info(f"Client {client_id}: All plots have been generated")

        except Exception as e:
            self._log.error(f"Error generating plots: {str(e)}", exc_info=True)

    def plot_tsne(self, step, embeddings, labels, output_dir):
        """Plot TSNE visualization with proper labeling."""
        try:
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            plt.figure(figsize=(10, 10))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=labels, cmap='tab10')
            plt.colorbar(scatter)
            plt.title(f'Final TSNE Visualization\nStep {step}')
            plt.xlabel('TSNE Dimension 1')
            plt.ylabel('TSNE Dimension 2')
            plt.grid(True)
            
            # Remove client_id from the path as it's already included in output_dir
            output_path = os.path.join(output_dir, 'tsne_visualization.png')
            plt.savefig(output_path)
            plt.close()

            self._log.info(f"TSNE visualization saved to {output_path}")
        except Exception as e:
            self._log.error(f"Error in TSNE visualization: {str(e)}", exc_info=True)
  
    def plot_mse_distribution(self, step, mse_values, output_dir):
        """Plot MSE distribution with proper labeling."""
        plt.figure(figsize=(10, 6))
        plt.hist(mse_values, bins=50, edgecolor='black')
        plt.title(f'Final MSE Distribution\nStep {step}')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'mse_distribution.png'))
        plt.close()

    def plot_confusion_matrix(self, step, cm, output_dir):
        """Plot a single confusion matrix with proper labeling."""
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Final Confusion Matrix\nStep {step}')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix.png'))
        plt.close()

    def plot_training_times(self, steps, times, output_dir, title, filename):
        """Plot training times."""
        plt.figure()
        plt.plot(steps, times, marker='o')
        plt.title(title, fontsize=14)
        plt.xlabel('Training Cycle', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def plot_loss(self, steps, loss_values, output_dir, title='Loss over Training Cycles', filename='loss_plot.png'):
        """Plot loss values."""
        plt.figure()
        plt.plot(steps, loss_values, marker='o')
        plt.title(title, fontsize=14)
        plt.xlabel('Training Cycle', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def plot_accuracy(self, steps, accuracy_values, output_dir, title='Accuracy over Training Cycles', filename='accuracy_plot.png'):
        """Plot accuracy values."""
        plt.figure()
        plt.plot(steps, accuracy_values, marker='o')
        plt.title(title, fontsize=14)
        plt.xlabel('Training Cycle', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def plot_staleness_over_time(self, timestamps, staleness_values, output_dir, client_id):
        """Plot staleness over time."""
        times = np.array(timestamps)
        plt.figure()
        plt.plot(times, staleness_values, marker='o')
        plt.title(f'Staleness Over Time for Client {client_id}', fontsize=14)
        plt.xlabel('Elapsed Time (seconds)', fontsize=14)
        plt.ylabel('Staleness', fontsize=14)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'staleness_over_time_{client_id}.png'))
        plt.close()

    def flush(self):
        """Write all metrics and generate all plots."""
        if not self.generate_visualizations:
            self._log.info("Visualizations generation disabled - skipping")
            return  
        
        print("\n\nWriting metrics to tensor\n\n")
        self.write_metrics_to_tensorboard()
        
        print("\n\nGenerating visualization\n\n")
        self.generate_visualizations()
        print("\n\nGenerating plots\n\n")
        self.generate_plots()
        print("\n\nDone\n\n")

        self.close()

    def close(self):
        """Close the TensorBoard writer."""
        try:
            self.writer.close()
        except Exception as e:
            self._log.error(f"Error closing TensorBoard writer: {str(e)}", exc_info=True)
