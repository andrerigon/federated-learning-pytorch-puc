import threading
import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any, List, OrderedDict, TypeVar, Optional, Union
from torch.utils.tensorboard import SummaryWriter
import torch.quantization as quant
from io import BytesIO
import gc
import numpy as np
import os
from datetime import datetime
import json
import pandas as pd
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix, silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F

from model_manager import ModelManager
from dataset_loader import DatasetLoader
from metrics_logger import MetricsLogger
from aggregation_strategy import AggregationStrategy, FedAvgStrategy, AsyncFedAvgStrategy, SAFAStrategy
from image_classifier_autoencoders import device as ae_device
from uav_metrics_logger import plot_clustering_metrics, plot_confusion_matrix, plot_mse, plot_tsne

class ConvergenceCriteria:
    def has_converged(self, metrics_history: List[Dict[str, float]]) -> bool:
        raise NotImplementedError()

class FederatedLearningAggregator:
    """
    A federated aggregator that:
    - Uses a given aggregation strategy (FedAvg, AsyncFedAvg, etc.).
    - Receives client model updates, handles staleness recording, and aggregates.
    - Uses a time-based interval for FedAvg aggregation instead of a fixed client count.
    - Periodically evaluates the global model in a background thread until convergence.
    - Logs metrics to TensorBoard for analysis.
    """

    def __init__(
        self,
        id: int,
        model_manager: ModelManager,
        dataset_loader: DatasetLoader,
        metrics_logger: MetricsLogger,
        strategy: AggregationStrategy,
        convergence_criteria: ConvergenceCriteria,
        output_dir: str = './',
        round_interval: float = 60.0,
        device=None
    ):
        """
        Args:
            id (int): Identifier for this aggregator (e.g., a server or UAV).
            model_manager (ModelManager): Handles model creation/loading.
            dataset_loader (DatasetLoader): Loads dataset for evaluation.
            metrics_logger (MetricsLogger): Handles custom metrics logging.
            strategy (AggregationStrategy): The federated aggregation strategy to use.
            convergence_criteria (ConvergenceCriteria): Criteria for determining model convergence.
            output_dir (str): Directory for logs and outputs.
            round_interval (float): Time-based interval (in seconds) for FedAvg rounds.
            device (torch.device): Device to run on (CPU or GPU).
        """
        self.id = id
        self.model_manager = model_manager
        self.dataset_loader = dataset_loader
        self.metrics_logger = metrics_logger
        self.strategy = strategy
        self.convergence_criteria = convergence_criteria
        self.output_dir = output_dir
        self.round_interval = round_interval
        self.converged = False
        self.start_time = time.time()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.global_model = self.model_manager.load_model().to(self.device)
        self.global_model_version = 0
        self.model_updates = 0
        self._log = logging.getLogger(__name__)

        # For FedAvg: buffer incoming client updates until time triggers aggregation
        self.client_updates_buffer: List[OrderedDict] = []
        self.last_aggregation_time = time.time()

        # For evaluation and convergence checking
        self.metrics_history: List[Dict[str, float]] = []
        self.stop_evaluation_flag = False
        self.evaluation_thread = threading.Thread(target=self.evaluate_global_model_periodically)
        self.evaluation_thread.daemon = True

        # Strategy-specific paths and metrics
        self.strategy_name = strategy.__class__.__name__
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_path = os.path.join(
            output_dir, 
            'aggregator_metrics',
            self.strategy_name,
            f'run_{self.run_timestamp}'
        )
        os.makedirs(self.metrics_path, exist_ok=True)

        # TensorBoard writer
        self.tb_writer = SummaryWriter(
            log_dir=os.path.join(output_dir, 'tensorboard', 
                               f'aggregator_{self.id}',
                               self.strategy_name,
                               f'run_{self.run_timestamp}')
        )

        # Initialize metrics storage
        self.communication_stats = {
            'total_messages': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'bandwidth_used': 0,
            'message_latencies': [],
            'update_timestamps': []
        }
        
        self.staleness_stats = {
            'values': [],
            'timestamps': [],
            'by_client': {},
            'max_staleness': 0,
            'total_staleness': 0
        }
        
        self.performance_metrics = {
            'convergence_time': None,
            'peak_accuracy': 0,
            'final_accuracy': None,
            'training_stability': [],
            'learning_rates': [],
            'model_sizes': []
        }

        # Load testset for evaluation
        self.testset = self.dataset_loader.testset

    def start(self):
        """Start the evaluation thread."""
        self.evaluation_thread.start()

    def receive_model_update(self, sender_id: str, client_dict: OrderedDict, local_model_version: int):
        """Called when a client model update arrives."""
        # Calculate staleness
        staleness = self.global_model_version - local_model_version
        
        # Record metrics
        message_time = time.time()
        self.communication_stats['total_messages'] += 1
        self.communication_stats['update_timestamps'].append(message_time)
        self.record_staleness(sender_id, staleness, message_time)
        
        # Log to TensorBoard
        self.tb_writer.add_scalar('Staleness/Value', staleness, self.model_updates + 1)

        if isinstance(self.strategy, (FedAvgStrategy, SAFAStrategy)):
            # For strategies that expect a list of updates
            self.client_updates_buffer.append(client_dict)
            self.check_and_aggregate_fedavg()
        else:
            # For strategies that work with single updates
            updated_state = self.strategy.aggregate(
                self.global_model,
                client_dict,
                extra_info={'staleness': staleness}
            )
            self.update_global_model(updated_state)
            self.communication_stats['successful_updates'] += 1

    def is_fedavg(self) -> bool:
        """Check if current strategy is FedAvg or SAFA."""
        return isinstance(self.strategy, (FedAvgStrategy, SAFAStrategy))

    def check_and_aggregate_fedavg(self):
        """Check if it's time to aggregate for FedAvg/SAFA strategies."""
        current_time = time.time()
        if (current_time - self.last_aggregation_time) >= self.round_interval:
            print("doing fedav/safa aggreg")    
            updated_state = self.strategy.aggregate(self.global_model, self.client_updates_buffer)
            self.update_global_model(updated_state)
            self.client_updates_buffer = []
            self.last_aggregation_time = current_time
        else: 
            print("skipping fedav/safa aggreg")

    def update_global_model(self, state: OrderedDict, new_version: int = None):
        """Updates the global model with a new aggregated state dict."""
        if new_version is None:
            new_version = self.global_model_version + 1

        self.global_model.load_state_dict(state)
        self.global_model_version = new_version
        self.model_updates += 1
        self.global_model.eval()
        self.global_model = torch.quantization.convert(self.global_model)
        self._log.info(f"Aggregator {self.id}: Updated global model to version {new_version}")

    def record_staleness(self, sender_id: str, staleness: int, timestamp: float):
        """Record detailed staleness metrics."""
        self.staleness_stats['values'].append(staleness)
        self.staleness_stats['timestamps'].append(timestamp)
        self.staleness_stats['max_staleness'] = max(self.staleness_stats['max_staleness'], staleness)
        self.staleness_stats['total_staleness'] += staleness
        
        if sender_id not in self.staleness_stats['by_client']:
            self.staleness_stats['by_client'][sender_id] = []
        self.staleness_stats['by_client'][sender_id].append((timestamp, staleness))

    def state_dict(self):
        return self.global_model.state_dict()

    def eval(self, images):
        return self.global_model(images)

    def evaluate_global_model_periodically(self):
        """Periodically evaluate the global model until convergence."""
        while not self.stop_evaluation_flag:
            metrics = self.evaluate_global_model()
            self.metrics_history.append(metrics)
            
            # Log metrics
            step = len(self.metrics_history)
            for name, value in metrics.items():
                self.tb_writer.add_scalar(f'Evaluation/{name}', value, step)
            
            if self.convergence_criteria.has_converged(self.metrics_history):
                print("\n\nConverged!\n\n")
                self.mode_converged_callback()
                break
            time.sleep(5)

    def evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate the global model on the testset."""
        testloader = DataLoader(self.testset, batch_size=4, shuffle=False, num_workers=2)
        criterion_reconstruction = nn.MSELoss()
        criterion_classification = nn.CrossEntropyLoss()
        total_reconstruction_loss = 0
        total_classification_loss = 0
        all_labels = []
        all_predictions = []

        self.global_model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(ae_device), data[1].to(ae_device)
                decoded, classified = self.global_model(images)
                reconstruction_loss = criterion_reconstruction(decoded, images)
                classification_loss = criterion_classification(classified, labels)
                total_reconstruction_loss += reconstruction_loss.item()
                total_classification_loss += classification_loss.item()
                _, predicted = torch.max(classified.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        mean_reconstruction_loss = total_reconstruction_loss / len(testloader)
        mean_classification_loss = total_classification_loss / len(testloader)
        accuracy = 100 * accuracy_score(all_labels, all_predictions)

        metrics = {
            'loss': mean_reconstruction_loss,
            'classification_loss': mean_classification_loss,
            'accuracy': accuracy
        }

        # Update performance tracking
        self.performance_metrics['peak_accuracy'] = max(
            self.performance_metrics['peak_accuracy'], 
            accuracy
        )
        self.performance_metrics['training_stability'].append(mean_reconstruction_loss)

        del testloader
        return metrics

    def mode_converged_callback(self):
        """Called when the global model has converged."""
        self._log.info("[Aggregator] Global model converged.")
        step = len(self.metrics_history)
        self.converged = True
        self.tb_writer.add_text('Convergence', f'Model converged at step {step}', step)

    def save_final_metrics(self):
        """Save comprehensive final metrics."""
        avg_staleness = (np.mean(self.staleness_stats['values']) 
                        if self.staleness_stats['values'] else 0)
        success_rate = (self.communication_stats['successful_updates'] / 
                       self.communication_stats['total_messages']
                       if self.communication_stats['total_messages'] > 0 else 0)
        
        metrics_summary = {
            'strategy': self.strategy_name,
            'run_timestamp': self.run_timestamp,
            'convergence_time': time.time() - self.start_time if self.converged else None,
            'total_updates': self.model_updates,
            'performance': {
                'final_accuracy': self.metrics_history[-1]['accuracy'] if self.metrics_history else None,
                'peak_accuracy': self.performance_metrics['peak_accuracy'],
                'final_loss': self.metrics_history[-1]['loss'] if self.metrics_history else None,
                'training_stability': self.performance_metrics['training_stability']
            },
            'communication': {
                'total_messages': self.communication_stats['total_messages'],
                'successful_updates': self.communication_stats['successful_updates'],
                'success_rate': success_rate,
                'average_model_size': np.mean(self.performance_metrics['model_sizes'])
            },
            'staleness': {
                'average': avg_staleness,
                'maximum': self.staleness_stats['max_staleness'],
                'by_client': {
                    client: {
                        'average': np.mean([s for _, s in values]),
                        'max': max(s for _, s in values)
                    }
                    for client, values in self.staleness_stats['by_client'].items()
                }
            }
        }
        
        # Save metrics to JSON
        metrics_file = os.path.join(self.metrics_path, 'final_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        
        # Save model
        model_file = os.path.join(self.metrics_path, 'final_model.pth')
        torch.save(self.global_model.state_dict(), model_file)
        
        # Save learning curves
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(os.path.join(self.metrics_path, 'learning_curves.csv'))
        
        # Log final hparams to tensorboard
        self.tb_writer.add_hparams(
            {
                'strategy': self.strategy_name,
                'round_interval': self.round_interval
            },
            {
                'hparam/final_accuracy': metrics_summary['performance']['final_accuracy'],
                'hparam/convergence_time': metrics_summary['convergence_time'],
                'hparam/total_updates': metrics_summary['total_updates'],
                'hparam/avg_staleness': avg_staleness,
                'hparam/success_rate': success_rate
            }
        )

    def stop(self):
        """Stop evaluation, cleanup resources, and finalize."""
        self.save_final_metrics()
        self.stop_evaluation_flag = True
        if self.evaluation_thread.is_alive():
            self.evaluation_thread.join()

        torch.cuda.empty_cache()
        self.global_model.cpu()
        del self.global_model
        gc.collect()

        self._log.info(f"Aggregator {self.id}: Stopped and cleaned up.")
        
        # Close TensorBoard writer
        self.tb_writer.close()