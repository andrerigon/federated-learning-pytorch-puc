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
from loguru import logger

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
        device=None,
        client_count: int = 10
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
        self.client_count = client_count
        self.client_versions = {}
        self.tracked_variables = {}

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.global_model = self.model_manager.load_model().to(self.device)
        self.global_model_version = 0
        self.model_updates = 0
        self.logger = logger.bind(source="uav", uav_id=self.id)

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

        self.received_client_ids = set()

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
        self.logger.info(f"Aggregator Started for {self.client_count} clients")

    def start(self):
        """Start the evaluation thread."""
        self.evaluation_thread.start()

    def tracked_vars(self): 
        self.tracked_variables['aggregation_clients_count'] = len(self.received_client_ids)
        self.tracked_variables['accuracy_checks'] = len(self.metrics_history)
        self.tracked_variables['strategy'] = self.strategy_name
        return self.tracked_variables

    def have_all_updates(self):
        """
        Check if we've received updates from 10 distinct client IDs.
        """
        self.logger.info(f"received_client_ids with size {len(self.received_client_ids)} ")
        return len(self.received_client_ids) == self.client_count  

    def receive_model_update(self, sender_id: str, client_dict: OrderedDict, local_model_version: int, extra_info: Dict = {}):
        """Called when a client model update arrives."""
        # Calculate staleness
        staleness = self.global_model_version - local_model_version

        current_version = self.client_versions.get(sender_id, -1)

        if current_version == local_model_version:
            self.logger.info(f"Client [{sender_id}] already sent version {local_model_version}")
            return 
        
        self.client_versions[sender_id] = local_model_version
        self.received_client_ids.add(sender_id)

        # Record metrics
        message_time = time.time()
        self.communication_stats['total_messages'] += 1
        self.communication_stats['update_timestamps'].append(message_time)
        self.record_staleness(sender_id, staleness, message_time)
        
        # Log to TensorBoard
        self.tb_writer.add_scalar('Staleness/Value', staleness, self.model_updates + 1)

        if isinstance(self.strategy, (FedAvgStrategy)):
            # For strategies that expect a list of updates
            self.client_updates_buffer.append(client_dict)
            self.check_and_aggregate_fedavg()
        else:
            # For strategies that work with single updates
            self.logger.info(f"Aggregating partialy for {self.strategy.__class__.__name__}")
            updated_state = self.strategy.aggregate(
                self.global_model,
                client_dict,
                extra_info = extra_info | {'staleness': staleness, 'client_id': sender_id}
            )
            self.update_global_model(updated_state)
            self.communication_stats['successful_updates'] += 1

    def is_fedavg(self) -> bool:
        """Check if current strategy is FedAvg or SAFA."""
        return isinstance(self.strategy, (FedAvgStrategy))

    def check_and_aggregate_fedavg(self):
        """
        Check if it's time to aggregate for FedAvg or other strategies.
        If it's FedAvg, also verify that we have updates from all clients before aggregating.
        """
        current_time = time.time()
            
            # Check if strategy is FedAvg
        if self.is_fedavg():
            # Example: a helper method that checks if we have updates from all clients
            self.logger.info("If fedavg")
            if self.have_all_updates():
                self.logger.info("Aggregating FedAvg model (all clients' updates present).")
                updated_state = self.strategy.aggregate(self.global_model, self.client_updates_buffer)
                self.update_global_model(updated_state)
                # Reset buffer and timer
                self.client_updates_buffer = []
                self.last_aggregation_time = current_time
                self.received_client_ids = set()
            else:
                self.logger.info("Skipping FedAvg aggregation: not all clients have submitted updates yet.")
            # if isinstance(self.strategy, (SAFAStrategy)) and (current_time - self.last_aggregation_time) < self.round_interval:
            #     self.logger.info("Skipping aggregation: round interval not reached yet.")
            #     return 
            # else:
            #     # Handle other strategies (e.g. SAFA) the same as before
            #     self.logger.info("Doing non-FedAvg aggregation (e.g., SAFA).")
            #     updated_state = self.strategy.aggregate(self.global_model, self.client_updates_buffer)
            #     self.update_global_model(updated_state)
            #     self.client_updates_buffer = []
            #     self.last_aggregation_time = current_time    
        
       
            
                
    # def check_and_aggregate_fedavg(self):
    #     """Check if it's time to aggregate for FedAvg/SAFA strategies."""
    #     current_time = time.time()
    #     if (current_time - self.last_aggregation_time) >= self.round_interval:
    #         self.logger.info("doing fedav/safa aggreg")    
    #         updated_state = self.strategy.aggregate(self.global_model, self.client_updates_buffer)
    #         self.update_global_model(updated_state)
    #         self.client_updates_buffer = []
    #         self.last_aggregation_time = current_time
    #     else: 
    #         self.logger.info("skipping fedav/safa aggreg")

    def update_global_model(self, state: OrderedDict, new_version: int = None):
        """Updates the global model with a new aggregated state dict."""
        if new_version is None:
            new_version = self.global_model_version + 1

        self.global_model.load_state_dict(state)
        self.global_model_version = new_version
        self.model_updates += 1
        self.global_model.eval()
        self.global_model = torch.quantization.convert(self.global_model)
        self.logger.info(f"Updated global model to version {new_version}")

    def record_staleness(self, sender_id: str, staleness: int, timestamp: float):
        self.staleness_stats['values'].append(staleness)
        self.staleness_stats['timestamps'].append(timestamp)
        self.staleness_stats['max_staleness'] = max(self.staleness_stats['max_staleness'], staleness)
        self.staleness_stats['total_staleness'] += staleness

        if sender_id not in self.staleness_stats['by_client']:
            self.staleness_stats['by_client'][sender_id] = {'staleness': [], 'timestamps': []}
        self.staleness_stats['by_client'][sender_id]['staleness'].append(staleness)
        self.staleness_stats['by_client'][sender_id]['timestamps'].append(timestamp)

    def state_dict(self):
        return self.global_model.state_dict()

    def eval(self, images):
        return self.global_model(images)

    def evaluate_global_model_periodically(self):
        """Periodically evaluate the global model until convergence."""
        last_checked_version = -1
        while not self.stop_evaluation_flag:
            if self.global_model_version <= last_checked_version:
                time.sleep(5)
                continue 
            
            self.logger.info(f"Convergency check for version {self.global_model_version}")
            last_checked_version = self.global_model_version
            metrics = self.evaluate_global_model()
            self.metrics_history.append(metrics)
            
            # Log metrics
            step = len(self.metrics_history)
            for name, value in metrics.items():
                self.tb_writer.add_scalar(f'Evaluation/{name}', value, step)
            
            if self.convergence_criteria.has_converged(self.metrics_history):
                self.logger.info("\n\nConverged!\n\n")
                self.mode_converged_callback()
                break
            time.sleep(5)

    def evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate the global model on the testset."""
        testloader = DataLoader(self.testset, batch_size=4, shuffle=False, num_workers=0)
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

        self.tracked_variables['loss'] = mean_reconstruction_loss
        self.tracked_variables['classification_loss'] = mean_classification_loss
        self.tracked_variables['accuracy'] = accuracy

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
        self.logger.info("Global model converged.")
        step = len(self.metrics_history)
        self.converged = True
        self.tb_writer.add_text('Convergence', f'Model converged at step {step}', step)

    def save_final_metrics(self):
        """Save comprehensive final metrics."""
        try:
            avg_staleness = (
                np.mean(self.staleness_stats['values']) 
                if self.staleness_stats['values'] else 0.0
            )
            success_rate = (
                self.communication_stats['successful_updates'] / self.communication_stats['total_messages']
                if self.communication_stats['total_messages'] > 0 else 0.0
            )

            metrics_summary = {
                'strategy': self.strategy_name,
                'run_timestamp': self.run_timestamp,
                'convergence_time': time.time() - self.start_time if self.converged else None,
                'total_updates': self.model_updates,
                'performance': {
                    'final_accuracy': self.metrics_history[-1]['accuracy'] if self.metrics_history else None,
                    'peak_accuracy': self.performance_metrics['peak_accuracy'],
                    'final_loss': self.metrics_history[-1]['loss'] if self.metrics_history else None,
                    'training_stability': self.performance_metrics['training_stability'],
                },
                'communication': {
                    'total_messages': self.communication_stats['total_messages'],
                    'successful_updates': self.communication_stats['successful_updates'],
                    'success_rate': success_rate,
                    'average_model_size': np.mean(self.performance_metrics['model_sizes'])
                    if self.performance_metrics['model_sizes'] else 0.0,
                },
                'staleness': {
                    'average': avg_staleness,
                    'maximum': self.staleness_stats['max_staleness'],
                    'by_client': {
                        client: {
                            'average': np.mean(values) if values and all(isinstance(v, (int, float)) for v in values) else 0.0,
                            'max': max(values) if values and all(isinstance(v, (int, float)) for v in values) else 0.0
                        }
                        for client, values in self.staleness_stats['by_client'].items()
                    }
                }
            }

            step = len(self.metrics_history)
            self.tb_writer.add_scalar('Metrics/FinalAccuracy', metrics_summary['performance']['final_accuracy'], step)
            self.tb_writer.add_scalar('Metrics/AverageStaleness', avg_staleness, step)
            self.tb_writer.add_scalar('Metrics/TotalUpdates', metrics_summary['total_updates'], step)
            self.tb_writer.add_scalar('Metrics/SuccessRate', success_rate, step)
            self.tb_writer.add_scalar('Metrics/AggregationRounds', self.model_updates, step)

            # Histogram for staleness
            self.tb_writer.add_histogram('Staleness/Distribution', np.array(self.staleness_stats['values']), step)

            # Per-client staleness
            for client, stats in self.staleness_stats['by_client'].items():
                if stats['staleness']:
                    self.tb_writer.add_scalar(f'Client/{client}/AverageStaleness', np.mean(stats['staleness']), step)
                    self.tb_writer.add_scalar(f'Client/{client}/MaxStaleness', max(stats['staleness']), step)

            metrics_file = os.path.join(self.metrics_path, 'final_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics_summary, f, indent=4)

            # Save model
            model_file = os.path.join(self.metrics_path, 'final_model.pth')
            torch.save(self.global_model.state_dict(), model_file)

            # Save learning curves
            metrics_df = pd.DataFrame(self.metrics_history)
            metrics_df.to_csv(os.path.join(self.metrics_path, 'learning_curves.csv'))

            # Log final hparams to TensorBoard
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
        except Exception as e:
            self.logger.info.error(f"Error in save_final_metrics: {e}", exc_info=True)

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

        self.logger.info(f"Aggregator {self.id}: Stopped and cleaned up.")
        
        # Close TensorBoard writer
        self.tb_writer.close()