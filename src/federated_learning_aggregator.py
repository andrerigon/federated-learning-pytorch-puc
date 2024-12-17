import threading
import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any, List, OrderedDict
from torch.utils.tensorboard import SummaryWriter
import torch.quantization as quant
from io import BytesIO
import gc
import numpy as np
import os
from datetime import datetime

from aggregation_strategy import AggregationStrategy
from model_manager import ModelManager
from dataset_loader import DatasetLoader
from metrics_logger import MetricsLogger
from sklearn.metrics import accuracy_score, adjusted_rand_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
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

        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard', f'aggregator_{self.id}'))

        # Load testset for evaluation
        self.testset = self.dataset_loader.testset

    def start(self):
        """
        Start the evaluation thread. Call this after initializing and setting everything up.
        """
        self.evaluation_thread.start()

    def receive_model_update(self, sender_id: str, client_dict: OrderedDict, local_model_version: int):
        """
        Called by the UAV or other orchestrator when a client model update arrives.

        Args:
            sender_id (str): The identifier of the client sending the update.
            client_dict (OrderedDict): The state dict of the client's model update.
            local_model_version (int): The client's local model version.
        """
        print(f"got model update: ! {sender_id}")
        # Calculate staleness
        staleness = self.global_model_version - local_model_version
        self.tb_writer.add_scalar('Staleness/Value', staleness, self.model_updates + 1)

        if self.is_fedavg():
            # For FedAvg, buffer updates
            self.client_updates_buffer.append(client_dict)
            self.check_and_aggregate_fedavg()
        else:
            # For async or incremental strategies, aggregate immediately
            updated_state = self.strategy.aggregate(
                self.global_model,
                client_dict,
                extra_info={'staleness': staleness}
            )
            self.update_global_model(updated_state)

    def is_fedavg(self) -> bool:
        # Check if the current strategy is FedAvg by type check
        return self.strategy.__class__.__name__ == 'FedAvgStrategy'

    def check_and_aggregate_fedavg(self):
        """
        For FedAvg: Check if the current time - last_aggregation_time >= round_interval.
        If yes, aggregate all collected updates and update the global model.
        """
        current_time = time.time()
        if (current_time - self.last_aggregation_time) >= self.round_interval:
            print("doing fedav aggreg")    
            updated_state = self.strategy.aggregate(self.global_model, self.client_updates_buffer)
            self.update_global_model(updated_state)
            self.client_updates_buffer = []
            self.last_aggregation_time = current_time
        else: 
            print("skipping fedav aggreg")    

    def update_global_model(self, state: OrderedDict, new_version: int = None):
        """
        Updates the global model with a new aggregated state dict.

        Args:
            state (OrderedDict): The aggregated model state.
            new_version (int, optional): The new global model version. If None, increments current version.
        """
        if new_version is None:
            new_version = self.global_model_version + 1

        self.global_model.load_state_dict(state)
        self.global_model_version = new_version
        self.model_updates += 1
        self.global_model.eval()
        self.global_model = torch.quantization.convert(self.global_model)
        self._log.info(f"Aggregator {self.id}: Updated global model to version {new_version}")

        # Log aggregation metrics
        self.tb_writer.add_scalar('Aggregation/GlobalModelVersion', self.global_model_version, self.model_updates)

    def state_dict(self):
        return self.global_model.state_dict()

    def eval(self, images):
        return self.global_model(images)

    def evaluate_global_model_periodically(self):
        """
        Periodically evaluate the global model until convergence criteria is met or stop_evaluation_flag is set.
        """
        while not self.stop_evaluation_flag:
            metrics = self.evaluate_global_model()
            self.metrics_history.append(metrics)
            self.log_metrics_to_tensorboard(metrics, step=len(self.metrics_history))
            if self.convergence_criteria.has_converged(self.metrics_history):
                print("\n\nConverged!\n\n")
                self.mode_converged_callback()
                break
            time.sleep(5)

    def evaluate_global_model(self) -> Dict[str, float]:
        """
        Evaluate the global model on the testset and return metrics (loss, classification_loss, accuracy).
        """
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
        del testloader
        return {
            'loss': mean_reconstruction_loss,
            'classification_loss': mean_classification_loss,
            'accuracy': accuracy
        }

    def mode_converged_callback(self):
        """
        Called when the global model has converged based on the convergence criteria.
        Logs a message and a text scalar to TensorBoard.
        """
        self._log.info("[Aggregator] Global model converged.")
        step = len(self.metrics_history)
        self.converged = True
        self.tb_writer.add_text('Convergence', f'Model converged at step {step}', step)

    def log_metrics_to_tensorboard(self, metrics: Dict[str, float], step: int):
        """
        Log evaluation metrics to TensorBoard.
        """
        if 'loss' in metrics:
            self.tb_writer.add_scalar('Evaluation/Loss', metrics['loss'], step)
        if 'classification_loss' in metrics:
            self.tb_writer.add_scalar('Evaluation/ClassificationLoss', metrics['classification_loss'], step)
        if 'accuracy' in metrics:
            self.tb_writer.add_scalar('Evaluation/Accuracy', metrics['accuracy'], step)

    def final_evaluation_and_cleanup(self):
        """
        Called at the end of the process. Performs final evaluation, logs final stats, and cleans up.
        """
        _, testset = self.dataset_loader.testset, self.dataset_loader.testset
        testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
        criterion_reconstruction = nn.MSELoss()
        criterion_classification = nn.CrossEntropyLoss()
        total_reconstruction_loss = 0
        total_classification_loss = 0
        mse_values = []
        all_labels = []
        all_predictions = []

        self.global_model.eval()
        with tqdm(total=len(testloader), desc="Final Testing") as progress_bar:
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(ae_device), data[1].to(ae_device)
                    decoded, classified = self.global_model(images)
                    reconstruction_loss = criterion_reconstruction(decoded, images)
                    classification_loss = criterion_classification(classified, labels)
                    total_reconstruction_loss += reconstruction_loss.item()
                    total_classification_loss += classification_loss.item()
                    mse_values.append(reconstruction_loss.item())
                    _, predicted = torch.max(classified.data, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    progress_bar.update(1)

        mean_reconstruction_loss = total_reconstruction_loss / len(testloader)
        mean_classification_loss = total_classification_loss / len(testloader)
        accuracy = 100 * accuracy_score(all_labels, all_predictions)
        features, labels = self.extract_features(testloader, self.global_model)
        clustering_accuracy, ari = self.evaluate_clustering(features, labels)

        print(f'MSE: {mean_reconstruction_loss}')
        print(f'Classification Loss: {mean_classification_loss}')
        print(f'Accuracy: {accuracy}%')
        print(f'Clustering Accuracy: {clustering_accuracy}')
        print(f'Adjusted Rand Index: {ari}')

        torch.save(self.global_model.state_dict(), os.path.join(self.output_dir, f'model_{self.id}.pth'))
        with open(os.path.join(self.output_dir, f'stats_{self.id}.txt'), 'w') as f:
            f.write(f'Mean Reconstruction Loss: {mean_reconstruction_loss}\n')
            f.write(f'Classification Loss: {mean_classification_loss}\n')
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'Clustering Accuracy: {clustering_accuracy}\n')
            f.write(f'Adjusted Rand Index: {ari}\n')

        final_step = len(self.metrics_history) + 1
        self.tb_writer.add_scalar('FinalEval/MSE', mean_reconstruction_loss, final_step)
        self.tb_writer.add_scalar('FinalEval/ClassificationLoss', mean_classification_loss, final_step)
        self.tb_writer.add_scalar('FinalEval/Accuracy', accuracy, final_step)
        self.tb_writer.add_scalar('FinalEval/ClusteringAccuracy', clustering_accuracy, final_step)
        self.tb_writer.add_scalar('FinalEval/ARI', ari, final_step)

        # Plotting and other logs
        plot_mse(mse_values, self.output_dir)
        plot_clustering_metrics(clustering_accuracy, ari, self.output_dir)

        # Cleanup
        testloader._iterator = None
        gc.collect()
        self.tb_writer.close()

    def extract_features(self, dataloader, model):
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for data in dataloader:
                images, targets = data[0].to(ae_device), data[1]
                encoded, _ = model(images)
                features.append(encoded.view(encoded.size(0), -1).cpu().numpy())
                labels.append(targets.cpu().numpy())
        return np.concatenate(features), np.concatenate(labels)

    def evaluate_clustering(self, features, labels, n_clusters=10):
        n_clusters = min(n_clusters, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        predicted_labels = kmeans.labels_
        labels = np.array(labels)
        label_mapping = {}
        for cluster in range(n_clusters):
            cluster_indices = np.where(predicted_labels == cluster)[0]
            if len(cluster_indices) == 0:
                continue
            true_labels = labels[cluster_indices]
            most_common_label = np.bincount(true_labels).argmax()
            label_mapping[cluster] = most_common_label

        mapped_predicted_labels = np.vectorize(label_mapping.get)(predicted_labels)
        accuracy = accuracy_score(labels, mapped_predicted_labels)
        ari = adjusted_rand_score(labels, mapped_predicted_labels)
        print(f'Clustering Accuracy: {accuracy}')
        print(f'Adjusted Rand Index: {ari}')

        plot_confusion_matrix(labels, mapped_predicted_labels, classes=range(10), output_dir=self.output_dir)
        plot_tsne(features, labels, self.output_dir)
        del kmeans
        return accuracy, ari

    def stop(self):
        """
        Stop evaluation, cleanup resources, and finalize.
        """
        self.stop_evaluation_flag = True
        if self.evaluation_thread.is_alive():
            self.evaluation_thread.join()
        self.final_evaluation_and_cleanup()

        torch.cuda.empty_cache()
        self.global_model.cpu()
        del self.global_model
        gc.collect()

        self._log.info(f"Aggregator {self.id}: Stopped and cleaned up.")