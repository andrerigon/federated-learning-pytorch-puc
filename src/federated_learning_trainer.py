import threading
import torch
import torch.nn as nn
import logging
import time
from model_manager import ModelManager
from dataset_loader import DatasetLoader
from metrics_logger import MetricsLogger
import torch.optim as optim
from tqdm import tqdm
from io import BytesIO
import torch.quantization as quant
import os
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from collections import OrderedDict


class FederatedLearningTrainer:
    """
    A class that manages local training for a federated learning client.
    It handles loading data, training the model locally, and preparing updates for the global model.
    """

    def __init__(
        self,
        id,
        model_manager: ModelManager,
        dataset_loader: DatasetLoader,
        metrics_logger: MetricsLogger,
        device=None,
        start_thread=True,
        synchronous=False,
        use_proximal_term=False
    ):
        """
        Initializes the FederatedLearningTrainer.

        Args:
            id (int): The identifier for the client.
            model_manager (ModelManager): An instance of ModelManager to handle model creation and loading.
            dataset_loader (DatasetLoader): An instance of DatasetLoader to handle data loading.
            device (torch.device, optional): The device to run the training on. Defaults to CUDA if available.
            start_thread (bool, optional): Whether to start the training thread upon initialization.
            output_dir (str, optional): Directory to save logs and outputs.
        """
        self.id = id
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_manager = model_manager
        self.start_time = time.time()
        self.local_model_version = 0
        self.synchronous = synchronous
        self.current_state = None
        self.use_proximal_term = use_proximal_term

        # Load the dataset for the client
        self.dataset_loader = dataset_loader
        self.loader = self.dataset_loader.loader(client_id=self.id)
        self.testset = self.dataset_loader.testset

        # Load the global model
        self.global_model = self.model_manager.load_model()

        # Initialize training parameters
        self.training_cycles = 0
        self.model_updates = 0
        self.global_model_version = 0
        self.global_model_changed = False

        self.finished = False
        self.model_updated = False
        self.current_training_loss = 0

        self.logger = logger.bind(source="sensor", sensor_id=self.id)

        # Initialize MetricsLogger
        self.metrics_logger = metrics_logger

        if start_thread:
            # Start the training thread
            self.thread = threading.Thread(target=self.start_training)
            self.thread.start()

    def update_model(self, state, new_version):
        """
        Updates the global model with new weights and version.

        Args:
            state (dict): The state dictionary of the new global model.
            new_version (int): The version number of the new global model.
        """
        self.global_model.load_state_dict(state)
        self.global_model_changed = True
        self.model_updates += 1
        self.global_model_version = new_version
        self.logger.info(f"Client {self.id}: Updated global model to version {new_version}")

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, alpha=0.3, extra_info=None) -> OrderedDict:
        global_dict = global_model.state_dict()
        staleness = extra_info.get('staleness', 0) if extra_info else 0
        logger.info(f"Staleness will be: {staleness}")
        effective_alpha = alpha # / (1 + staleness)
        for k in global_dict.keys():
            if k in client_dict and global_dict[k].size() == client_dict[k].size():
                global_dict[k] = (1 - effective_alpha)*global_dict[k] + effective_alpha*client_dict[k].dequantize()
        return global_dict
    
    def last_version(self):
        """
        Returns the current version of the global model.

        Returns:
            int: The version number of the global model.
        """
        return self.global_model_version
    

    def start_training(self):
        """
        The main training loop that runs in a separate thread.
        It continues training until self.finished is set to True.
        """
        while not self.finished:
            if self.should_train():
                self.train()
            else:
                time.sleep(1)    

    def should_train(self):
       if not self.synchronous:
           return True 
       
       return self.local_model_version == 0 or self.global_model_changed

    def log_model_sizes(self, model):
        """
        Calculates and accumulates the sizes of the non-quantized and quantized versions of the model.

        Args:
            model (nn.Module): The model whose sizes are to be logged.
        """
        non_quantized_size = self.get_model_size(model)
        quantized_size = self.get_model_size(torch.quantization.convert(model))
        self.logger.debug(f"Non-quantized model size: {non_quantized_size} bytes")
        self.logger.debug(f"Quantized model size: {quantized_size} bytes")

        # Register model sizes
        self.metrics_logger.register_model_size('non_quantized', self.training_cycles, non_quantized_size)
        self.metrics_logger.register_model_size('quantized', self.training_cycles, quantized_size)

    def get_model_size(self, model):
        """
        Calculates the size of a model in bytes.

        Args:
            model (nn.Module): The model whose size is to be calculated.

        Returns:
            int: The size of the model in bytes.
        """
        buffer = BytesIO()
        torch.save(model.state_dict(), buffer)
        size = len(buffer.getvalue())
        buffer.close()
        return size
    
    def stop_training(self):
        self.finished = True

    def stop(self):
        """
        Stops the training process and cleans up resources.
        """
        self.finished = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
            self.logger.info(f"Training thread stopped.")

        # Evaluate the local model if it exists
        if hasattr(self, 'local_model'):
            self.evaluate_model(self.local_model)
        else:
            self.logger.warning(f"No local model available for evaluation.")

        # Flush metrics (write to TensorBoard and generate plots)
        self.metrics_logger.flush()

        self.logger.info(f"Resources have been cleaned up.")

    def prepare_local_model(self):
        """
        Prepares the local model for training by loading the global model weights
        and setting up quantization-aware training configurations.

        Returns:
            nn.Module: The prepared local model.
        """
        # Create a new local model and load the global model's weights
        local_model = self.model_manager.create_model().to(self.device)
        local_model.load_state_dict(self.global_model.state_dict())

        # Define a custom quantization configuration
        qconfig = quant.QConfig(
            activation=quant.FakeQuantize.with_args(
                observer=quant.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            ),
            weight=quant.default_weight_fake_quant,
        )

        local_model.qconfig = qconfig
        # Prepare the model for quantization-aware training
        local_model = torch.quantization.prepare_qat(local_model, inplace=True)

        return local_model

    def get_loss_functions_and_optimizer(self, local_model):
        """
        Defines loss functions, optimizer, and learning rate scheduler.

        Args:
            local_model (nn.Module): The model to train.

        Returns:
            Tuple containing:
                - criterion_reconstruction (nn.Module)
                - criterion_classification (nn.Module)
                - optimizer (torch.optim.Optimizer)
                - scheduler (torch.optim.lr_scheduler._LRScheduler)
        """
        criterion_reconstruction = nn.MSELoss()
        criterion_classification = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(local_model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return criterion_reconstruction, criterion_classification, optimizer, scheduler

    def extra_info(self):
        return {'client_samples': len(self.loader),
        'client_loss': self.current_training_loss}
  
    def train_one_batch(self, local_model, inputs, labels, criterion_reconstruction, criterion_classification, optimizer):
        """Train on a single batch with error handling."""
        try:
            optimizer.zero_grad()

            # Forward pass
            decoded, classified = local_model(inputs)

            # Compute losses
            reconstruction_loss = criterion_reconstruction(decoded, inputs)
            classification_loss = criterion_classification(classified, labels)

            # Calculate proximal term if enabled
            global_model_state_dict = self.global_model.state_dict()
            proximal_term = self.calculate_proximal_term(
                local_model, 
                global_model_state_dict
            ) if global_model_state_dict is not None else 0.0

            # Total loss INCLUDING the proximal term
            loss = reconstruction_loss + classification_loss + proximal_term

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Detach losses for logging
            reconstruction_loss_value = reconstruction_loss.item()
            classification_loss_value = classification_loss.item()

            # Ensure we don't keep unnecessary graph history
            decoded = decoded.detach()
            classified = classified.detach()

            return loss, reconstruction_loss_value, classification_loss_value, decoded, classified
            
        except Exception as e:
            self.logger.error(f"Error in train_one_batch: {str(e)}", exc_info=True)
            raise  # Re-raise the exception after logging

    def calculate_proximal_term(self, local_model, global_model_state_dict, mu=0.01):
        """
        Calculates the proximal term from FedProx (Li et al., 2020).
        
        The proximal term helps to keep local models from deviating too far from the global model
        during training, which is particularly important in heterogeneous networks.
        
        Formula: (μ/2) * ||w - w_t||^2 
        where:
            μ: proximal term coefficient
            w: local model parameters (weights and biases)
            w_t: global model parameters at round t
        
        Args:
            local_model (nn.Module): The current local model
            global_model_state_dict (OrderedDict): State dict of global model parameters
            mu (float): Proximal term coefficient (default: 0.01)
            
        Returns:
            torch.Tensor: The calculated proximal term loss
        """
        if not hasattr(self, 'use_proximal_term') or not self.use_proximal_term:
            return 0.0
            
        try:
            proximal_term = 0.0
            
            # Work directly with model parameters
            for name, local_param in local_model.named_parameters():
                if name not in global_model_state_dict:
                    continue
                    
                global_param = global_model_state_dict[name]
                
                if local_param.shape != global_param.shape:
                    continue
                    
                param_diff = local_param - global_param
                proximal_term += (mu / 2) * torch.norm(param_diff) ** 2
                
            return proximal_term
            
        except Exception as e:
            self.logger.error(f"Error calculating proximal term: {str(e)}", exc_info=True)
            return 0.0  
    
    def train_one_epoch(self, local_model, criterion_reconstruction, criterion_classification, optimizer, scheduler):
        running_reconstruction_loss = 0.0
        running_classification_loss = 0.0
        
        # Only store metrics every N batches
        visualization_frequency = 10  # Adjust this value as needed
        batch_predictions = []
        batch_labels = []
        batch_embeddings = []

        progress_bar = tqdm(
            enumerate(self.loader, 0),
            total=len(self.loader),
            desc=f'S[{self.id+1}], # {self.training_cycles}',
            leave=False
        )

        for i, data in progress_bar:
            if self.finished:
                self.logger.info(f"Training has been stopped.")
                break

            try:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                loss, reconstruction_loss_value, classification_loss_value, decoded, classified = self.train_one_batch(
                    local_model,
                    inputs,
                    labels,
                    criterion_reconstruction,
                    criterion_classification,
                    optimizer
                )
                self.current_training_loss = loss.item()

                running_reconstruction_loss += reconstruction_loss_value
                running_classification_loss += classification_loss_value

                # Only collect visualization data periodically
                if i % visualization_frequency == 0:
                    with torch.no_grad():
                        _, predictions = torch.max(classified.data, 1)
                        # Use the encoder output we already have from decoded
                        encoded = decoded.flatten(1)  # Reuse existing tensor

                        # Store in batches instead of extending lists
                        batch_predictions.append(predictions.cpu())
                        batch_labels.append(labels.cpu())
                        batch_embeddings.append(encoded.cpu())

                        # Register MSE less frequently
                        self.metrics_logger.register_mse_per_sample(
                            self.training_cycles, 
                            decoded,
                            inputs
                        )

                progress_bar.set_postfix(
                    reconstruction_loss=running_reconstruction_loss / (i + 1),
                    classification_loss=running_classification_loss / (i + 1)
                )

            except Exception as batch_error:
                self.logger.error(f"Error processing batch {i}: {str(batch_error)}", exc_info=True)
                continue
            
            finally:
                # Clean up tensor references
                if 'inputs' in locals(): del inputs
                if 'labels' in locals(): del labels
                if 'loss' in locals(): del loss
                if 'decoded' in locals(): del decoded
                if 'classified' in locals(): del classified

        # Process collected visualization data at the end of epoch
        if batch_predictions:
            try:
                predictions_tensor = torch.cat(batch_predictions)
                labels_tensor = torch.cat(batch_labels)
                embeddings_tensor = torch.cat(batch_embeddings)

                # Register predictions and embeddings for visualization
                self.metrics_logger.register_predictions(
                    self.training_cycles,
                    predictions_tensor,
                    labels_tensor,
                    embeddings_tensor
                )
            except Exception as viz_error:
                self.logger.error(f"Error creating visualizations: {str(viz_error)}", exc_info=True)

        # Adjust learning rate
        scheduler.step(running_reconstruction_loss / len(self.loader))

        if self.training_cycles % 10 == 0:
            self.evaluate_model(self.local_model)

        return running_reconstruction_loss, running_classification_loss
   
    def evaluate_model(self, model):
        model.eval()
        correct = 0
        total = 0

        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=False, num_workers=0)

        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                _, outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs, predicted

        accuracy = 100 * correct / total
        self.metrics_logger.register_accuracy(self.training_cycles, accuracy)
        self.logger.info(f"Evaluation Accuracy: {accuracy}%")
        del test_loader

    def finalize_training(self, local_model):
        local_model.eval()
        self.log_model_sizes(local_model)
        self.local_model_version += 1
        local_model = torch.quantization.convert(local_model, mapping={'classifier': torch.nn.Identity})

        new_state = local_model.state_dict()
        if self.global_model_changed and not self.synchronous:
            self.logger.info("Global model changed while training. Aggregating")
            new_state = self.aggregate(self.global_model, new_state)
            
        self.global_model_changed = False    
        self.current_state = new_state

        self.model_updated = True

        self.logger.debug(f"Local model updated and ready for aggregation.")

    def train(self, epochs=1):
        # self.global_model_changed = False
        torch.backends.quantized.engine = 'qnnpack'
        try:
            # Record start time
            cycle_start_time = time.time()
            
            self.local_model = self.prepare_local_model()
            local_model = self.local_model

            current_version = self.global_model_version

            criterion_reconstruction, criterion_classification, optimizer, scheduler = \
                self.get_loss_functions_and_optimizer(local_model)

            self.training_cycles += 1

            for epoch in range(epochs):
                if self.finished:
                    self.logger.info(f"Training has been stopped.")
                    break

                running_reconstruction_loss, running_classification_loss = self.train_one_epoch(
                    local_model,
                    criterion_reconstruction,
                    criterion_classification,
                    optimizer,
                    scheduler
                )

                if self.finished:
                    break

            # Generate TSNE visualization
            self.metrics_logger.log_tsne_visualization(self.training_cycles)

            # Record cycle duration
            cycle_duration = time.time() - cycle_start_time
            self.metrics_logger.register_training_time(self.training_cycles, cycle_duration)

            # Calculate and log average losses
            average_reconstruction_loss = running_reconstruction_loss / len(self.loader)
            average_classification_loss = running_classification_loss / len(self.loader)

            self.metrics_logger.register_loss('reconstruction', self.training_cycles, average_reconstruction_loss)
            self.metrics_logger.register_loss('classification', self.training_cycles, average_classification_loss)
            self.metrics_logger.register_global_step(self.training_cycles)

            if self.finished:
                return

            self.finalize_training(local_model)

            current_timestamp = time.time()
            elapsed_time = current_timestamp - self.start_time
            staleness = self.global_model_version - current_version
            self.metrics_logger.register_staleness(self.model_updates, staleness, timestamp=elapsed_time)

            del local_model, criterion_reconstruction, criterion_classification, optimizer, scheduler

        except Exception as e:
            self.logger.error(f"Error in client {self.id}: {e}", exc_info=True)