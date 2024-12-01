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

# Assume MetricsLogger and plotting functions are imported or defined above

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
        start_thread=True
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

        # Load the dataset for the client
        self.dataset_loader = dataset_loader
        self.loader = self.dataset_loader.loader(client_id=self.id)
        self.testset = self.dataset_loader.testset  # Ensure this is the test dataset

        # Load the global model
        self.global_model = self.model_manager.load_model()

        # Initialize training parameters
        self.training_cycles = 0  # Number of completed training cycles
        self.model_updates = 0    # Number of model updates received
        self.global_model_version = 0  # Version of the global model
        self.global_model_changed = False  # Flag indicating if the global model has changed

        self.finished = False       # Flag to signal training termination
        self.model_updated = False  # Flag indicating if the local model has been updated

        self._log = logging.getLogger(__name__)  # Initialize the logger

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
        self._log.info(f"Client {self.id}: Updated global model to version {new_version}")

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
            self.train()

    def log_model_sizes(self, model):
        """
        Calculates and accumulates the sizes of the non-quantized and quantized versions of the model.

        Args:
            model (nn.Module): The model whose sizes are to be logged.
        """
        non_quantized_size = self.get_model_size(model)
        quantized_size = self.get_model_size(torch.quantization.convert(model))
        self._log.info(f"Client {self.id}: Non-quantized model size: {non_quantized_size} bytes")
        self._log.info(f"Client {self.id}: Quantized model size: {quantized_size} bytes")

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
            self._log.info(f"Client {self.id}: Training thread stopped.")

        # Evaluate the local model if it exists
        if hasattr(self, 'local_model'):
            self.evaluate_model(self.local_model)
        else:
            self._log.warning(f"Client {self.id}: No local model available for evaluation.")

        # Ensure all torch resources are freed
        torch.cuda.empty_cache()  # Clear CUDA memory if using GPU
        self.global_model.cpu()   # Move model to CPU to free GPU memory
        del self.global_model     # Remove model to free memory

        # Clean up the data loader
        if hasattr(self.loader, '_iterator'):
            self.loader._iterator = None  # Break reference cycle if any
        del self.loader  # Remove data loader to free memory

        # Flush metrics (write to TensorBoard and generate plots)
        self.metrics_logger.flush()

        self._log.info(f"Client {self.id}: Resources have been cleaned up.")

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

    def train_one_batch(self, local_model, inputs, labels, criterion_reconstruction, criterion_classification, optimizer):
        """
        Trains the model on a single batch.

        Args:
            local_model (nn.Module): The model to train.
            inputs (torch.Tensor): Input data.
            labels (torch.Tensor): Ground truth labels.
            criterion_reconstruction (nn.Module): The reconstruction loss function.
            criterion_classification (nn.Module): The classification loss function.
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            Tuple containing:
                - loss (torch.Tensor): The total loss.
                - reconstruction_loss_value (float): The reconstruction loss value.
                - classification_loss_value (float): The classification loss value.
        """
        optimizer.zero_grad()

        # Forward pass
        decoded, classified = local_model(inputs)

        # Compute losses
        reconstruction_loss = criterion_reconstruction(decoded, inputs)
        classification_loss = criterion_classification(classified, labels)

        # Total loss
        loss = reconstruction_loss + classification_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Detach losses for logging
        reconstruction_loss_value = reconstruction_loss.item()
        classification_loss_value = classification_loss.item()

        # Release memory
        del decoded, classified, reconstruction_loss, classification_loss

        return loss, reconstruction_loss_value, classification_loss_value

    def train_one_epoch(self, local_model, criterion_reconstruction, criterion_classification, optimizer, scheduler):
        """
        Trains the model for one epoch.

        Args:
            local_model (nn.Module): The model to train.
            criterion_reconstruction (nn.Module): The reconstruction loss function.
            criterion_classification (nn.Module): The classification loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.

        Returns:
            Tuple containing:
                - running_reconstruction_loss (float)
                - running_classification_loss (float)
        """
        running_reconstruction_loss = 0.0
        running_classification_loss = 0.0

        # Create a progress bar for monitoring
        progress_bar = tqdm(
            enumerate(self.loader, 0),
            total=len(self.loader),
            desc=f'Client {self.id+1}, Training Cycle {self.training_cycles}',
            leave=False
        )

        for i, data in progress_bar:
            if self.finished:
                self._log.info(f"Client {self.id}: Training has been stopped.")
                break  # Exit if training is stopped

            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            loss, reconstruction_loss_value, classification_loss_value = self.train_one_batch(
                local_model,
                inputs,
                labels,
                criterion_reconstruction,
                criterion_classification,
                optimizer
            )

            # Update running losses
            running_reconstruction_loss += reconstruction_loss_value
            running_classification_loss += classification_loss_value
            progress_bar.set_postfix(
                reconstruction_loss=running_reconstruction_loss / (i + 1),
                classification_loss=running_classification_loss / (i + 1)
            )

            # Release batch memory
            del inputs, labels, loss

        # Adjust learning rate
        scheduler.step(running_reconstruction_loss / len(self.loader))

        return running_reconstruction_loss, running_classification_loss

    def evaluate_model(self, model):
        """
        Evaluates the given model on the test dataset and accumulates accuracy.

        Args:
            model (nn.Module): The model to evaluate.
        """
        model.eval()
        correct = 0
        total = 0

        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=False, num_workers=2)

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
        self._log.info(f"Client {self.id}: Evaluation Accuracy: {accuracy}%")
        del test_loader

    def finalize_training(self, local_model):
        """
        Finalizes the training by converting the model to a quantized version,
        saving the state for uploading.

        Args:
            local_model (nn.Module): The trained local model.
        """

        # Set model to evaluation mode
        local_model.eval()
        # Log model sizes
        self.log_model_sizes(local_model)

        # Convert the model to a quantized version
        local_model = torch.quantization.convert(local_model, mapping={'classifier': torch.nn.Identity})

        # Save the current state for uploading to the server
        self.current_state = local_model.state_dict()
        self.model_updated = True  # Flag that the model has been updated

        self._log.info(f"Client {self.id}: Local model updated and ready for aggregation.")

    def train(self, epochs=1):
        """
        Performs local training on the client's data.

        Args:
            epochs (int, optional): Number of epochs to train. Defaults to 1.
        """
        torch.backends.quantized.engine = 'qnnpack'
        try:
            # Prepare the local model
            self.local_model = self.prepare_local_model()
            local_model = self.local_model

            current_version = self.global_model_version

            # Get loss functions and optimizer
            criterion_reconstruction, criterion_classification, optimizer, scheduler = self.get_loss_functions_and_optimizer(local_model)

            # Increment training cycle count
            self.training_cycles += 1

            for epoch in range(epochs):
                if self.finished:
                    self._log.info(f"Client {self.id}: Training has been stopped.")
                    break  # Exit if training is stopped

                # Train for one epoch
                running_reconstruction_loss, running_classification_loss = self.train_one_epoch(
                    local_model,
                    criterion_reconstruction,
                    criterion_classification,
                    optimizer,
                    scheduler
                )

                if self.finished:
                    break  # Exit if training is stopped

            # Accumulate average losses and global step
            average_reconstruction_loss = running_reconstruction_loss / len(self.loader)
            average_classification_loss = running_classification_loss / len(self.loader)

            self.metrics_logger.register_loss('reconstruction', self.training_cycles, average_reconstruction_loss)
            self.metrics_logger.register_loss('classification', self.training_cycles, average_classification_loss)
            self.metrics_logger.register_global_step(self.training_cycles)

            self._log.info(f"Client {self.id}: Completed training cycle {self.training_cycles}.")

            if self.finished:
                return

            # Finalize training
            self.finalize_training(local_model)

            # Calculate staleness before updating the version
            current_timestamp = time.time()
            elapsed_time = current_timestamp - self.start_time
            staleness = self.global_model_version - current_version
            self.metrics_logger.register_staleness(self.model_updates, staleness, timestamp=elapsed_time)

            # Clean up
            del local_model, criterion_reconstruction, criterion_classification, optimizer, scheduler

        except Exception as e:
            self._log.error(f"Error in client {self.id}: {e}", exc_info=True)