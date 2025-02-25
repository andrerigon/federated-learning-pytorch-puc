import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, adjusted_rand_score
import pandas as pd
import umap
from scipy.optimize import linear_sum_assignment
import plotly.express as px
import plotly.graph_objects as go
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms

# Data analysis and visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, adjusted_rand_score

# Utilities and system
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm

# TensorBoard visualization
from torch.utils.tensorboard import SummaryWriter

# Dimensionality reduction for visualization
import umap

class LandUseAutoencoderWithClustering(nn.Module):
    """
    Autoencoder architecture for land use classification with integrated clustering.
    This model combines reconstruction learning with cluster assignment optimization.
    
    The architecture consists of three main components:
    1. Encoder: Compresses the input image into a latent representation
    2. Decoder: Reconstructs the original image from the latent representation
    3. Clustering: Assigns the latent representation to clusters
    """
    def __init__(self, input_channels=3, n_clusters=10, embedding_dim=32):
        """
        Initialize the model components.
        
        Args:
            input_channels (int): Number of input image channels (3 for RGB)
            n_clusters (int): Number of clusters to learn
            embedding_dim (int): Dimension of the embedding space
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        
        # Encoder network: progressively reduce spatial dimensions while increasing channels
        self.encoder = nn.Sequential(
            # First encoding block: 3 -> 32 channels
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second encoding block: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third encoding block: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Decoder network: progressively increase spatial dimensions while decreasing channels
        self.decoder = nn.Sequential(
            # First decoding block: 128 -> 64 channels
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Second decoding block: 64 -> 32 channels
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final decoding block: 32 -> 3 channels
            nn.ConvTranspose2d(32, input_channels, kernel_size=2, stride=2),
            nn.Tanh()  # Output normalized image values
        )
        
        # Embedding network: convert feature maps to fixed-dimension embedding
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Clustering network with temperature scaling
        self.clustering = nn.Sequential(
            nn.Linear(embedding_dim, n_clusters)
            # Note: Softmax is applied in forward pass with temperature
        )
        
        # Learnable temperature parameter for cluster assignment
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize cluster centers buffer
        self.register_buffer('cluster_centers', 
                           torch.zeros(n_clusters, embedding_dim))
        self.register_buffer('cluster_centers_initialized', 
                           torch.tensor(False))

    def initialize_clusters(self, dataloader, device):
        """
        Initialize cluster centers using k-means on the embedding space.
        This should be called before training to establish initial cluster centers.
        
        Args:
            dataloader: DataLoader containing training data
            device: Device to perform computation on
        """
        if self.cluster_centers_initialized:
            return
            
        print("Collecting embeddings for cluster initialization...")
        embeddings = []
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            for batch, _ in tqdm(dataloader):
                batch = batch.to(device)
                # Get embeddings only
                _, embedding, _ = self.forward(batch)
                embeddings.append(embedding.cpu())
        
        # Concatenate all embeddings
        embeddings = torch.cat(embeddings).numpy()
        
        # Perform k-means clustering
        print("Performing k-means clustering...")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        kmeans.fit(embeddings)
        
        # Store cluster centers
        self.cluster_centers.data = torch.tensor(
            kmeans.cluster_centers_, 
            device=device
        )
        self.cluster_centers_initialized.data = torch.tensor(True)
        print("Cluster centers initialized")

    def forward(self, x):
        features = self.encoder(x)
        reconstructed = self.decoder(features)
        
        # Create embeddings with stronger normalization
        embedding = self.embedding(features)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalization
        
        # Generate cluster probabilities with sharpening
        logits = self.clustering[0](embedding)
        # Apply temperature scaling with a lower temperature for sharper distributions
        cluster_probs = F.softmax(logits / (self.temperature * 0.1), dim=1)
        
        return reconstructed, embedding, cluster_probs

class ClusteringLoss(nn.Module):
    """
    Custom loss function that encourages better cluster separation.
    Combines reconstruction loss with clustering objectives.
    """
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters
        
    def forward(self, reconstructed, original, cluster_probs, embedding):
        # Reconstruction loss - how well we recreate the input
        recon_loss = F.mse_loss(reconstructed, original)
        
        # Distribution loss - encourage uniform cluster assignment
        cluster_distribution = cluster_probs.mean(dim=0)
        distribution_loss = -torch.sum(cluster_distribution * torch.log(cluster_distribution + 1e-9))
        
        # Cluster separation loss - make embeddings more distinct
        similarity = torch.matmul(embedding, embedding.t())
        cluster_targets = torch.argmax(cluster_probs, dim=1)
        mask = (cluster_targets.unsqueeze(0) == cluster_targets.unsqueeze(1)).float()
        separation_loss = torch.mean(similarity * (1 - mask))
        
        # Combine losses
        total_loss = recon_loss + 0.1 * distribution_loss + 0.1 * separation_loss
        
        return total_loss, {
            'reconstruction': recon_loss.item(),
            'distribution': distribution_loss.item(),
            'separation': separation_loss.item()
        }
    
class ClusteringMetrics:
    """Handle clustering-related computations and metrics."""
    
    @staticmethod
    def target_distribution(cluster_probs):
        """
        Compute the target distribution P for clustering.
        Helps in learning by emphasizing confident assignments.
        """
        weight = (cluster_probs ** 2) / torch.sum(cluster_probs, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
    
    @staticmethod
    def clustering_loss(cluster_probs, target_dist):
        """
        Compute the clustering loss using KL divergence.
        """
        return F.kl_div(
            cluster_probs.log(), 
            target_dist, 
            reduction='batchmean'
        )
    
    @staticmethod
    def compute_metrics(embeddings, labels, predictions):
        """
        Compute various clustering quality metrics.
        """
        # Convert to numpy for sklearn metrics
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()
        
        # Compute metrics
        silhouette = silhouette_score(embeddings, predictions)
        ari = adjusted_rand_score(labels, predictions)
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)
        
        # Compute cluster purity
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)
        purity = conf_matrix[row_ind, col_ind].sum() / conf_matrix.sum()
        
        return {
            'silhouette_score': silhouette,
            'adjusted_rand_index': ari,
            'cluster_purity': purity,
            'confusion_matrix': conf_matrix
        }

class VisualizationUtils:
    """Utilities for creating visualizations for TensorBoard."""
    
    @staticmethod
    def plot_embeddings(embeddings, labels, predictions):
        """Create UMAP visualization of embeddings."""
        reducer = umap.UMAP(n_components=2)
        embedding_2d = reducer.fit_transform(embeddings)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot with true labels
        scatter1 = ax1.scatter(
            embedding_2d[:, 0], 
            embedding_2d[:, 1], 
            c=labels, 
            cmap='tab10'
        )
        ax1.set_title('Embeddings by True Labels')
        plt.colorbar(scatter1, ax=ax1)
        
        # Plot with predicted clusters
        scatter2 = ax2.scatter(
            embedding_2d[:, 0], 
            embedding_2d[:, 1], 
            c=predictions, 
            cmap='tab10'
        )
        ax2.set_title('Embeddings by Predicted Clusters')
        plt.colorbar(scatter2, ax=ax2)
        
        return fig
    
    @staticmethod
    def plot_reconstruction_samples(original, reconstructed, n_samples=8):
        """Create visualization of original vs reconstructed images."""
        fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
        
        for i in range(n_samples):
            # Original
            axes[0, i].imshow(
                original[i].cpu().permute(1, 2, 0).numpy()
            )
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')
            
            # Reconstructed
            axes[1, i].imshow(
                reconstructed[i].cpu().permute(1, 2, 0).numpy()
            )
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed')
        
        return fig
    
class TrainingLogger:
    """
    Handles training progress logging and visualization in TensorBoard.
    """
    def __init__(self, log_dir, generate_visualizations=False):
        """
        Initializes a training logger.
        
        Args:
            log_dir (str): Directory to save logs
            generate_visualizations (bool): Whether to generate visualizations
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.generate_visualizations = generate_visualizations
    
    def log_training_step(self, metrics, batch=None, reconstructed=None):
        """
        Logs training metrics and visualizations for the current step.
        
        We now explicitly handle the original and reconstructed images,
        making it clearer what we're visualizing.
        """
        # Log basic metrics first
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'training/{name}', value, self.step)
        
        # If we have images to visualize, create and log the comparison
        if batch is not None and reconstructed is not None:
            # Select a subset of images for visualization
            n_samples = min(8, batch.size(0))
            batch_samples = batch[:n_samples]
            recon_samples = reconstructed[:n_samples]
            
            # Create a grid of original and reconstructed images
            comparison = torch.cat([batch_samples, recon_samples])
            grid = torchvision.utils.make_grid(comparison, nrow=n_samples)
            
            # Log to TensorBoard
            self.writer.add_image('reconstruction', grid, self.step)
        
        # Log embeddings if available
        if 'embeddings' in metrics and 'labels' in metrics:
            self.writer.add_embedding(
                mat=torch.from_numpy(metrics['embeddings']),
                metadata=metrics['labels'],
                global_step=self.step
            )
        
        self.step += 1
    
    def log_evaluation(self, metrics, phase='validation'):
        """
        Logs evaluation metrics and creates visualizations.
        """
        for name, value in metrics.items():
            # Always log scalar metrics
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'{phase}/{name}', value, self.step)
            # Only create visualization figures if enabled
            elif name == 'confusion_matrix' and self.generate_visualizations:
                fig = plt.figure(figsize=(10, 8))
                sns.heatmap(value, annot=True, fmt='d')
                plt.title(f'{phase.capitalize()} Confusion Matrix')
                self.writer.add_figure(f'{phase}/confusion_matrix', fig, self.step)
                plt.close()
    
    def close(self):
        self.writer.close()

def train_epoch(model, train_loader, optimizer, device):
    """
    Trains the model for one epoch and returns comprehensive metrics.
    
    Args:
        model: The autoencoder model
        train_loader: DataLoader for the training data
        optimizer: The optimizer for model parameters
        device: The device to run training on (CPU/GPU)
        
    Returns:
        metrics: Dictionary containing training metrics
        batch_samples: A batch of original images for visualization
        recon_samples: Corresponding reconstructed images
    """
    model.train()
    total_recon_loss = 0.0
    total_cluster_loss = 0.0
    
    # Storage for embeddings and predictions
    all_embeddings = []
    all_labels = []
    all_predictions = []
    
    # Store first batch for visualization
    batch_samples = None
    recon_samples = None
    
    for batch_idx, (batch, labels) in enumerate(tqdm(train_loader, desc="Training")):
        batch = batch.to(device)
        labels = labels.to(device)
        
        # Forward pass through the model
        reconstructed, embedding, cluster_probs = model(batch)
        
        # Store first batch for visualization
        if batch_idx == 0:
            batch_samples = batch.detach()
            recon_samples = reconstructed.detach()
        
        # Calculate reconstruction loss
        recon_loss = F.mse_loss(reconstructed, batch)
        
        # Calculate clustering loss with temperature scaling
        # We integrate clustering objectives directly here
        cluster_distribution = cluster_probs.mean(dim=0)
        distribution_loss = -torch.sum(cluster_distribution * torch.log(cluster_distribution + 1e-9))
        
        # Combined loss
        total_loss = recon_loss + 0.1 * distribution_loss
        
        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_recon_loss += recon_loss.item()
        total_cluster_loss += distribution_loss.item()
        
        # Store results for analysis
        all_embeddings.append(embedding.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        predictions = torch.argmax(cluster_probs, dim=1)
        all_predictions.append(predictions.cpu().numpy())
    
    # Calculate average losses
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_cluster_loss = total_cluster_loss / len(train_loader)
    
    # Prepare metrics dictionary
    metrics = {
        'reconstruction_loss': avg_recon_loss,
        'clustering_loss': avg_cluster_loss,
        'embeddings': np.concatenate(all_embeddings),
        'labels': np.concatenate(all_labels),
        'predictions': np.concatenate(all_predictions)
    }
    
    return metrics, batch_samples, recon_samples

def evaluate_model(model, eval_loader, device):
    """
    Evaluates the model's performance on a given dataset.
    
    This function computes multiple metrics to assess both the reconstruction quality
    and the clustering performance of our autoencoder model. It runs in evaluation
    mode (no gradient computation) for efficiency.
    
    Args:
        model: The autoencoder model to evaluate
        eval_loader: DataLoader containing the evaluation dataset
        device: Device (CPU/GPU) to run the evaluation on
    
    Returns:
        Dictionary containing various performance metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metric accumulators
    total_recon_loss = 0.0
    all_embeddings = []
    all_labels = []
    all_predictions = []
    
    # Store samples for visualization
    sample_batch = None
    sample_reconstruction = None
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for batch_idx, (batch, labels) in enumerate(eval_loader):
            # Move data to appropriate device
            batch = batch.to(device)
            labels = labels.to(device)
            
            # Forward pass through the model
            reconstructed, embedding, cluster_probs = model(batch)
            
            # Store the first batch for visualization
            if batch_idx == 0:
                sample_batch = batch.detach()
                sample_reconstruction = reconstructed.detach()
            
            # Compute reconstruction loss
            recon_loss = F.mse_loss(reconstructed, batch)
            total_recon_loss += recon_loss.item()
            
            # Store embeddings and predictions for clustering metrics
            all_embeddings.append(embedding.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(
                torch.argmax(cluster_probs, dim=1).cpu().numpy()
            )
    
    # Concatenate accumulated data
    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    
    # Compute clustering metrics
    silhouette = silhouette_score(all_embeddings, all_predictions)
    ari = adjusted_rand_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Calculate cluster purity
    cluster_assignments = confusion_matrix(all_labels, all_predictions)
    purity = np.sum(np.max(cluster_assignments, axis=0)) / np.sum(cluster_assignments)
    
    return {
        'reconstruction_loss': total_recon_loss / len(eval_loader),
        'silhouette_score': silhouette,
        'adjusted_rand_index': ari,
        'cluster_purity': purity,
        'confusion_matrix': conf_matrix,
        'embeddings': all_embeddings,
        'labels': all_labels,
        'predictions': all_predictions,
        'sample_batch': sample_batch,
        'sample_reconstruction': sample_reconstruction
    }

def main():
    """
    Main training pipeline for the land use clustering autoencoder.
    This function handles the complete process of training, evaluation,
    and visualization of results.
    """
    # Set up logging directory with timestamp for unique runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f'runs/landuse_clustering_{timestamp}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Land Use Clustering Training Pipeline ===")
    print(f"Starting new run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using device: {device}")
    print(f"Logging results to: {log_dir}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define data transforms
    # Normalize using ImageNet statistics as they work well for transfer learning
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and prepare dataset
    print("\nLoading EuroSAT dataset...")
    try:
        dataset = datasets.EuroSAT(
            root='./data',
            transform=transform,
            download=True
        )
        print(f"Dataset loaded successfully: {len(dataset)} images")
        print(f"Classes: {dataset.classes}")
        
        # Create class distribution visualization
        class_counts = torch.zeros(len(dataset.classes))
        for _, label in dataset:
            class_counts[label] += 1
        
        # Log class distribution to TensorBoard
        fig = plt.figure(figsize=(10, 5))
        plt.bar(dataset.classes, class_counts)
        plt.xticks(rotation=45)
        plt.title("Class Distribution in Dataset")
        plt.tight_layout()
        
        logger = TrainingLogger(log_dir)
        logger.writer.add_figure('dataset/class_distribution', fig)
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"\nDataset split:")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples: {test_size}")
    
    # Create data loaders
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )
    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        return
    
    # Initialize model and training components
    print("\nInitializing model...")
    model = LandUseAutoencoderWithClustering(
        input_channels=3,  # RGB images
        n_clusters=len(dataset.classes),
        embedding_dim=32
    ).to(device)
    
    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Initialize cluster centers before training
    print("\nInitializing cluster centers...")
    model.initialize_clusters(train_loader, device)
    
    # Training configuration
    num_epochs = 1
    best_val_loss = float('inf')
    patience = 10  # For early stopping
    patience_counter = 0
    
    print("\nStarting training...")
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            metrics, batch_samples, recon_samples = train_epoch(
                model, train_loader, optimizer, device
            )
            
            # Log training metrics and visualizations
            logger.log_training_step(
                metrics,
                batch=batch_samples,
                reconstructed=recon_samples
            )
            
            # Validation phase (every 5 epochs)
            if epoch % 5 == 0:
                val_metrics = evaluate_model(model, val_loader, device)
                logger.log_evaluation(val_metrics, 'validation')
                
                # Update learning rate based on validation loss
                scheduler.step(val_metrics['reconstruction_loss'])
                
                # Early stopping check
                if val_metrics['reconstruction_loss'] < best_val_loss:
                    best_val_loss = val_metrics['reconstruction_loss']
                    # Save best model
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val_loss,
                        },
                        log_dir / 'best_model.pth'
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print("\nEarly stopping triggered")
                    break
                
                # Print current metrics
                print("\nValidation Metrics:")
                print(f"Reconstruction Loss: {val_metrics['reconstruction_loss']:.4f}")
                print(f"Silhouette Score: {val_metrics['silhouette_score']:.4f}")
                print(f"Cluster Purity: {val_metrics['cluster_purity']:.4f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise e
    
    # Final evaluation on test set
    print("\nPerforming final evaluation on test set...")
    try:
        # Load best model
        checkpoint = torch.load(log_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, test_loader, device)
        logger.log_evaluation(test_metrics, 'test')
        
        # Print final results
        print("\nFinal Test Set Metrics:")
        print(f"Reconstruction Loss: {test_metrics['reconstruction_loss']:.4f}")
        print(f"Silhouette Score: {test_metrics['silhouette_score']:.4f}")
        print(f"Cluster Purity: {test_metrics['cluster_purity']:.4f}")
        
        # Save final results summary
        results_summary = {
            'test_reconstruction_loss': test_metrics['reconstruction_loss'],
            'test_silhouette_score': test_metrics['silhouette_score'],
            'test_cluster_purity': test_metrics['cluster_purity'],
            'best_validation_loss': best_val_loss,
            'total_epochs': epoch + 1
        }
        
        with open(log_dir / 'results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=4)
        
    except Exception as e:
        print(f"\nError during final evaluation: {str(e)}")
        raise e
    finally:
        logger.close()
    
    print(f"\nTraining completed successfully")
    print(f"All results and visualizations saved to: {log_dir}")
    print("\nTo view training progress and results, run:")
    print(f"tensorboard --logdir={log_dir}")

if __name__ == '__main__':
    main()        