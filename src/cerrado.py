#!/usr/bin/env python3
"""
Cerrado Vegetation Monitoring System

This script implements an unsupervised deep learning system for monitoring 
vegetation patterns in the Brazilian Cerrado using satellite imagery. It includes
data loading, model training, TensorBoard logging, and interactive visualization.

Author: Your Name
Date: 2024
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import planetary_computer
import pystac_client
import rasterio
import stackstac
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import warnings
import xarray as xr
from pyproj import CRS
from rasterio.coords import BoundingBox
import pickle
import hashlib
import json

warnings.filterwarnings('ignore')

class CerradoDataset(Dataset):
    """
    A PyTorch Dataset for processing and analyzing Cerrado satellite imagery.
    
    This class handles the complete pipeline of satellite data acquisition,
    processing, and caching for the Brazilian Cerrado biome. It includes
    functionality for calculating vegetation indices and creating standardized
    patches for deep learning applications.
    """
    
    def __init__(self, root_dir, region_bounds, start_date, end_date, transform=None,
                 patch_size=64, cloud_cover_threshold=20, force_reload=False):
        """
        Initialize the CerradoDataset.

        Args:
            root_dir (str or Path): Directory to store downloaded and cached data
            region_bounds (dict): Geographic bounds with keys 'minx', 'miny', 'maxx', 'maxy'
            start_date (str): Start date for imagery collection (YYYY-MM-DD)
            end_date (str): End date for imagery collection (YYYY-MM-DD)
            transform (callable, optional): Transform to apply to the data
            patch_size (int): Size of image patches to create
            cloud_cover_threshold (int): Maximum acceptable cloud cover percentage
            force_reload (bool): If True, ignore cached data and reload from source
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.transform = transform
        self.patch_size = patch_size
        
        # Create cache directory structure
        config = {
            'bounds': region_bounds,
            'start_date': start_date,
            'end_date': end_date,
            'patch_size': patch_size,
            'cloud_cover': cloud_cover_threshold
        }
        self.cache_id = self._generate_cache_id(config)
        self.cache_dir = self.root_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.data_cache_path = self.cache_dir / f'data_{self.cache_id}.pt'
        self.metadata_cache_path = self.cache_dir / f'metadata_{self.cache_id}.json'
        
        print("\nInitializing Cerrado Dataset")
        print("============================")
        
        # Either load from cache or process from source
        if not force_reload and self._check_cache():
            print("Loading data from cache...")
            self._load_from_cache()
        else:
            print("Processing data from source...")
            self._process_from_source(region_bounds, start_date, end_date, cloud_cover_threshold)

    def _generate_cache_id(self, config):
        """Generate a unique identifier for the dataset configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _check_cache(self):
        """Check if valid cached data exists."""
        if self.data_cache_path.exists() and self.metadata_cache_path.exists():
            try:
                with open(self.metadata_cache_path, 'r') as f:
                    metadata = json.load(f)
                return True
            except Exception:
                return False
        return False

    def _save_to_cache(self):
        """Save processed data and metadata to cache."""
        try:
            torch.save(self.data, self.data_cache_path)
            
            metadata = {
                'creation_date': datetime.now().isoformat(),
                'num_patches': len(self.data),
                'patch_size': self.patch_size,
                'last_update': datetime.now().isoformat()
            }
            
            with open(self.metadata_cache_path, 'w') as f:
                json.dump(metadata, f)
                
            print(f"Data cached successfully at {self.cache_dir}")
        except Exception as e:
            print(f"Warning: Failed to cache data: {str(e)}")

    def _load_from_cache(self):
        """Load processed data from cache."""
        try:
            self.data = torch.load(self.data_cache_path)
            with open(self.metadata_cache_path, 'r') as f:
                metadata = json.load(f)
            print(f"Successfully loaded {len(self.data)} patches from cache")
            print(f"Cache creation date: {metadata['creation_date']}")
        except Exception as e:
            raise RuntimeError(f"Error loading cached data: {str(e)}")

    def _process_from_source(self, region_bounds, start_date, end_date, cloud_cover_threshold):
        """Process data with robust error handling."""
        try:
            self.bbox = BoundingBox(
                left=float(region_bounds['minx']),
                bottom=float(region_bounds['miny']),
                right=float(region_bounds['maxx']),
                top=float(region_bounds['maxy'])
            )
            
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace
            )
            
            # Get first 5 items for testing
            items = list(catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=[self.bbox.left, self.bbox.bottom, self.bbox.right, self.bbox.top],
                datetime=[start_date, end_date],
                query={"eo:cloud_cover": {"lt": cloud_cover_threshold}},
                limit=5
            ).get_items())
            
            if not items:
                raise ValueError("No imagery found")
                
            print(f"Loading {len(items)} images...")
            
            # Simplified stacking call without max_retries
            self.data_stack = stackstac.stack(
                items,
                assets=["B02", "B03", "B04", "B08"],  # Core bands only
                epsg=4326,
                resolution=10,
                bounds=[self.bbox.left, self.bbox.bottom, self.bbox.right, self.bbox.top],
                dtype='float64',
                rescale=False,
                chunksize=2048
            )
            
            self._process_data()
            self._save_to_cache()
            
        except Exception as e:
            print(f"Full error: {str(e)}")
            raise RuntimeError(f"Error during data processing: {str(e)}")
        
    def _process_data(self):
        """Process data with error checking."""
        try:
            raw_data = self.data_stack.to_numpy()
            raw_data = np.nan_to_num(raw_data, nan=0.0)
            raw_data = raw_data.astype(np.float32) / 10000.0
            
            self.data = []
            for i in range(0, raw_data.shape[1] - self.patch_size, self.patch_size):
                for j in range(0, raw_data.shape[2] - self.patch_size, self.patch_size):
                    patch = raw_data[0, i:i+self.patch_size, j:j+self.patch_size, :]
                    if np.mean(patch == 0) <= 0.5:
                        self.data.append(torch.from_numpy(patch.copy()))
                        
            print(f"Created {len(self.data)} patches")
            
        except Exception as e:
            raise RuntimeError(f"Data processing error: {str(e)}")

    def __len__(self):
        """Return the number of patches in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a specific patch from the dataset.
        
        Args:
            idx (int): Index of the desired patch
            
        Returns:
            torch.Tensor: The processed patch data
        """
        if self.transform:
            return self.transform(self.data[idx])
        return self.data[idx]

    def visualize_patch(self, idx):
        """
        Visualize a specific patch as an RGB image.
        
        Args:
            idx (int): Index of the patch to visualize
        """
        if idx >= len(self.data):
            raise IndexError("Patch index out of range")
            
        patch = self.data[idx].numpy()
        rgb = patch[..., [2, 1, 0]]  # Extract RGB bands
        
        # Normalize for visualization
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.title(f'Patch {idx}')
        plt.axis('off')
        plt.show()

class CerradoAutoencoder(nn.Module):
    """Autoencoder architecture for Cerrado vegetation pattern analysis."""
    
    def __init__(self, input_channels=12):
        super(CerradoAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, input_channels, 2, stride=2),
            nn.Tanh()
        )
        
        # Embedding network
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def get_embedding(self, x):
        z = self.encode(x)
        return self.embedding(z)
    
    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        embedding = self.embedding(z)
        return reconstructed, embedding

class TensorBoardLogger:
    """Handles logging of metrics and visualizations to TensorBoard."""
    
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.step = 0
    
    def log_training_metrics(self, metrics, images=None):
        """Log training metrics and visualizations."""
        for name, value in metrics.items():
            self.writer.add_scalar(f'training/{name}', value, self.step)
        
        if images is not None:
            self.writer.add_images('training/examples', images, self.step)
        
        if 'embeddings' in metrics:
            fig = plt.figure(figsize=(10, 6))
            plt.hist(metrics['embeddings'].cpu().numpy().flatten(), bins=50)
            plt.title('Embedding Distribution')
            self.writer.add_figure('training/embedding_distribution', fig, self.step)
            plt.close()
        
        self.step += 1
    
    def log_evaluation_metrics(self, metrics, confusion_matrix=None):
        """Log evaluation metrics and confusion matrix."""
        for name, value in metrics.items():
            self.writer.add_scalar(f'evaluation/{name}', value, self.step)
        
        if confusion_matrix is not None:
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt='d')
            plt.title('Clustering Confusion Matrix')
            self.writer.add_figure('evaluation/confusion_matrix', fig, self.step)
            plt.close()
    
    def close(self):
        self.writer.close()

def train_epoch(model, train_loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    reconstruction_loss = nn.MSELoss()
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        reconstructed, embedding = model(batch)
        loss = reconstruction_loss(reconstructed, batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return {
        'loss': total_loss / len(train_loader),
        'embeddings': embedding.detach()
    }

def evaluate_model(model, val_loader, device):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0
    reconstruction_loss = nn.MSELoss()
    embeddings = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            reconstructed, embedding = model(batch)
            loss = reconstruction_loss(reconstructed, batch)
            total_loss += loss.item()
            embeddings.append(embedding.cpu().numpy())
    
    embeddings = np.concatenate(embeddings)
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, clusters)
    
    return {
        'val_loss': total_loss / len(val_loader),
        'silhouette_score': silhouette
    }

def create_visualization_dashboard(model, dataset, device):
    """Create an interactive dashboard for result visualization."""
    app = dash.Dash(__name__)
    
    # Get embeddings and reconstructions
    embeddings = []
    reconstructions = []
    
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=32):
            batch = batch.to(device)
            recon, emb = model(batch)
            embeddings.append(emb.cpu().numpy())
            reconstructions.append(recon.cpu().numpy())
    
    embeddings = np.concatenate(embeddings)
    reconstructions = np.concatenate(reconstructions)
    
    # Reduce dimensionality
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create clusters
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Dashboard layout
    app.layout = html.Div([
        html.H1('Cerrado Vegetation Monitor'),
        
        html.Div([
            dcc.Graph(
                id='embedding-plot',
                figure=px.scatter(
                    x=embeddings_2d[:, 0],
                    y=embeddings_2d[:, 1],
                    color=clusters.astype(str),
                    title='Vegetation Pattern Clusters'
                )
            ),
            
            dcc.Graph(
                id='time-series',
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            y=[np.mean(batch) for batch in reconstructions],
                            mode='lines',
                            name='Average Vegetation Index'
                        )
                    ],
                    layout=dict(title='Temporal Vegetation Patterns')
                )
            )
        ], style={'display': 'flex'}),
        
        html.Div([
            dcc.Dropdown(
                id='metric-selector',
                options=[
                    {'label': 'NDVI', 'value': 'ndvi'},
                    {'label': 'SAVI', 'value': 'savi'},
                    {'label': 'Raw Bands', 'value': 'raw'}
                ],
                value='ndvi'
            ),
            dcc.RangeSlider(
                id='date-range',
                min=0,
                max=365,
                step=1,
                value=[0, 365],
                marks={0: 'Jan', 90: 'Apr', 180: 'Jul', 270: 'Oct', 365: 'Dec'}
            )
        ])
    ])
    
    return app

def main():
    """
    Main function to run the complete Cerrado vegetation monitoring pipeline.
    Handles data loading, model training, visualization, and interactive dashboard.
    """
    # Set up logging directory and format
    logging_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f'runs/cerrado_monitor_{logging_time}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Cerrado Vegetation Monitoring System ===")
    print(f"Starting new run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Logging results to: {log_dir}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Define study region (Central Cerrado near Brasília)
    region = {
        'minx': float(-47.0),  # Western boundary
        'miny': float(-15.0),  # Southern boundary
        'maxx': float(-46.0),  # Eastern boundary
        'maxy': float(-14.0)   # Northern boundary
    }
    
    # Validate geographic coordinates
    try:
        if not (-180 <= region['minx'] <= 180 and
                -90 <= region['miny'] <= 90 and
                -180 <= region['maxx'] <= 180 and
                -90 <= region['maxy'] <= 90):
            raise ValueError("Coordinates must be in valid ranges: longitude [-180, 180], latitude [-90, 90]")
        
        if region['minx'] >= region['maxx'] or region['miny'] >= region['maxy']:
            raise ValueError("Invalid region bounds: 'min' coordinates must be less than 'max' coordinates")
    except ValueError as e:
        print(f"\nError in region specification: {str(e)}")
        return

    print("\nStudy Region:")
    print(f"Longitude: {region['minx']}° to {region['maxx']}°")
    print(f"Latitude: {region['miny']}° to {region['maxy']}°")
    
    try:
        # Initialize dataset
        print("\nInitializing dataset...")
        dataset = CerradoDataset(
            root_dir='data/cerrado',
            region_bounds=region,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        if len(dataset) == 0:
            raise ValueError("No valid data patches were created. Please check region bounds and dates.")
        
        print(f"Successfully created dataset with {len(dataset)} patches")
        
        # Split dataset into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        print(f"\nDataset split:")
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        # Create data loaders with error handling
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
        except Exception as e:
            print(f"\nError creating data loaders: {str(e)}")
            return
        
        # Initialize model and move to device
        print("\nInitializing model...")
        model = CerradoAutoencoder().to(device)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Initialize TensorBoard logger
        logger = TensorBoardLogger(log_dir)
        
        # Training configuration
        num_epochs = 100
        best_val_loss = float('inf')
        patience = 10  # For early stopping
        patience_counter = 0
        
        print("\nStarting training...")
        print(f"Training for {num_epochs} epochs")
        
        try:
            for epoch in range(num_epochs):
                # Training phase
                train_metrics = train_epoch(model, train_loader, optimizer, device)
                logger.log_training_metrics(train_metrics)
                
                # Evaluation phase
                if epoch % 5 == 0:
                    eval_metrics = evaluate_model(model, val_loader, device)
                    logger.log_evaluation_metrics(eval_metrics)
                    
                    # Save best model
                    if eval_metrics['val_loss'] < best_val_loss:
                        best_val_loss = eval_metrics['val_loss']
                        torch.save(model.state_dict(), log_dir / 'best_model.pth')
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Early stopping check
                    if patience_counter >= patience:
                        print("\nEarly stopping triggered - validation loss not improving")
                        break
                    
                    print(f"\nEpoch {epoch+1}/{num_epochs}:")
                    print(f"Training Loss: {train_metrics['loss']:.4f}")
                    print(f"Validation Loss: {eval_metrics['val_loss']:.4f}")
                    print(f"Silhouette Score: {eval_metrics['silhouette_score']:.4f}")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
        finally:
            # Always close the logger
            logger.close()
        
        print("\nTraining completed")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Load best model for visualization
        print("\nPreparing visualization dashboard...")
        model.load_state_dict(torch.load(log_dir / 'best_model.pth'))
        model.eval()
        
        # Create and run dashboard
        app = create_visualization_dashboard(model, dataset, device)
        
        print("\nStarting visualization services:")
        print("1. Dashboard: http://localhost:8050")
        print("2. TensorBoard: http://localhost:6006")
        print("   (Run 'tensorboard --logdir=runs' in another terminal)")
        
        # Run the dashboard
        app.run_server(debug=True)
        
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        print("Please check your internet connection and try again")
        return

if __name__ == '__main__':
    main()