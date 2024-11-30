"""
ModelManager class that handles model loading and saving without direct model creation.
Uses dependency injection to allow flexible model initialization strategies.
"""

from typing import Callable, Optional, Union, Type
import torch
import torch.nn as nn
from pathlib import Path
import logging

ModelCreator = Union[Type[nn.Module], Callable[..., nn.Module]]

class ModelManager:
    """
    Manages model persistence without knowing model implementation details.
    
    The class can receive either a model class (constructor) or a factory function
    that creates model instances. This allows for flexible model creation strategies
    while keeping the manager focused on persistence concerns.
    
    Examples:
        # Using a model class (constructor):
        manager = ModelManager(Autoencoder, device='cuda', num_classes=10)
        
        # Using a factory function:
        def create_model():
            return Autoencoder(num_classes=10).to('cuda')
        manager = ModelManager(create_model)
    """
    
    def __init__(
        self,
        model_creator: ModelCreator,
        base_dir: str = 'output',
        from_scratch: bool = False,
        **model_kwargs
    ):
        """
        Initialize the model manager with a model creation strategy.
        
        Args:
            model_creator: Either a model class or a factory function that creates models
            base_dir: Directory for model persistence
            from_scratch: Whether to ignore existing saved models
            model_kwargs: Additional arguments passed to model creation
        """
        self.base_dir = Path(base_dir)
        self.from_scratch = from_scratch
        self.model_creator = model_creator
        self.model_kwargs = model_kwargs
        self._log = logging.getLogger(__name__)

    def create_model(self) -> nn.Module:
        """
        Create a new model instance using the provided creation strategy.
        """
        if isinstance(self.model_creator, type):
            # If model_creator is a class, instantiate it with kwargs
            return self.model_creator(**self.model_kwargs)
        else:
            # If model_creator is a factory function, call it with kwargs
            return self.model_creator(**self.model_kwargs)

    def get_last_model_path(self) -> Optional[Path]:
        """Find the most recent model file in the output directory."""
        if not self.base_dir.exists():
            return None
        
        # Look for the most recent model file
        model_files = list(self.base_dir.rglob('model.pth'))
        if not model_files:
            return None
            
        return max(model_files, key=lambda p: p.stat().st_mtime)

    def load_model(self) -> nn.Module:
        """
        Create a new model and optionally load saved weights.
        
        Returns:
            A PyTorch model, either newly initialized or loaded from saved weights
        """
        # Create new model instance
        model = self.create_model()
        
        # Load saved weights if available and desired
        if not self.from_scratch:
            model_path = self.get_last_model_path()
            if model_path:
                try:
                    model.load_state_dict(torch.load(model_path))
                    print(f"Model loaded from {model_path}")
                except Exception as e:
                    print(f"Failed to load model from {model_path}: {e}")
        
        return model

    def save_model(self, model: nn.Module, version: int) -> None:
        """
        Save model weights to disk.
        
        Args:
            model: The PyTorch model to save
            version: Version number for the save file
        """
        save_dir = self.base_dir / str(version)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / 'model.pth'
        torch.save(model.state_dict(), save_path)
        self._log.info(f"Model saved to {save_path}")