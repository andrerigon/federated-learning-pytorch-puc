"""
Utility functions for serializing and deserializing PyTorch state dictionaries.
These functions handle the compression and encoding of model parameters for efficient
transmission over networks or storage.

The process involves multiple steps of compression and encoding:
1. Converting PyTorch tensors to bytes
2. Compressing the bytes using gzip
3. Encoding the compressed data in base64 for safe transmission
"""

import base64
from io import BytesIO
import gzip
import json
import torch
from typing import Dict, Any
from loguru import logger

def serialize_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Serialize a PyTorch state dictionary into a compressed, base64-encoded JSON string.
    
    This function performs several transformations:
    1. Converts the PyTorch state dict to a binary format
    2. Compresses it using gzip to reduce size
    3. Encodes it in base64 for safe transmission
    4. Wraps the result in JSON for additional safety
    
    Args:
        state_dict: A PyTorch state dictionary containing model parameters
        
    Returns:
        str: A JSON string containing the base64-encoded compressed state dictionary
        
    Example:
        >>> model = torch.nn.Linear(10, 10)
        >>> encoded = serialize_state_dict(model.state_dict())
    """
    # First convert the state dict to bytes using PyTorch's save function
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    
    # Compress the bytes using gzip
    compressed_buffer = BytesIO()
    with gzip.GzipFile(fileobj=compressed_buffer, mode='wb') as f:
        f.write(buffer.getvalue())
    
    # Convert compressed data to base64 for safe transmission
    compressed_data = compressed_buffer.getvalue()
    compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
    
    # Log the compressed size for monitoring
    logger.debug(f"Serialized and compressed state_dict size: {len(compressed_data)} bytes")
    
    # Wrap in JSON for additional safety and transmission compatibility
    result = json.dumps(compressed_base64)
    
    # Clean up to prevent memory leaks
    del compressed_data, compressed_base64, buffer, compressed_buffer
    
    return result

def decompress_and_deserialize_state_dict(serialized_state_dict: str) -> Dict[str, torch.Tensor]:
    """
    Convert a serialized state dictionary back into a PyTorch state dictionary.
    
    Reverses the serialization process by:
    1. Decoding from JSON
    2. Decoding from base64
    3. Decompressing from gzip
    4. Loading back into PyTorch tensors
    
    Args:
        serialized_state_dict: JSON string containing base64-encoded compressed state dictionary
        
    Returns:
        Dict[str, torch.Tensor]: The original PyTorch state dictionary
        
    Raises:
        json.JSONDecodeError: If the input string is not valid JSON
        base64.binascii.Error: If the base64 decoding fails
        torch.serialization.pickle.UnpicklingError: If the PyTorch deserialization fails
        
    Example:
        >>> model = torch.nn.Linear(10, 10)
        >>> encoded = serialize_state_dict(model.state_dict())
        >>> decoded = decompress_and_deserialize_state_dict(encoded)
    """
    # Decode the JSON string back to base64
    serialized_state_dict = json.loads(serialized_state_dict)
    
    # Decode base64 back to compressed bytes
    compressed_data = base64.b64decode(serialized_state_dict.encode('utf-8'))
    compressed_buffer = BytesIO(compressed_data)
    
    # Decompress using gzip
    with gzip.GzipFile(fileobj=compressed_buffer, mode='rb') as f:
        buffer = BytesIO(f.read())
    buffer.seek(0)
    
    # Load back into PyTorch state dict
    result = torch.load(buffer, weights_only=True)
    
    # Clean up to prevent memory leaks
    del compressed_data, buffer, compressed_buffer
    
    return result