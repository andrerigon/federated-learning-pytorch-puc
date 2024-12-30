import threading
import torch
import torch.nn as nn
import logging
import time
import math
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
from abc import ABC, abstractmethod
from collections import OrderedDict
from loguru import logger
from typing import List, Optional, Dict
import traceback

class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, alpha=0.1, extra_info=None) -> OrderedDict:
        """
        Aggregates the client model parameters into the global model parameters.
        
        Args:
            global_model (nn.Module): The global model before aggregation.
            client_dict (OrderedDict or List[OrderedDict]): State dict(s) from the client(s).
            alpha (float): Blending factor or other weighting parameter.
            extra_info (dict): Additional info like staleness if needed.
        
        Returns:
            OrderedDict: The updated global state dictionary after aggregation.
        """
        pass

class AlphaWeightedStrategy(AggregationStrategy):
    """
    A simple incremental aggregation approach that linearly combines the global model
    and the incoming client model using a fixed alpha.
    """
    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, alpha=0.1, extra_info=None) -> OrderedDict:
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            if k in client_dict:
                if global_dict[k].size() == client_dict[k].size():
                    global_dict[k] = (1 - alpha) * global_dict[k] + alpha * client_dict[k].dequantize()
                else:
                    print(f"Size mismatch for key {k}. Skipping aggregation for this key.")
            else:
                print(f"Key {k} not found in client model. Skipping aggregation for this key.")
        return global_dict

class FedAvgStrategy(AggregationStrategy):
    """
    FedAvg: Synchronous aggregation strategy.
    Assumes client_dict is a list of state dicts from multiple clients.
    Uses a simple average (equal weighting) if data distribution is IID.
    """
    def aggregate(self, global_model: nn.Module, client_dict: list, alpha=0.1, extra_info=None) -> OrderedDict:
        global_dict = global_model.state_dict()
        keys = global_dict.keys()
        num_clients = len(client_dict)
        
        if num_clients == 0:
            return global_dict

        avg_dict = OrderedDict((k, torch.zeros_like(global_dict[k])) for k in keys)

        # Sum all updates
        for cdict in client_dict:
            for k in keys:
                avg_dict[k] += cdict[k]

        # Divide to get mean
        for k in keys:
            avg_dict[k] = avg_dict[k] / num_clients

        return avg_dict

class AsyncFedAvgStrategy(AggregationStrategy):
    """
    Asynchronous FedAvg: Handles staleness with adaptive weights.
    Reference: Asynchronous Federated Optimization
    DOI: 10.48550/arXiv.1903.03934
    """
    def __init__(self, staleness_threshold: int = 10, base_mixing_weight: float = 0.1):
        self.staleness_threshold = staleness_threshold
        self.last_update_time = {}
        # Base mixing weight determines how much we trust any client update
        self.base_mixing_weight = base_mixing_weight
        
    def _calculate_staleness_weight(self, client_id: str, current_time: float) -> float:
        """
        Calculate staleness-adjusted mixing weight.
        Returns a value between 0.1 * base_mixing_weight and base_mixing_weight
        """
        if client_id not in self.last_update_time:
            # First update still uses base_mixing_weight
            return 0.9 #self.base_mixing_weight
            
        staleness = current_time - self.last_update_time[client_id]
        staleness_factor = 1.0 / (1.0 + staleness/self.staleness_threshold)
        
        # Scale the base mixing weight by staleness
        return self.base_mixing_weight * max(0.1, staleness_factor)
        
    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, 
                 alpha: float = 0.1, extra_info: dict = None) -> OrderedDict:
        if not extra_info or 'client_id' not in extra_info:
            return global_model.state_dict()
            
        current_time = time.time()
        client_id = extra_info['client_id']
        
        # Get staleness-adjusted mixing weight
        mixing_weight = self._calculate_staleness_weight(client_id, current_time)
        
        # Update timestamp
        self.last_update_time[client_id] = current_time
        
        # Perform weighted update
        global_dict = global_model.state_dict()
        new_dict = OrderedDict()

        logger.debug(f"Client {client_id} will use weight {mixing_weight}")
        
        for k, v in global_dict.items():
            # Now the global model always retains (1 - mixing_weight) of its current state
            new_dict[k] = (1 - mixing_weight) * v + mixing_weight * client_dict[k]
            
        return new_dict

class SAFAStrategy(AggregationStrategy):
    """
    SAFA: Server-Aided Federated Analytics with differential privacy.
    Reference: SAFA: a Semi-Asynchronous Protocol for Fast Federated Learning
    DOI: 10.1109/TC.2020.2994391
    
    This strategy provides differential privacy guarantees by:
    1. Adding calibrated Gaussian noise to client updates
    2. Ensuring minimum participation thresholds
    3. Managing update timing to balance privacy and model freshness
    """
    def __init__(self, 
                 epsilon: float = 1.0, 
                 delta: float = 1e-5,
                 total_clients: int = 100,
                 timeout_seconds: float = 30,
                 initial_mixing_weight: float = 0.9,  # Higher initial weight
                 final_mixing_weight: float = 0.1): 
        
         # Privacy parameters
        self.epsilon = epsilon
        self.delta = delta
        
        # First, calculate minimum clients needed for privacy
        privacy_min = int(math.ceil(math.sqrt(2 * math.log(1.25/delta)) / epsilon))
        
        # Calculate system minimum (at least 2 clients or 10% of total)
        system_min = max(2, int(0.1 * total_clients))
        
        # Take the larger of privacy and system minimums, but don't exceed total
        self.min_clients = min(max(privacy_min, system_min), total_clients)
        
        # Maximum should be larger than minimum but not exceed total
        self.max_clients = min(total_clients, max(self.min_clients, int(0.8 * total_clients)))
        
        # Validate the thresholds
        if self.max_clients < self.min_clients:
            logger.warning("Maximum clients less than minimum. Adjusting max to equal min.")
            self.max_clients = self.min_clients
        
        self.last_aggregation_time = time.time()
        self.current_updates = 0
        self.mixing_weight = 0.1
        self.timeout_seconds = timeout_seconds

        self.initial_mixing_weight = initial_mixing_weight
        self.final_mixing_weight = final_mixing_weight
        self.mixing_weight = initial_mixing_weight  # Start with higher weight
        self.aggregation_count = 0  # Track number of successful aggregations
        
        logger.info(f"SAFA initialized with privacy (ε={epsilon}, δ={delta})")
        logger.info(f"Client thresholds: min={self.min_clients}, max={self.max_clients} "
                    f"out of {total_clients} total clients")
    
    def _adjust_mixing_weight(self):
        """
        Gradually decrease mixing weight from initial to final value
        over the first few aggregations.
        """
        if self.aggregation_count < 5:  # Transition period
            # Linearly decrease weight over first 5 aggregations
            progress = self.aggregation_count / 5
            self.mixing_weight = (self.initial_mixing_weight * (1 - progress) + 
                                self.final_mixing_weight * progress)
        else:
            self.mixing_weight = self.final_mixing_weight

    def _should_aggregate(self, extra_info: Optional[Dict] = None) -> bool:
        """
        Determines if we should perform aggregation based on:
        1. Having enough updates for privacy guarantees
        2. Either reaching max clients or timeout
        """
        current_time = time.time()
        timeout_reached = (current_time - self.last_aggregation_time) > self.timeout_seconds
        
        if self.current_updates < self.min_clients:
            return False
            
        return self.current_updates >= self.max_clients or timeout_reached
        
    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, 
                 alpha: float = 0.1, extra_info: Optional[Dict] = None) -> OrderedDict:
        """
        Aggregates client updates with differential privacy guarantees.
        
        The aggregation process:
        1. Updates client counter and checks aggregation conditions
        2. Calculates noise scale based on privacy parameters
        3. Adds calibrated noise to client updates
        4. Performs weighted averaging with noisy updates
        
        Args:
            global_model: Current global model
            client_dict: State dictionary from client
            alpha: Learning rate (unused in this implementation)
            extra_info: Additional information like client ID
        """
        # Track this update
        self.current_updates += 1
        
        # Check if we should aggregate yet
        if not self._should_aggregate(extra_info):
            logger.info("Not aggregating this time")
            return global_model.state_dict()
            
        logger.info("Will aggregate this time")    
        # Get global model state
        global_dict = global_model.state_dict()
        new_dict = OrderedDict()
        
        # Calculate privacy-preserving noise scale
        # Scale noise based on number of participating clients
        base_sigma = math.sqrt(2 * math.log(1.25/self.delta)) / self.epsilon
        sigma = base_sigma * (1.0 / self.current_updates)  # Scale by participation
        
        try:
            # Process each parameter with privacy-preserving noise
            for k, v in global_dict.items():
                noise = torch.normal(0, sigma, client_dict[k].shape, 
                                  device=client_dict[k].device)
                
                # Use current mixing weight
                new_dict[k] = ((1 - self.mixing_weight) * v + 
                              self.mixing_weight * (client_dict[k] + noise))
                              
            # Update aggregation state
            self.aggregation_count += 1
            self._adjust_mixing_weight()
            
            logger.info(f"Aggregated update with mixing weight {self.mixing_weight:.3f}")
            return new_dict
            
        except Exception as e:
            logger.error(f"Error during aggregation: {str(e)}")
            return global_dict
    
class RELAYStrategy(AggregationStrategy):
    """
    RELAY: Focuses on resource efficiency and uses a staleness-aware scaling.
    Prioritizes least-available learners and scales updates by similarity or staleness.
    
    As a placeholder, we do staleness-based scaling similar to AsyncFedAvg.
    """
    def __init__(self, base_alpha: float = 0.1):
        self.base_alpha = base_alpha

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, alpha=0.1, extra_info=None) -> OrderedDict:
        global_dict = global_model.state_dict()
        staleness = extra_info.get('staleness', 0) if extra_info else 0
        effective_alpha = self.base_alpha / (1 + staleness)
        for k in global_dict.keys():
            if k in client_dict and global_dict[k].size() == client_dict[k].size():
                global_dict[k] = (1 - effective_alpha)*global_dict[k] + effective_alpha*client_dict[k].dequantize()
        return global_dict

class AstraeaStrategy(AggregationStrategy):
    """
    Astraea: A sequential round training approach. 
    In a real Astraea scenario:
    - One client updates model
    - The mediator updates params
    - Next client trains and so forth

    Here, we simulate a simple sequential blending with a fixed alpha each time.
    """
    def __init__(self, base_alpha: float = 0.1):
        self.base_alpha = base_alpha

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, alpha=0.1, extra_info=None) -> OrderedDict:
        global_dict = global_model.state_dict()
        effective_alpha = self.base_alpha
        for k in global_dict.keys():
            if k in client_dict and global_dict[k].size() == client_dict[k].size():
                global_dict[k] = (1 - effective_alpha)*global_dict[k] + effective_alpha*client_dict[k].dequantize()
        return global_dict

class TimeWeightedStrategy(AggregationStrategy):
    """
    TimeWeighted: Weight updates by time or device capabilities.
    For demonstration, use staleness-based scaling like AsyncFedAvg.
    """
    def __init__(self, base_alpha: float = 0.1):
        self.base_alpha = base_alpha

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, alpha=0.1, extra_info=None) -> OrderedDict:
        global_dict = global_model.state_dict()
        staleness = extra_info.get('staleness', 0) if extra_info else 0
        effective_alpha = self.base_alpha / (1 + staleness)
        for k in global_dict.keys():
            if k in client_dict and global_dict[k].size() == client_dict[k].size():
                global_dict[k] = (1 - effective_alpha)*global_dict[k] + effective_alpha*client_dict[k].dequantize()
        return global_dict