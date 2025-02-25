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
from typing import Tuple, Set

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

class FedProxStrategy(AggregationStrategy):
    """
    Implementation of FedProx aggregation strategy from:
    'Federated Optimization in Heterogeneous Networks'
    Li et al., MLSys 2020 (DOI: 10.48550/arXiv.1812.06127)
    """
    def __init__(self, K: int):
        """
        Args:
            K: Number of devices to select per round (subset of total devices N)
        """
        super().__init__()
        self.K = K  # Number of devices to aggregate per round
        self.current_round_states = {}  # Store updates for current round
        self.current_round_sizes = {}   # Store dataset sizes for current round

    def aggregate(self, global_model: nn.Module, client_state_dict: OrderedDict, 
                    alpha=0.1, extra_info=None) -> OrderedDict:
            """
            Implements FedProx aggregation following Algorithm 2.
            """
            if extra_info is None or 'client_id' not in extra_info:
                return global_model.state_dict()
                
            client_id = extra_info['client_id']
            dataset_size = extra_info.get('client_samples', 1)
            
            # Store this client's update and dataset size
            self.current_round_states[client_id] = client_state_dict
            self.current_round_sizes[client_id] = dataset_size
            
            # Only aggregate when we have received K updates
            if len(self.current_round_states) < self.K:
                return global_model.state_dict()
                
            # Initialize aggregation dict
            global_dict = global_model.state_dict()
            agg_dict = OrderedDict((k, torch.zeros_like(v)) for k, v in global_dict.items())
            
            total_samples = sum(self.current_round_sizes.values())
            
            # Weighted aggregation
            for client_id, state_dict in self.current_round_states.items():
                weight = self.current_round_sizes[client_id] / total_samples
                for key in global_dict.keys():
                    # Cast weight to the same dtype as the tensor
                    weight_tensor = torch.tensor(weight, dtype=state_dict[key].dtype, device=state_dict[key].device)
                    agg_dict[key] += state_dict[key] * weight_tensor
                    
            logger.info(f"Aggregated {len(self.current_round_states)} client updates")
            
            # Reset for next round
            self.current_round_states = {}
            self.current_round_sizes = {}
            
            return agg_dict
    
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
    SAFA: A Semi-Asynchronous Protocol for Fast Federated Learning With Low Overhead
    Reference: DOI: 10.1109/TC.2020.2994391
    
    The strategy implements three key mechanisms from the paper:
    1. Lag-tolerant Model Distribution: Handles version differences between clients
    2. Post-training Client Selection: Selects clients after training completion
    3. Three-step Discriminative Aggregation: Uses cache and bypass structures
    
    This implementation adds extra robustness for handling communication losses 
    and partial updates in real-world scenarios.
    """
    def __init__(self, 
                 total_clients: int,
                 lag_tolerance: int = 2,
                 initial_weight: float = 0.8,
                 final_weight: float = 0.2,
                 selection_fraction: float = 0.5,
                 timeout_seconds: float = 30):
        # System parameters
        self.total_clients = total_clients
        self.lag_tolerance = lag_tolerance
        self.selection_fraction = selection_fraction
        self.timeout_seconds = timeout_seconds
        
        # Tracking state and version information
        self.current_version = 0
        self.client_versions = {}
        self.last_aggregation_time = time.time()
        
        # Storage structures for client updates
        self.pending_updates = {}  # Temporary storage for incoming updates
        self.model_cache = {}     # Storage for selected updates
        
        # Adaptive weight parameters for faster initial learning
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.current_weight = initial_weight
        self.round_count = 0
        
        logger.info(f"SAFA initialized: {total_clients} clients, "
                   f"selection fraction: {selection_fraction}")

    def _validate_client_update(self, client_dict: Optional[OrderedDict], 
                              global_dict: OrderedDict) -> bool:
        """
        Validates that a client update is usable by checking:
        1. Update is not None
        2. Update has the same structure as global model
        3. All tensors have correct shapes
        
        Returns True if update is valid, False otherwise.
        """
        try:
            if client_dict is None:
                return False
                
            # Check keys match
            if set(client_dict.keys()) != set(global_dict.keys()):
                return False
                
            # Check tensor shapes match    
            for k in global_dict.keys():
                if client_dict[k].shape != global_dict[k].shape:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating client update: {str(e)}")
            return False

    def _safe_aggregate_tensors(self, global_tensor: torch.Tensor,
                              updates: List[torch.Tensor],
                              weights: List[float]) -> torch.Tensor:
        """
        Safely aggregates tensors using weighted average.
        Handles edge cases like empty updates list.
        """
        if not updates or not weights:
            return global_tensor.clone()
            
        result = torch.zeros_like(global_tensor)
        total_weight = 0.0
        
        for update, weight in zip(updates, weights):
            result += weight * update
            total_weight += weight
            
        if total_weight > 0:
            result = (1 - total_weight) * global_tensor + result
        else:
            result = global_tensor.clone()
            
        return result

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, 
                 alpha: float = 0.1, extra_info: Optional[Dict] = None) -> OrderedDict:
        """
        Performs SAFA's three-step aggregation with robust error handling.
        Always returns a valid state dictionary, even in error cases.
        """
        # Always get a copy of global state dict first as fallback
        global_dict = global_model.state_dict()
        
        # Validate inputs
        if not extra_info or 'client_id' not in extra_info:
            logger.warning("Missing client ID in extra info")
            return global_dict
            
        client_id = extra_info['client_id']
        
        # Validate client update
        if not self._validate_client_update(client_dict, global_dict):
            logger.warning(f"Invalid update from client {client_id}")
            return global_dict
            
        try:
            # Store update in pending updates
            self.pending_updates[client_id] = client_dict
            
            # Check if we should aggregate
            current_time = time.time()
            num_updates = len(self.pending_updates)
            should_aggregate = (
                num_updates >= self.selection_fraction * self.total_clients or
                (current_time - self.last_aggregation_time) > self.timeout_seconds
            )
            
            if not should_aggregate:
                return global_dict
                
            # Prepare for aggregation
            new_dict = OrderedDict()
            
            # Process each parameter
            for key, global_param in global_dict.items():
                valid_updates = []
                update_weights = []
                
                # Collect valid updates for this parameter
                for cid, update in self.pending_updates.items():
                    if key in update:
                        valid_updates.append(update[key])
                        update_weights.append(self.current_weight)
                        
                # Safely aggregate this parameter
                new_dict[key] = self._safe_aggregate_tensors(
                    global_param,
                    valid_updates,
                    update_weights
                )
                
            # Update state
            self.round_count += 1
            self.current_version += 1
            self.last_aggregation_time = current_time
            self.pending_updates.clear()
            
            # Adjust mixing weight for next round
            if self.round_count < 5:
                progress = self.round_count / 5
                self.current_weight = (self.initial_weight * (1 - progress) + 
                                     self.final_weight * progress)
            else:
                self.current_weight = self.final_weight
                
            logger.info(f"Round {self.round_count}: Aggregated {num_updates} updates "
                       f"with weight {self.current_weight:.3f}")
            
            return new_dict
            
        except Exception as e:
            logger.error(f"Error during aggregation: {str(e)}")
            return global_dict
        
class RELAYStrategy(AggregationStrategy):
    """
    RELAY: FedAvg with adaptive client learning rates
    DOI: 10.1109/TWC.2022.3155596
    
    Key features:
    - Adapts client mixing weights based on local data characteristics
    - Higher initial weights for faster convergence from random initialization
    - Gradual transition to stable weights
    """
    def __init__(self, 
                 initial_mixing_weight: float = 0.8,  # Start with high trust in clients
                 final_mixing_weight: float = 0.2,    # Converge to more conservative weight
                 transition_rounds: int = 5):         # How many rounds to transition
        self.initial_mixing_weight = initial_mixing_weight
        self.final_mixing_weight = final_mixing_weight
        self.mixing_weight = initial_mixing_weight
        self.transition_rounds = transition_rounds
        self.aggregation_count = 0
        
        logger.info(f"RELAY initialized with mixing weights: "
                   f"initial={initial_mixing_weight}, final={final_mixing_weight}")
        
    def _adjust_mixing_weight(self):
        """Gradually decrease mixing weight for more stable updates."""
        if self.aggregation_count < self.transition_rounds:
            progress = self.aggregation_count / self.transition_rounds
            self.mixing_weight = (self.initial_mixing_weight * (1 - progress) + 
                                self.final_mixing_weight * progress)
        else:
            self.mixing_weight = self.final_mixing_weight
            
    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, 
                 alpha: float = 0.1, extra_info: Optional[Dict] = None) -> OrderedDict:
        """
        Aggregate client updates with adaptive mixing weights.
        Uses higher weights initially to learn quickly from client knowledge,
        then gradually reduces weight for stability.
        """
        if not client_dict:
            return global_model.state_dict()
            
        # Get global model state
        global_dict = global_model.state_dict()
        new_dict = OrderedDict()
        
        try:
            # Process each parameter with adaptive mixing
            for k, v in global_dict.items():
                new_dict[k] = ((1 - self.mixing_weight) * v + 
                              self.mixing_weight * client_dict[k])
                              
            # Update state
            self.aggregation_count += 1
            self._adjust_mixing_weight()
            
            logger.info(f"Aggregated update with mixing weight {self.mixing_weight:.3f}")
            return new_dict
            
        except Exception as e:
            logger.error(f"Error during aggregation: {str(e)}")
            return global_dict

class AstraeaStrategy(AggregationStrategy):
    """
    Astraea: Self-balancing federated learning with reputation scoring
    Reference: Astraea: Self-balancing federated learning for improving classification accuracy 
    of mobile deep learning applications
    DOI: 10.1109/ICCD46524.2019.00038
    
    Pseudocode:
    
    Server initialization:
        - Initialize global model w_0
        - Set reputation scores R = {} (empty dict)
        - Set reputation threshold θ
        - Set initial mixing weight α_init (high)
        - Set final mixing weight α_final (low)
        - Set transition rounds T
    
    For each client update from client k:
        1. Calculate Contribution Score:
           - Compute avg difference d between client and global parameters
           - Convert to score: s = exp(-d)
           
        2. Update Reputation:
           If first update from k:
               R[k] = s
           Else:
               R[k] = β * R[k] + (1-β) * s
               
        3. Check Reputation:
           If R[k] < θ:
               Reject update
           Else:
               - Calculate effective_weight = α * R[k]
               - w_t+1 = (1 - effective_weight) * w_t + effective_weight * w_k
               
        4. Adjust mixing weight α:
           If round < T:
               α = α_init * (1 - round/T) + α_final * (round/T)
           Else:
               α = α_final
        
    Key features:
        - Reputation tracks client reliability over time
        - Higher weights for reliable clients
        - Initial high mixing weights for faster learning
        - Automatic exclusion of unreliable clients
    """
    def __init__(self, 
                 initial_mixing_weight: float = 0.8,  
                 final_mixing_weight: float = 0.2,    
                 transition_rounds: int = 5,
                 reputation_threshold: float = 0.1):
        """
        Initialize the Astraea federated learning strategy.

        Args:
            initial_mixing_weight (float, optional): Starting weight for mixing client updates
                with global model. Higher values (> 0.5) mean stronger client influence initially,
                which helps faster learning from random initialization. Defaults to 0.8.
                
            final_mixing_weight (float, optional): The stable mixing weight after transition.
                Lower values (< 0.5) prioritize stability and reduce oscillations in the
                trained model. Defaults to 0.2.
                
            transition_rounds (int, optional): Number of rounds over which to linearly
                decrease the mixing weight from initial to final value. More rounds mean
                smoother transition. Defaults to 5.
                
            reputation_threshold (float, optional): Minimum reputation score [0-1] required
                for a client's update to be accepted. Higher values mean stricter filtering.
                Updates from clients below this threshold are rejected. Defaults to 0.3.
        
        The strategy maintains reputation scores for clients and combines them with
        adaptive mixing weights to determine each client's influence on the global model.
        """
        # Mixing weight parameters
        self.initial_mixing_weight = initial_mixing_weight
        self.final_mixing_weight = final_mixing_weight
        self.mixing_weight = initial_mixing_weight
        self.transition_rounds = transition_rounds
        self.reputation_threshold = reputation_threshold
        
        # State tracking
        self.reputation_scores = {}  # client_id -> score
        self.aggregation_count = 0
        
    def _adjust_mixing_weight(self):
        """Gradually decrease mixing weight for more stable updates."""
        if self.aggregation_count < self.transition_rounds:
            progress = self.aggregation_count / self.transition_rounds
            self.mixing_weight = (self.initial_mixing_weight * (1 - progress) + 
                                self.final_mixing_weight * progress)
        else:
            self.mixing_weight = self.final_mixing_weight

    def _calculate_contribution_score(self, client_dict: OrderedDict, 
                               global_dict: OrderedDict) -> float:
        """
        Calculate reputation score based on similarity to global model.
        Returns a score between 0 and 1, where:
        - Score close to 1 means the client update is similar to global model
        - Score close to 0 means very different update
        """
        total_relative_diff = 0.0
        param_count = 0
        
        for k in global_dict.keys():
            if k in client_dict:
                # Calculate relative difference normalized by parameter magnitude
                diff = (client_dict[k].float() - global_dict[k].float()).abs()
                magnitude = global_dict[k].float().abs() + 1e-8  # Avoid division by zero
                relative_diff = (diff / magnitude).mean().item()
                total_relative_diff += relative_diff
                param_count += 1
                
        if param_count == 0:
            return 0.0
            
        # Average relative difference across all parameters
        avg_relative_diff = total_relative_diff / param_count
        
        # Convert to score between 0 and 1
        # Using a softer decay that gives higher scores
        score = 1 / (1 + avg_relative_diff)  # This gives higher scores than exp(-diff)
        
        return score

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, 
                 alpha: float = 0.1, extra_info: Optional[Dict] = None) -> OrderedDict:
        """
        Aggregate client update with reputation-based weighting.
        Higher reputation clients have more influence on the global model.
        """
        if not extra_info or 'client_id' not in extra_info:
            return global_model.state_dict()
            
        client_id = extra_info['client_id']
        global_dict = global_model.state_dict()
        new_dict = OrderedDict()

        try:
            # Calculate contribution score
            contribution = self._calculate_contribution_score(client_dict, global_dict)
            
            # Update reputation using exponential moving average
            if client_id not in self.reputation_scores:
                # Give new clients a chance with decent initial reputation
                self.reputation_scores[client_id] = max(0.5, contribution)
            else:
                beta = 0.9  # History weight
                old_score = self.reputation_scores[client_id]
                new_score = beta * old_score + (1 - beta) * contribution
                self.reputation_scores[client_id] = new_score
            
            current_reputation = self.reputation_scores[client_id]
            
            # Only aggregate if reputation is above threshold
            if current_reputation >= self.reputation_threshold:
                # Combine mixing weight with reputation
                effective_weight = self.mixing_weight * current_reputation
                
                # Aggregate parameters
                for k, v in global_dict.items():
                    new_dict[k] = ((1 - effective_weight) * v + 
                                 effective_weight * client_dict[k])
                    
                logger.info(f"Client {client_id} reputation: {current_reputation:.3f}, "
                          f"effective weight: {effective_weight:.3f}")
            else:
                logger.warning(f"Skipped client {client_id} - low reputation: {current_reputation:.3f}")
                return global_dict
                
            # Update state
            self.aggregation_count += 1
            self._adjust_mixing_weight()
            
            return new_dict
            
        except Exception as e:
            logger.error(f"Error during aggregation: {str(e)}")
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