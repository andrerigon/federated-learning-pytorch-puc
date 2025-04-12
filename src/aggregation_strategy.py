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
import torch
from torch.optim import Adam
from collections import OrderedDict
from typing import Dict, Any
import random 

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

class FedAdaptiveRL(AggregationStrategy):
    """
    FedAdaptive strategy that uses reinforcement learning to find the optimal K.
    Uses Q-learning to learn which K values perform best in different situations.
    """
    def __init__(self, initial_K: int = 1, min_K: int = 1, max_K: int = 5, 
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 exploration_rate: float = 0.2, success_window: int = 5):
        super().__init__()
        self.K = initial_K
        self.min_K = min_K
        self.max_K = max_K
        self.success_window = success_window
        
        # RL parameters
        self.learning_rate = learning_rate  # Alpha in Q-learning
        self.discount_factor = discount_factor  # Gamma in Q-learning
        self.exploration_rate = exploration_rate  # Epsilon for exploration
        self.exploration_decay = 0.995  # Decrease exploration over time
        self.min_exploration = 0.05  # Minimum exploration rate
        
        # Q-table: maps K values to expected rewards
        self.q_values = {k: 0.0 for k in range(min_K, max_K + 1)}
        
        # State tracking
        self.current_round_states = {}
        self.current_round_sizes = {}
        self.current_round_success = {}
        self.historical_success = []
        self.last_state_K = initial_K
        self.last_reward = 0.0
        self.round_count = 0
        
    def _get_best_K(self) -> int:
        """
        Select the best K value according to current Q-values,
        with exploration based on epsilon-greedy policy.
        """
        # Exploration: random K
        if random.random() < self.exploration_rate:
            return random.randint(self.min_K, self.max_K)
        
        # Exploitation: best known K
        return max(self.q_values, key=self.q_values.get)
    
    def _calculate_reward(self, avg_success: float) -> float:
        """
        Calculate the reward signal for reinforcement learning.
        Rewards higher success rates and penalizes large K values to
        encourage efficiency.
        """
        # Base reward from success rate
        success_reward = avg_success * 2 - 1  # Maps [0,1] to [-1,1]
        
        # Efficiency penalty (larger K values cost more)
        efficiency_factor = (self.max_K - self.K) / (self.max_K - self.min_K)
        efficiency_bonus = 0.2 * efficiency_factor  # Smaller bonus for smaller K
        
        # Combined reward
        reward = success_reward + efficiency_bonus
        
        return reward
    
    def _update_q_values(self, state_K: int, action_K: int, reward: float, next_state_K: int):
        """Update Q-values based on the observed reward and next state"""
        # Current Q-value
        current_q = self.q_values[action_K]
        
        # Best Q-value for next state
        next_max_q = max(self.q_values.values())
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        # Update Q-table
        self.q_values[action_K] = new_q
        
    def aggregate(self, global_model: nn.Module, client_state_dict: OrderedDict, 
              alpha=0.1, extra_info=None) -> OrderedDict:
        if extra_info is None or 'client_id' not in extra_info:
            return global_model.state_dict()
        
        client_id = extra_info['client_id']
        dataset_size = extra_info.get('client_samples', 1)
        success_rate = extra_info.get('success_rate', 0.0)
        
        # Store this client's update
        self.current_round_states[client_id] = client_state_dict
        self.current_round_sizes[client_id] = dataset_size
        self.current_round_success[client_id] = success_rate
        
        # Only aggregate when we have received K updates
        if len(self.current_round_states) < self.K:
            return global_model.state_dict()
        
        # Initialize aggregation dict - all as float tensors for aggregation process
        global_dict = global_model.state_dict()
        agg_dict = OrderedDict()
        
        # Track original dtypes to convert back later
        original_dtypes = {}
        original_devices = {}
        
        for k, v in global_dict.items():
            agg_dict[k] = torch.zeros_like(v, dtype=torch.float32)
            original_dtypes[k] = v.dtype
            original_devices[k] = v.device
        
        # Standard weighted aggregation
        total_samples = sum(self.current_round_sizes.values())
        for client_id, state_dict in self.current_round_states.items():
            weight = self.current_round_sizes[client_id] / total_samples
            for key in global_dict.keys():
                # Always convert to float for weighted calculations
                weighted_param = state_dict[key].to(dtype=torch.float32) * weight
                agg_dict[key] += weighted_param
        
        # Convert back to original dtypes
        for key, dtype in original_dtypes.items():
            if dtype in [torch.int64, torch.long, torch.int32, torch.int16, torch.int8]:
                # Round before converting to integer types
                agg_dict[key] = agg_dict[key].round().to(dtype=dtype, device=original_devices[key])
            else:
                # Convert float tensors back to their original precision
                agg_dict[key] = agg_dict[key].to(dtype=dtype, device=original_devices[key])
        
        # Calculate average success rate for this round
        avg_success = sum(self.current_round_success.values()) / len(self.current_round_success)
        self.historical_success.append(avg_success)
        
        # Calculate moving average
        recent_success = self.historical_success[-self.success_window:] if len(self.historical_success) >= self.success_window else self.historical_success
        avg_success = sum(recent_success) / len(recent_success)
        
        # Calculate reward from this round
        reward = self._calculate_reward(avg_success)
        
        # Update Q-values based on last action and current reward
        if self.round_count > 0:
            self._update_q_values(
                state_K=self.last_state_K,
                action_K=self.K,
                reward=reward,
                next_state_K=self.K
            )
        
        self.round_count += 1
        self.last_state_K = self.K
        self.last_reward = reward
        
        # Reduce exploration rate over time
        self.exploration_rate = max(
            self.min_exploration, 
            self.exploration_rate * self.exploration_decay
        )
        
        # Choose K for next round using RL
        old_K = self.K
        self.K = self._get_best_K()
        
        logger.info(f"Round {self.round_count}: Aggregated {len(self.current_round_states)} "
                f"client updates. Avg success: {avg_success:.2f}, Reward: {reward:.2f}, "
                f"Adjusting K from {old_K} to {self.K} (exploration rate: {self.exploration_rate:.2f}).")
        
        # Log Q-values occasionally
        if self.round_count % 10 == 0:
            top_k_values = sorted(self.q_values.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Top 5 K values: {top_k_values}")
        
        # Reset for next round
        self.current_round_states = {}
        self.current_round_sizes = {}
        self.current_round_success = {}
        
        return agg_dict
   
    
class FedAdaptiveClientSelection(AggregationStrategy):
    """
    FedAdaptive strategy with intelligent client selection based on historical performance.
    Prioritizes clients with higher historical success rates rather than selecting randomly.
    """
    def __init__(self, initial_K: int = 1, min_K: int = 3, max_K: int = None, 
                 target_success: float = 0.7, adjustment_rate: float = 0.1,
                 success_window: int = 5, history_length: int = 10):
        super().__init__()
        self.K = initial_K
        self.min_K = min_K
        self.max_K = max_K if max_K is not None else float('inf')
        self.target_success = target_success
        self.adjustment_rate = adjustment_rate
        self.success_window = success_window
        self.history_length = history_length
        
        # State tracking
        self.current_round_states = {}
        self.current_round_sizes = {}
        self.current_round_success = {}
        self.historical_success = []
        self.client_history = defaultdict(lambda: deque(maxlen=history_length))
        self.available_clients = set()  # Tracks clients that have reported in the current round
        self.selected_clients = set()   # Tracks clients selected for this round's aggregation
        self.round_count = 0
    
    def _adjust_K(self, avg_success: float):
        # Same as original FedAdaptive
        deviation = avg_success - self.target_success
        
        if abs(deviation) < 0.05:
            return self.K
            
        if deviation > 0:
            adjustment_factor = 1 + (self.adjustment_rate * min(deviation * 2, 1.0))
            new_K = int(self.K * adjustment_factor)
        else:
            adjustment_factor = 1 - (self.adjustment_rate * min(abs(deviation) * 2, 1.0))
            new_K = int(self.K * adjustment_factor)
        
        return min(max(new_K, self.min_K), self.max_K)
    
    def _get_client_score(self, client_id: str) -> float:
        """Calculate a score for a client based on historical success rates"""
        if not self.client_history[client_id]:
            return 0.0  # No history yet
        
        # Return average success rate over client history
        return sum(rate for rate in self.client_history[client_id]) / len(self.client_history[client_id])
    
    def _select_best_clients(self) -> List[str]:
        """Select the K clients with highest historical success rates"""
        if len(self.available_clients) <= self.K:
            return list(self.available_clients)
        
        # Calculate scores for all available clients
        client_scores = [(client_id, self._get_client_score(client_id)) 
                         for client_id in self.available_clients]
        
        # Sort by score (highest first) and select top K
        client_scores.sort(key=lambda x: x[1], reverse=True)
        return [client_id for client_id, _ in client_scores[:self.K]]
    
    def aggregate(self, global_model: nn.Module, client_state_dict: OrderedDict, 
                  alpha=0.1, extra_info=None) -> OrderedDict:
        if extra_info is None or 'client_id' not in extra_info:
            return global_model.state_dict()
        
        client_id = extra_info['client_id']
        dataset_size = extra_info.get('client_samples', 1)
        success_rate = extra_info.get('success_rate', 0.0)
        
        # Store this client's update and add to available clients
        self.current_round_states[client_id] = client_state_dict
        self.current_round_sizes[client_id] = dataset_size
        self.current_round_success[client_id] = success_rate
        self.available_clients.add(client_id)
        
        # Update client history
        self.client_history[client_id].append(success_rate)
        
        # Check if we have enough clients to perform selection
        if len(self.available_clients) < self.K:
            return global_model.state_dict()
        
        # If we haven't selected clients for this round yet, do so now
        if not self.selected_clients:
            self.selected_clients = set(self._select_best_clients())
            logger.info(f"Selected {len(self.selected_clients)} clients for aggregation")
        
        # Check if all selected clients have reported
        if not all(client in self.current_round_states for client in self.selected_clients):
            return global_model.state_dict()
        
        # Perform aggregation with only the selected clients
        global_dict = global_model.state_dict()
        agg_dict = OrderedDict((k, torch.zeros_like(v)) for k, v in global_dict.items())
        
        total_samples = sum(self.current_round_sizes[client_id] 
                           for client_id in self.selected_clients)
        
        # Weighted aggregation
        for client_id in self.selected_clients:
            state_dict = self.current_round_states[client_id]
            weight = self.current_round_sizes[client_id] / total_samples
            for key in global_dict.keys():
                weight_tensor = torch.tensor(weight, dtype=state_dict[key].dtype, 
                                           device=state_dict[key].device)
                agg_dict[key] += state_dict[key] * weight_tensor
        
        # Calculate average success rate for selected clients
        avg_success = sum(self.current_round_success[client_id] 
                         for client_id in self.selected_clients) / len(self.selected_clients)
        self.historical_success.append(avg_success)
        
        # Calculate moving average
        if len(self.historical_success) > self.success_window:
            recent_success = self.historical_success[-self.success_window:]
            avg_success = sum(recent_success) / len(recent_success)
        
        self.round_count += 1
        
        # Adjust K for the next round
        old_K = self.K
        self.K = self._adjust_K(avg_success)
        
        logger.info(f"Round {self.round_count}: Aggregated {len(self.selected_clients)} "
                   f"client updates from {len(self.available_clients)} available. "
                   f"Avg success: {avg_success:.2f}, Adjusting K from {old_K} to {self.K}.")
        
        # Reset for next round
        self.current_round_states = {}
        self.current_round_sizes = {}
        self.current_round_success = {}
        self.available_clients = set()
        self.selected_clients = set()
        
        return agg_dict
    
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
            try:

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
            except Exception as e:
                return global_model.state_dict()

    
class FedAdamStrategy(AggregationStrategy):
    """
    Implementation of FedProx aggregation strategy from:
    'Federated Optimization in Heterogeneous Networks'
    Li et al., MLSys 2020 (DOI: 10.48550/arXiv.1812.06127)
    """
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """
        Initialize the FedAdam strategy.

        Args:
            lr (float): Learning rate for Adam optimizer.
            beta1 (float): Beta1 hyperparameter for Adam.
            beta2 (float): Beta2 hyperparameter for Adam.
            eps (float): Term added for numerical stability in Adam.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.optimizer = None

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict, alpha=0.1, extra_info=None) -> OrderedDict:
        """
        Aggregate the client model parameters into the global model parameters using Adam optimizer.
        
        Args:
            global_model (nn.Module): The global model before aggregation.
            client_dict (OrderedDict): State dict from the client model.
            alpha (float): Blending factor (not used in FedAdam).
            extra_info (dict): Additional info like staleness if needed.
        
        Returns:
            OrderedDict: The updated global state dictionary after aggregation.
        """
        from torch.optim import Adam
        
        # Initialize Adam optimizer for the global model
        self.optimizer = Adam(global_model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=self.eps)
        
        # Get the global model's state dictionary
        global_dict = global_model.state_dict()
        
        # Calculate the "gradient" as the difference between client and global model
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                # Find the corresponding key in state_dict
                for k in global_dict.keys():
                    if k.endswith(name.split('.')[-1]) and name in k:
                        # Set negative gradient (client - global)
                        update = client_dict[k] - global_dict[k]
                        param.grad = -update
                        break
        
        # Update global model parameters using Adam optimizer
        self.optimizer.step()
        
        return global_model.state_dict()

class QFedAvgStrategy(AggregationStrategy):
    """
    Implementation of q-FedAvg from:
    'Fair Resource Allocation in Federated Learning'
    Li et al., ICLR 2020 (DOI: 10.48550/arXiv.1905.10497)
    """
    def __init__(self, q=0.5, learning_rate=0.1, sampling_rate=0.1, total_clients=None):
        self.q = q
        self.lr = learning_rate
        self.client_losses = {}
        
        # Determine min_clients based on sampling rate
        if total_clients is not None:
            self.min_clients = max(1, int(total_clients * sampling_rate))
        else:
            self.min_clients = 1

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict,
                 alpha=0.1, extra_info=None) -> OrderedDict:
        """
        Aggregate client updates using q-FedAvg strategy.
        """
        if extra_info is None or 'client_id' not in extra_info or 'client_loss' not in extra_info:
            return global_model.state_dict()
            
        client_id = extra_info['client_id']
        loss = float(extra_info['client_loss'])
        
        # Store client update
        self.client_losses[client_id] = {
            'params': client_dict,
            'loss': max(loss, 1e-5)  # Ensure positive loss value
        }
        
        # Check if we have enough clients to perform aggregation
        if len(self.client_losses) < self.min_clients:
            return global_model.state_dict()
        
        # Compute client weights based on q-fairness
        total_q_loss = sum(v['loss']**self.q for v in self.client_losses.values())
        weights = {cid: (v['loss']**self.q / total_q_loss) for cid, v in self.client_losses.items()}
        
        # Get the global model state dict
        global_dict = global_model.state_dict()
        
        # Create a new dictionary for the aggregated model
        aggregated_dict = OrderedDict()
        
        # Process each key in the state dict
        for key in global_dict.keys():
            # Create a temporary tensor for each parameter's weighted sum
            if global_dict[key].dtype == torch.long or global_dict[key].dtype == torch.int64:
                # For integer tensors, first convert to float, then back to int at the end
                temp_tensor = torch.zeros_like(global_dict[key], dtype=torch.float)
                
                # Add each client's contribution
                for cid, info in self.client_losses.items():
                    # Skip if parameter missing in client model
                    if key not in info['params']:
                        continue
                    
                    # Get weight and parameter, convert to float
                    weight = weights[cid]
                    param = info['params'][key].float()
                    
                    # Add weighted parameter (avoiding in-place operation that caused the error)
                    temp_tensor = temp_tensor + (param * weight)
                
                # Apply server learning rate and update
                delta = (temp_tensor - global_dict[key].float()) * self.lr
                updated = global_dict[key].float() + delta
                
                # Convert back to original type with rounding
                aggregated_dict[key] = torch.round(updated).to(dtype=global_dict[key].dtype)
                
            else:
                # For float tensors, standard approach
                temp_tensor = torch.zeros_like(global_dict[key])
                
                # Add each client's contribution
                for cid, info in self.client_losses.items():
                    # Skip if parameter missing in client model
                    if key not in info['params']:
                        continue
                    
                    # Get weight and parameter
                    weight = weights[cid]
                    param = info['params'][key]
                    
                    # Add weighted parameter (avoid in-place addition)
                    temp_tensor = temp_tensor + (param * weight)
                
                # Apply server learning rate
                aggregated_dict[key] = global_dict[key] + self.lr * (temp_tensor - global_dict[key])
        
        # Reset client losses after aggregation
        self.client_losses = {}
        
        return aggregated_dict
    """
    Implementation of q-FedAvg from:
    'Fair Resource Allocation in Federated Learning'
    Li et al., ICLR 2020 (DOI: 10.48550/arXiv.1905.10497)
    """
    def __init__(self, q=0.5, learning_rate=0.1, sampling_rate=0.1, total_clients=None):
        """
        Initialize the q-FedAvg strategy.
        """
        self.q = q
        self.lr = learning_rate
        self.client_losses = {}
        
        # Determine min_clients based on sampling rate
        if total_clients is not None:
            self.min_clients = max(1, int(total_clients * sampling_rate))
        else:
            self.min_clients = 1

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict,
                 alpha=0.1, extra_info=None) -> OrderedDict:
        """
        Aggregate client updates using q-FedAvg strategy.
        """
        if extra_info is None or 'client_id' not in extra_info or 'client_loss' not in extra_info:
            return global_model.state_dict()
            
        client_id = extra_info['client_id']
        loss = float(extra_info['client_loss'])
        
        # Store client update
        self.client_losses[client_id] = {
            'params': client_dict,
            'loss': max(loss, 1e-5)  # Ensure positive loss value
        }
        
        # Check if we have enough clients to perform aggregation
        if len(self.client_losses) < self.min_clients:
            return global_model.state_dict()
        
        # Compute client weights based on q-fairness
        total_q_loss = sum(v['loss']**self.q for v in self.client_losses.values())
        weights = {cid: (v['loss']**self.q / total_q_loss) for cid, v in self.client_losses.items()}
        
        # Aggregate parameters
        global_dict = global_model.state_dict()
        aggregated_dict = OrderedDict()
        
        # For each parameter in the model
        for key in global_dict.keys():
            # Get parameter type
            param_type = global_dict[key].dtype
            
            # Initialize aggregate tensor as float
            agg_tensor = torch.zeros_like(global_dict[key], dtype=torch.float32)
            
            # Sum weighted client updates
            for cid, data in self.client_losses.items():
                if key in data['params']:
                    # Convert to float for computation
                    client_param = data['params'][key].to(dtype=torch.float32)
                    weight = float(weights[cid])
                    agg_tensor += client_param * weight
            
            # Apply learning rate to get delta
            global_param = global_dict[key].to(dtype=torch.float32)
            delta = (agg_tensor - global_param) * self.lr
            
            # Add delta to original parameter, converting back to original type
            updated_param = (global_param + delta).to(dtype=param_type)
            aggregated_dict[key] = updated_param
        
        # Reset client losses after aggregation
        self.client_losses = {}
        
        return aggregated_dict
    """
    Implementation of q-FedAvg from:
    'Fair Resource Allocation in Federated Learning'
    Li et al., ICLR 2020 (DOI: 10.48550/arXiv.1905.10497)
    """
    def __init__(self, q=0.5, learning_rate=0.1, sampling_rate=0.1, total_clients=None):
        """
        Initialize the q-FedAvg strategy.
        
        Args:
            q (float): Fairness parameter (higher q emphasizes clients with higher loss).
            learning_rate (float): Server learning rate.
            sampling_rate (float): Fraction of clients to wait for before aggregation.
            total_clients (int): Total number of clients in the simulation.
        """
        self.q = q
        self.lr = learning_rate
        self.client_losses = {}
        
        # Determine min_clients based on sampling rate
        if total_clients is not None:
            self.min_clients = max(1, int(total_clients * sampling_rate))
        else:
            # Default to at least 1 client if total_clients isn't provided
            self.min_clients = 1

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict,
                 alpha=0.1, extra_info=None) -> OrderedDict:
        """
        Aggregate client updates using q-FedAvg strategy.
        
        Args:
            global_model (nn.Module): The global model before aggregation.
            client_dict (OrderedDict): State dict from a client.
            alpha (float): Not used in q-FedAvg.
            extra_info (dict): Contains client_id, client_loss, and client_samples.
        
        Returns:
            OrderedDict: The updated global state dictionary after aggregation.
        """
        if extra_info is None or 'client_id' not in extra_info or 'client_loss' not in extra_info:
            return global_model.state_dict()
            
        client_id = extra_info['client_id']
        loss = float(extra_info['client_loss'])
        
        # Store client update
        self.client_losses[client_id] = {
            'params': client_dict,
            'loss': max(loss, 1e-5)  # Ensure positive loss value
        }
        
        # Check if we have enough clients to perform aggregation
        if len(self.client_losses) < self.min_clients:
            return global_model.state_dict()
        
        # Compute client weights based on q-fairness
        total_q_loss = sum(v['loss']**self.q for v in self.client_losses.values())
        weights = {cid: (v['loss']**self.q / total_q_loss) for cid, v in self.client_losses.items()}
        
        # Aggregate parameters with weighted average
        global_dict = global_model.state_dict()
        aggregated_dict = OrderedDict()
        
        for key in global_dict.keys():
            # Create a clone of the global parameter to modify
            aggregated_dict[key] = global_dict[key].clone()
            
            # Calculate weighted direction
            direction = torch.zeros_like(global_dict[key])
            
            for cid, data in self.client_losses.items():
                if key in data['params']:
                    # Use clone to avoid modifying the original tensors
                    client_param = data['params'][key].clone()
                    # Calculate direction from client to server for this parameter
                    direction += (client_param - global_dict[key]) * weights[cid]
            
            # Update parameter with server learning rate
            aggregated_dict[key] += direction * self.lr
        
        # Reset client losses after aggregation
        self.client_losses = {}
        
        return aggregated_dict
    """
    Implementation of q-FedAvg from:
    'Fair Resource Allocation in Federated Learning'
    Li et al., ICLR 2020 (DOI: 10.48550/arXiv.1905.10497)
    """
    def __init__(self, q=0.5, learning_rate=0.1, min_clients=1):
        """
        Initialize the q-FedAvg strategy.
        
        Args:
            q (float): Fairness parameter (higher q emphasizes clients with higher loss).
            learning_rate (float): Server learning rate.
            min_clients (int): Minimum number of clients required before performing aggregation.
        """
        self.q = q
        self.lr = learning_rate
        self.client_losses = {}
        self.min_clients = min_clients  # Add this parameter to control when aggregation happens

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict,
                 alpha=0.1, extra_info=None) -> OrderedDict:
        """
        Aggregate client updates using q-FedAvg strategy.
        
        Args:
            global_model (nn.Module): The global model before aggregation.
            client_dict (OrderedDict): State dict from a client.
            alpha (float): Not used in q-FedAvg.
            extra_info (dict): Contains client_id, client_loss, and client_samples.
        
        Returns:
            OrderedDict: The updated global state dictionary after aggregation.
        """
        if extra_info is None or 'client_id' not in extra_info or 'client_loss' not in extra_info:
            print("Warning: Required client information missing in extra_info")
            return global_model.state_dict()
            
        client_id = extra_info['client_id']
        loss = extra_info['client_loss']
        samples = extra_info.get('client_samples', 1)  # Default to 1 if not provided
        
        # Store client update
        self.client_losses[client_id] = {
            'params': client_dict,
            'loss': loss,
            'samples': samples
        }
        
        # Check if we have enough clients to perform aggregation
        if len(self.client_losses) < self.min_clients:
            print(f"Waiting for more clients ({len(self.client_losses)}/{self.min_clients})")
            return global_model.state_dict()
        
        print(f"Aggregating updates from {len(self.client_losses)} clients with q={self.q}")
        
        # Ensure we have valid loss values
        if any(v['loss'] <= 0 for v in self.client_losses.values()):
            print("Warning: Some clients have non-positive loss values")
            # Replace non-positive losses with a small positive value
            for cid, data in self.client_losses.items():
                if data['loss'] <= 0:
                    data['loss'] = 1e-5
        
        # Compute client weights based on q-fairness
        total_q_loss = sum(v['loss']**self.q for v in self.client_losses.values())
        if total_q_loss == 0:
            print("Warning: Total q-loss is zero, using uniform weights")
            weights = {cid: 1.0/len(self.client_losses) for cid in self.client_losses.keys()}
        else:
            weights = {cid: (v['loss']**self.q / total_q_loss) for cid, v in self.client_losses.items()}
        
        # Print weights for debugging
        print(f"Client weights: {weights}")
        
        # Aggregate parameters with weighted average
        global_dict = global_model.state_dict()
        agg_dict = OrderedDict()
        
        for key in global_dict.keys():
            # Initialize with zeros of the same shape - ensure correct dtype
            agg_tensor = torch.zeros_like(global_dict[key], dtype=torch.float32)
            
            # Weighted sum of client parameters
            for cid, data in self.client_losses.items():
                # Check if the key exists in the client parameters
                if key not in data['params']:
                    print(f"Warning: Key {key} missing in client {cid}")
                    continue
                    
                # Convert weights to the appropriate dtype and add weighted client parameters
                weight = float(weights[cid])  # Convert to Python float
                param_tensor = data['params'][key].float()  # Convert parameter tensor to float
                agg_tensor += param_tensor * weight
            
            # Update global parameters using server learning rate
            # Convert all tensors to the same dtype as the global model parameter
            target_dtype = global_dict[key].dtype
            delta = (agg_tensor - global_dict[key].float()) * self.lr
            agg_dict[key] = global_dict[key] + delta.to(dtype=target_dtype)
        
        # Reset client losses after aggregation
        self.client_losses = {}
        
        return agg_dict
    """
    Implementation of q-FedAvg from:
    'Fair Resource Allocation in Federated Learning'
    Li et al., ICLR 2020 (DOI: 10.48550/arXiv.1905.10497)
    """
    def __init__(self, q=0.5, learning_rate=0.1, min_clients=1):
        """
        Initialize the q-FedAvg strategy.
        
        Args:
            q (float): Fairness parameter (higher q emphasizes clients with higher loss).
            learning_rate (float): Server learning rate.
            min_clients (int): Minimum number of clients required before performing aggregation.
        """
        self.q = q
        self.lr = learning_rate
        self.client_losses = {}
        self.min_clients = min_clients  # Add this parameter to control when aggregation happens

    def aggregate(self, global_model: nn.Module, client_dict: OrderedDict,
                 alpha=0.1, extra_info=None) -> OrderedDict:
        """
        Aggregate client updates using q-FedAvg strategy.
        
        Args:
            global_model (nn.Module): The global model before aggregation.
            client_dict (OrderedDict): State dict from a client.
            alpha (float): Not used in q-FedAvg.
            extra_info (dict): Contains client_id, client_loss, and client_samples.
        
        Returns:
            OrderedDict: The updated global state dictionary after aggregation.
        """
        if extra_info is None or 'client_id' not in extra_info or 'client_loss' not in extra_info:
            print("Warning: Required client information missing in extra_info")
            return global_model.state_dict()
            
        client_id = extra_info['client_id']
        loss = extra_info['client_loss']
        samples = extra_info.get('client_samples', 1)  # Default to 1 if not provided
        
        # Store client update
        self.client_losses[client_id] = {
            'params': client_dict,
            'loss': loss,
            'samples': samples
        }
        
        # Check if we have enough clients to perform aggregation
        if len(self.client_losses) < self.min_clients:
            print(f"Waiting for more clients ({len(self.client_losses)}/{self.min_clients})")
            return global_model.state_dict()
        
        print(f"Aggregating updates from {len(self.client_losses)} clients with q={self.q}")
        
        # Ensure we have valid loss values
        if any(v['loss'] <= 0 for v in self.client_losses.values()):
            print("Warning: Some clients have non-positive loss values")
            # Replace non-positive losses with a small positive value
            for cid, data in self.client_losses.items():
                if data['loss'] <= 0:
                    data['loss'] = 1e-5
        
        # Compute client weights based on q-fairness
        total_q_loss = sum(v['loss']**self.q for v in self.client_losses.values())
        if total_q_loss == 0:
            print("Warning: Total q-loss is zero, using uniform weights")
            weights = {cid: 1.0/len(self.client_losses) for cid in self.client_losses.keys()}
        else:
            weights = {cid: (v['loss']**self.q / total_q_loss) for cid, v in self.client_losses.items()}
        
        # Print weights for debugging
        print(f"Client weights: {weights}")
        
        # Aggregate parameters with weighted average
        global_dict = global_model.state_dict()
        agg_dict = OrderedDict()
        
        for key in global_dict.keys():
            # Initialize with zeros of the same shape
            agg_tensor = torch.zeros_like(global_dict[key])
            
            # Weighted sum of client parameters
            for cid, data in self.client_losses.items():
                # Check if the key exists in the client parameters
                if key not in data['params']:
                    print(f"Warning: Key {key} missing in client {cid}")
                    continue
                    
                # Add weighted client parameters
                agg_tensor += data['params'][key] * weights[cid]
            
            # Update global parameters using server learning rate
            agg_dict[key] = global_dict[key] + self.lr * (agg_tensor - global_dict[key])
        
        # Reset client losses after aggregation
        self.client_losses = {}
        
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