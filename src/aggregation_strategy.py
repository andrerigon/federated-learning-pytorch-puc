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
from abc import ABC, abstractmethod
from collections import OrderedDict

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
    AsyncFedAvg: Asynchronous version of FedAvg that incorporates staleness.
    For example, scale the update by alpha/(1+staleness).
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

class SAFAStrategy(AggregationStrategy):
    """
    SAFA: Semi-Asynchronous Federated Averaging
    In SAFA, training runs on all learners, and a round ends when a preset fraction of participants return updates.
    Late updates are cached and applied in a later round.
    
    For demonstration, we do a simple average of available updates (like FedAvg).
    In a real scenario, you'd manage partial rounds, caching late updates, and apply them later.
    """
    def __init__(self, fraction: float = 0.8):
        self.fraction = fraction

    def aggregate(self, global_model: nn.Module, client_dict: list, alpha=0.1, extra_info=None) -> OrderedDict:
        # Similar to FedAvg for simplicity
        global_dict = global_model.state_dict()
        if len(client_dict) == 0:
            return global_dict

        keys = global_dict.keys()
        avg_dict = OrderedDict((k, torch.zeros_like(global_dict[k])) for k in keys)

        for cdict in client_dict:
            for k in keys:
                avg_dict[k] += cdict[k]

        for k in keys:
            avg_dict[k] = avg_dict[k] / len(client_dict)

        return avg_dict

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