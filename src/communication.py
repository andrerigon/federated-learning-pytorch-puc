"""
communication_mediator.py
A generic communication mediator that simulates realistic network conditions and tracks message delivery statistics.

The mediator implements a realistic network simulation with:
1. Success rates that start randomly between 0.25 and 1.0
2. Slow, gradual changes in network quality over time
3. Client correlation (nearby clients share similar network conditions)

This provides a more realistic simulation of real-world network behavior where:
- Network conditions start from a random baseline
- Changes in network quality happen very gradually
- Nearby clients experience similar network conditions (area-wide issues)

Generic Type Parameters:
    T: The type of command/message being mediated. This allows the mediator to work with
       different message types while maintaining type safety.
"""

import random
import math
import time
from typing import Generic, TypeVar, Any, Dict, Tuple, List
from collections import defaultdict
from loguru import logger

# Define the generic type for the command/message
T = TypeVar('T')

class CommunicationMediator(Generic[T]):
    """
    A generic mediator that handles message delivery with realistic network simulation.
    
    This class simulates real-world network conditions by:
    1. Starting with random success rates between 0.25 and 1.0
    2. Implementing very gradual changes in network quality
    3. Correlating network conditions between nearby clients
    
    Attributes:
        total_attempts (int): Count of all message delivery attempts
        successful_attempts (int): Count of successful message deliveries
        sticky_duration (float): How long to keep success rates (in seconds)
        gradual_factor (float): How quickly to blend new success rates (0-1)
        correlation_radius (float): How far to look for correlated clients
        max_change_per_second (float): Maximum allowed change in success rate per second
    """ 
    
    def __init__(self, origin: str, 
                 sticky_duration: float = 9000.0,  # 30 minutes in seconds
                 gradual_factor: float = 0.5,  # Reduced for slower changes
                 correlation_radius: float = 0.2,
                 success_rate: float = None) -> None:
        """
        Initialize the communication mediator with realistic network simulation.
        
        Args:
            origin: the origin for debug purposes
            sticky_duration: How long to keep success rates (in seconds)
            gradual_factor: How quickly to blend new success rates (0-1)
            correlation_radius: How far to look for correlated clients
            success_rate: Ignored, kept for backward compatibility
        """
        self.total_attempts = 0
        self.successful_attempts = 0
        self.origin = origin
        
        # Network simulation parameters
        self.sticky_duration = sticky_duration
        self.gradual_factor = gradual_factor
        self.correlation_radius = correlation_radius
        self.max_change_per_second = 0.01  # Maximum 1% change per second
        
        # Track client states
        self.client_success_rates = {}  # client_id -> (rate, expiry_time, last_update_time)
        self.client_positions = {}  # client_id -> (x, y) position
        self.client_attempts = defaultdict(int)  # client_id -> attempt count

    def _get_client_position(self, client_id: str) -> Tuple[float, float]:
        """Get or generate client position for correlation calculations."""
        if client_id in self.client_positions:
            return self.client_positions[client_id]
        
        # Generate a random position if not exists
        pos = (random.random(), random.random())
        self.client_positions[client_id] = pos
        return pos

    def _get_correlated_success_rate(self, client_id: str, new_rate: float) -> float:
        """
        Get success rate influenced by nearby clients.
        Uses spatial correlation to simulate area-wide network conditions.
        """
        if not self.client_success_rates:
            return new_rate
        
        client_pos = self._get_client_position(client_id)
        correlated_rates = []
        
        # Find rates from nearby clients
        for other_id, (rate, _, _) in self.client_success_rates.items():
            if other_id == client_id:
                continue
                
            other_pos = self._get_client_position(other_id)
            distance = math.sqrt((client_pos[0] - other_pos[0])**2 + 
                               (client_pos[1] - other_pos[1])**2)
            
            if distance <= self.correlation_radius:
                correlated_rates.append(rate)
        
        if not correlated_rates:
            return new_rate
        
        # Blend new rate with correlated rates
        avg_correlated = sum(correlated_rates) / len(correlated_rates)
        blended_rate = (new_rate * (1 - self.gradual_factor) + 
                       avg_correlated * self.gradual_factor)
        
        return blended_rate

    def _update_sticky_success_rate(self, client_id: str, new_rate: float) -> float:
        """
        Update the sticky success rate for a client.
        Implements gradual changes and correlation with nearby clients.
        Uses time-based sticky periods and enforces maximum change rate.
        """
        current_time = time.time()
        
        # Get correlated rate
        correlated_rate = self._get_correlated_success_rate(client_id, new_rate)
        
        if client_id in self.client_success_rates:
            current_rate, expiry_time, last_update = self.client_success_rates[client_id]
            
            # Calculate time since last update
            time_diff = current_time - last_update
            
            # Calculate maximum allowed change
            max_change = self.max_change_per_second * time_diff
            
            # Calculate desired change
            desired_change = correlated_rate - current_rate
            
            # Limit the change to maximum allowed
            actual_change = max(min(desired_change, max_change), -max_change)
            
            # Apply the limited change
            new_rate = current_rate + actual_change
            
            if current_time < expiry_time:
                # Still using sticky rate, but blend with new rate
                self.client_success_rates[client_id] = (new_rate, expiry_time, current_time)
                return new_rate
            else:
                # Sticky period ended, update with new rate
                new_expiry = current_time + self.sticky_duration
                self.client_success_rates[client_id] = (new_rate, new_expiry, current_time)
                return new_rate
        else:
            # First time seeing this client - start with random rate
            initial_rate = random.uniform(0.25, 1.0)
            new_expiry = current_time + self.sticky_duration
            self.client_success_rates[client_id] = (initial_rate, new_expiry, current_time)
            return initial_rate

    def send_message(self, command: T, provider: Any, client_id: str = None) -> bool:
        """
        Attempt to send a message through the provider with realistic network simulation.
        
        Args:
            command (T): The message/command to be sent
            provider: The communication provider that will handle actual message delivery
            client_id: Optional client identifier for tracking network conditions
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        
        Note:
            The provider must implement send_communication_command method
        """
        self.total_attempts += 1
        if client_id:
            self.client_attempts[client_id] += 1
        
        # Generate base success rate
        base_success_rate = random.uniform(0.25, 1.0)
        
        # Apply realistic network simulation if client_id provided
        if client_id:
            current_success_rate = self._update_sticky_success_rate(client_id, base_success_rate)
        else:
            current_success_rate = base_success_rate
        
        # Simulate network conditions based on current success rate
        value = random.random()
        if value < current_success_rate:
            self.successful_attempts += 1
            provider.send_communication_command(command)
            logger.trace("[{}] Message succeeded for client {}. Rate: {:.2f}, Value: {:.2f}", 
                        self.origin, client_id, current_success_rate, value)            
            return True
        else:
            logger.trace("[{}] Message failed for client {}. Rate: {:.2f}, Value: {:.2f}", 
                        self.origin, client_id, current_success_rate, value)            
            return False

    def log_metrics(self) -> dict:
        """
        Calculate and log communication success metrics.
        
        Returns:
            dict: Dictionary containing metrics:
                - success_rate: Percentage of successful deliveries
                - total_attempts: Total number of delivery attempts
                - successful_attempts: Number of successful deliveries
                - client_metrics: Per-client success rates and attempts
        """
        success_rate = (
            self.successful_attempts / self.total_attempts 
            if self.total_attempts > 0 else 0
        )
        
        # Calculate per-client metrics
        current_time = time.time()
        client_metrics = {}
        for client_id, attempts in self.client_attempts.items():
            if client_id in self.client_success_rates:
                rate, expiry_time, _ = self.client_success_rates[client_id]
                remaining_time = max(0, expiry_time - current_time)
                client_metrics[client_id] = {
                    'success_rate': rate,
                    'remaining_time': remaining_time,
                    'attempts': attempts
                }
        
        metrics = {
            'success_rate': success_rate,
            'total_attempts': self.total_attempts,
            'successful_attempts': self.successful_attempts,
            'client_metrics': client_metrics
        }
        
        # Log the metrics
        logger.info(f"Overall success rate: {success_rate:.2%}")
        logger.info(f"Total attempts: {self.total_attempts}, "
                   f"Successful attempts: {self.successful_attempts}")
        
        return metrics

    def get_current_success_rate(self, client_id: str = None) -> float:
        """
        Get the current success rate for a client, considering sticky rates and correlation.
        
        Args:
            client_id: Optional client identifier
            
        Returns:
            float: The current success rate for the client
        """
        if client_id and client_id in self.client_success_rates:
            rate, expiry_time, _ = self.client_success_rates[client_id]
            if time.time() < expiry_time:
                return rate
        return random.uniform(0.25, 1.0)

    def reset_metrics(self) -> None:
        """
        Reset all tracking metrics to initial state.
        Useful for testing or starting a new monitoring period.
        """
        self.total_attempts = 0
        self.successful_attempts = 0
        self.client_attempts.clear()
        # Don't reset client_success_rates to maintain network simulation continuity