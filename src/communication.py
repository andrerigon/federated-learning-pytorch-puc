"""
communication_mediator.py
A generic communication mediator that simulates network conditions and tracks message delivery statistics.

The mediator implements a probabilistic message delivery system where messages may fail
to be delivered based on a configured success rate. This is useful for simulating real-world
network conditions in testing and simulation environments.

Generic Type Parameters:
    T: The type of command/message being mediated. This allows the mediator to work with
       different message types while maintaining type safety.
"""

import random
from typing import Generic, TypeVar, Any
from loguru import logger

# Define the generic type for the command/message
T = TypeVar('T')

class CommunicationMediator(Generic[T]):
    """
    A generic mediator that handles message delivery with configurable reliability.
    
    This class simulates network conditions by introducing a probability of message
    failure, tracks delivery statistics, and provides reporting capabilities.
    
    Attributes:
        success_rate (float): Probability (0.0 to 1.0) that a message will be delivered successfully
        total_attempts (int): Count of all message delivery attempts
        successful_attempts (int): Count of successful message deliveries
    
    Example:
        >>> mediator = CommunicationMediator[str](success_rate=0.8)
        >>> mediator.send_message("test_message", provider)
    """ 
    
    def __init__(self, origin: str, success_rate: float) -> None:
        """
        Initialize the communication mediator.
        
        Args:
            origin: the origin for debug purporses
            success_rate (float): Probability of successful message delivery (0.0 to 1.0)
            
        Raises:
            ValueError: If success_rate is not between 0 and 1
        """
        if not 0 <= success_rate <= 1:
            raise ValueError("Success rate must be between 0 and 1")
            
        self.success_rate = success_rate
        self.total_attempts = 0
        self.successful_attempts = 0
        self.origin = origin

    def send_message(self, command: T, provider: Any) -> bool:
        """
        Attempt to send a message through the provider with probabilistic success.
        
        Args:
            command (T): The message/command to be sent
            provider: The communication provider that will handle actual message delivery
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        
        Note:
            The provider must implement send_communication_command method
        """
        self.total_attempts += 1
        
        # Simulate network conditions based on success rate
        value = random.random()
        if value < self.success_rate:
            self.successful_attempts += 1
            provider.send_communication_command(command)
            logger.trace(f"[{self.origin}] Message succeded. Result: {value}, success_rate: {self.success_rate}")
            return True
        else:
            logger.trace(f"[{self.origin}] Message failed to send due to simulated communication error.  Result: {value}, success_rate: {self.success_rate}")
            return False

    def log_metrics(self) -> dict:
        """
        Calculate and log communication success metrics.
        
        Returns:
            dict: Dictionary containing metrics:
                - success_rate: Percentage of successful deliveries
                - total_attempts: Total number of delivery attempts
                - successful_attempts: Number of successful deliveries
        """
        success_rate = (
            self.successful_attempts / self.total_attempts 
            if self.total_attempts > 0 else 0
        )
        
        metrics = {
            'success_rate': success_rate,
            'total_attempts': self.total_attempts,
            'successful_attempts': self.successful_attempts
        }
        
        # Log the metrics
        print(f"Message success rate: {success_rate:.2%}")
        print(f"Total attempts: {self.total_attempts}, "
              f"Successful attempts: {self.successful_attempts}")
        
        return metrics

    def reset_metrics(self) -> None:
        """
        Reset all tracking metrics to initial state.
        Useful for testing or starting a new monitoring period.
        """
        self.total_attempts = 0
        self.successful_attempts = 0