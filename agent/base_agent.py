"""
Base Agent Interface

Abstract base class for all agent implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for hospital resource management agents.
    
    Defines the interface that all agents must implement,
    including policy-based agents, heuristic agents, and random agents.
    
    Attributes:
        config: Agent configuration dictionary
        action_space: Action space specification
        observation_space: Observation space specification
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the agent.
        
        Args:
            config: Configuration dictionary with agent parameters
        """
        self.config = config
        self.action_space = None
        self.observation_space = None
        self._initialized = False
    
    @abstractmethod
    def act(self, observation: np.ndarray, info: Dict[str, Any] = None) -> np.ndarray:
        """
        Select an action given an observation.
        
        Args:
            observation: Current environment observation
            info: Additional information (optional)
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for new episode."""
        pass
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action with additional information.
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, additional_info)
        """
        action = self.act(observation)
        info = {"deterministic": deterministic}
        return action, info
    
    def save(self, filepath: str) -> None:
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save agent state
        """
        # Default: no state to save
        pass
    
    def load(self, filepath: str) -> None:
        """
        Load agent state from file.
        
        Args:
            filepath: Path to load agent state
        """
        # Default: no state to load
        pass
    
    def set_action_space(self, action_space) -> None:
        """
        Set the action space for the agent.
        
        Args:
            action_space: Action space specification
        """
        self.action_space = action_space
    
    def set_observation_space(self, observation_space) -> None:
        """
        Set the observation space for the agent.
        
        Args:
            observation_space: Observation space specification
        """
        self.observation_space = observation_space
    
    @property
    def name(self) -> str:
        """Return agent name."""
        return self.__class__.__name__
