"""
Random Agent

Simple baseline agent that selects actions randomly.
"""

from typing import Dict, Any
import numpy as np
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that selects random actions.
    
    Useful for baseline comparison and environment testing.
    
    Attributes:
        action_space: Action space to sample from
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize random agent.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config or {})
        self._rng = np.random.default_rng()
    
    def act(self, observation: np.ndarray, info: Dict[str, Any] = None) -> np.ndarray:
        """
        Select a random action.
        
        Args:
            observation: Current environment observation (unused)
            info: Additional information (unused)
            
        Returns:
            Random action
        """
        if self.action_space is None:
            # Fallback: return random integers
            return self._rng.integers(0, 10, size=(1,))
        
        return self.action_space.sample()
    
    def reset(self) -> None:
        """Reset agent state."""
        # No state to reset for random agent
        pass
    
    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        self._rng = np.random.default_rng(seed)
