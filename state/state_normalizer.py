"""
State Normalizer Module

Handles normalization and standardization of state features.
"""

from typing import Dict, Any, Optional
import numpy as np


class StateNormalizer:
    """
    Normalizes state features for stable RL training.
    
    Supports running statistics for online normalization.
    
    Attributes:
        running_mean: Running mean of each feature
        running_var: Running variance of each feature
        count: Number of observations seen
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, state_dim: int, epsilon: float = 1e-8) -> None:
        """
        Initialize state normalizer.
        
        Args:
            state_dim: Dimension of state vector
            epsilon: Small constant to avoid division by zero
        """
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.running_mean = np.zeros(state_dim)
        self.running_var = np.ones(state_dim)
        self.count = 0
    
    def update(self, state: np.ndarray) -> None:
        """
        Update running statistics with a new state observation.
        
        Args:
            state: New state observation
        """
        self.count += 1
        delta = state - self.running_mean
        self.running_mean += delta / self.count
        delta2 = state - self.running_mean
        self.running_var += delta * delta2
    
    def normalize(self, state: np.ndarray, clip: Optional[float] = 10.0) -> np.ndarray:
        """
        Normalize a state using running statistics.
        
        Args:
            state: State to normalize
            clip: Optional clipping threshold (standard deviations)
            
        Returns:
            Normalized state
        """
        if self.count < 2:
            return state
        
        std = np.sqrt(self.running_var / self.count)
        normalized = (state - self.running_mean) / (std + self.epsilon)
        
        if clip is not None:
            normalized = np.clip(normalized, -clip, clip)
        
        return normalized
    
    def reset_stats(self) -> None:
        """Reset running statistics."""
        self.running_mean = np.zeros(self.state_dim)
        self.running_var = np.ones(self.state_dim)
        self.count = 0
    
    def get_stats(self) -> Dict[str, np.ndarray]:
        """
        Get current normalization statistics.
        
        Returns:
            Dictionary with mean and variance
        """
        return {
            "mean": self.running_mean.copy(),
            "var": self.running_var.copy() / max(self.count, 1),
            "count": self.count
        }
    
    def set_stats(self, stats: Dict[str, np.ndarray]) -> None:
        """
        Set normalization statistics.
        
        Args:
            stats: Dictionary with mean, var, and count
        """
        self.running_mean = stats["mean"].copy()
        self.running_var = stats["var"].copy() * stats["count"]
        self.count = stats["count"]
