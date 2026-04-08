"""
Reward Shaping Module

Implements reward shaping techniques for improved learning.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np


class RewardShaper:
    """
    Applies reward shaping techniques to improve RL training.
    
    Supports potential-based reward shaping and other techniques
    to guide the agent toward better policies.
    
    Attributes:
        potential_function: Function that computes potential of a state
        gamma: Discount factor for potential-based shaping
        use_shaping: Whether to apply reward shaping
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        potential_function: Optional[Callable[[Dict[str, Any]], float]] = None
    ) -> None:
        """
        Initialize reward shaper.
        
        Args:
            config: Configuration for shaping parameters
            potential_function: Custom potential function (optional)
        """
        self.config = config
        self.gamma = config.get("gamma", 0.99)
        self.use_shaping = config.get("use_reward_shaping", True)
        self.potential_function = potential_function or self._default_potential
        self.prev_potential = 0.0
    
    def _default_potential(self, state: Dict[str, Any]) -> float:
        """
        Default potential function based on state features.
        
        Args:
            state: Current environment state
            
        Returns:
            Potential value
        """
        # TODO: Implement default potential calculation
        # - Based on resource utilization
        # - Based on patient satisfaction
        # - Based on operational efficiency
        return 0.0
    
    def shape_reward(
        self,
        reward: float,
        state: Dict[str, Any],
        next_state: Dict[str, Any],
        done: bool
    ) -> float:
        """
        Apply reward shaping to a transition.
        
        Args:
            reward: Original reward
            state: Current state
            next_state: Next state
            done: Whether episode is complete
            
        Returns:
            Shaped reward
        """
        if not self.use_shaping:
            return reward
        
        # Potential-based reward shaping: F = gamma * Phi(s') - Phi(s)
        current_potential = self.potential_function(state)
        next_potential = self.potential_function(next_state) if not done else 0.0
        
        shaping = self.gamma * next_potential - current_potential
        
        return reward + shaping
    
    def reset(self) -> None:
        """Reset potential tracking for new episode."""
        self.prev_potential = 0.0
    
    def set_potential_function(
        self,
        potential_function: Callable[[Dict[str, Any]], float]
    ) -> None:
        """
        Set a custom potential function.
        
        Args:
            potential_function: New potential function
        """
        self.potential_function = potential_function
