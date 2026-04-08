"""
Reward Function Core

Main reward computation interface for the hospital environment.
"""

from typing import Dict, Any
import numpy as np
from .reward_components import RewardComponents


class RewardFunction:
    """
    Computes reward signals for the RL agent.
    
    Aggregates multiple reward components to form a comprehensive
    reward signal that captures hospital efficiency metrics.
    
    Attributes:
        weights: Dictionary of component weights
        components: Reward component calculators
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize reward function.
        
        Args:
            config: Configuration with reward weights and parameters
        """
        self.config = config
        self.weights = {
            "patient_outcome": 50.0,
            "resource_utilization": 10.0,
            "waiting_time": -0.01,
            "workload_balance": 0.0,
            "operational_cost": 0.0,
        }
        self.components = RewardComponents(config)
    
    def compute_reward(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        next_state: Dict[str, Any],
        info: Dict[str, Any]
    ) -> float:
        """
        Compute total reward for a transition.
        
        Args:
            state: Previous state
            action: Action taken
            next_state: New state after action
            info: Additional information
            
        Returns:
            Total reward value
        """
        reward = 0.0
        
        # Compute individual components
        patient_outcome = self.components.compute_patient_outcome_reward(next_state)
        resource_util = self.components.compute_resource_utilization_reward(next_state)
        waiting_penalty = self.components.compute_waiting_time_penalty(next_state)
        workload_reward = self.components.compute_workload_balance_reward(next_state)
        cost_penalty = self.components.compute_operational_cost_penalty(state, action)
        emergency_reward = self.components.compute_emergency_handling_reward(next_state)
        
        # Weighted sum
        reward += self.weights.get("patient_outcome", 50.0) * patient_outcome
        reward += self.weights.get("resource_utilization", 10.0) * resource_util
        reward += self.weights.get("waiting_time", -0.01) * waiting_penalty
        reward += self.weights.get("workload_balance", 0.0) * workload_reward
        reward += self.weights.get("operational_cost", 0.0 ) * cost_penalty
        reward += self.weights.get("emergency_handling", 0.6) * emergency_reward
        
        return reward
    
    def compute_step_reward(self, state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """
        Compute reward for a single step (simplified interface).
        
        Args:
            state: Current state
            info: Additional information
            
        Returns:
            Reward value
        """
        action = np.array([info.get("action_id", 0)], dtype=np.int32)
        # Approximate with same-state transition when previous state is unavailable.
        return float(self.compute_reward(state, action, state, info))
    
    def get_reward_components(
        self,
        state: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get individual reward component values.
        
        Args:
            state: Previous state
            next_state: New state
            
        Returns:
            Dictionary of component names to values
        """
        return {
            "patient_outcome": self.components.compute_patient_outcome_reward(next_state),
            "resource_utilization": self.components.compute_resource_utilization_reward(next_state),
            "waiting_time": self.components.compute_waiting_time_penalty(next_state),
            "workload_balance": self.components.compute_workload_balance_reward(next_state),
            "emergency_handling": self.components.compute_emergency_handling_reward(next_state),
        }
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update reward component weights.
        
        Args:
            new_weights: Dictionary of new weights
        """
        self.weights.update(new_weights)
