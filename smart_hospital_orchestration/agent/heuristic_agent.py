"""
Heuristic Agent

Rule-based agent using domain knowledge heuristics.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    """
    Agent using hand-crafted heuristics for decision making.
    
    Implements priority-based and load-balancing heuristics
    commonly used in hospital resource management.
    
    Attributes:
        heuristic_type: Type of heuristic to use
        priority_weights: Weights for patient priority
    """
    
    HEURISTIC_TYPES = ["priority", "load_balance", "hybrid"]
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize heuristic agent.
        
        Args:
            config: Configuration with heuristic type and parameters
        """
        super().__init__(config)
        self.heuristic_type = config.get("heuristic_type", "hybrid")
        if self.heuristic_type not in self.HEURISTIC_TYPES:
            raise ValueError(f"Unknown heuristic type: {self.heuristic_type}")
        
        self.priority_weights = {
            "critical": 4.0,
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0
        }
    
    def act(self, observation: np.ndarray, info: Dict[str, Any] = None) -> np.ndarray:
        """
        Select action using heuristics.
        
        Args:
            observation: Current environment observation
            info: Additional information including state details
            
        Returns:
            Action based on heuristic
        """
        info = info or {}
        
        if self.heuristic_type == "priority":
            return self._priority_heuristic(info)
        elif self.heuristic_type == "load_balance":
            return self._load_balance_heuristic(info)
        else:  # hybrid
            return self._hybrid_heuristic(observation, info)
    
    def _priority_heuristic(self, info: Dict[str, Any]) -> np.ndarray:
        """
        Priority-based heuristic: admit highest priority patients first.
        
        Args:
            info: Environment information
            
        Returns:
            Action vector
        """
        # TODO: Implement priority-based action selection
        # - Identify highest priority waiting patients
        # - Assign available resources
        raise NotImplementedError("_priority_heuristic() not yet implemented")
    
    def _load_balance_heuristic(self, info: Dict[str, Any]) -> np.ndarray:
        """
        Load balancing heuristic: distribute patients evenly.
        
        Args:
            info: Environment information
            
        Returns:
            Action vector
        """
        # TODO: Implement load balancing action selection
        # - Calculate doctor workloads
        # - Assign patients to least loaded doctors
        raise NotImplementedError("_load_balance_heuristic() not yet implemented")
    
    def _hybrid_heuristic(
        self,
        observation: np.ndarray,
        info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Hybrid heuristic combining priority and load balancing.
        
        Args:
            observation: Environment observation
            info: Environment information
            
        Returns:
            Action vector
        """
        # TODO: Implement hybrid action selection
        # - Balance priority with workload
        # - Consider resource constraints
        raise NotImplementedError("_hybrid_heuristic() not yet implemented")
    
    def reset(self) -> None:
        """Reset agent state."""
        # No persistent state to reset
        pass
    
    def _find_best_bed_assignment(
        self,
        patient_priority: str,
        available_beds: List[str]
    ) -> Optional[str]:
        """
        Find best bed for a patient based on priority.
        
        Args:
            patient_priority: Patient priority level
            available_beds: List of available bed IDs
            
        Returns:
            Best bed ID or None
        """
        # TODO: Implement bed assignment logic
        # - Consider bed features
        # - Match with patient needs
        return available_beds[0] if available_beds else None
