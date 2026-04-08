"""
Base Task Configuration

Abstract base class for task configurations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import yaml
from pathlib import Path


class BaseTaskConfig(ABC):
    """
    Abstract base class for hospital task configurations.
    
    Defines the interface and common functionality for all
    difficulty levels of hospital resource management tasks.
    
    Attributes:
        difficulty: Task difficulty level
        config_dict: Complete configuration dictionary
    """
    
    def __init__(self, difficulty: str = "base") -> None:
        """
        Initialize base task configuration.
        
        Args:
            difficulty: Task difficulty identifier
        """
        self.difficulty = difficulty
        self.config_dict: Dict[str, Any] = {}
        self._build_config()
    
    @abstractmethod
    def _build_config(self) -> None:
        """Build the configuration dictionary. Must be implemented by subclasses."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config_dict.copy()
    
    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get environment-specific configuration.
        
        Returns:
            Environment configuration subset
        """
        return self.config_dict.get("environment", {})
    
    def get_reward_config(self) -> Dict[str, Any]:
        """
        Get reward function configuration.
        
        Returns:
            Reward configuration subset
        """
        return self.config_dict.get("reward", {})
    
    def get_state_config(self) -> Dict[str, Any]:
        """
        Get state representation configuration.
        
        Returns:
            State configuration subset
        """
        return self.config_dict.get("state", {})
    
    def save_to_yaml(self, filepath: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            filepath: Path to save YAML file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.config_dict, f, default_flow_style=False)
    
    @classmethod
    def load_from_yaml(cls, filepath: str) -> "BaseTaskConfig":
        """
        Load configuration from YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            TaskConfig instance
        """
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        instance = cls.__new__(cls)
        instance.config_dict = config
        instance.difficulty = config.get("difficulty", "unknown")
        return instance
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self._deep_update(self.config_dict, updates)
    
    def _deep_update(self, base: Dict, updates: Dict) -> None:
        """Recursively update nested dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
