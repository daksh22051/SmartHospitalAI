"""
Task Configuration Factory

Factory for creating task configurations.
"""

from typing import Dict, Type
from .base_config import BaseTaskConfig
from .easy_config import EasyTaskConfig
from .medium_config import MediumTaskConfig
from .hard_config import HardTaskConfig


class TaskConfigFactory:
    """
    Factory for creating task configuration instances.
    
    Provides a unified interface for creating different task
    configurations by difficulty level.
    """
    
    _registry: Dict[str, Type[BaseTaskConfig]] = {
        "easy": EasyTaskConfig,
        "medium": MediumTaskConfig,
        "hard": HardTaskConfig,
    }
    
    @classmethod
    def create(cls, difficulty: str) -> BaseTaskConfig:
        """
        Create a task configuration by difficulty level.
        
        Args:
            difficulty: Difficulty level ("easy", "medium", "hard")
            
        Returns:
            Task configuration instance
            
        Raises:
            ValueError: If difficulty level is not recognized
        """
        difficulty = difficulty.lower()
        if difficulty not in cls._registry:
            raise ValueError(
                f"Unknown difficulty: {difficulty}. "
                f"Available: {list(cls._registry.keys())}"
            )
        
        config_class = cls._registry[difficulty]
        return config_class()
    
    @classmethod
    def register(cls, difficulty: str, config_class: Type[BaseTaskConfig]) -> None:
        """
        Register a new task configuration class.
        
        Args:
            difficulty: Difficulty identifier
            config_class: Configuration class to register
        """
        cls._registry[difficulty.lower()] = config_class
    
    @classmethod
    def available_difficulties(cls) -> list:
        """
        Get list of available difficulty levels.
        
        Returns:
            List of difficulty strings
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create_all(cls) -> Dict[str, BaseTaskConfig]:
        """
        Create all available task configurations.
        
        Returns:
            Dictionary mapping difficulty to configuration
        """
        return {
            difficulty: cls.create(difficulty)
            for difficulty in cls._registry.keys()
        }
