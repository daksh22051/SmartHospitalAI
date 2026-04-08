"""
Task Configurations Module

Defines different task scenarios (easy, medium, hard) for the environment.
"""

from .base_config import BaseTaskConfig
from .easy_config import EasyTaskConfig
from .medium_config import MediumTaskConfig
from .hard_config import HardTaskConfig
from .config_factory import TaskConfigFactory

__all__ = [
    "BaseTaskConfig",
    "EasyTaskConfig",
    "MediumTaskConfig",
    "HardTaskConfig",
    "TaskConfigFactory",
]
