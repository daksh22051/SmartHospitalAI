"""
Configuration Validator

Validates configuration dictionaries against expected schemas.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigValidator:
    """
    Validates configuration parameters and structure.
    
    Ensures that loaded configurations contain required fields
    and have valid values.
    """
    
    REQUIRED_SECTIONS = [
        "environment",
        "state",
        "reward"
    ]
    
    ENVIRONMENT_REQUIRED = [
        "resources",
        "staff",
        "patients"
    ]
    
    def __init__(self) -> None:
        """Initialize config validator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []
        
        # Check required sections
        for section in self.REQUIRED_SECTIONS:
            if section not in config:
                self.errors.append(f"Missing required section: {section}")
        
        # Validate environment section
        if "environment" in config:
            self._validate_environment(config["environment"])
        
        # Validate state section
        if "state" in config:
            self._validate_state(config["state"])
        
        # Validate reward section
        if "reward" in config:
            self._validate_reward(config["reward"])
        
        return len(self.errors) == 0
    
    def _validate_environment(self, env_config: Dict[str, Any]) -> None:
        """Validate environment configuration."""
        for field in self.ENVIRONMENT_REQUIRED:
            if field not in env_config:
                self.errors.append(f"Missing environment field: {field}")
        
        # Validate resources
        if "resources" in env_config:
            resources = env_config["resources"]
            if "icu_beds" not in resources:
                self.warnings.append("icu_beds not specified, will use default")
            elif not isinstance(resources["icu_beds"], int) or resources["icu_beds"] <= 0:
                self.errors.append("icu_beds must be a positive integer")
        
        # Validate patient arrival rate
        if "patients" in env_config:
            patients = env_config["patients"]
            if "arrival_rate" in patients:
                rate = patients["arrival_rate"]
                if rate <= 0:
                    self.errors.append("arrival_rate must be positive")
                elif rate > 20:
                    self.warnings.append(f"arrival_rate {rate} is very high")
    
    def _validate_state(self, state_config: Dict[str, Any]) -> None:
        """Validate state configuration."""
        if "state_dim" in state_config:
            dim = state_config["state_dim"]
            if not isinstance(dim, int) or dim <= 0:
                self.errors.append("state_dim must be a positive integer")
        
        valid_normalizations = ["minmax", "standardize", "none"]
        if "normalization" in state_config:
            norm = state_config["normalization"]
            if norm not in valid_normalizations:
                self.errors.append(f"normalization must be one of {valid_normalizations}")
    
    def _validate_reward(self, reward_config: Dict[str, Any]) -> None:
        """Validate reward configuration."""
        if "reward_weights" in reward_config:
            weights = reward_config["reward_weights"]
            if not isinstance(weights, dict):
                self.errors.append("reward_weights must be a dictionary")
        
        if "gamma" in reward_config:
            gamma = reward_config["gamma"]
            if not (0 < gamma <= 1):
                self.errors.append("gamma must be in (0, 1]")
    
    def get_errors(self) -> List[str]:
        """Return list of validation errors."""
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Return list of validation warnings."""
        return self.warnings.copy()
    
    def print_report(self) -> None:
        """Print validation report."""
        print("Configuration Validation Report")
        print("=" * 40)
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            print("\nConfiguration is valid!")
