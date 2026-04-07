"""
Configuration Loader

Handles loading and merging of YAML configuration files.
"""

from typing import Dict, Any, Optional, Union
import yaml
from pathlib import Path
import os


class ConfigLoader:
    """
    Loads and manages configuration from YAML files.
    
    Supports loading from multiple sources with merging capabilities.
    
    Attributes:
        base_path: Base directory for config files
        default_config: Default configuration dictionary
    """
    
    def __init__(self, base_path: Optional[str] = None) -> None:
        """
        Initialize config loader.
        
        Args:
            base_path: Base directory for configuration files
        """
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.default_config: Dict[str, Any] = {}
    
    def load(
        self,
        config_path: Union[str, Path],
        merge_with_default: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            merge_with_default: Whether to merge with default config
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.is_absolute():
            config_path = self.base_path / config_path
        
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f) or {}
        
        if merge_with_default and self.default_config:
            return self._merge_configs(self.default_config, loaded_config)
        
        return loaded_config
    
    def load_from_string(self, yaml_string: str) -> Dict[str, Any]:
        """
        Load configuration from YAML string.
        
        Args:
            yaml_string: YAML-formatted configuration string
            
        Returns:
            Configuration dictionary
        """
        return yaml.safe_load(yaml_string) or {}
    
    def load_multiple(
        self,
        config_paths: list,
        merge_strategy: str = "deep"
    ) -> Dict[str, Any]:
        """
        Load and merge multiple configuration files.
        
        Args:
            config_paths: List of paths to configuration files
            merge_strategy: Strategy for merging ("deep" or "shallow")
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for path in config_paths:
            config = self.load(path, merge_with_default=False)
            if merge_strategy == "deep":
                merged = self._merge_configs(merged, config)
            else:
                merged.update(config)
        
        return merged
    
    def save(self, config: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            filepath: Path to save YAML file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def set_default(self, default_config: Dict[str, Any]) -> None:
        """
        Set default configuration for merging.
        
        Args:
            default_config: Default configuration dictionary
        """
        self.default_config = default_config
    
    def _merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def interpolate_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpolate environment variables in configuration.
        
        Args:
            config: Configuration dictionary with potential ${ENV_VAR} patterns
            
        Returns:
            Configuration with environment variables interpolated
        """
        config_str = yaml.dump(config)
        
        # Find and replace ${VAR} patterns
        import re
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        
        config_str = re.sub(pattern, replace_var, config_str)
        return yaml.safe_load(config_str)
