"""
Configuration Tests

Test cases for configuration management module.
"""

import unittest
import tempfile
import os
from smart_hospital_orchestration.config import ConfigLoader, ConfigValidator


class TestConfigLoader(unittest.TestCase):
    """Test cases for ConfigLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = ConfigLoader()
    
    def test_initialization(self):
        """Test loader initialization."""
        self.assertIsNotNone(self.loader)
    
    def test_load_from_string(self):
        """Test loading from YAML string."""
        yaml_str = """
        environment:
          icu_beds: 10
        state:
          state_dim: 64
        """
        config = self.loader.load_from_string(yaml_str)
        self.assertIn("environment", config)
        self.assertEqual(config["environment"]["icu_beds"], 10)
    
    def test_save_and_load(self):
        """Test saving and loading configuration."""
        config = {"test": {"key": "value"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            self.loader.save(config, temp_path)
            loaded = self.loader.load(temp_path, merge_with_default=False)
            self.assertEqual(loaded["test"]["key"], "value")
        finally:
            os.unlink(temp_path)
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base = {"a": 1, "nested": {"x": 1, "y": 2}}
        override = {"b": 2, "nested": {"y": 3, "z": 4}}
        merged = self.loader._merge_configs(base, override)
        
        self.assertEqual(merged["a"], 1)
        self.assertEqual(merged["b"], 2)
        self.assertEqual(merged["nested"]["x"], 1)
        self.assertEqual(merged["nested"]["y"], 3)
        self.assertEqual(merged["nested"]["z"], 4)


class TestConfigValidator(unittest.TestCase):
    """Test cases for ConfigValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()
    
    def test_initialization(self):
        """Test validator initialization."""
        self.assertIsNotNone(self.validator)
    
    def test_valid_config(self):
        """Test validating a valid configuration."""
        config = {
            "environment": {
                "resources": {"icu_beds": 10},
                "staff": {"doctors": 5},
                "patients": {"arrival_rate": 2.0}
            },
            "state": {"state_dim": 64},
            "reward": {"reward_weights": {}}
        }
        is_valid = self.validator.validate(config)
        self.assertTrue(is_valid)
    
    def test_missing_section(self):
        """Test validating config with missing section."""
        config = {
            "environment": {},
            "state": {}
            # Missing "reward"
        }
        is_valid = self.validator.validate(config)
        self.assertFalse(is_valid)
        self.assertIn("Missing required section: reward", self.validator.get_errors())
    
    def test_invalid_icu_beds(self):
        """Test validating config with invalid icu_beds."""
        config = {
            "environment": {
                "resources": {"icu_beds": -5},
                "staff": {},
                "patients": {}
            },
            "state": {},
            "reward": {}
        }
        is_valid = self.validator.validate(config)
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()
