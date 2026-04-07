"""
Task Configurations Tests

Test cases for task configuration module.
"""

import unittest
from smart_hospital_orchestration.tasks import (
    TaskConfigFactory,
    EasyTaskConfig,
    MediumTaskConfig,
    HardTaskConfig
)


class TestTaskConfigFactory(unittest.TestCase):
    """Test cases for TaskConfigFactory."""
    
    def test_create_easy(self):
        """Test creating easy configuration."""
        config = TaskConfigFactory.create("easy")
        self.assertIsInstance(config, EasyTaskConfig)
        self.assertEqual(config.difficulty, "easy")
    
    def test_create_medium(self):
        """Test creating medium configuration."""
        config = TaskConfigFactory.create("medium")
        self.assertIsInstance(config, MediumTaskConfig)
        self.assertEqual(config.difficulty, "medium")
    
    def test_create_hard(self):
        """Test creating hard configuration."""
        config = TaskConfigFactory.create("hard")
        self.assertIsInstance(config, HardTaskConfig)
        self.assertEqual(config.difficulty, "hard")
    
    def test_create_invalid(self):
        """Test creating invalid configuration."""
        with self.assertRaises(ValueError):
            TaskConfigFactory.create("invalid")
    
    def test_available_difficulties(self):
        """Test getting available difficulties."""
        difficulties = TaskConfigFactory.available_difficulties()
        self.assertIn("easy", difficulties)
        self.assertIn("medium", difficulties)
        self.assertIn("hard", difficulties)


class TestEasyTaskConfig(unittest.TestCase):
    """Test cases for EasyTaskConfig."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = EasyTaskConfig()
    
    def test_difficulty(self):
        """Test difficulty level."""
        self.assertEqual(self.config.difficulty, "easy")
    
    def test_get_config(self):
        """Test getting configuration dictionary."""
        cfg = self.config.get_config()
        self.assertIn("environment", cfg)
        self.assertIn("state", cfg)
        self.assertIn("reward", cfg)
    
    def test_environment_config(self):
        """Test environment configuration."""
        env_cfg = self.config.get_environment_config()
        self.assertIn("resources", env_cfg)
        self.assertEqual(env_cfg["resources"]["icu_beds"], 10)


class TestMediumTaskConfig(unittest.TestCase):
    """Test cases for MediumTaskConfig."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MediumTaskConfig()
    
    def test_difficulty(self):
        """Test difficulty level."""
        self.assertEqual(self.config.difficulty, "medium")
    
    def test_environment_config(self):
        """Test environment configuration."""
        env_cfg = self.config.get_environment_config()
        self.assertEqual(env_cfg["resources"]["icu_beds"], 20)


class TestHardTaskConfig(unittest.TestCase):
    """Test cases for HardTaskConfig."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HardTaskConfig()
    
    def test_difficulty(self):
        """Test difficulty level."""
        self.assertEqual(self.config.difficulty, "hard")
    
    def test_environment_config(self):
        """Test environment configuration."""
        env_cfg = self.config.get_environment_config()
        self.assertEqual(env_cfg["resources"]["icu_beds"], 30)


if __name__ == "__main__":
    unittest.main()
