"""
Integration Tests

End-to-end tests for the hospital orchestration system.
"""

import unittest
from smart_hospital_orchestration.environment import HospitalEnv
from smart_hospital_orchestration.agent import RandomAgent
from smart_hospital_orchestration.tasks import EasyTaskConfig
from smart_hospital_orchestration.state import StateRepresentation
from smart_hospital_orchestration.reward import RewardFunction


class TestEnvironmentIntegration(unittest.TestCase):
    """Integration tests for environment components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = EasyTaskConfig().get_config()
    
    def test_config_loading(self):
        """Test loading and using task configuration."""
        env_config = self.config.get("environment", {})
        self.assertIn("resources", env_config)
        self.assertIn("state", self.config)
    
    def test_agent_environment_interaction(self):
        """Test agent interacting with environment interface."""
        agent = RandomAgent()
        
        # Verify agent can be created
        self.assertIsNotNone(agent)
        
        # Verify agent can select actions
        obs = np.zeros(64)
        action = agent.act(obs)
        self.assertIsNotNone(action)


class TestFullPipeline(unittest.TestCase):
    """Full pipeline integration tests."""
    
    def test_component_initialization(self):
        """Test that all components can be initialized together."""
        config = EasyTaskConfig().get_config()
        
        # Initialize components
        env = HospitalEnv("easy")  # Pass string task name
        agent = RandomAgent(config.get("agent", {}))
        state_repr = StateRepresentation(config.get("state", {}))
        reward_fn = RewardFunction(config.get("reward", {}))
        
        # Verify all components initialized
        self.assertIsNotNone(env)
        self.assertIsNotNone(agent)
        self.assertIsNotNone(state_repr)
        self.assertIsNotNone(reward_fn)


import numpy as np

if __name__ == "__main__":
    unittest.main()
