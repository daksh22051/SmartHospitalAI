"""
Agent Tests

Test cases for agent module.
"""

import unittest
import numpy as np
from smart_hospital_orchestration.agent import RandomAgent, HeuristicAgent


class TestRandomAgent(unittest.TestCase):
    """Test cases for RandomAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = RandomAgent()
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.name, "RandomAgent")
    
    def test_act(self):
        """Test action selection."""
        obs = np.zeros(10)
        action = self.agent.act(obs)
        self.assertIsInstance(action, np.ndarray)
    
    def test_reset(self):
        """Test agent reset."""
        # Should not raise any exception
        self.agent.reset()
    
    def test_set_seed(self):
        """Test setting random seed."""
        self.agent.set_seed(42)


class TestHeuristicAgent(unittest.TestCase):
    """Test cases for HeuristicAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {"heuristic_type": "hybrid"}
        self.agent = HeuristicAgent(self.config)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.heuristic_type, "hybrid")
    
    def test_invalid_heuristic(self):
        """Test invalid heuristic type."""
        with self.assertRaises(ValueError):
            HeuristicAgent({"heuristic_type": "invalid"})
    
    def test_reset(self):
        """Test agent reset."""
        self.agent.reset()


if __name__ == "__main__":
    unittest.main()
