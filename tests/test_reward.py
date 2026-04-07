"""
Reward Tests

Test cases for reward computation module.
"""

import unittest
from smart_hospital_orchestration.reward import RewardFunction


class TestRewardFunction(unittest.TestCase):
    """Test cases for RewardFunction class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "reward_weights": {
                "patient_outcome": 1.0,
                "resource_utilization": 0.5
            }
        }
        self.reward_fn = RewardFunction(self.config)
    
    def test_initialization(self):
        """Test reward function initialization."""
        self.assertIsNotNone(self.reward_fn)
        self.assertEqual(self.reward_fn.weights["patient_outcome"], 1.0)
    
    def test_update_weights(self):
        """Test updating reward weights."""
        new_weights = {"patient_outcome": 2.0}
        self.reward_fn.update_weights(new_weights)
        self.assertEqual(self.reward_fn.weights["patient_outcome"], 2.0)


class TestRewardComponents(unittest.TestCase):
    """Test cases for RewardComponents class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smart_hospital_orchestration.reward import RewardComponents
        self.config = {"target_utilization": 0.80}
        self.components = RewardComponents(self.config)
    
    def test_initialization(self):
        """Test reward components initialization."""
        self.assertIsNotNone(self.components)
        self.assertEqual(self.components.target_utilization, 0.80)


if __name__ == "__main__":
    unittest.main()
