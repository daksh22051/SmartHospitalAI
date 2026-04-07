"""
State Tests

Test cases for state representation module.
"""

import unittest
import numpy as np
from smart_hospital_orchestration.state import StateRepresentation


class TestStateRepresentation(unittest.TestCase):
    """Test cases for StateRepresentation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {"state_dim": 64}
        self.state_repr = StateRepresentation(self.config)
    
    def test_initialization(self):
        """Test state representation initialization."""
        self.assertIsNotNone(self.state_repr)
        self.assertEqual(self.state_repr.get_state_dimension(), 64)
    
    def test_build_state_not_implemented(self):
        """Test that build_state raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.state_repr.build_state({}, {}, {}, None)


class TestStateNormalizer(unittest.TestCase):
    """Test cases for StateNormalizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smart_hospital_orchestration.state import StateNormalizer
        self.normalizer = StateNormalizer(state_dim=10)
    
    def test_initialization(self):
        """Test normalizer initialization."""
        self.assertIsNotNone(self.normalizer)
        self.assertEqual(self.normalizer.state_dim, 10)
    
    def test_update(self):
        """Test updating running statistics."""
        state = np.random.randn(10)
        self.normalizer.update(state)
        self.assertEqual(self.normalizer.count, 1)
    
    def test_normalize_before_update(self):
        """Test normalization before any updates."""
        state = np.random.randn(10)
        normalized = self.normalizer.normalize(state)
        np.testing.assert_array_equal(normalized, state)


class TestObservationSpace(unittest.TestCase):
    """Test cases for ObservationSpace class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smart_hospital_orchestration.state import ObservationSpace
        self.config = {"state_dim": 128}
        self.obs_space = ObservationSpace(self.config)
    
    def test_initialization(self):
        """Test observation space initialization."""
        self.assertIsNotNone(self.obs_space)
        self.assertEqual(self.obs_space.dim, 128)


if __name__ == "__main__":
    unittest.main()
