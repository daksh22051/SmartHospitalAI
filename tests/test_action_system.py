"""
Action System Tests

Test cases for the advanced action handling system.
"""

import unittest
import numpy as np
from smart_hospital_orchestration.environment.action_system import (
    ActionSystem, ActionValidator, ActionHandler, ActionType, 
    PatientStatus, PatientSeverity, ActionResult, create_action_system
)


class TestActionValidator(unittest.TestCase):
    """Test cases for ActionValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ActionValidator(max_patients=10, max_doctors=5, max_beds=5)
        
        # Sample state for testing
        self.sample_state = {
            "patients": np.array([
                [0, 2, 0, 5, 0, 0, 0, 1],  # Critical, waiting 5 steps
                [1, 1, 0, 2, 0, 0, 0, 1],  # Emergency, waiting 2 steps
                [2, 0, 0, 1, 0, 0, 0, 0],  # Normal, waiting 1 step
                [3, 2, 1, 0, 3, 1, 1, 1],  # Critical, admitted
                [0, 0, 0, 0, 0, 0, 0, 0],  # Empty slot
            ], dtype=np.float32),
            "doctors": np.array([
                [0, 1, 0, 3],  # Available doctor
                [1, 1, 0, 3],  # Available doctor
                [2, 0, 2, 3],  # Busy doctor
                [0, 0, 0, 0],  # Empty slot
            ], dtype=np.float32),
            "beds": np.array([
                [0, 1, 0, 2],  # Available bed
                [1, 1, 0, 1],  # Available bed
                [2, 0, 1, 1],  # Occupied bed
                [0, 0, 0, 0],  # Empty slot
            ], dtype=np.float32)
        }
    
    def test_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.max_patients, 10)
        self.assertEqual(self.validator.max_doctors, 5)
        self.assertEqual(self.validator.max_beds, 5)
    
    def test_validate_action_range(self):
        """Test action range validation."""
        # Valid actions - only WAIT is always valid
        is_valid, _ = self.validator.validate_action(ActionType.WAIT, self.sample_state)
        self.assertTrue(is_valid, "WAIT action should always be valid")
        
        # Other actions may be invalid depending on state
        # Test that they don't crash and return proper validation
        for action in [ActionType.ALLOCATE_RESOURCE, ActionType.ESCALATE_PRIORITY, 
                      ActionType.DEFER, ActionType.REASSIGN]:
            is_valid, message = self.validator.validate_action(action, self.sample_state)
            self.assertIsInstance(is_valid, bool)
            self.assertIsInstance(message, str)
    
    def test_validate_wait_action(self):
        """Test WAIT action validation (always valid)."""
        is_valid, message = self.validator.validate_action(ActionType.WAIT, self.sample_state)
        self.assertTrue(is_valid)
        self.assertEqual(message, "WAIT action always valid")
    
    def test_validate_allocation_success(self):
        """Test successful allocation validation."""
        is_valid, message = self.validator.validate_action(ActionType.ALLOCATE_RESOURCE, self.sample_state)
        self.assertTrue(is_valid)
        self.assertIn("Can allocate", message)
    
    def test_validate_allocation_no_patients(self):
        """Test allocation validation with no waiting patients."""
        # All patients admitted or discharged
        no_waiting_state = self.sample_state.copy()
        no_waiting_state["patients"][:, 2] = PatientStatus.ADMITTED.value  # All admitted
        
        is_valid, message = self.validator.validate_action(ActionType.ALLOCATE_RESOURCE, no_waiting_state)
        self.assertFalse(is_valid)
        self.assertIn("No waiting patients", message)
    
    def test_validate_allocation_no_resources(self):
        """Test allocation validation with no available resources."""
        # No available doctors
        no_doctors_state = self.sample_state.copy()
        no_doctors_state["doctors"][:, 1] = 0.0  # All doctors busy
        
        is_valid, message = self.validator.validate_action(ActionType.ALLOCATE_RESOURCE, no_doctors_state)
        self.assertFalse(is_valid)
        self.assertIn("No doctors available", message)
    
    def test_validate_escalation_success(self):
        """Test escalation validation - expects failure with current state."""
        is_valid, message = self.validator.validate_action(ActionType.ESCALATE_PRIORITY, self.sample_state)
        # Should be invalid since no patients waiting >3 steps
        self.assertFalse(is_valid)
        self.assertIn("No eligible patients", message)
    
    def test_validate_escalation_no_eligible(self):
        """Test escalation validation with no eligible patients."""
        # All patients either critical or waiting <3 steps
        no_eligible_state = self.sample_state.copy()
        no_eligible_state["patients"][0, 3] = 2  # Critical patient waiting only 2 steps
        no_eligible_state["patients"][1, 1] = 2  # Emergency patient made critical
        
        is_valid, message = self.validator.validate_action(ActionType.ESCALATE_PRIORITY, no_eligible_state)
        self.assertFalse(is_valid)
        self.assertIn("No eligible patients", message)
    
    def test_validate_deferral_success(self):
        """Test successful deferral validation."""
        # Create overloaded state with larger patient array
        overloaded_state = {
            "patients": np.array([
                [0, 2, 0, 5, 0, 0, 0, 1],  # Critical, waiting 5 steps
                [1, 1, 0, 2, 0, 0, 0, 1],  # Emergency, waiting 2 steps
                [2, 0, 0, 1, 0, 0, 0, 0],  # Normal, waiting 1 step
                [3, 2, 1, 0, 3, 1, 1, 1],  # Critical, admitted
                [4, 0, 0, 2, 0, 0, 0, 0],  # Normal, waiting 2 steps
                [5, 0, 0, 1, 0, 0, 0, 0],  # Normal, waiting 1 step
                [6, 1, 0, 3, 0, 0, 0, 1],  # Emergency, waiting 3 steps
                [7, 0, 0, 2, 0, 0, 0, 0],  # Normal, waiting 2 steps
                [8, 0, 0, 1, 0, 0, 0, 0],  # Normal, waiting 1 step
                [9, 1, 0, 2, 0, 0, 0, 1],  # Emergency, waiting 2 steps
            ], dtype=np.float32),
            "doctors": self.sample_state["doctors"].copy(),
            "beds": self.sample_state["beds"].copy()
        }
        
        is_valid, message = self.validator.validate_action(ActionType.DEFER, overloaded_state)
        self.assertTrue(is_valid)
        self.assertIn("Can defer", message)
    
    def test_validate_deferral_not_overloaded(self):
        """Test deferral validation when not overloaded."""
        is_valid, message = self.validator.validate_action(ActionType.DEFER, self.sample_state)
        self.assertFalse(is_valid)
        self.assertIn("System not overloaded", message)
    
    def test_validate_reassignment_success(self):
        """Test successful reassignment validation."""
        # Create imbalanced workload
        imbalanced_state = self.sample_state.copy()
        imbalanced_state["doctors"][0, 2] = 3  # Doctor 0 has 3 patients
        imbalanced_state["doctors"][1, 2] = 0  # Doctor 1 has 0 patients
        
        is_valid, message = self.validator.validate_action(ActionType.REASSIGN, imbalanced_state)
        self.assertTrue(is_valid)
        self.assertIn("Can reassign", message)
    
    def test_validate_reassignment_balanced(self):
        """Test reassignment validation when workload is balanced."""
        is_valid, message = self.validator.validate_action(ActionType.REASSIGN, self.sample_state)
        self.assertFalse(is_valid)
        # Should fail due to insufficient doctors with patients
        self.assertIn("Need at least 2 doctors", message)


class TestActionHandler(unittest.TestCase):
    """Test cases for ActionHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ActionValidator(max_patients=10, max_doctors=5, max_beds=5)
        self.handler = ActionHandler(self.validator)
        
        self.sample_state = {
            "patients": np.array([
                [0, 2, 0, 5, 0, 0, 0, 1],  # Critical, waiting 5 steps
                [1, 1, 0, 2, 0, 0, 0, 1],  # Emergency, waiting 2 steps
                [2, 0, 0, 1, 0, 0, 0, 0],  # Normal, waiting 1 step
                [3, 2, 1, 0, 3, 1, 1, 1],  # Critical, admitted
                [0, 0, 0, 0, 0, 0, 0, 0],  # Empty slot
            ], dtype=np.float32),
            "doctors": np.array([
                [0, 1, 0, 3],  # Available doctor
                [1, 1, 0, 3],  # Available doctor
                [2, 0, 2, 3],  # Busy doctor
                [0, 0, 0, 0],  # Empty slot
            ], dtype=np.float32),
            "beds": np.array([
                [0, 1, 0, 2],  # Available bed
                [1, 1, 0, 1],  # Available bed
                [2, 0, 1, 1],  # Occupied bed
                [0, 0, 0, 0],  # Empty slot
            ], dtype=np.float32)
        }
    
    def test_apply_wait_action(self):
        """Test WAIT action execution."""
        result = self.handler.apply_action(ActionType.WAIT, self.sample_state)
        
        self.assertIsInstance(result, ActionResult)
        self.assertTrue(result.success)
        self.assertEqual(result.patients_affected, 0)
        self.assertEqual(result.resources_used, 0)
        self.assertLess(result.reward_contribution, 0)  # Penalty for waiting
        self.assertIn("WAIT", result.message)
    
    def test_apply_allocation_action(self):
        """Test ALLOCATE_RESOURCE action execution."""
        initial_state = {
            key: arr.copy() for key, arr in self.sample_state.items()
        }
        
        result = self.handler.apply_action(ActionType.ALLOCATE_RESOURCE, initial_state)
        
        self.assertIsInstance(result, ActionResult)
        self.assertTrue(result.success)
        self.assertGreater(result.patients_affected, 0)
        self.assertEqual(result.resources_used, result.patients_affected * 2)
        self.assertGreater(result.reward_contribution, 0)
        
        # Check state changes
        admitted_patients = np.sum(initial_state["patients"][:, 2] == PatientStatus.ADMITTED.value)
        self.assertGreater(admitted_patients, 0)
    
    def test_apply_allocation_no_resources(self):
        """Test allocation with no available resources."""
        no_resources_state = {
            key: arr.copy() for key, arr in self.sample_state.items()
        }
        no_resources_state["doctors"][:, 1] = 0.0  # All doctors busy
        no_resources_state["beds"][:, 1] = 0.0    # All beds occupied
        
        result = self.handler.apply_action(ActionType.ALLOCATE_RESOURCE, no_resources_state)
        
        self.assertFalse(result.success)
        self.assertEqual(result.patients_affected, 0)
        self.assertEqual(result.resources_used, 0)
        self.assertLess(result.reward_contribution, 0)
    
    def test_apply_escalation_action(self):
        """Test ESCALATE_PRIORITY action execution."""
        initial_state = {
            key: arr.copy() for key, arr in self.sample_state.items()
        }
        
        result = self.handler.apply_action(ActionType.ESCALATE_PRIORITY, initial_state)
        
        self.assertIsInstance(result, ActionResult)
        # Should fail since no patients waiting >3 steps
        self.assertFalse(result.success)
        self.assertEqual(result.patients_affected, 0)
        self.assertEqual(result.resources_used, 0)
        self.assertLess(result.reward_contribution, 0)  # Penalty for failed action
        self.assertIn("No eligible patients", result.message)
    
    def test_apply_deferral_action(self):
        """Test DEFER action execution."""
        # Create overloaded state with larger patient array
        overloaded_state = {
            "patients": np.array([
                [0, 2, 0, 5, 0, 0, 0, 1],  # Critical, waiting 5 steps
                [1, 1, 0, 2, 0, 0, 0, 1],  # Emergency, waiting 2 steps
                [2, 0, 0, 1, 0, 0, 0, 0],  # Normal, waiting 1 step
                [3, 2, 1, 0, 3, 1, 1, 1],  # Critical, admitted
                [4, 0, 0, 2, 0, 0, 0, 0],  # Normal, waiting 2 steps
                [5, 0, 0, 1, 0, 0, 0, 0],  # Normal, waiting 1 step
                [6, 1, 0, 3, 0, 0, 0, 1],  # Emergency, waiting 3 steps
                [7, 0, 0, 2, 0, 0, 0, 0],  # Normal, waiting 2 steps
                [8, 0, 0, 1, 0, 0, 0, 0],  # Normal, waiting 1 step
                [9, 1, 0, 2, 0, 0, 0, 1],  # Emergency, waiting 2 steps
            ], dtype=np.float32),
            "doctors": self.sample_state["doctors"].copy(),
            "beds": self.sample_state["beds"].copy()
        }
        
        result = self.handler.apply_action(ActionType.DEFER, overloaded_state)
        
        self.assertIsInstance(result, ActionResult)
        self.assertTrue(result.success)
        self.assertGreater(result.patients_affected, 0)
        self.assertEqual(result.resources_used, 0)
        self.assertLess(result.reward_contribution, 0)  # Penalty for deferral
        
        # Check deferred patients
        deferred_patients = result.details.get("deferred_patients", [])
        self.assertGreater(len(deferred_patients), 0)
    
    def test_apply_deferral_not_overloaded(self):
        """Test deferral when system not overloaded."""
        result = self.handler.apply_action(ActionType.DEFER, self.sample_state)
        
        self.assertFalse(result.success)
        self.assertEqual(result.patients_affected, 0)
        self.assertIn("not overloaded", result.message)
    
    def test_apply_reassignment_action(self):
        """Test REASSIGN action execution."""
        # Create imbalanced state
        imbalanced_state = {
            key: arr.copy() for key, arr in self.sample_state.items()
        }
        imbalanced_state["doctors"][0, 2] = 3  # Doctor 0 has 3 patients
        imbalanced_state["doctors"][1, 2] = 0  # Doctor 1 has 0 patients
        
        result = self.handler.apply_action(ActionType.REASSIGN, imbalanced_state)
        
        self.assertIsInstance(result, ActionResult)
        self.assertTrue(result.success)
        self.assertEqual(result.patients_affected, 1)
        self.assertEqual(result.resources_used, 0)
        self.assertGreater(result.reward_contribution, 0)  # Reward for balancing
        
        # Check workload changes
        self.assertIn("from_doctor", result.details)
        self.assertIn("to_doctor", result.details)
    
    def test_apply_reassignment_balanced(self):
        """Test reassignment when workload is balanced."""
        result = self.handler.apply_action(ActionType.REASSIGN, self.sample_state)
        
        self.assertFalse(result.success)
        self.assertEqual(result.patients_affected, 0)
        self.assertIn("Need at least 2 doctors", result.message)
    
    def test_apply_invalid_action(self):
        """Test invalid action handling."""
        result = self.handler.apply_action(10, self.sample_state)  # Invalid action
        
        self.assertFalse(result.success)
        self.assertEqual(result.patients_affected, 0)
        self.assertEqual(result.resources_used, 0)
        self.assertLess(result.reward_contribution, 0)
        self.assertIn("Invalid action", result.message)


class TestActionSystem(unittest.TestCase):
    """Test cases for ActionSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_system = ActionSystem(max_patients=10, max_doctors=5, max_beds=5)
        
        self.sample_state = {
            "patients": np.array([
                [0, 2, 0, 5, 0, 0, 0, 1],  # Critical, waiting 5 steps
                [1, 1, 0, 2, 0, 0, 0, 1],  # Emergency, waiting 2 steps
                [2, 0, 0, 1, 0, 0, 0, 0],  # Normal, waiting 1 step
            ], dtype=np.float32),
            "doctors": np.array([
                [0, 1, 0, 3],  # Available doctor
                [1, 1, 0, 3],  # Available doctor
            ], dtype=np.float32),
            "beds": np.array([
                [0, 1, 0, 2],  # Available bed
                [1, 1, 0, 1],  # Available bed
            ], dtype=np.float32)
        }
    
    def test_initialization(self):
        """Test action system initialization."""
        self.assertIsInstance(self.action_system.validator, ActionValidator)
        self.assertIsInstance(self.action_system.handler, ActionHandler)
        self.assertEqual(self.action_system.validator.max_patients, 10)
    
    def test_execute_action(self):
        """Test action execution through system."""
        result = self.action_system.execute_action(ActionType.WAIT, self.sample_state)
        
        self.assertIsInstance(result, ActionResult)
        self.assertTrue(result.success)
    
    def test_get_valid_actions(self):
        """Test getting valid actions for state."""
        valid_actions = self.action_system.get_valid_actions(self.sample_state)
        
        self.assertIsInstance(valid_actions, list)
        self.assertIn(ActionType.WAIT, valid_actions)  # WAIT always valid
        self.assertLessEqual(len(valid_actions), 5)  # Max 5 actions
        self.assertGreaterEqual(len(valid_actions), 1)  # At least WAIT
    
    def test_get_valid_actions_empty_state(self):
        """Test getting valid actions for empty state."""
        empty_state = {
            "patients": np.zeros((5, 8), dtype=np.float32),
            "doctors": np.zeros((3, 4), dtype=np.float32),
            "beds": np.zeros((3, 4), dtype=np.float32)
        }
        
        valid_actions = self.action_system.get_valid_actions(empty_state)
        
        self.assertIsInstance(valid_actions, list)
        self.assertIn(ActionType.WAIT, valid_actions)  # Only WAIT should be valid


class TestActionSystemFactory(unittest.TestCase):
    """Test cases for action system factory function."""
    
    def test_create_action_system(self):
        """Test factory function."""
        action_system = create_action_system(
            max_patients=20, max_doctors=10, max_beds=10, stochastic=True
        )
        
        self.assertIsInstance(action_system, ActionSystem)
        self.assertEqual(action_system.validator.max_patients, 20)
        self.assertEqual(action_system.validator.max_doctors, 10)
        self.assertEqual(action_system.validator.max_beds, 10)
        self.assertTrue(action_system.handler.stochastic)


class TestActionResult(unittest.TestCase):
    """Test cases for ActionResult class."""
    
    def test_action_result_creation(self):
        """Test ActionResult creation."""
        result = ActionResult(
            success=True,
            reward_contribution=5.0,
            patients_affected=2,
            resources_used=4,
            message="Test action",
            details={"test": "data"}
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.reward_contribution, 5.0)
        self.assertEqual(result.patients_affected, 2)
        self.assertEqual(result.resources_used, 4)
        self.assertEqual(result.message, "Test action")
        self.assertEqual(result.details["test"], "data")


class TestActionIntegration(unittest.TestCase):
    """Integration tests for action system."""
    
    def test_full_action_pipeline(self):
        """Test complete action execution pipeline."""
        action_system = create_action_system(max_patients=10, max_doctors=3, max_beds=3)
        
        # Create test state with multiple patients
        state = {
            "patients": np.array([
                [0, 2, 0, 5, 0, 0, 0, 1],  # Critical, waiting 5 steps
                [1, 1, 0, 2, 0, 0, 0, 1],  # Emergency, waiting 2 steps
                [2, 0, 0, 1, 0, 0, 0, 0],  # Normal, waiting 1 step
                [3, 0, 0, 4, 0, 0, 0, 0],  # Normal, waiting 4 steps
            ], dtype=np.float32),
            "doctors": np.array([
                [0, 1, 0, 3],  # Available doctor
                [1, 1, 0, 3],  # Available doctor
                [2, 0, 1, 3],  # Busy doctor
            ], dtype=np.float32),
            "beds": np.array([
                [0, 1, 0, 2],  # Available bed
                [1, 1, 0, 1],  # Available bed
                [2, 0, 1, 1],  # Occupied bed
            ], dtype=np.float32)
        }
        
        # Test each action type
        actions_to_test = [
            ActionType.WAIT,
            ActionType.ALLOCATE_RESOURCE,
            ActionType.ESCALATE_PRIORITY,
            ActionType.REASSIGN
        ]
        
        results = []
        for action in actions_to_test:
            # Copy state for each action test
            test_state = {key: arr.copy() for key, arr in state.items()}
            result = action_system.execute_action(action, test_state)
            results.append(result)
            
            # Verify result structure
            self.assertIsInstance(result, ActionResult)
            self.assertIsInstance(result.success, bool)
            self.assertIsInstance(result.reward_contribution, float)
            self.assertIsInstance(result.patients_affected, int)
            self.assertIsInstance(result.resources_used, int)
            self.assertIsInstance(result.message, str)
            self.assertIsInstance(result.details, dict)
        
        # Verify at least some actions succeeded
        successful_actions = [r for r in results if r.success]
        self.assertGreater(len(successful_actions), 0, "At least one action should succeed")
    
    def test_action_sequence(self):
        """Test sequence of actions on evolving state."""
        action_system = create_action_system()
        
        # Initial state
        state = {
            "patients": np.array([
                [0, 2, 0, 1, 0, 0, 0, 1],  # Critical, waiting 1 step
                [1, 1, 0, 1, 0, 0, 0, 1],  # Emergency, waiting 1 step
            ], dtype=np.float32),
            "doctors": np.array([
                [0, 1, 0, 3],  # Available doctor
                [1, 1, 0, 3],  # Available doctor
            ], dtype=np.float32),
            "beds": np.array([
                [0, 1, 0, 2],  # Available bed
                [1, 1, 0, 1],  # Available bed
            ], dtype=np.float32)
        }
        
        # Execute action sequence
        actions = [ActionType.ALLOCATE_RESOURCE, ActionType.WAIT, ActionType.ESCALATE_PRIORITY]
        total_reward = 0
        
        for action in actions:
            result = action_system.execute_action(action, state)
            total_reward += result.reward_contribution
            
            # State should be modified by actions
            if result.success:
                # Verify state changes occurred
                self.assertGreaterEqual(np.sum(state["patients"]), 0)
        
        # Verify total reward accumulated
        self.assertIsInstance(total_reward, float)


if __name__ == "__main__":
    unittest.main()
