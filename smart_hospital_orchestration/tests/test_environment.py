"""
Environment Tests

Test cases for the hospital environment.
"""

import unittest
import numpy as np
from smart_hospital_orchestration.environment import HospitalEnv
from smart_hospital_orchestration.environment.hospital_env import (
    Patient, PatientSeverity, PatientStatus, 
    Doctor, ICUBed, ActionType
)


class TestHospitalEnv(unittest.TestCase):
    """Test cases for HospitalEnv class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = HospitalEnv(task="easy")
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.task, "easy")
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(len(self.env.doctors), 3)
        self.assertEqual(len(self.env.beds), 5)
    
    def test_reset(self):
        """Test environment reset."""
        state, info = self.env.reset(seed=42)
        
        self.assertIsInstance(state, dict)
        self.assertIn("patients", state)
        self.assertIn("beds", state)
        self.assertIn("doctors", state)
        self.assertIn("readable", state)
        self.assertIn("flat", state)
        
        self.assertIsInstance(info, dict)
        self.assertIn("num_patients", info)
        self.assertGreater(info["num_patients"], 0)
        self.assertIn("available_doctors", info)
        self.assertIn("available_beds", info)
    
    def test_action_space(self):
        """Test action space properties."""
        self.assertEqual(self.env.action_space.n, 5)
        for action in range(5):
            self.assertTrue(self.env.action_space.contains(action))
    
    def test_step_wait_action(self):
        """Test WAIT action (0)."""
        self.env.reset(seed=42)
        state, reward, done, info = self.env.step(0)
        
        self.assertIsInstance(state, dict)
        self.assertIsInstance(reward, float)
        self.assertEqual(info["action"], "WAIT")
        self.assertEqual(self.env.current_step, 1)
        self.assertIsInstance(done, bool)
        self.assertIn("terminated", info)
        self.assertIn("truncated", info)
    
    def test_step_allocate_action(self):
        """Test ALLOCATE_RESOURCE action (1)."""
        self.env.reset(seed=42)
        state, reward, done, info = self.env.step(1)
        
        self.assertEqual(info["action"], "ALLOCATE_RESOURCE")
        self.assertIsInstance(done, bool)
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        self.env.reset(seed=42)
        
        done = False
        steps = 0
        max_steps = 150
        
        while not done and steps < max_steps:
            action = self.env.action_space.sample()
            state, reward, done, info = self.env.step(action)
            steps += 1
        
        self.assertTrue(done or steps >= max_steps)
    
    def test_state_consistency(self):
        """Test that state remains consistent across steps."""
        self.env.reset(seed=42)
        
        for _ in range(10):
            action = self.env.action_space.sample()
            state, reward, done, info = self.env.step(action)
            
            readable = state["readable"]
            self.assertEqual(readable["step"], self.env.current_step)
            self.assertEqual(readable["total_patients"], len(self.env.patients))
    
    def test_render(self):
        """Test rendering functionality."""
        self.env.reset(seed=42)
        text = self.env.render()
        self.assertIsInstance(text, str)
        self.assertIn("Hospital State", text)


class TestPatient(unittest.TestCase):
    """Test cases for Patient class."""
    
    def test_patient_creation(self):
        """Test patient initialization."""
        patient = Patient(patient_id=0, severity=PatientSeverity.EMERGENCY)
        self.assertEqual(patient.patient_id, 0)
        self.assertEqual(patient.severity, PatientSeverity.EMERGENCY)
        self.assertEqual(patient.status, PatientStatus.WAITING)
    
    def test_patient_to_array(self):
        """Test patient array conversion."""
        patient = Patient(
            patient_id=1, severity=PatientSeverity.CRITICAL,
            status=PatientStatus.ADMITTED, wait_time=3
        )
        arr = patient.to_array()
        self.assertEqual(len(arr), 9)
        self.assertEqual(arr[0], 1)


class TestDoctor(unittest.TestCase):
    """Test cases for Doctor class."""
    
    def test_doctor_creation(self):
        """Test doctor initialization."""
        doctor = Doctor(resource_id=0)
        self.assertEqual(doctor.resource_id, 0)
        self.assertTrue(doctor.is_available)
    
    def test_doctor_allocate(self):
        """Test doctor patient allocation."""
        doctor = Doctor(resource_id=0)
        self.assertTrue(doctor.can_accept())
        self.assertTrue(doctor.allocate(0))


class TestDifferentTasks(unittest.TestCase):
    """Test different task difficulties."""
    
    def test_easy_task(self):
        """Test easy task configuration."""
        env = HospitalEnv(task="easy")
        self.assertEqual(len(env.doctors), 3)
        self.assertEqual(len(env.beds), 5)
    
    def test_medium_task(self):
        """Test medium task configuration."""
        env = HospitalEnv(task="medium")
        self.assertEqual(len(env.doctors), 4)
        self.assertEqual(len(env.beds), 6)
    
    def test_hard_task(self):
        """Test hard task configuration."""
        env = HospitalEnv(task="hard")
        self.assertEqual(len(env.doctors), 5)
        self.assertEqual(len(env.beds), 7)


if __name__ == "__main__":
    unittest.main()
