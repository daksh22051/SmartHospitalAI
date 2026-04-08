"""
State Representation Tests

Test cases for the advanced state representation system.
"""

import unittest
import numpy as np
from smart_hospital_orchestration.state.state import StateEncoder, StateConfig, PatientStatus, PatientSeverity, create_state_encoder


class TestStateEncoder(unittest.TestCase):
    """Test cases for StateEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = StateEncoder()
        
        # Sample data for testing
        self.sample_patients = [
            {"id": 0, "severity": 2, "status": 0, "wait_time": 3, "treatment_time": 0},
            {"id": 1, "severity": 1, "status": 1, "wait_time": 1, "treatment_time": 2},
            {"id": 2, "severity": 0, "status": 0, "wait_time": 5, "treatment_time": 0}
        ]
        
        self.sample_doctors = [
            {"id": 0, "available": True, "current_load": 0, "max_load": 3},
            {"id": 1, "available": False, "current_load": 2, "max_load": 3}
        ]
        
        self.sample_beds = [
            {"id": 0, "available": True, "assigned_patient": None, "equipment": ["monitor"]},
            {"id": 1, "available": False, "assigned_patient": 0, "equipment": ["ventilator"]}
        ]
    
    def test_initialization(self):
        """Test encoder initialization."""
        self.assertIsNotNone(self.encoder)
        self.assertIsInstance(self.encoder.config, StateConfig)
        self.assertEqual(self.encoder.config.MAX_PATIENTS, 50)
    
    def test_create_empty_state(self):
        """Test empty state creation."""
        state = self.encoder.create_empty_state()
        
        # Check all required keys exist
        required_keys = ["patients", "doctors", "beds", "resources", "time", "global", "metadata"]
        for key in required_keys:
            self.assertIn(key, state)
        
        # Check array shapes
        self.assertEqual(state["patients"].shape, (50, 8))
        self.assertEqual(state["doctors"].shape, (20, 4))
        self.assertEqual(state["beds"].shape, (20, 4))
        self.assertEqual(state["resources"].shape, (3, 3))
        self.assertEqual(state["time"].shape, (4,))
        self.assertEqual(state["global"].shape, (3,))
        
        # Check arrays are initialized to zeros
        self.assertTrue(np.all(state["patients"] == 0))
        self.assertTrue(np.all(state["doctors"] == 0))
    
    def test_encode_patients(self):
        """Test patient encoding."""
        patients_array = self.encoder.encode_patients(self.sample_patients)
        
        self.assertEqual(patients_array.shape, (50, 8))
        
        # Check first patient (critical, waiting)
        first_patient = patients_array[0]
        self.assertEqual(first_patient[0], 0)  # id
        self.assertEqual(first_patient[1], 2)  # critical severity
        self.assertEqual(first_patient[2], 0)  # waiting status
        self.assertEqual(first_patient[3], 3)  # wait_time
        self.assertEqual(first_patient[7], 1)  # emergency flag
        
        # Check second patient (emergency, admitted)
        second_patient = patients_array[1]
        self.assertEqual(second_patient[1], 1)  # emergency severity
        self.assertEqual(second_patient[2], 1)  # admitted status
        
        # Check padding (remaining patients should be zeros)
        self.assertTrue(np.all(patients_array[3:] == 0))
    
    def test_encode_doctors(self):
        """Test doctor encoding."""
        doctors_array = self.encoder.encode_doctors(self.sample_doctors)
        
        self.assertEqual(doctors_array.shape, (20, 4))
        
        # Check first doctor (available)
        first_doctor = doctors_array[0]
        self.assertEqual(first_doctor[0], 0)  # id
        self.assertEqual(first_doctor[1], 1)  # available
        self.assertEqual(first_doctor[2], 0)  # current_load
        self.assertEqual(first_doctor[3], 3)  # max_load
        
        # Check second doctor (busy)
        second_doctor = doctors_array[1]
        self.assertEqual(second_doctor[1], 0)  # not available
        self.assertEqual(second_doctor[2], 2)  # current_load
    
    def test_encode_beds(self):
        """Test bed encoding."""
        beds_array = self.encoder.encode_beds(self.sample_beds)
        
        self.assertEqual(beds_array.shape, (20, 4))
        
        # Check first bed (available)
        first_bed = beds_array[0]
        self.assertEqual(first_bed[0], 0)  # id
        self.assertEqual(first_bed[1], 1)  # available
        self.assertEqual(first_bed[2], 0)  # no assigned patient
        self.assertEqual(first_bed[3], 1)  # 1 equipment item
        
        # Check second bed (occupied)
        second_bed = beds_array[1]
        self.assertEqual(second_bed[1], 0)  # not available
        self.assertEqual(second_bed[2], 1)  # has assigned patient
    
    def test_encode_resources(self):
        """Test resource encoding."""
        resources_array = self.encoder.encode_resources(
            self.sample_doctors, self.sample_beds
        )
        
        self.assertEqual(resources_array.shape, (3, 3))
        
        # Check doctors resource
        doctors_res = resources_array[0]
        self.assertEqual(doctors_res[0], 1)  # 1 available doctor
        self.assertEqual(doctors_res[1], 2)  # 2 total doctors
        self.assertEqual(doctors_res[2], 0.5)  # 50% utilization
        
        # Check beds resource
        beds_res = resources_array[1]
        self.assertEqual(beds_res[0], 1)  # 1 available bed
        self.assertEqual(beds_res[1], 2)  # 2 total beds
        self.assertEqual(beds_res[2], 0.5)  # 50% utilization
    
    def test_encode_time(self):
        """Test time encoding."""
        time_array = self.encoder.encode_time(step=25, max_steps=100)
        
        self.assertEqual(time_array.shape, (4,))
        self.assertEqual(time_array[0], 0.25)  # normalized step
        self.assertGreaterEqual(time_array[1], 0)   # hour (can be 0)
        self.assertGreaterEqual(time_array[2], 0)   # day (can be 0)
        self.assertEqual(time_array[3], 0)     # load ratio (will be updated)
    
    def test_encode_global(self):
        """Test global feature encoding."""
        global_array = self.encoder.encode_global(self.sample_patients, step=10)
        
        self.assertEqual(global_array.shape, (3,))
        self.assertEqual(global_array[0], 3/50)  # normalized total patients
        self.assertEqual(global_array[1], 1/50)  # 1 critical waiting
        self.assertLessEqual(global_array[2], 1)  # efficiency score
    
    def test_build_state(self):
        """Test complete state building."""
        state = self.encoder.build_state(
            patients=self.sample_patients,
            doctors=self.sample_doctors,
            beds=self.sample_beds,
            step=15,
            max_steps=100
        )
        
        # Validate state structure
        self.assertTrue(self.encoder.validate_state(state))
        
        # Check metadata
        metadata = state["metadata"]
        self.assertEqual(metadata["step"], 15)
        self.assertIn("episode_time", metadata)
        self.assertGreater(metadata["utilization"], 0)
        self.assertEqual(metadata["emergency_level"], 2)  # 2 patients with severity >= 1
    
    def test_to_tensor(self):
        """Test tensor conversion."""
        state = self.encoder.build_state(
            patients=self.sample_patients,
            doctors=self.sample_doctors,
            beds=self.sample_beds,
            step=10,
            max_steps=100
        )
        
        tensor_state = self.encoder.to_tensor(state)
        
        # Check all arrays are present and correct type
        self.assertIn("patients", tensor_state)
        self.assertIn("doctors", tensor_state)
        self.assertIn("beds", tensor_state)
        
        # Check arrays are numpy arrays
        for key, array in tensor_state.items():
            self.assertIsInstance(array, np.ndarray)
    
    def test_flatten_state(self):
        """Test state flattening."""
        state = self.encoder.build_state(
            patients=self.sample_patients,
            doctors=self.sample_doctors,
            beds=self.sample_beds,
            step=10,
            max_steps=100
        )
        
        flattened = self.encoder.flatten_state(state)
        
        # Check flattened dimension
        expected_dim = (50 * 8) + (20 * 4) + (20 * 4) + (3 * 3) + 4 + 3
        self.assertEqual(len(flattened), expected_dim)
        self.assertEqual(flattened.dtype, np.float32)
    
    def test_get_state_dimension(self):
        """Test state dimension calculation."""
        dim = self.encoder.get_state_dimension()
        expected_dim = (50 * 8) + (20 * 4) + (20 * 4) + (3 * 3) + 4 + 3
        self.assertEqual(dim, expected_dim)
    
    def test_validate_state(self):
        """Test state validation."""
        # Valid state should pass
        valid_state = self.encoder.build_state(
            patients=self.sample_patients,
            doctors=self.sample_doctors,
            beds=self.sample_beds,
            step=10,
            max_steps=100
        )
        self.assertTrue(self.encoder.validate_state(valid_state))
        
        # Invalid state should fail
        invalid_state = {"patients": np.zeros((10, 8))}  # Wrong shape
        with self.assertRaises(AssertionError):
            self.encoder.validate_state(invalid_state)
    
    def test_custom_config(self):
        """Test encoder with custom configuration."""
        custom_config = StateConfig(MAX_PATIENTS=10, MAX_DOCTORS=5, MAX_BEDS=5)
        custom_encoder = StateEncoder(custom_config)
        
        state = custom_encoder.create_empty_state()
        
        self.assertEqual(state["patients"].shape, (10, 8))
        self.assertEqual(state["doctors"].shape, (5, 4))
        self.assertEqual(state["beds"].shape, (5, 4))
    
    def test_create_state_encoder_factory(self):
        """Test factory function."""
        encoder = create_state_encoder(max_patients=20, max_doctors=10, max_beds=10)
        
        self.assertEqual(encoder.config.MAX_PATIENTS, 20)
        self.assertEqual(encoder.config.MAX_DOCTORS, 10)
        self.assertEqual(encoder.config.MAX_BEDS, 10)


class TestStateEnums(unittest.TestCase):
    """Test state enumeration classes."""
    
    def test_patient_status_enum(self):
        """Test PatientStatus enum."""
        self.assertEqual(PatientStatus.WAITING, 0)
        self.assertEqual(PatientStatus.ADMITTED, 1)
        self.assertEqual(PatientStatus.IN_TREATMENT, 2)
        self.assertEqual(PatientStatus.DISCHARGED, 3)
        self.assertEqual(PatientStatus.DEFERRED, 4)
    
    def test_patient_severity_enum(self):
        """Test PatientSeverity enum."""
        self.assertEqual(PatientSeverity.NORMAL, 0)
        self.assertEqual(PatientSeverity.EMERGENCY, 1)
        self.assertEqual(PatientSeverity.CRITICAL, 2)


class TestStateIntegration(unittest.TestCase):
    """Integration tests for state system."""
    
    def test_large_patient_count(self):
        """Test handling of many patients."""
        encoder = StateEncoder()
        
        # Create many patients (more than default max)
        many_patients = [
            {"id": i, "severity": i % 3, "status": 0, "wait_time": i, "treatment_time": 0}
            for i in range(100)  # 100 patients, but max is 50
        ]
        
        state = encoder.build_state(
            patients=many_patients,
            doctors=[],
            beds=[],
            step=0,
            max_steps=100
        )
        
        # Should only have first 50 patients
        self.assertEqual(state["patients"].shape[0], 50)
        # First patient should be from original data
        self.assertEqual(state["patients"][0][0], 0)
        # 49th patient should be from original data
        self.assertEqual(state["patients"][49][0], 49)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        encoder = StateEncoder()
        
        state = encoder.build_state(
            patients=[],
            doctors=[],
            beds=[],
            step=0,
            max_steps=100
        )
        
        # Should handle empty data gracefully
        self.assertEqual(len(state["patients"]), 50)  # Default max patients
        self.assertEqual(state["metadata"]["emergency_level"], 0)
        self.assertEqual(state["metadata"]["utilization"], 0.0)
        self.assertTrue(np.all(state["patients"] == 0))
        self.assertTrue(np.all(state["doctors"] == 0))
        self.assertTrue(np.all(state["beds"] == 0))


if __name__ == "__main__":
    unittest.main()
