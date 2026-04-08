"""
State Representation Module

Advanced state representation for Smart Hospital Resource Orchestration Environment.
Designed for RL compatibility with fixed-size arrays and tensor conversion capabilities.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum


class PatientStatus(IntEnum):
    """Patient status enumeration for numerical encoding."""
    WAITING = 0
    ADMITTED = 1
    IN_TREATMENT = 2
    DISCHARGED = 3
    DEFERRED = 4


class PatientSeverity(IntEnum):
    """Patient severity enumeration for numerical encoding."""
    NORMAL = 0
    EMERGENCY = 1
    CRITICAL = 2


class ResourceType(IntEnum):
    """Resource type enumeration."""
    DOCTORS = 0
    ICU_BEDS = 1
    OPERATION_ROOMS = 2


@dataclass
class StateConfig:
    """Configuration for state representation dimensions."""
    MAX_PATIENTS: int = 50
    MAX_DOCTORS: int = 20
    MAX_BEDS: int = 20
    NUM_RESOURCES: int = 3
    TIME_FEATURES: int = 4
    GLOBAL_FEATURES: int = 3
    
    # Feature indices for clarity
    PATIENT_FEATURES = 8
    DOCTOR_FEATURES = 4
    BED_FEATURES = 4
    RESOURCE_FEATURES = 3


class StateEncoder:
    """
    Advanced state encoder for hospital resource orchestration.
    
    Provides RL-friendly fixed-size state representations with tensor conversion.
    """
    
    def __init__(self, config: Optional[StateConfig] = None):
        """
        Initialize state encoder with configuration.
        
        Args:
            config: State configuration parameters
        """
        self.config = config or StateConfig()
        self._validate_dimensions()
    
    def _validate_dimensions(self) -> None:
        """Validate dimension consistency."""
        assert self.config.PATIENT_FEATURES == 8, "Patient features must be 8"
        assert self.config.DOCTOR_FEATURES == 4, "Doctor features must be 4"
        assert self.config.BED_FEATURES == 4, "Bed features must be 4"
    
    def create_empty_state(self) -> Dict[str, Any]:
        """
        Create initialized empty state with zeros.
        
        Returns:
            Dictionary with all state arrays initialized to zeros
        """
        return {
            "patients": np.zeros(
                (self.config.MAX_PATIENTS, self.config.PATIENT_FEATURES), 
                dtype=np.float32
            ),
            "doctors": np.zeros(
                (self.config.MAX_DOCTORS, self.config.DOCTOR_FEATURES), 
                dtype=np.float32
            ),
            "beds": np.zeros(
                (self.config.MAX_BEDS, self.config.BED_FEATURES), 
                dtype=np.float32
            ),
            "resources": np.zeros(
                (self.config.NUM_RESOURCES, self.config.RESOURCE_FEATURES), 
                dtype=np.float32
            ),
            "time": np.zeros(
                (self.config.TIME_FEATURES,), 
                dtype=np.float32
            ),
            "global": np.zeros(
                (self.config.GLOBAL_FEATURES,), 
                dtype=np.float32
            ),
            "metadata": {
                "step": 0,
                "episode_time": "00:00",
                "utilization": 0.0,
                "emergency_level": 0
            }
        }
    
    def encode_patients(self, patients_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode patient data into fixed-size array with padding.
        
        Args:
            patients_data: List of patient dictionaries
            
        Returns:
            Fixed-size patient array (MAX_PATIENTS, 8)
        """
        patient_array = np.zeros(
            (self.config.MAX_PATIENTS, self.config.PATIENT_FEATURES), 
            dtype=np.float32
        )
        
        for i, patient in enumerate(patients_data[:self.config.MAX_PATIENTS]):
            # Extract and encode patient features
            patient_array[i, 0] = patient.get("id", 0)
            patient_array[i, 1] = PatientSeverity(patient.get("severity", 0))
            patient_array[i, 2] = PatientStatus(patient.get("status", 0))
            patient_array[i, 3] = patient.get("wait_time", 0)
            patient_array[i, 4] = patient.get("treatment_time", 0)
            patient_array[i, 5] = 1.0 if patient.get("assigned_bed") is not None else 0.0
            patient_array[i, 6] = 1.0 if patient.get("assigned_doctor") is not None else 0.0
            patient_array[i, 7] = 1.0 if patient.get("severity", 0) >= 1 else 0.0  # Emergency flag
        
        return patient_array
    
    def encode_doctors(self, doctors_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode doctor data into fixed-size array.
        
        Args:
            doctors_data: List of doctor dictionaries
            
        Returns:
            Fixed-size doctor array (MAX_DOCTORS, 4)
        """
        doctor_array = np.zeros(
            (self.config.MAX_DOCTORS, self.config.DOCTOR_FEATURES), 
            dtype=np.float32
        )
        
        for i, doctor in enumerate(doctors_data[:self.config.MAX_DOCTORS]):
            doctor_array[i, 0] = doctor.get("id", 0)
            doctor_array[i, 1] = 1.0 if doctor.get("available", True) else 0.0
            doctor_array[i, 2] = doctor.get("current_load", 0)
            doctor_array[i, 3] = doctor.get("max_load", 3)
        
        return doctor_array
    
    def encode_beds(self, beds_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode ICU bed data into fixed-size array.
        
        Args:
            beds_data: List of bed dictionaries
            
        Returns:
            Fixed-size bed array (MAX_BEDS, 4)
        """
        bed_array = np.zeros(
            (self.config.MAX_BEDS, self.config.BED_FEATURES), 
            dtype=np.float32
        )
        
        for i, bed in enumerate(beds_data[:self.config.MAX_BEDS]):
            bed_array[i, 0] = bed.get("id", 0)
            bed_array[i, 1] = 1.0 if bed.get("available", True) else 0.0
            bed_array[i, 2] = 1.0 if bed.get("assigned_patient") is not None else 0.0
            bed_array[i, 3] = len(bed.get("equipment", []))
        
        return bed_array
    
    def encode_resources(self, doctors: List[Dict], beds: List[Dict], 
                        operation_rooms: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Encode resource availability summary.
        
        Args:
            doctors: List of doctor data
            beds: List of bed data
            operation_rooms: Optional list of operation room data
            
        Returns:
            Resource summary array (NUM_RESOURCES, 3)
        """
        resource_array = np.zeros(
            (self.config.NUM_RESOURCES, self.config.RESOURCE_FEATURES), 
            dtype=np.float32
        )
        
        # Doctors
        total_doctors = len(doctors)
        available_doctors = sum(1 for d in doctors if d.get("available", True))
        resource_array[ResourceType.DOCTORS] = [available_doctors, total_doctors, 
                                              (total_doctors - available_doctors) / max(total_doctors, 1)]
        
        # ICU Beds
        total_beds = len(beds)
        available_beds = sum(1 for b in beds if b.get("available", True))
        resource_array[ResourceType.ICU_BEDS] = [available_beds, total_beds,
                                                (total_beds - available_beds) / max(total_beds, 1)]
        
        # Operation Rooms (optional)
        if operation_rooms:
            total_rooms = len(operation_rooms)
            available_rooms = sum(1 for r in operation_rooms if r.get("available", True))
            resource_array[ResourceType.OPERATION_ROOMS] = [available_rooms, total_rooms,
                                                           (total_rooms - available_rooms) / max(total_rooms, 1)]
        else:
            resource_array[ResourceType.OPERATION_ROOMS] = [0, 0, 0]
        
        return resource_array
    
    def encode_time(self, step: int, max_steps: int) -> np.ndarray:
        """
        Encode time-related features.
        
        Args:
            step: Current simulation step
            max_steps: Maximum simulation steps
            
        Returns:
            Time feature array (TIME_FEATURES,)
        """
        # Simulate 24-hour hospital cycle
        hour = int((step % 100) * 24 / 100)  # Map steps to hours
        day = int(step / 100) % 7  # Day of week
        
        # Calculate load ratio (patients / total capacity)
        load_ratio = 0.0  # Will be updated with actual patient data
        
        return np.array([
            step / max_steps,  # Normalized step
            hour / 24,         # Normalized hour
            day / 7,           # Normalized day
            load_ratio         # Hospital load
        ], dtype=np.float32)
    
    def encode_global(self, patients: List[Dict], step: int) -> np.ndarray:
        """
        Encode global hospital metrics.
        
        Args:
            patients: List of patient data
            step: Current simulation step
            
        Returns:
            Global feature array (GLOBAL_FEATURES,)
        """
        total_patients = len(patients)
        critical_waiting = sum(1 for p in patients 
                             if p.get("severity", 0) == 2 and p.get("status", 0) == 0)
        
        # Efficiency score: ratio of admitted patients to total
        admitted = sum(1 for p in patients if p.get("status", 0) == 1)
        efficiency = admitted / max(total_patients, 1)
        
        return np.array([
            total_patients / self.config.MAX_PATIENTS,
            critical_waiting / self.config.MAX_PATIENTS,
            efficiency
        ], dtype=np.float32)
    
    def build_state(self, patients: List[Dict], doctors: List[Dict], 
                    beds: List[Dict], step: int, max_steps: int,
                    operation_rooms: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Build complete state representation from raw data.
        
        Args:
            patients: List of patient data
            doctors: List of doctor data
            beds: List of bed data
            step: Current simulation step
            max_steps: Maximum simulation steps
            operation_rooms: Optional operation room data
            
        Returns:
            Complete state dictionary
        """
        state = self.create_empty_state()
        
        # Encode each component
        state["patients"] = self.encode_patients(patients)
        state["doctors"] = self.encode_doctors(doctors)
        state["beds"] = self.encode_beds(beds)
        state["resources"] = self.encode_resources(doctors, beds, operation_rooms)
        
        # Time and global features
        state["time"] = self.encode_time(step, max_steps)
        state["global"] = self.encode_global(patients, step)
        
        # Update time features with actual load ratio
        total_capacity = len(doctors) + len(beds)
        load_ratio = len(patients) / max(total_capacity, 1)
        state["time"][3] = load_ratio
        
        # Update metadata
        state["metadata"].update({
            "step": step,
            "episode_time": f"{int((step % 100) * 24 / 100):02d}:00",
            "utilization": load_ratio,
            "emergency_level": sum(1 for p in patients if p.get("severity", 0) >= 1)
        })
        
        return state
    
    def to_tensor(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert state to tensor-friendly format.
        
        Args:
            state: State dictionary
            
        Returns:
            Dictionary with tensor-ready arrays
        """
        return {
            "patients": state["patients"],
            "doctors": state["doctors"],
            "beds": state["beds"],
            "resources": state["resources"],
            "time": state["time"],
            "global": state["global"]
        }
    
    def flatten_state(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Flatten state into single vector for simple RL algorithms.
        
        Args:
            state: State dictionary
            
        Returns:
            Flattened state vector
        """
        tensor_state = self.to_tensor(state)
        
        # Concatenate all arrays
        flattened = np.concatenate([
            tensor_state["patients"].flatten(),
            tensor_state["doctors"].flatten(),
            tensor_state["beds"].flatten(),
            tensor_state["resources"].flatten(),
            tensor_state["time"],
            tensor_state["global"]
        ])
        
        return flattened.astype(np.float32)
    
    def get_state_dimension(self) -> int:
        """
        Get total dimension of flattened state.
        
        Returns:
            Total number of features in flattened state
        """
        empty_state = self.create_empty_state()
        return len(self.flatten_state(empty_state))
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate state structure and dimensions.
        
        Args:
            state: State dictionary to validate
            
        Returns:
            True if valid, raises AssertionError if invalid
        """
        required_keys = ["patients", "doctors", "beds", "resources", "time", "global", "metadata"]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"
        
        assert state["patients"].shape == (self.config.MAX_PATIENTS, self.config.PATIENT_FEATURES)
        assert state["doctors"].shape == (self.config.MAX_DOCTORS, self.config.DOCTOR_FEATURES)
        assert state["beds"].shape == (self.config.MAX_BEDS, self.config.BED_FEATURES)
        assert state["resources"].shape == (self.config.NUM_RESOURCES, self.config.RESOURCE_FEATURES)
        assert state["time"].shape == (self.config.TIME_FEATURES,)
        assert state["global"].shape == (self.config.GLOBAL_FEATURES,)
        
        return True


# Factory function for easy instantiation
def create_state_encoder(max_patients: int = 50, max_doctors: int = 20, 
                       max_beds: int = 20) -> StateEncoder:
    """
    Create state encoder with custom dimensions.
    
    Args:
        max_patients: Maximum number of patients to track
        max_doctors: Maximum number of doctors to track
        max_beds: Maximum number of beds to track
        
    Returns:
        Configured StateEncoder instance
    """
    config = StateConfig(
        MAX_PATIENTS=max_patients,
        MAX_DOCTORS=max_doctors,
        MAX_BEDS=max_beds
    )
    return StateEncoder(config)
