"""
Reward Components Module

Individual reward component calculations.
"""

from typing import Dict, Any, List
import numpy as np


class RewardComponents:
    """
    Computes individual reward components.
    
    Each method computes a specific aspect of hospital performance
    that contributes to the overall reward signal.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize reward components.
        
        Args:
            config: Configuration for component parameters
        """
        self.config = config
        self.target_utilization = config.get("target_utilization", 0.85)
        self.max_wait_threshold = config.get("max_wait_threshold", 4.0)  # hours
    
    def compute_patient_outcome_reward(self, state: Dict[str, Any]) -> float:
        """
        Reward based on patient outcomes.
        
        Rewards successful treatments, penalizes complications.
        
        Args:
            state: Current environment state
            
        Returns:
            Patient outcome reward
        """
        readable = state.get("readable", {})
        admitted = float(readable.get("admitted", 0.0))
        red_admitted = float(readable.get("red_admitted", 0.0))
        green_admitted = float(readable.get("green_admitted", max(0.0, admitted - red_admitted)))

        # RED admissions count stronger than GREEN admissions.
        return (2.0 * red_admitted) + (1.0 * green_admitted)
    
    def compute_resource_utilization_reward(self, state: Dict[str, Any]) -> float:
        """
        Reward based on resource utilization efficiency.
        
        Rewards maintaining optimal utilization levels.
        
        Args:
            state: Current environment state
            
        Returns:
            Resource utilization reward
        """
        readable = state.get("readable", {})
        total_beds = float(readable.get("available_beds", 0.0) + readable.get("admitted", 0.0))
        total_doctors = float(readable.get("available_doctors", 0.0) + readable.get("admitted", 0.0))

        # Fallback to matrix-based estimate only if readable totals are unavailable.
        if total_beds <= 0.0 or total_doctors <= 0.0:
            beds = state.get("beds", np.array([]))
            doctors = state.get("doctors", np.array([]))

            bed_util = 0.0
            if len(beds) > 0:
                # Consider only initialized bed rows (max capacity is non-zero in column 3).
                valid_beds = beds[beds[:, 3] > 0.0] if beds.shape[1] > 3 else beds
                if len(valid_beds) > 0:
                    bed_util = float(np.mean(valid_beds[:, 1] == 0.0))

            doctor_util = 0.0
            if len(doctors) > 0:
                valid_doctors = doctors[doctors[:, 3] > 0.0] if doctors.shape[1] > 3 else doctors
                if len(valid_doctors) > 0:
                    doctor_util = float(np.mean(valid_doctors[:, 1] == 0.0))
        else:
            occupied_beds = max(total_beds - float(readable.get("available_beds", 0.0)), 0.0)
            busy_doctors = max(total_doctors - float(readable.get("available_doctors", 0.0)), 0.0)
            bed_util = occupied_beds / max(total_beds, 1.0)
            doctor_util = busy_doctors / max(total_doctors, 1.0)

        util = 0.5 * (bed_util + doctor_util)

        # Strictly positive reward in [0, 2], peaking near target utilization.
        return max(0.0, 2.0 - (2.0 * abs(util - self.target_utilization)))
    
    def compute_waiting_time_penalty(self, state: Dict[str, Any]) -> float:
        """
        Penalty for patient waiting times.
        
        Penalizes excessive waiting, especially for high-priority patients.
        
        Args:
            state: Current environment state
            
        Returns:
            Waiting time penalty (negative value)
        """
        patients = state.get("patients", np.array([]))
        if len(patients) == 0:
            return 0.0

        # patient array columns: [id, severity, status, wait_time, ...]
        waiting = patients[patients[:, 2] == 0.0]  # status WAITING
        if len(waiting) == 0:
            return 0.0

        severities = waiting[:, 1]
        waits = waiting[:, 3]

        # Higher severity => higher waiting cost. Return POSITIVE magnitude.
        severity_weight = np.where(severities == 2.0, 3.0, np.where(severities == 1.0, 1.8, 1.0))
        priority_weight = 1.0
        # Optional column 8 = priority (0 GREEN, 1 RED) in updated patient vector.
        if waiting.shape[1] > 8:
            priority_weight = np.where(waiting[:, 8] == 1.0, 3.0, 1.0)

        weighted_wait = float(np.mean(severity_weight * waits * priority_weight))

        # Smooth bounded magnitude.
        return float(min(weighted_wait / max(self.max_wait_threshold, 1.0), 10.0))
    
    def compute_workload_balance_reward(self, state: Dict[str, Any]) -> float:
        """
        Reward for balanced doctor workload distribution.
        
        Rewards equitable distribution of patients among doctors.
        
        Args:
            state: Current environment state
            
        Returns:
            Workload balance reward
        """
        doctors = state.get("doctors", np.array([]))
        if len(doctors) == 0:
            return 0.0

        # doctor array columns: [id, is_available, current_patients, max_patients]
        max_cap = np.maximum(doctors[:, 3], 1.0)
        load = doctors[:, 2] / max_cap
        variance = float(np.var(load))

        # Reward low variance, bounded [0,1].
        return max(0.0, 1.0 - min(variance * 4.0, 1.0))
    
    def compute_operational_cost_penalty(
        self,
        state: Dict[str, Any],
        action: np.ndarray
    ) -> float:
        """
        Penalty for operational costs.
        
        Penalizes unnecessary resource usage.
        
        Args:
            state: Current environment state
            action: Action taken
            
        Returns:
            Cost penalty (negative value)
        """
        action_id = int(action[0]) if np.ndim(action) > 0 else int(action)

        # Action-sensitive operational cost (POSITIVE magnitude).
        base = {
            0: 0.05,  # WAIT
            1: 0.18,  # ALLOCATE
            2: 0.10,  # ESCALATE
            3: 0.12,  # DEFER
            4: 0.15,  # REASSIGN
        }.get(action_id, 0.10)

        readable = state.get("readable", {})
        waiting = float(readable.get("waiting", 0.0))
        admitted = float(readable.get("admitted", 0.0))

        # Small congestion-dependent operating cost.
        return base + (0.01 * waiting) + (0.005 * admitted)
    
    def compute_emergency_handling_reward(self, state: Dict[str, Any]) -> float:
        """
        Reward for proper emergency handling.
        
        Rewards quick response to critical patients.
        
        Args:
            state: Current environment state
            
        Returns:
            Emergency handling reward
        """
        readable = state.get("readable", {})
        critical_waiting = float(readable.get("critical_waiting", 0.0))
        emergency_waiting = float(readable.get("emergency_waiting", 0.0))

        return max(0.0, 2.0 - (critical_waiting * 0.8 + emergency_waiting * 0.3))
