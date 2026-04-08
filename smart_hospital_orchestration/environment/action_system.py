"""
Action Logic System

Advanced action handling for Smart Hospital Resource Orchestration Environment.
Designed for deterministic, constraint-aware execution with RL compatibility.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class ActionType(IntEnum):
    """Discrete action enumeration for hospital operations."""
    WAIT = 0
    ALLOCATE_RESOURCE = 1
    ESCALATE_PRIORITY = 2
    DEFER = 3
    REASSIGN = 4


class PatientStatus(IntEnum):
    """Patient status for state transitions."""
    WAITING = 0
    ADMITTED = 1
    IN_TREATMENT = 2
    DISCHARGED = 3
    DEFERRED = 4


class PatientSeverity(IntEnum):
    """Patient severity levels for escalation logic."""
    NORMAL = 0
    EMERGENCY = 1
    CRITICAL = 2


@dataclass
class ActionResult:
    """Result of action execution with detailed metrics."""
    success: bool
    reward_contribution: float
    patients_affected: int
    resources_used: int
    message: str
    details: Dict[str, Any]


class ActionValidator:
    """Validates actions against system constraints and state."""
    
    def __init__(self, max_patients: int = 50, max_doctors: int = 20, max_beds: int = 20):
        """
        Initialize action validator.
        
        Args:
            max_patients: Maximum patients the system can handle
            max_doctors: Maximum doctors available
            max_beds: Maximum beds available
        """
        self.max_patients = max_patients
        self.max_doctors = max_doctors
        self.max_beds = max_beds
    
    def validate_action(self, action: int, state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate if action can be executed given current state.
        
        Args:
            action: Discrete action (0-4)
            state: Current environment state
            
        Returns:
            Tuple of (is_valid, reason_message)
        """
        if not (0 <= action <= 4):
            return False, f"Invalid action: {action}. Must be in [0, 4]"
        
        if action == ActionType.ALLOCATE_RESOURCE:
            return self._validate_allocation(state)
        elif action == ActionType.ESCALATE_PRIORITY:
            return self._validate_escalation(state)
        elif action == ActionType.DEFER:
            return self._validate_deferral(state)
        elif action == ActionType.REASSIGN:
            return self._validate_reassignment(state)
        
        # WAIT is always valid
        return True, "WAIT action always valid"
    
    def _validate_allocation(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate resource allocation feasibility."""
        patients = state["patients"]
        doctors = state["doctors"]
        beds = state["beds"]
        
        # Check for waiting patients
        waiting_mask = patients[:, 2] == PatientStatus.WAITING.value
        if not np.any(waiting_mask):
            return False, "No waiting patients to allocate"
        
        # Check resource availability
        available_doctors = np.sum(doctors[:, 1] == 1.0)  # available flag
        available_beds = np.sum(beds[:, 1] == 1.0)  # available flag
        
        if available_doctors == 0:
            return False, "No doctors available for allocation"
        
        if available_beds == 0:
            return False, "No beds available for allocation"
        
        return True, f"Can allocate: {available_doctors} doctors, {available_beds} beds available"
    
    def _validate_escalation(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate priority escalation feasibility."""
        patients = state["patients"]
        
        # Find patients waiting >3 timesteps who aren't critical
        waiting_mask = patients[:, 2] == PatientStatus.WAITING.value
        long_wait_mask = patients[:, 3] > 3  # wait_time > 3
        non_critical_mask = patients[:, 1] < PatientSeverity.CRITICAL.value
        
        eligible_mask = waiting_mask & long_wait_mask & non_critical_mask
        
        if not np.any(eligible_mask):
            return False, "No eligible patients for escalation (none waiting >3 steps or non-critical)"
        
        eligible_count = np.sum(eligible_mask)
        return True, f"Can escalate {eligible_count} patients"
    
    def _validate_deferral(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate patient deferral feasibility."""
        patients = state["patients"]
        
        # Check system load (patients / total capacity)
        total_patients = np.sum(patients[:, 0] > 0)  # non-zero patient IDs
        total_capacity = self.max_doctors + self.max_beds
        load_ratio = total_patients / max(total_capacity, 1)
        
        if load_ratio <= 0.8:
            return False, f"System not overloaded (load: {load_ratio:.2f} <= 0.8)"
        
        # Check for normal-priority waiting patients
        waiting_mask = patients[:, 2] == PatientStatus.WAITING.value
        normal_mask = patients[:, 1] == PatientSeverity.NORMAL.value
        
        eligible_mask = waiting_mask & normal_mask
        
        if not np.any(eligible_mask):
            return False, "No normal-priority waiting patients to defer"
        
        eligible_count = np.sum(eligible_mask)
        return True, f"Can defer {eligible_count} normal patients (load: {load_ratio:.2f})"
    
    def _validate_reassignment(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate doctor reassignment feasibility."""
        doctors = state["doctors"]
        
        # Check for workload imbalance
        loads = doctors[:, 2]  # current_load
        max_load = np.max(loads)
        min_load = np.min(loads)
        
        if max_load - min_load <= 1:
            return False, f"Workload balanced (max: {max_load}, min: {min_load})"
        
        # Check if any doctor has patients to reassign
        doctors_with_patients = doctors[doctors[:, 2] > 0]
        if len(doctors_with_patients) < 2:
            return False, "Need at least 2 doctors with patients for reassignment"
        
        return True, f"Can reassign (imbalance: {max_load - min_load} patients)"


class ActionHandler:
    """Handles execution of discrete actions with realistic hospital operations."""
    
    def __init__(self, validator: ActionValidator, stochastic: bool = False):
        """
        Initialize action handler.
        
        Args:
            validator: Action validator instance
            stochastic: Enable stochastic behavior for realism
        """
        self.validator = validator
        self.stochastic = stochastic
    
    def apply_action(self, action: int, state: Dict[str, Any]) -> ActionResult:
        """
        Apply action to environment state with validation and execution.
        
        Args:
            action: Discrete action (0-4)
            state: Current environment state
            
        Returns:
            ActionResult with execution details
        """
        # Validate action first
        is_valid, message = self.validator.validate_action(action, state)
        
        if not is_valid:
            return ActionResult(
                success=False,
                reward_contribution=-0.5,  # Penalty for invalid action
                patients_affected=0,
                resources_used=0,
                message=f"Invalid action: {message}",
                details={}
            )
        
        # Execute action based on type
        if action == ActionType.WAIT:
            return self._handle_wait(state)
        elif action == ActionType.ALLOCATE_RESOURCE:
            return self._handle_allocation(state)
        elif action == ActionType.ESCALATE_PRIORITY:
            return self._handle_escalation(state)
        elif action == ActionType.DEFER:
            return self._handle_deferral(state)
        elif action == ActionType.REASSIGN:
            return self._handle_reassignment(state)
        
        return ActionResult(
            success=False,
            reward_contribution=0.0,
            patients_affected=0,
            resources_used=0,
            message="Unknown action",
            details={}
        )
    
    def _handle_wait(self, state: Dict[str, Any]) -> ActionResult:
        """
        Handle WAIT action - no active intervention.
        
        Small penalty for inaction when patients are waiting to encourage proactivity.
        """
        patients = state["patients"]
        waiting_mask = patients[:, 2] == PatientStatus.WAITING.value
        waiting_count = np.sum(waiting_mask)
        
        penalty = -0.1 * waiting_count if waiting_count > 0 else 0.0
        
        return ActionResult(
            success=True,
            reward_contribution=penalty,
            patients_affected=0,
            resources_used=0,
            message=f"WAIT - {waiting_count} patients waiting",
            details={"waiting_patients": int(waiting_count)}
        )
    
    def _handle_allocation(self, state: Dict[str, Any]) -> ActionResult:
        """
        Handle ALLOCATE_RESOURCE action - assign doctors and beds to waiting patients.
        
        Prioritizes by severity (critical > emergency > normal) and wait time.
        Requires both doctor and bed availability for each allocation.
        """
        patients = state["patients"]
        doctors = state["doctors"]
        beds = state["beds"]
        
        # Find waiting patients sorted by priority
        waiting_mask = patients[:, 2] == PatientStatus.WAITING.value
        waiting_indices = np.where(waiting_mask)[0]
        
        if len(waiting_indices) == 0:
            return ActionResult(
                success=False,
                reward_contribution=-0.2,
                patients_affected=0,
                resources_used=0,
                message="No waiting patients to allocate",
                details={}
            )
        
        # Sort by severity (descending) then wait time (descending)
        waiting_patients = patients[waiting_indices]
        severity_priority = waiting_patients[:, 1]  # severity
        wait_priority = waiting_patients[:, 3]     # wait_time
        
        # Combined priority: severity * 10 + wait_time
        combined_priority = severity_priority * 10 + wait_priority
        sorted_indices = np.argsort(-combined_priority)  # descending
        prioritized_indices = waiting_indices[sorted_indices]
        
        # Get available resources
        available_doctor_mask = doctors[:, 1] == 1.0  # available flag
        available_bed_mask = beds[:, 1] == 1.0        # available flag
        
        available_doctor_indices = np.where(available_doctor_mask)[0]
        available_bed_indices = np.where(available_bed_mask)[0]
        
        allocations_made = 0
        total_reward = 0.0
        patients_admitted = []
        
        # Allocate to as many patients as possible
        for patient_idx in prioritized_indices:
            if len(available_doctor_indices) == 0 or len(available_bed_indices) == 0:
                break
            
            # Assign first available doctor and bed
            doctor_idx = available_doctor_indices[0]
            bed_idx = available_bed_indices[0]
            
            # Update patient state
            patients[patient_idx, 2] = PatientStatus.ADMITTED.value  # status
            patients[patient_idx, 5] = 1.0  # has_bed
            patients[patient_idx, 6] = 1.0  # has_doctor
            
            # Update doctor state
            doctors[doctor_idx, 1] = 0.0  # not available
            doctors[doctor_idx, 2] += 1  # current_load
            
            # Update bed state
            beds[bed_idx, 1] = 0.0  # not available
            beds[bed_idx, 2] = 1.0  # assigned
            
            # Calculate reward based on severity
            severity = int(patients[patient_idx, 1])
            wait_time = int(patients[patient_idx, 3])
            
            if severity == PatientSeverity.CRITICAL.value:
                reward = 10.0
            elif severity == PatientSeverity.EMERGENCY.value:
                reward = 5.0
            else:
                reward = 2.0
            
            # Penalty for long wait times
            reward -= 0.5 * min(wait_time, 10)
            
            total_reward += reward
            allocations_made += 1
            patients_admitted.append(int(patient_idx))
            
            # Remove used resources
            available_doctor_indices = available_doctor_indices[1:]
            available_bed_indices = available_bed_indices[1:]
        
        # Penalty for critical patients remaining unallocated
        critical_waiting = patients[
            (patients[:, 2] == PatientStatus.WAITING.value) & 
            (patients[:, 1] == PatientSeverity.CRITICAL.value)
        ]
        
        if len(critical_waiting) > 0 and (len(available_doctor_indices) == 0 or len(available_bed_indices) == 0):
            total_reward -= 5.0 * len(critical_waiting)
        
        return ActionResult(
            success=allocations_made > 0,
            reward_contribution=total_reward,
            patients_affected=allocations_made,
            resources_used=allocations_made * 2,  # doctor + bed per patient
            message=f"Allocated {allocations_made} patients",
            details={
                "patients_admitted": patients_admitted,
                "critical_remaining": len(critical_waiting),
                "resources_available": {
                    "doctors": len(available_doctor_indices),
                    "beds": len(available_bed_indices)
                }
            }
        )
    
    def _handle_escalation(self, state: Dict[str, Any]) -> ActionResult:
        """
        Handle ESCALATE_PRIORITY action - increase severity of long-waiting patients.
        
        Escalates patients who have waited >3 timesteps and aren't already critical.
        Small penalty for needing escalation (indicates system stress).
        """
        patients = state["patients"]
        
        # Find eligible patients for escalation
        waiting_mask = patients[:, 2] == PatientStatus.WAITING.value
        long_wait_mask = patients[:, 3] > 3  # wait_time > 3
        non_critical_mask = patients[:, 1] < PatientSeverity.CRITICAL.value
        
        eligible_mask = waiting_mask & long_wait_mask & non_critical_mask
        eligible_indices = np.where(eligible_mask)[0]
        
        if len(eligible_indices) == 0:
            return ActionResult(
                success=False,
                reward_contribution=-0.1,
                patients_affected=0,
                resources_used=0,
                message="No eligible patients for escalation",
                details={}
            )
        
        escalations_made = 0
        total_penalty = 0.0
        escalated_patients = []
        
        for patient_idx in eligible_indices:
            old_severity = int(patients[patient_idx, 1])
            
            # Escalate severity (but not beyond critical)
            new_severity = min(old_severity + 1, PatientSeverity.CRITICAL.value)
            patients[patient_idx, 1] = new_severity
            
            if new_severity != old_severity:
                escalations_made += 1
                total_penalty -= 0.5  # Penalty for needing escalation
                escalated_patients.append({
                    "patient_id": int(patients[patient_idx, 0]),
                    "old_severity": old_severity,
                    "new_severity": new_severity,
                    "wait_time": int(patients[patient_idx, 3])
                })
        
        return ActionResult(
            success=escalations_made > 0,
            reward_contribution=total_penalty,
            patients_affected=escalations_made,
            resources_used=0,
            message=f"Escalated {escalations_made} patients",
            details={
                "escalated_patients": escalated_patients,
                "eligible_count": len(eligible_indices)
            }
        )
    
    def _handle_deferral(self, state: Dict[str, Any]) -> ActionResult:
        """
        Handle DEFER action - defer normal-priority patients when system overloaded.
        
        Only defers normal patients when system load >80%.
        Significant penalty for deferring (negative patient outcome).
        """
        patients = state["patients"]
        
        # Calculate system load
        total_patients = np.sum(patients[:, 0] > 0)
        total_capacity = self.validator.max_doctors + self.validator.max_beds
        load_ratio = total_patients / max(total_capacity, 1)
        
        if load_ratio <= 0.8:
            return ActionResult(
                success=False,
                reward_contribution=-0.1,
                patients_affected=0,
                resources_used=0,
                message=f"System not overloaded (load: {load_ratio:.2f})",
                details={"load_ratio": load_ratio}
            )
        
        # Find normal-priority waiting patients
        waiting_mask = patients[:, 2] == PatientStatus.WAITING.value
        normal_mask = patients[:, 1] == PatientSeverity.NORMAL.value
        
        eligible_mask = waiting_mask & normal_mask
        eligible_indices = np.where(eligible_mask)[0]
        
        if len(eligible_indices) == 0:
            return ActionResult(
                success=False,
                reward_contribution=-0.1,
                patients_affected=0,
                resources_used=0,
                message="No normal-priority waiting patients to defer",
                details={"load_ratio": load_ratio}
            )
        
        # Defer all eligible normal patients
        deferrals_made = 0
        total_penalty = 0.0
        deferred_patients = []
        
        for patient_idx in eligible_indices:
            patients[patient_idx, 2] = PatientStatus.DEFERRED.value
            deferrals_made += 1
            total_penalty -= 1.0  # Significant penalty for deferring
            
            deferred_patients.append({
                "patient_id": int(patients[patient_idx, 0]),
                "wait_time": int(patients[patient_idx, 3])
            })
        
        return ActionResult(
            success=deferrals_made > 0,
            reward_contribution=total_penalty,
            patients_affected=deferrals_made,
            resources_used=0,
            message=f"Deferred {deferrals_made} normal patients (load: {load_ratio:.2f})",
            details={
                "deferred_patients": deferred_patients,
                "load_ratio": load_ratio
            }
        )
    
    def _handle_reassignment(self, state: Dict[str, Any]) -> ActionResult:
        """
        Handle REASSIGN action - balance doctor workloads by moving patients.
        
        Identifies workload imbalance and moves patients from most loaded to least loaded doctors.
        Small reward for load balancing.
        """
        doctors = state["doctors"]
        patients = state["patients"]
        
        # Find workload imbalance
        doctor_loads = doctors[:, 2]  # current_load
        max_load = np.max(doctor_loads)
        min_load = np.min(doctor_loads)
        
        if max_load - min_load <= 1:
            return ActionResult(
                success=False,
                reward_contribution=-0.1,
                patients_affected=0,
                resources_used=0,
                message=f"Workload balanced (max: {max_load}, min: {min_load})",
                details={"current_loads": doctor_loads.tolist()}
            )
        
        # Find most and least loaded doctors
        most_loaded_idx = int(np.argmax(doctor_loads))
        least_loaded_idx = int(np.argmin(doctor_loads))
        
        most_loaded_doctor = doctors[most_loaded_idx]
        least_loaded_doctor = doctors[least_loaded_idx]
        
        # Check if most loaded has patients to reassign
        if most_loaded_doctor[2] == 0:
            return ActionResult(
                success=False,
                reward_contribution=-0.1,
                patients_affected=0,
                resources_used=0,
                message="Most loaded doctor has no patients",
                details={}
            )
        
        # Find a patient to reassign (last assigned)
        patients_with_most_loaded = patients[
            (patients[:, 6] == 1.0) &  # has_doctor
            (patients[:, 2] == PatientStatus.ADMITTED.value)  # admitted
        ]
        
        # Find patients assigned to most loaded doctor
        # Note: In a real implementation, we'd track doctor-patient assignments more precisely
        # For now, we'll simulate by finding any admitted patient
        if len(patients_with_most_loaded) == 0:
            return ActionResult(
                success=False,
                reward_contribution=-0.1,
                patients_affected=0,
                resources_used=0,
                message="No patients found for reassignment",
                details={}
            )
        
        # Reassign first available patient
        patient_to_reassign_idx = np.where(patients[:, 6] == 1.0)[0][0]  # First patient with doctor
        
        # Update patient assignment
        patients[patient_to_reassign_idx, 6] = least_loaded_idx  # New doctor assignment
        
        # Update doctor loads
        doctors[most_loaded_idx, 2] -= 1  # Reduce load on most loaded
        doctors[least_loaded_idx, 2] += 1  # Increase load on least loaded
        
        # Update availability if needed
        if doctors[least_loaded_idx, 2] >= doctors[least_loaded_idx, 3]:  # at max capacity
            doctors[least_loaded_idx, 1] = 0.0  # not available
        
        if doctors[most_loaded_idx, 2] < doctors[most_loaded_idx, 3]:  # below max capacity
            doctors[most_loaded_idx, 1] = 1.0  # available
        
        return ActionResult(
            success=True,
            reward_contribution=0.5,  # Small reward for balancing
            patients_affected=1,
            resources_used=0,
            message=f"Reassigned patient from doctor {most_loaded_idx} to {least_loaded_idx}",
            details={
                "from_doctor": most_loaded_idx,
                "to_doctor": least_loaded_idx,
                "old_loads": [int(max_load), int(min_load)],
                "new_loads": [int(doctors[most_loaded_idx, 2]), int(doctors[least_loaded_idx, 2])],
                "patient_id": int(patients[patient_to_reassign_idx, 0])
            }
        )


class ActionSystem:
    """Main action system integrating validation and execution."""
    
    def __init__(self, max_patients: int = 50, max_doctors: int = 20, max_beds: int = 20, 
                 stochastic: bool = False):
        """
        Initialize complete action system.
        
        Args:
            max_patients: Maximum patients system can handle
            max_doctors: Maximum doctors available
            max_beds: Maximum beds available
            stochastic: Enable stochastic behavior
        """
        self.validator = ActionValidator(max_patients, max_doctors, max_beds)
        self.handler = ActionHandler(self.validator, stochastic)
    
    def execute_action(self, action: int, state: Dict[str, Any]) -> ActionResult:
        """
        Execute action with full validation and execution pipeline.
        
        Args:
            action: Discrete action (0-4)
            state: Current environment state
            
        Returns:
            ActionResult with execution details
        """
        return self.handler.apply_action(action, state)
    
    def get_valid_actions(self, state: Dict[str, Any]) -> List[int]:
        """
        Get list of valid actions for current state.
        
        Args:
            state: Current environment state
            
        Returns:
            List of valid action indices
        """
        valid_actions = []
        
        for action in range(5):  # 0-4
            is_valid, _ = self.validator.validate_action(action, state)
            if is_valid:
                valid_actions.append(action)
        
        return valid_actions


# Factory function for easy instantiation
def create_action_system(max_patients: int = 50, max_doctors: int = 20, 
                       max_beds: int = 20, stochastic: bool = False) -> ActionSystem:
    """
    Create action system with custom dimensions.
    
    Args:
        max_patients: Maximum patients system can handle
        max_doctors: Maximum doctors available
        max_beds: Maximum beds available
        stochastic: Enable stochastic behavior
        
    Returns:
        Configured ActionSystem instance
    """
    return ActionSystem(max_patients, max_doctors, max_beds, stochastic)
