"""
Advanced Reward System for Smart Hospital Resource Orchestration

Implements multi-objective reward design that balances patient outcomes,
resource efficiency, and operational priorities for reinforcement learning.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class PatientSeverity(Enum):
    """Patient severity levels for reward calculation."""
    NORMAL = 0
    EMERGENCY = 1
    CRITICAL = 2


class PatientStatus(Enum):
    """Patient status for reward calculation."""
    WAITING = 0
    ADMITTED = 1
    IN_TREATMENT = 2
    DISCHARGED = 3
    DEFERRED = 4


@dataclass
class RewardComponents:
    """Container for detailed reward breakdown."""
    emergency_handling: float = 0.0
    critical_prioritization: float = 0.0
    wait_time_reduction: float = 0.0
    resource_efficiency: float = 0.0
    delay_penalties: float = 0.0
    conflict_penalties: float = 0.0
    critical_ignored: float = 0.0
    step_efficiency: float = 0.0
    event_bonuses: float = 0.0
    
    def total(self) -> float:
        """Calculate total reward."""
        return (self.emergency_handling + self.critical_prioritization + 
                self.wait_time_reduction + self.resource_efficiency + 
                self.event_bonuses + self.delay_penalties + 
                self.conflict_penalties + self.critical_ignored + 
                self.step_efficiency)


@dataclass
class RewardConfig:
    """Configuration for reward system - OPTIMIZED VERSION."""
    
    # Positive rewards - BALANCED FOR REALISM
    allocation_reward: float = 10.0         # +10 for each successful allocation
    emergency_handling: float = 15.0        # +15 for handling emergency/critical
    wait_reduction: float = 4.0             # +4 for reducing wait times
    resource_efficiency: float = 5.0        # +5 for good resource usage
    system_improvement: float = 3.0         # +3 for overall system improvement
    
    # Penalties - CONTROLLED AND CAPPED
    step_penalty: float = 0.1               # Minimal step cost
    delay_penalty_base: float = 1.0         # Base delay penalty per patient
    critical_ignored: float = 10.0          # Strong but controlled
    resource_waste: float = 2.0             # Penalty for wasting resources
    
    # Reward clipping bounds
    reward_clip_min: float = -20.0          # Prevent extreme negative
    reward_clip_max: float = 20.0           # Prevent extreme positive
    
    # Episode target range
    target_episode_reward_min: float = -100.0
    target_episode_reward_max: float = 100.0
    
    # Thresholds
    critical_wait_threshold: int = 4        # Tolerance for critical patients
    emergency_wait_threshold: int = 6         # Tolerance for emergency patients
    max_delay_penalty_per_step: float = 5.0 # CAP: Max -5 per step from delays
    


class HospitalRewardCalculator:
    """
    Advanced reward calculator for hospital resource orchestration.
    
    Designs balanced, multi-objective rewards that encourage:
    - Fast emergency response
    - Critical patient prioritization
    - Efficient resource utilization
    - Minimized wait times
    - Avoidance of dangerous delays
    """
    
    def __init__(self, config: Optional[RewardConfig] = None, difficulty: str = "medium"):
        """
        Initialize reward calculator.
        
        Args:
            config: Reward configuration parameters
            difficulty: Task difficulty level ("easy", "medium", "hard")
        """
        self.config = config or RewardConfig()
        self.difficulty = difficulty
        self._apply_difficulty_scaling()
    
    def _apply_difficulty_scaling(self) -> None:
        """Apply difficulty-based scaling to reward parameters."""
        if self.difficulty == "easy":
            scale = self.config.easy_scaling
        elif self.difficulty == "hard":
            scale = self.config.hard_scaling
        else:
            scale = self.config.medium_scaling
        
        # Scale all reward parameters
        for attr in dir(self.config):
            if not attr.startswith('_') and isinstance(getattr(self.config, attr), float):
                current_value = getattr(self.config, attr)
                setattr(self.config, attr, current_value * scale)
    
    def compute_reward(self, state: Dict[str, Any], action: int, 
                      next_state: Dict[str, Any], events: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute comprehensive reward for hospital orchestration actions.
        
        Args:
            state: Environment state before action
            action: Action taken (0-4)
            next_state: Environment state after action
            events: Dynamic events that occurred during step
            
        Returns:
            Tuple of (total_reward, detailed_breakdown)
        """
        components = RewardComponents()
        
        # ===== POSITIVE REWARD COMPONENTS =====
        
        # 1. Emergency handling speed
        components.emergency_handling = self._calculate_emergency_handling_reward(
            state, next_state, action
        )
        
        # 2. Critical patient prioritization
        components.critical_prioritization = self._calculate_prioritization_reward(
            state, next_state, action
        )
        
        # 3. Wait time reduction
        components.wait_time_reduction = self._calculate_wait_time_reduction_reward(
            state, next_state
        )
        
        # 4. Resource utilization efficiency
        components.resource_efficiency = self._calculate_resource_efficiency_reward(
            state, next_state
        )
        
        # 5. Event handling bonuses
        components.event_bonuses = self._calculate_event_handling_reward(
            events, next_state
        )
        
        # ===== NEGATIVE REWARD COMPONENTS =====
        
        # 6. Delay penalties
        components.delay_penalties = self._calculate_delay_penalties(
            state, next_state
        )
        
        # 7. Conflict penalties
        components.conflict_penalties = self._calculate_conflict_penalties(
            state, next_state, action
        )
        
        # 8. Critical patient ignored penalties
        components.critical_ignored = self._calculate_critical_ignored_penalties(
            state, next_state
        )
        
        # 9. Step efficiency penalty
        components.step_efficiency = -self.config.step_penalty
        
        # ===== FINAL REWARD CALCULATION =====
        total_reward = components.total()
        
        # REWARD CLIPPING - Prevent extreme values
        # Clip to [-20, +20] per step as per requirements
        clipped_reward = max(self.config.reward_clip_min, 
                           min(self.config.reward_clip_max, total_reward))
        
        # Log if clipping occurred
        if clipped_reward != total_reward:
            clipped_amount = total_reward - clipped_reward
            # Add clipping info to breakdown
            
        # Create detailed breakdown for debugging
        breakdown = {
            "total_reward": clipped_reward,
            "unclipped_reward": total_reward,
            "clipping_applied": clipped_reward != total_reward,
            "components": {
                "emergency_handling": components.emergency_handling,
                "critical_prioritization": components.critical_prioritization,
                "wait_time_reduction": components.wait_time_reduction,
                "resource_efficiency": components.resource_efficiency,
                "delay_penalties": components.delay_penalties,
                "conflict_penalties": components.conflict_penalties,
                "critical_ignored": components.critical_ignored,
                "step_efficiency": components.step_efficiency,
                "event_bonuses": components.event_bonuses
            },
            "metrics": self._calculate_reward_metrics(state, next_state, events),
            "difficulty": self.difficulty
        }
        
        return clipped_reward, breakdown
    
    def _calculate_emergency_handling_reward(self, state: Dict[str, Any], 
                                          next_state: Dict[str, Any], 
                                          action: int) -> float:
        """
        Calculate reward for handling emergency/critical patients - OPTIMIZED.
        
        Rewards:
        - +15.0 for emergency/critical patients handled (using config.emergency_handling)
        """
        reward = 0.0
        
        # Find newly admitted critical/emergency patients
        prev_waiting = self._get_waiting_patients_by_severity(state)
        next_waiting = self._get_waiting_patients_by_severity(next_state)
        
        # Check for critical/emergency patients that were admitted
        for severity in [PatientSeverity.CRITICAL, PatientSeverity.EMERGENCY]:
            prev_count = prev_waiting.get(severity, 0)
            next_count = next_waiting.get(severity, 0)
            
            if next_count < prev_count:  # Patients were admitted
                admitted_count = prev_count - next_count
                # Use new config value: +15 for each emergency/critical handled
                reward += admitted_count * self.config.emergency_handling
        
        return reward
    
    def _calculate_prioritization_reward(self, state: Dict[str, Any], 
                                      next_state: Dict[str, Any], 
                                      action: int) -> float:
        """
        Calculate reward for proper patient prioritization - OPTIMIZED.
        
        Rewards:
        - +10.0 for any patient allocation (using config.allocation_reward)
        """
        reward = 0.0
        
        # Get admitted patients by severity
        prev_admitted = self._count_admitted_patients(state)
        next_admitted = self._count_admitted_patients(next_state)
        
        # Reward for each new admission (+10 per allocation)
        if next_admitted > prev_admitted:
            new_admissions = next_admitted - prev_admitted
            reward += new_admissions * self.config.allocation_reward
        
        return reward
    
    def _calculate_wait_time_reduction_reward(self, state: Dict[str, Any], 
                                           next_state: Dict[str, Any]) -> float:
        """
        Calculate reward for reducing patient wait times - OPTIMIZED.
        
        Rewards:
        - +4.0 per patient with reduced wait time (using config.wait_reduction)
        - Additional bonus for significant wait time reduction
        """
        reward = 0.0
        
        # Compare wait times between states
        prev_wait_times = self._extract_wait_times(state)
        next_wait_times = self._extract_wait_times(next_state)
        
        for patient_id, prev_wait in prev_wait_times.items():
            if patient_id in next_wait_times:
                next_wait = next_wait_times[patient_id]
                
                # Reward for wait time reduction
                if next_wait < prev_wait:
                    reduction = prev_wait - next_wait
                    reward += reduction * self.config.wait_reduction_bonus
                    
                    # Bonus for significant reduction
                    if reduction >= 3:
                        reward += self.config.wait_reduction_bonus
        
        return reward
    
    def _calculate_resource_efficiency_reward(self, state: Dict[str, Any], 
                                           next_state: Dict[str, Any]) -> float:
        """
        Calculate reward for efficient resource utilization.
        
        Rewards:
        - +0.3 per efficiently used resource pair
        - Bonus for high utilization without overloading
        """
        reward = 0.0
        
        # Calculate resource utilization
        total_resources = self._get_total_resources(next_state)
        used_resources = self._get_used_resources(next_state)
        
        if total_resources > 0:
            utilization = used_resources / total_resources
            
            # Reward for efficient utilization
            if utilization >= self.config.resource_efficiency_threshold:
                efficiency_score = min(utilization, 1.0)
                reward += efficiency_score * self.config.resource_efficiency_bonus * 10
                
                # Bonus for optimal utilization
                if 0.8 <= utilization <= 0.95:
                    reward += self.config.resource_efficiency_bonus * 5
        
        return reward
    
    def _calculate_event_handling_reward(self, events: Dict[str, Any], 
                                       next_state: Dict[str, Any]) -> float:
        """
        Calculate reward for handling dynamic events well.
        
        Rewards:
        - +2.0 per emergency event handled effectively
        - +1.0 per resource disruption managed
        """
        reward = 0.0
        
        # Emergency event handling
        if events.get("emergency_events", 0) > 0:
            # Check if critical patients were admitted after emergency
            critical_admitted = self._count_critical_admitted(next_state)
            if critical_admitted > 0:
                reward += critical_admitted * self.config.event_handling_bonus
        
        # Resource disruption handling
        if events.get("resource_disruptions", 0) > 0:
            # Check if system maintained reasonable performance
            waiting_patients = self._count_waiting_patients(next_state)
            if waiting_patients < 10:  # Reasonable performance despite disruption
                reward += self.config.event_handling_bonus
        
        return reward
    
    def _calculate_delay_penalties(self, state: Dict[str, Any], 
                                 next_state: Dict[str, Any]) -> float:
        """
        Calculate penalties for patient delays - OPTIMIZED VERSION.
        
        Controlled penalties:
        - -1.0 per critical patient waiting >4 steps
        - -0.5 per emergency patient waiting >6 steps
        - MAX -5.0 total per step (prevent collapse)
        """
        penalty = 0.0
        
        waiting_patients = self._get_waiting_patients_with_details(next_state)
        
        for patient in waiting_patients:
            wait_time = patient.get("wait_time", 0)
            severity = patient.get("severity", 0)
            
            if severity == PatientSeverity.CRITICAL.value:
                if wait_time > self.config.critical_wait_threshold:
                    # Small penalty per step over threshold
                    excess = wait_time - self.config.critical_wait_threshold
                    penalty -= self.config.delay_penalty_base * excess  # -1.0 per step
                    
            elif severity == PatientSeverity.EMERGENCY.value:
                if wait_time > self.config.emergency_wait_threshold:
                    # Half penalty for emergency
                    excess = wait_time - self.config.emergency_wait_threshold
                    penalty -= self.config.delay_penalty_base * 0.5 * excess  # -0.5 per step
        
        # HARD CAP at -5.0 per step (critical for preventing collapse)
        if penalty < -self.config.max_delay_penalty_per_step:
            penalty = -self.config.max_delay_penalty_per_step
        
        return penalty
    
    def _calculate_conflict_penalties(self, state: Dict[str, Any], 
                                   next_state: Dict[str, Any], 
                                   action: int) -> float:
        """
        Calculate penalties for resource conflicts and bad decisions.
        
        Penalties:
        - -1.0 for resource allocation conflicts
        - -0.5 for inefficient resource usage
        """
        penalty = 0.0
        
        # Check for resource conflicts
        conflicts = self._detect_resource_conflicts(next_state)
        penalty -= conflicts * self.config.resource_conflict_penalty
        
        # Check for inefficient resource usage
        inefficiency = self._detect_resource_inefficiency(state, next_state, action)
        penalty -= inefficiency * self.config.resource_conflict_penalty * 0.5
        
        return penalty
    
    def _calculate_critical_ignored_penalties(self, state: Dict[str, Any], 
                                          next_state: Dict[str, Any]) -> float:
        """
        Calculate penalties for ignoring critical patients - OPTIMIZED.
        
        Penalties:
        - -10.0 per critical patient ignored while treating normal patients
        - But capped by overall reward clipping to [-20, +20]
        """
        penalty = 0.0
        
        # Check if critical patients are waiting while normal patients are treated
        critical_waiting = self._count_waiting_by_severity(next_state, PatientSeverity.CRITICAL)
        normal_admitted = self._count_admitted_by_severity(next_state, PatientSeverity.NORMAL)
        
        if critical_waiting > 0 and normal_admitted > 0:
            # Strong but controlled penalty per critical patient ignored
            penalty -= critical_waiting * self.config.critical_ignored
        
        return penalty
    
    def _calculate_reward_metrics(self, state: Dict[str, Any], 
                                 next_state: Dict[str, Any], 
                                 events: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional metrics for reward analysis."""
        return {
            "total_patients": self._count_total_patients(next_state),
            "waiting_patients": self._count_waiting_patients(next_state),
            "admitted_patients": self._count_admitted_patients(next_state),
            "critical_waiting": self._count_waiting_by_severity(next_state, PatientSeverity.CRITICAL),
            "emergency_waiting": self._count_waiting_by_severity(next_state, PatientSeverity.EMERGENCY),
            "resource_utilization": self._calculate_resource_utilization(next_state),
            "avg_wait_time": self._calculate_average_wait_time(next_state),
            "events_occurred": sum(events.values()),
            "system_load": self._calculate_system_load(next_state)
        }
    
    # ===== HELPER METHODS =====
    
    def _get_waiting_patients_by_severity(self, state: Dict[str, Any]) -> Dict[PatientSeverity, int]:
        """Get count of waiting patients by severity."""
        waiting_counts = {}
        
        if "patients" in state:
            patients = state["patients"]
            waiting_mask = patients[:, 2] == PatientStatus.WAITING.value
            
            for severity in PatientSeverity:
                severity_mask = patients[:, 1] == severity.value
                waiting_counts[severity] = np.sum(waiting_mask & severity_mask)
        
        return waiting_counts
    
    def _get_waiting_patients_with_details(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get detailed information about waiting patients."""
        waiting_patients = []
        
        if "patients" in state:
            patients = state["patients"]
            waiting_mask = patients[:, 2] == PatientStatus.WAITING.value
            
            for patient in patients[waiting_mask]:
                waiting_patients.append({
                    "id": int(patient[0]),
                    "severity": int(patient[1]),
                    "status": int(patient[2]),
                    "wait_time": int(patient[3])
                })
        
        return waiting_patients
    
    def _extract_wait_times(self, state: Dict[str, Any]) -> Dict[int, int]:
        """Extract wait times for all patients."""
        wait_times = {}
        
        if "patients" in state:
            patients = state["patients"]
            for patient in patients:
                if patient[0] > 0:  # Valid patient ID
                    wait_times[int(patient[0])] = int(patient[3])
        
        return wait_times
    
    def _were_patients_admitted_fast(self, state: Dict[str, Any], next_state: Dict[str, Any], 
                                   severity: PatientSeverity, threshold: int) -> bool:
        """Check if patients were admitted within fast threshold."""
        # This is a simplified check - in practice, you'd track individual patient times
        waiting_patients = self._get_waiting_patients_with_details(state)
        
        for patient in waiting_patients:
            if patient["severity"] == severity.value and patient["wait_time"] <= threshold:
                return True
        
        return False
    
    def _get_total_resources(self, state: Dict[str, Any]) -> int:
        """Get total number of resources."""
        total = 0
        
        if "doctors" in state:
            total += np.sum(state["doctors"][:, 0] > 0)  # Valid doctors
        
        if "beds" in state:
            total += np.sum(state["beds"][:, 0] > 0)  # Valid beds
        
        return total
    
    def _get_used_resources(self, state: Dict[str, Any]) -> int:
        """Get number of used resources."""
        used = 0
        
        if "doctors" in state:
            used += np.sum(state["doctors"][:, 1] == 0)  # Unavailable doctors
        
        if "beds" in state:
            used += np.sum(state["beds"][:, 1] == 0)  # Unavailable beds
        
        return used
    
    def _count_critical_admitted(self, state: Dict[str, Any]) -> int:
        """Count critical patients that were admitted."""
        if "patients" not in state:
            return 0
        
        patients = state["patients"]
        critical_mask = patients[:, 1] == PatientSeverity.CRITICAL.value
        admitted_mask = patients[:, 2] == PatientStatus.ADMITTED.value
        
        return np.sum(critical_mask & admitted_mask)
    
    def _count_waiting_patients(self, state: Dict[str, Any]) -> int:
        """Count total waiting patients."""
        if "patients" not in state:
            return 0
        
        return np.sum(state["patients"][:, 2] == PatientStatus.WAITING.value)
    
    def _count_admitted_patients(self, state: Dict[str, Any]) -> int:
        """Count total admitted patients."""
        if "patients" not in state:
            return 0
        
        return np.sum(state["patients"][:, 2] == PatientStatus.ADMITTED.value)
    
    def _count_waiting_by_severity(self, state: Dict[str, Any], severity: PatientSeverity) -> int:
        """Count waiting patients by severity."""
        if "patients" not in state:
            return 0
        
        patients = state["patients"]
        waiting_mask = patients[:, 2] == PatientStatus.WAITING.value
        severity_mask = patients[:, 1] == severity.value
        
        return np.sum(waiting_mask & severity_mask)
    
    def _count_admitted_by_severity(self, state: Dict[str, Any], severity: PatientSeverity) -> int:
        """Count admitted patients by severity."""
        if "patients" not in state:
            return 0
        
        patients = state["patients"]
        admitted_mask = patients[:, 2] == PatientStatus.ADMITTED.value
        severity_mask = patients[:, 1] == severity.value
        
        return np.sum(admitted_mask & severity_mask)
    
    def _count_total_patients(self, state: Dict[str, Any]) -> int:
        """Count total patients in system."""
        if "patients" not in state:
            return 0
        
        return np.sum(state["patients"][:, 0] > 0)
    
    def _detect_resource_conflicts(self, state: Dict[str, Any]) -> int:
        """Detect resource allocation conflicts."""
        conflicts = 0
        
        # Check for double allocations (simplified)
        if "patients" in state:
            patients = state["patients"]
            # Patients admitted without proper resources
            admitted_mask = patients[:, 2] == PatientStatus.ADMITTED.value
            no_bed_mask = patients[:, 5] == 0  # No bed assigned
            no_doctor_mask = patients[:, 6] == 0  # No doctor assigned
            
            conflicts += np.sum(admitted_mask & (no_bed_mask | no_doctor_mask))
        
        return conflicts
    
    def _detect_resource_inefficiency(self, state: Dict[str, Any], 
                                    next_state: Dict[str, Any], 
                                    action: int) -> int:
        """Detect inefficient resource usage."""
        inefficiency = 0
        
        # Check for action that doesn't utilize available resources
        if action == 0:  # WAIT action
            available_doctors = self._count_available_resources(next_state, "doctors")
            available_beds = self._count_available_resources(next_state, "beds")
            waiting_patients = self._count_waiting_patients(next_state)
            
            if waiting_patients > 0 and available_doctors > 0 and available_beds > 0:
                inefficiency += 1
        
        return inefficiency
    
    def _count_available_resources(self, state: Dict[str, Any], resource_type: str) -> int:
        """Count available resources of a specific type."""
        if resource_type not in state:
            return 0
        
        resources = state[resource_type]
        return np.sum(resources[:, 1] == 1)  # Available flag
    
    def _calculate_resource_utilization(self, state: Dict[str, Any]) -> float:
        """Calculate overall resource utilization."""
        total = self._get_total_resources(state)
        used = self._get_used_resources(state)
        
        return used / max(total, 1)
    
    def _calculate_average_wait_time(self, state: Dict[str, Any]) -> float:
        """Calculate average wait time for waiting patients."""
        if "patients" not in state:
            return 0.0
        
        patients = state["patients"]
        waiting_mask = patients[:, 2] == PatientStatus.WAITING.value
        
        if np.sum(waiting_mask) == 0:
            return 0.0
        
        wait_times = patients[waiting_mask, 3]
        return np.mean(wait_times)
    
    def _calculate_system_load(self, state: Dict[str, Any]) -> float:
        """Calculate system load ratio."""
        total_patients = self._count_total_patients(state)
        total_resources = self._get_total_resources(state)
        
        return total_patients / max(total_resources, 1)


# Factory function for easy instantiation
def create_reward_calculator(difficulty: str = "medium", 
                           config: Optional[RewardConfig] = None) -> HospitalRewardCalculator:
    """
    Create a reward calculator with specified difficulty and configuration.
    
    Args:
        difficulty: Task difficulty level ("easy", "medium", "hard")
        config: Optional custom configuration
        
    Returns:
        Configured reward calculator
    """
    return HospitalRewardCalculator(config, difficulty)


# Convenience function for direct reward calculation
def compute_reward(state: Dict[str, Any], action: int, 
                  next_state: Dict[str, Any], events: Dict[str, Any],
                  difficulty: str = "medium") -> Tuple[float, Dict[str, Any]]:
    """
    Compute reward for hospital orchestration action.
    
    Args:
        state: Environment state before action
        action: Action taken (0-4)
        next_state: Environment state after action
        events: Dynamic events that occurred during step
        difficulty: Task difficulty level
        
    Returns:
        Tuple of (total_reward, detailed_breakdown)
    """
    calculator = create_reward_calculator(difficulty)
    return calculator.compute_reward(state, action, next_state, events)
