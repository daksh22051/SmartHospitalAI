"""
Hospital Environment Core

Main environment class implementing the OpenEnv-compatible interface
for hospital resource management simulation.
"""

from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from gymnasium import spaces

try:
    from reward.reward_function import RewardFunction
except ImportError:
    from smart_hospital_orchestration.reward.reward_function import RewardFunction


class PatientSeverity(Enum):
    """Patient severity levels."""
    NORMAL = 0
    EMERGENCY = 1
    CRITICAL = 2


class PatientPriority(Enum):
    """Triage priority levels for showcase logic."""
    GREEN = 0
    RED = 1


class PatientStatus(Enum):
    """Patient status in the system."""
    WAITING = 0
    ADMITTED = 1
    IN_TREATMENT = 2
    DISCHARGED = 3
    DEFERRED = 4


class ActionType(Enum):
    """Discrete action space for the agent."""
    WAIT = 0
    ALLOCATE_RESOURCE = 1
    ESCALATE_PRIORITY = 2
    DEFER = 3
    REASSIGN = 4


@dataclass
class Patient:
    """Represents a patient in the hospital system."""
    patient_id: int
    severity: PatientSeverity
    priority: PatientPriority = PatientPriority.GREEN
    status: PatientStatus = PatientStatus.WAITING
    wait_time: int = 0
    treatment_time: int = 0
    assigned_bed: Optional[int] = None
    assigned_doctor: Optional[int] = None
    was_escalated: bool = False
    
    def to_array(self) -> np.ndarray:
        """Convert patient to numpy array representation."""
        return np.array([
            self.patient_id,
            self.severity.value,
            self.status.value,
            self.wait_time,
            self.treatment_time,
            1 if self.assigned_bed is not None else 0,
            1 if self.assigned_doctor is not None else 0,
            1 if self.was_escalated else 0,
            self.priority.value,
        ], dtype=np.float32)


@dataclass
class Resource:
    """Base class for hospital resources."""
    resource_id: int
    is_available: bool = True
    assigned_patient: Optional[int] = None
    
    def release(self) -> None:
        """Release resource from current assignment."""
        self.is_available = True
        self.assigned_patient = None
    
    def allocate(self, patient_id: int) -> bool:
        """Allocate resource to a patient."""
        if self.is_available:
            self.is_available = False
            self.assigned_patient = patient_id
            return True
        return False


@dataclass
class Doctor(Resource):
    """Doctor resource with specialty and capacity."""
    specialty: str = "general"
    max_patients: int = 3
    current_patients: List[int] = field(default_factory=list)
    
    def can_accept(self) -> bool:
        """Check if doctor can accept another patient."""
        return len(self.current_patients) < self.max_patients
    
    def allocate(self, patient_id: int) -> bool:
        """Allocate patient to doctor."""
        if self.can_accept():
            self.current_patients.append(patient_id)
            self.is_available = len(self.current_patients) < self.max_patients
            self.assigned_patient = patient_id
            return True
        return False
    
    def release(self, patient_id: int) -> None:
        """Release specific patient from doctor."""
        if patient_id in self.current_patients:
            self.current_patients.remove(patient_id)
            self.is_available = True
            if not self.current_patients:
                self.assigned_patient = None


@dataclass
class ICUBed(Resource):
    """ICU bed resource with equipment."""
    equipment: List[str] = field(default_factory=list)
    # When True, this bed is forcibly unavailable due to a simulated crisis/maintenance lock.
    crisis_locked: bool = False
    
    def to_array(self) -> np.ndarray:
        """Convert bed to numpy array representation."""
        return np.array([
            self.resource_id,
            1 if self.is_available else 0,
            1 if self.assigned_patient is not None else 0,
            len(self.equipment)
        ], dtype=np.float32)


class HospitalEnv:
    """
    Hospital Resource Orchestration Environment.
    
    Implements OpenEnv-compatible interface for managing hospital resources
    including ICU beds, doctors, and patient admissions.
    
    Discrete Action Space:
        0 = WAIT: Do nothing, let time progress
        1 = ALLOCATE_RESOURCE: Assign bed and doctor to highest priority patient
        2 = ESCALATE_PRIORITY: Increase severity of waiting patients
        3 = DEFER: Mark non-critical patients for later treatment
        4 = REASSIGN: Move patient to different doctor/bed
    
    Attributes:
        task: Difficulty level ("easy", "medium", "hard")
        max_steps: Maximum timesteps per episode
        current_step: Current timestep
        patients: List of all patients
        doctors: List of available doctors
        beds: List of ICU beds
        rng: Random number generator for reproducibility
    """
    
    def __init__(self, task: str = "easy") -> None:
        """
        Initialize the hospital environment.
        
        Args:
            task: Difficulty level ("easy", "medium", "hard")
        """
        self.task = task.lower()
        self._load_task_config()
        
        # Environment state
        self.current_step: int = 0
        self.patients: List[Patient] = []
        self.doctors: List[Doctor] = []
        self.beds: List[ICUBed] = []
        self.rng: Optional[np.random.Generator] = None
        
        # Statistics for reward calculation
        self.total_admissions: int = 0
        self.total_deferrals: int = 0
        self.total_escalations: int = 0
        self.critical_wait_time: int = 0

        # Per-step diagnostics for reward shaping and API info
        self._last_action_name: str = "WAIT"
        self._last_allocations_made: int = 0
        self._last_critical_allocated: int = 0
        self._last_waiting_delta: int = 0
        self._last_base_reward: float = 0.0
        self._last_component_reward: float = 0.0
        self._last_city_transfer: bool = False
        self._last_city_transfer_target: Optional[str] = None
        self._last_city_crisis: bool = False

        # Multi-facility city network (mock nearby hospitals)
        self.city_hospitals: List[Dict[str, Any]] = []

        # Blood bank inventory is part of the backend state.
        self.blood_inventory: Dict[str, float] = {}

        # Crisis simulator flags (used by the web UI)
        self._crisis_active: bool = False
        self._crisis_message: Optional[str] = None

        # Initialize spaces and resources
        self._init_spaces()
        self._init_resources()
        self._init_city_network()
        self.reward_function = RewardFunction(self.config)

    def _init_city_network(self) -> None:
        """Initialize nearby hospitals for city load balancing."""
        self.city_hospitals = [
            {"name": "City General", "total_beds": 8, "available_beds": 4},
            {"name": "Sterling Care", "total_beds": 6, "available_beds": 3},
            {"name": "Apollo Metro", "total_beds": 10, "available_beds": 5},
        ]

    def _init_blood_inventory(self) -> None:
        """Initialize blood bank levels as backend-owned state."""
        blood_types = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
        if self.rng is not None:
            self.blood_inventory = {
                blood_type: float(self.rng.integers(62, 96))
                for blood_type in blood_types
            }
        else:
            self.blood_inventory = {blood_type: 80.0 for blood_type in blood_types}

    def _consume_blood_supply(self, patient: Optional[Patient]) -> None:
        """Reduce blood inventory when a patient is admitted."""
        if not self.blood_inventory or patient is None:
            return

        if patient.priority == PatientPriority.RED or patient.severity == PatientSeverity.CRITICAL:
            consumed_types = ["O-", "AB-", "A-"]
            drop = 2.5
        elif patient.severity == PatientSeverity.EMERGENCY:
            consumed_types = ["O+", "A+"]
            drop = 1.5
        else:
            consumed_types = ["A+", "B+"]
            drop = 1.0

        for blood_type in consumed_types:
            if blood_type in self.blood_inventory:
                self.blood_inventory[blood_type] = max(0.0, float(self.blood_inventory[blood_type]) - drop)

    def _update_city_network(self) -> None:
        """Simulate live bed availability drift in nearby hospitals."""
        if not self.rng:
            return

        for hospital in self.city_hospitals:
            drift = int(self.rng.integers(-1, 2))
            next_available = int(hospital["available_beds"]) + drift
            hospital["available_beds"] = max(0, min(int(hospital["total_beds"]), next_available))

    def _total_city_available_beds(self) -> int:
        """Total available beds across nearby hospitals."""
        return int(sum(int(h.get("available_beds", 0)) for h in self.city_hospitals))

    def _best_city_hospital(self) -> Optional[Dict[str, Any]]:
        """Return nearby hospital with highest availability."""
        candidates = [h for h in self.city_hospitals if int(h.get("available_beds", 0)) > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda h: int(h.get("available_beds", 0)))
    
    def _load_task_config(self) -> None:
        """Load configuration based on task difficulty."""
        configs = {
            "easy": {
                "max_steps": 50,
                "num_doctors": 3,
                "num_beds": 5,
                "initial_arrivals_min": 5,
                "initial_arrivals_max": 6,
                "max_patient_capacity": 45,
                "queue_soft_limit": 8,
                "arrival_rate": 0.25,
                "critical_prob": 0.08,
                "emergency_prob": 0.18,
                "max_arrivals": 2,
                "normal_to_emergency_wait": 5,
                "emergency_to_critical_wait": 8,
                "enable_dynamic_events": False,
                "resource_disruption_prob": 0.0,
                "emergency_event_prob": 0.0,
                "reward_blend_alpha": 0.30,
                "reward_weights": {
                    "patient_outcome": 2.2,
                    "resource_utilization": 0.6,
                    "waiting_time": -1.4,
                    "workload_balance": 0.35,
                    "operational_cost": -0.25,
                    "emergency_handling": 1.6,
                },
            },
            "medium": {
                "max_steps": 75,
                "num_doctors": 4,
                "num_beds": 6,
                "initial_arrivals_min": 8,
                "initial_arrivals_max": 11,
                "max_patient_capacity": 60,
                "queue_soft_limit": 12,
                "arrival_rate": 0.22,
                "critical_prob": 0.12,
                "emergency_prob": 0.22,
                "max_arrivals": 2,
                "normal_to_emergency_wait": 4,
                "emergency_to_critical_wait": 7,
                "enable_dynamic_events": True,
                "resource_disruption_prob": 0.04,
                "emergency_event_prob": 0.02,
                "reward_blend_alpha": 0.35,
                "reward_weights": {
                    "patient_outcome": 2.4,
                    "resource_utilization": 0.7,
                    "waiting_time": -1.7,
                    "workload_balance": 0.4,
                    "operational_cost": -0.3,
                    "emergency_handling": 1.9,
                },
            },
            "hard": {
                "max_steps": 100,
                "num_doctors": 5,
                "num_beds": 7,
                "initial_arrivals_min": 10,
                "initial_arrivals_max": 14,
                "max_patient_capacity": 80,
                "queue_soft_limit": 16,
                "arrival_rate": 0.30,
                "critical_prob": 0.18,
                "emergency_prob": 0.30,
                "max_arrivals": 2,
                "normal_to_emergency_wait": 3,
                "emergency_to_critical_wait": 6,
                "enable_dynamic_events": True,
                "resource_disruption_prob": 0.08,
                "emergency_event_prob": 0.04,
                "reward_blend_alpha": 0.40,
                "reward_weights": {
                    "patient_outcome": 2.6,
                    "resource_utilization": 0.8,
                    "waiting_time": -2.0,
                    "workload_balance": 0.45,
                    "operational_cost": -0.35,
                    "emergency_handling": 2.2,
                },
            }
        }
        
        if self.task not in configs:
            raise ValueError(f"Unknown task: {self.task}. Choose from {list(configs.keys())}")
        
        self.config = configs[self.task]
        self.max_steps = self.config["max_steps"]
    
    def _init_spaces(self) -> None:
        """Initialize observation and action spaces."""
        # Discrete action space: 5 actions (0-4)
        self._action_space = spaces.Discrete(5)
        
        # Observation space: structured but flattened for RL
        # Max 20 patients, 9 features each + 10 beds, 4 features each + 10 doctors, 4 features each + time
        obs_dim = (20 * 9) + (10 * 4) + (10 * 4) + 5
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    @property
    def action_space(self) -> spaces.Discrete:
        """Return action space specification."""
        return self._action_space
    
    @property
    def observation_space(self) -> spaces.Box:
        """Return observation space specification."""
        return self._observation_space
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (initial_state, info_dict)
        """
        # Initialize random number generator
        self.rng = np.random.default_rng(seed)
        
        # Reset time
        self.current_step = 0
        
        # Reset statistics
        self.total_admissions = 0
        self.total_deferrals = 0
        self.total_escalations = 0
        self.critical_wait_time = 0
        self._last_action_name = "WAIT"
        self._last_allocations_made = 0
        self._last_critical_allocated = 0
        self._last_waiting_delta = 0
        self._last_base_reward = 0.0
        self._last_component_reward = 0.0
        self._last_city_transfer = False
        self._last_city_transfer_target = None
        self._last_city_crisis = False

        # Reset crisis mode
        self._crisis_active = False
        self._crisis_message = None
        
        # Initialize resources
        self._init_resources()
        self._init_city_network()
        self._init_blood_inventory()
        
        # Initialize with some patients
        self.patients = []
        self._generate_arrivals(initial=True)
        
        # Get initial state
        state = self.state()
        info = {
            "step": self.current_step,
            "num_patients": len(self.patients),
            "available_doctors": sum(1 for d in self.doctors if d.is_available),
            "available_beds": sum(1 for b in self.beds if b.is_available)
        }
        
        return state, info
    
    def _init_resources(self) -> None:
        """Initialize doctors and beds."""
        # Create doctors
        self.doctors = [
            Doctor(
                resource_id=i,
                specialty="general",
                max_patients=1
            )
            for i in range(self.config["num_doctors"])
        ]
        
        # Create ICU beds with equipment
        equipment_lists = [
            ["monitor", "ventilator"],
            ["monitor", "infusion_pump"],
            ["monitor", "defibrillator"],
            ["monitor"]
        ]
        
        self.beds = [
            ICUBed(
                resource_id=i,
                equipment=equipment_lists[i % len(equipment_lists)]
            )
            for i in range(self.config["num_beds"])
        ]

    def trigger_resource_crisis(self, lock_ratio: float = 0.50, seed: Optional[int] = None) -> Dict[str, Any]:
        """Lock a fraction of currently available beds (simulate maintenance/crisis).

        The lock only applies to beds that are currently available and unassigned.
        Locked beds remain unavailable until environment reset (or manual unlock feature).

        Args:
            lock_ratio: fraction of *currently available* beds to lock
            seed: optional RNG seed for reproducible selection

        Returns:
            Dict with counts and locked bed ids.
        """

        rng = self.rng
        if rng is None:
            rng = np.random.default_rng(seed)
        elif seed is not None:
            rng = np.random.default_rng(seed)

        candidates = [
            b for b in self.beds
            if b.is_available and b.assigned_patient is None and not getattr(b, "crisis_locked", False)
        ]

        available_before = len(candidates)
        if available_before <= 0:
            self._crisis_active = True
            self._crisis_message = "CITY-WIDE ALERT: RESOURCE SHORTAGE DETECTED"
            return {
                "available_before": 0,
                "locked": 0,
                "locked_bed_ids": [],
            }

        lock_ratio = float(max(0.0, min(1.0, lock_ratio)))
        lock_n = int(np.ceil(available_before * lock_ratio))
        lock_n = max(1, min(available_before, lock_n))

        idx = rng.choice(np.arange(available_before), size=lock_n, replace=False)
        locked_ids: List[int] = []
        for i in idx.tolist():
            bed = candidates[int(i)]
            bed.crisis_locked = True
            bed.is_available = False
            bed.assigned_patient = None
            locked_ids.append(int(bed.resource_id))

        self._crisis_active = True
        self._crisis_message = "CITY-WIDE ALERT: RESOURCE SHORTAGE DETECTED"

        return {
            "available_before": int(available_before),
            "locked": int(lock_n),
            "locked_bed_ids": locked_ids,
        }
    
    def _generate_arrivals(self, initial: bool = False) -> None:
        """Generate new patient arrivals."""
        if initial:
            # Task-specific initial patient load:
            # easy ~5, medium ~8-10, hard ~10+
            low = int(self.config.get("initial_arrivals_min", 5))
            high = int(self.config.get("initial_arrivals_max", low + 1))
            num_arrivals = self.rng.integers(low, max(low + 1, high))
        else:
            # Probabilistic arrivals based on arrival rate
            arrival_prob = self.config["arrival_rate"]
            num_arrivals = self.rng.binomial(3, arrival_prob)
        
        for _ in range(num_arrivals):
            patient_id = len(self.patients)
            
            # Determine severity
            r = self.rng.random()
            if r < self.config["critical_prob"]:
                severity = PatientSeverity.CRITICAL
            elif r < self.config["critical_prob"] + self.config["emergency_prob"]:
                severity = PatientSeverity.EMERGENCY
            else:
                severity = PatientSeverity.NORMAL
            
            priority = PatientPriority.RED if self.rng.random() < 0.20 else PatientPriority.GREEN

            patient = Patient(
                patient_id=patient_id,
                severity=severity,
                priority=priority,
                status=PatientStatus.WAITING
            )
            self.patients.append(patient)
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one timestep of the environment with advanced modular pipeline.
        
        This is the core brain of the system that orchestrates:
        1. Action validation and application
        2. Constraint-aware state transitions
        3. Patient lifecycle updates
        4. Dynamic event simulation
        5. Reward computation and episode termination
        
        Args:
            action: Discrete action (0=wait, 1=allocate, 2=escalate, 3=defer, 4=reassign)
            
        Returns:
            Tuple of (state, reward, done, info) - Standard RL API
        """
        # ===== STAGE 1: Action Validation & Interpretation =====
        action_result = self._validate_and_interpret_action(action)

        # STRICT GUARD: invalid ALLOCATE due to no resources must not mutate state.
        if (
            action_result["action_type"] == ActionType.ALLOCATE_RESOURCE
            and not action_result["is_feasible"]
        ):
            self._last_action_name = ActionType.ALLOCATE_RESOURCE.name
            self._last_allocations_made = 0
            self._last_critical_allocated = 0
            self._last_waiting_delta = 0

            # Heavy negative penalty for invalid allocation attempt.
            total_reward = -50.0
            self._last_base_reward = float(total_reward)
            self._last_component_reward = 0.0

            # Keep timeline moving while preserving state counters.
            self.current_step += 1
            terminated, truncated = self._check_termination_conditions()
            done = terminated or truncated

            event_info = {
                "new_arrivals": 0,
                "resource_disruptions": 0,
                "emergency_events": 0,
            }
            current_state = self.state()
            info = self._assemble_info_dict(action_result, event_info, total_reward, terminated, truncated)
            info["invalid_allocation"] = True
            info["error"] = "No resources"
            info["action_message"] = "Action ALLOCATE_RESOURCE infeasible: no available doctor or bed"
            return current_state, total_reward, done, info
        
        # ===== STAGE 2: Action Application with Constraints =====
        action_reward = self._apply_action_with_constraints(action, action_result)
        
        # ===== STAGE 3: Patient State Updates =====
        self._update_patient_lifecycle()
        
        # ===== STAGE 4: Dynamic Event Simulation =====
        event_info = self._simulate_dynamic_events()
        
        # ===== STAGE 5: Resource State Updates =====
        self._update_resource_states()
        self._update_city_network()
        
        # ===== STAGE 6: Reward Computation =====
        total_reward = self._compute_step_reward(action_reward, event_info)
        
        # ===== STAGE 7: Time Progression =====
        self.current_step += 1
        
        # ===== STAGE 8: Termination Check =====
        terminated, truncated = self._check_termination_conditions()
        
        # Combine terminated and truncated into single done flag
        done = terminated or truncated
        
        # ===== STAGE 9: Result Assembly =====
        current_state = self.state()
        info = self._assemble_info_dict(action_result, event_info, total_reward, terminated, truncated)
        
        return current_state, total_reward, done, info
    
    def _validate_and_interpret_action(self, action: int) -> Dict[str, Any]:
        """
        Validate action and convert to internal representation.
        
        Args:
            action: Raw action integer (0-4)
            
        Returns:
            Dictionary with action metadata and validation result
        """
        # Validate action range
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be in [0, 4]")
        
        # Convert to enum for clarity
        action_type = ActionType(action)
        
        # Check resource constraints
        constraints = self._check_current_constraints()
        
        # Determine action feasibility
        is_feasible = self._check_action_feasibility(action_type, constraints)
        
        return {
            "action_type": action_type,
            "action_id": action,
            "is_feasible": is_feasible,
            "constraints": constraints,
            "validation_message": f"Action {action_type.name} {'feasible' if is_feasible else 'infeasible'}"
        }
    
    def _check_current_constraints(self) -> Dict[str, Any]:
        """
        Check current system constraints and resource availability.
        
        Returns:
            Dictionary with constraint information
        """
        # Count available resources
        available_doctors = sum(1 for d in self.doctors if d.is_available)
        available_beds = sum(1 for b in self.beds if b.is_available)
        
        # Count patients by status
        waiting_patients = sum(1 for p in self.patients if p.status == PatientStatus.WAITING)
        admitted_patients = sum(1 for p in self.patients if p.status == PatientStatus.ADMITTED)
        critical_waiting = sum(1 for p in self.patients 
                             if p.status == PatientStatus.WAITING and p.severity == PatientSeverity.CRITICAL)
        emergency_waiting = sum(1 for p in self.patients
                             if p.status == PatientStatus.WAITING and p.severity == PatientSeverity.EMERGENCY)
        normal_waiting = sum(1 for p in self.patients
                          if p.status == PatientStatus.WAITING and p.severity == PatientSeverity.NORMAL)

        deferred_patients = sum(1 for p in self.patients if p.status == PatientStatus.DEFERRED)
        
        # Calculate system load
        total_capacity = len(self.doctors) + len(self.beds)
        system_load = len(self.patients) / max(total_capacity, 1)
        
        return {
            "available_doctors": available_doctors,
            "available_beds": available_beds,
            "city_available_beds": self._total_city_available_beds(),
            "waiting_patients": waiting_patients,
            "deferred_patients": deferred_patients,
            "admitted_patients": admitted_patients,
            "critical_waiting": critical_waiting,
            "emergency_waiting": emergency_waiting,
            "normal_waiting": normal_waiting,
            "system_load": system_load,
            "overloaded": system_load > 0.8,
            "queue_overflow": waiting_patients > int(self.config.get("queue_soft_limit", 12))
        }
    
    def _check_action_feasibility(self, action_type: ActionType, constraints: Dict[str, Any]) -> bool:
        """
        Check if action is feasible given current constraints.
        
        Args:
            action_type: The action to check
            constraints: Current system constraints
            
        Returns:
            True if action is feasible
        """
        if action_type == ActionType.WAIT:
            return True  # WAIT always feasible
        
        elif action_type == ActionType.ALLOCATE_RESOURCE:
            return (constraints["waiting_patients"] > 0 and 
                   constraints["available_doctors"] > 0 and 
                   constraints["available_beds"] > 0)
        
        elif action_type == ActionType.ESCALATE_PRIORITY:
            # Check for patients waiting >3 steps who aren't critical
            eligible_patients = [p for p in self.patients 
                                if (p.status == PatientStatus.WAITING and 
                                    p.wait_time > 3 and 
                                    p.severity != PatientSeverity.CRITICAL)]
            return len(eligible_patients) > 0
        
        elif action_type == ActionType.DEFER:
            red_waiting = any(
                p.status == PatientStatus.WAITING and p.priority == PatientPriority.RED
                for p in self.patients
            )
            has_green_admitted = any(
                p.status == PatientStatus.ADMITTED and p.priority == PatientPriority.GREEN
                for p in self.patients
            )

            if constraints["available_beds"] <= 0 and red_waiting and has_green_admitted:
                return True

            return (constraints["overloaded"] and 
                   any(p.status == PatientStatus.WAITING and p.severity == PatientSeverity.NORMAL 
                       for p in self.patients))
        
        elif action_type == ActionType.REASSIGN:
            # City transfer path when local beds are full but nearby hospitals have space.
            if (
                constraints["available_beds"] <= 0
                and constraints["waiting_patients"] > 0
                and constraints["city_available_beds"] > 0
            ):
                return True

            # Check for workload imbalance
            doctor_loads = [len(d.current_patients) for d in self.doctors]
            return max(doctor_loads) - min(doctor_loads) > 1
        
        return False
    
    def _apply_action_with_constraints(self, action: int, action_result: Dict[str, Any]) -> float:
        """
        Apply action effects respecting system constraints.
        
        Args:
            action: Raw action integer
            action_result: Action validation result
            
        Returns:
            Immediate reward contribution from action
        """
        action_type = action_result["action_type"]

        # Reset per-step diagnostics
        self._last_action_name = action_type.name
        self._last_allocations_made = 0
        self._last_critical_allocated = 0
        self._last_waiting_delta = 0
        
        if not action_result["is_feasible"]:
            # Penalty for infeasible action
            return -0.5
        
        if action_type == ActionType.WAIT:
            return self._handle_wait_action()
        
        elif action_type == ActionType.ALLOCATE_RESOURCE:
            waiting_before = sum(1 for p in self.patients if p.status == PatientStatus.WAITING)
            return self._handle_allocation_action()
        
        elif action_type == ActionType.ESCALATE_PRIORITY:
            return self._handle_escalation_action()
        
        elif action_type == ActionType.DEFER:
            return self._handle_deferral_action()
        
        elif action_type == ActionType.REASSIGN:
            return self._handle_reassignment_action()
        
        return 0.0
    
    def _handle_wait_action(self) -> float:
        """
        Handle WAIT action - no active intervention.
        
        Returns:
            Penalty for inaction when patients are waiting
        """
        waiting_patients = [p for p in self.patients if p.status == PatientStatus.WAITING]
        if not waiting_patients:
            return 0.0

        critical_waiting = [p for p in waiting_patients if p.severity == PatientSeverity.CRITICAL]
        emergency_waiting = [p for p in waiting_patients if p.severity == PatientSeverity.EMERGENCY]

        # Priority-aware inaction penalty.
        penalty = 0.1 * len(waiting_patients)
        penalty += 1.5 * len(critical_waiting)
        penalty += 0.5 * len(emergency_waiting)
        return -penalty
    
    def _handle_allocation_action(self) -> float:
        """
        Handle ALLOCATE_RESOURCE action with constraint-aware allocation.
        
        Returns:
            Reward based on successful allocations and patient severity
        """
        # Get waiting patients sorted by triage + severity + wait:
        # RED first, then criticality, then longest wait.
        waiting_patients = [p for p in self.patients if p.status == PatientStatus.WAITING]
        waiting_patients.sort(
            key=lambda p: (p.priority == PatientPriority.RED, p.severity.value, p.wait_time),
            reverse=True,
        )
        
        # Get available resources
        available_doctors = [d for d in self.doctors if d.can_accept()]
        available_beds = [b for b in self.beds if b.is_available]

        if not waiting_patients:
            return -0.05

        if not available_doctors or not available_beds:
            critical_waiting = sum(1 for p in waiting_patients if p.severity == PatientSeverity.CRITICAL)
            emergency_waiting = sum(1 for p in waiting_patients if p.severity == PatientSeverity.EMERGENCY)
            return -(0.3 + 1.2 * critical_waiting + 0.4 * emergency_waiting)
        
        total_reward = 0.0
        allocations_made = 0
        critical_allocated = 0
        had_critical_waiting = any(p.severity == PatientSeverity.CRITICAL for p in waiting_patients)
        normal_allocated = 0

        # Allocate exactly ONE patient per ALLOCATE action.
        patient = waiting_patients[0]
        doctor = available_doctors[0]
        bed = available_beds[0]

        # Allocate resources
        if doctor.allocate(patient.patient_id) and bed.allocate(patient.patient_id):
            patient.status = PatientStatus.ADMITTED
            patient.assigned_doctor = doctor.resource_id
            patient.assigned_bed = bed.resource_id
            self._consume_blood_supply(patient)

            # Priority-aware reward shaping
            if patient.severity == PatientSeverity.CRITICAL:
                reward = 8.0
                critical_allocated += 1
                if patient.wait_time <= 2:
                    reward += 4.0  # Critical treated quickly
                elif patient.wait_time > 2:
                    reward -= min(1.5 * (patient.wait_time - 2), 6.0)
            elif patient.severity == PatientSeverity.EMERGENCY:
                reward = 4.0
                if patient.wait_time <= 3:
                    reward += 2.0
                elif patient.wait_time > 3:
                    reward -= min(0.8 * (patient.wait_time - 3), 4.0)
            else:
                reward = 1.5
                reward -= 0.2 * min(patient.wait_time, 8)
                normal_allocated += 1

            # Priority-based triage stakes: RED allocation gets 2x reward.
            if patient.priority == PatientPriority.RED:
                reward *= 2.0

            total_reward += reward
            allocations_made = 1
            self.total_admissions += 1

            # Remove used resources
            available_doctors.remove(doctor)
            available_beds.remove(bed)
        
        # Penalty for critical patients REMAINING unallocated (post-allocation state).
        remaining_critical_waiting = [
            p for p in self.patients
            if p.status == PatientStatus.WAITING and p.severity == PatientSeverity.CRITICAL
        ]
        if remaining_critical_waiting and (not available_doctors or not available_beds):
            total_reward -= 2.0 * len(remaining_critical_waiting)

        # Heavy penalty: critical ignored while normal were prioritized/processed.
        if had_critical_waiting and normal_allocated > 0 and any(
            p.status == PatientStatus.WAITING and p.severity == PatientSeverity.CRITICAL
            for p in self.patients
        ):
            total_reward -= 8.0

        # Save diagnostics for reward shaping + info dict.
        waiting_after = sum(1 for p in self.patients if p.status == PatientStatus.WAITING)
        self._last_allocations_made = allocations_made
        self._last_critical_allocated = critical_allocated
        self._last_waiting_delta = max(0, len(waiting_patients) - waiting_after)
        
        return total_reward
    
    def _handle_escalation_action(self) -> float:
        """
        Handle ESCALATE_PRIORITY action for long-waiting patients.
        
        Returns:
            Penalty for needing escalation
        """
        constraints = self._check_current_constraints()
        waiting_pressure = constraints["waiting_patients"] / max(len(self.beds), 1)
        threshold_shift = 1 if waiting_pressure > 1.0 else 0

        n2e_threshold = max(2, int(self.config.get("normal_to_emergency_wait", 4)) - 1 - threshold_shift)
        e2c_threshold = max(3, int(self.config.get("emergency_to_critical_wait", 7)) - 1 - threshold_shift)

        escalations_made = 0
        total_penalty = 0.0

        for patient in self.patients:
            if patient.status not in {PatientStatus.WAITING, PatientStatus.DEFERRED}:
                continue

            if patient.severity == PatientSeverity.NORMAL and patient.wait_time >= n2e_threshold:
                patient.severity = PatientSeverity.EMERGENCY
                patient.was_escalated = True
                if patient.status == PatientStatus.DEFERRED:
                    patient.status = PatientStatus.WAITING
                escalations_made += 1
                self.total_escalations += 1
                total_penalty -= 0.2

            elif patient.severity == PatientSeverity.EMERGENCY and patient.wait_time >= e2c_threshold:
                patient.severity = PatientSeverity.CRITICAL
                patient.was_escalated = True
                if patient.status == PatientStatus.DEFERRED:
                    patient.status = PatientStatus.WAITING
                escalations_made += 1
                self.total_escalations += 1
                total_penalty -= 0.4

        return total_penalty if escalations_made > 0 else -0.1
    
    def _handle_deferral_action(self) -> float:
        """
        Handle DEFER action for normal-priority patients when overloaded.
        
        Returns:
            Penalty for deferring patients
        """
        constraints = self._check_current_constraints()

        # Life-saving prioritization bonus:
        # If no beds are available and RED is waiting, defer a GREEN admitted patient.
        red_waiting = [
            p for p in self.patients
            if p.status == PatientStatus.WAITING and p.priority == PatientPriority.RED
        ]
        if constraints["available_beds"] <= 0 and red_waiting:
            green_admitted = [
                p for p in self.patients
                if p.status == PatientStatus.ADMITTED and p.priority == PatientPriority.GREEN
            ]
            if green_admitted:
                # Choose the least urgent GREEN admitted (shortest treatment progress).
                patient = sorted(green_admitted, key=lambda p: p.treatment_time)[0]
                self._release_patient_resources(patient)
                patient.status = PatientStatus.DEFERRED
                patient.assigned_bed = None
                patient.assigned_doctor = None
                self.total_deferrals += 1
                return 6.0
        
        if not constraints["overloaded"]:
            return -0.1  # Penalty for attempting defer when not overloaded
        
        deferrals_made = 0
        total_penalty = 0.0
        
        for patient in self.patients:
            if (patient.status == PatientStatus.WAITING and 
                patient.severity == PatientSeverity.NORMAL):
                
                patient.status = PatientStatus.DEFERRED
                deferrals_made += 1
                total_penalty -= 1.0  # Significant penalty for deferring
        
        return total_penalty
    
    def _handle_reassignment_action(self) -> float:
        """
        Handle REASSIGN action for city transfer (primary) or
        internal doctor workload balancing (fallback).
        
        Returns:
            Small reward for load balancing
        """
        constraints = self._check_current_constraints()

        # Reset city diagnostics
        self._last_city_transfer = False
        self._last_city_transfer_target = None
        self._last_city_crisis = False

        # Strategic city-level transfer when local beds are full.
        if constraints["available_beds"] <= 0 and constraints["waiting_patients"] > 0:
            transfer_target = self._best_city_hospital()
            waiting = [p for p in self.patients if p.status == PatientStatus.WAITING]
            waiting.sort(
                key=lambda p: (p.priority == PatientPriority.RED, p.severity.value, p.wait_time),
                reverse=True,
            )

            if transfer_target and waiting:
                patient = waiting[0]
                transfer_target["available_beds"] = max(0, int(transfer_target["available_beds"]) - 1)
                self.patients.remove(patient)

                self._last_city_transfer = True
                self._last_city_transfer_target = str(transfer_target["name"])
                self._last_waiting_delta = max(self._last_waiting_delta, 1)

                strategic_bonus = 9.0
                if patient.priority == PatientPriority.RED:
                    strategic_bonus += 2.0
                return strategic_bonus

            # City-wide healthcare crisis.
            self._last_city_crisis = True
            return -30.0

        doctor_loads = [(d, len(d.current_patients)) for d in self.doctors]
        doctor_loads.sort(key=lambda x: x[1], reverse=True)
        
        if len(doctor_loads) < 2:
            return -0.1  # Cannot reassign with <2 doctors
        
        most_loaded = doctor_loads[0]
        least_loaded = doctor_loads[-1]
        
        # Check if significant imbalance exists
        if most_loaded[1] - least_loaded[1] <= 1:
            return -0.1  # No significant imbalance
        
        # Move a patient from most loaded to least loaded
        doctor_from = most_loaded[0]
        doctor_to = least_loaded[0]
        
        if doctor_from.current_patients:
            patient_id = doctor_from.current_patients[-1]
            doctor_from.release(patient_id)
            
            # Find patient and update assignment
            for patient in self.patients:
                if patient.patient_id == patient_id:
                    patient.assigned_doctor = doctor_to.resource_id
                    doctor_to.allocate(patient_id)
                    return 0.5  # Small reward for balancing
        
        return -0.1
    
    def _update_patient_lifecycle(self) -> None:
        """
        Update patient states including wait times and treatment progression.
        """
        waiting_pool = [p for p in self.patients if p.status in {PatientStatus.WAITING, PatientStatus.DEFERRED}]
        waiting_pressure = len(waiting_pool) / max(len(self.beds), 1)
        threshold_shift = 1 if waiting_pressure > 1.0 else 0
        n2e_threshold = max(2, int(self.config.get("normal_to_emergency_wait", 4)) - threshold_shift)
        e2c_threshold = max(3, int(self.config.get("emergency_to_critical_wait", 7)) - threshold_shift)

        for patient in self.patients:
            if patient.status in {PatientStatus.WAITING, PatientStatus.DEFERRED}:
                patient.wait_time += 1

                # Dynamic priority escalation based on wait time + system pressure.
                if patient.severity == PatientSeverity.NORMAL and patient.wait_time >= n2e_threshold:
                    patient.severity = PatientSeverity.EMERGENCY
                    patient.was_escalated = True
                    self.total_escalations += 1

                elif patient.severity == PatientSeverity.EMERGENCY and patient.wait_time >= e2c_threshold:
                    patient.severity = PatientSeverity.CRITICAL
                    patient.was_escalated = True
                    self.total_escalations += 1
                    if patient.status == PatientStatus.DEFERRED:
                        patient.status = PatientStatus.WAITING

                # Track critical wait time for termination
                if patient.severity == PatientSeverity.CRITICAL and patient.status == PatientStatus.WAITING:
                    self.critical_wait_time += 1
            
            elif patient.status == PatientStatus.ADMITTED:
                patient.treatment_time += 1
                
                # Check for recovery and discharge
                if patient.treatment_time >= self._get_recovery_time(patient):
                    patient.status = PatientStatus.DISCHARGED
                    self._release_patient_resources(patient)
    
    def _get_recovery_time(self, patient: Patient) -> int:
        """
        Calculate recovery time based on patient severity.
        
        Args:
            patient: Patient to calculate recovery time for
            
        Returns:
            Recovery time in timesteps
        """
        base_time = 5
        if patient.severity == PatientSeverity.CRITICAL:
            return base_time + 5
        elif patient.severity == PatientSeverity.EMERGENCY:
            return base_time + 2
        return base_time
    
    def _release_patient_resources(self, patient: Patient) -> None:
        """
        Release resources when patient is discharged.
        
        Args:
            patient: Patient being discharged
        """
        # Release bed
        if patient.assigned_bed is not None:
            for bed in self.beds:
                if bed.resource_id == patient.assigned_bed:
                    bed.release()
                    break
        
        # Release doctor
        if patient.assigned_doctor is not None:
            for doctor in self.doctors:
                if doctor.resource_id == patient.assigned_doctor:
                    doctor.release(patient.patient_id)
                    break
    
    def _simulate_dynamic_events(self) -> Dict[str, Any]:
        """
        Simulate dynamic events like patient arrivals and resource disruptions.
        
        Returns:
            Dictionary with event information
        """
        events = {
            "new_arrivals": 0,
            "resource_disruptions": 0,
            "emergency_events": 0
        }
        
        # Generate new patient arrivals
        prev_patient_count = len(self.patients)
        self._generate_patient_arrivals()
        events["new_arrivals"] = len(self.patients) - prev_patient_count

        if not self.config.get("enable_dynamic_events", True):
            return events

        # Simulate task-specific resource disruptions
        if self.rng and self.rng.random() < self.config.get("resource_disruption_prob", 0.1):
            self._simulate_resource_disruption()
            events["resource_disruptions"] = 1

        # Simulate task-specific emergency events
        if self.rng and self.rng.random() < self.config.get("emergency_event_prob", 0.05):
            self._simulate_emergency_event()
            events["emergency_events"] = 1
        
        return events
    
    def _generate_patient_arrivals(self) -> None:
        """
        Generate new patient arrivals based on task configuration.
        """
        if not self.rng:
            return
        
        # Probabilistic arrivals based on arrival rate
        arrival_prob = self.config["arrival_rate"]
        max_arrivals_cfg = int(self.config.get("max_arrivals", 3))
        max_arrivals = self.rng.binomial(max_arrivals_cfg, arrival_prob)
        
        for _ in range(max_arrivals):
            if len(self.patients) >= int(self.config.get("max_patient_capacity", 80)):
                break

            patient_id = len(self.patients)
            
            # Determine severity
            r = self.rng.random()
            if r < self.config["critical_prob"]:
                severity = PatientSeverity.CRITICAL
            elif r < self.config["critical_prob"] + self.config["emergency_prob"]:
                severity = PatientSeverity.EMERGENCY
            else:
                severity = PatientSeverity.NORMAL
            
            priority = PatientPriority.RED if self.rng.random() < 0.20 else PatientPriority.GREEN

            patient = Patient(
                patient_id=patient_id,
                severity=severity,
                priority=priority,
                status=PatientStatus.WAITING
            )
            self.patients.append(patient)
    
    def _simulate_resource_disruption(self) -> None:
        """
        Simulate temporary resource disruption (e.g., doctor unavailable).
        """
        # Randomly make a doctor temporarily unavailable
        available_doctors = [d for d in self.doctors if d.is_available]
        if available_doctors and self.rng:
            disrupted_doctor = self.rng.choice(available_doctors)
            # In a real implementation, this would affect availability for several steps
            # For now, we just log the event
            pass
    
    def _simulate_emergency_event(self) -> None:
        """
        Simulate emergency event (e.g., mass casualty).
        """
        # Add multiple critical patients at once
        if self.rng:
            emergency_patients = self.rng.integers(2, 5)
            for _ in range(emergency_patients):
                patient_id = len(self.patients)
                patient = Patient(
                    patient_id=patient_id,
                    severity=PatientSeverity.CRITICAL,
                    priority=PatientPriority.RED,
                    status=PatientStatus.WAITING
                )
                self.patients.append(patient)
    
    def _update_resource_states(self) -> None:
        """
        Update resource states based on current assignments.
        """
        # Resources are updated during patient lifecycle and action application
        # This method can be used for additional resource state logic
        pass
    
    def _compute_step_reward(self, action_reward: float, event_info: Dict[str, Any]) -> float:
        """
        Compute total reward for the step.
        
        Args:
            action_reward: Reward from action execution
            event_info: Information about dynamic events
            
        Returns:
            Total step reward
        """
        reward = action_reward

        waiting = [p for p in self.patients if p.status == PatientStatus.WAITING]
        critical_waiting = [p for p in waiting if p.severity == PatientSeverity.CRITICAL]
        emergency_waiting = [p for p in waiting if p.severity == PatientSeverity.EMERGENCY]

        # Safety penalties: critical ignored => heavy, emergency delayed => moderate.
        # If this step successfully allocated critical patients, avoid over-penalizing
        # the same decision step.
        critical_penalty_scale = 1.0
        if self._last_action_name == "ALLOCATE_RESOURCE" and self._last_critical_allocated > 0:
            critical_penalty_scale = 0.5

        for patient in critical_waiting:
            if patient.wait_time > 2:
                reward -= critical_penalty_scale * min(2.5 * (patient.wait_time - 2), 8.0)

        for patient in emergency_waiting:
            if patient.wait_time > 3:
                reward -= min(1.0 * (patient.wait_time - 3), 3.0)

        # Priority-aware waiting penalty: RED waiting is 3x GREEN waiting.
        for patient in waiting:
            if patient.wait_time > 2:
                base_wait_penalty = min(0.6 * (patient.wait_time - 2), 4.0)
                multiplier = 3.0 if patient.priority == PatientPriority.RED else 1.0
                reward -= multiplier * base_wait_penalty

        # Throughput + efficiency bonus.
        admitted = [p for p in self.patients if p.status == PatientStatus.ADMITTED]
        reward += 0.08 * len(admitted)

        bed_utilization = sum(1 for b in self.beds if not b.is_available) / max(len(self.beds), 1)
        if 0.6 <= bed_utilization <= 0.95 and len(critical_waiting) == 0:
            reward += 0.5

        # Overflow handling.
        queue_soft_limit = int(self.config.get("queue_soft_limit", 12))
        if len(waiting) > queue_soft_limit:
            reward -= 0.2 * (len(waiting) - queue_soft_limit)

        # Event response bonus.
        if event_info["emergency_events"] > 0:
            critical_handled = sum(
                1 for p in self.patients
                if p.status == PatientStatus.ADMITTED and p.severity == PatientSeverity.CRITICAL
            )
            reward += 1.0 * critical_handled

        # Positive reinforcement when ALLOCATE truly improves throughput.
        if self._last_action_name == "ALLOCATE_RESOURCE" and self._last_allocations_made > 0:
            reward += 0.6 * self._last_allocations_made
            reward += 0.2 * self._last_waiting_delta

        base_reward = reward

        # Reward module integration (weighted components), blended with existing reward.
        action_id = ActionType[self._last_action_name].value if self._last_action_name in ActionType.__members__ else 0
        component_reward = self.reward_function.compute_step_reward(
            self.state(),
            {"action_id": action_id},
        )
        alpha = float(self.config.get("reward_blend_alpha", 0.35))
        blended_reward = ((1.0 - alpha) * base_reward) + (alpha * component_reward)

        self._last_base_reward = float(base_reward)
        self._last_component_reward = float(component_reward)

        return float(blended_reward)
    
    def _check_termination_conditions(self) -> Tuple[bool, bool]:
        """
        Check if episode should terminate.
        
        Returns:
            Tuple of (terminated, truncated)
        """
        # Terminate if too many critical patients have waited too long
        critical_timeout = sum(
            1 for p in self.patients 
            if p.status == PatientStatus.WAITING 
            and p.severity == PatientSeverity.CRITICAL 
            and p.wait_time > 10
        )
        
        terminated = critical_timeout >= 3
        truncated = self.current_step >= self.max_steps
        
        return terminated, truncated
    
    def _assemble_info_dict(self, action_result: Dict[str, Any], 
                          event_info: Dict[str, Any], 
                          total_reward: float,
                          terminated: bool,
                          truncated: bool) -> Dict[str, Any]:
        """
        Assemble information dictionary for the agent.
        
        Args:
            action_result: Result of action execution
            event_info: Information about dynamic events
            total_reward: Total reward for the step
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            
        Returns:
            Information dictionary
        """
        constraints = self._check_current_constraints()
        
        return {
            "step": self.current_step,
            "action": action_result["action_type"].name,
            "action_feasible": action_result["is_feasible"],
            "action_message": action_result["validation_message"],
            "reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_patients": len(self.patients),
            "waiting_patients": constraints["waiting_patients"],
            "admitted_patients": constraints["admitted_patients"],
            "critical_waiting": constraints["critical_waiting"],
            "emergency_waiting": constraints["emergency_waiting"],
            "normal_waiting": constraints["normal_waiting"],
            "available_doctors": constraints["available_doctors"],
            "available_beds": constraints["available_beds"],
            "city_available_beds": constraints["city_available_beds"],
            "system_load": constraints["system_load"],
            "queue_overflow": constraints["queue_overflow"],
            "events": event_info,
            "allocations_made": int(self._last_allocations_made),
            "critical_allocations_made": int(self._last_critical_allocated),
            "waiting_reduction": int(self._last_waiting_delta),
            "base_reward": float(self._last_base_reward),
            "component_reward": float(self._last_component_reward),
            "reward_blend_alpha": float(self.config.get("reward_blend_alpha", 0.35)),
            "escalations_made": action_result.get("escalations_made", 0),
            "deferrals_made": action_result.get("deferrals_made", 0),
            "reassignments_made": action_result.get("reassignments_made", 0),
            "city_transfer": bool(self._last_city_transfer),
            "city_transfer_target": self._last_city_transfer_target,
            "city_crisis": bool(self._last_city_crisis),
        }
    
    def _apply_action(self, action: ActionType, info: Dict[str, Any]) -> float:
        """
        Apply the agent's action to the environment.
        
        Args:
            action: The action to apply
            info: Dictionary to update with action metrics
            
        Returns:
            Immediate reward contribution from the action
        """
        reward = 0.0
        
        if action == ActionType.WAIT:
            # No action taken, small penalty for inaction when patients waiting
            waiting = [p for p in self.patients if p.status == PatientStatus.WAITING]
            if waiting:
                reward -= 0.1 * len(waiting)  # Small penalty for waiting
        
        elif action == ActionType.ALLOCATE_RESOURCE:
            # Allocate resources to highest priority waiting patients
            reward += self._allocate_resources(info)
        
        elif action == ActionType.ESCALATE_PRIORITY:
            # Escalate priority of waiting patients
            reward += self._escalate_patients(info)
        
        elif action == ActionType.DEFER:
            # Defer non-critical patients
            reward += self._defer_patients(info)
        
        elif action == ActionType.REASSIGN:
            # Reassign patients to balance load
            reward += self._reassign_patients(info)
        
        return reward
    
    def _allocate_resources(self, info: Dict[str, Any]) -> float:
        """Allocate doctors and beds to waiting patients."""
        reward = 0.0
        
        # Get waiting patients sorted by severity (highest first)
        waiting = [p for p in self.patients if p.status == PatientStatus.WAITING]
        waiting.sort(key=lambda p: p.severity.value, reverse=True)
        
        # Get available resources
        available_doctors = [d for d in self.doctors if d.can_accept()]
        available_beds = [b for b in self.beds if b.is_available]
        
        # Allocate to as many patients as possible
        for patient in waiting:
            if not available_doctors or not available_beds:
                break
            
            # Find best doctor (general priority, could be specialty-based)
            doctor = available_doctors[0]
            bed = available_beds[0]
            
            # Allocate
            if doctor.allocate(patient.patient_id) and bed.allocate(patient.patient_id):
                patient.status = PatientStatus.ADMITTED
                patient.assigned_doctor = doctor.resource_id
                patient.assigned_bed = bed.resource_id
                
                # Reward based on severity (higher reward for treating severe patients)
                if patient.severity == PatientSeverity.CRITICAL:
                    reward += 10.0
                elif patient.severity == PatientSeverity.EMERGENCY:
                    reward += 5.0
                else:
                    reward += 2.0
                
                # Penalty for long wait time
                reward -= 0.5 * min(patient.wait_time, 10)
                
                info["allocations"] += 1
                self.total_admissions += 1
                
                # Remove used resources from available lists
                available_doctors.remove(doctor)
                available_beds.remove(bed)
        
        # Penalty if critical patients remain unallocated
        critical_waiting = [p for p in waiting if p.severity == PatientSeverity.CRITICAL]
        if critical_waiting and (not available_doctors or not available_beds):
            reward -= 5.0 * len(critical_waiting)
        
        return reward
    
    def _escalate_patients(self, info: Dict[str, Any]) -> float:
        """Escalate priority of patients who have waited too long."""
        reward = 0.0
        
        for patient in self.patients:
            if patient.status == PatientStatus.WAITING:
                # Escalate if waited more than 3 timesteps or is already severe
                if patient.wait_time >= 3 and patient.severity != PatientSeverity.CRITICAL:
                    old_severity = patient.severity
                    patient.severity = PatientSeverity(min(patient.severity.value + 1, 2))
                    patient.was_escalated = True
                    
                    if old_severity != patient.severity:
                        info["escalations"] += 1
                        self.total_escalations += 1
                        # Small penalty for needing escalation
                        reward -= 0.5
        
        return reward
    
    def _defer_patients(self, info: Dict[str, Any]) -> float:
        """Defer non-critical patients to manage load."""
        reward = 0.0
        
        # Only defer normal patients when system is overloaded
        load_ratio = len([p for p in self.patients if p.status == PatientStatus.WAITING]) / max(len(self.beds), 1)
        
        if load_ratio > 0.8:  # High load threshold
            for patient in self.patients:
                if patient.status == PatientStatus.WAITING and patient.severity == PatientSeverity.NORMAL:
                    patient.status = PatientStatus.DEFERRED
                    info["deferrals"] += 1
                    self.total_deferrals += 1
                    reward -= 1.0  # Penalty for deferring
        
        return reward
    
    def _reassign_patients(self, info: Dict[str, Any]) -> float:
        """Reassign patients to balance doctor workloads."""
        reward = 0.0
        
        # Find overloaded and underloaded doctors
        doctor_loads = [(d, len(d.current_patients)) for d in self.doctors]
        doctor_loads.sort(key=lambda x: x[1], reverse=True)
        
        if len(doctor_loads) >= 2:
            most_loaded = doctor_loads[0]
            least_loaded = doctor_loads[-1]
            
            # If significant imbalance (>1 patient difference)
            if most_loaded[1] - least_loaded[1] > 1:
                # Move a patient from most loaded to least loaded
                doctor_from = most_loaded[0]
                doctor_to = least_loaded[0]
                
                if doctor_from.current_patients:
                    patient_id = doctor_from.current_patients[-1]
                    doctor_from.release(patient_id)
                    
                    # Find patient and update assignment
                    for patient in self.patients:
                        if patient.patient_id == patient_id:
                            patient.assigned_doctor = doctor_to.resource_id
                            doctor_to.allocate(patient_id)
                            info["reassignments"] += 1
                            reward += 0.5  # Small reward for balancing
                            break
        
        return reward
    
    def _update_state(self) -> None:
        """Update patient states and track time progression."""
        for patient in self.patients:
            if patient.status == PatientStatus.WAITING:
                patient.wait_time += 1
                if patient.severity == PatientSeverity.CRITICAL:
                    self.critical_wait_time += 1
            
            elif patient.status == PatientStatus.ADMITTED:
                patient.treatment_time += 1
                
                # Patients recover after some treatment time
                if patient.treatment_time >= self._get_recovery_time(patient):
                    patient.status = PatientStatus.DISCHARGED
                    # Release resources
                    if patient.assigned_bed is not None:
                        for bed in self.beds:
                            if bed.resource_id == patient.assigned_bed:
                                bed.release()
                                break
                    if patient.assigned_doctor is not None:
                        for doctor in self.doctors:
                            if doctor.resource_id == patient.assigned_doctor:
                                doctor.release(patient.patient_id)
                                break
    
    def _get_recovery_time(self, patient: Patient) -> int:
        """Calculate recovery time based on severity."""
        base_time = 5
        if patient.severity == PatientSeverity.CRITICAL:
            return base_time + 5
        elif patient.severity == PatientSeverity.EMERGENCY:
            return base_time + 2
        return base_time
    
    def _compute_reward(self, action_reward: float, info: Dict[str, Any]) -> float:
        """
        Compute total reward for the step.
        
        Args:
            action_reward: Immediate reward from action
            info: Step information dictionary
            
        Returns:
            Total reward value
        """
        reward = action_reward
        
        # Penalty for critical patients waiting too long
        critical_waiting = [
            p for p in self.patients 
            if p.status == PatientStatus.WAITING and p.severity == PatientSeverity.CRITICAL
        ]
        for patient in critical_waiting:
            if patient.wait_time > 2:
                reward -= 2.0 * (patient.wait_time - 2)
        
        # Small reward for each successful treatment in progress
        admitted = [p for p in self.patients if p.status == PatientStatus.ADMITTED]
        reward += 0.1 * len(admitted)
        
        # Penalty for resource underutilization
        utilization = sum(1 for b in self.beds if not b.is_available) / max(len(self.beds), 1)
        if utilization < 0.3 and len(self.patients) > 3:
            reward -= 0.5
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if too many critical patients have waited too long
        critical_timeout = sum(
            1 for p in self.patients 
            if p.status == PatientStatus.WAITING 
            and p.severity == PatientSeverity.CRITICAL 
            and p.wait_time > 10
        )
        
        if critical_timeout >= 3:
            return True
        
        return False

    def _compute_ai_suggestion(self, readable: Dict[str, Any]) -> Dict[str, Any]:
        """Compute a deterministic dashboard suggestion based on current readable state.

        This is intended to be the single source of truth for the UI "AI Suggestion Box"
        so the backend gym environment and frontend logic stay synced.

        Priority order (highest first):
          1) Emergency: waiting_red_patients > 0 AND beds_available > 0 -> URGENT ALLOCATE
          2) Normal:    waiting_patients > 0 AND beds_available > 0     -> ALLOCATE
          3) Crisis:    waiting_patients > 0 AND beds_available == 0    -> REASSIGN or DEFER
          4) Idle:      waiting_patients == 0                           -> HOLD
        """

        waiting_patients = int(readable.get("waiting", 0))
        waiting_red_patients = int(readable.get("red_waiting", 0))
        available_doctors = int(readable.get("available_doctors", 0))
        beds_available = int(readable.get("available_beds", 0))
        city_available_beds = int(readable.get("city_available_beds", 0))
        crisis_active = bool(readable.get("crisis_active", False))
        queue_soft_limit = int(readable.get("queue_soft_limit", self.config.get("queue_soft_limit", 12)))
        blood_inventory = readable.get("blood_inventory", {}) if isinstance(readable.get("blood_inventory"), dict) else {}
        blood_values = [float(v) for v in blood_inventory.values() if isinstance(v, (int, float))]
        lowest_blood = min(blood_values) if blood_values else 100.0

        crisis_reasons: List[str] = []
        if available_doctors == 0:
            crisis_reasons.append("doctors=0")
        if lowest_blood <= 0:
            crisis_reasons.append("blood supply=0")
        elif lowest_blood < 30:
            crisis_reasons.append(f"blood supply low ({lowest_blood:.0f}%)")
        if waiting_patients > queue_soft_limit:
            crisis_reasons.append(f"waiting={waiting_patients} above limit={queue_soft_limit}")
        if beds_available == 0:
            crisis_reasons.append("beds=0")

        if crisis_reasons:
            alert_reason = "Crisis triggered because " + " and ".join(crisis_reasons)
        else:
            alert_bits: List[str] = []
            if waiting_patients > 0:
                alert_bits.append(f"{waiting_patients} patient(s) waiting")
                if waiting_red_patients > 0:
                    alert_bits.append(f"{waiting_red_patients} RED priority waiting")
                if beds_available > 0:
                    alert_bits.append(f"{beds_available} beds available")
                else:
                    alert_bits.append("no local beds available")
                if available_doctors > 0:
                    alert_bits.append(f"{available_doctors} doctors on duty")
                else:
                    alert_bits.append("no doctors on duty")
                alert_reason = "Queue active; " + "; ".join(alert_bits)
            else:
                alert_reason = "No active queue pressure"

        # Priority 4 (Idle)
        if waiting_patients == 0:
            return {
                "ai_suggestion_action": "HOLD",
                "ai_suggestion_reason": "No patients waiting",
                "ai_suggestion_priority": 4,
                "alert_explanation": alert_reason,
            }

        # If there are no doctors, reassigning is not realistic; escalate/request help instead.
        if available_doctors == 0:
            if lowest_blood <= 0 or crisis_active:
                return {
                    "ai_suggestion_action": "ESCALATE",
                    "ai_suggestion_reason": "No doctors available and crisis conditions are active",
                    "ai_suggestion_priority": 1,
                    "alert_explanation": alert_reason,
                }
            return {
                "ai_suggestion_action": "REQUEST HELP",
                "ai_suggestion_reason": "No doctors available; request emergency staffing support",
                "ai_suggestion_priority": 1,
                "alert_explanation": alert_reason,
            }

        # Priority 1 (Emergency)
        if waiting_red_patients > 0 and beds_available > 0 and available_doctors > 0:
            return {
                "ai_suggestion_action": "URGENT ALLOCATE",
                "ai_suggestion_reason": "RED patients waiting and beds available",
                "ai_suggestion_priority": 1,
                "alert_explanation": alert_reason,
            }

        # Crisis override: if crisis is active, prefer local allocation first,
        # then city transfer only when local beds are exhausted.
        if crisis_active and waiting_patients > 0:
            if beds_available > 0 and available_doctors > 0:
                return {
                    "ai_suggestion_action": "URGENT ALLOCATE",
                    "ai_suggestion_reason": "Crisis active: treat waiting patients locally while beds are still available",
                    "ai_suggestion_priority": 1,
                    "alert_explanation": alert_reason,
                }
            if city_available_beds > 0 and available_doctors > 0:
                return {
                    "ai_suggestion_action": "REASSIGN",
                    "ai_suggestion_reason": "Crisis active: local beds exhausted; shift overflow to city network",
                    "ai_suggestion_priority": 2,
                    "alert_explanation": alert_reason,
                }
            return {
                "ai_suggestion_action": "DEFER",
                "ai_suggestion_reason": "Crisis active: no city capacity; defer non-critical patients",
                "ai_suggestion_priority": 1,
                "alert_explanation": alert_reason,
            }

        # Priority 2 (Normal)
        if waiting_patients > 0 and beds_available > 0 and available_doctors > 0:
            return {
                "ai_suggestion_action": "ALLOCATE",
                "ai_suggestion_reason": "Patients waiting and beds available",
                "ai_suggestion_priority": 2,
                "alert_explanation": alert_reason,
            }

        # Priority 3 (Resource Crisis)
        # Choose REASSIGN only if doctors exist and the city network can absorb overflow.
        if beds_available == 0:
            if city_available_beds > 0 and available_doctors > 0:
                return {
                    "ai_suggestion_action": "REASSIGN",
                    "ai_suggestion_reason": "Local beds full; city capacity available",
                    "ai_suggestion_priority": 3,
                    "alert_explanation": alert_reason,
                }
            if waiting_patients > queue_soft_limit or lowest_blood <= 0:
                return {
                    "ai_suggestion_action": "ESCALATE",
                    "ai_suggestion_reason": "Capacity exhausted or blood supply critical; escalate immediately",
                    "ai_suggestion_priority": 2,
                    "alert_explanation": alert_reason,
                }
            return {
                "ai_suggestion_action": "DEFER",
                "ai_suggestion_reason": "Local beds full; defer non-critical patients",
                "ai_suggestion_priority": 3,
                "alert_explanation": alert_reason,
            }

        return {
            "ai_suggestion_action": "HOLD",
            "ai_suggestion_reason": "No applicable suggestion rule matched",
            "ai_suggestion_priority": 4,
            "alert_explanation": alert_reason,
        }
    
    def state(self) -> Dict[str, Any]:
        """
        Get current state representation.
        
        Returns:
            Dictionary with structured state information
        """
        # Patient state matrix (max 20 patients, padded if fewer)
        max_patients = 20
        patient_array = np.zeros((max_patients, 9), dtype=np.float32)

        # IMPORTANT: expose the most decision-relevant patients first.
        # If we only take self.patients[:max_patients], older patients dominate
        # and newly waiting patients can become invisible to the agent.
        prioritized_patients = sorted(
            self.patients,
            key=lambda p: (
                p.status == PatientStatus.WAITING,  # waiting patients first
                p.priority == PatientPriority.RED,  # RED before GREEN
                p.severity.value,                   # critical > emergency > normal
                p.wait_time                         # longest waits first
            ),
            reverse=True
        )

        for i, patient in enumerate(prioritized_patients[:max_patients]):
            patient_array[i] = patient.to_array()
        
        # Resource states
        bed_array = np.zeros((10, 4), dtype=np.float32)
        for i, bed in enumerate(self.beds[:10]):
            bed_array[i] = bed.to_array()
        
        doctor_array = np.zeros((10, 4), dtype=np.float32)
        for i, doctor in enumerate(self.doctors[:10]):
            doctor_array[i] = np.array([
                doctor.resource_id,
                1 if doctor.is_available else 0,
                len(doctor.current_patients),
                doctor.max_patients
            ], dtype=np.float32)
        
        # Time and statistics
        normal_waiting = sum(
            1 for p in self.patients
            if p.status == PatientStatus.WAITING and p.severity == PatientSeverity.NORMAL
        )
        emergency_waiting = sum(
            1 for p in self.patients
            if p.status == PatientStatus.WAITING and p.severity == PatientSeverity.EMERGENCY
        )
        critical_waiting = sum(
            1 for p in self.patients
            if p.status == PatientStatus.WAITING and p.severity == PatientSeverity.CRITICAL
        )

        red_waiting = sum(
            1 for p in self.patients
            if p.status == PatientStatus.WAITING and p.priority == PatientPriority.RED
        )
        green_waiting = sum(
            1 for p in self.patients
            if p.status == PatientStatus.WAITING and p.priority == PatientPriority.GREEN
        )
        red_admitted = sum(
            1 for p in self.patients
            if p.status == PatientStatus.ADMITTED and p.priority == PatientPriority.RED
        )
        green_admitted = sum(
            1 for p in self.patients
            if p.status == PatientStatus.ADMITTED and p.priority == PatientPriority.GREEN
        )

        overdue_normal = sum(
            1 for p in self.patients
            if p.status == PatientStatus.WAITING and p.severity == PatientSeverity.NORMAL and p.wait_time >= int(self.config.get("normal_to_emergency_wait", 4))
        )
        overdue_emergency = sum(
            1 for p in self.patients
            if p.status == PatientStatus.WAITING and p.severity == PatientSeverity.EMERGENCY and p.wait_time >= int(self.config.get("emergency_to_critical_wait", 7))
        )

        time_stats = np.array([
            self.current_step,
            self.max_steps,
            len(self.patients),
            sum(1 for p in self.patients if p.status == PatientStatus.WAITING),
            sum(1 for p in self.patients if p.status == PatientStatus.ADMITTED)
        ], dtype=np.float32)
        
        # Flatten for RL agent
        flat_state = np.concatenate([
            patient_array.flatten(),
            bed_array.flatten(),
            doctor_array.flatten(),
            time_stats
        ])
        
        locked_beds = sum(1 for b in self.beds if getattr(b, "crisis_locked", False))

        base_state: Dict[str, Any] = {
            "patients": patient_array,
            "beds": bed_array,
            "doctors": doctor_array,
            "time": time_stats,
            "flat": flat_state,
            "metadata": {
                "severity_counts": {
                    "normal_waiting": normal_waiting,
                    "emergency_waiting": emergency_waiting,
                    "critical_waiting": critical_waiting,
                },
                "overdue_counts": {
                    "normal_overdue": overdue_normal,
                    "emergency_overdue": overdue_emergency,
                },
                "risk_load": float((1.0 * normal_waiting) + (2.0 * emergency_waiting) + (4.0 * critical_waiting)),
                "priority_counts": {
                    "red_waiting": red_waiting,
                    "green_waiting": green_waiting,
                    "red_admitted": red_admitted,
                    "green_admitted": green_admitted,
                },
                "city_network": [
                    {
                        "name": str(h["name"]),
                        "total_beds": int(h["total_beds"]),
                        "available_beds": int(h["available_beds"]),
                    }
                    for h in self.city_hospitals
                ],
            },
            "readable": {
                "step": self.current_step,
                "total_patients": len(self.patients),
                "waiting": sum(1 for p in self.patients if p.status == PatientStatus.WAITING),
                "admitted": sum(1 for p in self.patients if p.status == PatientStatus.ADMITTED),
                "available_doctors": sum(1 for d in self.doctors if d.is_available),
                "available_beds": sum(1 for b in self.beds if b.is_available),
                "locked_beds": int(locked_beds),
                "crisis_active": bool(self._crisis_active),
                "city_wide_alert": str(self._crisis_message) if self._crisis_active and self._crisis_message else None,
                "normal_waiting": normal_waiting,
                "emergency_waiting": emergency_waiting,
                "critical_waiting": critical_waiting,
                "red_waiting": red_waiting,
                "green_waiting": green_waiting,
                "red_admitted": red_admitted,
                "green_admitted": green_admitted,
                "overdue_normal": overdue_normal,
                "overdue_emergency": overdue_emergency,
                "city_available_beds": self._total_city_available_beds(),
                "queue_soft_limit": int(self.config.get("queue_soft_limit", 12)),
                "city_hospitals": [
                    {
                        "name": str(h["name"]),
                        "total_beds": int(h["total_beds"]),
                        "available_beds": int(h["available_beds"]),
                    }
                    for h in self.city_hospitals
                ],
                "blood_inventory": {
                    blood_type: float(max(0.0, min(100.0, level)))
                    for blood_type, level in self.blood_inventory.items()
                },
                "alert_explanation": None,
            }
        }

        # Inject deterministic suggestion so UI/backend stay in sync.
        try:
            readable = base_state.get("readable")
            if isinstance(readable, dict):
                suggestion = self._compute_ai_suggestion(readable)
                readable.update(suggestion)
                readable["alert_explanation"] = suggestion.get("alert_explanation")
        except Exception:
            # Suggestion must never break state() (RL/infra safety).
            pass

        return base_state
    
    def get_state_dimension(self) -> int:
        """Return the dimension of the flattened state vector."""
        return self._observation_space.shape[0]
    
    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render current state as text.
        
        Args:
            mode: Rendering mode ("human" for text)
            
        Returns:
            Text representation of state
        """
        s = self.state()["readable"]
        
        text = f"""
=== Hospital State (Step {s['step']}) ===
Patients: {s['total_patients']} total | {s['waiting']} waiting | {s['admitted']} admitted
Resources: {s['available_doctors']}/{len(self.doctors)} doctors | {s['available_beds']}/{len(self.beds)} beds
Critical: {s['critical_waiting']} waiting
        """.strip()
        
        if mode == "human":
            print(text)
        
        return text
    
    def close(self) -> None:
        """Clean up environment resources."""
        self.patients = []
        self.doctors = []
        self.beds = []
        self.rng = None
