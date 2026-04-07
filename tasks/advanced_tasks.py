"""
Advanced Task Configuration System for Smart Hospital Resource Orchestration

Implements progressive difficulty scenarios with realistic hospital operational challenges.
Designed to develop robust RL agents through structured curriculum learning.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class TaskDifficulty(Enum):
    """Task difficulty levels for progressive learning."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PatientSeverity(Enum):
    """Patient severity levels for task configuration."""
    NORMAL = 0
    EMERGENCY = 1
    CRITICAL = 2


@dataclass
class PatientDistribution:
    """Configuration for patient severity distribution."""
    normal_count: int = 0
    emergency_count: int = 0
    critical_count: int = 0
    
    def total(self) -> int:
        """Get total patient count."""
        return self.normal_count + self.emergency_count + self.critical_count
    
    def to_list(self) -> List[int]:
        """Convert to list of severity values."""
        severity_list = []
        severity_list.extend([PatientSeverity.NORMAL.value] * self.normal_count)
        severity_list.extend([PatientSeverity.EMERGENCY.value] * self.emergency_count)
        severity_list.extend([PatientSeverity.CRITICAL.value] * self.critical_count)
        return severity_list


@dataclass
class ResourceConfiguration:
    """Configuration for hospital resources."""
    doctors: int = 3
    beds: int = 5
    doctor_capacity: int = 3  # Max patients per doctor
    bed_equipment: List[str] = field(default_factory=lambda: ["monitor", "infusion_pump"])
    
    def total_capacity(self) -> int:
        """Get total patient capacity."""
        return self.doctors + self.beds


@dataclass
class EventConfiguration:
    """Configuration for dynamic events."""
    enabled: bool = False
    arrival_rate: float = 0.2
    emergency_prob: float = 0.1
    critical_prob: float = 0.05
    disruption_prob: float = 0.05
    emergency_event_prob: float = 0.02
    
    def validate(self) -> None:
        """Validate event probabilities."""
        total_prob = self.emergency_prob + self.critical_prob
        if total_prob > 1.0:
            raise ValueError(f"Patient severity probabilities sum to {total_prob}, must be ≤ 1.0")


@dataclass
class TaskConfiguration:
    """Complete task configuration for hospital environment."""
    name: str
    difficulty: TaskDifficulty
    patients: PatientDistribution
    resources: ResourceConfiguration
    events: EventConfiguration
    max_steps: int = 100
    seed: Optional[int] = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for environment initialization."""
        return {
            "name": self.name,
            "difficulty": self.difficulty.value,
            "environment": {
                "max_patients": self.patients.total() + 20,  # Extra capacity for arrivals
                "max_doctors": self.resources.doctors + 5,
                "max_beds": self.resources.beds + 5
            },
            "initial_patients": self.patients.to_list(),
            "resources": {
                "doctors": self.resources.doctors,
                "beds": self.resources.beds,
                "doctor_capacity": self.resources.doctor_capacity,
                "bed_equipment": self.resources.bed_equipment
            },
            "events": {
                "enabled": self.events.enabled,
                "arrival_rate": self.events.arrival_rate,
                "critical_prob": self.events.critical_prob,
                "emergency_prob": self.events.emergency_prob,
                "disruption_prob": self.events.disruption_prob,
                "emergency_event_prob": self.events.emergency_event_prob
            },
            "max_steps": self.max_steps,
            "seed": self.seed,
            "description": self.description
        }


class TaskFactory:
    """
    Factory class for creating hospital orchestration tasks.
    
    Generates progressively difficult scenarios that test different aspects
    of resource management and decision-making under pressure.
    """
    
    @staticmethod
    def easy(seed: Optional[int] = None) -> TaskConfiguration:
        """
        Create easy difficulty task configuration.
        
        Characteristics:
        - Small patient load (~5 patients)
        - Minimal emergency cases
        - Sufficient resources
        - Deterministic behavior
        - Focus on basic resource allocation
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            Easy task configuration
        """
        patients = PatientDistribution(
            normal_count=4,
            emergency_count=1,
            critical_count=0
        )
        
        resources = ResourceConfiguration(
            doctors=3,
            beds=5,
            doctor_capacity=3,
            bed_equipment=["monitor"]
        )
        
        events = EventConfiguration(
            enabled=False,  # No dynamic events for easy mode
            arrival_rate=0.1,
            emergency_prob=0.1,
            critical_prob=0.0,
            disruption_prob=0.0,
            emergency_event_prob=0.0
        )
        
        return TaskConfiguration(
            name="Easy Hospital Management",
            difficulty=TaskDifficulty.EASY,
            patients=patients,
            resources=resources,
            events=events,
            max_steps=50,
            seed=seed,
            description="Basic resource allocation with minimal pressure. Focus on learning fundamental operations."
        )
    
    @staticmethod
    def medium(seed: Optional[int] = None) -> TaskConfiguration:
        """
        Create medium difficulty task configuration.
        
        Characteristics:
        - Moderate patient load (~8-10 patients)
        - Emergency cases included
        - Limited resources
        - Some dynamic events
        - Time pressure introduced
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            Medium task configuration
        """
        patients = PatientDistribution(
            normal_count=5,
            emergency_count=3,
            critical_count=1
        )
        
        resources = ResourceConfiguration(
            doctors=4,
            beds=6,
            doctor_capacity=3,
            bed_equipment=["monitor", "infusion_pump"]
        )
        
        events = EventConfiguration(
            enabled=True,
            arrival_rate=0.2,
            emergency_prob=0.15,
            critical_prob=0.1,
            disruption_prob=0.05,
            emergency_event_prob=0.02
        )
        
        return TaskConfiguration(
            name="Medium Hospital Management",
            difficulty=TaskDifficulty.MEDIUM,
            patients=patients,
            resources=resources,
            events=events,
            max_steps=75,
            seed=seed,
            description="Balanced scenario with emergency cases and resource constraints. Introduces time pressure and dynamic events."
        )
    
    @staticmethod
    def hard(seed: Optional[int] = None) -> TaskConfiguration:
        """
        Create hard difficulty task configuration.
        
        Characteristics:
        - High patient load (10+ patients)
        - Critical and emergency-heavy
        - Severe resource constraints
        - Dynamic events enabled
        - Stochastic behavior
        - Crisis management focus
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            Hard task configuration
        """
        patients = PatientDistribution(
            normal_count=4,
            emergency_count=4,
            critical_count=3
        )
        
        resources = ResourceConfiguration(
            doctors=5,
            beds=7,
            doctor_capacity=4,  # Higher capacity due to load
            bed_equipment=["monitor", "infusion_pump", "ventilator"]
        )
        
        events = EventConfiguration(
            enabled=True,
            arrival_rate=0.3,
            emergency_prob=0.2,
            critical_prob=0.15,
            disruption_prob=0.1,
            emergency_event_prob=0.05
        )
        
        return TaskConfiguration(
            name="Hard Hospital Management",
            difficulty=TaskDifficulty.HARD,
            patients=patients,
            resources=resources,
            events=events,
            max_steps=100,
            seed=seed,
            description="Crisis scenario with severe resource constraints and high patient load. Tests advanced decision-making under extreme pressure."
        )
    
    @staticmethod
    def custom(patients: PatientDistribution,
              resources: ResourceConfiguration,
              events: EventConfiguration,
              max_steps: int = 100,
              seed: Optional[int] = None,
              name: str = "Custom Task") -> TaskConfiguration:
        """
        Create custom task configuration.
        
        Args:
            patients: Patient distribution configuration
            resources: Resource configuration
            events: Event configuration
            max_steps: Maximum episode length
            seed: Optional random seed
            name: Task name
            
        Returns:
            Custom task configuration
        """
        events.validate()
        
        return TaskConfiguration(
            name=name,
            difficulty=TaskDifficulty.MEDIUM,  # Default to medium
            patients=patients,
            resources=resources,
            events=events,
            max_steps=max_steps,
            seed=seed,
            description="Custom task configuration"
        )


class TaskCurriculum:
    """
    Curriculum learning system for progressive skill development.
    
    Provides structured learning path from basic to advanced scenarios.
    """
    
    @staticmethod
    def beginner_curriculum() -> List[TaskConfiguration]:
        """
        Generate beginner curriculum focusing on basic skills.
        
        Returns:
            List of progressively harder easy tasks
        """
        curriculum = []
        
        # Stage 1: Pure normal patients
        curriculum.append(TaskConfiguration(
            name="Beginner Stage 1: Normal Patients Only",
            difficulty=TaskDifficulty.EASY,
            patients=PatientDistribution(normal_count=3, emergency_count=0, critical_count=0),
            resources=ResourceConfiguration(doctors=2, beds=3),
            events=EventConfiguration(enabled=False),
            max_steps=30,
            description="Learn basic resource allocation with normal patients only."
        ))
        
        # Stage 2: Introduce emergencies
        curriculum.append(TaskConfiguration(
            name="Beginner Stage 2: First Emergencies",
            difficulty=TaskDifficulty.EASY,
            patients=PatientDistribution(normal_count=3, emergency_count=1, critical_count=0),
            resources=ResourceConfiguration(doctors=2, beds=3),
            events=EventConfiguration(enabled=False),
            max_steps=40,
            description="Practice prioritizing emergency patients."
        ))
        
        # Stage 3: Resource constraints
        curriculum.append(TaskConfiguration(
            name="Beginner Stage 3: Limited Resources",
            difficulty=TaskDifficulty.EASY,
            patients=PatientDistribution(normal_count=4, emergency_count=1, critical_count=0),
            resources=ResourceConfiguration(doctors=2, beds=2),  # Tighter constraints
            events=EventConfiguration(enabled=False),
            max_steps=50,
            description="Learn efficient resource utilization under constraints."
        ))
        
        return curriculum
    
    @staticmethod
    def advanced_curriculum() -> List[TaskConfiguration]:
        """
        Generate advanced curriculum for skill refinement.
        
        Returns:
            List of challenging scenarios
        """
        curriculum = []
        
        # Stage 1: High patient volume
        curriculum.append(TaskConfiguration(
            name="Advanced Stage 1: High Volume",
            difficulty=TaskDifficulty.HARD,
            patients=PatientDistribution(normal_count=8, emergency_count=3, critical_count=2),
            resources=ResourceConfiguration(doctors=4, beds=6),
            events=EventConfiguration(enabled=True, arrival_rate=0.2),
            max_steps=80,
            description="Manage high patient volume with dynamic arrivals."
        ))
        
        # Stage 2: Resource scarcity
        curriculum.append(TaskConfiguration(
            name="Advanced Stage 2: Resource Scarcity",
            difficulty=TaskDifficulty.HARD,
            patients=PatientDistribution(normal_count=3, emergency_count=3, critical_count=2),
            resources=ResourceConfiguration(doctors=3, beds=4),  # Severe constraints
            events=EventConfiguration(enabled=True, arrival_rate=0.15),
            max_steps=100,
            description="Crisis management with severe resource shortages."
        ))
        
        # Stage 3: Emergency surge
        curriculum.append(TaskConfiguration(
            name="Advanced Stage 3: Emergency Surge",
            difficulty=TaskDifficulty.HARD,
            patients=PatientDistribution(normal_count=2, emergency_count=5, critical_count=3),
            resources=ResourceConfiguration(doctors=5, beds=7),
            events=EventConfiguration(enabled=True, emergency_event_prob=0.1),
            max_steps=120,
            description="Handle mass casualty scenario with emergency surge."
        ))
        
        return curriculum


class TaskValidator:
    """
    Validator for task configurations to ensure consistency and correctness.
    """
    
    @staticmethod
    def validate_config(config: TaskConfiguration) -> List[str]:
        """
        Validate task configuration and return list of issues.
        
        Args:
            config: Task configuration to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Validate patient distribution
        if config.patients.total() <= 0:
            issues.append("Task must have at least one patient")
        
        if config.patients.total() > 50:
            issues.append("Task has too many patients (>50)")
        
        # Validate resources
        if config.resources.doctors <= 0:
            issues.append("Task must have at least one doctor")
        
        if config.resources.beds <= 0:
            issues.append("Task must have at least one bed")
        
        if config.resources.doctor_capacity <= 0:
            issues.append("Doctor capacity must be positive")
        
        # Validate resource-patient balance
        total_capacity = config.resources.doctors * config.resources.doctor_capacity
        if total_capacity < config.patients.total():
            issues.append(f"Insufficient doctor capacity: {total_capacity} < {config.patients.total()}")
        
        if config.resources.beds < config.patients.total():
            issues.append(f"Insufficient beds: {config.resources.beds} < {config.patients.total()}")
        
        # Validate events
        try:
            config.events.validate()
        except ValueError as e:
            issues.append(str(e))
        
        # Validate time horizon
        if config.max_steps <= 0:
            issues.append("Max steps must be positive")
        
        if config.max_steps > 1000:
            issues.append("Max steps too long (>1000)")
        
        return issues
    
    @staticmethod
    def is_valid(config: TaskConfiguration) -> bool:
        """
        Check if configuration is valid.
        
        Args:
            config: Task configuration to check
            
        Returns:
            True if valid, False otherwise
        """
        return len(TaskValidator.validate_config(config)) == 0


# Convenience functions for direct use
def easy(seed: Optional[int] = None) -> Dict[str, Any]:
    """Create easy task configuration dictionary."""
    return TaskFactory.easy(seed).to_dict()


def medium(seed: Optional[int] = None) -> Dict[str, Any]:
    """Create medium task configuration dictionary."""
    return TaskFactory.medium(seed).to_dict()


def hard(seed: Optional[int] = None) -> Dict[str, Any]:
    """Create hard task configuration dictionary."""
    return TaskFactory.hard(seed).to_dict()


def custom(patients: PatientDistribution,
          resources: ResourceConfiguration,
          events: EventConfiguration,
          max_steps: int = 100,
          seed: Optional[int] = None,
          name: str = "Custom Task") -> Dict[str, Any]:
    """Create custom task configuration dictionary."""
    return TaskFactory.custom(patients, resources, events, max_steps, seed, name).to_dict()


# Task registry for easy access
TASK_REGISTRY = {
    TaskDifficulty.EASY: TaskFactory.easy,
    TaskDifficulty.MEDIUM: TaskFactory.medium,
    TaskDifficulty.HARD: TaskFactory.hard
}


def get_task(difficulty: Union[str, TaskDifficulty], seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Get task configuration by difficulty level.
    
    Args:
        difficulty: Task difficulty level
        seed: Optional random seed
        
    Returns:
        Task configuration dictionary
    """
    if isinstance(difficulty, str):
        difficulty = TaskDifficulty(difficulty)
    
    if difficulty not in TASK_REGISTRY:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    
    return TASK_REGISTRY[difficulty](seed).to_dict()


def list_available_tasks() -> List[str]:
    """List all available task names."""
    return [difficulty.value for difficulty in TaskDifficulty]


def validate_task_config(config: Dict[str, Any]) -> bool:
    """
    Validate task configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Convert back to TaskConfiguration for validation
        # This is a simplified validation - in practice, you'd reconstruct the full object
        required_keys = ["name", "difficulty", "initial_patients", "resources", "events", "max_steps"]
        return all(key in config for key in required_keys)
    except Exception:
        return False
