"""
Medium Task Configuration

Moderate scenario with balanced resources and variable patient flow.
Represents typical hospital operating conditions.
"""

from typing import Dict, Any
from .base_config import BaseTaskConfig


class MediumTaskConfig(BaseTaskConfig):
    """
    Medium difficulty hospital task.
    
    Features:
    - 20 ICU beds (moderate utilization target)
    - 10 doctors (moderate availability)
    - Medium patient arrival rate (3-5 per hour)
    - Mixed priority distribution
    - Tighter resource constraints
    - Dynamic patient flow patterns
    """
    
    def __init__(self) -> None:
        super().__init__(difficulty="medium")
    
    def _build_config(self) -> None:
        """Build medium task configuration."""
        self.config_dict = {
            "difficulty": "medium",
            "description": "Typical hospital scenario with balanced load",
            
            "environment": {
                "duration": 168,  # 1 week in hours
                "time_step": 1,   # 1 hour per step
                "random_seed": None,
                
                "resources": {
                    "icu_beds": 20,
                    "target_utilization": 0.75,
                    "equipment_per_bed": [
                        "ventilator", "monitor", "infusion_pump", "defibrillator"
                    ],
                    "maintenance_schedule": "random"
                },
                
                "staff": {
                    "doctors": 10,
                    "max_patients_per_doctor": 4,
                    "specialties": [
                        "general", "cardiology", "neurology", "trauma", "icu_specialist"
                    ],
                    "shift_duration": 12  # hours
                },
                
                "patients": {
                    "arrival_rate": 4.0,  # patients per hour
                    "arrival_pattern": "time_varying",  # Higher during day
                    "priority_distribution": {
                        "critical": 0.10,
                        "high": 0.25,
                        "medium": 0.35,
                        "low": 0.30
                    },
                    "avg_stay_duration": 72,  # hours
                    "readmission_rate": 0.05
                }
            },
            
            "state": {
                "state_dim": 128,
                "include_patient_history": True,
                "history_length": 12,
                "include_doctor_workload": True,
                "include_resource_queue": True,
                "include_time_features": True,
                "normalization": "standardize"
            },
            
            "reward": {
                "reward_weights": {
                    "patient_outcome": 1.0,
                    "resource_utilization": 0.5,
                    "waiting_time": -0.3,
                    "workload_balance": 0.2,
                    "operational_cost": -0.1,
                    "emergency_response": 0.3
                },
                "use_reward_shaping": True,
                "target_utilization": 0.75,
                "max_wait_threshold": 3.0,  # hours
                "gamma": 0.99
            },
            
            "agent": {
                "action_space_type": "discrete",
                "num_actions": 50,  # More granular actions
                "learning_rate": 0.0005,
                "exploration_schedule": "linear_decay"
            }
        }
