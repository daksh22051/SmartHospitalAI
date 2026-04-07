"""
Hard Task Configuration

Challenging scenario with constrained resources and high patient load.
Tests the limits of the agent's decision-making capabilities.
"""

from typing import Dict, Any
from .base_config import BaseTaskConfig


class HardTaskConfig(BaseTaskConfig):
    """
    Hard difficulty hospital task.
    
    Features:
    - 30 ICU beds (high utilization target)
    - 15 doctors (stretched thin)
    - High patient arrival rate (6-8 per hour)
    - High proportion of critical patients
    - Severe resource constraints
    - Complex patient interactions
    - Equipment failures and emergencies
    - Staff scheduling challenges
    """
    
    def __init__(self) -> None:
        super().__init__(difficulty="hard")
    
    def _build_config(self) -> None:
        """Build hard task configuration."""
        self.config_dict = {
            "difficulty": "hard",
            "description": "High-load hospital scenario with resource constraints",
            
            "environment": {
                "duration": 336,  # 2 weeks in hours
                "time_step": 1,   # 1 hour per step
                "random_seed": None,
                
                "resources": {
                    "icu_beds": 30,
                    "target_utilization": 0.90,
                    "equipment_per_bed": [
                        "ventilator", "monitor", "infusion_pump", 
                        "defibrillator", "dialysis_machine"
                    ],
                    "shared_equipment": {
                        "portable_xray": 3,
                        "ecmo_machines": 2
                    },
                    "maintenance_schedule": "random",
                    "equipment_failure_rate": 0.02,
                    "surge_capacity": 5  # Temporary beds
                },
                
                "staff": {
                    "doctors": 15,
                    "max_patients_per_doctor": 5,
                    "specialties": [
                        "general", "cardiology", "neurology", "trauma", 
                        "pediatrics", "icu_specialist"
                    ],
                    "shift_duration": 12,
                    "overtime_threshold": 60,  # hours per week
                    "burnout_factor": True
                },
                
                "patients": {
                    "arrival_rate": 7.0,  # patients per hour
                    "arrival_pattern": "burst",  # Unpredictable spikes
                    "burst_probability": 0.1,
                    "burst_multiplier": 3.0,
                    "priority_distribution": {
                        "critical": 0.20,
                        "high": 0.30,
                        "medium": 0.30,
                        "low": 0.20
                    },
                    "avg_stay_duration": 96,  # hours
                    "condition_deterioration_rate": 0.05,
                    "readmission_rate": 0.10,
                    "complication_rate": 0.08
                },
                
                "events": {
                    "enable_emergencies": True,
                    "emergency_frequency": 0.05,
                    "enable_mass_casualty": True,
                    "mass_casualty_probability": 0.02
                }
            },
            
            "state": {
                "state_dim": 256,
                "include_patient_history": True,
                "history_length": 24,
                "include_doctor_workload": True,
                "include_doctor_fatigue": True,
                "include_resource_queue": True,
                "include_equipment_status": True,
                "include_time_features": True,
                "include_predicted_arrivals": True,
                "normalization": "standardize"
            },
            
            "reward": {
                "reward_weights": {
                    "patient_outcome": 1.5,
                    "resource_utilization": 0.4,
                    "waiting_time": -0.5,
                    "workload_balance": 0.3,
                    "operational_cost": -0.2,
                    "emergency_response": 0.5,
                    "staff_wellbeing": 0.2
                },
                "use_reward_shaping": True,
                "target_utilization": 0.90,
                "max_wait_threshold": 1.5,  # hours
                "gamma": 0.99,
                "death_penalty": -10.0
            },
            
            "agent": {
                "action_space_type": "multi_discrete",
                "action_dimensions": [10, 10, 5, 5],  # Multiple decision axes
                "learning_rate": 0.0001,
                "exploration_schedule": "exponential_decay",
                "recurrent_architecture": True,
                "attention_mechanism": True
            }
        }
