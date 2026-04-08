"""
Easy Task Configuration

Simple scenario with abundant resources and stable patient flow.
Ideal for initial training and debugging.
"""

from typing import Dict, Any
from .base_config import BaseTaskConfig


class EasyTaskConfig(BaseTaskConfig):
    """
    Easy difficulty hospital task.
    
    Features:
    - 10 ICU beds (low utilization target)
    - 5 doctors (high availability)
    - Low patient arrival rate (1-2 per hour)
    - Mostly low-priority patients
    - Generous resource buffer
    - Simple reward structure
    """
    
    def __init__(self) -> None:
        super().__init__(difficulty="easy")
    
    def _build_config(self) -> None:
        """Build easy task configuration."""
        self.config_dict = {
            "difficulty": "easy",
            "description": "Low-load hospital scenario for initial training",
            
            "environment": {
                "duration": 168,  # 1 week in hours
                "time_step": 1,   # 1 hour per step
                "random_seed": 42,
                
                "resources": {
                    "icu_beds": 10,
                    "target_utilization": 0.60,
                    "equipment_per_bed": ["ventilator", "monitor", "infusion_pump"]
                },
                
                "staff": {
                    "doctors": 5,
                    "max_patients_per_doctor": 3,
                    "specialties": ["general", "icu_specialist"]
                },
                
                "patients": {
                    "arrival_rate": 1.5,  # patients per hour (Poisson)
                    "priority_distribution": {
                        "critical": 0.05,
                        "high": 0.15,
                        "medium": 0.30,
                        "low": 0.50
                    },
                    "avg_stay_duration": 48  # hours
                }
            },
            
            "state": {
                "state_dim": 64,
                "include_patient_history": False,
                "include_doctor_workload": True,
                "include_resource_queue": True,
                "normalization": "minmax"
            },
            
            "reward": {
                "reward_weights": {
                    "patient_outcome": 1.0,
                    "resource_utilization": 0.3,
                    "waiting_time": -0.1,
                    "workload_balance": 0.1,
                    "operational_cost": -0.05
                },
                "use_reward_shaping": True,
                "target_utilization": 0.60,
                "max_wait_threshold": 6.0  # hours
            },
            
            "agent": {
                "action_space_type": "discrete",
                "num_actions": 20,  # Simplified action space
                "learning_rate": 0.001
            }
        }
