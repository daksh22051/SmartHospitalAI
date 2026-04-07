"""
Default Configuration

Default configuration values for the hospital environment.
"""

from typing import Dict, Any


DEFAULT_CONFIG: Dict[str, Any] = {
    "environment": {
        "duration": 168,
        "time_step": 1,
        "random_seed": None,
        
        "resources": {
            "icu_beds": 15,
            "target_utilization": 0.80,
            "equipment_per_bed": ["monitor", "infusion_pump"]
        },
        
        "staff": {
            "doctors": 8,
            "max_patients_per_doctor": 4,
            "specialties": ["general"]
        },
        
        "patients": {
            "arrival_rate": 3.0,
            "priority_distribution": {
                "critical": 0.10,
                "high": 0.20,
                "medium": 0.40,
                "low": 0.30
            },
            "avg_stay_duration": 60
        }
    },
    
    "state": {
        "state_dim": 128,
        "include_patient_history": True,
        "include_doctor_workload": True,
        "include_resource_queue": True,
        "normalization": "standardize"
    },
    
    "reward": {
        "reward_weights": {
            "patient_outcome": 1.0,
            "resource_utilization": 0.5,
            "waiting_time": -0.3,
            "workload_balance": 0.2,
            "operational_cost": -0.1
        },
        "use_reward_shaping": True,
        "target_utilization": 0.80,
        "max_wait_threshold": 3.0,
        "gamma": 0.99
    },
    
    "agent": {
        "action_space_type": "discrete",
        "num_actions": 30,
        "learning_rate": 0.0005
    },
    
    "logging": {
        "level": "INFO",
        "log_interval": 100,
        "save_interval": 1000,
        "tensorboard": True
    },
    
    "training": {
        "total_timesteps": 1000000,
        "eval_frequency": 10000,
        "save_frequency": 50000,
        "num_eval_episodes": 10
    }
}
