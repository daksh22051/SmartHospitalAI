"""
Baseline Agent Inference System for Smart Hospital Resource Orchestration
Fixed for Scaler Hackathon LLM Proxy Compliance.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import argparse
import time
import os
from dataclasses import dataclass

# ==========================================
# SCALER PROXY COMPLIANCE INJECTION
# ==========================================
# Force libraries to use Scaler Proxy if available
API_BASE = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")

if API_BASE:
    os.environ["OPENAI_API_BASE"] = API_BASE
    os.environ["OPENAI_API_KEY"] = API_KEY
    os.environ["LITELLM_API_BASE"] = API_BASE
    os.environ["LITELLM_API_KEY"] = API_KEY

@dataclass
class AgentMetrics:
    """Container for agent performance metrics."""
    total_reward: float = 0.0
    steps: int = 0
    patients_treated: int = 0
    critical_patients_treated: int = 0
    emergency_patients_treated: int = 0
    normal_patients_treated: int = 0
    escalations_made: int = 0
    deferrals_made: int = 0
    reassignments_made: int = 0
    wait_actions: int = 0
    allocation_success: int = 0
    allocation_failures: int = 0
    
    def print_summary(self) -> None:
        """Print performance summary."""
        print("\n" + "="*60)
        print("BASELINE AGENT PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Reward: {self.total_reward:.2f}")
        print(f"Episode Length: {self.steps} steps")
        print(f"Patients Treated: {self.patients_treated}")
        print(f"  - Critical: {self.critical_patients_treated}")
        print(f"  - Emergency: {self.emergency_patients_treated}")
        print(f"  - Normal: {self.normal_patients_treated}")
        print(f"Actions Taken:")
        print(f"  - Escalations: {self.escalations_made}")
        print(f"  - Deferrals: {self.deferrals_made}")
        print(f"  - Reassignments: {self.reassignments_made}")
        print(f"  - Wait: {self.wait_actions}")
        print(f"Allocation Success Rate: {self.allocation_success}/{self.allocation_success + self.allocation_failures} "
              f"({self.allocation_success/(self.allocation_success + self.allocation_failures + 1e-6):.1%})")
        print(f"Average Reward per Step: {self.total_reward/max(self.steps, 1):.3f}")
        print("="*60)


class HospitalBaselineAgent:
    """
    Intelligent rule-based agent for hospital resource orchestration.
    """
    
    def __init__(self, verbose: bool = True, log_interval: int = 10):
        self.verbose = verbose
        self.log_interval = log_interval
        self.metrics = AgentMetrics()
        self.step_count = 0
        
        # Action constants for readability
        self.WAIT = 0
        self.ALLOCATE = 1
        self.ESCALATE = 2
        self.DEFER = 3
        self.REASSIGN = 4

    def select_action(self, state: Dict[str, Any]) -> int:
        """Select action based on priority rules."""
        # Extract information
        patients_raw = state.get("patients", np.array([]))
        doctors = state.get("doctors", np.array([]))
        beds = state.get("beds", np.array([]))
        time_stats = state.get("time", np.array([]))
        
        current_step = int(time_stats[0]) if len(time_stats) > 0 else 0
        total_patients = int(time_stats[2]) if len(time_stats) > 2 else len(patients_raw)

        visible_patients = min(total_patients, len(patients_raw))
        patients = patients_raw[:visible_patients] if visible_patients > 0 else np.array([])
        
        if len(patients) == 0:
            return self.WAIT
        
        available_doctors = self._count_available_doctors(doctors)
        available_beds = self._count_available_beds(beds)
        waiting_patients = self._get_waiting_patients(patients)
        total_waiting = len(waiting_patients)
        
        critical_waiting = self._find_critical_patients(waiting_patients)
        emergency_waiting = self._find_emergency_patients(waiting_patients)
        normal_waiting = self._find_normal_patients(waiting_patients)
        
        has_resources = available_doctors > 0 and available_beds > 0
        
        # 1. CRITICAL ESCALATION (Priority)
        if len(critical_waiting) > 0:
            critical_waiting_long = self._find_critical_patients_waiting_too_long(waiting_patients)
            if len(critical_waiting_long) > 0 or (not has_resources and len(critical_waiting) >= 2):
                self.metrics.escalations_made += 1
                return self.ESCALATE
        
        # 2. ALLOCATE
        if has_resources and total_waiting > 0:
            self.metrics.allocation_success += 1
            return self.ALLOCATE
        
        # 3. DEFER (Load Balancing)
        system_load = self._calculate_system_load(total_patients, doctors, beds)
        if system_load > 0.7 and len(normal_waiting) > 0 and not has_resources:
            self.metrics.deferrals_made += 1
            return self.DEFER
        
        # 4. REASSIGN
        if self._needs_reassignment(doctors):
            self.metrics.reassignments_made += 1
            return self.REASSIGN
        
        # 5. WAIT
        self.metrics.wait_actions += 1
        return self.WAIT
    
    def _count_available_doctors(self, doctors: np.ndarray) -> int:
        if len(doctors) == 0: return 0
        active_mask = doctors[:, 3] > 0
        return np.sum((doctors[:, 1] == 1.0) & active_mask)
    
    def _count_available_beds(self, beds: np.ndarray) -> int:
        if len(beds) == 0: return 0
        active_mask = beds[:, 3] > 0
        return np.sum((beds[:, 1] == 1.0) & active_mask)
    
    def _get_waiting_patients(self, patients: np.ndarray) -> np.ndarray:
        if len(patients) == 0: return np.array([])
        return patients[patients[:, 2] == 0.0]
    
    def _find_critical_patients(self, waiting_patients: np.ndarray) -> np.ndarray:
        if len(waiting_patients) == 0: return np.array([])
        return waiting_patients[waiting_patients[:, 1] == 2.0]
    
    def _find_critical_patients_waiting_too_long(self, waiting_patients: np.ndarray) -> np.ndarray:
        if len(waiting_patients) == 0: return np.array([])
        return waiting_patients[(waiting_patients[:, 1] == 2.0) & (waiting_patients[:, 3] > 2)]
    
    def _find_emergency_patients(self, waiting_patients: np.ndarray) -> np.ndarray:
        if len(waiting_patients) == 0: return np.array([])
        return waiting_patients[waiting_patients[:, 1] == 1.0]
    
    def _find_normal_patients(self, waiting_patients: np.ndarray) -> np.ndarray:
        if len(waiting_patients) == 0: return np.array([])
        return waiting_patients[waiting_patients[:, 1] == 0.0]
    
    def _calculate_system_load(self, total_patients: int, doctors: np.ndarray, beds: np.ndarray) -> float:
        active_res = np.sum(doctors[:, 3] > 0) + np.sum(beds[:, 3] > 0)
        return total_patients / max(active_res, 1)
    
    def _needs_reassignment(self, doctors: np.ndarray) -> bool:
        if len(doctors) < 2: return False
        loads = doctors[:, 2]
        return np.max(loads) - np.min(loads) > 1

    def update_metrics(self, state: Dict[str, Any], action: int, reward: float, info: Dict[str, Any]) -> None:
        self.metrics.total_reward += reward
        self.step_count += 1
        if "patients" in state:
            admitted = state["patients"][state["patients"][:, 2] == 1.0]
            for p in admitted:
                sev = int(p[1])
                if sev == 2: self.metrics.critical_patients_treated += 1
                elif sev == 1: self.metrics.emergency_patients_treated += 1
                else: self.metrics.normal_patients_treated += 1
                self.metrics.patients_treated += 1
        if action == self.ALLOCATE and reward < 0:
            self.metrics.allocation_failures += 1
    
    def reset_metrics(self) -> None:
        self.metrics = AgentMetrics()
        self.step_count = 0


class EpisodeRunner:
    def __init__(self, agent: HospitalBaselineAgent, max_episodes: int = 1):
        self.agent = agent
        self.max_episodes = max_episodes
    
    def run_episode(self, env, task_name: str, seed: Optional[int] = None) -> AgentMetrics:
        state, info = env.reset(seed=seed)
        self.agent.reset_metrics()
        done = False
        step = 0
        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done, next_info = env.step(action)
            self.agent.update_metrics(state, action, reward, next_info)
            state = next_state
            step += 1
        return self.agent.metrics
    
    def run_multiple_episodes(self, env, task_configs: List[Tuple[str, Dict[str, Any]]]) -> List[AgentMetrics]:
        all_metrics = []
        for episode_idx, (task_name, config) in enumerate(task_configs, 1):
            metrics = self.run_episode(env, task_name, seed=episode_idx * 42)
            all_metrics.append(metrics)
            if self.agent.verbose:
                metrics.print_summary()
        return all_metrics


def create_environment(task: str = "medium") -> Any:
    try:
        from smart_hospital_orchestration.environment import HospitalEnv
        return HospitalEnv(task)
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Run baseline agent inference")
    parser.add_argument("--task", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    
    args = parser.parse_args()
    agent = HospitalBaselineAgent(verbose=args.verbose, log_interval=args.log_interval)
    env = create_environment(args.task)
    if env is None: return
    
    runner = EpisodeRunner(agent, max_episodes=args.episodes)
    task_configs = [(f"{args.task.upper()} Task", {"difficulty": args.task})]
    
    start_time = time.time()
    all_metrics = runner.run_multiple_episodes(env, task_configs)
    end_time = time.time()
    
    if args.verbose and len(all_metrics) > 1:
        print(f"Average Reward: {np.mean([m.total_reward for m in all_metrics]):.2f}")


if __name__ == "__main__":
    main()