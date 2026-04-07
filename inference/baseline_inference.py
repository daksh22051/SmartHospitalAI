"""
Baseline Agent Inference System for Smart Hospital Resource Orchestration

Implements intelligent rule-based decision making that serves as a performance
benchmark for reinforcement learning agents in hospital resource management.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import argparse
import time
from dataclasses import dataclass


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
    
    Implements medical triage principles with priority-based decision making
    to provide a strong baseline for RL agent evaluation.
    """
    
    def __init__(self, verbose: bool = True, log_interval: int = 10):
        """
        Initialize baseline agent.
        
        Args:
            verbose: Whether to print detailed logs
            log_interval: How often to print step logs
        """
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
        """
        Select action based on IMPROVED intelligent rule-based policy.
        
        SMART PRIORITY ORDER:
        1. CRITICAL patients waiting → ESCALATE immediately (don't wait too long)
        2. Any patients with available resources → ALLOCATE (don't waste resources)
        3. CRITICAL waiting without resources → ESCALATE (prioritize them)
        4. System overloaded → DEFER normal patients
        5. Workload imbalance → REASSIGN
        6. Default → WAIT only when truly idle
        
        Args:
            state: Current environment state
            
        Returns:
            Selected action (0-4)
        """
        # Extract key information from state
        patients_raw = state.get("patients", np.array([]))
        doctors = state.get("doctors", np.array([]))
        beds = state.get("beds", np.array([]))
        time_stats = state.get("time", np.array([]))
        current_step = int(time_stats[0]) if len(time_stats) > 0 else 0
        total_patients = int(time_stats[2]) if len(time_stats) > 2 else len(patients_raw)

        # State arrays are padded with zeros; use only real patient rows
        visible_patients = min(total_patients, len(patients_raw))
        patients = patients_raw[:visible_patients] if visible_patients > 0 else np.array([])
        
        if len(patients) == 0:
            return self.WAIT
        
        # Count resources and patients by status
        available_doctors = self._count_available_doctors(doctors)
        available_beds = self._count_available_beds(beds)
        waiting_patients = self._get_waiting_patients(patients)
        total_waiting = len(waiting_patients)
        
        # Categorize waiting patients by severity
        critical_waiting = self._find_critical_patients(waiting_patients)
        emergency_waiting = self._find_emergency_patients(waiting_patients)
        normal_waiting = self._find_normal_patients(waiting_patients)
        
        has_resources = available_doctors > 0 and available_beds > 0
        
        # ===== PRIORITY 1: IMMEDIATE CRITICAL ESCALATION =====
        # If critical patients are waiting, escalate them immediately
        # This is the highest priority - don't let critical patients wait
        if len(critical_waiting) > 0:
            # Check if any critical patient has been waiting too long
            critical_waiting_long = self._find_critical_patients_waiting_too_long(waiting_patients)
            if len(critical_waiting_long) > 0:
                if self.verbose and self.step_count % self.log_interval == 0:
                    print(f"  → Escalating {len(critical_waiting_long)} critical patients (waiting too long)")
                self.metrics.escalations_made += len(critical_waiting_long)
                return self.ESCALATE
            
            # Even if not waiting long, if we have no resources, escalate to prioritize
            if not has_resources and len(critical_waiting) >= 2:
                if self.verbose and self.step_count % self.log_interval == 0:
                    print(f"  → Escalating {len(critical_waiting)} critical patients (resource shortage)")
                self.metrics.escalations_made += len(critical_waiting)
                return self.ESCALATE
        
        # ===== PRIORITY 2: ALLOCATE IF RESOURCES AVAILABLE =====
        # Don't waste idle resources - allocate to ANY waiting patients
        if has_resources and total_waiting > 0:
            # Priority order: critical > emergency > normal
            if len(critical_waiting) > 0:
                if self.verbose and self.step_count % self.log_interval == 0:
                    print(f"  → Allocating to {len(critical_waiting)} critical patients")
                self.metrics.allocation_success += 1
                return self.ALLOCATE
            elif len(emergency_waiting) > 0:
                if self.verbose and self.step_count % self.log_interval == 0:
                    print(f"  → Allocating to {len(emergency_waiting)} emergency patients")
                self.metrics.allocation_success += 1
                return self.ALLOCATE
            elif len(normal_waiting) > 0:
                if self.verbose and self.step_count % self.log_interval == 0:
                    print(f"  → Allocating to {len(normal_waiting)} normal patients")
                self.metrics.allocation_success += 1
                return self.ALLOCATE
        
        # ===== PRIORITY 3: SYSTEM OVERLOAD - DEFER NORMAL PATIENTS =====
        # If system is overloaded, defer lower priority patients
        system_load = self._calculate_system_load(total_patients, doctors, beds)
        if system_load > 0.7 and len(normal_waiting) > 0 and not has_resources:
            if self.verbose and self.step_count % self.log_interval == 0:
                print(f"  → Deferring {len(normal_waiting)} normal patients (system load: {system_load:.1%})")
            self.metrics.deferrals_made += len(normal_waiting)
            return self.DEFER
        
        # ===== PRIORITY 4: WORKLOAD BALANCING =====
        if self._needs_reassignment(doctors):
            if self.verbose and self.step_count % self.log_interval == 0:
                print("  → Reassigning patients to balance doctor workloads")
            self.metrics.reassignments_made += 1
            return self.REASSIGN
        
        # ===== PRIORITY 5: SMART WAIT =====
        # Only wait if truly idle (no waiting patients or no resources and can't help)
        if total_waiting == 0:
            if self.verbose and self.step_count % self.log_interval == 0:
                print("  → Waiting (no patients waiting)")
            self.metrics.wait_actions += 1
            return self.WAIT
        
        # If we have resources but chose not to allocate, try to allocate
        if has_resources and total_waiting > 0:
            if self.verbose and self.step_count % self.log_interval == 0:
                print(f"  → Allocating to {total_waiting} waiting patients (opportunistic)")
            self.metrics.allocation_success += 1
            return self.ALLOCATE
        
        # Default: WAIT with explanation
        if self.verbose and self.step_count % self.log_interval == 0:
            print(f"  → Waiting ({total_waiting} waiting, {available_doctors} doctors, {available_beds} beds)")
        self.metrics.wait_actions += 1
        return self.WAIT
    
    def _count_available_doctors(self, doctors: np.ndarray) -> int:
        """Count available doctors."""
        if len(doctors) == 0:
            return 0
        # Ignore padded rows (max_patients == 0)
        active_mask = doctors[:, 3] > 0
        return np.sum((doctors[:, 1] == 1.0) & active_mask)  # available flag
    
    def _count_available_beds(self, beds: np.ndarray) -> int:
        """Count available beds."""
        if len(beds) == 0:
            return 0
        # Ignore padded rows (equipment count == 0)
        active_mask = beds[:, 3] > 0
        return np.sum((beds[:, 1] == 1.0) & active_mask)  # available flag
    
    def _get_waiting_patients(self, patients: np.ndarray) -> np.ndarray:
        """Get all waiting patients."""
        if len(patients) == 0:
            return np.array([])
        waiting_mask = patients[:, 2] == 0.0  # WAITING status
        return patients[waiting_mask]
    
    def _find_critical_patients(self, waiting_patients: np.ndarray) -> np.ndarray:
        """Find all critical patients (any wait time)."""
        if len(waiting_patients) == 0:
            return np.array([])
        critical_mask = waiting_patients[:, 1] == 2.0  # CRITICAL severity
        return waiting_patients[critical_mask]
    
    def _find_critical_patients_waiting_too_long(self, waiting_patients: np.ndarray) -> np.ndarray:
        """Find critical patients waiting more than 2 steps."""
        if len(waiting_patients) == 0:
            return np.array([])
        critical_mask = waiting_patients[:, 1] == 2.0  # CRITICAL severity
        long_wait_mask = waiting_patients[:, 3] > 2  # wait_time > 2
        return waiting_patients[critical_mask & long_wait_mask]
    
    def _find_emergency_patients(self, waiting_patients: np.ndarray) -> np.ndarray:
        """Find emergency patients."""
        if len(waiting_patients) == 0:
            return np.array([])
        emergency_mask = waiting_patients[:, 1] == 1.0  # EMERGENCY severity
        return waiting_patients[emergency_mask]
    
    def _find_normal_patients(self, waiting_patients: np.ndarray) -> np.ndarray:
        """Find normal patients."""
        if len(waiting_patients) == 0:
            return np.array([])
        normal_mask = waiting_patients[:, 1] == 0.0  # NORMAL severity
        return waiting_patients[normal_mask]
    
    def _calculate_system_load(self, total_patients: int, doctors: np.ndarray, beds: np.ndarray) -> float:
        """Calculate system load ratio."""
        active_doctors = np.sum(doctors[:, 3] > 0) if len(doctors) > 0 else 0
        active_beds = np.sum(beds[:, 3] > 0) if len(beds) > 0 else 0
        total_resources = active_doctors + active_beds
        return total_patients / max(total_resources, 1)
    
    def _needs_reassignment(self, doctors: np.ndarray) -> bool:
        """Check if workload rebalancing is needed."""
        if len(doctors) < 2:
            return False
        
        loads = doctors[:, 2]  # current_load column
        max_load = np.max(loads)
        min_load = np.min(loads)
        
        return max_load - min_load > 1
    
    def update_metrics(self, state: Dict[str, Any], action: int, reward: float, info: Dict[str, Any]) -> None:
        """
        Update performance metrics after each step.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            info: Additional info from environment
        """
        self.metrics.total_reward += reward
        self.step_count += 1
        
        # Track patient treatment
        if "patients" in state:
            patients = state["patients"]
            admitted_mask = patients[:, 2] == 1.0  # ADMITTED status
            admitted_patients = patients[admitted_mask]
            
            for patient in admitted_patients:
                severity = int(patient[1])
                if severity == 2:  # CRITICAL
                    self.metrics.critical_patients_treated += 1
                elif severity == 1:  # EMERGENCY
                    self.metrics.emergency_patients_treated += 1
                else:  # NORMAL
                    self.metrics.normal_patients_treated += 1
                self.metrics.patients_treated += 1
        
        # Track allocation failures
        if action == self.ALLOCATE and reward < 0:
            self.metrics.allocation_failures += 1
    
    def reset_metrics(self) -> None:
        """Reset all metrics for new episode."""
        self.metrics = AgentMetrics()
        self.step_count = 0


class EpisodeRunner:
    """
    Handles episode execution for the baseline agent.
    """
    
    def __init__(self, agent: HospitalBaselineAgent, max_episodes: int = 1):
        """
        Initialize episode runner.
        
        Args:
            agent: Baseline agent instance
            max_episodes: Number of episodes to run
        """
        self.agent = agent
        self.max_episodes = max_episodes
    
    def run_episode(self, env, task_name: str, seed: Optional[int] = None) -> AgentMetrics:
        """
        Run a single episode with the baseline agent.
        
        Args:
            env: Hospital environment instance
            task_name: Name of the task being run
            seed: Optional random seed
            
        Returns:
            Episode metrics
        """
        if self.agent.verbose:
            print(f"\n{'='*60}")
            print(f"RUNNING EPISODE: {task_name.upper()}")
            if seed is not None:
                print(f"Seed: {seed}")
            print(f"{'='*60}")
        
        # Reset environment and agent
        state, info = env.reset(seed=seed)
        self.agent.reset_metrics()
        
        terminated = False
        truncated = False
        done = False
        step = 0
        
        if self.agent.verbose:
            print(f"Initial State: {info.get('num_patients', 0)} patients, "
                  f"{info.get('available_doctors', 0)} doctors, "
                  f"{info.get('available_beds', 0)} beds")
            print("-" * 60)
        
        # Main episode loop
        while not done:
            # Select action using rule-based policy
            action = self.agent.select_action(state)
            
            # Execute action
            next_state, reward, done, next_info = env.step(action)
            
            # Extract terminated and truncated from info
            terminated = next_info.get("terminated", False)
            truncated = next_info.get("truncated", False)
            
            # Update metrics
            self.agent.update_metrics(state, action, reward, next_info)
            
            # Log step information
            if self.agent.verbose and (step % self.agent.log_interval == 0 or step < 5):
                action_names = ["WAIT", "ALLOCATE", "ESCALATE", "DEFER", "REASSIGN"]
                print(f"Step {step+1:3d}: {action_names[action]:12s} | "
                      f"Reward: {reward:6.2f} | "
                      f"Patients: {next_info.get('num_patients', 0):2d} | "
                      f"Waiting: {next_info.get('waiting_patients', 0):2d} | "
                      f"Admitted: {next_info.get('admitted_patients', 0):2d}")
            
            # Update state
            state = next_state
            step += 1
        
        if self.agent.verbose:
            print(f"\nEpisode terminated after {step} steps")
            print(f"Final state: {next_info.get('waiting_patients', 0)} waiting, "
                  f"{next_info.get('admitted_patients', 0)} admitted")
        
        return self.agent.metrics
    
    def run_multiple_episodes(self, env, task_configs: List[Tuple[str, Dict[str, Any]]]) -> List[AgentMetrics]:
        """
        Run multiple episodes with different task configurations.
        
        Args:
            env: Hospital environment instance
            task_configs: List of (task_name, config) tuples
            
        Returns:
            List of episode metrics
        """
        all_metrics = []
        
        for episode_idx, (task_name, config) in enumerate(task_configs, 1):
            if self.agent.verbose:
                print(f"\n{'='*80}")
                print(f"EPISODE {episode_idx}/{len(task_configs)}: {task_name}")
                print(f"{'='*80}")
            
            # Configure environment for this task
            # Note: In practice, you'd reinitialize the environment with the new config
            # For this demo, we'll use the same environment with different seeds
            
            metrics = self.run_episode(env, task_name, seed=episode_idx * 42)
            all_metrics.append(metrics)
            
            if self.agent.verbose:
                metrics.print_summary()
        
        return all_metrics


def create_environment(task: str = "medium") -> Any:
    """
    Create hospital environment instance.
    
    Args:
        task: Task difficulty level
        
    Returns:
        Environment instance
    """
    try:
        # Import here to avoid circular imports
        from smart_hospital_orchestration.environment import HospitalEnv
        return HospitalEnv(task)
    except ImportError as e:
        print(f"Error importing HospitalEnv: {e}")
        print("Please ensure the hospital environment is properly installed")
        return None


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description="Run baseline agent inference")
    parser.add_argument("--task", type=str, default="medium", 
                       choices=["easy", "medium", "hard"],
                       help="Task difficulty level")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to run")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose logging")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Logging interval (steps)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create baseline agent
    agent = HospitalBaselineAgent(verbose=args.verbose, log_interval=args.log_interval)
    
    # Create environment
    env = create_environment(args.task)
    if env is None:
        return
    
    # Create episode runner
    runner = EpisodeRunner(agent, max_episodes=args.episodes)
    
    # Define task configurations
    task_configs = [
        (f"{args.task.upper()} Task", {"difficulty": args.task})
    ]
    
    # Run episodes
    start_time = time.time()
    all_metrics = runner.run_multiple_episodes(env, task_configs)
    end_time = time.time()
    
    # Print overall summary
    if args.verbose and len(all_metrics) > 1:
        print(f"\n{'='*80}")
        print("OVERALL PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Total Episodes: {len(all_metrics)}")
        print(f"Total Time: {end_time - start_time:.2f} seconds")
        print(f"Average Reward: {np.mean([m.total_reward for m in all_metrics]):.2f}")
        print(f"Average Steps: {np.mean([m.steps for m in all_metrics]):.1f}")
        print(f"Average Patients Treated: {np.mean([m.patients_treated for m in all_metrics]):.1f}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
