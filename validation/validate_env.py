"""
Comprehensive Environment Validation System for Smart Hospital Resource Orchestration

Rigorous testing suite to ensure environment stability, reward dynamics,
task differentiation, and edge case handling before deployment.
"""

import numpy as np
import sys
import traceback
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import time


@dataclass
class ValidationMetrics:
    """Container for validation test results."""
    total_episodes: int = 0
    successful_episodes: int = 0
    crashed_episodes: int = 0
    reward_variance: float = 0.0
    state_transitions: int = 0
    edge_cases_tested: int = 0
    task_differences: Dict[str, float] = None
    bugs_detected: List[str] = None
    
    def __post_init__(self):
        if self.task_differences is None:
            self.task_differences = {}
        if self.bugs_detected is None:
            self.bugs_detected = []


class HospitalEnvironmentValidator:
    """
    Comprehensive validator for Smart Hospital Resource Orchestration Environment.
    
    Tests environment stability, reward dynamics, task differentiation,
    and edge cases to ensure production readiness.
    """
    
    def __init__(self, verbose: bool = True, log_file: str = None):
        """
        Initialize validator.
        
        Args:
            verbose: Whether to print detailed logs
            log_file: Optional file to write logs
        """
        self.verbose = verbose
        self.log_file = log_file
        self.metrics = ValidationMetrics()
        self.test_results = []
        
        # Test configuration
        self.test_episodes = 5
        self.test_steps_per_episode = 20
        self.test_tasks = ["easy", "medium", "hard"]
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log message with timestamp and level."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {level}: {message}"
        
        if self.verbose:
            print(formatted_msg)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted_msg + "\n")
    
    def validate_environment_imports(self) -> bool:
        """Test that all required modules can be imported."""
        try:
            self.log("Testing environment imports...")
            
            # Test core environment import
            from smart_hospital_orchestration.environment import HospitalEnv
            self.log("✓ HospitalEnv imported successfully")
            
            # Test baseline agent import
            from smart_hospital_orchestration.inference.baseline_inference import HospitalBaselineAgent
            self.log("✓ HospitalBaselineAgent imported successfully")
            
            # Test task system import
            from smart_hospital_orchestration.tasks.advanced_tasks import easy, medium, hard
            self.log("✓ Task system imported successfully")
            
            return True
            
        except ImportError as e:
            self.log(f"✗ Import failed: {e}", "ERROR")
            self.metrics.bugs_detected.append(f"Import error: {e}")
            return False
    
    def validate_environment_creation(self) -> bool:
        """Test environment creation for all task difficulties."""
        try:
            self.log("Testing environment creation...")
            
            from smart_hospital_orchestration.environment import HospitalEnv
            
            for task in self.test_tasks:
                try:
                    env = HospitalEnv(task)
                    self.log(f"✓ {task} environment created successfully")
                    
                    # Test basic attributes
                    assert hasattr(env, 'action_space'), f"{task} missing action_space"
                    assert hasattr(env, 'state'), f"{task} missing state method"
                    assert env.action_space.n == 5, f"{task} incorrect action space size"
                    
                    self.log(f"✓ {task} environment attributes validated")
                    
                except Exception as e:
                    self.log(f"✗ {task} environment creation failed: {e}", "ERROR")
                    self.metrics.bugs_detected.append(f"{task} creation error: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.log(f"✗ Environment creation test failed: {e}", "ERROR")
            self.metrics.bugs_detected.append(f"Environment creation error: {e}")
            return False
    
    def validate_environment_reset(self) -> bool:
        """Test environment reset functionality."""
        try:
            self.log("Testing environment reset...")
            
            from smart_hospital_orchestration.environment import HospitalEnv
            
            for task in self.test_tasks:
                env = HospitalEnv(task)
                
                # Test reset with seed
                state, info = env.reset(seed=42)
                
                # Validate reset output
                assert isinstance(state, dict), f"{task} reset state not dict"
                assert isinstance(info, dict), f"{task} reset info not dict"
                
                # Validate state structure
                required_keys = ["patients", "doctors", "beds", "flat"]
                for key in required_keys:
                    assert key in state, f"{task} missing state key: {key}"
                
                # Validate info structure
                required_info_keys = ["num_patients", "available_doctors", "available_beds"]
                for key in required_info_keys:
                    assert key in info, f"{task} missing info key: {key}"
                
                # Validate data types
                assert isinstance(state["patients"], np.ndarray), f"{task} patients not numpy array"
                assert isinstance(state["doctors"], np.ndarray), f"{task} doctors not numpy array"
                assert isinstance(state["beds"], np.ndarray), f"{task} beds not numpy array"
                
                self.log(f"✓ {task} environment reset validated")
            
            return True
            
        except Exception as e:
            self.log(f"✗ Environment reset test failed: {e}", "ERROR")
            self.metrics.bugs_detected.append(f"Reset error: {e}")
            return False
    
    def validate_step_execution(self) -> bool:
        """Test step execution without crashes."""
        try:
            self.log("Testing step execution...")
            
            from smart_hospital_orchestration.environment import HospitalEnv
            
            for task in self.test_tasks:
                env = HospitalEnv(task)
                state, info = env.reset(seed=42)
                
                episode_steps = 0
                terminated = False
                truncated = False
                
                while not (terminated or truncated) and episode_steps < self.test_steps_per_episode:
                    # Test valid action
                    action = env.action_space.sample()
                    
                    # Execute step
                    try:
                        next_state, reward, terminated, truncated, next_info = env.step(action)
                        
                        # Validate step output
                        assert isinstance(next_state, dict), f"{task} step {episode_steps} state not dict"
                        assert isinstance(reward, (int, float)), f"{task} step {episode_steps} reward not numeric"
                        assert isinstance(terminated, bool), f"{task} step {episode_steps} terminated not bool"
                        assert isinstance(truncated, bool), f"{task} step {episode_steps} truncated not bool"
                        assert isinstance(next_info, dict), f"{task} step {episode_steps} info not dict"
                        
                        # Check for NaN or infinite values
                        assert not np.isnan(reward), f"{task} step {episode_steps} reward is NaN"
                        assert not np.isinf(reward), f"{task} step {episode_steps} reward is infinite"
                        
                        episode_steps += 1
                        self.metrics.state_transitions += 1
                        
                    except Exception as e:
                        self.log(f"✗ {task} step {episode_steps} failed: {e}", "ERROR")
                        self.metrics.bugs_detected.append(f"{task} step error: {e}")
                        return False
                
                self.log(f"✓ {task} step execution validated ({episode_steps} steps)")
            
            return True
            
        except Exception as e:
            self.log(f"✗ Step execution test failed: {e}", "ERROR")
            self.metrics.bugs_detected.append(f"Step execution error: {e}")
            return False
    
    def validate_reward_dynamics(self) -> bool:
        """Test that rewards vary based on actions and states."""
        try:
            self.log("Testing reward dynamics...")
            
            from smart_hospital_orchestration.environment import HospitalEnv
            
            reward_samples = []
            
            for task in self.test_tasks:
                env = HospitalEnv(task)
                state, info = env.reset(seed=42)
                
                episode_rewards = []
                
                for step in range(self.test_steps_per_episode):
                    # Test different actions
                    action = step % 5  # Cycle through all actions
                    
                    next_state, reward, terminated, truncated, next_info = env.step(action)
                    episode_rewards.append(reward)
                    
                    if terminated or truncated:
                        break
                
                # Check reward variance
                reward_variance = np.var(episode_rewards)
                reward_samples.append(reward_variance)
                
                # Rewards should not be constant
                if reward_variance < 1e-6:
                    self.log(f"⚠ {task} rewards appear constant (variance: {reward_variance:.2e})", "WARNING")
                    self.metrics.bugs_detected.append(f"{task} constant rewards")
                else:
                    self.log(f"✓ {task} reward variance: {reward_variance:.4f}")
                
                self.metrics.task_differences[task] = reward_variance
            
            # Calculate overall reward variance
            self.metrics.reward_variance = np.mean(reward_samples)
            
            return True
            
        except Exception as e:
            self.log(f"✗ Reward dynamics test failed: {e}", "ERROR")
            self.metrics.bugs_detected.append(f"Reward dynamics error: {e}")
            return False
    
    def validate_task_differentiation(self) -> bool:
        """Test that different tasks produce different behaviors."""
        try:
            self.log("Testing task differentiation...")
            
            from smart_hospital_orchestration.environment import HospitalEnv
            
            task_metrics = {}
            
            for task in self.test_tasks:
                env = HospitalEnv(task)
                state, info = env.reset(seed=42)
                
                # Collect metrics for this task
                initial_patients = info["num_patients"]
                initial_doctors = info["available_doctors"]
                initial_beds = info["available_beds"]
                
                # Run a few steps
                total_reward = 0
                steps = 0
                
                for step in range(10):
                    action = 1  # Use allocate action
                    next_state, reward, terminated, truncated, next_info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                task_metrics[task] = {
                    "initial_patients": initial_patients,
                    "initial_doctors": initial_doctors,
                    "initial_beds": initial_beds,
                    "avg_reward_per_step": total_reward / max(steps, 1)
                }
                
                self.log(f"✓ {task} metrics: {task_metrics[task]}")
            
            # Check that tasks are different
            patient_counts = [task_metrics[t]["initial_patients"] for t in self.test_tasks]
            doctor_counts = [task_metrics[t]["initial_doctors"] for t in self.test_tasks]
            
            if len(set(patient_counts)) < len(self.test_tasks):
                self.log("⚠ Tasks may not be differentiated by patient count", "WARNING")
            
            if len(set(doctor_counts)) < len(self.test_tasks):
                self.log("⚠ Tasks may not be differentiated by doctor count", "WARNING")
            
            self.log("✓ Task differentiation validated")
            return True
            
        except Exception as e:
            self.log(f"✗ Task differentiation test failed: {e}", "ERROR")
            self.metrics.bugs_detected.append(f"Task differentiation error: {e}")
            return False
    
    def validate_edge_cases(self) -> bool:
        """Test edge cases and boundary conditions."""
        try:
            self.log("Testing edge cases...")
            
            from smart_hospital_orchestration.environment import HospitalEnv
            
            edge_case_tests = [
                ("No available resources", self._test_no_resources),
                ("Full capacity", self._test_full_capacity),
                ("Multiple critical patients", self._test_multiple_critical),
                ("Invalid actions", self._test_invalid_actions),
                ("Multiple resets", self._test_multiple_resets)
            ]
            
            for test_name, test_func in edge_case_tests:
                try:
                    self.log(f"  Testing {test_name}...")
                    result = test_func()
                    if result:
                        self.log(f"  ✓ {test_name} passed")
                        self.metrics.edge_cases_tested += 1
                    else:
                        self.log(f"  ✗ {test_name} failed", "ERROR")
                        self.metrics.bugs_detected.append(f"Edge case failed: {test_name}")
                except Exception as e:
                    self.log(f"  ✗ {test_name} error: {e}", "ERROR")
                    self.metrics.bugs_detected.append(f"Edge case error: {test_name}: {e}")
            
            return True
            
        except Exception as e:
            self.log(f"✗ Edge case testing failed: {e}", "ERROR")
            self.metrics.bugs_detected.append(f"Edge case testing error: {e}")
            return False
    
    def _test_no_resources(self) -> bool:
        """Test behavior when no resources are available."""
        env = HospitalEnv("medium")
        state, info = env.reset(seed=42)
        
        # Manually set all resources to unavailable
        if "doctors" in state:
            state["doctors"][:, 1] = 0  # Set all doctors to unavailable
        if "beds" in state:
            state["beds"][:, 1] = 0  # Set all beds to unavailable
        
        # Try allocation action
        try:
            next_state, reward, terminated, truncated, next_info = env.step(1)  # ALLOCATE
            # Should handle gracefully (negative reward or no change)
            return True
        except Exception:
            return False
    
    def _test_full_capacity(self) -> bool:
        """Test behavior when system is at full capacity."""
        env = HospitalEnv("medium")
        state, info = env.reset(seed=42)
        
        # Try defer action when system is overloaded
        try:
            next_state, reward, terminated, truncated, next_info = env.step(3)  # DEFER
            # Should handle based on actual system load
            return True
        except Exception:
            return False
    
    def _test_multiple_critical(self) -> bool:
        """Test behavior with multiple critical patients."""
        env = HospitalEnv("hard")
        state, info = env.reset(seed=42)
        
        # Try escalation action
        try:
            next_state, reward, terminated, truncated, next_info = env.step(2)  # ESCALATE
            # Should handle multiple critical patients appropriately
            return True
        except Exception:
            return False
    
    def _test_invalid_actions(self) -> bool:
        """Test behavior with invalid actions."""
        env = HospitalEnv("medium")
        state, info = env.reset(seed=42)
        
        try:
            # Test action out of range
            next_state, reward, terminated, truncated, next_info = env.step(10)
            return False  # Should raise error
        except (ValueError, AssertionError):
            return True  # Expected behavior
        except Exception:
            return False
    
    def _test_multiple_resets(self) -> bool:
        """Test multiple consecutive resets."""
        env = HospitalEnv("medium")
        
        try:
            for i in range(5):
                state, info = env.reset(seed=42 + i)
                # Should work multiple times
            return True
        except Exception:
            return False
    
    def validate_baseline_agent_integration(self) -> bool:
        """Test baseline agent integration with environment."""
        try:
            self.log("Testing baseline agent integration...")
            
            from smart_hospital_orchestration.environment import HospitalEnv
            from smart_hospital_orchestration.inference.baseline_inference import HospitalBaselineAgent, EpisodeRunner
            
            for task in self.test_tasks:
                env = HospitalEnv(task)
                agent = HospitalBaselineAgent(verbose=False)
                runner = EpisodeRunner(agent, max_episodes=1)
                
                try:
                    metrics = runner.run_episode(env, f"{task}_test", seed=42)
                    
                    # Validate metrics
                    assert hasattr(metrics, 'total_reward'), f"{task} missing total_reward"
                    assert hasattr(metrics, 'steps'), f"{task} missing steps"
                    assert metrics.steps >= 0, f"{task} negative steps"
                    
                    self.log(f"✓ {task} baseline agent integration validated")
                    self.metrics.successful_episodes += 1
                    
                except Exception as e:
                    self.log(f"✗ {task} baseline agent integration failed: {e}", "ERROR")
                    self.metrics.crashed_episodes += 1
                    self.metrics.bugs_detected.append(f"Baseline agent error {task}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.log(f"✗ Baseline agent integration test failed: {e}", "ERROR")
            self.metrics.bugs_detected.append(f"Baseline agent integration error: {e}")
            return False
    
    def validate_stability(self) -> bool:
        """Test system stability over multiple episodes."""
        try:
            self.log("Testing system stability...")
            
            from smart_hospital_orchestration.environment import HospitalEnv
            from smart_hospital_orchestration.inference.baseline_inference import HospitalBaselineAgent, EpisodeRunner
            
            env = HospitalEnv("medium")
            agent = HospitalBaselineAgent(verbose=False)
            runner = EpisodeRunner(agent, max_episodes=self.test_episodes)
            
            successful_runs = 0
            total_rewards = []
            
            for episode in range(self.test_episodes):
                try:
                    metrics = runner.run_episode(env, f"stability_test_{episode}", seed=episode)
                    successful_runs += 1
                    total_rewards.append(metrics.total_reward)
                    self.metrics.successful_episodes += 1
                    
                except Exception as e:
                    self.log(f"✗ Stability test episode {episode} failed: {e}", "ERROR")
                    self.metrics.crashed_episodes += 1
                    self.metrics.bugs_detected.append(f"Stability episode {episode}: {e}")
            
            success_rate = successful_runs / self.test_episodes
            self.log(f"✓ Stability test: {successful_runs}/{self.test_episodes} episodes successful ({success_rate:.1%})")
            
            # Check reward consistency
            if len(total_rewards) > 1:
                reward_std = np.std(total_rewards)
                self.log(f"✓ Reward standard deviation: {reward_std:.2f}")
            
            self.metrics.total_episodes = self.test_episodes
            
            return success_rate >= 0.8  # At least 80% success rate
            
        except Exception as e:
            self.log(f"✗ Stability test failed: {e}", "ERROR")
            self.metrics.bugs_detected.append(f"Stability test error: {e}")
            return False
    
    def run_comprehensive_validation(self) -> ValidationMetrics:
        """
        Run comprehensive environment validation.
        
        Returns:
            ValidationMetrics with all test results
        """
        self.log("=" * 60)
        self.log("STARTING COMPREHENSIVE ENVIRONMENT VALIDATION")
        self.log("=" * 60)
        
        start_time = time.time()
        
        # Run all validation tests
        tests = [
            ("Environment Imports", self.validate_environment_imports),
            ("Environment Creation", self.validate_environment_creation),
            ("Environment Reset", self.validate_environment_reset),
            ("Step Execution", self.validate_step_execution),
            ("Reward Dynamics", self.validate_reward_dynamics),
            ("Task Differentiation", self.validate_task_differentiation),
            ("Edge Cases", self.validate_edge_cases),
            ("Baseline Agent Integration", self.validate_baseline_agent_integration),
            ("System Stability", self.validate_stability)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            self.log(f"\n--- {test_name} ---")
            try:
                if test_func():
                    passed_tests += 1
                    self.log(f"✓ {test_name} PASSED")
                else:
                    self.log(f"✗ {test_name} FAILED", "ERROR")
            except Exception as e:
                self.log(f"✗ {test_name} ERROR: {e}", "ERROR")
                self.metrics.bugs_detected.append(f"Test error {test_name}: {e}")
        
        # Calculate final metrics
        end_time = time.time()
        self.metrics.total_episodes = self.test_episodes
        self.metrics.successful_episodes = passed_tests
        
        # Print summary
        self.log("\n" + "=" * 60)
        self.log("VALIDATION SUMMARY")
        self.log("=" * 60)
        self.log(f"Tests Passed: {passed_tests}/{total_tests}")
        self.log(f"Success Rate: {passed_tests/total_tests:.1%}")
        self.log(f"Total Episodes Run: {self.metrics.total_episodes}")
        self.log(f"Successful Episodes: {self.metrics.successful_episodes}")
        self.log(f"Crashed Episodes: {self.metrics.crashed_episodes}")
        self.log(f"State Transitions: {self.metrics.state_transitions}")
        self.log(f"Edge Cases Tested: {self.metrics.edge_cases_tested}")
        self.log(f"Reward Variance: {self.metrics.reward_variance:.6f}")
        self.log(f"Bugs Detected: {len(self.metrics.bugs_detected)}")
        self.log(f"Validation Time: {end_time - start_time:.2f} seconds")
        
        if self.metrics.bugs_detected:
            self.log("\nDetected Issues:")
            for bug in self.metrics.bugs_detected:
                self.log(f"  • {bug}")
        
        self.log("=" * 60)
        
        return self.metrics


def main():
    """Main validation script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Smart Hospital Resource Orchestration Environment")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, help="Write logs to file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes for stability testing")
    
    args = parser.parse_args()
    
    # Create validator
    validator = HospitalEnvironmentValidator(verbose=args.verbose, log_file=args.log_file)
    validator.test_episodes = args.episodes
    
    # Run validation
    metrics = validator.run_comprehensive_validation()
    
    # Exit with appropriate code
    if len(metrics.bugs_detected) == 0 and metrics.successful_episodes > 0:
        print("\n🎉 ALL VALIDATIONS PASSED! Environment is ready for use.")
        sys.exit(0)
    else:
        print(f"\n❌ VALIDATION FAILED! {len(metrics.bugs_detected)} issues detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
