"""
Web Interface for Smart Hospital Resource Orchestration
Interactive dashboard for running and monitoring hospital simulation
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import json
import subprocess
import importlib.util
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import logging
import requests
from functools import lru_cache
import time

# NOTE:
# This module is intended to be runnable as:
#   - `py -m smart_hospital_orchestration.web_interface` (preferred)
# and also tolerates direct execution in some setups.

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Preferred when executed as a package module.
    from .environment.hospital_env import HospitalEnv
    from .inference.baseline_inference import HospitalBaselineAgent, EpisodeRunner
    from .tasks.advanced_tasks import easy, medium, hard
    from .reward.advanced_reward import create_reward_calculator
except Exception:
    try:
        # Fallback for direct execution (e.g. `python smart_hospital_orchestration/web_interface.py`).
        from smart_hospital_orchestration.environment.hospital_env import HospitalEnv

        # Avoid issues where root-level `inference.py` shadows `smart_hospital_orchestration/inference/`.
        from smart_hospital_orchestration.inference.baseline_inference import (
            HospitalBaselineAgent,
            EpisodeRunner,
        )
        from smart_hospital_orchestration.tasks.advanced_tasks import easy, medium, hard
        from smart_hospital_orchestration.reward.advanced_reward import create_reward_calculator
    except ImportError as e:
        logger.error(f"Import error: {e}")
        sys.exit(1)

app = Flask(__name__)


def _load_local_env_file() -> None:
    """Load key=value pairs from the workspace .env file if they are not already set."""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, 'r', encoding='utf-8') as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        logger.warning(f"Unable to load local .env file: {exc}")


_load_local_env_file()

class EnvironmentManager:
    """
    Robust environment lifecycle manager for hospital orchestration system.
    Handles creation, initialization, state management, and cleanup.
    """
    
    def __init__(self):
        self.env: Optional[HospitalEnv] = None
        self.agent: Optional[HospitalBaselineAgent] = None
        self.current_task: Optional[str] = None
        self.is_initialized: bool = False
        self.episode_count: int = 0
        self.last_activity: Optional[datetime] = None

        # --- Time-Travel Replay (server-side timeline frames) ---
        # Stores lightweight snapshots so UI can scrub through time.
        self._timeline: List[Dict[str, Any]] = []
        self._cumulative_reward: float = 0.0

        # --- Drone Fleet (demo feature; managed server-side so it works across clients/tabs) ---
        self._drone_fleet: List[Dict[str, Any]] = self._default_drone_fleet()

        # --- AI Lab tuning (server-side source of truth) ---
        self.ai_lab_tuning: Dict[str, Any] = {
            "emergency_weight": 1.0,
            "efficiency_weight": 1.0,
            "profile": "Balanced",
            "updated_at": None,
        }

    @staticmethod
    def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
        try:
            v = float(value)
        except Exception:
            v = float(default)
        return float(max(lo, min(hi, v)))

    @staticmethod
    def _profile_from_tuning(emergency_weight: float, efficiency_weight: float) -> str:
        if emergency_weight >= 1.6 and efficiency_weight <= 1.1:
            return "Aggressive Triage"
        if efficiency_weight >= 1.6 and emergency_weight <= 1.1:
            return "Efficiency First"
        if emergency_weight >= 1.3 and efficiency_weight >= 1.3:
            return "High Response"
        if emergency_weight <= 0.9 and efficiency_weight <= 0.9:
            return "Conservative"
        return "Balanced"

    def set_ai_lab_tuning(self, emergency_weight: Any, efficiency_weight: Any) -> Dict[str, Any]:
        ew = self._clamp_float(emergency_weight, 0.5, 3.0, 1.0)
        rw = self._clamp_float(efficiency_weight, 0.5, 3.0, 1.0)
        profile = self._profile_from_tuning(ew, rw)
        self.ai_lab_tuning = {
            "emergency_weight": round(ew, 2),
            "efficiency_weight": round(rw, 2),
            "profile": profile,
            "updated_at": datetime.now().isoformat(),
        }
        return dict(self.ai_lab_tuning)

    def get_ai_lab_tuning(self) -> Dict[str, Any]:
        return dict(self.ai_lab_tuning)

    def _reset_timeline(self) -> None:
        self._timeline = []
        self._cumulative_reward = 0.0

    def _append_timeline_frame(
        self,
        *,
        event: str,
        source: str,
        action_name: Optional[str] = None,
        reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a snapshot frame for time-travel replay.

        Frame schema (stable):
          - ts, event, source
          - state: same keys as /api/get_state
          - action_name, reward, cumulative_reward
          - info: step info (if any)
          - events: derived notable transitions
        """

        try:
            ok, state = self.get_current_state()
            if not ok:
                return

            # Derived event detection based on previous frame.
            derived: List[str] = []
            try:
                prev_state = (
                    self._timeline[-1].get("state") if self._timeline else None
                )
                if isinstance(prev_state, dict):
                    if bool(state.get("crisis_active")) and not bool(prev_state.get("crisis_active")):
                        derived.append("CRISIS_ON")
                    if int(state.get("red_waiting", 0)) > int(prev_state.get("red_waiting", 0)):
                        derived.append("NEW_RED_WAITING")
                    if int(state.get("waiting", 0)) > int(prev_state.get("waiting", 0)):
                        derived.append("QUEUE_GROWING")
            except Exception:
                derived = []

            frame: Dict[str, Any] = {
                "ts": datetime.now().isoformat(),
                "event": str(event),
                "source": str(source),
                "state": state,
                "action_name": str(action_name) if action_name is not None else None,
                "reward": float(reward) if reward is not None else None,
                "cumulative_reward": float(self._cumulative_reward),
                "info": info if isinstance(info, dict) else ({} if info is None else {"raw": info}),
                "events": derived,
            }

            self._timeline.append(frame)

            # Prevent unbounded growth in long-running demos.
            if len(self._timeline) > 600:
                self._timeline = self._timeline[-600:]
        except Exception:
            # Timeline must never break the simulation.
            return

    @staticmethod
    def _default_drone_fleet() -> List[Dict[str, Any]]:
        return [
            {"name": "Drone Alpha", "status": "IDLE", "package": None, "dispatch_step": None},
            {"name": "Drone Bravo", "status": "IDLE", "package": None, "dispatch_step": None},
            {"name": "Drone Charlie", "status": "IDLE", "package": None, "dispatch_step": None},
            {"name": "Drone Delta", "status": "IDLE", "package": None, "dispatch_step": None},
        ]

    def _reset_drone_fleet(self) -> None:
        self._drone_fleet = self._default_drone_fleet()

    def _tick_drone_fleet(self, current_step: int) -> None:
        """Advance drone statuses based on how long they've been dispatched.

        Simple lifecycle:
          DISPATCHED (0-2 steps) -> RETURNING (3-4 steps) -> IDLE (>=5 steps)
        """
        for d in self._drone_fleet:
            st = str(d.get("status", "IDLE"))
            ds = d.get("dispatch_step")
            if st not in ["DISPATCHED", "RETURNING"]:
                continue
            if ds is None:
                continue
            elapsed = int(current_step) - int(ds)
            if elapsed >= 5:
                d["status"] = "IDLE"
                d["package"] = None
                d["dispatch_step"] = None
            elif elapsed >= 3:
                d["status"] = "RETURNING"

    def _can_dispatch_drone(self, readable: Dict[str, Any]) -> bool:
        beds_available = int(readable.get("available_beds", 0))
        red_waiting = int(readable.get("red_waiting", 0))
        return beds_available <= 0 and red_waiting > 0

    def dispatch_drone(
        self,
        drone_name: Optional[str] = None,
        package: str = "Medical Kit (Red)",
        mode: str = "manual",
    ) -> Tuple[bool, Dict[str, Any]]:
        """Dispatch an emergency drone if conditions are met.

        Conditions:
          - beds_available == 0 AND red_waiting > 0

        mode:
          - 'manual' or 'autopilot' (autopilot can earn bonus reward in /api/ai_action)
        """
        try:
            if not self.ensure_initialized():
                return False, {"error": "Failed to initialize environment"}
            if not self.env:
                return False, {"error": "Environment not available"}

            readable = self.env.state().get("readable", {})
            if not self._can_dispatch_drone(readable):
                return False, {"error": "Dispatch conditions not met"}

            # Update lifecycle first so we don't block on stale RETURNING state.
            self._tick_drone_fleet(int(readable.get("step", readable.get("current_step", 0)) or 0))

            # Choose drone.
            chosen: Optional[Dict[str, Any]] = None
            if drone_name:
                for d in self._drone_fleet:
                    if d.get("name") == drone_name and d.get("status") == "IDLE":
                        chosen = d
                        break
            if chosen is None:
                for d in self._drone_fleet:
                    if d.get("status") == "IDLE":
                        chosen = d
                        break

            if chosen is None:
                return False, {"error": "No idle drones available"}

            step_now = int(readable.get("step", 0))
            chosen["status"] = "DISPATCHED"
            chosen["package"] = str(package)
            chosen["dispatch_step"] = step_now

            return True, {
                "mode": str(mode),
                "drone": str(chosen.get("name")),
                "status": str(chosen.get("status")),
                "package": str(chosen.get("package")),
                "dispatch_step": int(step_now),
            }
        except Exception as e:
            logger.error(f"Failed to dispatch drone: {e}")
            return False, {"error": str(e)}
        
    def create_environment(self, task: str, seed: int = 42) -> bool:
        """
        Create and initialize environment with proper lifecycle management.
        
        Args:
            task: Task difficulty level
            seed: Random seed for reproducibility
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Creating environment for task: {task}")
            
            # Clean up any existing environment
            self.cleanup()

            # Reset demo subsystems
            self._reset_drone_fleet()

            # Reset timeline
            self._reset_timeline()
            
            # Create new environment
            self.env = HospitalEnv(task)
            self.current_task = task
            
            # Initialize environment state
            state, info = self.env.reset(seed=seed)

            # Initial timeline frame
            self._append_timeline_frame(event="INIT", source="system")
            
            # Create baseline agent
            self.agent = HospitalBaselineAgent(verbose=False)
            
            # Mark as initialized
            self.is_initialized = True
            self.last_activity = datetime.now()
            # Count the freshly initialized simulation as the first episode.
            # This keeps the UI from showing 0 after a successful init.
            self.episode_count = 1
            
            logger.info(f"Environment successfully created and initialized")
            logger.info(f"Initial state: {info.get('num_patients', 0)} patients, "
                       f"{info.get('available_doctors', 0)} doctors, "
                       f"{info.get('available_beds', 0)} beds")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            self.cleanup()
            return False
    
    def ensure_initialized(self, task: str = "medium") -> bool:
        """
        Ensure environment is initialized, auto-initialize if needed.
        
        Args:
            task: Task difficulty level (used for auto-initialization)
            
        Returns:
            True if environment is ready, False otherwise
        """
        if self.is_initialized and self.env is not None and self.agent is not None:
            logger.info("Environment already initialized")
            return True
        
        logger.info("Environment not initialized, auto-initializing...")
        return self.create_environment(task, seed=42)
    
    def reset_environment(self, seed: int = 42) -> Tuple[bool, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (success, info_dict)
        """
        try:
            if not self.env:
                logger.error("Cannot reset: No environment exists")
                return False, {"error": "No environment exists"}
            
            logger.info("Resetting environment...")
            state, info = self.env.reset(seed=seed)

            self._reset_drone_fleet()

            self._reset_timeline()
            self._append_timeline_frame(event="RESET", source="user")
            
            self.is_initialized = True
            self.last_activity = datetime.now()
            self.episode_count += 1
            
            logger.info(f"Environment reset successfully. Episode {self.episode_count}")
            return True, info
            
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            return False, {"error": str(e)}
    
    def execute_step(self, action: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a single step in the environment.
        
        Args:
            action: Action to execute (0-4)
            
        Returns:
            Tuple of (success, step_result)
        """
        try:
            # Ensure environment is ready
            if not self.ensure_initialized():
                return False, {"error": "Failed to initialize environment"}
            
            if not self.env or not self.agent:
                return False, {"error": "Environment or agent not available"}
            
            logger.info(f"Executing step with action: {action}")
            
            # Execute step
            next_state, reward, done, info = self.env.step(action)

            # Update timeline cumulative reward and append a frame.
            self._cumulative_reward += float(reward)
            self._append_timeline_frame(
                event="STEP",
                source="manual",
                action_name=['WAIT', 'ALLOCATE', 'ESCALATE', 'DEFER', 'REASSIGN'][action],
                reward=float(reward),
                info=info if isinstance(info, dict) else {},
            )
            
            self.last_activity = datetime.now()
            
            result = {
                "action": action,
                "action_name": ['WAIT', 'ALLOCATE', 'ESCALATE', 'DEFER', 'REASSIGN'][action],
                "reward": reward,
                "terminated": info.get("terminated", False),
                "truncated": info.get("truncated", False),
                "done": done,
                "info": info,
                "total_steps": info.get("step", 0),  # Use step from environment info
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Step executed: {result['action_name']} -> reward: {reward:.2f}")
            return True, result
            
        except Exception as e:
            logger.error(f"Failed to execute step: {e}")
            return False, {"error": str(e)}

    def trigger_resource_crisis(self, lock_ratio: float = 0.50, seed: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """Trigger a simulated resource crisis by locking a portion of currently available beds."""
        try:
            if not self.ensure_initialized():
                return False, {"error": "Failed to initialize environment"}

            if not self.env:
                return False, {"error": "Environment not available"}

            details = self.env.trigger_resource_crisis(lock_ratio=lock_ratio, seed=seed)  # type: ignore[attr-defined]
            self._append_timeline_frame(event="CRISIS_TRIGGERED", source="user", info=details)
            return True, details
        except Exception as e:
            logger.error(f"Failed to trigger resource crisis: {e}")
            return False, {"error": str(e)}

    def _heuristic_action(self) -> int:
        """Fallback AI policy for showcase when no model prediction is available."""
        if not self.env:
            return 0

        tuning = self.get_ai_lab_tuning()
        emergency_weight = float(tuning.get("emergency_weight", 1.0))
        efficiency_weight = float(tuning.get("efficiency_weight", 1.0))

        state = self.env.state().get("readable", {})
        waiting = int(state.get("waiting", 0))
        available_doctors = int(state.get("available_doctors", 0))
        available_beds = int(state.get("available_beds", 0))
        city_available_beds = int(state.get("city_available_beds", 0))
        red_waiting = int(state.get("red_waiting", 0))
        green_waiting = int(state.get("green_waiting", 0))

        # Optional richer constraints if available.
        constraints = {}
        try:
            constraints = self.env._check_current_constraints()  # type: ignore[attr-defined]
        except Exception:
            constraints = {}

        critical_waiting = int(state.get("critical_waiting", constraints.get("critical_waiting", 0)))
        emergency_waiting = int(state.get("emergency_waiting", constraints.get("emergency_waiting", 0)))
        overloaded = bool(constraints.get("overloaded", waiting > 10))
        has_normal_waiting = int(constraints.get("normal_waiting", 0)) > 0

        # City-first load balancing: if local beds are full but external capacity exists.
        if waiting > 0 and available_beds <= 0 and city_available_beds > 0:
            # Efficiency-heavy tuning prefers reassignment over escalation.
            if efficiency_weight >= 1.0:
                return 4  # REASSIGN

        escalate_threshold = max(6, int(round(12 - (emergency_weight - 1.0) * 4)))

        # Emergency-heavy tuning escalates sooner under critical pressure.
        if red_waiting > 0 and waiting >= escalate_threshold and (available_doctors <= 0 or available_beds <= 0):
            return 2  # ESCALATE

        # Efficiency-heavy tuning de-prioritizes escalation when non-critical queues dominate.
        if green_waiting > red_waiting and efficiency_weight >= 1.4 and available_beds <= 0 and city_available_beds <= 0:
            return 3  # DEFER

        # Emergency-heavy tuning reassigns if city network can absorb RED backlog.
        if red_waiting > 0 and city_available_beds > 0 and available_beds <= 0 and emergency_weight >= 1.2:
            return 4  # REASSIGN

        # 1) Strong ALLOCATE priority only when strictly feasible.
        # Prefer RED triage first in allocation-capable states.
        if waiting > 0 and available_doctors > 0 and available_beds > 0:
            return 1  # ALLOCATE

        # 2) Never ALLOCATE when resources are missing.
        if available_doctors <= 0 or available_beds <= 0:
            # 3) Smart waiting/deferral when resources are blocked.
            if overloaded and has_normal_waiting:
                return 3  # DEFER

            # 4) ESCALATE only at very high pressure / last resort.
            if red_waiting >= 2 and waiting > 10:
                return 2  # ESCALATE (last resort under heavy RED pressure)
            if waiting > 12 or critical_waiting >= 3 or emergency_waiting >= 6:
                return 2  # ESCALATE

            return 0  # WAIT

        return 0  # WAIT

    def execute_ai_step(self) -> Tuple[bool, Dict[str, Any]]:
        """Choose action using model/heuristic and execute one environment step."""
        try:
            if not self.ensure_initialized():
                return False, {"error": "Failed to initialize environment"}

            if not self.env:
                return False, {"error": "Environment not available"}

            # --- Autonomous Drone Fleet (auto-dispatch during AI autopilot) ---
            drone_bonus = 0.0
            drone_details: Optional[Dict[str, Any]] = None
            try:
                readable = self.env.state().get("readable", {})
                if self._can_dispatch_drone(readable):
                    ok_d, det = self.dispatch_drone(drone_name="Drone Alpha", package="Medical Kit (Red)", mode="autopilot")
                    if ok_d:
                        drone_details = det
                        drone_bonus = 8.0  # Strategic bonus reward
            except Exception:
                drone_bonus = 0.0
                drone_details = None

            chosen_action: Optional[int] = None

            # Try model-based decision first.
            try:
                if self.agent is not None:
                    agent_action = self.agent.select_action(self.env.state())
                    if isinstance(agent_action, (int, np.integer)) and int(agent_action) in [0, 1, 2, 3, 4]:
                        chosen_action = int(agent_action)
            except Exception:
                chosen_action = None

            if chosen_action is None:
                chosen_action = self._heuristic_action()

            # Safety override to avoid invalid/low-value model choices in demo mode.
            try:
                if chosen_action == 1:
                    constraints = self.env._check_current_constraints()  # type: ignore[attr-defined]
                    if constraints.get("available_doctors", 0) <= 0 or constraints.get("available_beds", 0) <= 0:
                        chosen_action = self._heuristic_action()
                elif chosen_action == 2:
                    constraints = self.env._check_current_constraints()  # type: ignore[attr-defined]
                    if int(constraints.get("waiting_patients", 0)) <= 10:
                        chosen_action = self._heuristic_action()
            except Exception:
                chosen_action = self._heuristic_action()

            ok, result = self.execute_step(chosen_action)
            if not ok:
                return False, result

            # Apply strategic drone bonus to AI action when auto-dispatch happened.
            if drone_bonus > 0:
                result["reward"] = float(result.get("reward", 0.0)) + float(drone_bonus)
                info = result.get("info") if isinstance(result.get("info"), dict) else {}
                info["drone_dispatched"] = (drone_details or {}).get("drone")
                info["drone_bonus_reward"] = float(drone_bonus)
                info["drone_package"] = (drone_details or {}).get("package")
                result["info"] = info

                # Keep replay totals consistent: the STEP frame already added base reward,
                # so we add a separate bonus frame and update cumulative.
                self._cumulative_reward += float(drone_bonus)
                self._append_timeline_frame(
                    event="DRONE_BONUS",
                    source="autopilot",
                    action_name="DRONE_DISPATCH",
                    reward=float(drone_bonus),
                    info={
                        "drone": (drone_details or {}).get("drone"),
                        "package": (drone_details or {}).get("package"),
                        "bonus": float(drone_bonus),
                    },
                )

            # Adjust timeline frame (latest STEP frame was recorded as manual by execute_step).
            # Add an AI marker frame so the replay timeline shows autopilot boundaries.
            try:
                self._append_timeline_frame(
                    event="AI_STEP",
                    source="autopilot",
                    action_name=result.get("ai_action_name"),
                    reward=None,
                    info=result.get("info") if isinstance(result.get("info"), dict) else {},
                )
            except Exception:
                pass

            result["ai_action"] = chosen_action
            result["ai_action_name"] = ['WAIT', 'ALLOCATE', 'ESCALATE', 'DEFER', 'REASSIGN'][chosen_action]
            return True, result

        except Exception as e:
            logger.error(f"Failed to execute AI step: {e}")
            return False, {"error": str(e)}
    
    def run_complete_episode(self, max_steps: int = 50) -> Tuple[bool, Dict[str, Any]]:
        """
        Run a complete episode with baseline agent.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (success, episode_result)
        """
        try:
            # Ensure environment is ready
            if not self.ensure_initialized():
                return False, {"error": "Failed to initialize environment"}
            
            logger.info(f"Starting complete episode run (max steps: {max_steps})")
            
            # Reset environment
            success, info = self.reset_environment(seed=42)
            if not success:
                return False, {"error": "Failed to reset environment"}
            
            # Run episode
            terminated = False
            truncated = False
            done = False
            total_reward = 0
            steps = 0
            episode_history = []
            
            while not done and steps < max_steps:
                # Get current state
                state_result = self.env.state()
                state = state_result if isinstance(state_result, dict) else state_result[0]
                
                # Agent selects action
                action = self.agent.select_action(state)
                
                # Execute step
                success, step_result = self.execute_step(action)
                if not success:
                    return False, {"error": f"Step execution failed: {step_result.get('error')}"}
                
                # Extract done flag from step result
                done = step_result.get("done", False)
                terminated = step_result.get("terminated", False)
                truncated = step_result.get("truncated", False)
                
                # Record step
                episode_history.append({
                    "step": steps + 1,
                    "action": action,
                    "action_name": step_result["action_name"],
                    "reward": step_result["reward"],
                    "terminated": terminated,
                    "truncated": truncated,
                    "done": done,
                    "info": step_result["info"]
                })
                
                total_reward += step_result["reward"]
                terminated = step_result["terminated"]
                truncated = step_result["truncated"]
                done = step_result["done"]
                steps += 1
            
            # Build final_info with correct field names for frontend
            final_info = {
                "num_patients": info.get('num_patients', 0),
                "waiting_patients": info.get('waiting_patients', 0),
                "admitted_patients": info.get('admitted_patients', 0),
                "available_doctors": info.get('available_doctors', 0),
                "available_beds": info.get('available_beds', 0),
                "step": info.get('step', 0)
            }
            
            result = {
                "total_reward": total_reward,
                "steps": steps,
                "terminated": terminated,
                "truncated": truncated,
                "final_info": final_info,  # Use the properly structured final_info
                "episode_history": episode_history[-10:],  # Last 10 steps
                "success": True
            }
            
            logger.info(f"Episode completed: {steps} steps, total reward: {total_reward:.2f}")
            return True, result
            
        except Exception as e:
            logger.error(f"Failed to run complete episode: {e}")
            return False, {"error": str(e)}
    
    def get_current_state(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Get current environment state with proper value extraction.
        
        Returns:
            Tuple of (success, state_dict)
        """
        try:
            if not self.env:
                return False, {"error": "Environment not initialized"}
            
            # Get state from environment
            state_result = self.env.state()
            
            # Handle dictionary return (expected format)
            if isinstance(state_result, dict):
                state = state_result
                info = state.get("readable", {})
            else:
                logger.error(f"Unexpected state result: {state_result}")
                return False, {"error": f"Unexpected state format: {type(state_result)}"}
            
            # Extract values with proper conversion (NumPy → Python)
            total_patients = int(info.get('total_patients', 0))
            waiting = int(info.get('waiting', 0))
            admitted = int(info.get('admitted', 0))
            available_doctors = int(info.get('available_doctors', 0))
            available_beds = int(info.get('available_beds', 0))
            system_load = float(info.get('system_load', 0.0))
            step = int(info.get('step', 0))
            red_waiting = int(info.get('red_waiting', 0))
            green_waiting = int(info.get('green_waiting', 0))
            red_admitted = int(info.get('red_admitted', 0))
            green_admitted = int(info.get('green_admitted', 0))

            waiting_triage = []
            avg_wait_time = 0.0
            red_treated_total = 0
            if self.env:
                waiting_patients = [
                    p for p in self.env.patients
                    if getattr(p, 'status', None) and str(p.status.name) == 'WAITING'
                ]
                waiting_patients.sort(
                    key=lambda p: (
                        int(getattr(p, 'wait_time', 0)),
                        str(getattr(p, 'priority', None).name) == 'RED',
                        int(getattr(getattr(p, 'severity', None), 'value', 0)),
                    ),
                    reverse=True,
                )

                waiting_triage = [
                    {
                        "patient_id": int(getattr(p, 'patient_id', -1)),
                        "priority": str(getattr(getattr(p, 'priority', None), 'name', 'GREEN')),
                        "severity": str(getattr(getattr(p, 'severity', None), 'name', 'NORMAL')),
                        "wait_time": int(getattr(p, 'wait_time', 0)),
                    }
                    for p in waiting_patients[:8]
                ]

                avg_wait_time = (
                    float(sum(int(getattr(p, 'wait_time', 0)) for p in waiting_patients)) / len(waiting_patients)
                    if waiting_patients else 0.0
                )
                red_treated_total = int(getattr(self.env, 'total_admissions', 0)) if red_admitted > 0 else 0
            
            # Build clean state dict with exact keys frontend expects
            state_dict = {
                "total_patients": total_patients,
                "waiting": waiting,
                "admitted": admitted,
                "available_doctors": available_doctors,
                "available_beds": available_beds,
                "total_doctors": int(len(self.env.doctors)) if self.env else 0,
                "total_beds": int(len(self.env.beds)) if self.env else 0,
                "system_load": system_load,
                "step": step,
                "red_waiting": red_waiting,
                "green_waiting": green_waiting,
                "red_admitted": red_admitted,
                "green_admitted": green_admitted,
                "avg_wait_time": round(avg_wait_time, 2),
                "red_treated_total": red_treated_total,
                "waiting_triage": waiting_triage,
                "city_available_beds": int(info.get('city_available_beds', 0)),
                "city_hospitals": info.get('city_hospitals', []),
                "queue_soft_limit": int(info.get('queue_soft_limit', 12)),
                "blood_inventory": info.get('blood_inventory', {}),
                # Crisis simulator
                "crisis_active": bool(info.get("crisis_active", False)),
                "locked_beds": int(info.get("locked_beds", 0)),
                "city_wide_alert": info.get("city_wide_alert"),
                # AI Suggestion Box (backend source of truth)
                "ai_suggestion_action": str(info.get("ai_suggestion_action")) if info.get("ai_suggestion_action") is not None else None,
                "ai_suggestion_reason": str(info.get("ai_suggestion_reason")) if info.get("ai_suggestion_reason") is not None else None,
                "ai_suggestion_priority": int(info.get("ai_suggestion_priority")) if info.get("ai_suggestion_priority") is not None else None,
                "alert_explanation": str(info.get("alert_explanation")) if info.get("alert_explanation") is not None else None,
                "episode_count": int(self.episode_count),
                "is_initialized": bool(self.is_initialized),
                "current_task": str(self.current_task) if self.current_task else "none"
            }
            
            # Debug logging
            logger.info(f"Sending state: {state_dict}")
            
            return True, state_dict
            
        except Exception as e:
            logger.error(f"Failed to get current state: {e}")
            return False, {"error": str(e)}
    
    def cleanup(self):
        """Clean up environment and reset state."""
        try:
            if self.env:
                logger.info("Cleaning up environment...")
                self.env = None
            
            if self.agent:
                self.agent = None
            
            self.is_initialized = False
            self.current_task = None
            self.last_activity = None
            
            logger.info("Environment cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Global environment manager
env_manager = EnvironmentManager()

# Episode history for web display
episode_history = []

@app.route('/')
def index():
    """Overview dashboard page"""
    return render_template('overview.html')


@app.route('/controls')
def controls_page():
    """Simulation controls page"""
    return render_template('controls.html')


@app.route('/analytics')
def analytics_page():
    """Analytics and charts page"""
    return render_template('analytics.html')


@app.route('/ai_lab')
def ai_lab_page():
    """AI Lab page (tuning + benchmarks + status)."""
    return render_template('ai_lab.html')


@app.route('/api/ai_lab/tuning', methods=['GET', 'POST'])
def ai_lab_tuning_api():
    """Get or set AI Lab tuning values persisted on server."""
    try:
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'tuning': env_manager.get_ai_lab_tuning(),
            })

        data = request.get_json(silent=True) or {}
        emergency_weight = data.get('emergency_weight', data.get('emergencyWeight', 1.0))
        efficiency_weight = data.get('efficiency_weight', data.get('efficiencyWeight', 1.0))
        tuning = env_manager.set_ai_lab_tuning(emergency_weight, efficiency_weight)

        return jsonify({
            'success': True,
            'message': 'AI tuning applied to server policy heuristics',
            'tuning': tuning,
        })
    except Exception as e:
        logger.error(f"AI Lab tuning API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai_lab/preview', methods=['POST'])
def ai_lab_preview_api():
    """Return a tuned recommendation preview for the current state without stepping the env."""
    try:
        if not env_manager.ensure_initialized():
            return jsonify({'success': False, 'error': 'Failed to initialize environment'}), 500

        success, state = env_manager.get_current_state()
        if not success:
            return jsonify({'success': False, 'error': state.get('error', 'State unavailable')}), 500

        data = request.get_json(silent=True) or {}
        emergency_weight = EnvironmentManager._clamp_float(
            data.get('emergency_weight', data.get('emergencyWeight', 1.0)),
            0.5,
            3.0,
            1.0,
        )
        efficiency_weight = EnvironmentManager._clamp_float(
            data.get('efficiency_weight', data.get('efficiencyWeight', 1.0)),
            0.5,
            3.0,
            1.0,
        )

        waiting = int(state.get('waiting', 0))
        red_waiting = int(state.get('red_waiting', 0))
        available_doctors = int(state.get('available_doctors', 0))
        available_beds = int(state.get('available_beds', 0))
        city_available_beds = int(state.get('city_available_beds', 0))

        critical_pressure = red_waiting * emergency_weight
        resource_capacity = (available_doctors + available_beds + city_available_beds) * efficiency_weight

        if waiting <= 0:
            action = 'HOLD'
            reason = 'No patients waiting in queue.'
        elif available_doctors > 0 and available_beds > 0:
            action = 'URGENT ALLOCATE' if red_waiting > 0 and emergency_weight >= 1.1 else 'ALLOCATE'
            reason = 'Local doctor and bed capacity available for immediate admissions.'
        elif available_beds <= 0 and city_available_beds > 0 and efficiency_weight >= 1.0:
            action = 'REASSIGN'
            reason = 'Local beds are full; city network capacity is available.'
        elif red_waiting > 0 and emergency_weight >= 1.2:
            action = 'ESCALATE'
            reason = 'Critical queue pressure is high under emergency-biased policy.'
        else:
            action = 'DEFER'
            reason = 'Capacity constraints detected; defer non-critical flow temporarily.'

        return jsonify({
            'success': True,
            'preview': {
                'action': action,
                'reason': reason,
                'critical_pressure': round(float(critical_pressure), 2),
                'resource_capacity': round(float(resource_capacity), 2),
                'profile': EnvironmentManager._profile_from_tuning(emergency_weight, efficiency_weight),
            },
            'state_snapshot': {
                'waiting': waiting,
                'red_waiting': red_waiting,
                'available_doctors': available_doctors,
                'available_beds': available_beds,
                'city_available_beds': city_available_beds,
            }
        })
    except Exception as e:
        logger.error(f"AI Lab preview API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/3d_view')
def three_d_view_page():
    """3D Hospital Visualization page."""
    return render_template('3d_view.html')


@app.route('/clinical_ops')
def clinical_ops_page():
    """Clinical Ops page (specialized hospital services)."""
    return render_template('clinical_ops.html')


@app.route('/api/bonus_reward', methods=['POST'])
def bonus_reward():
    """Apply a small reward bonus used by non-simulation modules (e.g., Clinical Ops).

    This intentionally affects the global episode_history aggregation so the
    header Efficiency Score (tied to /api/episode_history) updates.
    """
    try:
        data = request.get_json(silent=True) or {}
        amount = float(data.get('amount', 0.0))
        reason = str(data.get('reason', 'CLINICAL_OPS_BONUS'))

        # Clamp to prevent abuse in demos.
        amount = max(0.0, min(100.0, amount))

        episode_history.append({
            'action': 'CLINICAL_OPS_BONUS',
            'action_name': reason,
            'reward': amount,
            'timestamp': datetime.now().isoformat(),
        })

        return jsonify({'success': True, 'amount': amount, 'reason': reason})
    except Exception as e:
        logger.error(f"Error applying bonus reward: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status')
def get_status():
    """Get current system status"""
    try:
        success, state = env_manager.get_current_state()
        
        return jsonify({
            'status': 'running',
            'environment': env_manager.env is not None,
            'agent': env_manager.agent is not None,
            'episode_count': env_manager.episode_count,
            'is_initialized': env_manager.is_initialized,
            'current_task': env_manager.current_task,
            'last_activity': env_manager.last_activity.isoformat() if env_manager.last_activity else None,
            'timestamp': datetime.now().isoformat(),
            'state': state if success else None
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            'status': 'error',
            'is_initialized': False,
            'environment': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/init', methods=['POST'])
def initialize_environment():
    """Initialize environment with specified task"""
    try:
        data = request.get_json()
        task = data.get('task', 'medium')
        seed = data.get('seed', 42)
        
        logger.info(f"Initializing environment: task={task}, seed={seed}")
        
        success = env_manager.create_environment(task, seed)
        
        if success:
            # Get initial state
            success_state, state_info = env_manager.get_current_state()
            
            return jsonify({
                'success': True,
                'task': task,
                'seed': seed,
                'initial_state': state_info if success_state else {},
                'message': f'{task} environment initialized successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to initialize environment',
                'message': 'Environment creation failed'
            })
            
    except Exception as e:
        logger.error(f"Error initializing environment: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Initialization failed due to error'
        })

@app.route('/api/step', methods=['POST'])
def execute_step():
    """Execute one step in the environment"""
    global episode_history
    
    try:
        data = request.get_json()
        action = data.get('action', 0)  # Default to WAIT
        
        # Validate action
        if action not in [0, 1, 2, 3, 4]:
            return jsonify({
                'success': False,
                'error': 'Invalid action',
                'message': 'Action must be 0-4 (WAIT, ALLOCATE, ESCALATE, DEFER, REASSIGN)'
            })
        
        logger.info(f"Executing step: action={action}")
        
        # Execute step
        success, step_result = env_manager.execute_step(action)
        
        if success:
            # Record step in history
            episode_history.append(step_result)
            
            # Calculate cumulative reward
            total_reward = sum(h['reward'] for h in episode_history)
            
            return jsonify({
                'success': True,
                'step': step_result,
                'total_reward': total_reward,
                'episode_length': len(episode_history),
                'invalid_action': bool(step_result.get('info', {}).get('invalid_allocation', False)),
                'error': step_result.get('info', {}).get('error'),
                'message': f"Step executed: {step_result['action_name']}"
            })
        else:
            return jsonify({
                'success': False,
                'error': step_result.get('error', 'Unknown error'),
                'message': 'Step execution failed'
            })
            
    except Exception as e:
        logger.error(f"Error executing step: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Step execution failed due to error'
        })


@app.route('/api/ai_action', methods=['POST'])
def execute_ai_action():
    """Execute one AI-selected action step (model or heuristic)."""
    global episode_history

    try:
        logger.info("Executing AI action step")
        success, step_result = env_manager.execute_ai_step()

        if success:
            episode_history.append(step_result)
            total_reward = sum(h['reward'] for h in episode_history)

            return jsonify({
                'success': True,
                'step': step_result,
                'ai_action': step_result.get('ai_action'),
                'ai_action_name': step_result.get('ai_action_name'),
                'total_reward': total_reward,
                'episode_length': len(episode_history),
                'invalid_action': bool(step_result.get('info', {}).get('invalid_allocation', False)),
                'error': step_result.get('info', {}).get('error'),
                'message': f"AI step executed: {step_result.get('ai_action_name', step_result.get('action_name'))}"
            })

        return jsonify({
            'success': False,
            'error': step_result.get('error', 'Unknown error'),
            'message': 'AI step execution failed'
        })

    except Exception as e:
        logger.error(f"Error executing AI action: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'AI step execution failed due to error'
        })

@app.route('/api/run_episode', methods=['POST'])
def run_episode():
    """Run a complete episode with baseline agent"""
    global episode_history
    
    try:
        data = request.get_json()
        max_steps = data.get('max_steps', 50)
        task = data.get('task', 'medium')
        
        logger.info(f"Running complete episode: task={task}, max_steps={max_steps}")

        # Ensure the requested task is the active environment before running.
        if (not env_manager.is_initialized) or (str(env_manager.current_task or '').lower() != str(task).lower()):
            if not env_manager.create_environment(task, seed=42):
                return jsonify({
                    'success': False,
                    'error': f'Failed to initialize task {task}',
                    'message': 'Episode execution failed'
                })
        
        # Clear episode history
        episode_history = []
        
        # Run episode
        success, result = env_manager.run_complete_episode(max_steps)
        
        if success:
            # Update episode history
            episode_history = result.get('episode_history', [])
            
            return jsonify({
                'success': True,
                'total_reward': result['total_reward'],
                'steps': result['steps'],
                'terminated': result['terminated'],
                'truncated': result['truncated'],
                'final_info': result['final_info'],
                'episode_history': episode_history,
                'message': f'Episode completed: {result["steps"]} steps, reward: {result["total_reward"]:.2f}'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'message': 'Episode execution failed'
            })
            
    except Exception as e:
        logger.error(f"Error running episode: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Episode execution failed due to error'
        })

@app.route('/api/get_state')
def get_current_state():
    """Get current environment state"""
    try:
        auto_init_param = str(request.args.get('auto_init', '1')).strip().lower()
        auto_init = auto_init_param not in {'0', 'false', 'no'}
        requested_task = str(request.args.get('task', env_manager.current_task or 'medium'))

        if auto_init and not env_manager.is_initialized:
            logger.info(f"/api/get_state: environment not initialized, auto-init requested (task={requested_task})")
            if not env_manager.ensure_initialized(requested_task):
                logger.error("/api/get_state: auto-initialization failed")
                return jsonify({
                    'success': False,
                    'error': 'Environment auto-initialization failed',
                    'message': 'Failed to initialize environment before reading state',
                    'is_initialized': False
                })

        success, state = env_manager.get_current_state()
        
        if success:
            return jsonify({
                'success': True,
                'state': state,
                'is_initialized': env_manager.is_initialized,
                'auto_initialized': auto_init,
                'message': 'State retrieved successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': state.get('error', 'Unknown error'),
                'is_initialized': env_manager.is_initialized,
                'message': 'Failed to get current state'
            })
            
    except Exception as e:
        logger.error(f"Error getting current state: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'State retrieval failed due to error'
        })

@app.route('/api/episode_history')
def get_episode_history():
    """Get episode execution history"""
    try:
        total_reward = sum(h['reward'] for h in episode_history)
        
        return jsonify({
            'success': True,
            'history': episode_history,
            'total_steps': len(episode_history),
            'total_reward': total_reward,
            'message': f'Episode history: {len(episode_history)} steps, reward: {total_reward:.2f}'
        })
        
    except Exception as e:
        logger.error(f"Error getting episode history: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to get episode history'
        })

@app.route('/api/reset', methods=['POST'])
def reset_environment():
    """Reset environment to initial state"""
    global episode_history
    try:
        data = request.get_json()
        seed = data.get('seed', 42) if data else 42
        task = data.get('task') if data else None
        
        logger.info(f"Resetting environment: seed={seed}, task={task}")
        
        # Clear episode history
        episode_history = []

        if task and (not env_manager.is_initialized or str(env_manager.current_task or '').lower() != str(task).lower()):
            success = env_manager.create_environment(str(task), seed)
            if success:
                return jsonify({
                    'success': True,
                    'info': {'task': str(task), 'seed': seed},
                    'episode_count': env_manager.episode_count,
                    'message': 'Environment reset successfully'
                })
            return jsonify({
                'success': False,
                'error': f'Failed to initialize task {task}',
                'message': 'Environment reset failed'
            })
        
        # Reset environment
        success, info = env_manager.reset_environment(seed)
        
        if success:
            return jsonify({
                'success': True,
                'info': info,
                'episode_count': env_manager.episode_count,
                'message': 'Environment reset successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': info.get('error', 'Unknown error'),
                'message': 'Environment reset failed'
            })
            
    except Exception as e:
        logger.error(f"Error resetting environment: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Environment reset failed due to error'
        })

@app.route('/api/tasks')
def get_available_tasks():
    """Get available task configurations"""
    try:
        # advanced_tasks convenience helpers already return plain dicts.
        tasks = {
            'easy': easy(),
            'medium': medium(),
            'hard': hard()
        }
        
        # Simplify for web display
        simplified_tasks = {}
        for task_name, task_config in tasks.items():
            simplified_tasks[task_name] = {
                'name': task_config['name'],
                'difficulty': task_config['difficulty'],
                'max_steps': task_config['max_steps'],
                'description': task_config['description'],
                'patients': len(task_config['initial_patients']),
                'doctors': task_config['resources']['doctors'],
                'beds': task_config['resources']['beds'],
                'events_enabled': task_config['events']['enabled']
            }
        
        return jsonify({
            'success': True,
            'tasks': simplified_tasks,
            'current_task': env_manager.current_task,
            'message': 'Task configurations retrieved successfully'
        })
        
    except Exception as e:
        logger.error(f"Error getting available tasks: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to get task configurations'
        })


@app.route('/api/trigger_crisis', methods=['POST'])
def trigger_crisis():
    """Trigger a simulated resource crisis by locking a portion of available beds."""
    try:
        data = request.get_json(silent=True) or {}
        lock_ratio = float(data.get('lock_ratio', 0.50))
        seed = data.get('seed', None)

        logger.info(f"Triggering resource crisis: lock_ratio={lock_ratio}, seed={seed}")

        ok, details = env_manager.trigger_resource_crisis(lock_ratio=lock_ratio, seed=seed)
        if not ok:
            return jsonify({
                'success': False,
                'error': details.get('error', 'Unknown error'),
                'message': 'Failed to trigger crisis'
            })

        # Return updated state as well for immediate UI refresh.
        success_state, state = env_manager.get_current_state()
        return jsonify({
            'success': True,
            'details': details,
            'state': state if success_state else {},
            'message': 'Resource crisis triggered'
        })

    except Exception as e:
        logger.error(f"Error triggering crisis: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to trigger crisis due to error'
        })


@app.route('/api/drone_status')
def drone_status():
    """Get current drone fleet status for UI."""
    try:
        # Fleet exists even if env isn't initialized (demo-ready).
        if env_manager.env is not None:
            try:
                readable = env_manager.env.state().get("readable", {})  # type: ignore[union-attr]
                step = int(readable.get("step", readable.get("current_step", 0)) or 0)
                env_manager._tick_drone_fleet(step)  # type: ignore[attr-defined]
            except Exception:
                pass

        return jsonify({
            "success": True,
            "fleet": getattr(env_manager, "_drone_fleet", []),
        })
    except Exception as e:
        logger.error(f"Error getting drone status: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/dispatch_drone', methods=['POST'])
def dispatch_drone():
    """Dispatch a medicine drone when resources are full and RED patients are waiting."""
    try:
        data = request.get_json(silent=True) or {}
        drone_name = data.get("drone", "Drone Alpha")
        package = data.get("package", "Medical Kit (Red)")

        ok, details = env_manager.dispatch_drone(drone_name=drone_name, package=package, mode="manual")
        return jsonify({
            "success": bool(ok),
            "details": details,
        })
    except Exception as e:
        logger.error(f"Error dispatching drone: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        success, state = env_manager.get_current_state()
        
        health_status = {
            'status': 'healthy',
            'environment_ready': env_manager.is_initialized,
            'timestamp': datetime.now().isoformat(),
            'uptime': 'running' if env_manager.env else 'stopped',
            'episode_count': env_manager.episode_count,
            'last_activity': env_manager.last_activity.isoformat() if env_manager.last_activity else None
        }
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/timeline')
def get_timeline():
    """Time-Travel Replay API.

    Query params:
      - limit: int (max frames to return)
      - offset: int (start index)

    Returns:
      { success, total, offset, limit, frames }
    """
    try:
        limit = int(request.args.get('limit', 200))
        offset = int(request.args.get('offset', 0))
        limit = max(1, min(600, limit))
        offset = max(0, offset)

        frames = getattr(env_manager, '_timeline', [])
        total = len(frames)
        sliced = frames[offset: offset + limit]

        return jsonify({
            'success': True,
            'total': total,
            'offset': offset,
            'limit': limit,
            'frames': sliced,
        })
    except Exception as e:
        logger.error(f"Error getting timeline: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/timeline/clear', methods=['POST'])
def clear_timeline():
    """Clear the current replay timeline (does not reset the env)."""
    try:
        if hasattr(env_manager, '_reset_timeline'):
            env_manager._reset_timeline()  # type: ignore[attr-defined]
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error clearing timeline: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Weather API cache (5 minutes)
_weather_cache = {}
_weather_cache_time = 0
WEATHER_CACHE_DURATION = 60  # 1 minute
WEATHER_REQUEST_TIMEOUT = 4
WEATHER_MAX_RETRIES = 2
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '').strip()


def _get_json_with_retry(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Fetch JSON with small retry/backoff for transient weather API failures."""
    headers = {
        'User-Agent': 'smart-hospital-orchestration/1.0',
        'Accept': 'application/json',
    }

    for attempt in range(WEATHER_MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=WEATHER_REQUEST_TIMEOUT, headers=headers)
            status = response.status_code

            if status == 200:
                return response.json()

            if status == 429 or status >= 500:
                wait_seconds = min(1.5 * (attempt + 1), 4.0)
                logger.warning(f"Transient weather API status {status}; retrying in {wait_seconds:.1f}s")
                time.sleep(wait_seconds)
                continue

            logger.warning(f"Weather API request failed with status {status}: {response.text[:160]}")
            return None
        except requests.RequestException as e:
            wait_seconds = min(1.5 * (attempt + 1), 4.0)
            logger.warning(f"Weather API request error: {e}; retrying in {wait_seconds:.1f}s")
            time.sleep(wait_seconds)

    return None

def fetch_weather_from_open_meteo(city_name: str = 'Ahmedabad', lat: float = 23.0225, lon: float = 72.5714) -> Optional[Dict[str, Any]]:
    """
    Fetch weather from Open-Meteo API (free, no API key required).
    Most reliable free option with 100% uptime.
    """
    try:
        url = 'https://api.open-meteo.com/v1/forecast'
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m',
            'timezone': 'Asia/Kolkata',
        }
        data = _get_json_with_retry(url, params)
        if not data:
            return None

        current = data.get('current', {})
        
        # Extract data
        temp = round(float(current.get('temperature_2m', 0)), 1)
        humidity = int(current.get('relative_humidity_2m', 0))
        wind_kph = round(float(current.get('wind_speed_10m', 0)), 1)
        weather_code = int(current.get('weather_code', 0))
        
        # Map WMO weather codes to conditions
        weather_map = {
            0: 'Clear Sky', 1: 'Mainly Clear', 2: 'Partly Cloudy', 3: 'Overcast',
            45: 'Foggy', 48: 'Foggy', 51: 'Light Drizzle', 53: 'Moderate Drizzle',
            61: 'Slight Rain', 63: 'Moderate Rain', 65: 'Heavy Rain', 71: 'Slight Snow',
            80: 'Slight Rain Showers', 81: 'Moderate Showers', 82: 'Violent Showers',
            95: 'Thunderstorm'
        }
        condition = weather_map.get(weather_code, 'Clear')
        
        logger.info(f"✓ Open-Meteo API SUCCESS: {temp}°C, {condition}")
        
        return {
            'temp': temp,
            'condition': condition,
            'humidity': humidity,
            'wind_kph': wind_kph,
            'aqi': None,  # Will fetch separately
            'source': 'Open-Meteo'
        }
    except Exception as e:
        logger.warning(f"Open-Meteo API failed: {e}")
        return None


def fetch_weather_from_openweather(lat: float = 23.0225, lon: float = 72.5714) -> Optional[Dict[str, Any]]:
    """Fetch current weather from OpenWeather using API key."""
    if not OPENWEATHER_API_KEY:
        logger.warning("OpenWeather API key not configured")
        return None

    try:
        url = 'https://api.openweathermap.org/data/2.5/weather'
        params = {
            'lat': lat,
            'lon': lon,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
        }
        data = _get_json_with_retry(url, params)
        if not data:
            return None

        main = data.get('main', {})
        wind = data.get('wind', {})
        weather_items = data.get('weather', [])
        weather_text = 'Clear Sky'
        if weather_items and isinstance(weather_items[0], dict):
            weather_text = str(weather_items[0].get('description', 'Clear Sky')).title()

        temp = round(float(main.get('temp', 0.0)), 1)
        humidity = int(float(main.get('humidity', 0)))
        wind_kph = round(float(wind.get('speed', 0.0)) * 3.6, 1)  # m/s -> km/h

        logger.info(f"OpenWeather API SUCCESS: {temp} C, {weather_text}")
        return {
            'temp': temp,
            'condition': weather_text,
            'humidity': humidity,
            'wind_kph': wind_kph,
            'aqi': None,
            'source': 'OpenWeather'
        }
    except Exception as e:
        logger.warning(f"OpenWeather API failed: {e}")
        return None


def fetch_weather_from_wttr(city_name: str = 'Ahmedabad') -> Optional[Dict[str, Any]]:
    """Secondary live weather provider when Open-Meteo is unavailable."""
    try:
        url = f'https://wttr.in/{city_name}'
        params = {'format': 'j1'}
        data = _get_json_with_retry(url, params)
        if not data:
            return None

        current_list = data.get('current_condition', [])
        if not current_list:
            return None
        current = current_list[0]

        temp = round(float(current.get('temp_C', 0)), 1)
        humidity = int(float(current.get('humidity', 0)))
        wind_kph = round(float(current.get('windspeedKmph', 0)), 1)
        desc = current.get('weatherDesc', [])
        condition = (desc[0].get('value') if desc and isinstance(desc[0], dict) else 'Clear') or 'Clear'

        logger.info(f"WTTR fallback API SUCCESS: {temp} C, {condition}")

        return {
            'temp': temp,
            'condition': condition,
            'humidity': humidity,
            'wind_kph': wind_kph,
            'aqi': None,
            'source': 'WTTR'
        }
    except Exception as e:
        logger.warning(f"WTTR fallback API failed: {e}")
        return None


def fetch_weather_from_met_no(lat: float = 23.0225, lon: float = 72.5714) -> Optional[Dict[str, Any]]:
    """Fallback live weather from MET Norway location forecast API."""
    try:
        url = 'https://api.met.no/weatherapi/locationforecast/2.0/compact'
        params = {'lat': lat, 'lon': lon}
        data = _get_json_with_retry(url, params)
        if not data:
            return None

        timeseries = (data.get('properties') or {}).get('timeseries') or []
        if not timeseries:
            return None

        first = timeseries[0] if isinstance(timeseries[0], dict) else {}
        instant = (first.get('data') or {}).get('instant') or {}
        details = instant.get('details') or {}
        if not details:
            return None

        symbol = (((first.get('data') or {}).get('next_1_hours') or {}).get('summary') or {}).get('symbol_code', '')
        symbol_l = str(symbol).lower()
        if 'thunder' in symbol_l:
            condition = 'Thunderstorm'
        elif 'snow' in symbol_l or 'sleet' in symbol_l:
            condition = 'Snow'
        elif 'rain' in symbol_l or 'drizzle' in symbol_l:
            condition = 'Rain'
        elif 'fog' in symbol_l:
            condition = 'Fog'
        elif 'cloud' in symbol_l:
            condition = 'Partly Cloudy'
        else:
            condition = 'Clear Sky'

        temp = round(float(details.get('air_temperature', 0.0)), 1)
        humidity = int(float(details.get('relative_humidity', 0.0)))
        wind_kph = round(float(details.get('wind_speed', 0.0)) * 3.6, 1)

        logger.info(f"MET Norway weather SUCCESS: {temp} C, {condition}")
        return {
            'temp': temp,
            'condition': condition,
            'humidity': humidity,
            'wind_kph': wind_kph,
            'aqi': None,
            'source': 'MET Norway'
        }
    except Exception as e:
        logger.warning(f"MET Norway weather failed: {e}")
        return None


def fetch_aqi_from_open_meteo(lat: float = 23.0225, lon: float = 72.5714) -> Optional[int]:
    """
    Fetch AQI from Open-Meteo Air Quality API.
    """
    try:
        url = 'https://air-quality-api.open-meteo.com/v1/air-quality'
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'us_aqi,pm2_5,pm10',
            'timezone': 'Asia/Kolkata',
        }
        data = _get_json_with_retry(url, params)
        if not data:
            return None

        current = data.get('current', {})

        # Prefer directly provided US AQI.
        us_aqi = current.get('us_aqi')
        if us_aqi is not None:
            aqi = int(round(float(us_aqi)))
            logger.info(f"Air Quality API SUCCESS: us_aqi={aqi}")
            return aqi

        pm25 = float(current.get('pm2_5', 0))
        
        # Convert PM2.5 to AQI (US EPA formula)
        if pm25 <= 12:
            aqi = int(pm25 * 4.17)  # 0-50
        elif pm25 <= 35.4:
            aqi = int(50 + ((pm25 - 12) / 23.4) * 50)  # 51-100
        elif pm25 <= 55.4:
            aqi = int(100 + ((pm25 - 35.4) / 20) * 50)  # 101-150
        elif pm25 <= 150.4:
            aqi = int(150 + ((pm25 - 55.4) / 95) * 50)  # 151-200
        else:
            aqi = int(200 + ((pm25 - 150.4) / 250) * 50)  # 201+
        
        logger.info(f"Air Quality API SUCCESS: PM2.5={pm25}, AQI={aqi}")
        return aqi
        
    except Exception as e:
        logger.warning(f"Air Quality API failed: {e}")
        return None


def fetch_aqi_from_openweather(lat: float = 23.0225, lon: float = 72.5714) -> Optional[int]:
    """Fetch AQI from OpenWeather air pollution API and map to US-style index."""
    if not OPENWEATHER_API_KEY:
        return None

    try:
        url = 'https://api.openweathermap.org/data/2.5/air_pollution'
        params = {
            'lat': lat,
            'lon': lon,
            'appid': OPENWEATHER_API_KEY,
        }
        data = _get_json_with_retry(url, params)
        if not data:
            return None

        rows = data.get('list', [])
        if not rows:
            return None
        row = rows[0] if isinstance(rows[0], dict) else {}

        # Prefer PM2.5-based US AQI conversion when component is available.
        pm25 = float((row.get('components') or {}).get('pm2_5', 0.0))
        if pm25 > 0:
            if pm25 <= 12:
                return int(pm25 * 4.17)
            if pm25 <= 35.4:
                return int(50 + ((pm25 - 12) / 23.4) * 50)
            if pm25 <= 55.4:
                return int(100 + ((pm25 - 35.4) / 20) * 50)
            if pm25 <= 150.4:
                return int(150 + ((pm25 - 55.4) / 95) * 50)
            return int(200 + ((pm25 - 150.4) / 250) * 50)

        # Fallback from OpenWeather AQI scale (1-5) to approximate US AQI bands.
        owm_aqi = int(float((row.get('main') or {}).get('aqi', 0)))
        mapping = {1: 30, 2: 75, 3: 125, 4: 175, 5: 300}
        return mapping.get(owm_aqi)
    except Exception as e:
        logger.warning(f"OpenWeather AQI API failed: {e}")
        return None


@app.route('/api/weather')
def get_weather():
    """Fetch live weather data for Ahmedabad with smart hospital impact analysis."""
    global _weather_cache, _weather_cache_time
    
    # Return cached data if still valid
    current_time = time.time()
    if _weather_cache and (current_time - _weather_cache_time) < WEATHER_CACHE_DURATION:
        logger.info("✓ Returning cached weather data")
        return jsonify(_weather_cache)
    
    try:
        city = 'Ahmedabad'
        
        # Primary live source: OpenWeather. Fallbacks prefer observed feeds.
        weather_data = fetch_weather_from_openweather()
        if weather_data is None:
            weather_data = fetch_weather_from_open_meteo()
        if weather_data is None:
            weather_data = fetch_weather_from_wttr()
        if weather_data is None:
            weather_data = fetch_weather_from_met_no()

        if weather_data is None:
            if _weather_cache:
                logger.warning("Weather APIs failed; returning cached weather reading")
                cached = dict(_weather_cache)
                cached['stale'] = True
                cached['source'] = f"{cached.get('source', 'OpenWeather')} (cached)"
                cached['message'] = 'Returning last known live reading'
                return jsonify(cached)

            logger.warning("Weather APIs failed; no cached weather available")
            return jsonify({
                'success': False,
                'error': 'Live weather data unavailable',
                'city': city,
                'country': 'India',
                'source': 'OpenWeather',
                'timestamp': current_time,
                'last_updated': datetime.now().isoformat(),
            }), 503
        
        # Fetch AQI separately (OpenWeather first, then Open-Meteo fallback)
        aqi = fetch_aqi_from_openweather()
        if aqi is None:
            aqi = fetch_aqi_from_open_meteo()
        weather_data['aqi'] = aqi
        
        # Determine icon based on condition
        condition_lower = weather_data['condition'].lower()
        if 'clear' in condition_lower or 'sunny' in condition_lower:
            icon = '☀️'
        elif 'cloud' in condition_lower:
            if 'partly' in condition_lower:
                icon = '🌤️'
            else:
                icon = '☁️'
        elif 'rain' in condition_lower or 'drizzle' in condition_lower:
            icon = '🌧️'
        elif 'thunder' in condition_lower or 'storm' in condition_lower:
            icon = '🌩️'
        elif 'fog' in condition_lower or 'mist' in condition_lower:
            icon = '🌫️'
        else:
            icon = '🌡️'
        
        # Smart Hospital Impact Logic
        temp = weather_data['temp']
        condition = weather_data['condition']
        humidity = weather_data['humidity']
        wind_kph = weather_data['wind_kph']
        aqi = weather_data['aqi']
        
        impact_messages = []
        
        # Temperature-based impact
        if temp > 40:
            impact_messages.append('🔥 EXTREME HEAT ALERT: Heatstroke risk critical. Emergency cooling active.')
        elif temp > 35:
            impact_messages.append('🌡️ Heat risk: Hydration stations required. Monitor heat-related admissions.')
        elif temp < 18:
            impact_messages.append('❄️ Cold weather: ICU capacity reserved for hypothermia cases.')
        
        # Condition-based impact
        if 'rain' in condition.lower() or 'shower' in condition.lower():
            impact_messages.append('🌧️ Increased vector-borne diseases (Dengue/Malaria) & trauma cases expected.')
        
        if 'thunderstorm' in condition.lower():
            impact_messages.append('⚡ Emergency protocols: Power backup confirmed & ready.')
        
        if 'fog' in condition.lower():
            impact_messages.append('🌫️ Respiratory issues expected. Pulmonology on standby.')
        
        # AQI-based impact (AQI may be unavailable from upstream API)
        if isinstance(aqi, (int, float)):
            if aqi > 200:
                impact_messages.append('🚨 CRITICAL POLLUTION: Respiratory distress admissions expected.')
            elif aqi > 150:
                impact_messages.append('⚠️ High pollution: Enhanced monitoring for respiratory patients.')
            elif aqi > 100:
                impact_messages.append('📊 Moderate pollution: Standard protocols active.')
        
        if not impact_messages:
            impact_messages.append('✅ Normal conditions. Standard operations.')
        
        hospital_impact = ' '.join(impact_messages)
        
        # Determine background gradient
        if temp > 38 or 'clear' in condition.lower() or 'sunny' in condition.lower():
            bg_gradient = 'linear-gradient(135deg, #f59e0b, #ef4444)'  # Orange to Red
        elif 'rain' in condition.lower() or 'thunderstorm' in condition.lower():
            bg_gradient = 'linear-gradient(135deg, #475569, #1e293b)'  # Dark Gray
        elif temp < 20:
            bg_gradient = 'linear-gradient(135deg, #0ea5e9, #6366f1)'  # Blue to Indigo
        else:
            bg_gradient = 'linear-gradient(135deg, #0ea5e9, #3b82f6)'  # Sky Blue
        
        # Build final response
        weather_data = {
            'success': True,
            'city': city,
            'country': 'India',
            'temp': temp,
            'condition': condition,
            'icon': icon,
            'humidity': humidity,
            'wind_kph': wind_kph,
            'aqi': aqi,
            'hospital_impact': hospital_impact,
            'bg_gradient': bg_gradient,
            'source': weather_data.get('source', 'Open-Meteo'),
            'stale': False,
            'timestamp': current_time,
            'last_updated': datetime.now().isoformat()
        }
        
        # Cache the result
        _weather_cache = weather_data
        _weather_cache_time = current_time
        
        logger.info(f"✓ Weather endpoint returning: {temp}°C, {condition}, AQI={aqi}")
        return jsonify(weather_data)
        
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to fetch weather data'
        }), 500

if __name__ == '__main__':
    # Use ASCII-only startup logs so Windows default code pages (e.g. cp1252)
    # don't crash when PYTHONIOENCODING is not explicitly set.
    print("Starting Smart Hospital Orchestration Web Interface...")
    port = int(os.getenv('PORT', '7860'))
    print(f"Open your browser: http://localhost:{port}")
    print("Hospital Resource Management Dashboard Ready")

    # Emit one baseline inference run at startup for deployment proof logs.
    try:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, os.path.join(repo_root, 'inference.py'), '--task', 'medium', '--seed', '42']
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, timeout=180)
        if proc.stdout:
            print(proc.stdout, end='')
        if proc.stderr:
            print(proc.stderr, end='')
    except Exception as e:
        print(f"Startup inference skipped due to error: {e}")
    
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
