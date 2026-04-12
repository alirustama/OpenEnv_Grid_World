#!/usr/bin/env python3
"""
OpenEnv Specification Compliance Validator

Validates:
1. openenv.yaml structure and configuration
2. Pydantic typed models (Action, Observation, State)
3. Environment API: reset(), step(), state() endpoints
4. Response schemas and types
"""

import os
import sys
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import project components
from env.environment import StochasticGridEnvironment
from models.action import Action
from models.observation import Observation
from models.state import State
from env.tasks import get_all_tasks


class OpenEnvSpecValidator:
    """Validates OpenEnv specification compliance."""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.results = []
        self.project_root = Path(__file__).parent

    def log_check(self, category: str, name: str, passed: bool, details: str = ""):
        """Log validation result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} [{category}] {name}")
        if details:
            print(f"            └─ {details}")
        
        if passed:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
        
        self.results.append({
            "category": category,
            "check": name,
            "passed": passed,
            "details": details
        })

    # ==================== SECTION 1: openenv.yaml ====================

    def validate_openenv_yaml(self):
        """Validate openenv.yaml structure."""
        print("\n" + "="*70)
        print("1️⃣  OPENENV.YAML VALIDATION")
        print("="*70)
        
        yaml_path = self.project_root / "openenv.yaml"
        
        # Check file exists
        if not yaml_path.exists():
            self.log_check("YAML", "File exists", False, f"Not found: {yaml_path}")
            return False
        
        self.log_check("YAML", "File exists", True, str(yaml_path))
        
        # Parse YAML
        try:
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
            self.log_check("YAML", "Valid syntax", True, "Successfully parsed")
        except Exception as e:
            self.log_check("YAML", "Valid syntax", False, str(e))
            return False
        
        # Check required keys
        required_keys = ["class", "stochastic_params"]
        has_keys = all(key in config for key in required_keys)
        missing_keys = [k for k in required_keys if k not in config]
        self.log_check(
            "YAML",
            "Required keys present",
            has_keys,
            f"Keys found: {', '.join(required_keys) if has_keys else f'Missing: {missing_keys}'}"
        )
        
        if not has_keys:
            return False
        
        # Check class specification
        class_spec = config.get("class", "")
        expected_class = "env.environment:StochasticGridEnvironment"
        is_correct = class_spec == expected_class
        self.log_check(
            "YAML",
            "Correct class specification",
            is_correct,
            f"Class: {class_spec}"
        )
        
        # Check stochastic_params structure
        stoch_params = config.get("stochastic_params", {})
        has_difficulties = "difficulties" in stoch_params
        self.log_check(
            "YAML",
            "Stochastic params configured",
            has_difficulties,
            f"Has difficulties: {has_difficulties}"
        )
        
        # Check difficulties
        difficulties = stoch_params.get("difficulties", {})
        expected_difficulties = {"easy", "medium", "hard"}
        has_all_difficulties = all(d in difficulties for d in expected_difficulties)
        self.log_check(
            "YAML",
            "All difficulties defined",
            has_all_difficulties,
            f"Difficulties: {list(difficulties.keys())}"
        )
        
        # Validate each difficulty
        for difficulty in expected_difficulties:
            if difficulty in difficulties:
                diff_config = difficulties[difficulty]
                required_diff_keys = ["obstacle_density", "num_obstacles", "max_steps"]
                has_diff_keys = all(k in diff_config for k in required_diff_keys)
                self.log_check(
                    "YAML",
                    f"{difficulty.capitalize()} config complete",
                    has_diff_keys,
                    f"Density: {diff_config.get('obstacle_density')}, "
                    f"Obstacles: {diff_config.get('num_obstacles')}, "
                    f"MaxSteps: {diff_config.get('max_steps')}"
                )
        
        return True

    # ==================== SECTION 2: Pydantic Models ====================

    def validate_pydantic_models(self):
        """Validate Pydantic typed models."""
        print("\n" + "="*70)
        print("2️⃣  PYDANTIC MODELS VALIDATION")
        print("="*70)
        
        # Test Action model
        try:
            action = Action(direction="right", magnitude=0.8)
            self.log_check(
                "Models",
                "Action model works",
                True,
                f"Created: {action.direction} @ {action.magnitude}"
            )
        except Exception as e:
            self.log_check("Models", "Action model works", False, str(e))
        
        # Test Observation model
        try:
            obs = Observation(
                agent_position=[5, 5],
                target_position=[15, 15],
                distance_to_target=20,
                obstacles=[],
                step_count=0,
                episode_reward=0.0,
                grid_size=20,
                difficulty="medium"
            )
            self.log_check(
                "Models",
                "Observation model works",
                True,
                f"Position: {obs.agent_position}, Target: {obs.target_position}"
            )
        except Exception as e:
            self.log_check("Models", "Observation model works", False, str(e))
        
        # Test State model (used internally for state management)
        try:
            state_obj = State(
                agent_position=[5, 5],
                target_position=[15, 15],
                obstacles=[[10, 10]],
                step_count=0,
                episode_reward=0.0,
                grid_size=20
            )
            self.log_check(
                "Models",
                "State model works",
                True,
                f"Steps: {state_obj.step_count}, Terminal: False"
            )
        except Exception as e:
            self.log_check("Models", "State model works", False, str(e))
        
        # Test model validation
        try:
            invalid_action = Action(direction="invalid", magnitude=0.5)
            self.log_check(
                "Models",
                "Action validation works",
                False,
                "Should have rejected invalid direction"
            )
        except Exception as e:
            self.log_check(
                "Models",
                "Action validation works",
                True,
                "Correctly rejected invalid direction"
            )

    # ==================== SECTION 3: API Endpoints ====================

    def validate_api_endpoints(self):
        """Validate reset(), step(), state() API."""
        print("\n" + "="*70)
        print("3️⃣  API ENDPOINTS VALIDATION")
        print("="*70)
        
        env = None
        
        # Test initialization
        try:
            env = StochasticGridEnvironment(
                grid_size=20,
                difficulty="easy",
                seed=42
            )
            self.log_check(
                "API",
                "Environment initialization",
                True,
                "StochasticGridEnvironment created"
            )
        except Exception as e:
            self.log_check("API", "Environment initialization", False, str(e))
            return False
        
        # Test reset() endpoint
        try:
            observation = env.reset(seed=42)
            
            # Validate observation type
            is_dict = isinstance(observation, dict)
            self.log_check(
                "API",
                "reset() returns dict",
                is_dict,
                f"Type: {type(observation).__name__}"
            )
            
            # Validate required observation keys
            required_obs_keys = [
                "agent_position",
                "target_position",
                "distance_to_target",
                "obstacles",
                "step_count",
                "episode_reward",
                "grid_size",
                "difficulty"
            ]
            
            has_all_keys = all(key in observation for key in required_obs_keys)
            self.log_check(
                "API",
                "reset() returns valid observation",
                has_all_keys,
                f"Keys: {len(observation)}/8 required"
            )
            
            # Validate observation value types
            right_types = (
                isinstance(observation.get("agent_position"), list) and
                isinstance(observation.get("target_position"), list) and
                isinstance(observation.get("distance_to_target"), (int, float)) and
                isinstance(observation.get("obstacles"), list) and
                isinstance(observation.get("step_count"), int) and
                isinstance(observation.get("episode_reward"), (int, float)) and
                isinstance(observation.get("grid_size"), int) and
                isinstance(observation.get("difficulty"), str)
            )
            self.log_check(
                "API",
                "reset() observation types correct",
                right_types,
                "All value types validated"
            )
            
        except Exception as e:
            self.log_check("API", "reset() endpoint", False, str(e))
            return False
        
        # Test step() endpoint
        try:
            action = {"direction": "right", "magnitude": 0.8}
            observation, reward, done = env.step(action)
            
            # Validate return types
            is_tuple = isinstance((observation, reward, done), tuple)
            obs_is_dict = isinstance(observation, dict)
            reward_is_float = isinstance(reward, (int, float))
            done_is_bool = isinstance(done, bool)
            
            all_types_correct = obs_is_dict and reward_is_float and done_is_bool
            self.log_check(
                "API",
                "step() returns (obs, reward, done)",
                all_types_correct,
                f"Obs: {type(observation).__name__}, "
                f"Reward: {type(reward).__name__}, "
                f"Done: {type(done).__name__}"
            )
            
            # Validate observation still valid
            has_keys = all(key in observation for key in required_obs_keys)
            self.log_check(
                "API",
                "step() observation valid",
                has_keys,
                f"Keys: {len(observation)}/8 required"
            )
            
            # Validate reward is reasonable
            reward_reasonable = -100 <= reward <= 100
            self.log_check(
                "API",
                "step() reward reasonable",
                reward_reasonable,
                f"Reward: {reward}"
            )
            
        except Exception as e:
            self.log_check("API", "step() endpoint", False, str(e))
            return False
        
        # Test state() endpoint
        try:
            # First get an observation from reset/step
            obs_from_reset = env.reset(seed=42)
            
            # Then call state() without modifying environment
            state_result = env.state()
            
            # Validate state type
            is_dict = isinstance(state_result, dict)
            self.log_check(
                "API",
                "state() returns dict",
                is_dict,
                f"Type: {type(state_result).__name__}"
            )
            
            # Validate state keys (state() returns observation format)
            expected_state_keys = {"agent_position", "target_position", "step_count", "episode_reward"}
            has_keys = expected_state_keys.issubset(state_result.keys())
            self.log_check(
                "API",
                "state() has required keys",
                has_keys,
                f"Keys: {list(state_result.keys())}"
            )
            
        except Exception as e:
            self.log_check("API", "state() endpoint", False, str(e))
            return False
        
        # Test consistency across environments with same seed
        try:
            # Create two environments with same seed
            env1 = StochasticGridEnvironment(difficulty="easy", seed=42)
            env2 = StochasticGridEnvironment(difficulty="easy", seed=42)
            
            # Reset both with same seed
            obs1 = env1.reset(seed=42)
            obs2 = env2.reset(seed=42)
            
            # Compare specific fields
            same_target = obs1["target_position"] == obs2["target_position"]
            same_obstacles = obs1["obstacles"] == obs2["obstacles"]
            
            reproducible = same_target and same_obstacles
            self.log_check(
                "API",
                "Reproducibility with seed",
                reproducible,
                "Same seed produces identical scenarios"
            )
            
        except Exception as e:
            self.log_check("API", "Reproducibility", False, str(e))

    # ==================== SECTION 4: Tasks ====================

    def validate_tasks(self):
        """Validate task definitions."""
        print("\n" + "="*70)
        print("4️⃣  TASKS VALIDATION")
        print("="*70)
        
        try:
            tasks = get_all_tasks()
            
            # Check we have tasks
            has_tasks = len(tasks) > 0
            self.log_check(
                "Tasks",
                "Tasks defined",
                has_tasks,
                f"Found {len(tasks)} tasks"
            )
            
            # Check required tasks
            required_tasks = {"easy_navigation", "medium_navigation", "hard_navigation"}
            has_required = required_tasks.issubset(tasks.keys())
            self.log_check(
                "Tasks",
                "All difficulty levels",
                has_required,
                f"Tasks: {list(tasks.keys())}"
            )
            
            # Validate each task
            for task_name, task in tasks.items():
                has_required_attrs = all(
                    hasattr(task, attr) for attr in ["name", "difficulty", "max_steps"]
                )
                self.log_check(
                    "Tasks",
                    f"{task_name} structure",
                    has_required_attrs,
                    f"Difficulty: {task.difficulty}, MaxSteps: {task.max_steps}"
                )
                
                # Test calculate_score method
                try:
                    score = task.calculate_score(
                        episode_reward=10.0,
                        steps_taken=50,
                        collision_count=2
                    )
                    is_valid_score = 0.0 <= score <= 1.0
                    self.log_check(
                        "Tasks",
                        f"{task_name} scoring",
                        is_valid_score,
                        f"Score: {score:.2f}"
                    )
                except Exception as e:
                    self.log_check(
                        "Tasks",
                        f"{task_name} scoring",
                        False,
                        str(e)
                    )
            
        except Exception as e:
            self.log_check("Tasks", "Task system", False, str(e))

    # ==================== MAIN ====================

    def run_all(self) -> bool:
        """Run all validation checks."""
        print("\n" + "="*70)
        print("🔍 OPENENV SPECIFICATION COMPLIANCE VALIDATOR")
        print("="*70)
        
        self.validate_openenv_yaml()
        self.validate_pydantic_models()
        self.validate_api_endpoints()
        self.validate_tasks()
        
        # Summary
        print("\n" + "="*70)
        print("📊 VALIDATION SUMMARY")
        print("="*70)
        
        total = self.checks_passed + self.checks_failed
        
        # Group by category
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0}
            if result["passed"]:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1
        
        for cat in sorted(categories.keys()):
            stats = categories[cat]
            total_cat = stats["passed"] + stats["failed"]
            status = "✅" if stats["failed"] == 0 else "⚠️ "
            print(f"{status} {cat:15} {stats['passed']}/{total_cat} passed")
        
        print("\n" + "="*70)
        print(f"Overall: {self.checks_passed}/{total} checks passed")
        
        if self.checks_failed == 0:
            print("\n🎉 ✨ FULL OpenEnv COMPLIANCE VERIFIED ✨ 🎉")
        else:
            print(f"\n⚠️  {self.checks_failed} issue(s) to address")
        
        print("="*70 + "\n")
        
        return self.checks_failed == 0


def main():
    """Main entry point."""
    validator = OpenEnvSpecValidator()
    success = validator.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
