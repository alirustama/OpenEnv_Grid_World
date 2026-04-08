#!/usr/bin/env python3
"""
Test OpenEnv HF Space API
Tests: https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World
"""

import os
import sys
import json
import requests
from typing import Dict, Any, Optional

# Your HF Space Configuration
HF_SPACE_URL = "https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World"
HF_SPACE_API = "https://indianhacker001-openenv-grid-world.hf.space"


class SpaceAPITester:
    """Test suite for OpenEnv HF Space API."""

    def __init__(self, api_url: str):
        """Initialize tester."""
        self.api_url = api_url.rstrip('/')
        self.tests_passed = 0
        self.tests_failed = 0
        self.results = []

    def log_test(self, name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {name}")
        if details:
            print(f"       {details}")
        
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        
        self.results.append({
            "test": name,
            "passed": passed,
            "details": details
        })

    def test_1_ping(self) -> bool:
        """Test 1: Space responds to ping."""
        try:
            response = requests.get(
                f"{self.api_url}/info",
                timeout=5
            )
            passed = response.status_code in [200, 404, 405]
            self.log_test(
                "Space Ping",
                passed,
                f"Status: {response.status_code}"
            )
            return passed
        except Exception as e:
            self.log_test("Space Ping", False, str(e))
            return False

    def test_2_reset(self) -> bool:
        """Test 2: reset() endpoint works."""
        try:
            payload = {
                "difficulty": "easy",
                "seed": 42
            }
            response = requests.post(
                f"{self.api_url}/api/reset",
                json=payload,
                timeout=10
            )
            
            if response.status_code != 200:
                self.log_test("Reset Endpoint", False, f"Status: {response.status_code}")
                return False
            
            data = response.json()
            required_keys = ["agent_position", "target_position", "obstacles"]
            has_keys = all(key in data for key in required_keys)
            
            details = f"Keys: {', '.join(list(data.keys())[:3])}"
            self.log_test("Reset Endpoint", has_keys, details)
            return has_keys
        except Exception as e:
            self.log_test("Reset Endpoint", False, str(e))
            return False

    def test_3_step(self) -> bool:
        """Test 3: step() endpoint works."""
        try:
            # Reset first
            requests.post(
                f"{self.api_url}/api/reset",
                json={"difficulty": "easy", "seed": 42},
                timeout=10
            )
            
            # Step
            action = {
                "direction": "right",
                "magnitude": 0.8
            }
            response = requests.post(
                f"{self.api_url}/api/step",
                json=action,
                timeout=10
            )
            
            passed = (
                response.status_code == 200 and
                "reward" in response.json()
            )
            
            reward = response.json().get("reward", "N/A")
            self.log_test("Step Endpoint", passed, f"Reward: {reward}")
            return passed
        except Exception as e:
            self.log_test("Step Endpoint", False, str(e))
            return False

    def test_4_inference(self) -> bool:
        """Test 4: inference() endpoint works."""
        try:
            payload = {
                "task": "easy_navigation",
                "seed": 42,
                "max_steps": 50
            }
            response = requests.post(
                f"{self.api_url}/api/inference",
                json=payload,
                timeout=30
            )
            
            passed = (
                response.status_code == 200 and
                "episode_reward" in response.json()
            )
            
            if passed:
                data = response.json()
                details = f"Reward: {data.get('episode_reward', 0):.2f}, Steps: {data.get('steps_taken', 0)}"
            else:
                details = f"Status: {response.status_code}"
            
            self.log_test("Inference Endpoint", passed, details)
            return passed
        except Exception as e:
            self.log_test("Inference Endpoint", False, str(e))
            return False

    def test_5_grading(self) -> bool:
        """Test 5: grading endpoint works."""
        try:
            payload = {
                "task": "medium_navigation",
                "episode_reward": 25.5,
                "steps_taken": 45,
                "completed": True,
                "collisions": 2
            }
            response = requests.post(
                f"{self.api_url}/api/grade",
                json=payload,
                timeout=10
            )
            
            passed = (
                response.status_code == 200 and
                "score" in response.json()
            )
            
            score = response.json().get("score", "N/A")
            self.log_test("Grading Endpoint", passed, f"Score: {score}")
            return passed
        except Exception as e:
            self.log_test("Grading Endpoint", False, str(e))
            return False

    def test_6_response_format(self) -> bool:
        """Test 6: Response format is valid."""
        try:
            response = requests.post(
                f"{self.api_url}/api/reset",
                json={"difficulty": "medium", "seed": 123},
                timeout=10
            )
            
            data = response.json()
            required_keys = [
                "agent_position",
                "target_position",
                "distance_to_target",
                "obstacles",
                "difficulty"
            ]
            
            passed = all(key in data for key in required_keys)
            details = f"Has {len(data)} keys, needs ≥ {len(required_keys)}"
            self.log_test("Response Format Valid", passed, details)
            return passed
        except Exception as e:
            self.log_test("Response Format Valid", False, str(e))
            return False

    def run_all(self) -> bool:
        """Run all tests."""
        print("\n" + "="*70)
        print("🧪 OpenEnv Grid World - API Test Suite")
        print("="*70)
        print(f"Space: {HF_SPACE_URL}")
        print(f"API:   {self.api_url}\n")
        
        self.test_1_ping()
        self.test_2_reset()
        self.test_3_step()
        self.test_4_inference()
        self.test_5_grading()
        self.test_6_response_format()
        
        # Summary
        print("\n" + "="*70)
        total = self.tests_passed + self.tests_failed
        print(f"Results: {self.tests_passed}/{total} tests passed")
        
        if self.tests_failed == 0:
            print("\n✨ ALL TESTS PASSED - Space is production ready! ✨")
        else:
            print(f"\n⚠️  {self.tests_failed} test(s) failed")
        
        print("="*70 + "\n")
        
        return self.tests_failed == 0


def main():
    """Main entry point."""
    # Use hardcoded Space URL
    api_url = HF_SPACE_API.rstrip('/')
    
    print(f"\n📍 Testing: {HF_SPACE_URL}\n")
    
    tester = SpaceAPITester(api_url)
    success = tester.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
