#!/usr/bin/env python3
"""
Pre-Submission Check - Verify HF Space Deployment
Tests the OpenEnv Grid World Space at: https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World
"""

import os
import sys
import requests
import json
from typing import Dict, Tuple

# Your HF Space URL
HF_SPACE_URL = "https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World"
# For API calls, use the direct endpoint
SPACE_API_URL = "https://indianhacker001-openenv-grid-world.hf.space"

class PreSubmissionChecker:
    """Comprehensive pre-submission validation."""
    
    def __init__(self, space_url: str):
        self.space_url = space_url.rstrip('/')
        self.api_url = SPACE_API_URL.rstrip('/')
        self.checks_passed = 0
        self.checks_failed = 0
        self.results = []

    def log_check(self, name: str, passed: bool, details: str = ""):
        """Log check result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {name}")
        if details:
            print(f"       {details}")
        
        if passed:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
        
        self.results.append({"check": name, "passed": passed, "details": details})

    def check_1_space_exists(self) -> bool:
        """Check 1: HF Space page is accessible."""
        try:
            response = requests.head(self.space_url, timeout=5, allow_redirects=True)
            passed = response.status_code == 200
            self.log_check(
                "HF Space Page Accessible",
                passed,
                f"Status: {response.status_code}"
            )
            return passed
        except Exception as e:
            self.log_check("HF Space Page Accessible", False, str(e))
            return False

    def check_2_api_responds(self) -> bool:
        """Check 2: API endpoint responds."""
        try:
            response = requests.get(
                f"{self.api_url}/info",
                timeout=5
            )
            passed = response.status_code in [200, 404, 405]
            self.log_check(
                "API Endpoint Responds",
                passed,
                f"Status: {response.status_code}"
            )
            return passed
        except Exception as e:
            self.log_check("API Endpoint Responds", False, str(e))
            return False

    def check_3_reset_works(self) -> bool:
        """Check 3: reset() endpoint works."""
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
                self.log_check("Reset Endpoint", False, f"Status: {response.status_code}")
                return False
            
            data = response.json()
            required_keys = ["agent_position", "target_position", "obstacles"]
            has_required = all(key in data for key in required_keys)
            
            details = f"Status: {response.status_code}, Keys: {list(data.keys())[:3]}"
            self.log_check("Reset Endpoint Works", has_required, details)
            return has_required
        except Exception as e:
            self.log_check("Reset Endpoint Works", False, str(e))
            return False

    def check_4_step_works(self) -> bool:
        """Check 4: step() endpoint works."""
        try:
            # First reset
            reset_response = requests.post(
                f"{self.api_url}/api/reset",
                json={"difficulty": "easy", "seed": 42},
                timeout=10
            )
            
            if reset_response.status_code != 200:
                self.log_check("Step Endpoint Works", False, "Reset failed")
                return False
            
            # Then step
            action = {"direction": "right", "magnitude": 0.8}
            response = requests.post(
                f"{self.api_url}/api/step",
                json=action,
                timeout=10
            )
            
            passed = (
                response.status_code == 200 and
                "reward" in response.json()
            )
            
            details = f"Status: {response.status_code}"
            self.log_check("Step Endpoint Works", passed, details)
            return passed
        except Exception as e:
            self.log_check("Step Endpoint Works", False, str(e))
            return False

    def check_5_inference_works(self) -> bool:
        """Check 5: inference endpoint works."""
        try:
            payload = {
                "task": "easy_navigation",
                "seed": 42,
                "max_steps": 30
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
            
            self.log_check("Inference Endpoint Works", passed, f"Status: {response.status_code}")
            return passed
        except Exception as e:
            self.log_check("Inference Endpoint Works", False, str(e))
            return False

    def check_6_grading_works(self) -> bool:
        """Check 6: grading endpoint works."""
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
            self.log_check("Grading Endpoint Works", passed, f"Score: {score}")
            return passed
        except Exception as e:
            self.log_check("Grading Endpoint Works", False, str(e))
            return False

    def run_all(self) -> bool:
        """Run all checks."""
        print("\n" + "="*70)
        print("🚀 OpenEnv Grid World - Pre-Submission Check")
        print("="*70)
        print(f"Space: {self.space_url}")
        print(f"API:   {self.api_url}\n")
        
        self.check_1_space_exists()
        self.check_2_api_responds()
        self.check_3_reset_works()
        self.check_4_step_works()
        self.check_5_inference_works()
        self.check_6_grading_works()
        
        # Summary
        print("\n" + "="*70)
        total = self.checks_passed + self.checks_failed
        print(f"Results: {self.checks_passed}/{total} checks passed")
        
        if self.checks_failed == 0:
            print("\n✨ ALL CHECKS PASSED - Space is ready for submission! ✨")
        else:
            print(f"\n⚠️  {self.checks_failed} check(s) failed - Review and fix above")
        
        print("="*70 + "\n")
        
        return self.checks_failed == 0


def main():
    """Main entry point."""
    checker = PreSubmissionChecker(HF_SPACE_URL)
    success = checker.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
