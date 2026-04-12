#!/usr/bin/env python3
"""Quick Space Check - Rapid validation of HF Space deployment."""

import requests
import sys

# Your HF Space configuration
SPACE_URL = "https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World"
API_URL = "https://indianhacker001-openenv-grid-world.hf.space"

def quick_check():
    """Run quick validation checks."""
    print("\n⚡ Quick Space Check")
    print("=" * 60)
    print(f"Space: {SPACE_URL}")
    print(f"API:   {API_URL}\n")
    
    checks = {
        "1. Space Page": lambda: requests.head(SPACE_URL, timeout=5).status_code == 200,
        "2. API Responds": lambda: requests.get(f"{API_URL}/info", timeout=5).status_code in [200, 404, 405],
        "3. Reset Works": lambda: requests.post(
            f"{API_URL}/api/reset",
            json={"difficulty": "easy", "seed": 42},
            timeout=10
        ).status_code == 200,
    }
    
    passed = 0
    for name, check in checks.items():
        try:
            result = check()
            status = "✅" if result else "❌"
            print(f"{status} {name}")
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {name} - {str(e)[:40]}")
    
    print("\n" + "=" * 60)
    print(f"Result: {passed}/{len(checks)} quick checks passed\n")
    
    return passed == len(checks)


if __name__ == "__main__":
    success = quick_check()
    sys.exit(0 if success else 1)
