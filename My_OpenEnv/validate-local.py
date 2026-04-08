#!/usr/bin/env python3
"""
validate-local.py — OpenEnv Local Pre-Submission Validator

Comprehensive validation script for local testing before deployment.
Tests all critical components to ensure submission readiness.

Prerequisites:
  - Python 3.8+
  - All packages: pip install -r requirements.txt

Run:
  python validate-local.py [--verbose] [--no-color]

Examples:
  python validate-local.py
  python validate-local.py --verbose
  python validate-local.py --no-color
"""

import os
import sys
import py_compile
import argparse
from pathlib import Path

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BOLD = '\033[1m'
    NC = '\033[0m'
    
    @staticmethod
    def disable():
        Colors.RED = Colors.GREEN = Colors.YELLOW = Colors.BOLD = Colors.NC = ''

def print_header(text):
    """Print section header"""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.NC}")
    print(f"{Colors.BOLD}  {text}{Colors.NC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.NC}\n")

def check_syntax(verbose=False):
    """1️⃣  Check Python syntax"""
    print_header("1️⃣  SYNTAX CHECK")
    files = [
        'inference.py',
        'env/environment.py',
        'env/graders.py',
        'env/tasks.py',
        'models/action.py',
        'models/observation.py'
    ]
    
    all_valid = True
    for f in files:
        if not os.path.exists(f):
            print(f"   {Colors.RED}❌{Colors.NC} {f} - NOT FOUND")
            all_valid = False
            continue
        try:
            py_compile.compile(f)
            print(f"   {Colors.GREEN}✅{Colors.NC} {f}")
        except Exception as e:
            print(f"   {Colors.RED}❌{Colors.NC} {f}: {e}")
            all_valid = False
    
    return all_valid

def check_imports(verbose=False):
    """2️⃣  Check module imports"""
    print_header("2️⃣  IMPORT CHECK")
    
    try:
        from env.environment import StochasticGridEnvironment
        print(f"   {Colors.GREEN}✅{Colors.NC} env.environment.StochasticGridEnvironment")
    except Exception as e:
        print(f"   {Colors.RED}❌{Colors.NC} env.environment: {e}")
        return False
    
    try:
        from env.graders import TaskGrader
        print(f"   {Colors.GREEN}✅{Colors.NC} env.graders.TaskGrader")
    except Exception as e:
        print(f"   {Colors.RED}❌{Colors.NC} env.graders: {e}")
        return False
    
    try:
        from env.tasks import get_all_tasks
        print(f"   {Colors.GREEN}✅{Colors.NC} env.tasks.get_all_tasks")
    except Exception as e:
        print(f"   {Colors.RED}❌{Colors.NC} env.tasks: {e}")
        return False
    
    try:
        from models import Action, Observation
        print(f"   {Colors.GREEN}✅{Colors.NC} models.Action, models.Observation")
    except Exception as e:
        print(f"   {Colors.RED}❌{Colors.NC} models: {e}")
        return False
    
    print(f"   {Colors.GREEN}✅{Colors.NC} All module imports successful\n")
    return True

def check_environment(verbose=False):
    """3️⃣  Test environment functionality"""
    print_header("3️⃣  ENVIRONMENT FUNCTIONALITY")
    
    try:
        from env.environment import StochasticGridEnvironment
        
        for difficulty in ['easy', 'medium', 'hard']:
            try:
                env = StochasticGridEnvironment(difficulty=difficulty, seed=42)
                obs = env.reset(seed=42)
                
                if not isinstance(obs, dict):
                    print(f"   {Colors.RED}❌{Colors.NC} {difficulty}: reset() did not return dict")
                    return False
                
                required_obs_keys = ['agent_position', 'target_position', 'distance_to_target']
                missing = [k for k in required_obs_keys if k not in obs]
                if missing:
                    print(f"   {Colors.RED}❌{Colors.NC} {difficulty}: missing observation keys: {missing}")
                    return False
                
                obs, reward, done = env.step({'direction': 'right', 'magnitude': 0.5})
                
                if not all([isinstance(obs, dict), isinstance(reward, (int, float)), isinstance(done, bool)]):
                    print(f"   {Colors.RED}❌{Colors.NC} {difficulty}: step() returned invalid types")
                    return False
                
                print(f"   {Colors.GREEN}✅{Colors.NC} {difficulty} environment works")
            except Exception as e:
                print(f"   {Colors.RED}❌{Colors.NC} {difficulty}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                return False
        
        return True
    except Exception as e:
        print(f"   {Colors.RED}❌{Colors.NC} Environment test failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

def check_graders(verbose=False):
    """4️⃣  Test grader functionality and score variation"""
    print_header("4️⃣  GRADER TESTS")
    
    try:
        from env.graders import TaskGrader
        grader = TaskGrader()
        
        test_trajectories = [
            {'episode_reward': 5.0, 'steps_taken': 10, 'completed': True, 'partial_reward': 0.0, 'collisions': 0},
            {'episode_reward': 2.0, 'steps_taken': 20, 'completed': False, 'partial_reward': 0.5, 'collisions': 2},
            {'episode_reward': 0.0, 'steps_taken': 50, 'completed': False, 'partial_reward': 0.0, 'collisions': 5}
        ]
        
        all_passed = True
        for task in ['easy_navigation', 'medium_navigation', 'hard_navigation']:
            scores = []
            try:
                for traj in test_trajectories:
                    score = grader.grade_trajectory(task, traj)
                    if not (0.0 <= score <= 1.0):
                        print(f"   {Colors.RED}❌{Colors.NC} {task}: score {score} out of range [0.0, 1.0]")
                        all_passed = False
                    scores.append(score)
                
                unique_scores = len(set(scores))
                if unique_scores == 1:
                    print(f"   {Colors.YELLOW}⚠️{Colors.NC}  {task}: grader returns same score (bad - must vary!)")
                    all_passed = False
                else:
                    print(f"   {Colors.GREEN}✅{Colors.NC} {task}: scores vary ✓ {scores}")
            except Exception as e:
                print(f"   {Colors.RED}❌{Colors.NC} {task}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"   {Colors.RED}❌{Colors.NC} Grader test failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

def check_required_files(verbose=False):
    """5️⃣  Check required files exist"""
    print_header("5️⃣  REQUIRED FILES")
    
    required = {
        'inference.py': 'Baseline agent script',
        'Dockerfile': 'Container configuration',
        'requirements.txt': 'Python dependencies',
        'openenv.yaml': 'Environment specification',
        'README.md': 'Documentation'
    }
    
    all_exist = True
    for filename, description in required.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"   {Colors.GREEN}✅{Colors.NC} {filename:20s} ({size:5d}B) — {description}")
        else:
            print(f"   {Colors.RED}❌{Colors.NC} {filename:20s} — MISSING ({description})")
            all_exist = False
    
    return all_exist

def check_directory_structure(verbose=False):
    """6️⃣  Check directory structure"""
    print_header("6️⃣  DIRECTORY STRUCTURE")
    
    dirs = {
        'env': ['__init__.py', 'environment.py', 'graders.py', 'tasks.py'],
        'models': ['__init__.py', 'action.py', 'observation.py'],
        'test': ['__init__.py', 'test_graders.py']
    }
    
    all_valid = True
    for dirname, files in dirs.items():
        if not os.path.isdir(dirname):
            print(f"   {Colors.RED}❌{Colors.NC} Directory missing: {dirname}/")
            all_valid = False
            continue
        
        print(f"   {Colors.GREEN}✅{Colors.NC} {dirname}/")
        for filename in files:
            filepath = os.path.join(dirname, filename)
            if os.path.exists(filepath):
                print(f"       ├─ {filename}")
            else:
                print(f"       └─ {Colors.RED}❌ {filename} - MISSING{Colors.NC}")
                all_valid = False
    
    return all_valid

def check_openenv_yaml(verbose=False):
    """7️⃣  Check openenv.yaml structure"""
    print_header("7️⃣  OPENENV.YAML VALIDATION")
    
    try:
        import yaml
        
        if not os.path.exists('openenv.yaml'):
            print(f"   {Colors.RED}❌{Colors.NC} openenv.yaml not found")
            return False
        
        with open('openenv.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['class', 'stochastic_params']
        missing = [k for k in required_keys if k not in config]
        
        if missing:
            print(f"   {Colors.RED}❌{Colors.NC} openenv.yaml missing keys: {missing}")
            return False
        
        print(f"   {Colors.GREEN}✅{Colors.NC} openenv.yaml structure valid")
        print(f"       ├─ class: {config['class']}")
        print(f"       └─ stochastic_params: configured")
        return True
    except Exception as e:
        print(f"   {Colors.RED}❌{Colors.NC} openenv.yaml validation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

def check_dockerfile(verbose=False):
    """8️⃣  Check Dockerfile content"""
    print_header("8️⃣  DOCKERFILE VALIDATION")
    
    if not os.path.exists('Dockerfile'):
        print(f"   {Colors.RED}❌{Colors.NC} Dockerfile not found")
        return False
    
    with open('Dockerfile', 'r') as f:
        content = f.read()
    
    required_strings = [
        'FROM python:3.1',
        'requirements.txt',
        'inference.py',
        'EXPOSE 7860'
    ]
    
    all_valid = True
    for req in required_strings:
        if req in content:
            print(f"   {Colors.GREEN}✅{Colors.NC} Contains: {req}")
        else:
            print(f"   {Colors.YELLOW}⚠️{Colors.NC}  Missing: {req}")
            all_valid = False
    
    return all_valid

def check_requirements(verbose=False):
    """9️⃣  Check requirements.txt content"""
    print_header("9️⃣  REQUIREMENTS.TXT VALIDATION")
    
    if not os.path.exists('requirements.txt'):
        print(f"   {Colors.RED}❌{Colors.NC} requirements.txt not found")
        return False
    
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    required_packages = [
        'pydantic',
        'numpy',
        'openenv',
        'pyyaml'
    ]
    
    all_valid = True
    for pkg in required_packages:
        if pkg.lower() in content.lower():
            print(f"   {Colors.GREEN}✅{Colors.NC} Contains: {pkg}")
        else:
            print(f"   {Colors.YELLOW}⚠️{Colors.NC}  Missing: {pkg}")
    
    return all_valid

def main():
    parser = argparse.ArgumentParser(
        description='OpenEnv Local Pre-Submission Validator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python validate-local.py
  python validate-local.py --verbose
  python validate-local.py --no-color
        '''
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    
    args = parser.parse_args()
    
    if args.no_color:
        Colors.disable()
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.NC}")
    print(f"{Colors.BOLD}  OpenEnv Local Pre-Submission Validator{Colors.NC}")
    print(f"{Colors.BOLD}  April 8, 2026{Colors.NC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.NC}\n")
    
    checks = [
        ('Syntax', check_syntax),
        ('Imports', check_imports),
        ('Environment', check_environment),
        ('Graders', check_graders),
        ('Files', check_required_files),
        ('Structure', check_directory_structure),
        ('Config', check_openenv_yaml),
        ('Dockerfile', check_dockerfile),
        ('Requirements', check_requirements)
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn(verbose=args.verbose)
            results.append((name, result))
        except Exception as e:
            print(f"\n{Colors.RED}❌ Unexpected error in {name}: {e}{Colors.NC}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print_header("VALIDATION SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{Colors.GREEN}✅ PASS{Colors.NC}" if result else f"{Colors.RED}❌ FAIL{Colors.NC}"
        print(f"  {status} — {name}")
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.NC}")
    
    if all(result for _, result in results):
        print(f"{Colors.GREEN}{Colors.BOLD}✨ ALL CHECKS PASSED ({passed}/{total}){Colors.NC}")
        print(f"{Colors.GREEN}{Colors.BOLD}Your project is ready for submission!{Colors.NC}\n")
        print("📋 Next steps:")
        print("   1. Deploy to HF Spaces (after April 1)")
        print("   2. Run: python test_api.py (set HF_SPACE_URL env var)")
        print("   3. Submit HF Space URL (before April 8, 11:59 PM IST)\n")
        print(f"{Colors.BOLD}{'='*70}{Colors.NC}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ SOME CHECKS FAILED ({passed}/{total}){Colors.NC}")
        print(f"{Colors.RED}{Colors.BOLD}Fix the issues above and try again.{Colors.NC}\n")
        if not args.verbose:
            print("💡 Tip: Run with --verbose for more details\n")
        print(f"{Colors.BOLD}{'='*70}{Colors.NC}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
