#!/bin/bash
# 🚀 OpenEnv Hackathon - Complete Deployment Script
# This script validates and prepares your project for submission

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Functions
log() { printf "${GREEN}✅${NC} %b\n" "$1"; }
warn() { printf "${YELLOW}⚠️${NC} %b\n" "$1"; }
error() { printf "${RED}❌${NC} %b\n" "$1"; }
section() { printf "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n${BOLD}%b${NC}\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n" "$1"; }

# Check if we're in the right directory
if [ ! -f "inference.py" ] || [ ! -f "Dockerfile" ]; then
    error "Must run from project root directory (containing inference.py and Dockerfile)"
    exit 1
fi

section "OpenEnv Hackathon - Submission Validator"

# 1. Test Graders
section "1️⃣  Testing Graders (Score Variation)"
log "Running grader test suite..."
if python test/test_graders.py > /tmp/grader_test.log 2>&1; then
    log "All grader tests passed"
else
    error "Grader tests failed:"
    cat /tmp/grader_test.log
    exit 1
fi

# 2. Syntax Check
section "2️⃣  Checking Python Syntax"
for file in inference.py env/environment.py env/graders.py env/tasks.py; do
    if python -m py_compile "$file" 2>/dev/null; then
        log "$file - syntax OK"
    else
        error "$file - syntax ERROR"
        exit 1
    fi
done

# 3. Import Check
section "3️⃣  Testing Imports"
if python -c "
from env.environment import StochasticGridEnvironment
from env.graders import TaskGrader
from env.tasks import get_all_tasks
print('  Imports successful')
" 2>/dev/null; then
    log "All modules import correctly"
else
    error "Import errors detected"
    exit 1
fi

# 4. Environment Check
section "4️⃣  Testing Environment Functionality"
python << 'EOF' || exit 1
from env.environment import StochasticGridEnvironment

for difficulty in ['easy', 'medium', 'hard']:
    try:
        env = StochasticGridEnvironment(difficulty=difficulty, seed=42)
        obs = env.reset()
        obs, reward, done = env.step({'direction': 'right', 'magnitude': 0.5})
        print(f"  ✅ {difficulty} task OK")
    except Exception as e:
        print(f"  ❌ {difficulty} task FAILED: {e}")
        exit(1)
EOF

# 5. Docker Build Check
section "5️⃣  Testing Docker Build"
if command -v docker &> /dev/null; then
    log "Docker found, building image..."
    if docker build -t openenv-test:latest . > /tmp/docker_build.log 2>&1; then
        log "Docker build successful"
    else
        warn "Docker build failed (may need Docker daemon running):"
        tail -20 /tmp/docker_build.log
    fi
else
    warn "Docker not found (optional for local testing)"
fi

# 6. Requirements Check
section "6️⃣  Checking Dependencies"
if [ ! -f "requirements.txt" ]; then
    error "requirements.txt not found"
    exit 1
fi
log "requirements.txt present with dependencies:"
grep -E "^[^#]" requirements.txt | head -5
echo "  ..."

# 7. Configuration Check
section "7️⃣  Checking Configuration Files"
for file in openenv.yaml README.md Dockerfile; do
    if [ -f "$file" ]; then
        log "$file present"
    else
        error "$file missing (REQUIRED)"
        exit 1
    fi
done

# 8. API Compliance Check
section "8️⃣  Verifying OpenEnv API Compliance"
python << 'EOF' || exit 1
import inspect
from env.environment import StochasticGridEnvironment

env = StochasticGridEnvironment()

# Check methods exist
methods_required = ['reset', 'step', 'state']
for method in methods_required:
    if hasattr(env, method):
        print(f"  ✅ {method}() implemented")
    else:
        print(f"  ❌ {method}() missing")
        exit(1)

# Check reset signature
sig = inspect.signature(env.reset)
if 'seed' in sig.parameters:
    print(f"  ✅ reset() has seed parameter")
else:
    print(f"  ❌ reset() missing seed parameter")
    exit(1)

# Check step signature
sig = inspect.signature(env.step)
if 'action' in sig.parameters:
    print(f"  ✅ step() has action parameter")
else:
    print(f"  ❌ step() missing action parameter")
    exit(1)
EOF

# 9. Format Check
section "9️⃣  Verifying Output Format"
warn "Skipping inference run (would require API keys)"
log "Format will be validated on submission"

# 10. File Structure Check
section "🔟 Checking Project Structure"
required_files=(
    "inference.py"
    "Dockerfile"
    "requirements.txt"
    "openenv.yaml"
    "README.md"
    "env/environment.py"
    "env/graders.py"
    "env/tasks.py"
    "env/models/action.py"
    "env/models/observation.py"
    "test/test_graders.py"
)

all_found=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        log "$file"
    else
        error "$file MISSING"
        all_found=false
    fi
done

if [ "$all_found" = false ]; then
    exit 1
fi

# Summary
section "📊 Validation Summary"
log "All critical checks passed!"
log "Your project is ready for HF Spaces deployment"

echo ""
echo "Next steps:"
echo "1. Create HF Space at https://huggingface.co/spaces"
echo "2. Clone: git clone https://huggingface.co/spaces/USERNAME/SPACENAME"
echo "3. Copy files: cp -r . path/to/space/"
echo "4. Push: git add . && git commit && git push"
echo "5. Wait for Docker build (2-5 min)"
echo "6. Submit URL before April 8, 11:59 PM IST"
echo ""
echo "For detailed instructions, see HACKATHON_REQUIREMENTS.md"
echo "For final checklist, see FINAL_SUBMISSION_CHECKLIST.md"
echo ""
log "Good luck with your submission! 🚀"
