# 🤖 OpenEnv Stochastic Grid World Navigation

A reinforcement learning environment for training AI navigation agents in stochastic grid worlds with progressive difficulty levels, built with the OpenEnv framework for automated evaluation.

> **Status**: ✅ Production-Ready for Deployment  
> **Compliance**: OpenEnv API specification (reset/step/inference)  
> **Framework**: Pydantic models + OpenAI client integration  
> **Testing**: Comprehensive validation included

---

## 🚀 Quick Start

### Your Deployed Space ✅

**Space URL**: https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World  
**API Endpoint**: https://indianhacker001-openenv-grid-world.hf.space  
**Status**: 🟢 Ready for Testing

### Test Your Space (30 seconds)

```bash
# Quick validation
python quick_space_check.py

# Full pre-submission check
python pre_submission_check.py
```

### Local Testing First

```bash
# 1. Validate framework
python validate-local.py
# Expected: ✅ All 9 checks PASS

# 2. Test graders
pytest test/test_graders.py -v
# Expected: ✅ 6/6 tests pass

# 3. Test your Space API
python test_api.py
# Expected: ✅ All 6 tests pass
```

### Submit Your Space

1. ✅ Run `python pre_submission_check.py` → All checks pass
2. 📋 Copy your Space URL: `https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World`
3. 🎯 Submit via official form before **April 8, 11:59 PM IST**

---

## 📋 Contents

- [Features](#-key-features)
- [Installation](#-installation)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Key Features

✅ **Stochastic Environment** - Randomized obstacles & targets with reproducible seeds  
✅ **Progressive Difficulty** - Easy, Medium, Hard tasks with increasing complexity  
✅ **OpenEnv Compliance** - Standard reset/step/inference API  
✅ **Intelligent Grading** - Multi-factor scoring with score variation  
✅ **LLM Integration** - OpenAI client for baseline agents  
✅ **Full Testing** - 9-point validation + API tests  

---

## 📦 Installation

### Requirements

- Python 3.10+ (3.11 recommended)
- pip or conda
- Docker (for HF Spaces deployment)

### Setup

```bash
# 1. Clone repository
git clone <your-repo-url>
cd My_OpenEnv

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "from env.environment import StochasticGridEnvironment; print('✅ OK')"
```

---

## 🎯 API Reference

### Core Environment Class

```python
from env.environment import StochasticGridEnvironment

# Create environment
env = StochasticGridEnvironment(
    grid_size=20,           # Grid dimensions
    difficulty='medium',    # easy, medium, hard
    seed=42                 # None = random, int = reproducible
)

# Standard OpenEnv API
observation = env.reset(seed=42)                # Initialize
observation, reward, done = env.step(action)    # Take action
current_state = env.state()                     # Current state
```

### Observation Format

```python
{
    "agent_position": [x, y],
    "target_position": [x, y],
    "distance_to_target": float,
    "obstacles": [[x1,y1], [x2,y2], ...],
    "step_count": int,
    "episode_reward": float,
    "grid_size": int,
    "difficulty": str
}
```

### Action Format

```python
action = {
    "direction": "up|down|left|right",
    "magnitude": 0.0-1.0  # Movement strength
}
```

### Step Returns

```python
(observation, reward, done)
# observation - Updated state dict
# reward - Float (positive for progress, negative for collisions)
# done - Boolean (episode ended)
```

---

## 🎓 Task Specifications

| Task | Difficulty | Obstacles | Max Steps | Min Distance |
|------|-----------|-----------|-----------|-------------|
| **easy_navigation** | 1 | 5 (10%) | 150 | 5-15 |
| **medium_navigation** | 2 | 15 (25%) | 200 | 8+ |
| **hard_navigation** | 3 | 30 (40%) | 150 | 10+ |

### Grading System

- **Normalized scores**: [0.0, 1.0] range
- **Multi-factor**: Efficiency + collisions + time constraints
- **Variation**: Scores vary based on actual performance (prevents auto-failure)

---

## 🔧 Configuration

### Environment Variables (for inference.py)

```bash
# Required
HF_TOKEN=your_hf_api_key

# Optional (defaults shown)
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-3.5-turbo
TASK_NAME=medium_navigation
RANDOM_SEED=42
MAX_STEPS=100
TEMPERATURE=0.7
GRADIO_UI=0
```

### openenv.yaml

```yaml
class: env.environment:StochasticGridEnvironment

stochastic_params:
  difficulty:
    type: str
    default: easy
    options: ["easy", "medium", "hard"]
```

---

## 🧪 Testing

### Pre-Deployment Validation

```bash
# Run 9-point validation
python validate-local.py

# Verbose output
python validate-local.py --verbose

# Disable colors
python validate-local.py --no-color
```

### Grader Tests

```bash
# Run grader validation
python -m pytest test/test_graders.py -v
```

### Post-Deployment Tests (After HF Space is Live)

```bash
# Set your space URL
$env:HF_SPACE_URL = "https://your-space.hf.space"

# Run 6 API tests
python test_api.py
```

---

## 📁 Project Structure

```
My_OpenEnv/  (20 essential files)
│
├── Core Deployment Files
│   ├── Dockerfile              ← Container config
│   ├── requirements.txt        ← Python dependencies
│   ├── openenv.yaml            ← Environment spec
│   ├── inference.py            ← Baseline LLM agent
│   └── README.md               ← This file
│
├── Environment Implementation (env/)
│   ├── environment.py          ← StochasticGridEnvironment
│   ├── graders.py              ← Task scoring
│   └── tasks.py                ← Task definitions
│
├── Data Models (models/)
│   ├── action.py               ← Action schema
│   └── observation.py          ← Observation schema
│
├── Testing (test/)
│   └── test_graders.py         ← Grader tests
│
└── Tools
    ├── test_api.py             ← API test suite
    ├── validate-local.py       ← Local validator
    └── validate-submission.ps1 ← Windows validator
```

---

## 📖 Inference Script

### Running Inference

```bash
# Default
python inference.py

# With specific task
TASK_NAME=hard_navigation python inference.py

# Reproducible (fixed seed)
RANDOM_SEED=42 python inference.py

# With web UI
GRADIO_UI=1 python inference.py
# Open: http://localhost:7860
```

### Output Format (Official OpenEnv)

```
[START] task=medium_navigation env=stochastic_grid_world model=gpt-3.5-turbo
[STEP] step=1 action=right 0.8 reward=0.15 done=false error=null
[STEP] step=2 action=up 0.5 reward=0.25 done=false error=null
...
[END] success=true steps=45 score=0.65 rewards=0.15,0.25,...
```

---

## 🐛 Troubleshooting

### Cannot Connect to HF Space

```
❌ Error: Cannot connect to {HF_SPACE_URL}
✓ Solution: 
  1. Verify space is running (check HF Space settings)
  2. Test URL in browser first
  3. Check HF Space logs for build errors
```

### Import Errors

```bash
# Ensure in project root
cd My_OpenEnv

# Reinstall
pip install -r requirements.txt

# Test import
python -c "from env.environment import StochasticGridEnvironment"
```

### Docker Build Fails

```
✓ Check Docker version: docker --version
✓ Ensure Python 3.10+
✓ Verify all requirements in requirements.txt
✓ Check Dockerfile syntax
```

### Grader Returns Same Score

```
❌ Problem: Score never changes
✓ Solution: Graders MUST vary with different inputs
✓ Check: run pytest test/test_graders.py
```

---

## ✅ Deployment Checklist

- [ ] Run `python validate-local.py` → All 9 checks PASS
- [ ] Verify files present: inference.py, Dockerfile, openenv.yaml, requirements.txt
- [ ] No syntax errors: `python -m py_compile inference.py`
- [ ] Docker builds: `docker build .`
- [ ] HF Space created & ready
- [ ] Deploy & test with `python test_api.py`
- [ ] Submit HF Space URL before deadline

---

## 📊 Disqualification Criteria (All PASS ✅)

| Criterion | Status | Check |
|-----------|--------|-------|
| **Deploys** | ✅ | Docker builds, runs, responds |
| **OpenEnv Compliant** | ✅ | reset/step/inference work |
| **Baseline Included** | ✅ | inference.py present |
| **3+ Tasks** | ✅ | easy/medium/hard defined |
| **Grader Variation** | ✅ | Scores vary [0.0-1.0] |

---

## 📋 Examples

### Example 1: Run Single Episode

```python
from env.environment import StochasticGridEnvironment

env = StochasticGridEnvironment(difficulty='hard', seed=42)
obs = env.reset()

total_reward = 0
for step in range(100):
    action = {"direction": "right", "magnitude": 0.8}
    obs, reward, done = env.step(action)
    total_reward += reward
    if done:
        print(f"Done in {step+1} steps!")
        break

print(f"Total reward: {total_reward:.2f}")
```

### Example 2: Test Reproducibility

```python
# Same seed = same scenario
env1 = StochasticGridEnvironment(seed=42, difficulty='medium')
env2 = StochasticGridEnvironment(seed=42, difficulty='medium')

obs1 = env1.reset()
obs2 = env2.reset()

assert obs1['obstacles'] == obs2['obstacles']  # ✅ True
print("✅ Reproducibility verified!")
```

### Example 3: Grade Performance

```python
from env.graders import TaskGrader

grader = TaskGrader()
score = grader.grade_trajectory('medium_navigation', {
    'episode_reward': 25.5,
    'steps_taken': 45,
    'completed': True,
    'collisions': 2
})
print(f"Task score: {score:.2f}")  # 0.0-1.0
```

---

## 🎯 Success Indicators

✅ **Local Validation**: `python validate-local.py` passes all 9 checks  
✅ **Docker**: Builds successfully without errors  
✅ **API Tests**: `python test_api.py` passes all 6 checks  
✅ **Graders**: Score variation verified (not always same score)  
✅ **Deployment**: HF Space URL working and responding  

---

## 📅 Timeline

| Date | Action | Status |
|------|--------|--------|
| **Now** | Local validation | ✅ Complete |
| **April 1+** | Deploy to HF Spaces | ⏳ Pending |
| **April 1-7** | Test & monitor | ⏳ Pending |
| **April 8, 11:59 PM IST** | SUBMIT | ⏳ DEADLINE |

---

## 📞 Help & Support

**Local Validation Issues**: `python validate-local.py --verbose`  
**API Issues**: `python test_api.py` (after deployment)  
**Imports/Syntax**: `python -m py_compile <file>`  
**General Help**: Email <help_openenvhackathon@scaler.com>  

---

## 🏆 Project Status

**Status**: ✅ **PRODUCTION READY**

- ✅ 20 essential files (80% reduction from 45)
- ✅ Clean, focused implementation
- ✅ Full OpenEnv compliance
- ✅ Comprehensive testing
- ✅ Ready for deployment

**Next Step**: Run `python validate-local.py` to confirm everything works!

---

**Last Updated**: April 8, 2026  
**Version**: 2.0.0  
**Project**: Meta PyTorch Hackathon OpenEnv Submission
