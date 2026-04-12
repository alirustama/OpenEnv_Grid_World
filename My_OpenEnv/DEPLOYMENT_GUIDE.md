# 🚀 HF Space Deployment Guide

## Your Space Details

**Space URL**: <https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World>  
**API URL**: <https://indianhacker001-openenv-grid-world.hf.space>  
**Status**: ✅ Ready for Pre-Submission Testing

---

## Quick Start - Test Your Space

### 1️⃣ Quick Check (30 seconds)
```bash
python quick_space_check.py
```

Expected output:
```
✅ 1. Space Page
✅ 2. API Responds
✅ 3. Reset Works

Result: 3/3 quick checks passed
```

### 2️⃣ Full Pre-Submission Check (1-2 minutes)
```bash
python pre_submission_check.py
```

Expected output:
```
✅ PASS | HF Space Page Accessible
✅ PASS | API Endpoint Responds
✅ PASS | Reset Endpoint Works
✅ PASS | Step Endpoint Works
✅ PASS | Inference Endpoint Works
✅ PASS | Grading Endpoint Works

Results: 6/6 checks passed
✨ ALL CHECKS PASSED - Space is ready for submission!
```

---

## Configuration

### Environment Variables (Optional)
```bash
# For local testing if needed
export HF_SPACE_URL="https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World"
export HF_SPACE_API="https://indianhacker001-openenv-grid-world.hf.space"
```

### API Endpoints

Your Space exposes these endpoints:

#### Reset Environment
```bash
curl -X POST https://indianhacker001-openenv-grid-world.hf.space/api/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy", "seed": 42}'
```

#### Take Step
```bash
curl -X POST https://indianhacker001-openenv-grid-world.hf.space/api/step \
  -H "Content-Type: application/json" \
  -d '{"direction": "right", "magnitude": 0.8}'
```

#### Run Inference
```bash
curl -X POST https://indianhacker001-openenv-grid-world.hf.space/api/inference \
  -H "Content-Type: application/json" \
  -d '{"task": "medium_navigation", "seed": 42, "max_steps": 100}'
```

#### Grade Performance
```bash
curl -X POST https://indianhacker001-openenv-grid-world.hf.space/api/grade \
  -H "Content-Type: application/json" \
  -d '{
    "task": "easy_navigation",
    "episode_reward": 25.5,
    "steps_taken": 45,
    "completed": true,
    "collisions": 2
  }'
```

---

## Pre-Submission Checklist

- [ ] Run `python quick_space_check.py` → All 3 checks pass ✅
- [ ] Run `python pre_submission_check.py` → All 6 checks pass ✅
- [ ] Space page loads: <https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World>
- [ ] API responds to `/info` endpoint
- [ ] reset() returns valid observation
- [ ] step() takes actions and returns rewards
- [ ] inference() completes episodes
- [ ] grade() returns scores [0.0, 1.0]

---

## Deployment Timeline

| Date | Status | Action |
|------|--------|--------|
| Now | ✅ Space Created | indianhacker001/OpenEnv_Grid_World |
| Now | ✅ Local Validation | All 9 checks pass |
| Now | ✅ Grader Tests | All 6 tests pass |
| Now | ⏳ Pre-Submission | Run automated checks |
| April 8, 11:59 PM IST | 📅 DEADLINE | Submit Space URL |

---

## Troubleshooting

### Space URL Not Responding

**Issue**: `Connection refused` or timeout

**Solution**:
1. Check HF Space page loads: <https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World>
2. Wait 2-3 minutes for Space to start (may be in sleep mode)
3. Check Space logs on HF dashboard for errors
4. Try with browser first: <https://indianhacker001-openenv-grid-world.hf.space>

### API Returns 404

**Issue**: Endpoint not found

**Solution**:
1. Verify endpoint URL is correct
2. Check Space has deployed recent code
3. Look at Space build logs for errors
4. Try `python quick_space_check.py` for rapid diagnosis

### reset() Fails

**Issue**: `TypeError` or `KeyError` in response

**Solution**:
1. Verify request schema matches expected format
2. Check difficulty is one of: "easy", "medium", "hard"
3. Run local `python -c "from env.environment import StochasticGridEnvironment; print('✅ OK')"`
4. Review Space logs on HF dashboard

---

## Local Testing Before Submission

Test locally first to ensure everything works:

```bash
# 1. Validate framework
python validate-local.py

# 2. Run grader tests
pytest test/test_graders.py -v

# 3. Run environment
python test_graders_manual.py

# 4. Then test Space
python pre_submission_check.py
```

---

## Submission Instructions

Once all checks pass:

1. ✅ Ensure Space is running and responding
2. ✅ Run `python pre_submission_check.py` one final time
3. 📋 Copy your Space URL: `https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World`
4. 🎯 Submit via official form before **April 8, 11:59 PM IST**
5. ✨ Wait for evaluation

---

## Space Management

### Restart Space
- Go to: <https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World/settings>
- Click "Restart Space"

### View Logs
- Click "Logs" tab in Space dashboard
- Check for startup errors or runtime issues

### Update Code
```bash
git push origin main  # If using git with HF
# or update on HF dashboard directly
```

---

## Deployment Success Indicators

✅ **Quick Check**: All 3 checks pass  
✅ **Pre-Submission**: All 6 checks pass  
✅ **API Response Time**: < 5 seconds  
✅ **Grader Variation**: Scores vary by performance  
✅ **Space URL**: Public and accessible  

---

**Space URL**: <https://huggingface.co/spaces/indianhacker001/OpenEnv_Grid_World>  
**API Endpoint**: <https://indianhacker001-openenv-grid-world.hf.space>  
**Status**: 🟢 Ready for Submission
