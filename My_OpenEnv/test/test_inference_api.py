#!/usr/bin/env python3
"""Quick test of inference script with limited steps"""

import json
from datetime import datetime
from env.environment import StochasticGridEnvironment
from models.action import Action
from env.graders import TaskGrader, MetricsTracker

print("Testing OpenEnv API compliance in inference pipeline...\n")

# Create environment
env = StochasticGridEnvironment(grid_size=20, difficulty='easy', seed=100)

# Start event
print(json.dumps({
    "event": "START",
    "timestamp": datetime.now().isoformat(),
    "task": "easy_navigation",
    "difficulty": "easy"
}))

# Reset using standard API
obs = env.reset()
print(json.dumps({
    "event": "RESET",
    "agent_position": obs['agent_position'],
    "target_position": obs['target_position'],
    "obstacles_count": len(obs['obstacles']),
    "grid_size": obs['grid_size']
}))

# Run a few steps
tracker = MetricsTracker()
for step in range(5):
    # Check state without stepping
    state_check = env.state()
    assert state_check['step_count'] == step, "state() should not advance"
    
    # Create action using new API
    agent_x, agent_y = obs['agent_position']
    target_x, target_y = obs['target_position']
    direction = 'right' if agent_x < target_x else 'left'
    action = Action(direction=direction, magnitude=0.8)
    
    # Execute step
    obs, reward, done = env.step(action)
    
    tracker.record_step(
        distance=obs['distance_to_target'],
        reward=reward,
        collision=(reward < -0.5)
    )
    
    print(json.dumps({
        "event": "STEP",
        "step": step,
        "action": {"direction": action.direction, "magnitude": action.magnitude},
        "reward": round(reward, 4),
        "distance_to_target": obs['distance_to_target'],
        "agent_position": obs['agent_position'],
        "episode_reward": round(obs['episode_reward'], 4),
        "done": done
    }))
    
    if done:
        break

# End event
summary = tracker.get_summary()
print(json.dumps({
    "event": "END",
    "task": "easy_navigation",
    "steps": summary['steps'],
    "total_reward": round(summary['total_reward'], 4),
    "target_reached": summary['target_reached'],
    "final_distance": summary['final_distance']
}))

print("\n✅ Inference pipeline test passed!")
print("   ├─ reset() returns observation ✓")
print("   ├─ step() accepts Action model ✓")
print("   ├─ state() non-destructive ✓")
print("   ├─ JSON output format correct ✓")
print("   └─ Standard OpenEnv API working ✓")
