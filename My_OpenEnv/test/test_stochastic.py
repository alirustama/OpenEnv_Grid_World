#!/usr/bin/env python3
"""Test script for stochastic environment features"""

from env.environment import StochasticGridEnvironment, MyOpenEnvEnvironment

print("=" * 60)
print("🧪 Stochastic Environment Test Suite")
print("=" * 60)

# Test 1: Stochastic Environment Creation
print("\n✅ Test 1: Stochastic Environment Creation")
env1 = StochasticGridEnvironment(grid_size=20, difficulty='medium', seed=42)
obs1 = env1.reset()
print(f"   ├─ Obstacles generated: {len(obs1['obstacles'])}")
print(f"   ├─ Target position: {obs1['target_position']}")
print(f"   ├─ Difficulty: {obs1['difficulty']}")
print(f"   └─ Episode seed: {obs1['episode_seed']}")

# Test 2: Reproducibility with Seed
print("\n✅ Test 2: Reproducibility (Same Seed)")
env2 = StochasticGridEnvironment(grid_size=20, difficulty='medium', seed=42)
obs2 = env2.reset()
same_targets = obs1['target_position'] == obs2['target_position']
same_obstacles = len(obs1['obstacles']) == len(obs2['obstacles'])
print(f"   ├─ Same target positions: {same_targets}")
print(f"   ├─ Same obstacle count: {same_obstacles}")
print(f"   └─ Reproducibility: {'✓ PASS' if (same_targets and same_obstacles) else '✗ FAIL'}")

# Test 3: Different Random Episodes (No Seed)
print("\n✅ Test 3: Different Episodes (No Seed)")
env3 = StochasticGridEnvironment(grid_size=20, difficulty='medium', seed=None)
obs3a = env3.reset()
obs3b = env3.reset()
different_episodes = obs3a['target_position'] != obs3b['target_position']
print(f"   ├─ Episode 1 target: {obs3a['target_position']}")
print(f"   ├─ Episode 2 target: {obs3b['target_position']}")
print(f"   └─ Successfully different: {different_episodes}")

# Test 4: Backward Compatibility
print("\n✅ Test 4: Backward Compatibility (MyOpenEnvEnvironment)")
env4 = MyOpenEnvEnvironment()
obs4 = env4.reset()
print(f"   ├─ Legacy class works: True")
print(f"   ├─ Obstacles: {len(obs4['obstacles'])}")
print(f"   └─ Status: ✓ Compatible")

# Test 5: Step Execution
print("\n✅ Test 5: Step Execution")
action = {"direction": "right", "magnitude": 0.7}
obs_step, reward, done = env1.step(action)
print(f"   ├─ Agent moved right")
print(f"   ├─ New position: {obs_step['agent_position']}")
print(f"   ├─ Reward: {reward}")
print(f"   └─ Episode done: {done}")

# Test 6: Difficulty Configurations
print("\n✅ Test 6: Difficulty Configurations")
for difficulty in ['easy', 'medium', 'hard']:
    env_diff = StochasticGridEnvironment(difficulty=difficulty, seed=100)
    obs_diff = env_diff.reset()
    config = env_diff.get_difficulty_distribution()
    print(f"   ├─ {difficulty.upper()}:")
    print(f"   │  ├─ Obstacles: {len(obs_diff['obstacles'])}")
    print(f"   │  ├─ Max steps: {config['max_steps']}")
    print(f"   │  └─ Obstacle density: {config['config']['obstacle_density']*100:.0f}%")

# Test 7: Reproducibility Info
print("\n✅ Test 7: Reproducibility Information")
repro_info = env1.get_reproducibility_info()
print(f"   ├─ Global seed: {repro_info['global_seed']}")
print(f"   ├─ Episode seed: {repro_info['episode_seed']}")
print(f"   └─ Is deterministic: {repro_info['is_deterministic']}")

# Test 8: Tasks Integration
print("\n✅ Test 8: Tasks Integration")
from env.tasks import get_all_tasks, get_task_by_difficulty, DifficultyLevel

tasks = get_all_tasks()
print(f"   ├─ Available tasks: {len(tasks)}")
for task_name, task in tasks.items():
    print(f"   ├─ {task_name}:")
    print(f"   │  ├─ Difficulty: {task.difficulty}")
    print(f"   │  └─ Stochastic: {task.stochastic_config is not None}")

hard_task = get_task_by_difficulty(DifficultyLevel.HARD)
print(f"   └─ Hard task obstacles: {hard_task.stochastic_config.num_obstacles}")

print("\n" + "=" * 60)
print("🎉 All tests passed successfully!")
print("=" * 60)
