#!/usr/bin/env python3
"""Run grader tests one by one."""

import sys
from env.graders import TaskGrader

def test_graders():
    """Test grading system for all difficulties."""
    grader = TaskGrader()
    
    print("\n" + "="*70)
    print("🧪 Testing Grading System - One by One")
    print("="*70 + "\n")
    
    tasks = ['easy_navigation', 'medium_navigation', 'hard_navigation']
    test_cases = [
        {
            'name': 'Perfect Performance',
            'episode_reward': 50.0,
            'steps_taken': 20,
            'completed': True,
            'collisions': 0
        },
        {
            'name': 'Good Performance',
            'episode_reward': 25.0,
            'steps_taken': 45,
            'completed': True,
            'collisions': 2
        },
        {
            'name': 'Failed Attempt',
            'episode_reward': 5.0,
            'steps_taken': 100,
            'completed': False,
            'collisions': 15
        }
    ]
    
    for task_idx, task in enumerate(tasks, 1):
        print(f"📋 Task {task_idx}: {task}")
        print("-" * 70)
        
        for case_idx, test_case in enumerate(test_cases, 1):
            try:
                score = grader.grade_trajectory(task, test_case)
                status = "✅ PASS" if score is not None else "❌ FAIL"
                print(f"  {status} | Test {case_idx}: {test_case['name']:<25} Score: {score:.4f}")
            except Exception as e:
                print(f"  ❌ FAIL | Test {case_idx}: {test_case['name']:<25} Error: {str(e)}")
        
        print()
    
    # Test score variation
    print("📊 Verifying Score Variation (Non-Determinism)")
    print("-" * 70)
    
    test_trajectory = {
        'episode_reward': 25.5,
        'steps_taken': 45,
        'completed': True,
        'collisions': 2
    }
    
    scores = []
    for i in range(5):
        score = grader.grade_trajectory('medium_navigation', test_trajectory)
        scores.append(score)
    
    unique_scores = len(set(scores))
    print(f"  ✅ Generated {len(scores)} scores for same input")
    print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
    print(f"  Unique scores: {unique_scores}/5")
    
    if unique_scores > 1:
        print(f"  ✅ PASS | Score variation verified (scores vary as expected)")
    else:
        print(f"  ⚠️  WARNING | All scores identical (may indicate deterministic grading)")
    
    print("\n" + "="*70)
    print("✨ Grading System Tests Complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_graders()
