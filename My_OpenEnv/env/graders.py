"""Task Graders and Evaluation Logic - Following OpenEnv Standard"""

from typing import Dict, Any, List, Optional
from .tasks import get_all_tasks


class TaskGrader:
    """Evaluate task performance and assign normalized scores [0.0, 1.0]
    
    Implements the OpenEnv scoring standard where all scores are normalized
    to the range [0.0, 1.0], with 0.0 being complete failure and 1.0 being
    perfect performance.
    """
    
    def __init__(self):
        """Initialize grader with task configurations"""
        self.tasks = get_all_tasks()
    
    def grade_trajectory(self, task_name: str, trajectory: Dict[str, Any]) -> float:
        """
        Grade a single episode trajectory - Returns normalized score [0.0, 1.0]
        
        Args:
            task_name: Name of the task (e.g., 'easy_navigation')
            trajectory: Dict with episode metrics:
                - episode_reward: Total reward accumulated
                - steps_taken: Number of steps executed
                - completed: Boolean, whether task was completed
                - partial_reward: Partial reward if not completed
        
        Returns:
            float: Score in [0.0, 1.0] where:
                - 0.0 = Complete failure
                - 0.5 = Partial success (incomplete but made progress)
                - 1.0 = Perfect success
        """
        if task_name not in self.tasks:
            return 0.0
        
        task = self.tasks[task_name]
        episode_reward = trajectory.get("episode_reward", 0.0)
        steps_taken = trajectory.get("steps_taken", 0)
        completed = trajectory.get("completed", False)
        partial_reward = trajectory.get("partial_reward", 0.0)
        
        # Task not completed - score based on partial progress with 0.5 cap
        if not completed:
            # Progress-based partial score (0.0 - 0.5)
            if partial_reward <= 0:
                progress_score = 0.0
            else:
                progress_score = min(0.5, partial_reward / 10.0)
            return max(0.0, min(progress_score, 0.5))
        
        # Task completed - use task's scoring function
        score = task.calculate_score(episode_reward, steps_taken)
        # Ensure score is normalized to [0.0, 1.0]
        return max(0.0, min(score, 1.0))
    
    def calculate_normalized_reward(self, episode_reward: float, max_possible: float = 10.0) -> float:
        """
        Normalize raw episode reward to [0.0, 1.0] range
        
        Args:
            episode_reward: Raw accumulated reward
            max_possible: Maximum possible reward (default 10.0 for target reach)
        
        Returns:
            Normalized reward in [0.0, 1.0]
        """
        if max_possible <= 0:
            return 0.0
        
        normalized = episode_reward / max_possible
        return max(0.0, min(normalized, 1.0))
    
    def grade_all_tasks(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Grade multiple task trajectories
        
        Args:
            results: Dict mapping task names to trajectory dicts
        
        Returns:
            Dict mapping task names to normalized scores [0.0, 1.0]
        """
        scores = {}
        for task_name, trajectory in results.items():
            scores[task_name] = self.grade_trajectory(task_name, trajectory)
        
        return scores
    
    def get_final_score(self, task_scores: Dict[str, float]) -> float:
        """
        Calculate final normalized score from all task scores
        
        Args:
            task_scores: Dict mapping task names to scores
        
        Returns:
            Average score in [0.0, 1.0]
        """
        if not task_scores:
            return 0.0
        
        avg_score = sum(task_scores.values()) / len(task_scores)
        return max(0.0, min(avg_score, 1.0))


class RewardCalculator:
    """Calculate rewards during environment execution"""
    
    @staticmethod
    def distance_reward(current_distance: int, previous_distance: int) -> float:
        """
        Reward for getting closer to target
        
        Args:
            current_distance: Current Manhattan distance to target
            previous_distance: Previous Manhattan distance
        
        Returns:
            Reward value
        """
        if current_distance < previous_distance:
            return 0.1 * (previous_distance - current_distance)
        else:
            return -0.05
    
    @staticmethod
    def collision_penalty() -> float:
        """Penalty for hitting an obstacle"""
        return -1.0
    
    @staticmethod
    def target_bonus(target_reached: bool) -> float:
        """Bonus for reaching target"""
        return 10.0 if target_reached else 0.0
    
    @staticmethod
    def step_penalty() -> float:
        """Small penalty for each step (encourage faster completion)"""
        return -0.01


class MetricsTracker:
    """Track metrics throughout an episode"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.steps = 0
        self.total_reward = 0.0
        self.collisions = 0
        self.target_reached = False
        self.distances = []
    
    def record_step(self, distance: int, reward: float, collision: bool = False):
        """Record a single step"""
        self.steps += 1
        self.total_reward += reward
        self.distances.append(distance)
        if collision:
            self.collisions += 1
    
    def set_target_reached(self, reached: bool = True):
        """Set target reached flag"""
        self.target_reached = reached
    
    def get_summary(self) -> Dict[str, Any]:
        """Get episode summary"""
        return {
            "steps": self.steps,
            "total_reward": self.total_reward,
            "collisions": self.collisions,
            "target_reached": self.target_reached,
            "min_distance": min(self.distances) if self.distances else float('inf'),
            "final_distance": self.distances[-1] if self.distances else float('inf'),
            "average_distance": sum(self.distances) / len(self.distances) if self.distances else 0.0
        }
