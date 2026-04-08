"""Task Definitions for OpenEnv - Stochastic Environment Edition"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class DifficultyLevel(Enum):
    """Difficulty levels with stochastic parameters"""
    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'


@dataclass
class StochasticConfig:
    """Configuration for stochastic environment parameters"""
    obstacle_density: float          # Fraction of grid occupied by obstacles
    num_obstacles: int              # Number of obstacles to place
    target_min_distance: int        # Minimum distance from agent to target
    target_max_distance: Optional[int]  # Maximum distance (if any)
    seed: Optional[int]             # Seed for reproducibility
    description: str                # Human-readable description


@dataclass
class Task:
    """Enhanced Task class with stochastic parameters"""
    name: str
    description: str
    difficulty: int
    max_steps: int
    target_distance: int
    stochastic_config: Optional[StochasticConfig] = None
    reward_threshold: float = 5.0
    
    def __post_init__(self):
        """Validate task parameters"""
        if self.difficulty not in [1, 2, 3]:
            raise ValueError("Difficulty must be 1 (easy), 2 (medium), or 3 (hard)")


@dataclass
class EasyNavigation:
    """Easy Navigation Task - Beginner Level
    
    Stochastic Features:
    - 10% grid cell obstacle density
    - 5-8 obstacles randomly placed
    - Target at least 5 units away from agent
    """
    name: str = "easy_navigation"
    description: str = "Navigate to a nearby target with few obstacles"
    difficulty: int = 1
    max_steps: int = 150
    target_distance: int = 5
    reward_threshold: float = 8.0
    stochastic_config: Optional[StochasticConfig] = field(default_factory=lambda: StochasticConfig(
        obstacle_density=0.10,
        num_obstacles=5,
        target_min_distance=5,
        target_max_distance=15,
        seed=None,
        description="Low obstacle density - lots of open space, easy escape routes"
    ))
    
    def check_completion(self, state: Dict[str, Any]) -> bool:
        """Check if task is completed"""
        return state.get("distance_to_target", float('inf')) == 0
    
    def calculate_score(self, episode_reward: float, steps_taken: int, 
                       collision_count: int = 0, path_efficiency: float = 1.0) -> float:
        """Calculate task score (0.0 to 1.0) considering stochasticity
        
        Args:
            episode_reward: Total reward accumulated in episode
            steps_taken: Number of steps to completion
            collision_count: Number of obstacle collisions
            path_efficiency: Ratio of optimal path to actual path
        
        Returns:
            Score between 0.0 and 1.0
        """
        # Penalize collisions more heavily (0.5 per collision)
        collision_penalty = collision_count * 0.5
        score = episode_reward - collision_penalty
        
        if score >= 10.0:
            return 1.0
        elif score >= 7.0:
            return 0.75
        elif score >= 4.0:
            return 0.5
        elif score >= 1.0:
            return 0.25
        else:
            return 0.0


@dataclass
class MediumNavigation:
    """Medium Navigation Task - Intermediate Level
    
    Stochastic Features:
    - 25% grid cell obstacle density
    - 15-20 obstacles randomly scattered
    - Target at least 8 units away
    - Requires obstacle avoidance strategy
    - Different obstacle distributions per episode
    """
    name: str = "medium_navigation"
    description: str = "Navigate through moderate obstacles to reach target"
    difficulty: int = 2
    max_steps: int = 200
    target_distance: int = 8
    reward_threshold: float = 12.0
    stochastic_config: Optional[StochasticConfig] = field(default_factory=lambda: StochasticConfig(
        obstacle_density=0.25,
        num_obstacles=15,
        target_min_distance=8,
        target_max_distance=None,
        seed=None,
        description="Moderate obstacle density - significant planning required"
    ))
    
    def check_completion(self, state: Dict[str, Any]) -> bool:
        """Check if task is completed"""
        return state.get("distance_to_target", float('inf')) == 0
    
    def calculate_score(self, episode_reward: float, steps_taken: int,
                       collision_count: int = 0, path_efficiency: float = 1.0) -> float:
        """Calculate task score (0.0 to 1.0) with efficiency bonus
        
        Args:
            episode_reward: Total reward accumulated
            steps_taken: Number of steps taken
            collision_count: Number of collisions
            path_efficiency: Ratio of shortest path to actual path (0-1]
        
        Returns:
            Score between 0.0 and 1.0
        """
        collision_penalty = collision_count * 0.1
        efficiency_bonus = path_efficiency * 0.2 if path_efficiency > 0 else 0.0
        
        score = episode_reward - collision_penalty + efficiency_bonus
        
        if score >= 18.0 and steps_taken <= 120:
            return 1.0
        elif score >= 14.0:
            return 0.75
        elif score >= 9.0:
            return 0.5
        elif score >= 5.0:
            return 0.25
        else:
            return 0.0


@dataclass
class HardNavigation:
    """Hard Navigation Task - Expert Level
    
    Stochastic Features:
    - 40% grid cell obstacle density
    - 30-35 obstacles creating tight corridors
    - Target 10+ units away
    - Highly stochastic - different layout each episode
    - Requires optimal pathfinding and collision avoidance
    """
    name: str = "hard_navigation"
    description: str = "Navigate through dense obstacles with time constraint"
    difficulty: int = 3
    max_steps: int = 150
    target_distance: int = 10
    reward_threshold: float = 15.0
    stochastic_config: Optional[StochasticConfig] = field(default_factory=lambda: StochasticConfig(
        obstacle_density=0.40,
        num_obstacles=30,
        target_min_distance=10,
        target_max_distance=None,
        seed=None,
        description="High obstacle density - tight corridors, challenging navigation"
    ))
    
    def check_completion(self, state: Dict[str, Any]) -> bool:
        """Check if task is completed"""
        return state.get("distance_to_target", float('inf')) == 0
    
    def calculate_score(self, episode_reward: float, steps_taken: int,
                       collision_count: int = 0, path_efficiency: float = 1.0) -> float:
        """Calculate task score (0.0 to 1.0) with strict criteria
        
        Args:
            episode_reward: Total reward accumulated
            steps_taken: Number of steps taken
            collision_count: Number of collisions
            path_efficiency: Ratio of shortest to actual path
        
        Returns:
            Score between 0.0 and 1.0
        """
        collision_penalty = collision_count * 0.15
        efficiency_bonus = path_efficiency * 0.3 if path_efficiency > 0 else 0.0
        
        # Strict time limit bonus
        time_bonus = 0.5 if steps_taken <= 100 else 0.0
        
        score = episode_reward - collision_penalty + efficiency_bonus + time_bonus
        
        if score >= 25.0 and steps_taken <= 80:
            return 1.0
        elif score >= 20.0:
            return 0.75
        elif score >= 12.0:
            return 0.5
        elif score >= 7.0:
            return 0.25
        else:
            return 0.0


def get_all_tasks() -> Dict[str, Any]:
    """Get all available tasks
    
    Returns:
        Dictionary mapping task names to Task objects
    """
    return {
        'easy_navigation': EasyNavigation(),
        'medium_navigation': MediumNavigation(),
        'hard_navigation': HardNavigation()
    }


def get_task_by_difficulty(difficulty: DifficultyLevel) -> Any:
    """Get a task by difficulty level
    
    Args:
        difficulty: DifficultyLevel enum value
    
    Returns:
        Task object
    """
    tasks = get_all_tasks()
    difficulty_map = {
        DifficultyLevel.EASY: 'easy_navigation',
        DifficultyLevel.MEDIUM: 'medium_navigation',
        DifficultyLevel.HARD: 'hard_navigation'
    }
    return tasks[difficulty_map[difficulty]]
