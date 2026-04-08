"""Main OpenEnv Environment Class - Stochastic Grid World

This module implements the standard OpenEnv environment API with:
- reset(seed: Optional[int] = None) -> Observation: Initialize environment
- step(action: Action) -> Tuple[Observation, float, bool]: Execute action
- state() -> Observation: Get current observation without advancing

The environment is fully stochastic with reproducible randomness via seeds.
"""

import random
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from models.action import Action
from models.observation import Observation


class StochasticGridEnvironment:
    """
    Stochastic Grid World Environment
    
    Features:
    - Randomized obstacle positions at episode start (reproducible with seed)
    - Randomized target positions based on difficulty
    - Configurable obstacle density distributions (easy: low, medium: moderate, hard: high)
    - Support for multiple randomization strategies
    """
    
    # Difficulty-based obstacle density configuration
    DIFFICULTY_CONFIG = {
        'easy': {
            'obstacle_density': 0.10,        # 10% of grid cells
            'num_obstacles': 5,
            'target_min_distance': 5,
            'description': 'Low obstacle density, far target'
        },
        'medium': {
            'obstacle_density': 0.25,        # 25% of grid cells
            'num_obstacles': 15,
            'target_min_distance': 8,
            'description': 'Moderate obstacle density, medium distance'
        },
        'hard': {
            'obstacle_density': 0.40,        # 40% of grid cells
            'num_obstacles': 30,
            'target_min_distance': 10,
            'description': 'High obstacle density, far hard-to-reach target'
        }
    }
    
    def __init__(self, 
                 grid_size: int = 20,
                 seed: Optional[int] = None,
                 difficulty: str = 'medium'):
        """
        Initialize stochastic grid environment
        
        Args:
            grid_size: Size of the square grid (default: 20x20)
            seed: Random seed for reproducibility (None = random)
            difficulty: 'easy', 'medium', or 'hard'
        """
        self.grid_size = grid_size
        self.difficulty = difficulty.lower()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Validate difficulty
        if self.difficulty not in self.DIFFICULTY_CONFIG:
            raise ValueError(f"Difficulty must be one of {list(self.DIFFICULTY_CONFIG.keys())}")
        
        self.config = self.DIFFICULTY_CONFIG[self.difficulty]
        
        # State variables
        self.agent_pos = np.array([0, 0])
        self.target_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        self.obstacles: List[np.ndarray] = []
        self.step_count = 0
        self.episode_reward = 0.0
        self.max_steps = 100 + (self.config['num_obstacles'] * 2)
        self.episode_seed = None
        
    def set_seed(self, seed: Optional[int]) -> None:
        """
        Set the random seed for reproducibility
        
        Args:
            seed: Random seed value (None = random)
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def _generate_stochastic_obstacles(self) -> List[np.ndarray]:
        """
        Generate stochastically distributed obstacles based on difficulty
        
        Uses stratified random placement to ensure coverage across the grid
        and specified obstacle density.
        
        Returns:
            List of obstacle positions as [x, y] coordinates
        """
        obstacles = []
        grid_area = self.grid_size * self.grid_size
        target_num_obstacles = max(
            int(grid_area * self.config['obstacle_density']),
            self.config['num_obstacles']
        )
        
        # Stratified sampling: divide grid into regions and place obstacles randomly
        attempts = 0
        max_attempts = target_num_obstacles * 3
        
        while len(obstacles) < target_num_obstacles and attempts < max_attempts:
            # Random position excluding agent start and nearby cells
            x = self.rng.randint(1, self.grid_size - 1)
            y = self.rng.randint(1, self.grid_size - 1)
            
            pos = np.array([x, y])
            
            # Avoid placing obstacle at agent start or target position
            if not (np.array_equal(pos, self.agent_pos) or 
                    np.array_equal(pos, self.target_pos)):
                # Avoid duplicates
                if not any(np.array_equal(pos, obs) for obs in obstacles):
                    obstacles.append(pos)
            
            attempts += 1
        
        return obstacles
    
    def _generate_stochastic_target(self) -> np.ndarray:
        """
        Generate stochastically placed target position based on difficulty
        
        Returns:
            Target position as [x, y] coordinates
        """
        min_distance = self.config['target_min_distance']
        
        # Keep trying until we find a valid position far enough from agent
        while True:
            x = self.rng.randint(0, self.grid_size)
            y = self.rng.randint(0, self.grid_size)
            
            target = np.array([x, y])
            distance = np.linalg.norm(target - self.agent_pos)
            
            if distance >= min_distance and not np.array_equal(target, self.agent_pos):
                return target
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset environment with stochastic initialization
        
        This method initializes a new episode with stochastically generated
        obstacles and target positions. The randomization is reproducible
        if a seed is provided.
        
        Args:
            seed: Optional seed for reproducible episode generation.
                  If None, uses the environment's global seed.
                  If provided, overrides the global seed for this episode.
        
        Returns:
            observation: Dictionary with environment state matching Observation schema:
                - agent_position: [x, y] coordinates
                - target_position: [x, y] coordinates
                - distance_to_target: Manhattan distance
                - obstacles: List of [x, y] obstacle positions
                - step_count: Steps taken in episode (0 at reset)
                - episode_reward: Cumulative reward (0.0 at reset)
                - grid_size: Size of grid
                - difficulty: Difficulty level ('easy', 'medium', 'hard')
                - episode_seed: Seed used for this episode
        
        Raises:
            ValueError: If difficulty is not valid
        """
        if seed is not None:
            self.set_seed(seed)
            self.episode_seed = seed
        else:
            self.episode_seed = self.seed
        
        # Reset agent and state
        self.agent_pos = np.array([0, 0])
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Generate stochastic elements
        self.target_pos = self._generate_stochastic_target()
        self.obstacles = self._generate_stochastic_obstacles()
        
        return self._get_observation()
    
    def step(self, action: Union[Dict[str, Any], Action]) -> Tuple[Dict[str, Any], float, bool]:
        """
        Execute one step in the environment using standard OpenEnv API
        
        This method executes an action in the environment and returns the
        resulting state, reward, and termination flag. This is the standard
        OpenEnv environment step interface.
        
        Args:
            action: Agent action, either:
                - Dict with 'direction' and 'magnitude' keys
                - Action pydantic model instance
                where direction: "up", "down", "left", "right"
                      magnitude: float [0.0, 1.0]
        
        Returns:
            Tuple of:
            - observation: Dict matching Observation schema with current state
                - agent_position: Updated [x, y] position
                - target_position: [x, y] target location
                - distance_to_target: Manhattan distance to target
                - obstacles: List of obstacle positions
                - step_count: Total steps taken in episode
                - episode_reward: Cumulative reward so far
                - grid_size: Grid dimensions
                - difficulty: Current difficulty level
                - episode_seed: Episode's seed
            - reward: Float reward for this step
                Reward breakdown:
                - +10.0 for reaching target
                - -1.0 for collision with obstacle
                - 1.0/(distance+1) for progressive step rewards
                - -0.01 per step (time penalty)
            - done: Boolean flag indicating episode termination
                True if: (a) target reached, (b) obstacle hit, (c) max steps exceeded
                False if: episode still ongoing
        
        Raises:
            ValueError: If action format is invalid
        """
        # Handle both Action model and dict inputs
        if isinstance(action, Action):
            direction = action.direction
            magnitude = action.magnitude
        elif isinstance(action, dict):
            direction = action.get("direction", "up")
            magnitude = action.get("magnitude", 0.5)
        else:
            raise ValueError(f"Action must be dict or Action model, got {type(action)}")
        
        # Move agent based on direction and magnitude
        step_size = max(1, int(magnitude * 3))  # 0 to 3 units
        
        if direction == "up":
            self.agent_pos[1] = min(self.agent_pos[1] + step_size, self.grid_size - 1)
        elif direction == "down":
            self.agent_pos[1] = max(self.agent_pos[1] - step_size, 0)
        elif direction == "left":
            self.agent_pos[0] = max(self.agent_pos[0] - step_size, 0)
        elif direction == "right":
            self.agent_pos[0] = min(self.agent_pos[0] + step_size, self.grid_size - 1)
        
        self.step_count += 1
        
        # Check if agent is in obstacle
        if any(np.array_equal(self.agent_pos, obs) for obs in self.obstacles):
            reward = -1.0  # Penalty for hitting obstacle
            done = True
        # Check if agent reached target
        elif np.array_equal(self.agent_pos, self.target_pos):
            reward = 10.0  # Big reward for reaching target
            done = True
        else:
            # Distance-based reward (closer to target = higher reward)
            distance = np.abs(self.agent_pos[0] - self.target_pos[0]) + \
                      np.abs(self.agent_pos[1] - self.target_pos[1])
            reward = 1.0 / (distance + 1)  # Inverse distance reward
            done = self.step_count >= self.max_steps
        
        self.episode_reward += reward
        
        return self._get_observation(), reward, done
    
    def state(self) -> Dict[str, Any]:
        """
        Get current environment state without advancing time (standard OpenEnv API)
        
        This method returns the current observation without modifying the environment
        state or incrementing the step counter. It's useful for inspecting the
        environment or implementing custom control logic.
        
        Returns:
            observation: Dict with current environment state (same format as reset/step)
                - agent_position: Current [x, y] position
                - target_position: [x, y] target location
                - distance_to_target: Manhattan distance to target
                - obstacles: List of obstacle positions
                - step_count: Current step count
                - episode_reward: Total reward so far
                - grid_size: Grid dimensions
                - difficulty: Difficulty level
                - episode_seed: Episode seed
        """
        return self._get_observation()
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Generate observation from current state
        
        Returns:
            Observation dictionary
        """
        distance = int(np.abs(self.agent_pos[0] - self.target_pos[0]) + \
                      np.abs(self.agent_pos[1] - self.target_pos[1]))
        
        return {
            "agent_position": self.agent_pos.tolist(),
            "target_position": self.target_pos.tolist(),
            "distance_to_target": distance,
            "obstacles": [obs.tolist() for obs in self.obstacles],
            "step_count": self.step_count,
            "episode_reward": round(self.episode_reward, 4),
            "grid_size": self.grid_size,
            "difficulty": self.difficulty,
            "episode_seed": self.episode_seed
        }
    
    def get_difficulty_distribution(self) -> Dict[str, Any]:
        """
        Get the current difficulty configuration
        
        Returns:
            Dictionary with difficulty parameters and obstacle distribution info
        """
        return {
            "difficulty": self.difficulty,
            "config": self.config,
            "grid_size": self.grid_size,
            "num_obstacles": len(self.obstacles),
            "max_steps": self.max_steps,
            "target_distance": int(np.linalg.norm(self.target_pos - self.agent_pos))
        }
    
    def get_reproducibility_info(self) -> Dict[str, Any]:
        """
        Get seed and reproducibility information
        
        Returns:
            Dictionary with seed information for reproducibility
        """
        return {
            "global_seed": self.seed,
            "episode_seed": self.episode_seed,
            "is_deterministic": self.seed is not None
        }
    
    def get_tasks(self) -> list:
        """Return available tasks"""
        return ["easy_navigation", "medium_navigation", "hard_navigation"]


# Backward compatibility wrapper
class MyOpenEnvEnvironment(StochasticGridEnvironment):
    """Backward compatible wrapper for existing code"""
    
    def __init__(self):
        """Initialize with default configuration"""
        super().__init__(grid_size=20, seed=None, difficulty='medium')
