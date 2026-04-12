"""Observation Schema - What environment sends to agent (Standard OpenEnv API)

This module defines the Observation schema that the environment returns
via reset(), step(), and state() methods. All fields are observed by the agent.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class Observation(BaseModel):
    """
    Observation schema sent by environment to agent
    
    This is the standard OpenEnv observation format returned by reset(), step()
    and state() methods. The agent receives this as a dictionary to perceive
    the environment state and make decisions.
    
    Attributes:
        agent_position: Current agent [x, y] position on grid
        target_position: Target/goal [x, y] position on grid
        distance_to_target: Manhattan distance to target (|dx| + |dy|)
        obstacles: List of obstacle [x, y] positions
        step_count: Number of steps executed in current episode
        episode_reward: Cumulative reward accumulated so far
        grid_size: Size of square grid (grid_size x grid_size)
        difficulty: Task difficulty level (affects obstacle density)
        episode_seed: Random seed used for this episode (for reproducibility)
    """
    
    agent_position: List[int] = Field(
        ...,
        min_items=2,
        max_items=2,
        description="Current [x, y] position of agent on grid"
    )
    
    target_position: List[int] = Field(
        ...,
        min_items=2,
        max_items=2,
        description="Target [x, y] position on grid"
    )
    
    distance_to_target: int = Field(
        ...,
        ge=0,
        description="Manhattan distance (L1 norm) to target"
    )
    
    obstacles: List[List[int]] = Field(
        default_factory=list,
        description="List of obstacle [x, y] positions [[x1, y1], [x2, y2], ...]"
    )
    
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps executed so far in episode"
    )
    
    episode_reward: float = Field(
        default=0.0,
        description="Cumulative reward accumulated in episode"
    )
    
    grid_size: int = Field(
        default=20,
        ge=5,
        description="Size of square grid (grid_size x grid_size)"
    )
    
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium",
        description="Task difficulty: easy (10% obstacles), medium (25%), hard (40%)"
    )
    
    episode_seed: Optional[int] = Field(
        default=None,
        description="Random seed used for this episode (for reproducibility)"
    )
    
    class Config:
        """Pydantic config"""
        json_schema_extra = {
            "example": {
                "agent_position": [0, 0],
                "target_position": [15, 18],
                "distance_to_target": 33,
                "obstacles": [[5, 5], [10, 8], [12, 12]], 
                "step_count": 0,
                "episode_reward": 0.0,
                "grid_size": 20,
                "difficulty": "medium",
                "episode_seed": 42
            }
        }
