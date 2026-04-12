"""State Schema - Internal environment state"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class State(BaseModel):
    """
    Complete environment state representation
    """
    
    agent_position: List[int] = Field(..., description="Agent position [x, y]")
    target_position: List[int] = Field(..., description="Target position [x, y]")
    obstacles: List[List[int]] = Field(..., description="Obstacle positions")
    step_count: int = Field(default=0, description="Current step number")
    episode_reward: float = Field(default=0.0, description="Total episode reward")
    grid_size: int = Field(default=10, description="Grid size")
    
    class Config:
        """Pydantic config"""
        json_schema_extra = {
            "example": {
                "agent_position": [0, 0],
                "target_position": [9, 9],
                "obstacles": [[2, 2], [3, 3], [5, 5], [7, 7]],
                "step_count": 0,
                "episode_reward": 0.0,
                "grid_size": 10
            }
        }
