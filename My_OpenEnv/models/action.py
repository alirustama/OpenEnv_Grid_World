"""Action Schema - What agent sends to environment (Standard OpenEnv API)

This module defines the Action schema that agents send to the environment's
step() method to interact with the world.
"""

from pydantic import BaseModel, Field
from typing import Literal


class Action(BaseModel):
    """
    Action schema sent by agent to environment's step() method
    
    This is the standard OpenEnv action format. Agents use this to communicate
    desired movements to the environment.
    
    Attributes:
        direction: Cardinal direction to move (up/down/left/right)
        magnitude: Strength/distance of action (0.0=no movement, 1.0=max movement)
    
    Example:
        Move right with medium strength:
        >>> action = Action(direction="right", magnitude=0.7)
        >>> obs, reward, done = env.step(action)
    """
    
    direction: Literal["up", "down", "left", "right"] = Field(
        ..., 
        description="Cardinal direction: 'up' (y+), 'down' (y-), 'left' (x-), 'right' (x+)"
    )
    
    magnitude: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Action strength: 0.0 (no movement) to 1.0 (max movement, ~3 units)"
    )
    
    class Config:
        """Pydantic config"""
        json_schema_extra = {
            "example": {
                "direction": "right",
                "magnitude": 0.8
            }
        }
        json_encoders = {
            float: lambda v: round(v, 2)
        }
