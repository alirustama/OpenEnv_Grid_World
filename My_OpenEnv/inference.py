#!/usr/bin/env python3
"""
OpenEnv Reproducible Inference Script - Stochastic Grid World Navigation

Official OpenEnv specification compliance:
✓ Standard API: reset() -> Observation, step(action) -> (Observation, reward, done)
✓ Reproducible with configurable seeds for consistent evaluation
✓ LLM-based baseline agent using OpenAI client
✓ Logging: [START], [STEP], [END] format (strict spec compliance)
✓ Normalized scores in [0, 1] range
✓ Visualization with matplotlib & Gradio web UI (optional)

ENVIRONMENT VARIABLES:
  HF_TOKEN            - API key (REQUIRED)
  TASK_NAME           - easy|medium|hard_navigation (default: medium_navigation)
  RANDOM_SEED         - Fixed seed for reproducibility (optional)
  MAX_STEPS           - Max steps per episode (default: 100)
  MODEL_NAME          - LLM model (default: gpt-3.5-turbo)
  API_BASE_URL        - LLM endpoint (default: https://api.openai.com/v1)
  TEMPERATURE         - LLM temperature (default: 0.7)
  MAX_TOKENS          - LLM max tokens (default: 150)
  GRADIO_UI           - Enable web UI (0 or 1, default: 0)

STDOUT FORMAT (OpenEnv Spec):
  [START] task=<name> env=stochastic_grid_world model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

Usage:
  python inference.py
  TASK_NAME=hard_navigation RANDOM_SEED=42 python inference.py
  GRADIO_UI=1 python inference.py
"""

import os
import sys
import json
import textwrap
import io
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from openai import OpenAI, APIError
except ImportError:
    print("[ERROR] OpenAI not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

try:
    import gradio as gr 
except ImportError:
    HAS_GRADIO = False
    gr = None

from env.environment import StochasticGridEnvironment


# ==================== CONFIGURATION WITH VALIDATION ====================

def validate_env_vars() -> Dict[str, Any]:
    """Validate and load environment variables with defaults"""
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")
    
    task_name = os.getenv("TASK_NAME", "medium_navigation").lower()
    valid_tasks = ["easy_navigation", "medium_navigation", "hard_navigation"]
    if task_name not in valid_tasks:
        raise ValueError(f"TASK_NAME must be one of {valid_tasks}, got {task_name}")
    
    random_seed = os.getenv("RANDOM_SEED")
    if random_seed is not None:
        try:
            random_seed = int(random_seed)
        except ValueError:
            raise ValueError(f"RANDOM_SEED must be integer, got {random_seed}")
    
    max_steps = int(os.getenv("MAX_STEPS", "100"))
    if max_steps < 1:
        raise ValueError(f"MAX_STEPS must be >= 1, got {max_steps}")
    
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("MAX_TOKENS", "150"))
    enable_gradio = os.getenv("GRADIO_UI", "0").lower() in ["1", "true", "yes"]
    
    if enable_gradio and not HAS_GRADIO:
        raise ImportError("Gradio requested but not installed. Run: pip install gradio")
    
    return {
        "hf_token": hf_token,
        "task_name": task_name,
        "random_seed": random_seed,
        "max_steps": max_steps,
        "model_name": model_name,
        "api_base_url": api_base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_gradio": enable_gradio,
    }


# System prompt for LLM agent
SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent navigation agent in a 2D grid world.
    
    OBJECTIVE: Reach target while avoiding obstacles.
    
    OBSERVATIONS:
    - agent_position: Current [x, y]
    - target_position: Goal [x, y]
    - distance_to_target: Steps to goal
    - obstacles: List of blocked cells
    - grid_size: Arena size
    - episode_reward: Total reward accumulated
    
    ACTIONS: direction magnitude
    - direction: up | down | left | right
    - magnitude: 0.0 to 1.0 (step size)
    
    STRATEGY:
    1. Calculate path to target (Manhattan distance)
    2. Identify obstacles in direct path
    3. Plan efficient route around obstacles
    4. Move toward target with magnitude based on confidence
    
    RESPONSE FORMAT: ONLY "direction magnitude"
    Examples: "right 0.8" or "up 0.5"
    Do not include explanations - just direction and magnitude.
""").strip()



# ==================== LOGGING FUNCTIONS ====================

def log_start(task: str, model: str, benchmark: str = "stochastic_grid_world") -> None:
    """Log episode start in OpenEnv format"""
    print(f"[START] task={task} env={benchmark} model={model}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, 
             error: Optional[str] = None) -> None:
    """Log step in OpenEnv format"""
    error_val = f'"{error}"' if error else "null"
    done_val = "true" if done else "false"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", 
          flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end in OpenEnv format"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)



# ==================== VISUALIZATION ====================

def visualize_grid_state(agent_pos: List[int], target_pos: List[int], 
                        obstacles: List[List[int]], grid_size: int = 20,
                        title: str = "Grid State") -> Optional[np.ndarray]:
    """Create matplotlib visualization - returns numpy RGB array or None on error"""
    try:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        grid = np.zeros((grid_size, grid_size))
        
        # Mark obstacles
        for obs in obstacles:
            if 0 <= obs[0] < grid_size and 0 <= obs[1] < grid_size:
                grid[obs[1], obs[0]] = 0.3
        
        # Mark agent
        if 0 <= agent_pos[0] < grid_size and 0 <= agent_pos[1] < grid_size:
            grid[agent_pos[1], agent_pos[0]] = 0.7
        
        # Mark target
        if 0 <= target_pos[0] < grid_size and 0 <= target_pos[1] < grid_size:
            grid[target_pos[1], target_pos[0]] = 1.0
        
        ax.imshow(grid, cmap='RdYlGn', origin='upper', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        
        legend_elements = [
            Patch(facecolor='red', label='Obstacles'),
            Patch(facecolor='yellow', label='Agent'),
            Patch(facecolor='green', label='Target')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Convert to numpy array
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig)
        
        return (img * 255).astype(np.uint8)
        
    except Exception as e:
        print(f"[WARNING] Visualization failed: {e}", file=sys.stderr)
        plt.close('all')
        return None


# ==================== LLM AGENT ====================

def parse_llm_response(response: str, obs: Dict[str, Any]) -> Tuple[str, str]:
    """Parse LLM response and extract action - returns (action_string, parsed_action)"""
    if not response:
        return "right 0.5", "right 0.5"
    
    parts = response.strip().split()
    if len(parts) >= 2:
        direction = parts[0].lower()
        try:
            magnitude = float(parts[1])
            magnitude = max(0.0, min(1.0, magnitude))
        except (ValueError, IndexError):
            magnitude = 0.5
    else:
        # Fallback: use simple heuristic
        agent_x, agent_y = obs["agent_position"]
        target_x, target_y = obs["target_position"]
        dx = target_x - agent_x
        dy = target_y - agent_y
        
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        magnitude = 0.8
    
    return f"{direction} {magnitude:.1f}", f"{direction} {magnitude:.1f}"


def get_llm_action(client: OpenAI, config: Dict[str, Any], step: int, 
                  obs: Dict[str, Any], history: List[str]) -> Tuple[str, Optional[str]]:
    """
    Get action from LLM with error handling
    Returns: (action_string, error_message_or_None)
    """
    agent_pos = obs.get("agent_position", [0, 0])
    target_pos = obs.get("target_position", [10, 10])
    distance = obs.get("distance_to_target", 0)
    obstacles = obs.get("obstacles", [])
    episode_reward = obs.get("episode_reward", 0.0)
    
    # Build compact prompt
    history_str = "\n".join(history[-2:]) if history else "None"
    prompt = f"""Step {step}: Navigate to target
Pos: {agent_pos} → Target: {target_pos}, Distance: {distance}
Obstacles: {len(obstacles)}, Reward: {episode_reward:.2f}
Recent: {history_str}
Action (direction magnitude):"""
    
    try:
        response = client.chat.completions.create(
            model=config["model_name"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout=10.0,
        )
        
        action_text = (response.choices[0].message.content or "").strip()
        action_str, _ = parse_llm_response(action_text, obs)
        return action_str, None
        
    except (APIError, Exception) as e:
        error_msg = str(e)
        _, fallback_action = parse_llm_response("", obs)
        return fallback_action, error_msg


# ==================== EPISODE RUNNER ====================

def run_episode(client: OpenAI, env: StochasticGridEnvironment, config: Dict[str, Any],
                episode_seed: Optional[int] = None) -> Tuple[bool, int, float, List[float]]:
    """
    Run single episode
    Returns: (success, steps, score, rewards_list)
    """
    rewards_list = []
    history = []
    benchmark = "stochastic_grid_world"
    success = False
    score = 0.0
    
    # Log start
    task = config["task_name"]
    model = config["model_name"]
    log_start(task=task, model=model, benchmark=benchmark)
    
    try:
        # Reset environment
        obs = env.reset(seed=episode_seed)
        if obs is None:
            raise RuntimeError("Environment reset returned None")
        
        # Run steps
        for step in range(1, config["max_steps"] + 1):
            if obs.get("done", False):
                break
            
            # Get action from LLM
            action_str, error = get_llm_action(client, config, step, obs, history)
            
            # Parse action
            parts = action_str.split()
            direction = parts[0].lower() if parts else "right"
            try:
                magnitude = float(parts[1]) if len(parts) > 1 else 0.5
            except ValueError:
                magnitude = 0.5
            
            action = {"direction": direction, "magnitude": magnitude}
            
            # Step environment
            obs, reward, done = env.step(action)
            rewards_list.append(float(reward))
            history.append(f"Step {step}: {action_str}")
            
            # Log step
            log_step(step=step, action_str=action_str, reward=reward, 
                    done=done, error=error)
            
            if done:
                break
        
        # Compute score
        max_reward = 10.0 * config["max_steps"]
        total_reward = sum(rewards_list)
        score = total_reward / max_reward if max_reward > 0 else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= 0.5
        
    except Exception as e:
        error_msg = f"Episode failed: {e}"
        print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
        score = 0.0
        success = False
    
    # Log end
    log_end(success=success, steps=len(rewards_list), score=score, 
           rewards=rewards_list)
    
    return success, len(rewards_list), score, rewards_list


# ==================== GRADIO UI (OPTIONAL) ====================

def create_gradio_ui(client: OpenAI, config: Dict[str, Any]) -> Optional[Any]:
    """Create Gradio web interface"""
    if not HAS_GRADIO:
        print("[ERROR] Gradio not installed", file=sys.stderr)
        return None
    
    def run_visualization(task: str, seed: Optional[int] = None, 
                         max_steps_ui: int = 100) -> Tuple[List, str, str]:
        """Run episode and display"""
        try:
            env = StochasticGridEnvironment(
                grid_size=20,
                difficulty=task.split("_")[0],
                seed=int(seed) if seed else None
            )
            
            local_config = dict(config)
            local_config["task_name"] = task
            local_config["max_steps"] = max_steps_ui
            
            success, steps, score, rewards = run_episode(
                client, env, local_config, 
                episode_seed=int(seed) if seed else None
            )
            
            summary = f"""
            ### Episode Results
            - **Success**: {success}
            - **Steps**: {steps}
            - **Score**: {score:.2f}
            - **Total Reward**: {sum(rewards):.2f}
            """
            
            metrics = json.dumps({
                "success": success,
                "steps": steps,
                "score": float(score),
                "total_reward": float(sum(rewards)),
                "rewards": [float(r) for r in rewards]
            }, indent=2)
            
            return [], summary, metrics
            
        except Exception as e:
            return [], f"Error: {e}", "{}"
    
    with gr.Blocks(title="OpenEnv Grid World") as interface:
        gr.Markdown("# 🤖 OpenEnv Stochastic Grid World")
        gr.Markdown("LLM-based navigation agent in dynamic environments")
        
        with gr.Row():
            with gr.Column(scale=1):
                task = gr.Dropdown(
                    choices=["easy_navigation", "medium_navigation", "hard_navigation"],
                    value="medium_navigation",
                    label="Task"
                )
                seed = gr.Number(label="Seed", value=42, precision=0)
                max_steps = gr.Number(label="Max Steps", value=100, precision=0)
                run_btn = gr.Button("Run Episode", variant="primary")
            
            with gr.Column(scale=2):
                summary = gr.Markdown("Configure task and click Run")
                metrics = gr.JSON(label="Metrics")
        
        gallery = gr.Gallery(label="Trajectory", columns=3)
        
        run_btn.click(
            run_visualization,
            inputs=[task, seed, max_steps],
            outputs=[gallery, summary, metrics]
        )
    
    return interface


# ==================== MAIN ====================

def main():
    """Main entry point with configuration validation"""
    
    try:
        # Validate configuration
        config = validate_env_vars()
        
    except (ValueError, ImportError) as e:
        print(f"[ERROR] Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Initialize OpenAI client
        client = OpenAI(base_url=config["api_base_url"], api_key=config["hf_token"])
        
    except Exception as e:
        print(f"[ERROR] Client initialization failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Create environment
        difficulty = config["task_name"].split("_")[0]
        env = StochasticGridEnvironment(
            grid_size=20,
            difficulty=difficulty,
            seed=config["random_seed"]
        )
        
    except Exception as e:
        print(f"[ERROR] Environment creation failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run UI or episode
    if config["enable_gradio"]:
        try:
            print("[INFO] Starting Gradio UI at http://localhost:7860", file=sys.stderr)
            interface = create_gradio_ui(client, config)
            if interface:
                interface.launch(share=False)
            else:
                print("[ERROR] Failed to create UI", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] UI error: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        # Run single episode
        print(f"[INFO] Configuration:", file=sys.stderr)
        print(f"  Task: {config['task_name']}", file=sys.stderr)
        print(f"  Model: {config['model_name']}", file=sys.stderr)
        print(f"  Seed: {config['random_seed']}", file=sys.stderr)
        print(f"  Max Steps: {config['max_steps']}", file=sys.stderr)
        print(f"", file=sys.stderr)
        
        success, steps, score, rewards = run_episode(
            client, env, config, 
            episode_seed=config["random_seed"]
        )
        
        # Print summary
        print(f"", file=sys.stderr)
        print(f"[SUMMARY]", file=sys.stderr)
        print(f"  Success: {success}", file=sys.stderr)
        print(f"  Steps: {steps}", file=sys.stderr)
        print(f"  Score: {score:.2f}", file=sys.stderr)
        print(f"  Total Reward: {sum(rewards):.2f}", file=sys.stderr)
        if rewards:
            print(f"  Avg Reward/Step: {np.mean(rewards):.2f}", file=sys.stderr)
        print(f"  Status: {'PASS' if success else 'FAIL'}", file=sys.stderr)


if __name__ == "__main__":
    main()

