"""
Baseline inference script for DataCleanerEnv using OpenAI API.

Demonstrates how an LLM agent can interact with the DataCleanerEnv environment.
Logs output in OpenEnv-standard format: [START], [STEP], [END].

Required environment variables:
- OPENAI_API_KEY: OpenAI API key
- API_BASE_URL: (optional) base URL for OpenAI API
- MODEL_NAME: (optional) model name, defaults to "gpt-4-turbo"
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Optional
from datetime import datetime

from openai import AsyncOpenAI
from environment import DataCleanerEnv, Action, OperationType

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
MAX_STEPS_PER_TASK = 30
MAX_TOTAL_REWARD = 3.0
SUCCESS_SCORE_THRESHOLD = 0.7

# Tasks to run
TASKS = ["task_easy", "task_medium", "task_hard"]


def log_start(task: str, env: str = "DataCleanerEnv", model: str = MODEL_NAME):
    """Log episode start in OpenEnv format."""
    timestamp = datetime.utcnow().isoformat()
    print(f"[START] task={task!r} env={env!r} model={model!r} timestamp={timestamp!r}")
    sys.stdout.flush()


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    """Log step in OpenEnv format."""
    if error:
        print(f"[STEP] step={step} action={action!r} reward={reward:+.2f} done={done} error={error!r}")
    else:
        print(f"[STEP] step={step} action={action!r} reward={reward:+.2f} done={done} error=null")
    sys.stdout.flush()


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    """Log episode end in OpenEnv format."""
    timestamp = datetime.utcnow().isoformat()
    rewards_str = "[" + ", ".join(f"{r:+.2f}" for r in rewards) + "]"
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards_str} timestamp={timestamp!r}")
    sys.stdout.flush()


def extract_action_from_response(response_text: str) -> Optional[Action]:
    """
    Parse LLM response and extract Action object.
    
    Expected format:
    {
      "operation": "OPERATION_TYPE",
      "column": "column_name",
      "strategy": "strategy_name",
      "params": {...}
    }
    """
    try:
        # Try to find JSON block in response
        if "{" in response_text and "}" in response_text:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            json_str = response_text[start_idx:end_idx]
            action_dict = json.loads(json_str)
            
            # Validate and construct Action
            operation = action_dict.get("operation", "").upper()
            column = action_dict.get("column")
            strategy = action_dict.get("strategy")
            params = action_dict.get("params")
            
            # Map string to enum
            if operation == "FILL_MISSING":
                op = OperationType.FILL_MISSING
            elif operation == "REMOVE_DUPLICATES":
                op = OperationType.REMOVE_DUPLICATES
            elif operation == "FIX_TYPES":
                op = OperationType.FIX_TYPES
            elif operation == "REMOVE_OUTLIERS":
                op = OperationType.REMOVE_OUTLIERS
            elif operation == "DECLARE_CLEAN":
                op = OperationType.DECLARE_CLEAN
            else:
                return None
            
            return Action(
                operation=op,
                column=column,
                strategy=strategy,
                params=params
            )
    except Exception as e:
        print(f"[DEBUG] Failed to parse action: {e}", flush=True)
    
    return None


async def get_model_message(
    client: AsyncOpenAI,
    step: int,
    last_observation: dict,
    last_reward: float,
    history: List[str]
) -> str:
    """
    Get next action from LLM based on current state.
    
    Args:
        client: AsyncOpenAI client
        step: Current step number
        last_observation: Last observation dict
        last_reward: Reward from last action
        history: History of actions taken
    
    Returns:
        JSON string representing next action
    """
    
    # Build context for LLM
    instruction = f"""You are a data cleaning agent. Your goal is to clean messy datasets.

Current Dataset State:
- Total rows: {last_observation['dataset_stats']['total_rows']}
- Total columns: {last_observation['dataset_stats']['total_columns']}
- Quality score: {last_observation['dataset_stats']['quality_score']:.2f}/1.0
- Detected issues: {', '.join(last_observation['dataset_stats']['detected_issues']) or 'None'}
- Steps remaining: {last_observation['steps_remaining']}

Available columns: {', '.join(col['name'] for col in last_observation['column_info'])}

Last reward: {last_reward:+.2f}

Available operations:
1. FILL_MISSING - Fill missing values in a column (strategies: mean, median, mode, forward_fill, drop)
2. REMOVE_DUPLICATES - Remove duplicate rows
3. FIX_TYPES - Fix type errors in a column (strategy: infer)
4. REMOVE_OUTLIERS - Remove outlier rows (strategies: iqr, zscore)
5. DECLARE_CLEAN - Declare the dataset clean and end the episode

Recent actions taken:
{chr(10).join(history[-5:]) if history else 'None'}

Based on the detected issues and current state, choose the NEXT action that would best clean the dataset.

Respond with a JSON object:
{{
  "operation": "OPERATION_NAME",
  "column": "column_name_or_null",
  "strategy": "strategy_name_or_null",
  "params": {{}} 
}}

Think about what issues need fixing and take the most impactful action."""
    
    try:
        response = await client.messages.create(
            model=MODEL_NAME,
            max_tokens=500,
            messages=[
                {"role": "user", "content": instruction}
            ]
        )
        
        return response.content[0].text
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return '{"operation": "DECLARE_CLEAN"}'


async def run_episode(task_id: str) -> tuple[float, bool, List[float]]:
    """
    Run a single episode of the environment.
    
    Args:
        task_id: Task identifier (e.g., "task_easy")
    
    Returns:
        Tuple of (final_score, success, rewards)
    """
    
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = DataCleanerEnv(dataset_dir="datasets", max_steps=MAX_STEPS_PER_TASK)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=task_id, env="DataCleanerEnv", model=MODEL_NAME)
    
    try:
        # Reset environment
        result = await env.reset(task_id=task_id, dataset_idx=0)
        last_obs = result.dict()
        last_reward = 0.0
        
        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if step == MAX_STEPS_PER_TASK + 1:
                break
            
            # Get next action from LLM
            response_text = await get_model_message(client, step, last_obs, last_reward, history)
            action_obj = extract_action_from_response(response_text)
            
            if action_obj is None:
                # Fallback: declare clean if we can't parse
                action_obj = Action(operation=OperationType.DECLARE_CLEAN)
            
            action_str = f"{action_obj.operation.value}"
            if action_obj.column:
                action_str += f":{action_obj.column}"
            if action_obj.strategy:
                action_str += f":{action_obj.strategy}"
            
            # Execute action
            result = await env.step(action_obj)
            obs, reward, done, info = result
            
            rewards.append(reward or 0.0)
            steps_taken = step
            last_obs = obs.dict()
            last_reward = reward or 0.0
            
            # Log step
            log_step(step=step, action=action_str, reward=last_reward, done=done, error=None)
            
            history.append(f"Step {step}: {action_str} -> reward {last_reward:+.2f}")
            
            if done:
                break
        
        # Calculate final score
        score = sum(max(r, 0.0) for r in rewards) / MAX_TOTAL_REWARD
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    
    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        score = 0.0
        success = False
    
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return score, success, rewards


async def main():
    """Run baseline inference on all tasks."""
    
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Starting DataCleanerEnv baseline inference", flush=True)
    print(f"[INFO] Model: {MODEL_NAME}", flush=True)
    print(f"[INFO] API Base URL: {API_BASE_URL}", flush=True)
    print(f"[INFO] Tasks: {TASKS}", flush=True)
    print()
    
    all_scores = {}
    all_successes = {}
    
    for task_id in TASKS:
        print(f"\n{'='*60}")
        print(f"Running task: {task_id}")
        print(f"{'='*60}\n")
        
        task_score, task_success, task_rewards = await run_episode(task_id)
        all_scores[task_id] = task_score
        all_successes[task_id] = task_success
        
        print(f"\n[RESULT] {task_id}: score={task_score:.4f}, success={task_success}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for task_id in TASKS:
        print(f"{task_id}: {all_scores[task_id]:.4f}")
    
    overall_score = sum(all_scores.values()) / len(TASKS)
    print(f"\nOverall Score: {overall_score:.4f}")
    print(f"Tasks Succeeded: {sum(all_successes.values())}/{len(TASKS)}")


if __name__ == "__main__":
    asyncio.run(main())
