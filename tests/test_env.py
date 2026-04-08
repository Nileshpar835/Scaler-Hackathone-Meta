"""Unit tests for DataCleanerEnv environment."""

import pytest
import asyncio
import pandas as pd
import numpy as np
from environment import DataCleanerEnv, Action, OperationType


@pytest.fixture
def env():
    """Create a fresh environment for each test."""
    return DataCleanerEnv(dataset_dir="datasets", max_steps=30)


@pytest.mark.asyncio
async def test_reset(env):
    """Test environment reset."""
    obs = await env.reset(task_id="task_easy")
    
    assert obs is not None
    assert obs.task_id == "task_easy"
    assert obs.dataset_stats.total_rows > 0
    assert obs.dataset_stats.total_columns > 0
    assert obs.steps_remaining == 30


@pytest.mark.asyncio
async def test_step_remove_duplicates(env):
    """Test removing duplicates."""
    obs = await env.reset(task_id="task_easy")
    
    initial_rows = obs.dataset_stats.total_rows
    
    action = Action(operation=OperationType.REMOVE_DUPLICATES)
    obs, reward, done, info = await env.step(action)
    
    assert obs.dataset_stats.total_rows <= initial_rows
    assert isinstance(reward, float)
    assert -1.0 <= reward <= 1.0
    assert isinstance(done, bool)


@pytest.mark.asyncio
async def test_step_fill_missing(env):
    """Test filling missing values."""
    obs = await env.reset(task_id="task_medium")
    
    # Find a column with missing values
    col_with_missing = None
    for col_info in obs.column_info:
        if col_info.missing_count > 0:
            col_with_missing = col_info.name
            break
    
    if col_with_missing:
        action = Action(
            operation=OperationType.FILL_MISSING,
            column=col_with_missing,
            strategy="mean"
        )
        obs, reward, done, info = await env.step(action)
        
        assert isinstance(reward, float)
        assert -1.0 <= reward <= 1.0


@pytest.mark.asyncio
async def test_step_declare_clean(env):
    """Test declaring dataset clean."""
    obs = await env.reset(task_id="task_easy")
    
    action = Action(operation=OperationType.DECLARE_CLEAN)
    obs, reward, done, info = await env.step(action)
    
    assert done is True
    assert isinstance(reward, float)
    assert -1.0 <= reward <= 1.0
    assert info.get("episode_complete") is True


@pytest.mark.asyncio
async def test_episode_flow(env):
    """Test a complete episode flow."""
    obs = await env.reset(task_id="task_easy")
    
    steps_taken = 0
    total_reward = 0.0
    
    for _ in range(5):
        action = Action(operation=OperationType.REMOVE_DUPLICATES)
        obs, reward, done, info = await env.step(action)
        
        steps_taken += 1
        total_reward += reward
        
        if done:
            break
    
    assert steps_taken > 0
    assert isinstance(total_reward, float)


@pytest.mark.asyncio
async def test_reward_bounds(env):
    """Test that all rewards are within [-1, 1]."""
    obs = await env.reset(task_id="task_medium")
    
    # Try various actions
    actions = [
        Action(operation=OperationType.REMOVE_DUPLICATES),
        Action(operation=OperationType.REMOVE_DUPLICATES),  # Should penalize (no duplicates left)
    ]
    
    for action in actions:
        obs, reward, done, info = await env.step(action)
        assert -1.0 <= reward <= 1.0, f"Reward out of bounds: {reward}"


@pytest.mark.asyncio
async def test_state_method(env):
    """Test the state() method."""
    obs = await env.reset(task_id="task_easy")
    
    state = env.state()
    
    assert state is not None
    assert state["current_task_id"] == "task_easy"
    assert state["steps_taken"] == 0
    assert isinstance(state["current_dataset"], dict)


@pytest.mark.asyncio
async def test_invalid_action(env):
    """Test handling of invalid actions."""
    obs = await env.reset(task_id="task_easy")
    
    # Invalid column
    action = Action(
        operation=OperationType.FILL_MISSING,
        column="nonexistent_column",
        strategy="mean"
    )
    obs, reward, done, info = await env.step(action)
    
    assert reward < 0  # Should be penalized


@pytest.mark.asyncio
async def test_all_tasks(env):
    """Test that all tasks can be loaded and run."""
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        obs = await env.reset(task_id=task_id)
        
        assert obs.task_id == task_id
        assert obs.dataset_stats.total_rows > 0
        
        # Run one step
        action = Action(operation=OperationType.DECLARE_CLEAN)
        obs, reward, done, info = await env.step(action)
        
        assert done is True


def test_observation_schema():
    """Test that observation follows the correct schema."""
    from environment import Observation
    
    # Try to create an observation (schema validation)
    obs = Observation(
        current_dataset=[],
        dataset_stats__total_rows=10,
        dataset_stats__total_columns=5,
        dataset_stats__pct_missing=0.1,
        dataset_stats__pct_duplicates=0.0,
        dataset_stats__detected_issues=[],
        dataset_stats__quality_score=0.9,
        column_info=[],
        steps_remaining=30,
        previous_action_result="test",
        task_id="test",
        task_description="test task"
    )


def test_action_schema():
    """Test that action follows the correct schema."""
    from environment import Action
    
    action = Action(operation=OperationType.FILL_MISSING, column="age", strategy="mean")
    assert action.operation == OperationType.FILL_MISSING
    assert action.column == "age"
    assert action.is_terminal_action is False
    
    action2 = Action(operation=OperationType.DECLARE_CLEAN)
    assert action2.is_terminal_action is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
