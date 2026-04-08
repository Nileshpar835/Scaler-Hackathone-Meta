"""Reward model and calculation logic for the DataCleanerEnv environment."""

from pydantic import BaseModel, Field
from typing import Dict, Any, List
from dataclasses import dataclass


class Reward(BaseModel):
    """Reward returned by the DataCleanerEnv environment."""
    
    value: float = Field(
        ge=-1.0, le=1.0,
        description="Scalar reward value, normalized to [-1.0, 1.0]"
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of reward components (for debugging)"
    )


@dataclass
class RewardConfig:
    """Configuration for reward function."""
    # Per-step rewards
    correct_operation_reward: float = 0.15
    issue_reduction_bonus: float = 0.10
    no_effect_penalty: float = -0.02
    data_corruption_penalty: float = -0.30
    invalid_operation_penalty: float = -0.05
    
    # Episode completion rewards
    clean_declaration_bonus: float = 0.50
    efficiency_multiplier: float = 0.10
    false_declaration_penalty: float = -0.50
    
    # Episode parameters
    max_steps: int = 30
    max_total_reward: float = 3.0


def calculate_step_reward(
    operation_valid: bool,
    operation_was_effective: bool,
    data_was_corrupted: bool,
    issue_was_real: bool,
    config: RewardConfig = None
) -> Reward:
    """
    Calculate reward for a single step.
    
    Args:
        operation_valid: Was the operation executable (correct column/type)?
        operation_was_effective: Did the operation reduce an issue?
        data_was_corrupted: Did the operation corrupt valid data?
        issue_was_real: Did the operation target an actual issue?
        config: RewardConfig instance
    
    Returns:
        Reward object with value and breakdown
    """
    if config is None:
        config = RewardConfig()
    
    breakdown: Dict[str, float] = {}
    value = 0.0
    
    if not operation_valid:
        value += config.invalid_operation_penalty
        breakdown["invalid_operation"] = config.invalid_operation_penalty
        return Reward(value=value, breakdown=breakdown)
    
    if data_was_corrupted:
        value += config.data_corruption_penalty
        breakdown["data_corruption"] = config.data_corruption_penalty
        return Reward(value=value, breakdown=breakdown)
    
    if operation_was_effective and issue_was_real:
        value += config.correct_operation_reward
        breakdown["correct_operation"] = config.correct_operation_reward
        
        value += config.issue_reduction_bonus
        breakdown["issue_reduction"] = config.issue_reduction_bonus
    elif not operation_was_effective and issue_was_real:
        value += config.no_effect_penalty
        breakdown["no_effect"] = config.no_effect_penalty
    elif not issue_was_real:
        value += config.invalid_operation_penalty
        breakdown["invalid_issue"] = config.invalid_operation_penalty
    
    return Reward(value=value, breakdown=breakdown)


def calculate_episode_completion_reward(
    steps_used: int,
    had_remaining_issues: bool,
    config: RewardConfig = None
) -> float:
    """
    Calculate reward for episode completion.
    
    Args:
        steps_used: Number of steps taken to declare clean
        had_remaining_issues: Were there unresolved issues when declared clean?
        config: RewardConfig instance
    
    Returns:
        Float reward value
    """
    if config is None:
        config = RewardConfig()
    
    if had_remaining_issues:
        return config.false_declaration_penalty
    
    # Successfully cleaned: base bonus + efficiency bonus
    efficiency_bonus = (config.max_steps - steps_used) / config.max_steps * config.efficiency_multiplier
    return config.clean_declaration_bonus + efficiency_bonus


def normalize_rewards(rewards: List[float], total_max_reward: float) -> float:
    """
    Normalize episode rewards to [0.0, 1.0].
    
    Args:
        rewards: List of step rewards
        total_max_reward: Maximum possible total reward
    
    Returns:
        Normalized score in [0.0, 1.0]
    """
    if total_max_reward <= 0:
        return 0.0
    
    total = sum(max(r, 0.0) for r in rewards)  # Only count positive rewards
    score = total / total_max_reward
    return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
