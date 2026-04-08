"""Main DataCleanerEnv environment implementation."""

import json
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

from .observation import Observation, DatasetStats, ColumnInfo
from .action import Action, OperationType
from .reward import Reward, RewardConfig, calculate_step_reward, normalize_rewards
from .data_loader import load_task_from_json
from .grader import grade_episode


class DataCleanerEnv:
    """
    Real-world data cleaning environment for OpenEnv.
    
    Tasks:
    - task_easy: Remove duplicates from email list
    - task_medium: Clean multi-issue customer database
    - task_hard: Clean financial data with outlier detection challenge
    """
    
    def __init__(self, dataset_dir: str = "datasets", max_steps: int = 30):
        """
        Initialize environment.
        
        Args:
            dataset_dir: Directory containing task JSON files
            max_steps: Maximum steps per episode
        """
        self.dataset_dir = dataset_dir
        self.max_steps = max_steps
        self.reward_config = RewardConfig(max_steps=max_steps)
        
        # Episode state
        self.current_task_id: Optional[str] = None
        self.current_dataset: Optional[pd.DataFrame] = None
        self.original_dataset: Optional[pd.DataFrame] = None
        self.task_description: str = ""
        self.steps_taken: int = 0
        self.episode_rewards: List[float] = []
        self.action_history: List[Dict[str, Any]] = []
        
        # Track quality issues by type
        self.known_issues: Dict[str, List[int]] = {}
        self.last_action_result: str = "Environment initialized"
        
        # Task metadata
        self.all_tasks = self._load_all_tasks()
    
    def _load_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Load all available tasks from JSON files."""
        tasks = {}
        dataset_path = Path(self.dataset_dir)
        
        for task_file in dataset_path.glob("task_*.json"):
            try:
                with open(task_file, "r") as f:
                    task_data = json.load(f)
                    task_id = task_data["task_id"]
                    tasks[task_id] = task_data
            except Exception as e:
                print(f"Warning: Failed to load {task_file}: {e}")
        
        if not tasks:
            print(f"Warning: No tasks found in {self.dataset_dir}")
        
        return tasks
    
    async def reset(self, task_id: Optional[str] = None, dataset_idx: int = 0) -> Observation:
        """
        Reset environment to initial state.
        
        Args:
            task_id: Task to use (random if None). Options: "task_easy", "task_medium", "task_hard"
            dataset_idx: Which dataset within the task to use (0-2)
        
        Returns:
            Initial observation
        """
        # Select task
        if task_id is None:
            task_id = np.random.choice(list(self.all_tasks.keys()))
        
        if task_id not in self.all_tasks:
            raise ValueError(f"Unknown task_id: {task_id}. Available: {list(self.all_tasks.keys())}")
        
        task_data = self.all_tasks[task_id]
        self.current_task_id = task_id
        self.task_description = task_data["description"]
        
        # Load dataset
        dataset_idx = min(dataset_idx, len(task_data["datasets"]) - 1)
        dataset_spec = task_data["datasets"][dataset_idx]
        
        self.current_dataset = pd.DataFrame(dataset_spec["dirty_data"])
        self.original_dataset = pd.DataFrame(dataset_spec["clean_data"])
        
        # Reset episode state
        self.steps_taken = 0
        self.episode_rewards = []
        self.action_history = []
        self.last_action_result = "Environment reset. Ready for cleaning operations."
        
        # Detect and track known issues
        self._detect_issues()
        
        return self._get_observation()
    
    def _detect_issues(self):
        """Detect and track data quality issues in current dataset."""
        self.known_issues = {
            "duplicates": [],
            "missing_values": [],
            "type_errors": [],
            "outliers": []
        }
        
        if self.current_dataset is None:
            return
        
        # Detect duplicates
        dup_mask = self.current_dataset.duplicated(keep=False)
        self.known_issues["duplicates"] = dup_mask[dup_mask].index.tolist()
        
        # Detect missing values
        missing_mask = self.current_dataset.isnull().any(axis=1)
        self.known_issues["missing_values"] = missing_mask[missing_mask].index.tolist()
        
        # Detect type errors: expect matching with original dataset types
        self.known_issues["type_errors"] = self._find_type_errors()
        
        # Detect outliers (simple: values > 3 std from mean)
        self.known_issues["outliers"] = self._find_outliers()
    
    def _find_type_errors(self) -> List[int]:
        """Find rows with type mismatches compared to original."""
        type_errors = []
        
        if self.original_dataset is None:
            return type_errors
        
        for col in self.current_dataset.columns:
            if col not in self.original_dataset.columns:
                continue
            
            orig_dtype = self.original_dataset[col].dtype
            curr_dtype = self.current_dataset[col].dtype
            
            # Check for type mismatches in non-nullable columns
            if orig_dtype in [np.int64, np.float64] and curr_dtype == object:
                # Try to convert; if it fails, it's a type error
                for idx, val in self.current_dataset[col].items():
                    if pd.isna(val):
                        continue
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        if idx not in type_errors:
                            type_errors.append(idx)
        
        return type_errors
    
    def _find_outliers(self) -> List[int]:
        """Find potential outliers using zscore method."""
        outliers = []
        
        numeric_cols = self.current_dataset.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                col_data = pd.to_numeric(self.current_dataset[col], errors="coerce")
                if col_data.notna().sum() < 2:
                    continue
                
                mean = col_data.mean()
                std = col_data.std()
                
                if std > 0:
                    z_scores = np.abs((col_data - mean) / std)
                    outlier_mask = z_scores > 3.0
                    outliers.extend(outlier_mask[outlier_mask].index.tolist())
            except Exception:
                pass
        
        return list(set(outliers))  # Remove duplicates
    
    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        Args:
            action: Action object specifying operation to perform
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.steps_taken += 1
        
        # Check if action is terminal (declare clean)
        if action.is_terminal_action:
            return await self._handle_declare_clean()
        
        # Validate and execute action
        reward_obj, info = self._execute_action(action)
        reward = reward_obj.value
        
        self.episode_rewards.append(reward)
        
        # Check episode termination
        done = self.steps_taken >= self.max_steps
        
        # Get updated observation
        obs = self._get_observation()
        
        return obs, reward, done, info
    
    def _execute_action(self, action: Action) -> Tuple[Reward, Dict[str, Any]]:
        """
        Execute a cleaning action and compute reward.
        
        Returns:
            Tuple of (reward_obj, info_dict)
        """
        info = {
            "action": action.dict(),
            "step": self.steps_taken,
            "dataset_shape_before": tuple(self.current_dataset.shape) if self.current_dataset is not None else None,
        }
        
        try:
            operation = action.operation
            
            if operation == OperationType.FILL_MISSING:
                reward_obj = self._fill_missing(action)
            elif operation == OperationType.REMOVE_DUPLICATES:
                reward_obj = self._remove_duplicates(action)
            elif operation == OperationType.FIX_TYPES:
                reward_obj = self._fix_types(action)
            elif operation == OperationType.REMOVE_OUTLIERS:
                reward_obj = self._remove_outliers(action)
            else:
                reward_obj = Reward(value=-0.05, breakdown={"invalid_operation": -0.05})
                self.last_action_result = f"Unknown operation: {operation}"
        except Exception as e:
            reward_obj = Reward(value=-0.05, breakdown={"error": -0.05})
            self.last_action_result = f"Operation failed: {str(e)}"
            info["error"] = str(e)
        
        info["dataset_shape_after"] = tuple(self.current_dataset.shape) if self.current_dataset is not None else None
        info["reward"] = reward_obj.value
        info["action_result"] = self.last_action_result
        
        self.action_history.append(info)
        
        return reward_obj, info
    
    def _fill_missing(self, action: Action) -> Reward:
        """Fill missing values in specified column."""
        if action.column is None:
            self.last_action_result = "FILL_MISSING requires a column name"
            return Reward(value=-0.05, breakdown={"missing_column": -0.05})
        
        if action.column not in self.current_dataset.columns:
            self.last_action_result = f"Column '{action.column}' not found"
            return Reward(value=-0.05, breakdown={"invalid_column": -0.05})
        
        col_data = self.current_dataset[action.column]
        missing_count_before = col_data.isnull().sum()
        
        if missing_count_before == 0:
            self.last_action_result = f"No missing values in '{action.column}' to fill"
            return Reward(value=-0.02, breakdown={"no_effect": -0.02})
        
        strategy = action.strategy or "mean"
        
        try:
            if strategy == "mean":
                fill_val = pd.to_numeric(col_data, errors="coerce").mean()
                self.current_dataset[action.column].fillna(fill_val, inplace=True)
            elif strategy == "median":
                fill_val = pd.to_numeric(col_data, errors="coerce").median()
                self.current_dataset[action.column].fillna(fill_val, inplace=True)
            elif strategy == "mode":
                fill_val = col_data.mode()[0] if len(col_data.mode()) > 0 else col_data.iloc[0]
                self.current_dataset[action.column].fillna(fill_val, inplace=True)
            elif strategy == "forward_fill":
                self.current_dataset[action.column].fillna(method="ffill", inplace=True)
            elif strategy == "drop":
                # Remove rows with missing values in this column
                self.current_dataset = self.current_dataset.dropna(subset=[action.column], inplace=False)
            else:
                self.last_action_result = f"Unknown strategy: {strategy}"
                return Reward(value=-0.05, breakdown={"bad_strategy": -0.05})
            
            missing_count_after = self.current_dataset[action.column].isnull().sum()
            
            if missing_count_after < missing_count_before:
                self.last_action_result = f"Filled {missing_count_before - missing_count_after} missing values in '{action.column}'"
                return Reward(value=0.25, breakdown={"successful_fill": 0.25})
            else:
                self.last_action_result = f"Fill strategy did not reduce missing values"
                return Reward(value=-0.02, breakdown={"no_effect": -0.02})
        
        except Exception as e:
            self.last_action_result = f"Error filling missing values: {str(e)}"
            return Reward(value=-0.05, breakdown={"error": -0.05})
    
    def _remove_duplicates(self, action: Action) -> Reward:
        """Remove duplicate rows."""
        dup_count_before = self.current_dataset.duplicated(keep=False).sum()
        
        if dup_count_before == 0:
            self.last_action_result = "No duplicate rows found"
            return Reward(value=-0.02, breakdown={"no_effect": -0.02})
        
        try:
            self.current_dataset = self.current_dataset.drop_duplicates()
            
            dup_count_after = self.current_dataset.duplicated(keep=False).sum()
            
            if dup_count_after < dup_count_before:
                removed = dup_count_before - dup_count_after
                self.last_action_result = f"Removed {removed} duplicate rows"
                return Reward(value=0.25, breakdown={"duplicates_removed": 0.25})
            else:
                self.last_action_result = "Failed to remove duplicates"
                return Reward(value=-0.05, breakdown={"error": -0.05})
        
        except Exception as e:
            self.last_action_result = f"Error removing duplicates: {str(e)}"
            return Reward(value=-0.05, breakdown={"error": -0.05})
    
    def _fix_types(self, action: Action) -> Reward:
        """Fix type errors in specified column."""
        if action.column is None:
            self.last_action_result = "FIX_TYPES requires a column name"
            return Reward(value=-0.05, breakdown={"missing_column": -0.05})
        
        if action.column not in self.current_dataset.columns:
            self.last_action_result = f"Column '{action.column}' not found"
            return Reward(value=-0.05, breakdown={"invalid_column": -0.05})
        
        strategy = action.strategy or "infer"
        
        try:
            orig_type = self.original_dataset[action.column].dtype if self.original_dataset is not None else None
            
            if strategy == "infer":
                # Try to infer and convert to appropriate type
                if orig_type in [np.int64, int]:
                    self.current_dataset[action.column] = pd.to_numeric(
                        self.current_dataset[action.column], errors="coerce"
                    ).astype("Int64")  # nullable int
                elif orig_type in [np.float64, float]:
                    self.current_dataset[action.column] = pd.to_numeric(
                        self.current_dataset[action.column], errors="coerce"
                    )
            
            self.last_action_result = f"Fixed types in '{action.column}'"
            return Reward(value=0.20, breakdown={"types_fixed": 0.20})
        
        except Exception as e:
            self.last_action_result = f"Error fixing types: {str(e)}"
            return Reward(value=-0.05, breakdown={"error": -0.05})
    
    def _remove_outliers(self, action: Action) -> Reward:
        """Remove outlier rows."""
        if action.column is None:
            self.last_action_result = "REMOVE_OUTLIERS requires a column name"
            return Reward(value=-0.05, breakdown={"missing_column": -0.05})
        
        if action.column not in self.current_dataset.columns:
            self.last_action_result = f"Column '{action.column}' not found"
            return Reward(value=-0.05, breakdown={"invalid_column": -0.05})
        
        strategy = action.strategy or "iqr"
        threshold = 3.0 if strategy == "zscore" else 1.5
        
        if action.params and "threshold" in action.params:
            threshold = action.params["threshold"]
        
        try:
            col_data = pd.to_numeric(self.current_dataset[action.column], errors="coerce")
            
            if col_data.notna().sum() < 2:
                self.last_action_result = f"Not enough numeric values in '{action.column}' for outlier detection"
                return Reward(value=-0.02, breakdown={"no_effect": -0.02})
            
            rows_before = len(self.current_dataset)
            
            if strategy == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (col_data >= lower_bound) & (col_data <= upper_bound)
            elif strategy == "zscore":
                mean = col_data.mean()
                std = col_data.std()
                if std > 0:
                    z_scores = np.abs((col_data - mean) / std)
                    mask = z_scores <= threshold
                else:
                    mask = pd.Series([True] * len(col_data))
            else:
                self.last_action_result = f"Unknown outlier strategy: {strategy}"
                return Reward(value=-0.05, breakdown={"bad_strategy": -0.05})
            
            self.current_dataset = self.current_dataset[mask]
            rows_after = len(self.current_dataset)
            rows_removed = rows_before - rows_after
            
            if rows_removed > 0:
                # Penalty for too aggressive removal (false positives)
                max_safe_removal = int(len(self.original_dataset) * 0.05)
                penalty = 0.0
                if rows_removed > max_safe_removal:
                    penalty = (rows_removed - max_safe_removal) * 0.05
                
                reward_val = 0.20 - penalty
                self.last_action_result = f"Removed {rows_removed} outlier rows from '{action.column}'"
                return Reward(value=max(-0.30, reward_val), breakdown={"outliers_removed": reward_val})
            else:
                self.last_action_result = f"No outliers detected in '{action.column}'"
                return Reward(value=-0.02, breakdown={"no_effect": -0.02})
        
        except Exception as e:
            self.last_action_result = f"Error removing outliers: {str(e)}"
            return Reward(value=-0.05, breakdown={"error": -0.05})
    
    async def _handle_declare_clean(self) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Handle terminal action: agent declares dataset clean."""
        # Grade the final state
        if self.current_dataset is None or self.original_dataset is None:
            final_reward = -0.50
            success = False
        else:
            final_score = grade_episode(
                self.current_dataset,
                self.original_dataset,
                self.current_task_id
            )
            
            # Reward based on how clean the dataset is
            if final_score >= 0.8:
                final_reward = 0.50
                success = True
            elif final_score >= 0.5:
                final_reward = 0.20
                success = False
            else:
                final_reward = -0.50
                success = False
        
        # Add efficiency bonus
        efficiency_bonus = (self.max_steps - self.steps_taken) / self.max_steps * 0.10
        if success and self.steps_taken < self.max_steps:
            final_reward += efficiency_bonus
        
        self.episode_rewards.append(final_reward)
        
        obs = self._get_observation()
        
        info = {
            "action": "DECLARE_CLEAN",
            "step": self.steps_taken,
            "final_score": final_score if self.current_dataset is not None else 0.0,
            "success": success,
            "total_reward": sum(self.episode_rewards),
            "episode_complete": True,
        }
        
        done = True
        
        return obs, final_reward, done, info
    
    def _get_observation(self) -> Observation:
        """Generate current observation."""
        if self.current_dataset is None:
            return self._empty_observation()
        
        # Recalculate quality metrics
        duplicate_rows = self.current_dataset.duplicated(keep=False).sum()
        missing_cells = self.current_dataset.isnull().sum().sum()
        total_cells = len(self.current_dataset) * len(self.current_dataset.columns)
        
        detected_issues = []
        if duplicate_rows > 0:
            detected_issues.append("duplicates")
        if missing_cells > 0:
            detected_issues.append("missing_values")
        if len(self.known_issues["type_errors"]) > 0:
            detected_issues.append("type_errors")
        if len(self.known_issues["outliers"]) > 0:
            detected_issues.append("outliers")
        
        quality_score = 1.0 - (
            (duplicate_rows / max(len(self.current_dataset), 1)) * 0.25 +
            (missing_cells / max(total_cells, 1)) * 0.25 +
            (len(self.known_issues["type_errors"]) / max(len(self.current_dataset), 1)) * 0.25 +
            (len(self.known_issues["outliers"]) / max(len(self.current_dataset), 1)) * 0.25
        )
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Build column info
        column_info_list = []
        for col in self.current_dataset.columns:
            col_data = self.current_dataset[col]
            column_info_list.append(ColumnInfo(
                name=col,
                data_type=str(col_data.dtype),
                missing_count=col_data.isnull().sum(),
                missing_pct=col_data.isnull().sum() / len(self.current_dataset),
                duplicate_count=col_data.duplicated().sum(),
                has_type_errors=any(idx in self.known_issues["type_errors"] for idx in range(len(self.current_dataset))),
                has_outliers=any(idx in self.known_issues["outliers"] for idx in range(len(self.current_dataset))),
                sample_values=[str(v)[:50] for v in col_data.dropna().head(3).values],
            ))
        
        dataset_stats = DatasetStats(
            total_rows=len(self.current_dataset),
            total_columns=len(self.current_dataset.columns),
            pct_missing=missing_cells / max(total_cells, 1),
            pct_duplicates=duplicate_rows / max(len(self.current_dataset), 1),
            detected_issues=detected_issues,
            quality_score=quality_score,
        )
        
        return Observation(
            current_dataset=self.current_dataset.head(10).to_dict(orient="records"),
            dataset_stats=dataset_stats,
            column_info=column_info_list,
            steps_remaining=self.max_steps - self.steps_taken,
            previous_action_result=self.last_action_result,
            task_id=self.current_task_id or "unknown",
            task_description=self.task_description,
        )
    
    def _empty_observation(self) -> Observation:
        """Generate empty observation for uninitialized state."""
        return Observation(
            current_dataset=[],
            dataset_stats=DatasetStats(
                total_rows=0,
                total_columns=0,
                pct_missing=0.0,
                pct_duplicates=0.0,
                detected_issues=[],
                quality_score=0.0,
            ),
            column_info=[],
            steps_remaining=self.max_steps,
            previous_action_result="Environment not initialized",
            task_id="none",
            task_description="",
        )
    
    def state(self) -> Dict[str, Any]:
        """
        Get complete internal state for debugging/inspection.
        
        Returns:
            Dictionary with full environment state
        """
        return {
            "current_task_id": self.current_task_id,
            "current_dataset": self.current_dataset.to_dict() if self.current_dataset is not None else None,
            "original_dataset": self.original_dataset.to_dict() if self.original_dataset is not None else None,
            "steps_taken": self.steps_taken,
            "episode_rewards": self.episode_rewards,
            "action_history": self.action_history,
            "known_issues": self.known_issues,
        }
    
    async def close(self):
        """Clean up resources."""
        pass


# For compatibility with OpenEnv discovery
__all__ = ["DataCleanerEnv"]
