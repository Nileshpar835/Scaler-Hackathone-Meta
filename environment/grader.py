"""Graders for evaluating DataCleanerEnv task completion."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from .data_loader import load_task_from_json


class TaskGrader:
    """Base class for task grading."""
    
    def grade(self, final_df: pd.DataFrame, original_df: pd.DataFrame, task_id: str) -> float:
        """
        Grade task completion.
        
        Args:
            final_df: Final cleaned DataFrame
            original_df: Original clean DataFrame (ground truth)
            task_id: Task identifier
        
        Returns:
            Score in [0.0, 1.0]
        """
        raise NotImplementedError


class EasyTaskGrader(TaskGrader):
    """Grader for easy task (remove duplicates)."""
    
    def grade(self, final_df: pd.DataFrame, original_df: pd.DataFrame, task_id: str) -> float:
        """
        Grade easy task: Perfect score if no duplicates remain.
        
        Args:
            final_df: Final DataFrame (should have all duplicates removed)
            original_df: Original clean DataFrame
            task_id: Task identifier
        
        Returns:
            Score [0.0, 1.0] where 1.0 = no duplicates, 0.0 = many duplicates
        """
        if len(final_df) == 0:
            return 0.0  # Empty dataset is wrong
        
        # Count duplicate rows (considering all columns)
        duplicates = final_df.duplicated(keep=False).sum()
        
        if duplicates == 0:
            return 1.0
        
        # Partial credit: score based on how many duplicates remain
        score = 1.0 - (duplicates / (2 * len(final_df)))  # Each duplicate is 2 entries
        return max(0.0, score)


class MediumTaskGrader(TaskGrader):
    """Grader for medium task (multiple issue types)."""
    
    def grade(self, final_df: pd.DataFrame, original_df: pd.DataFrame, task_id: str) -> float:
        """
        Grade medium task: Multi-criteria scoring for duplicates, missing values, and type errors.
        
        Args:
            final_df: Final cleaned DataFrame
            original_df: Original clean DataFrame (ground truth)
            task_id: Task identifier
        
        Returns:
            Score [0.0, 1.0] based on resolution of all three issue types
        """
        if len(final_df) == 0:
            return 0.0
        
        scores = {}
        
        # 1. Duplicate score: penalize remaining duplicates
        duplicates_remaining = final_df.duplicated(keep=False).sum()
        duplicate_score = max(0.0, 1.0 - (duplicates_remaining / (2 * len(final_df))))
        scores["duplicates"] = duplicate_score
        
        # 2. Missing values score: penalize remaining missing values
        missing_pct = final_df.isnull().sum().sum() / (len(final_df) * len(final_df.columns))
        missing_score = max(0.0, 1.0 - missing_pct)
        scores["missing_values"] = missing_score
        
        # 3. Type consistency score: check if all columns match original types
        type_score = self._evaluate_type_consistency(final_df, original_df)
        scores["type_errors"] = type_score
        
        # Weighted average
        weighted_score = (
            0.33 * scores["duplicates"] +
            0.33 * scores["missing_values"] +
            0.34 * scores["type_errors"]
        )
        
        return min(1.0, weighted_score)
    
    def _evaluate_type_consistency(self, final_df: pd.DataFrame, original_df: pd.DataFrame) -> float:
        """Evaluate how well types match between final and original."""
        if final_df.shape[1] != original_df.shape[1]:
            return 0.5  # Column mismatch
        
        type_matches = 0
        total_cols = len(final_df.columns)
        
        for col in final_df.columns:
            if col in original_df.columns:
                orig_dtype = original_df[col].dtype
                final_dtype = final_df[col].dtype
                
                # Check if dtype is reasonable (numeric, object, datetime)
                orig_category = self._dtype_category(orig_dtype)
                final_category = self._dtype_category(final_dtype)
                
                if orig_category == final_category:
                    type_matches += 1
        
        return type_matches / total_cols if total_cols > 0 else 0.0
    
    @staticmethod
    def _dtype_category(dtype) -> str:
        """Categorize dtype."""
        if dtype in [np.int64, np.int32, int]:
            return "integer"
        elif dtype in [np.float64, np.float32, float]:
            return "float"
        elif dtype in ["object", "string"]:
            return "string"
        else:
            return "other"


class HardTaskGrader(TaskGrader):
    """Grader for hard task (outlier detection with false positive risk)."""
    
    def grade(self, final_df: pd.DataFrame, original_df: pd.DataFrame, task_id: str) -> float:
        """
        Grade hard task: Reward removing outliers, penalize removing valid data.
        
        Args:
            final_df: Final DataFrame (outliers should be removed, but not valid data)
            original_df: Original clean DataFrame (ground truth)
            task_id: Task identifier
        
        Returns:
            Score [0.0, 1.0] based on outlier removal accuracy and false positives
        """
        if len(final_df) == 0:
            return 0.0
        
        # Primary score: how much of original data was kept
        # Ideal: keep all of original, remove only true outliers
        
        rows_removed = len(original_df) - len(final_df)
        if rows_removed < 0:
            # Added rows (bad sign)
            return 0.0
        
        # Rows removed should roughly match the number of actual outliers introduced
        # In our hard task, we introduce ~10 outliers per 750 rows
        max_removable = int(len(original_df) * 0.05)  # Max 5% should be outliers
        
        if rows_removed <= max_removable:
            # Good: removed limited rows (conservative outlier removal)
            removal_score = 1.0 - (rows_removed / (max_removable + 1))
        else:
            # Bad: removed too many rows (likely false positives)
            removal_score = max(0.1, 1.0 - (rows_removed / len(original_df)))
        
        # Secondary score: does final data match original structure?
        structural_score = 0.0
        if len(final_df) > 0 and set(final_df.columns) == set(original_df.columns):
            # Check shape and types
            structural_score = 0.8
            
            # Bonus if no missing values remain
            if final_df.isnull().sum().sum() == 0:
                structural_score = 1.0
        
        # Combined score: prioritize not removing valid data
        final_score = 0.7 * removal_score + 0.3 * structural_score
        return min(1.0, max(0.0, final_score))


def grade_episode(
    final_df: pd.DataFrame,
    original_df: pd.DataFrame,
    task_id: str
) -> float:
    """
    Unified interface to grade an episode.
    
    Args:
        final_df: Final cleaned DataFrame
        original_df: Original clean DataFrame (ground truth)
        task_id: Task identifier (e.g., "task_easy", "task_medium", "task_hard")
    
    Returns:
        Float score in [0.0, 1.0]
    """
    if "easy" in task_id:
        grader = EasyTaskGrader()
    elif "medium" in task_id:
        grader = MediumTaskGrader()
    elif "hard" in task_id:
        grader = HardTaskGrader()
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
    
    return grader.grade(final_df, original_df, task_id)
