"""Observation model for the DataCleanerEnv environment."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ColumnInfo(BaseModel):
    """Information about a single column in the dataset."""
    name: str
    data_type: str  # e.g., "int", "float", "object", "datetime"
    missing_count: int = Field(ge=0)
    missing_pct: float = Field(ge=0.0, le=1.0)
    duplicate_count: int = Field(ge=0)
    has_type_errors: bool = False
    has_outliers: bool = False
    sample_values: List[str] = Field(max_items=3)  # Example values


class DatasetStats(BaseModel):
    """Statistics about the current dataset state."""
    total_rows: int = Field(gt=0)
    total_columns: int = Field(gt=0)
    pct_missing: float = Field(ge=0.0, le=1.0)
    pct_duplicates: float = Field(ge=0.0, le=1.0)
    detected_issues: List[str] = Field(
        description="List of detected quality issues in dataset",
        examples=[["missing_values", "duplicates", "type_errors"]]
    )
    quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall data quality score (1.0 = perfect)"
    )


class Observation(BaseModel):
    """Observation returned by the DataCleanerEnv environment."""
    
    current_dataset: List[Dict[str, Any]] = Field(
        description="Current dataset as list of dicts (5-10 sample rows)",
        max_items=10
    )
    dataset_stats: DatasetStats
    column_info: List[ColumnInfo] = Field(
        description="Information about each column"
    )
    steps_remaining: int = Field(
        ge=0, le=100,
        description="Number of steps remaining in episode"
    )
    previous_action_result: str = Field(
        description="Human-readable feedback on previous action"
    )
    task_id: str = Field(
        description="Identifier of current task (e.g., 'task_easy', 'task_medium')"
    )
    task_description: str = Field(
        description="Natural language description of task objective"
    )
