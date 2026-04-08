"""Action model for the DataCleanerEnv environment."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal
from enum import Enum


class OperationType(str, Enum):
    """Available operations for data cleaning."""
    FILL_MISSING = "FILL_MISSING"
    REMOVE_DUPLICATES = "REMOVE_DUPLICATES"
    FIX_TYPES = "FIX_TYPES"
    REMOVE_OUTLIERS = "REMOVE_OUTLIERS"
    DECLARE_CLEAN = "DECLARE_CLEAN"


class Action(BaseModel):
    """Action taken by agent in the DataCleanerEnv environment."""
    
    operation: OperationType = Field(
        description="Type of cleaning operation to perform"
    )
    column: Optional[str] = Field(
        default=None,
        description="Column name to operate on (required for most operations except DECLARE_CLEAN)"
    )
    strategy: Optional[str] = Field(
        default=None,
        description="Strategy for operation (e.g., 'mean', 'forward_fill', 'drop', 'iqr', 'zscore')"
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for operation (e.g., {'threshold': 2.5} for zscore)"
    )
    
    @property
    def is_terminal_action(self) -> bool:
        """Check if this action ends the episode."""
        return self.operation == OperationType.DECLARE_CLEAN


# Action examples for documentation
ACTION_EXAMPLES = {
    "fill_missing_mean": {
        "operation": "FILL_MISSING",
        "column": "age",
        "strategy": "mean"
    },
    "fill_missing_forward": {
        "operation": "FILL_MISSING",
        "column": "date",
        "strategy": "forward_fill"
    },
    "remove_duplicates": {
        "operation": "REMOVE_DUPLICATES",
        "column": None,
        "strategy": "drop"
    },
    "fix_types": {
        "operation": "FIX_TYPES",
        "column": "revenue",
        "strategy": "infer"
    },
    "remove_outliers_iqr": {
        "operation": "REMOVE_OUTLIERS",
        "column": "salary",
        "strategy": "iqr",
        "params": {"multiplier": 1.5}
    },
    "remove_outliers_zscore": {
        "operation": "REMOVE_OUTLIERS",
        "column": "score",
        "strategy": "zscore",
        "params": {"threshold": 3.0}
    },
    "declare_clean": {
        "operation": "DECLARE_CLEAN",
        "column": None,
        "strategy": None
    }
}
