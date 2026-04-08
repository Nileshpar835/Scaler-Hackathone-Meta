"""Data loader and generator for DataCleanerEnv tasks and datasets."""

import json
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class DataQualityIssue:
    """Represents a single data quality issue."""
    issue_type: str  # "missing_value", "duplicate", "type_error", "outlier"
    column: str
    row_indices: List[int]
    severity: float  # 0.0-1.0


@dataclass
class TaskDefinition:
    """Definition of a single task."""
    task_id: str
    difficulty: str  # "easy", "medium", "hard"
    description: str
    expected_operations: List[str]
    expected_issues: List[str]
    datasets: List[Dict[str, Any]]


class DataGenerator:
    """Generates synthetic dirty datasets with known issues."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_clean_dataset(
        self,
        num_rows: int = 100,
        columns: List[Tuple[str, str]] = None
    ) -> pd.DataFrame:
        """
        Generate a clean base dataset.
        
        Args:
            num_rows: Number of rows
            columns: List of (name, type) tuples. Types: "int", "float", "string", "date"
        
        Returns:
            Clean pandas DataFrame
        """
        if columns is None:
            columns = [
                ("id", "int"),
                ("name", "string"),
                ("email", "string"),
                ("age", "int"),
                ("salary", "float"),
            ]
        
        data = {}
        for col_name, col_type in columns:
            if col_type == "int":
                data[col_name] = list(range(num_rows))
            elif col_type == "float":
                data[col_name] = [float(i) * 1.5 + random.random() * 100 for i in range(num_rows)]
            elif col_type == "string":
                if "name" in col_name:
                    data[col_name] = [f"person_{i}" for i in range(num_rows)]
                elif "email" in col_name:
                    data[col_name] = [f"user_{i}@example.com" for i in range(num_rows)]
                else:
                    data[col_name] = [f"value_{i}" for i in range(num_rows)]
            elif col_type == "date":
                data[col_name] = pd.date_range("2020-01-01", periods=num_rows).astype(str)
        
        return pd.DataFrame(data)
    
    def introduce_duplicates(
        self,
        df: pd.DataFrame,
        num_duplicates: int = 5,
        duplicate_of_rows: List[int] = None
    ) -> Tuple[pd.DataFrame, List[DataQualityIssue]]:
        """
        Introduce duplicate rows.
        
        Args:
            df: Clean DataFrame
            num_duplicates: Number of duplicate rows to create
            duplicate_of_rows: Specific row indices to duplicate (random if None)
        
        Returns:
            Tuple of (dirty DataFrame, list of issues)
        """
        if duplicate_of_rows is None:
            duplicate_of_rows = random.sample(range(len(df)), min(num_duplicates, len(df)))
        
        duplicated_rows = [df.iloc[i].to_dict() for i in duplicate_of_rows]
        df_dirty = pd.concat(
            [df, pd.DataFrame(duplicated_rows)],
            ignore_index=True
        )
        
        new_indices = list(range(len(df), len(df) + len(duplicated_rows)))
        issue = DataQualityIssue(
            issue_type="duplicate",
            column=None,
            row_indices=new_indices,
            severity=len(new_indices) / len(df_dirty)
        )
        
        return df_dirty, [issue]
    
    def introduce_missing_values(
        self,
        df: pd.DataFrame,
        columns_to_affect: List[str] = None,
        pct_missing: float = 0.1
    ) -> Tuple[pd.DataFrame, List[DataQualityIssue]]:
        """
        Introduce missing values.
        
        Args:
            df: DataFrame
            columns_to_affect: Columns to introduce missing in (random if None)
            pct_missing: Percentage of values to remove
        
        Returns:
            Tuple of (dirty DataFrame, list of issues)
        """
        df_dirty = df.copy()
        if columns_to_affect is None:
            columns_to_affect = random.sample(list(df.columns), max(1, len(df.columns) // 2))
        
        issues = []
        for col in columns_to_affect:
            num_to_remove = max(1, int(len(df_dirty) * pct_missing))
            rows_to_affect = random.sample(range(len(df_dirty)), num_to_remove)
            df_dirty.loc[rows_to_affect, col] = None
            
            issue = DataQualityIssue(
                issue_type="missing_value",
                column=col,
                row_indices=rows_to_affect,
                severity=pct_missing
            )
            issues.append(issue)
        
        return df_dirty, issues
    
    def introduce_type_errors(
        self,
        df: pd.DataFrame,
        columns_to_affect: List[str] = None,
        pct_errors: float = 0.05
    ) -> Tuple[pd.DataFrame, List[DataQualityIssue]]:
        """
        Introduce type errors (e.g., strings in numeric columns).
        
        Args:
            df: DataFrame
            columns_to_affect: Columns to introduce errors in
            pct_errors: Percentage of values to corrupt
        
        Returns:
            Tuple of (dirty DataFrame, list of issues)
        """
        df_dirty = df.copy()
        if columns_to_affect is None:
            numeric_cols = df_dirty.select_dtypes(include=[np.number]).columns.tolist()
            columns_to_affect = random.sample(numeric_cols, min(2, len(numeric_cols)))
        
        issues = []
        for col in columns_to_affect:
            if df_dirty[col].dtype in [np.int64, np.float64]:
                num_to_corrupt = max(1, int(len(df_dirty) * pct_errors))
                rows_to_affect = random.sample(range(len(df_dirty)), num_to_corrupt)
                corruptions = ["invalid", "error", "N/A", "unknown"]
                for row_idx in rows_to_affect:
                    df_dirty.at[row_idx, col] = random.choice(corruptions)
                
                issue = DataQualityIssue(
                    issue_type="type_error",
                    column=col,
                    row_indices=rows_to_affect,
                    severity=pct_errors
                )
                issues.append(issue)
        
        return df_dirty, issues
    
    def introduce_outliers(
        self,
        df: pd.DataFrame,
        columns_to_affect: List[str] = None,
        num_outliers: int = 5
    ) -> Tuple[pd.DataFrame, List[DataQualityIssue]]:
        """
        Introduce outlier values.
        
        Args:
            df: DataFrame
            columns_to_affect: Numeric columns to affect
            num_outliers: Number of outlier rows to create
        
        Returns:
            Tuple of (dirty DataFrame, list of issues)
        """
        df_dirty = df.copy()
        if columns_to_affect is None:
            numeric_cols = df_dirty.select_dtypes(include=[np.number]).columns.tolist()
            columns_to_affect = random.sample(numeric_cols, min(2, len(numeric_cols)))
        
        issues = []
        rows_affected = set()
        for col in columns_to_affect:
            if df_dirty[col].dtype in [np.int64, np.float64]:
                col_mean = df_dirty[col].mean()
                col_std = df_dirty[col].std()
                
                num_to_add = min(num_outliers, len(df_dirty) // 10)
                for _ in range(num_to_add):
                    row_idx = len(df_dirty)
                    outlier_val = col_mean + random.choice([-1, 1]) * col_std * random.uniform(4, 10)
                    new_row = df_dirty.iloc[-1].to_dict()
                    new_row[col] = outlier_val
                    df_dirty = pd.concat([df_dirty, pd.DataFrame([new_row])], ignore_index=True)
                    rows_affected.add(row_idx)
        
        if rows_affected:
            issue = DataQualityIssue(
                issue_type="outlier",
                column=None,
                row_indices=list(rows_affected),
                severity=len(rows_affected) / len(df_dirty)
            )
            issues.append(issue)
        
        return df_dirty, issues


def generate_all_tasks() -> List[TaskDefinition]:
    """Generate all 3 task definitions with datasets."""
    
    generator = DataGenerator(seed=42)
    tasks = []
    
    # TASK 1: EASY - Single issue type (Duplicates only)
    task_easy = TaskDefinition(
        task_id="task_easy",
        difficulty="easy",
        description="Remove all duplicate rows from an employee email list",
        expected_operations=["REMOVE_DUPLICATES"],
        expected_issues=["duplicates"],
        datasets=[]
    )
    
    for dataset_idx in range(3):
        clean_df = generator.generate_clean_dataset(
            num_rows=150,
            columns=[
                ("employee_id", "int"),
                ("name", "string"),
                ("email", "string"),
                ("department", "string"),
            ]
        )
        dirty_df, issues = generator.introduce_duplicates(
            clean_df,
            num_duplicates=8
        )
        
        task_easy.datasets.append({
            "id": f"task_easy_dataset_{dataset_idx}",
            "name": f"employees_{dataset_idx}.csv",
            "dirty_data": dirty_df.to_dict(orient="records"),
            "clean_data": clean_df.to_dict(orient="records"),
            "expected_issues": ["duplicates"],
            "grading_criteria": {
                "metric": "duplicate_count",
                "target": 0,
                "tolerance": 0,
                "description": "All duplicate rows must be removed"
            }
        })
    
    tasks.append(task_easy)
    
    # TASK 2: MEDIUM - Multiple issue types
    task_medium = TaskDefinition(
        task_id="task_medium",
        difficulty="medium",
        description="Clean a customer database with missing values, duplicates, and type errors",
        expected_operations=["REMOVE_DUPLICATES", "FILL_MISSING", "FIX_TYPES"],
        expected_issues=["duplicates", "missing_values", "type_errors"],
        datasets=[]
    )
    
    for dataset_idx in range(3):
        clean_df = generator.generate_clean_dataset(
            num_rows=400,
            columns=[
                ("customer_id", "int"),
                ("name", "string"),
                ("email", "string"),
                ("phone", "string"),
                ("age", "int"),
                ("revenue", "float"),
            ]
        )
        # Add duplicates
        dirty_df, _ = generator.introduce_duplicates(clean_df, num_duplicates=12)
        # Add missing values
        dirty_df, _ = generator.introduce_missing_values(dirty_df, pct_missing=0.08)
        # Add type errors
        dirty_df, _ = generator.introduce_type_errors(dirty_df, pct_errors=0.05)
        
        task_medium.datasets.append({
            "id": f"task_medium_dataset_{dataset_idx}",
            "name": f"customers_{dataset_idx}.csv",
            "dirty_data": dirty_df.to_dict(orient="records"),
            "clean_data": clean_df.to_dict(orient="records"),
            "expected_issues": ["duplicates", "missing_values", "type_errors"],
            "grading_criteria": {
                "metric": "multi_issue_resolution",
                "weights": {
                    "duplicates": 0.33,
                    "missing_values": 0.33,
                    "type_errors": 0.34
                },
                "description": "Resolve all three issue types proportionally"
            }
        })
    
    tasks.append(task_medium)
    
    # TASK 3: HARD - Outlier detection with false positive risk
    task_hard = TaskDefinition(
        task_id="task_hard",
        difficulty="hard",
        description="Clean financial dataset with mixed quality issues including ambiguous outliers",
        expected_operations=["REMOVE_DUPLICATES", "FILL_MISSING", "FIX_TYPES", "REMOVE_OUTLIERS"],
        expected_issues=["duplicates", "missing_values", "type_errors", "outliers"],
        datasets=[]
    )
    
    for dataset_idx in range(3):
        clean_df = generator.generate_clean_dataset(
            num_rows=750,
            columns=[
                ("transaction_id", "int"),
                ("timestamp", "date"),
                ("customer_id", "int"),
                ("amount", "float"),
                ("category", "string"),
                ("merchant", "string"),
                ("status", "string"),
                ("fee", "float"),
            ]
        )
        # Add all issues
        dirty_df, _ = generator.introduce_duplicates(clean_df, num_duplicates=15)
        dirty_df, _ = generator.introduce_missing_values(dirty_df, pct_missing=0.10)
        dirty_df, _ = generator.introduce_type_errors(dirty_df, pct_errors=0.06)
        dirty_df, _ = generator.introduce_outliers(dirty_df, num_outliers=10)
        
        task_hard.datasets.append({
            "id": f"task_hard_dataset_{dataset_idx}",
            "name": f"transactions_{dataset_idx}.csv",
            "dirty_data": dirty_df.to_dict(orient="records"),
            "clean_data": clean_df.to_dict(orient="records"),
            "expected_issues": ["duplicates", "missing_values", "type_errors", "outliers"],
            "grading_criteria": {
                "metric": "outlier_accuracy",
                "weights": {
                    "correct_removals": 0.5,
                    "false_positives_penalty": -1.0,  # Heavy penalty for removing valid data
                },
                "description": "Maximize true outlier removals while minimizing false positives"
            }
        })
    
    tasks.append(task_hard)
    
    return tasks


def save_tasks_to_json(tasks: List[TaskDefinition], output_dir: str = "datasets"):
    """Save task definitions to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for task in tasks:
        filename = output_path / f"{task.task_id}.json"
        with open(filename, "w") as f:
            json.dump({
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "description": task.description,
                "expected_operations": task.expected_operations,
                "expected_issues": task.expected_issues,
                "datasets": task.datasets,
            }, f, indent=2, default=str)
        print(f"Saved {filename}")


def load_task_from_json(task_id: str, dataset_dir: str = "datasets") -> TaskDefinition:
    """Load a task definition from JSON file."""
    filename = Path(dataset_dir) / f"{task_id}.json"
    with open(filename, "r") as f:
        data = json.load(f)
    
    return TaskDefinition(
        task_id=data["task_id"],
        difficulty=data["difficulty"],
        description=data["description"],
        expected_operations=data["expected_operations"],
        expected_issues=data["expected_issues"],
        datasets=data["datasets"],
    )


if __name__ == "__main__":
    # Generate and save all tasks
    tasks = generate_all_tasks()
    save_tasks_to_json(tasks)
    print(f"\nGenerated {len(tasks)} tasks successfully!")
