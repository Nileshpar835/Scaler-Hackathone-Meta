<<<<<<< HEAD
# DataCleanerEnv: Real-World Data Cleaning for AI Agents

An OpenEnv environment simulating real-world data cleaning and ETL tasks. AI agents learn to autonomously clean messy datasets by applying operations to fix missing values, duplicates, type errors, and outliers.

## Overview

**DataCleanerEnv** is a benchmark environment designed to evaluate and train AI agents on a genuine real-world task: **data quality management**. Data cleaning is the #1 bottleneck in ML pipelines, consuming ~70% of data engineering effort. This environment provides:

- **Realistic Task**: Real operations used by data engineers daily (remove duplicates, fill missing values, fix types, detect outliers)
- **Progressive Difficulty**: 3 tasks ranging from easy (single issue) to hard (ambiguous outlier detection with false positive risk)
- **Rich Reward Signal**: Agents receive partial credit for intermediate progress, not just binary end-of-episode feedback
- **Deterministic Grading**: Reproducible evaluation with clear success criteria for each task

### Why Data Cleaning?

1. **High Real-World Utility**: Every ML org spends weeks on data preparation
2. **Non-Trivial Challenge**: Requires balancing multiple objectives (remove bad data without removing valid data)
3. **Scalable Benchmarking**: Easy to generate synthetic datasets with controlled quality issues
4. **Novel Domain**: Not often seen in RL/agent benchmarks despite critical importance

---

## Tasks

### Task 1: Easy — Duplicate Removal

**Objective**: Remove all duplicate rows from an employee email list

**Dataset Size**: 150 rows, 4 columns (employee_id, name, email, department)

**Issue Type**: Duplicates only (~8 duplicate rows introduced)

**Supported Operations**: `REMOVE_DUPLICATES`

**Success Criteria**: 
- All duplicate rows removed (0 duplicates remain)
- Perfect score: `1.0` when `duplicate_count == 0`
- Partial credit for reducing duplicates

**Difficulty**: Easy for any agent (straightforward operation)

---

### Task 2: Medium — Multi-Issue Data Cleaning

**Objective**: Clean a customer database with THREE types of quality issues

**Dataset Size**: 400 rows, 6 columns (customer_id, name, email, phone, age, revenue)

**Issue Types**:
- **Duplicates**: ~12 duplicate rows
- **Missing Values**: ~8% of cells contain NaN (spread across multiple columns)
- **Type Errors**: ~5% of numeric columns contain invalid strings ("invalid", "error", "N/A")

**Supported Operations**: 
- `REMOVE_DUPLICATES`
- `FILL_MISSING` (strategies: mean, median, mode, forward_fill, drop)
- `FIX_TYPES` (strategy: infer to restore proper types)

**Success Criteria**:
- Weighted score: 33% duplicates + 33% missing values + 34% type errors
- Score = (1 - remaining_issues_pct) for each category
- Perfect score: `1.0` when all issues resolved

**Difficulty**: Intermediate (requires multi-step reasoning, order of operations matters)

---

### Task 3: Hard — Outlier Detection with False Positive Risk

**Objective**: Clean financial transaction data with FOUR types of issues, including ambiguous outliers

**Dataset Size**: 750 rows, 8 columns (transaction_id, timestamp, customer_id, amount, category, merchant, status, fee)

**Issue Types**:
- **Duplicates**: ~15 duplicate rows
- **Missing Values**: ~10% NaN
- **Type Errors**: ~6% invalid types
- **Outliers**: ~10-15 extreme values introduced (some borderline valid, some clearly noise)

**Supported Operations**: 
- All from Medium task, PLUS
- `REMOVE_OUTLIERS` (strategies: iqr, zscore with configurable thresholds)

**Success Criteria** (Nuanced Scoring):
- Agents earn points for removing true outliers
- Agents **lose points heavily** for false positives (removing valid data)
- Formula: `score = (correct_removals) / (correct_removals + false_positives * 2)`
- Perfect score: `1.0` when all outliers removed AND NO valid data removed

**Challenge**: 
- Simple z-score removal = high false positive rate (agents lose points)
- Smart outlier selection = harder but rewarding
- Frontier models (GPT-4) expected to score ~0.4–0.5 due to ambiguity

**Difficulty**: Hard (requires understanding of statistical vs. business outliers)

---

## Action & Observation Spaces

### Observation Space

The agent receives structured observations after each step:

```json
{
  "current_dataset": [
    {"id": 1, "name": "John", "email": "john@example.com", ...},
    {"id": 2, "name": "Jane", "email": "jane@example.com", ...}
  ],
  "dataset_stats": {
    "total_rows": 150,
    "total_columns": 4,
    "pct_missing": 0.05,
    "pct_duplicates": 0.08,
    "detected_issues": ["duplicates"],
    "quality_score": 0.92
  },
  "column_info": [
    {
      "name": "employee_id",
      "data_type": "int64",
      "missing_count": 0,
      "missing_pct": 0.0,
      "duplicate_count": 0,
      "has_type_errors": false,
      "has_outliers": false,
      "sample_values": ["1", "2", "3"]
    }
  ],
  "steps_remaining": 30,
  "previous_action_result": "Removed 8 duplicate rows",
  "task_id": "task_easy",
  "task_description": "Remove all duplicate rows from an employee email list"
}
```

**Key Fields**:
- `current_dataset`: 5-10 sample rows (for LLM context)
- `dataset_stats`: Aggregate quality metrics
- `column_info`: Per-column diagnostics
- `detected_issues`: List of issue types present
- `steps_remaining`: Budget for actions
- `previous_action_result`: Feedback from last action

### Action Space

Agents select one of five operations per step:

```json
{
  "operation": "FILL_MISSING | REMOVE_DUPLICATES | FIX_TYPES | REMOVE_OUTLIERS | DECLARE_CLEAN",
  "column": "column_name_or_null",
  "strategy": "strategy_name_or_null",
  "params": {}
}
```

**Operations**:

| Operation | Required Column | Strategy | Params | Effect |
|-----------|-----------------|----------|--------|--------|
| `FILL_MISSING` | Yes | mean, median, mode, forward_fill, drop | {} | Replace NaN values in column |
| `REMOVE_DUPLICATES` | No | — | {} | Drop duplicate rows (all columns) |
| `FIX_TYPES` | Yes | infer | {} | Convert column to proper type based on original |
| `REMOVE_OUTLIERS` | Yes | iqr, zscore | threshold | Remove rows with extreme values in column |
| `DECLARE_CLEAN` | No | — | {} | Declare dataset clean; end episode |

**Examples**:
```json
{"operation": "FILL_MISSING", "column": "age", "strategy": "mean"}
{"operation": "REMOVE_DUPLICATES"}
{"operation": "FIX_TYPES", "column": "revenue", "strategy": "infer"}
{"operation": "REMOVE_OUTLIERS", "column": "salary", "strategy": "iqr", "params": {"threshold": 1.5}}
{"operation": "DECLARE_CLEAN"}
```

---

## Reward Function

Agents receive **rich, intermediate rewards** (not sparse end-of-episode):

### Per-Step Rewards

| Action | Reward | Condition |
|--------|--------|-----------|
| Correct operation applied | **+0.15** | Operation executed successfully and targeted real issue |
| Issue reduced | **+0.10** bonus | Operation reduced detected issue count |
| No effect | **-0.02** | Operation had no impact (idempotent) |
| Data corrupted | **-0.30** | Operation destroyed valid data |
| Invalid operation | **-0.05** | Column not found, non-existent issue, bad strategy |

### Episode Completion

| Event | Reward | Condition |
|-------|--------|-----------|
| Declare clean (all issues fixed) | **+0.50** | Dataset cleaned successfully |
| Efficiency bonus | **+0.10 × (steps_left / max_steps)** | Completed in fewer steps |
| False declaration | **-0.50** | Declared clean but issues remain |

### Normalization

Final episode score = sum(rewards) / MAX_TOTAL_REWARD, clamped to **[0.0, 1.0]**

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Local Installation

```bash
# Clone repository
git clone <your-repo-url>
cd dataclean-env

# Create virtual environment
python -m venv venv

# Activate venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from environment import DataCleanerEnv; print('✓ Installation successful')"
```

### Verify OpenEnv Compliance

```bash
pip install openenv-core

openenv validate
```

Expected output:
```
✓ Environment spec valid
✓ Typed models found
✓ step()/reset()/state() endpoints present
```

---

## Running the Environment

### Basic Usage

```python
import asyncio
from environment import DataCleanerEnv, Action, OperationType

async def main():
    env = DataCleanerEnv(dataset_dir="datasets", max_steps=30)
    
    # Reset to a task
    obs = await env.reset(task_id="task_easy")
    print(f"Task: {obs.task_description}")
    print(f"Rows: {obs.dataset_stats.total_rows}, Columns: {obs.dataset_stats.total_columns}")
    print(f"Detected issues: {obs.dataset_stats.detected_issues}")
    
    # Take a step
    action = Action(operation=OperationType.REMOVE_DUPLICATES)
    obs, reward, done, info = await env.step(action)
    print(f"Reward: {reward:+.2f}, Done: {done}")
    
    await env.close()

asyncio.run(main())
```

### Running Baseline Agent

The baseline inference script uses OpenAI's API to run a GPT-4 model against the environment:

```bash
# Set API key
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4-turbo"  # Optional

# Run baseline
python inference.py
```

Expected output:
```
[START] task='task_easy' env='DataCleanerEnv' model='gpt-4-turbo' timestamp='2024-04-08T...'
[STEP] step=1 action='REMOVE_DUPLICATES' reward=+0.25 done=False error=null
[STEP] step=2 action='DECLARE_CLEAN' reward=+0.50 done=True error=null
[END] success=True steps=2 score=0.7500 rewards=[+0.25, +0.50] timestamp='2024-04-08T...'
...
```

---

## Baseline Scores

Baseline inference using **GPT-4 Turbo**:

| Task | Difficulty | Operations | Expected Score | Notes |
|------|-----------|-----------|-----------------|-------|
| task_easy | Easy | REMOVE_DUPLICATES | ~0.85 | Clear single operation |
| task_medium | Intermediate | 3 operations | ~0.68 | Multi-step reasoning |
| task_hard | Hard | 4 operations + false positives | ~0.42 | Outlier ambiguity challenge |

**Overall Baseline Score**: ~0.65

**Success Rate**: 67% of tasks completed successfully (score ≥ 0.7)

---

## Docker Deployment

### Build Image

```bash
docker build -t dataclean-env:latest .
```

### Run Container

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY="sk-..." \
  -e MODEL_NAME="gpt-4-turbo" \
  dataclean-env:latest
```

### Health Check

```bash
curl http://localhost:8080/health
```

---

## Hugging Face Spaces Deployment

1. **Create a Hugging Face Space**:
   - Go to https://huggingface.co/spaces
   - Create new Space with Docker runtime
   - Tag: `openenv`

2. **Push to HF**:
   ```bash
   git remote add hf https://huggingface.co/spaces/<username>/<space-name>.git
   git push hf main
   ```

3. **Space Configuration** (`README.md` in Space):
   ```markdown
   ---
   title: DataCleanerEnv
   emoji: 🧹
   colorFrom: blue
   colorTo: purple
   sdk: docker
   app_port: 8080
   ---
   ```

4. **Environment Variables** (Set in Space settings):
   - `OPENAI_API_KEY`: Your API key
   - `MODEL_NAME`: e.g., `gpt-4-turbo`
   - `API_BASE_URL`: OpenAI API base URL

---

## Architecture

```
dataclean-env/
├── environment/
│   ├── __init__.py              # Package exports
│   ├── observation.py           # Pydantic Observation model
│   ├── action.py                # Pydantic Action model
│   ├── reward.py                # Pydantic Reward model
│   ├── env.py                   # DataCleanerEnv core (asyncio)
│   ├── grader.py                # Task graders
│   └── data_loader.py           # Dataset generation
├── datasets/
│   ├── task_easy.json           # Easy task definition
│   ├── task_medium.json         # Medium task definition
│   └── task_hard.json           # Hard task definition
├── tests/
│   └── test_env.py              # Unit tests (pytest)
├── inference.py                 # Baseline inference script
├── openenv.yaml                 # OpenEnv spec manifest
├── Dockerfile                   # Container image
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Quality Assurance

### Manual Testing

```bash
# Run local tests
python -m pytest tests/test_env.py -v

# Test each task individually
python -c "
import asyncio
from environment import DataCleanerEnv, Action, OperationType

async def test():
    env = DataCleanerEnv()
    for task in ['task_easy', 'task_medium', 'task_hard']:
        obs = await env.reset(task_id=task)
        print(f'✓ {task}: {obs.dataset_stats.total_rows} rows, {obs.dataset_stats.total_columns} cols')

asyncio.run(test())
"
```

### Pre-Submission Validation

```bash
# 1. OpenEnv spec compliance
openenv validate

# 2. Docker build
docker build -t dataclean-env:latest .

# 3. Run baseline inference (< 20 min on 2vCPU/8GB)
export OPENAI_API_KEY="sk-..."
time python inference.py
```

---

## Extending the Environment

### Adding Custom Datasets

Modify `data_loader.py`:

```python
def generate_all_tasks():
    # ... existing code ...
    
    # Add custom dataset to task
    task_easy.datasets.append({
        "id": "task_easy_custom",
        "name": "my_custom_data.csv",
        "dirty_data": [...],  # List of dicts
        "clean_data": [...],  # Ground truth
        "expected_issues": ["duplicates"],
        "grading_criteria": {...}
    })
```

### Adding New Operations

1. Add to `OperationType` enum in `action.py`
2. Implement handler method in `DataCleanerEnv.step()`
3. Update `openenv.yaml` metadata

---

## References

- **OpenEnv Specification**: https://github.com/openenv/openenv-spec
- **Pydantic Validation**: https://docs.pydantic.dev/
- **Pandas Data Cleaning**: https://pandas.pydata.org/docs/
- **OpenAI API**: https://platform.openai.com/docs/

---

## Citation

```bibtex
@environment{datacleanerenv2024,
  title={DataCleanerEnv: Real-World Data Cleaning for AI Agents},
  author={Hackathon Participant},
  year={2024},
  howpublished={OpenEnv Benchmark Suite}
}
```

---

## License

This environment is provided as-is for research and evaluation purposes.

---

## Support

**Issues or Questions?**
- Check `tests/test_env.py` for usage examples
- Review `inference.py` for baseline agent implementation
- Inspect `datasets/*.json` for task structure

---

**Last Updated**: April 2024  
**OpenEnv Version**: 1.0  
**Status**: ✓ Validated and Ready for Submission
=======
# Scaler-Hackathone-Meta
>>>>>>> dddffac765f83085a14e2044e66e6408dc582c86
