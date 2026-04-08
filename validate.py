"""Quick validation script."""
import json
from pathlib import Path

# Check openenv.yaml exists
if Path('openenv.yaml').exists():
    print('✓ openenv.yaml found')
else:
    print('✗ openenv.yaml MISSING')

# Check task JSON files exist
for task_file in ['task_easy.json', 'task_medium.json', 'task_hard.json']:
    path = Path('datasets') / task_file
    if path.exists():
        with open(path) as f:
            task_data = json.load(f)
        print(f'✓ {task_file}: {len(task_data["datasets"])} datasets, issues={task_data["expected_issues"]}')
    else:
        print(f'✗ {task_file} not found')

# Check key files exist
required_files = [
    'environment/__init__.py',
    'environment/env.py',
    'environment/observation.py',
    'environment/action.py',
    'environment/reward.py',
    'environment/grader.py',
    'environment/data_loader.py',
    'inference.py',
    'Dockerfile',
    'README.md',
]

print()
for file in required_files:
    if Path(file).exists():
        print(f'✓ {file}')
    else:
        print(f'✗ {file} MISSING')

print()
print('Build validation complete!')
