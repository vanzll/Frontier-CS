# Problem Count Tracker

This directory contains tools to automatically track and update the number of problems in Frontier-CS.

## How It Works

The `update_problem_count.py` script automatically:
- Counts research problems in the `research/` directory
- Counts algorithmic problems in the `algorithmic/problems/` directory
- Updates the README.md with badge indicators showing current counts

## Manual Usage

To manually update the problem count:

```bash
python3 scripts/update_problem_count.py
```

This will:
1. Scan the repository for problems
2. Display statistics in the terminal
3. Update the README.md with new badge counts

## Automatic Updates

### GitHub Actions (CI/CD)

The `.github/workflows/update_problem_count.yml` workflow automatically:
- Runs when changes are pushed to `research/` or `algorithmic/problems/`
- Updates the README with new counts
- Commits the changes automatically

### Pre-commit Hook (Local)

To enable automatic updates on every commit:

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install
```

Now the problem count will be updated automatically whenever you commit changes to problem directories.

## Problem Counting Logic

### Research Problems
Counts directories in `research/` that contain:
- `config.yaml` file, OR
- `readme` or `README.md` file, OR
- `evaluator.py` file

Excludes: `results/`, `scripts/`, `test_scripts/`, and hidden directories

### Algorithmic Problems
Counts numbered directories in `algorithmic/problems/`
- Only directories with numeric names (e.g., `1/`, `42/`, `107/`)

## Badge Display

The script adds/updates badges in the README:

![Research Problems](https://img.shields.io/badge/Research_Problems-12-blue) ![Algorithmic Problems](https://img.shields.io/badge/Algorithmic_Problems-107-green)

These badges are automatically generated and will update whenever the script runs.
