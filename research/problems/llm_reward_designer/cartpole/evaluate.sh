#!/usr/bin/env bash
# evaluate.sh
set -e

# Install dependencies inside the container
pip install "gymnasium<1.0" numpy

# Run the python evaluator using absolute path for solution
python3 evaluator.py --solution-path /work/solution/solution.py
