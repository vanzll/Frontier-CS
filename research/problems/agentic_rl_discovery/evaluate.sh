#!/bin/bash
# Entry point for evaluating Agentic RL Algorithm Discovery solutions.
#
# This script runs inside Docker container and:
# 1. Sets up the environment (verl-agent, ALFWorld)
# 2. Runs the evaluator with the user's solution
# 3. Outputs the final score

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="${SCRIPT_DIR}/resources"
VERL_AGENT_DIR="${RESOURCES_DIR}/verl-agent"

# Set up Python path - include verl-agent and resources
export PYTHONPATH="${VERL_AGENT_DIR}:${RESOURCES_DIR}:${PYTHONPATH:-}"

# Run environment setup (installs verl-agent, downloads ALFWorld data)
echo "Running environment setup..."
source "${SCRIPT_DIR}/set_up_env.sh"

# Find solution file (Frontier-CS Docker framework puts it here)
EXEC_ROOT="/work/execution_env"
if [[ -f "${EXEC_ROOT}/solution_env/solution.py" ]]; then
    SOLUTION_PATH="${EXEC_ROOT}/solution_env/solution.py"
elif [[ -f "/workspace/solution/solution.py" ]]; then
    SOLUTION_PATH="/workspace/solution/solution.py"
elif [[ -n "$1" ]]; then
    SOLUTION_PATH="$1"
else
    echo "Error: No solution file found" >&2
    echo "0"
    exit 1
fi

echo "=== Agentic RL Algorithm Discovery Evaluator ==="
echo "Solution: ${SOLUTION_PATH}"
echo "Resources: ${RESOURCES_DIR}"
echo "verl-agent: ${VERL_AGENT_DIR}"
echo ""

# Run the evaluator
python3 "${SCRIPT_DIR}/evaluator.py" --solution "${SOLUTION_PATH}"
