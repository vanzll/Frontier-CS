#!/bin/bash
# Generate solutions for Frontier-CS problems using LLM APIs
#
# Usage:
#   ./generate_solutions.sh --problem agentic_rl_discovery --model gemini/gemini-2.5-pro
#   ./generate_solutions.sh --problem flash_attn --model gpt-5 --indices 3
#   ./generate_solutions.sh --dryrun  # Preview without generating

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# API Keys - set your keys here or export them as environment variables
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-AIzaSyDwQ5ecrTjx-xuMz_EJZIQgzpmAm4AVi9s}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-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}"
# export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
# export XAI_API_KEY="${XAI_API_KEY:-}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-0df1ac2770a1406eb878e535d6b2b795}"

# Activate virtual environment
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Run the Python script with all arguments
cd "$REPO_ROOT"
python research/scripts/generate_solutions.py "$@"
