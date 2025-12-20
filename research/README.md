# Research Problems

Real-world systems challenges requiring domain expertise in GPU computing, distributed systems, ML pipelines, databases, and security.

## Basic Usage

```bash
# List all problems
frontier-eval --list

# Evaluate a solution (requires Docker)
frontier-eval flash_attn <your_solution.py>

# Evaluate multiple problems
frontier-eval --problems flash_attn,cross_entropy <your_solution.py>
```

## Cloud Evaluation with SkyPilot

Some problems require GPUs or specific hardware. Use [SkyPilot](https://skypilot.readthedocs.io/) to run evaluations on cloud VMs.

**Setup:**

```bash
sky check
```

See [SkyPilot docs](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html) for cloud credential setup.

**Usage:**

```bash
frontier-eval flash_attn <your_solution.py> --skypilot
```

## Batch Evaluation

For evaluating multiple solutions at once, create a pairs file mapping solutions to problems:

```
# pairs.txt format: solution_path:problem_id
solutions/my_flash_attn_v1.py:flash_attn
solutions/my_flash_attn_v2.py:flash_attn
solutions/my_cross_entropy.py:cross_entropy
```

Then run:

```bash
# Evaluate all pairs
frontier-eval batch --pairs-file pairs.txt

# Resume interrupted evaluation
frontier-eval batch --pairs-file pairs.txt --resume

# Check status
frontier-eval batch --status --results-dir results/batch
```

## Python API

```python
from frontier_cs import FrontierCSEvaluator

evaluator = FrontierCSEvaluator()

# Single problem
result = evaluator.evaluate("research", problem_id="flash_attn", code=my_code)
print(f"Score: {result.score}")

# With SkyPilot
result = evaluator.evaluate("research", problem_id="flash_attn", code=my_code,
                           backend="skypilot")

# Batch evaluation
results = evaluator.evaluate_batch("research",
                                  problem_ids=["flash_attn", "cross_entropy"],
                                  code=my_code)
```

## Problem Structure

Each problem is in its own directory under `research/problems/`:

```
research/problems/
├── flash_attn/           # Single problem
│   ├── config.yaml
│   ├── readme
│   ├── evaluator.py
│   └── resources/
├── gemm_optimization/    # Problem with variants
│   ├── squares/
│   ├── rectangles/
│   └── ...
└── ...
```

### File Reference

| File | Purpose |
|------|---------|
| `config.yaml` | Runtime config (Docker image, GPU requirement, timeout) |
| `readme` | Problem description, API spec, scoring formula |
| `set_up_env.sh` | Environment setup (install deps, check CUDA) |
| `download_datasets.sh` | Download datasets (for local pre-download) |
| `evaluate.sh` | Evaluation entry point |
| `run_evaluator.sh` | Invokes `evaluator.py` |
| `evaluator.py` | Core evaluation logic |
| `resources/` | Baseline code, benchmark, test data |

### config.yaml Example

```yaml
dependencies:
  uv_project: resources    # Optional: uv project in resources/
datasets: []               # Optional: dataset URLs
tag: hpc                   # Category: os, hpc, ai, db, pl, security
runtime:
  docker:
    image: andylizf/triton-tlx:tlx-nv-cu122
    gpu: true
  timeout_seconds: 1800
```

## Evaluation Flow

Inside the Docker container, the execution order is:

```
1. set_up_env.sh         →  Initialize environment
2. Copy solution.py      →  /work/execution_env/solution_env/
3. evaluate.sh           →  Check files, call run_evaluator.sh
4. run_evaluator.sh      →  python3 evaluator.py
5. evaluator.py          →  Load Solution.solve(), run benchmark, print score
```

The final score is extracted from the last numeric line of stdout.

## Solution Interface

Submit a `solution.py` implementing the `Solution` class. The interface varies by problem type:

### Triton Kernel Problems (flash_attn, cross_entropy, gemm_optimization...)

```python
class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns either:
        - {"code": "python_code_string"}
        - {"program_path": "path/to/kernel.py"}
        """
        kernel_code = '''
import triton
import triton.language as tl

@triton.jit
def my_kernel(...):
    ...

def entry_function(...):
    ...
'''
        return {"code": kernel_code}
```

### ML Training Problems (imagenet_pareto...)

```python
class Solution:
    def solve(self, train_loader, val_loader, metadata: dict) -> torch.nn.Module:
        """
        Train and return a model.

        metadata contains: num_classes, input_dim, param_limit,
                          baseline_accuracy, device, etc.
        """
        model = MyModel(...)
        # training loop
        return model
```

### Other Problems

Check each problem's `readme` for the specific `solve()` signature and return type.

## Generating Solutions with LLMs

Use `generate_solutions.py` to generate solutions using LLMs.

```bash
# Generate one solution
python research/scripts/generate_solutions.py --problem flash_attn --model gpt-5 --variants 1

# Preview what would be generated
python research/scripts/generate_solutions.py --dryrun
```

### Two Modes

**Problem mode** (generate new solutions):

```bash
python research/scripts/generate_solutions.py --problem flash_attn --model gpt-5
```

Generates **problems × models × variants** (Cartesian product):
- Problems: `--problem` patterns or `--problems-file` (default: `problems.txt`)
- Models: `--model` list or `--models-file` (default: `models.txt`)
- Variants: `--variants N` (default: from `num_solutions.txt`, currently 5)

Solution naming: `{model_prefix}_{problem}` for variant 0, `{model_prefix}_{problem}_{i}` for variant i.

**Solution mode** (regenerate existing solutions):

```bash
python research/scripts/generate_solutions.py --solution "gpt5_flash*" --force
```

- Matches existing solutions in `solutions/` by pattern
- Model inferred from solution name prefix (e.g., `gpt5_` → `gpt-5`)
- Requires `--force` since solutions already exist
- Still needs `models.txt` or `--model` to map prefix to model name

### Options

| Option | Description |
|--------|-------------|
| `--problem` / `--problems-file` | Problem pattern or file (default: `problems.txt`) |
| `--model` / `--models-file` | Model(s) or file (default: `models.txt`) |
| `--variants` / `--variants-file` | Variant count or file (default: `num_solutions.txt`) |
| `--solution PATTERN` | Regenerate existing solutions by pattern (mutually exclusive with `--problem`) |
| `--force` | Overwrite existing solutions |
| `--dryrun` | Preview without generating |
| `--concurrency N` | Parallel API calls |
| `--timeout SECONDS` | API timeout (default: 600s) |
| `--temperature TEMP` | Sampling temperature (default: 0.7) |
| `--reasoning-model` | Force reasoning mode (o1/o3 models) |

### Output

```
solutions/{model_prefix}_{problem}[_{variant}]/
├── config.yaml
├── prepare_env.sh
├── solve.sh
└── resources/solution.py
```

### API Keys

```bash
export OPENAI_API_KEY=sk-...      # GPT models
export ANTHROPIC_API_KEY=sk-...   # Claude models
export GOOGLE_API_KEY=...         # Gemini models
```
