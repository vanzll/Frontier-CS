# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Frontier-CS is an evaluation framework for challenging computer science problems, including research problems (GPU kernels, ML systems, security) and algorithmic problems (competitive programming style). Solutions are evaluated in Docker containers or via SkyPilot cloud VMs.

### Public vs Internal Repos

| Repo | Contents | Purpose |
|------|----------|---------|
| `Frontier-CS` (public) | Problems with 1 test case, solutions, evaluation tools | Open benchmark, prevent test case leakage |
| `Frontier-CS-internal` | Full test cases (10-100+ per problem) | Accurate evaluation, not leaked to LLMs |
| `Frontier-CS-Result` | Evaluation state and results | Incremental runs, leaderboard data |

Why separate? Public test cases may leak into LLM training data. Internal repo keeps full test suites private for accurate evaluation.

**Important**: Only the `problems/` directories from internal repo are used during evaluation. The evaluation code (`src/`) always runs from the public repo. This means changes to `src/` in internal repo have no effect and can be safely discarded.

**Note**: `Frontier-CS-Result` stores evaluation state with `problem_hash` computed from internal repo. Don't run `frontier batch` directly on public repo—use `run_eval.sh` which handles the three-repo setup correctly.

### Scoring: Bounded vs Unbounded

| Type | Range | Use Case |
|------|-------|----------|
| Bounded (`score`) | 0-100 | Human-readable, leaderboard display |
| Unbounded (`score_unbounded`) | 0-∞ | ELO rating calculation |

**Why unbounded?** When multiple models score 100 (bounded), we can't differentiate them. Unbounded scores (e.g., 150 vs 200) enable meaningful ELO ratings for ranking models that all "max out" the bounded scale.

```bash
frontier eval flash_attn solution.py              # Returns bounded (0-100)
frontier eval --unbounded flash_attn solution.py  # Returns unbounded
frontier batch                                    # Stores both in results
```

## Commands

```bash
# Installation
uv sync  # or: pip install -e .

# Evaluate single solution
frontier eval flash_attn solution.py              # research (Docker)
frontier eval flash_attn solution.py --skypilot   # research (cloud)
frontier eval --algorithmic 1 solution.cpp        # algorithmic (Docker)

# Batch evaluation
frontier batch                                    # scan solutions/ directory
frontier batch --workers 10                       # parallel Docker
frontier batch --skypilot --workers 20 --clusters 4  # cloud with cluster pool
frontier batch --status                           # check progress
frontier batch --retry-failed                     # retry failures (incl. score=0)
frontier batch --report                           # aggregated results

# Generate solutions with LLMs
python research/scripts/generate_solutions.py --problem flash_attn --model gpt-5
python research/scripts/generate_solutions.py --only-failed  # retry failed generations

# Check solution coverage
python research/scripts/check_solutions.py
```

## Architecture

### Two Problem Tracks

| Aspect    | Research                        | Algorithmic               |
| --------- | ------------------------------- | ------------------------- |
| Location  | `research/problems/`            | `algorithmic/problems/`   |
| Solution  | Python (`Solution.solve()`)     | C++17                     |
| Evaluator | `evaluator.py` in each problem  | go-judge server           |
| Docker    | Per-problem image (GPU support) | Shared go-judge container |
| SkyPilot  | Dedicated VM per problem        | Single go-judge VM        |

### Backends: Docker vs SkyPilot

| Backend         | Research                                  | Algorithmic                            |
| --------------- | ----------------------------------------- | -------------------------------------- |
| Docker (local)  | `evaluate.sh` in problem-specific image   | HTTP to go-judge (auto `docker compose up`) |
| SkyPilot (cloud)| Cloud VM with GPU per evaluation          | Single go-judge VM, shared via HTTP    |
| `--clusters N`  | N reusable VMs for load-balancing         | Not applicable (single judge)          |

### Source Structure

```
src/frontier_cs/
├── cli.py                 # CLI entry point (frontier command)
├── evaluator.py           # FrontierCSEvaluator - unified API
├── config.py              # ProblemConfig, RuntimeConfig, DockerConfig
├── runner/
│   ├── base.py            # Runner ABC, EvaluationResult, EvaluationStatus
│   ├── docker.py          # DockerRunner (research)
│   ├── skypilot.py        # SkyPilotRunner (research)
│   ├── algorithmic.py     # AlgorithmicRunner (go-judge HTTP)
│   └── algorithmic_skypilot.py
├── batch/
│   ├── evaluator.py       # BatchEvaluator (parallel, resume, export)
│   ├── state.py           # EvaluationState, PairResult, hash functions
│   └── pair.py            # Pair dataclass, expand_pairs()
├── storage/
│   └── bucket.py          # BucketStorage (S3/GCS sync)
└── gen/
    ├── llm.py             # Multi-provider LLM client
    ├── llm_interface.py
    └── solution_format.py # Filename parsing, FAILED_EXTENSION
```

### Problem Structure

**Research problems** (`research/problems/{problem_name}/`):
```
flash_attn/
├── config.yaml          # Docker image, GPU, timeout, uv_project
├── readme               # Problem statement (markdown-like)
├── evaluator.py         # Scoring logic, imports solution
├── evaluate.sh          # Entry point inside container
├── download_datasets.sh # Optional: download large datasets
├── run_evaluator.sh     # Optional: helper script
└── resources/           # Problem-specific dependencies
    ├── pyproject.toml   # uv project for deps (if uv_project: resources)
    ├── baseline.py      # Reference implementation
    ├── benchmark.py     # Performance testing
    └── submission_spec.json  # API spec for solutions
```

**Excluded directories**: `resources/`, `common/`, `__pycache__/` are not treated as problems. A valid research problem must have `evaluator.py` or `evaluate.py`.

**config.yaml fields**:
```yaml
dependencies:
  uv_project: resources    # Dir with pyproject.toml for deps
tag: hpc                   # Problem category
runtime:
  environment: "..."       # Human-readable env description
  docker:
    image: andylizf/triton-tlx:tlx-nv-cu122
    gpu: true              # Requires GPU passthrough
    dind: false            # Docker-in-Docker support
```

**Algorithmic problems** (`algorithmic/problems/{problem_id}/`):
```
1/                       # Problem ID is numeric
├── config.yaml          # Time/memory limits, checker, subtasks
├── statement.txt        # Problem statement (plain text)
├── chk.cc               # Custom checker (C++) for scoring
└── testdata/            # Test cases
    ├── 1.in             # Input for test case 1
    ├── 1.ans            # Expected output for test case 1
    ├── 2.in
    ├── 2.ans
    └── ...
```

**config.yaml fields**:
```yaml
type: default
time: 1s                 # Time limit per test case
memory: 1024m            # Memory limit
checker: chk.cc          # Custom checker file
subtasks:
  - score: 100           # Points for this subtask
    n_cases: 3           # Number of test cases
```

### Solution Structure

**Research solutions** (`research/solutions/{problem_name}/`):
```
flash_attn/
├── gpt5.py              # Base solution from model
├── gpt5_1.py            # Variant 1 (index starts at 1)
├── gpt5_2.py            # Variant 2
├── gemini2.5pro.py
├── gemini2.5pro_1.py
└── deepseekreasoner.FAILED  # Generation failure marker (JSON)
```

Solution file must define a `Solution` class:
```python
class Solution:
    def solve(self, spec_path: str = None) -> dict:
        # Return {"code": "..."} or {"program_path": "..."}
        pass
```

**Algorithmic solutions** (`algorithmic/solutions/{problem_id}/`):
```
1/
├── gpt5.cpp             # Base solution from model
├── gpt5_1.cpp           # Variant 1
├── gpt5_2.cpp           # Variant 2
├── gemini2.5pro.cpp
└── deepseekreasoner.FAILED  # Generation failure marker
```

Standard C++17 solution reading from stdin, writing to stdout.

**Naming convention**:
```
{model}.{ext}            # Base solution: gpt5.py, gpt5.cpp
{model}_{i}.{ext}        # Variant i (1-indexed): gpt5_1.py, gpt5_2.cpp
{model}.FAILED           # Generation failure marker (JSON content)
```

Model names use lowercase without hyphens: `gpt5`, `gemini2.5pro`, `deepseekreasoner`, `grok4fastreasoning`.

**Special directories**:
- `_deleted/`: Solutions moved here are excluded from evaluation

## Workflow: Generate → Evaluate → Report

### End-to-End Flow

```
1. Generate solutions:
   python research/scripts/generate_solutions.py --problem flash_attn --model gpt-5
   → Creates: research/solutions/flash_attn/gpt5.py
   → On failure: research/solutions/flash_attn/gpt5.FAILED

2. Check coverage:
   python research/scripts/check_solutions.py
   → Compares expected (models × problems × variants) vs actual

3. Batch evaluate:
   frontier batch --solutions-dir research/solutions
   → Scans solutions/, runs evaluations, saves to results/batch/

4. Check results:
   frontier batch --report
   → Aggregates by model and problem
```

### Solution Filename Convention

```
{solutions_dir}/{problem}/{model}.py           # e.g., solutions/flash_attn/gpt5.py
{solutions_dir}/{problem}/{model}_{i}.py       # e.g., solutions/flash_attn/gpt5_1.py (variant)
{solutions_dir}/{problem}/{model}.FAILED       # Generation failure marker
{solutions_dir}/_deleted/...                   # Excluded from evaluation
```

Pair ID format: `{problem}/{model}.py:{problem}` (e.g., `flash_attn/gpt5.py:flash_attn`)

### check_solutions.py

Coverage report comparing expected vs actual solutions:

1. Auto-discovers problems from `research/problems/` (leaf dirs with `readme`)
2. Reads models from `models.txt`, variants from `indices.txt`
3. Computes expected: `models × problems × variants`
4. Scans `research/solutions/{problem}/{model}.py` for actual files
5. Reports:
   - Coverage bar (generated / expected)
   - Missing by model
   - Failed generations (`.FAILED` files) with error messages
   - Empty file warnings

## Architectural Decisions

### Runner Hierarchy (Strategy Pattern)

```
Runner (ABC)
├── ResearchRunner (adds base_dir, problems_dir)
│   ├── DockerRunner (local container)
│   └── SkyPilotRunner (cloud VM)
└── AlgorithmicRunner (HTTP to judge server)
    └── AlgorithmicSkyPilotRunner (cloud judge)
```

**Why two-level hierarchy?**
- `Runner`: Generic interface with `evaluate()` and `evaluate_file()` methods
- `ResearchRunner`: Adds directory auto-detection (`_find_base_dir()`) and problem discovery
- Algorithmic runners extend Runner directly—they use HTTP, not filesystem scanning

**Why two evaluation methods?**
- `evaluate(problem_id, solution_code)`: For code strings (generated in memory)
- `evaluate_file(problem_id, solution_path)`: For existing files (batch mode)
- Both return unified `EvaluationResult` dataclass

### DockerRunner Design

**Workspace isolation**: Each evaluation creates temp directory with:
1. Problem files (evaluator.py, config.yaml, resources/)
2. Solution file (copied to workspace)
3. Common directory if present (parent-level shared code)

**Score parsing strategy**:
- Looks for last numeric line in stdout
- Skips log lines containing `[`, `INFO`, `ERROR`
- Supports single number (`85.5`) or two numbers (`85.5 120.3` for bounded + unbounded)

**GPU handling**: Auto-detects via cached `nvidia-smi` check; returns `SKIPPED` status if problem requires GPU but unavailable.

### SkyPilotRunner Design

**Cluster pooling** (key for batch efficiency):
```python
# Create N reusable clusters at batch start
cluster_pool = queue.Queue()
for i in range(num_clusters):
    cluster_pool.put(runner.create_cluster())

# Workers acquire from pool, execute, return
cluster = cluster_pool.get(timeout=300)
result = runner.exec_on_cluster(cluster, ...)
cluster_pool.put(cluster)  # Return to pool
```

Why not `sky.exec`? File mounts only sync with `sky.launch`.

**Dual result storage modes**:
- **Bucket mode** (`--bucket-url`): Results written directly to S3/GCS during job; avoids scp round-trip
- **SCP mode** (legacy): Fetches results from remote after job completes

### BatchEvaluator Design

**Worker pool pattern**: ThreadPoolExecutor with cluster pool for SkyPilot load-balancing.

**Hash-based cache invalidation**:
```python
solution_hash = sha256(solution_file)[:16]
problem_hash = sha256(problem_directory)[:16]  # Includes evaluator.py, config.yaml
```
If either hash changes → pair re-evaluated. Prevents stale results when evaluator is fixed.

**Atomic state writes**: Uses `tempfile` + `os.replace()` to prevent corruption from crashes.

**Pair interleaving** (load-balancing):
```
# Without: A1, A2, A3, B1, B2, B3 (one slow problem blocks others)
# With:    A1, B1, C1, A2, B2, C2 (round-robin across problems)
```

**Retry logic** (`--retry-failed` includes score=0):
Cannot distinguish "legitimate 0 score" from "evaluator bug printing 0 before exit(1)". Cost of re-running is low; benefit of catching missed scores is high.

### AlgorithmicRunner Design

**Judge server lifecycle**: Double-check locking pattern for thread-safe auto-start:
```python
if not self._judge_started:        # Fast path (no lock)
    with self._startup_lock:       # Slow path
        if not self._judge_started:  # Verify again
            self._start_judge()
```

**Submission flow**: POST /submit → poll GET /result/{sid} → parse score from JSON response.

## Evaluation Internals

### Research Problem Flow (DockerRunner)

```
1. Load config.yaml (Docker image, GPU, timeout, uv_project)
2. Setup workspace (copy problem files + solution.py)
3. Build docker run command:
   - Mount workspace, datasets
   - GPU passthrough if needed
   - DinD socket if dind: true
4. Inside container:
   - Install uv, deps from uv_project
   - Run set_up_env.sh (dataset prep)
   - Run evaluate.sh → evaluator.py
5. Parse score from last numeric line of stdout
```

**Evaluator output format**: Last line should be either:
- Single number: `85.5` → both `score` and `score_unbounded` set to this value
- Two numbers: `85.5 120.3` → `score=85.5`, `score_unbounded=120.3`

### Algorithmic Problem Flow (AlgorithmicRunner)

```
1. Ensure go-judge running (auto docker compose up)
2. POST /submit {pid, code} → get submission ID
3. Poll GET /result/{sid} until status=done or timeout
4. Parse score and scoreUnbounded from response
```

### Batch Evaluation Flow (BatchEvaluator)

```
1. Sync from bucket (if --bucket-url)
2. Compute solution/problem hashes for cache invalidation
3. Filter pending pairs (skip completed with matching hashes)
4. Create worker pool (ThreadPoolExecutor)
5. For SkyPilot research: create cluster pool
6. Workers pull pairs from queue:
   - Mark running in state
   - Run evaluation (Docker or SkyPilot)
   - Record result with hashes
   - Save state (atomic write)
7. Export results (CSV, aggregated reports)
```

**Hash invalidation**: When problem files change (evaluator fixes), `problem_hash` changes and results are re-evaluated. **New solutions**: Only pairs existing at batch start are evaluated; re-run batch to pick up new solutions.

## Data Structures & State

### Key Data Structures

**EvaluationResult** (runner/base.py) - Return type from runners:
```python
@dataclass
class EvaluationResult:
    problem_id: str
    score: Optional[float]           # Bounded (0-100)
    score_unbounded: Optional[float] # Raw score
    status: EvaluationStatus         # SUCCESS, ERROR, TIMEOUT, SKIPPED
    message: Optional[str]
    logs: Optional[str]
    duration_seconds: Optional[float]
```

**EvaluationState** (batch/state.py) - Persistent batch state:
```python
@dataclass
class EvaluationState:
    results: Dict[str, PairResult]   # "solution:problem" -> result
    started_at: Optional[str]
    updated_at: Optional[str]
    total_pairs: int
    backend: str                     # "docker" or "skypilot"
```

**PairResult** (batch/state.py) - Single evaluation result:
```python
@dataclass
class PairResult:
    pair_id: str                     # "flash_attn/gpt5.py:flash_attn"
    score: Optional[float]
    score_unbounded: Optional[float]
    status: str                      # pending, running, success, error, timeout, skipped
    solution_hash: Optional[str]     # For cache invalidation
    problem_hash: Optional[str]
```

### State Files

Results are stored in `Frontier-CS-Result` repo (or local `results/` for dev):

| Track       | Directory              | State File                |
| ----------- | ---------------------- | ------------------------- |
| Research    | `research/batch/`      | `.state.research.json`    |
| Algorithmic | `algorithmic/batch/`   | `.state.algorithmic.json` |

Each directory contains:

| File                    | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| `.state.{track}.json`   | Persistent state (results dict, hashes, timestamps)  |
| `results.csv`           | All results (solution, problem, score, status, etc.) |
| `by_model.csv`          | Aggregated by model (avg_score, successful, failed)  |
| `by_problem.csv`        | Aggregated by problem                                |
| `failed.txt`            | Failed pairs for retry (`solution:problem` per line) |
| `pending.txt`           | Incomplete pairs                                     |

**Important**: State files (from `Frontier-CS-Result` repo) contain `problem_hash` computed from **internal repo** problems. Running `frontier batch` locally with public repo will show hash mismatches and trigger re-evaluation—this is expected. For official evaluation, always use `run_eval.sh` which uses internal repo problems.

**Orphaned results**: After problem restructuring, old results for deleted/renamed problems remain in state but are excluded from aggregation. **Running status**: Interrupted evaluations leave `status=running` results without scores; these are re-evaluated on next batch run.

## Error Handling & Retry

### Generation Failure (.FAILED files)

When LLM generation fails, creates `.FAILED` marker instead of `.py`:

```
solutions/flash_attn/gpt5.FAILED   # JSON: {error, model, timestamp}
```

- `--only-failed`: Only regenerate solutions with `.FAILED` markers
- `--mark-failed`: Create `.FAILED` files without generating (for tracking)
- On success: Deletes `.FAILED` file if existed

Runners detect `.FAILED` files → return `ERROR` with `"Generation failed: {error}"`.

### Retry Logic (--retry-failed)

`frontier batch --retry-failed` retries:

1. **Explicit failures**: status = `error` or `timeout`
2. **Zero-score successes**: status = `success` AND score = 0

Why retry score=0? Cannot distinguish between:
- Legitimate 0 score
- Evaluator bug printing "0" before exit(1)
- Infrastructure issues

See `get_failed_pairs()` in state.py - this is intentional.

## CI/CD

### run_eval.sh

Main orchestration script for batch evaluation (used by CI and locally):

```bash
./scripts/run_eval.sh --track research              # Auto-clones internal + results repos
./scripts/run_eval.sh --track algorithmic --workers 10
./scripts/run_eval.sh --track research --skypilot --clusters 4
./scripts/run_eval.sh --check-overlap               # Verify internal ⊇ public
```

**What it does**:

1. Auto-clone repos (or use `--internal-dir`, `--results-repo`):
   - `Frontier-CS-internal`: Full test cases (public has only 1 test per problem)
   - `Frontier-CS-Result`: Incremental state for resume
2. Check internal ⊇ public (all public problems exist in internal)
3. Run `frontier batch` with:
   - Solutions from public repo
   - Problems from internal repo (more test cases)
   - Results saved to results repo
4. Push results to remote (interactive prompt or `--push`/`--no-push`)
5. Cleanup SkyPilot clusters

**Key flags**:
- `--track research|algorithmic`: Required
- `--workers N`: Parallel workers (default: 4)
- `--clusters N`: SkyPilot clusters (default: 4)
- `--skypilot`: Use cloud evaluation
- `--check-overlap`: Only verify internal ⊇ public, then exit
- `--dry-run`: Print command without executing

### weekly-eval.yml

Weekly GitHub Actions workflow that runs `run_eval.sh`:

1. Checkout public + internal repos
2. Setup GCP credentials + SkyPilot
3. Run `scripts/run_eval.sh` for research and algorithmic tracks
4. Push results to Frontier-CS-Result repo
5. Cleanup SkyPilot clusters

## Configuration

### API Keys for LLM Generation

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export OPENROUTER_API_KEY=...
```
