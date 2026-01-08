import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "my_strategy"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Cache scalar values (seconds) for efficiency/robustness.
        def _as_scalar(x):
            if isinstance(x, (list, tuple)):
                if not x:
                    return 0.0
                x = x[0]
            return float(x)

        self._task_duration_value = _as_scalar(getattr(self, "task_duration", float(config["duration"]) * 3600.0))
        self._restart_overhead_value = _as_scalar(
            getattr(self, "restart_overhead", float(config["overhead"]) * 3600.0)
        )
        self._deadline_value = _as_scalar(getattr(self, "deadline", float(config["deadline"]) * 3600.0))

        # Per-run state initialization flags.
        self._last_elapsed = None
        self._progress_done = 0.0
        self._last_task_segment_count = 0
        self.safe_mode = False

        return self

    def _init_run_state(self):
        # Called at the beginning of each new environment run.
        self.safe_mode = False
        self._progress_done = 0.0
        # Initialize from current task_done_time if any (should be empty at start, but be robust).
        td = getattr(self, "task_done_time", None)
        if td:
            total = 0.0
            for v in td:
                total += v
            self._progress_done = total
            self._last_task_segment_count = len(td)
        else:
            self._last_task_segment_count = 0

    def _update_progress(self):
        # Incrementally update accumulated progress to avoid O(n) summation each step.
        td = self.task_done_time
        current_len = len(td)
        last_len = self._last_task_segment_count
        if current_len > last_len:
            inc = 0.0
            for i in range(last_len, current_len):
                inc += td[i]
            self._progress_done += inc
            self._last_task_segment_count = current_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env

        # Detect new run by elapsed time reset.
        if self._last_elapsed is None or env.elapsed_seconds < self._last_elapsed:
            self._init_run_state()
        self._last_elapsed = env.elapsed_seconds

        # Update progress.
        if hasattr(self, "task_done_time"):
            self._update_progress()

        task_total = self._task_duration_value
        remaining_work = task_total - self._progress_done
        if remaining_work <= 0.0:
            # Task completed; no need to run more.
            return ClusterType.NONE

        time_left = self._deadline_value - env.elapsed_seconds
        gap = getattr(env, "gap_seconds", 0.0)
        overhead = self._restart_overhead_value

        if time_left <= 0.0:
            # Already at/past deadline but work remains; just use on-demand.
            return ClusterType.ON_DEMAND

        # Decide whether to enter "safe mode" (pure on-demand).
        # Risking another SPOT or idle step is safe only if:
        #   time_left > remaining_work + gap + 2 * overhead
        if not self.safe_mode:
            if time_left <= remaining_work + gap + 2.0 * overhead:
                self.safe_mode = True

        if self.safe_mode:
            return ClusterType.ON_DEMAND

        # Risk-taking mode: prefer Spot when available, otherwise wait.
        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE