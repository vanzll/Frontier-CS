from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_heuristic_v1"

    def solve(self, spec_path: str) -> "Solution":
        # Initialize internal state
        self._forced_od = False
        self._cached_work_done = 0.0
        self._last_segments_len = 0
        self._segments_ref = None
        return self

    def _update_progress_cache(self) -> None:
        segments = getattr(self, "task_done_time", None)

        if not segments:
            self._cached_work_done = 0.0
            self._last_segments_len = 0
            self._segments_ref = segments
            return

        # Initialize cache if needed
        if not hasattr(self, "_cached_work_done"):
            self._cached_work_done = 0.0
            self._last_segments_len = 0
            self._segments_ref = None

        # If list object changed or shrank, recompute from scratch
        if segments is not self._segments_ref or len(segments) < self._last_segments_len:
            total = 0.0
            for seg in segments:
                try:
                    if isinstance(seg, (list, tuple)):
                        if len(seg) >= 2:
                            total += float(seg[1]) - float(seg[0])
                        elif len(seg) == 1:
                            total += float(seg[0])
                    else:
                        total += float(seg)
                except Exception:
                    continue
            self._cached_work_done = max(total, 0.0)
            self._last_segments_len = len(segments)
            self._segments_ref = segments
            return

        # Incremental update for newly appended segments
        for seg in segments[self._last_segments_len:]:
            try:
                if isinstance(seg, (list, tuple)):
                    if len(seg) >= 2:
                        self._cached_work_done += float(seg[1]) - float(seg[0])
                    elif len(seg) == 1:
                        self._cached_work_done += float(seg[0])
                else:
                    self._cached_work_done += float(seg)
            except Exception:
                continue

        if self._cached_work_done < 0.0:
            self._cached_work_done = 0.0

        self._last_segments_len = len(segments)
        self._segments_ref = segments

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update cached progress
        self._update_progress_cache()
        work_done = getattr(self, "_cached_work_done", 0.0)

        # Basic attributes with safe defaults
        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = work_done

        remaining_work = max(task_duration - work_done, 0.0)
        if remaining_work <= 0.0:
            # Task already completed
            return ClusterType.NONE

        env = getattr(self, "env", None)
        if env is not None:
            try:
                elapsed = float(env.elapsed_seconds)
            except Exception:
                elapsed = 0.0
        else:
            elapsed = 0.0

        try:
            deadline = float(self.deadline)
        except Exception:
            # No meaningful deadline; be conservative with ON_DEMAND
            self._forced_od = True
            return ClusterType.ON_DEMAND

        time_left = deadline - elapsed
        if time_left <= 0.0:
            # Already at/past deadline; best effort with ON_DEMAND
            self._forced_od = True
            return ClusterType.ON_DEMAND

        slack = time_left - remaining_work
        if slack <= 0.0:
            # Cannot afford any more delay; force on-demand
            self._forced_od = True
            return ClusterType.ON_DEMAND

        # Global slack characteristics (constant over episode)
        total_slack = max(deadline - task_duration, 0.0)
        try:
            oh = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        except Exception:
            oh = 0.0

        # Derive commit and idle thresholds based on total slack and restart overhead
        if total_slack <= 0.0:
            commit_slack = 0.0
            idle_slack = 0.0
        else:
            base_commit = max(5.0 * oh, 0.25 * total_slack)
            commit_slack = min(base_commit, 0.6 * total_slack)
            if commit_slack < 2.0 * oh:
                commit_slack = min(2.0 * oh, total_slack)
            # Ensure commit_slack is at most total_slack
            if commit_slack > total_slack:
                commit_slack = total_slack

            idle_base = min(total_slack * 0.75, commit_slack * 2.0)
            idle_slack = max(idle_base, commit_slack + oh)
            if idle_slack > total_slack:
                idle_slack = total_slack

        # Once forced to on-demand, stay there until completion
        if getattr(self, "_forced_od", False) or slack <= commit_slack:
            self._forced_od = True
            return ClusterType.ON_DEMAND

        # Pre-commit behavior
        if has_spot:
            # If slack is getting close to commit threshold, avoid risking another restart
            if slack <= commit_slack + oh:
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT
        else:
            # No spot available: decide to wait or use on-demand
            if slack > idle_slack:
                # Plenty of slack left: wait for spot to save cost
                return ClusterType.NONE
            else:
                # Slack is limited: keep making progress with on-demand
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)