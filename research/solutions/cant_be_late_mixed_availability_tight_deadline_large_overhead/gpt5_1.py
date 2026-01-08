from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
import argparse


class Solution(Strategy):
    NAME = "cant_be_late_safe_spot_v2"

    def __init__(self, args=None):
        super().__init__(args)
        self.commit_od = False
        self.preemptions = 0
        self.spot_avail_ema = 0.5
        self.ema_alpha = 0.05
        self.consec_nospot = 0
        self._overhead_left = 0.0  # seconds of startup overhead remaining before compute resumes
        self._done_est = 0.0  # estimated compute progress (seconds)
        self._last_decision = ClusterType.NONE

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _sum_task_done_time(self):
        total = 0.0
        try:
            for seg in getattr(self, "task_done_time", []):
                try:
                    if isinstance(seg, (int, float)):
                        if seg > 0:
                            total += float(seg)
                    elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
                        a, b = seg[0], seg[1]
                        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                            dur = float(b) - float(a)
                            if dur > 0:
                                total += dur
                except Exception:
                    continue
        except Exception:
            pass
        return max(0.0, min(total, getattr(self, "task_duration", total)))

    def _compute_overhead_needed_if_switch_to_od(self, last_cluster_type: ClusterType) -> float:
        # If already on OD, remaining overhead is whatever is left to consume; else need full restart_overhead
        if last_cluster_type == ClusterType.ON_DEMAND:
            return max(0.0, self._overhead_left)
        return float(self.restart_overhead)

    def _is_new_launch(self, last_cluster_type: ClusterType, next_cluster_type: ClusterType, has_spot: bool) -> bool:
        # Determine if starting a new instance next step (incurs restart overhead)
        if next_cluster_type == ClusterType.NONE:
            return False
        if last_cluster_type == ClusterType.NONE:
            return True
        if next_cluster_type == ClusterType.ON_DEMAND:
            if last_cluster_type == ClusterType.ON_DEMAND:
                return False
            # last was SPOT; if we switch to OD, it's a launch
            return True
        if next_cluster_type == ClusterType.SPOT:
            if not has_spot:
                return False  # shouldn't happen as caller guards, but safe
            # Continue SPOT only if last was SPOT and spot still available
            if last_cluster_type == ClusterType.SPOT:
                return False
            # From OD->SPOT or NONE->SPOT
            return True
        return True

    def _update_progress_from_last_step(self, last_cluster_type: ClusterType):
        # Update estimated compute progress for the last executed step
        try:
            gap = float(self.env.gap_seconds)
        except Exception:
            gap = 60.0
        if last_cluster_type in (ClusterType.SPOT, ClusterType.ON_DEMAND):
            overhead_consumed = min(self._overhead_left, gap)
            effective_compute = max(gap - overhead_consumed, 0.0)
            self._overhead_left = max(self._overhead_left - gap, 0.0)
            self._done_est = min(self.task_duration, self._done_est + effective_compute)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update progress estimate from the previous step
        self._update_progress_from_last_step(last_cluster_type)

        # Update availability stats and preemption count
        self.spot_avail_ema = (1.0 - self.ema_alpha) * self.spot_avail_ema + self.ema_alpha * (1.0 if has_spot else 0.0)
        if has_spot:
            self.consec_nospot = 0
        else:
            self.consec_nospot += 1

        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.preemptions += 1

        # Compute remaining work/time and slack
        try:
            gap = float(self.env.gap_seconds)
        except Exception:
            gap = 60.0
        rem_time = max(0.0, float(self.deadline) - float(self.env.elapsed_seconds))
        # Prefer our continuous estimate (accounts for current running segment)
        done_est = max(0.0, min(self._done_est, float(self.task_duration)))
        # As a guard, if task_done_time reports more, use that (avoid underestimating progress)
        done_reported = self._sum_task_done_time()
        done = max(done_est, done_reported)
        rem_work = max(0.0, float(self.task_duration) - done)
        slack_left = rem_time - rem_work

        # Determine overhead needed if we switch/commit to OD now
        od_overhead_needed = self._compute_overhead_needed_if_switch_to_od(last_cluster_type)

        # Safety margins
        # Fudge accounts for small estimation errors and step granularity
        fudge = 2.0 * gap + 0.5 * float(self.restart_overhead)

        # Commit conditions: once set, we stick to OD forever
        if not self.commit_od:
            # Primary safety condition: ensure we can complete on OD including overhead
            if slack_left <= od_overhead_needed + fudge:
                self.commit_od = True
            else:
                # Additional conservative conditions based on poor spot availability and high preemptions
                if (self.spot_avail_ema < 0.25 and slack_left <= (od_overhead_needed + 3 * gap + 0.5 * self.restart_overhead)) \
                   or (self.preemptions >= 4 and slack_left <= (2 * gap + self.restart_overhead)):
                    self.commit_od = True

        # Decision logic
        if self.commit_od:
            decision = ClusterType.ON_DEMAND
        else:
            if has_spot:
                decision = ClusterType.SPOT
            else:
                # Decide between waiting (NONE) or switching to OD
                # Allow waiting if we have sufficient slack beyond OD overhead + fudge.
                # Be less patient when spot reliability is low or after many consecutive misses.
                reliability_factor = 1
                if self.spot_avail_ema >= 0.70:
                    reliability_factor = 3
                elif self.spot_avail_ema >= 0.45:
                    reliability_factor = 2
                else:
                    reliability_factor = 1

                dynamic_wait_budget = gap * (reliability_factor + min(self.consec_nospot, 6))
                wait_threshold = od_overhead_needed + fudge + dynamic_wait_budget

                if slack_left > wait_threshold:
                    decision = ClusterType.NONE
                else:
                    decision = ClusterType.ON_DEMAND

        # Update overhead state for the next step based on the decision (launch detection)
        if decision in (ClusterType.SPOT, ClusterType.ON_DEMAND):
            if self._is_new_launch(last_cluster_type, decision, has_spot):
                # Replace any remaining overhead with a new one (doesn't stack)
                self._overhead_left = float(self.restart_overhead)
            # else continue consuming any remaining overhead (if any) on the next steps
        # If decision == NONE, we neither consume nor start overhead here.

        self._last_decision = decision
        return decision

    @classmethod
    def _from_args(cls, parser):
        if not isinstance(parser, argparse.ArgumentParser):
            parser = argparse.ArgumentParser()
        args, _ = parser.parse_known_args()
        return cls(args)