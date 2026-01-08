from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the parameters for the adaptive scheduling strategy.
        This method is called once before the simulation starts.
        """
        # We aim to finish with a safety buffer to absorb unexpected delays.
        # A 1-hour buffer is chosen out of the 4-hour total slack.
        self.SAFETY_BUFFER_S = 3600.0

        # The effective deadline is our internal target for job completion.
        self.effective_deadline = self.deadline - self.SAFETY_BUFFER_S

        # This threshold determines when we are "too far behind". If our work
        # deficit exceeds this value, we switch to a more aggressive strategy
        # (on-demand) to catch up. It's tuned to be a multiple of the
        # restart overhead to tolerate several spot instance preemptions.
        self.CATCH_UP_THRESHOLD_S = 5.0 * self.restart_overhead

        # This factor is used to decide when we are "comfortably ahead". If
        # we are ahead of schedule by more than this factor times the time
        # step, we can afford to pause (use NONE) to save costs when spot
        # instances are unavailable.
        self.RELAX_THRESHOLD_FACTOR = 2.0

        # A flag to track job completion to avoid unnecessary calculations.
        self.job_finished = False

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides which cluster type to use for the next time step.
        """
        if self.job_finished:
            return ClusterType.NONE

        work_done = sum(end - start for start, end in self.task_done_time)

        if work_done >= self.task_duration:
            self.job_finished = True
            return ClusterType.NONE

        remaining_work = self.task_duration - work_done
        remaining_time_to_deadline = self.deadline - self.env.elapsed_seconds

        # 1. Panic Mode (Safety Net)
        # This is a critical check to guarantee we meet the hard deadline.
        # It calculates the minimum time required to finish the job using only
        # reliable on-demand instances. If this time exceeds the remaining
        # time to the deadline, we have no choice but to use on-demand.

        # A restart is needed if the previous instance was NONE or a preempted SPOT.
        needs_restart = (last_cluster_type == ClusterType.NONE or
                         (last_cluster_type == ClusterType.SPOT and not has_spot))

        on_demand_required_time = remaining_work
        if needs_restart:
            on_demand_required_time += self.restart_overhead

        if on_demand_required_time >= remaining_time_to_deadline:
            return ClusterType.ON_DEMAND

        # 2. Deficit-Based Adaptive Control
        # This is the core of the strategy. We define a target progress schedule
        # and adjust our resource choice based on whether we are ahead of,
        # on-track with, or behind this schedule.

        # The target schedule is a linear ramp to finish the job by our effective_deadline.
        if self.effective_deadline > 0:
            progress_ratio = min(1.0, self.env.elapsed_seconds / self.effective_deadline)
            target_work_done = progress_ratio * self.task_duration
        else:
            target_work_done = self.task_duration

        # The work deficit measures our progress against the target schedule.
        # A positive deficit means we are behind schedule.
        # A negative deficit means we are ahead of schedule.
        work_deficit = target_work_done - work_done

        # 3. Decision Logic
        # Based on the work deficit, we choose the next cluster type.

        # Case A: We are significantly behind schedule.
        # We must use the reliable on-demand instance to catch up.
        if work_deficit > self.CATCH_UP_THRESHOLD_S:
            return ClusterType.ON_DEMAND

        # Case B: We are on-track or ahead of schedule.
        # We prioritize cost savings by using spot instances if available.
        if has_spot:
            return ClusterType.SPOT

        # Case C: Spot is unavailable, and we are on-track or ahead.
        # We decide between ON_DEMAND (to keep progressing) and NONE (to save cost)
        # based on how large our lead is.

        # We can afford to wait (NONE) only if we are comfortably ahead.
        relax_threshold = -self.RELAX_THRESHOLD_FACTOR * self.env.gap_seconds

        if work_deficit < relax_threshold:
            # We have a sufficient lead; pause and wait for a spot instance.
            return ClusterType.NONE
        else:
            # Our lead is small; use on-demand to avoid falling behind.
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)