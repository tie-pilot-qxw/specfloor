"""Read-only timing export for this benchmark's server processes only."""
import collections
import importlib.abc
import importlib.machinery
import sys


def patch(module):
    if module.__name__.endswith("req_time_stats"):
        cls = module.SchedulerReqTimeStats
        original_state = cls.__getstate__
        original_meta = cls.convert_to_output_meta_info

        def state(self):
            result = original_state(self)
            if result:
                result["completion_time"] = self.completion_time
            return result

        def meta(self):
            result = original_meta(self)
            if self.completion_time > 0 and self.prefill_finished_time > 0:
                result["sweep_scheduler_decode_elapsed_s"] = (
                    self.completion_time - self.prefill_finished_time)
                result["sweep_scheduler_prefill_elapsed_s"] = (
                    self.prefill_finished_time - self.forward_entry_time)
            return result

        cls.__getstate__ = state
        cls.convert_to_output_meta_info = meta
    elif module.__name__.endswith("metrics_reporter"):
        cls = module.SchedulerMetricsReporter
        original = cls.report_decode_stats

        def report(self, can_run_cuda_graph, running_batch=None, num_correct_drafts=0):
            batch = running_batch or self.scheduler.running_batch
            if not hasattr(self, "sweep_decode_counts"):
                self.sweep_decode_counts = collections.Counter()
            self.sweep_decode_counts[str(len(batch.reqs))] += 1
            self.sweep_decode_counts["graph" if can_run_cuda_graph else "eager"] += 1
            return original(self, can_run_cuda_graph, running_batch, num_correct_drafts)

        cls.report_decode_stats = report
    elif module.__name__.endswith("scheduler"):
        cls = module.Scheduler
        original = cls.get_internal_state

        def internal(self, req):
            result = original(self, req)
            result.internal_state["sweep_decode_counts"] = dict(
                getattr(self.metrics_reporter, "sweep_decode_counts", {}))
            return result

        cls.get_internal_state = internal


TARGETS = {
    "sglang.srt.observability.req_time_stats",
    "sglang.srt.managers.scheduler_components.metrics_reporter",
    "sglang.srt.managers.scheduler",
}


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        patch(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in TARGETS:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None:
                spec.loader = Loader(spec.loader)
            return spec


sys.meta_path.insert(0, Finder())
