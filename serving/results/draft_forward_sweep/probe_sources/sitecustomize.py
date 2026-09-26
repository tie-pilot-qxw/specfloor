"""Install the probe in scheduler subprocesses as well as the launcher."""
import importlib.abc
import importlib.machinery
import sys


class ProbeLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        import draft_forward_probe_server  # patches the now-initialized class


class ProbeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang.srt.speculative.dspark_components.dspark_draft":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = ProbeLoader(spec.loader)
        return spec


sys.meta_path.insert(0, ProbeFinder())
