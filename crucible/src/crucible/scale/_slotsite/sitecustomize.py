"""Interpreter hook of the resource governor (crucible.scale.slots): GPU memory cap and GPU usage reporting inside
every Python process of a governed GPU job.

The supervisor prepends this directory to PYTHONPATH of jobs that were granted GPU memory. At start-up of each Python
process of the job this file arranges that, when the process initialises CUDA through torch:

* torch's caching allocator is capped at the job's grant (`torch.cuda.set_per_process_memory_fraction`), so an
  allocation beyond it raises an out-of-memory error inside the job instead of taking memory from other tenants;
* a daemon thread writes the process's allocator usage (reserved, peak reserved, allocated) and the time CUDA was
  initialised to `<job dir>/gpu/<pid>.json` twice a second. The supervisor sums these over the job's live
  processes, kills the job when the sum exceeds the grant (a process that lifted its own cap) and accounts GPU time
  as the seconds each process holds a CUDA context.

Per-process device memory cannot be read from outside this container (nvidia-smi reports host pids), which is why the
measurement is taken here. The hook imports nothing heavy, never imports torch itself and does nothing in processes
that do not use CUDA. Any sitecustomize further down sys.path still runs.
"""
import os
import sys


def _chain_next_sitecustomize():
    here = os.path.dirname(os.path.abspath(__file__))
    import importlib.machinery
    import importlib.util
    path = [p for p in sys.path if os.path.abspath(p or ".") != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", path)
    if spec is not None and spec.origin and os.path.dirname(os.path.abspath(spec.origin)) != here:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def _install():
    jobdir = os.environ.get("CRUCIBLE_SLOT_JOBDIR")
    try:
        grant_mb = int(os.environ.get("CRUCIBLE_SLOT_GPU_MB", "0"))
    except ValueError:
        grant_mb = 0
    if not jobdir or grant_mb <= 0:
        return
    import importlib.abc
    import importlib.util
    import json
    import threading
    import time

    state = {"done": False, "t0": None, "cap_mb": None}
    MB = 1 << 20

    def report(torch, final=False):
        try:
            rec = {"pid": os.getpid(), "t0": state["t0"], "t": time.time(), "cap_mb": state["cap_mb"],
                   "reserved_mb": torch.cuda.memory_reserved() / MB,
                   "max_reserved_mb": torch.cuda.max_memory_reserved() / MB,
                   "allocated_mb": torch.cuda.memory_allocated() / MB, "exited": bool(final)}
            chain = [x for x in os.environ.get("CRUCIBLE_SLOT_JOBDIR_CHAIN", jobdir).split(os.pathsep) if x]
            for jd in chain:                        # this job and every enclosing governed job
                if not os.path.isdir(jd):
                    continue
                d = os.path.join(jd, "gpu")
                os.makedirs(d, exist_ok=True)
                p = os.path.join(d, f"{os.getpid()}.json")
                with open(p + ".tmp", "w") as fh:
                    json.dump(rec, fh)
                os.replace(p + ".tmp", p)
        except Exception:
            pass

    def apply(torch):
        if state["done"]:
            return
        state["done"] = True
        state["t0"] = time.time()
        try:
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / MB
            frac = max(0.0, min(1.0, grant_mb / total))
            torch.cuda.set_per_process_memory_fraction(float(frac))
            state["cap_mb"] = frac * total
        except Exception:
            state["cap_mb"] = None
        report(torch)

        def loop():
            while True:
                time.sleep(0.5)
                report(torch)
        threading.Thread(target=loop, name="crucible-slot-gpu", daemon=True).start()
        import atexit
        atexit.register(lambda: report(torch, final=True))

    def patch(torch):
        cuda = getattr(torch, "cuda", None)
        if cuda is None or getattr(cuda, "_crucible_slot_patched", False):
            return
        orig = cuda._lazy_init

        def _lazy_init():
            orig()
            if not state["done"]:
                apply(torch)
        cuda._lazy_init = _lazy_init
        cuda._crucible_slot_patched = True

        def watch():            # CUDA initialised through a path that bypasses the module attribute
            while not state["done"]:
                time.sleep(0.25)
                try:
                    if cuda.is_initialized():
                        apply(torch)
                except Exception:
                    return
        threading.Thread(target=watch, name="crucible-slot-gpu-watch", daemon=True).start()

    class _TorchHook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "torch":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec("torch")
            if spec is None or spec.loader is None:
                return spec
            exec_module = spec.loader.exec_module

            def exec_and_patch(module):
                exec_module(module)
                patch(module)
            spec.loader.exec_module = exec_and_patch
            return spec

    if "torch" in sys.modules:
        patch(sys.modules["torch"])
    else:
        sys.meta_path.insert(0, _TorchHook())


try:
    _install()
finally:
    _chain_next_sitecustomize()
