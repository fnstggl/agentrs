"""Crucible 4.0 benchmark suite: measured kernel, world-family, storage, memory, scaling, rollout, browser and
episode-cost numbers, each with the conditions it was measured under. Writes docs/benchmarks/openworld.json
(the 3.0 files bench.json and internet.json are never touched).

    python examples/bench_openworld.py --quick                      # a few minutes
    python examples/bench_openworld.py                              # the complete table (30-day episodes)
    python examples/bench_openworld.py --only kernel,scaling        # any subset, merged into the existing file
    python examples/bench_openworld.py --quick --price-per-cpu-hour 0.04 --llm-file llm_throughput.json
    python examples/bench_openworld.py --list                       # sections and the families found right now

World families are discovered when the script runs (`crucible.worldgen.families()` after importing every package
that registers families), so newly added ``mech_*`` and ``ow_*`` families are benchmarked without changes here.
A family that fails to build or to run is reported as an error row. The implementation is `crucible.scale.bench`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from crucible.scale import bench  # noqa: E402

if __name__ == "__main__":         # required: worker processes re-import this file
    bench.main(default_out=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs",
                                                         "benchmarks", "openworld.json")))
