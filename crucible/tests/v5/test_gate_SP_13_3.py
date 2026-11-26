"""SP-13.3 (MASTER_SPEC Part 13.3): inference backends benchmarked and chosen by measured workload.

The class-2 shared network (and the class-3 architectures) timed on NumPy, compiled numerical kernels (torch.compile
/ Inductor, C++ code generated with gcc), a JIT compiler standing in for Numba (TorchScript; Numba is not installed and
may not be installed into the shared environment), PyTorch CPU eager, dynamic int8 quantization and batched GPU
(fp32, fp16), at batch sizes 1-1024 (crucible.scale.backends). Behavioural equivalence to fp32 is measured under the
population's choice rule; the choice per batch size is the fastest equivalent backend. High-frequency paths: the
batched path's cost per decision is compared with one call per participant."""
import pytest

from crucible.acceptance.v5 import evidence
from crucible.scale import backends

import _perf_kit as K

pytestmark = [pytest.mark.slow, pytest.mark.timeout(6 * 3600)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 1500 --ram-mb 8000 -- "
       "python -m crucible.acceptance.v5 run --requirement SP-13.3")
FAMILIES = {"numpy": "numpy", "compiled_kernels": "inductor_cpu", "numba_equivalent_jit": "torchscript",
            "pytorch_cpu": "torch_cpu", "quantized": "int8_dynamic", "batched_gpu": "gpu_fp32"}


def test_SP_13_3_backends_benchmarked_and_chosen():
    bm = K.backend_matrix()["result"]
    rows = bm["adaptive"]["rows"]
    measured = {f: [r for r in rows if r.get("backend") == b and r.get("available")] for f, b in FAMILIES.items()}
    ok = [f for f, rs in measured.items() if rs and all(r["wall_s"] > 0 for r in rs)]
    choice = backends.choose(bm["adaptive"])
    justified = bool(choice) and all(
        any(r["backend"] == c["best"] and r["B"] == int(B) and r["equivalent"] for r in rows) and
        all(c["best_wall_s"] <= r["wall_s"] for r in rows if r["B"] == int(B) and r.get("available") and r.get("equivalent"))
        for B, c in choice.items())
    m = {"backends_benchmarked": len(ok), "backend_choices_justified_by_measurement": justified,
         "families": ",".join(ok)}
    for B, c in choice.items():
        m[f"choice_B{B}"] = c["best"]
        m[f"choice_B{B}_cpu"] = c["best_cpu"] or "none"
    for r in rows:
        if r.get("available") and r["B"] in (1, 256, 1024):
            m[f"{r['backend']}_B{r['B']}_decisions_per_s"] = float(r["decisions_per_s"])
            m[f"{r['backend']}_B{r['B']}_agreement"] = float(r["paired_agreement"])
    evidence.record("SP-13.3", m, command=CMD, notes=f"choice {choice}; strategic {bm['strategic']}; encoder {bm['encoder']}")
    assert m["backends_benchmarked"] >= 6, measured.keys() - set(ok)
    assert m["backend_choices_justified_by_measurement"]
