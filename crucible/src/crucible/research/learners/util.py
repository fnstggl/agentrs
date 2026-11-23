"""Small helpers shared by the learners that carry weights."""
from __future__ import annotations


def cpu_adam(params, lr: float, eps: float = 1e-8):
    """Adam for learners that live on the CPU by design.

    `torch.optim.Optimizer.step` asks the accelerator whether a graph capture is in progress whenever a CUDA
    device is visible, which creates a CUDA context (hundreds of megabytes of GPU memory, or an out-of-memory
    error when the GPU is full) in a process that never puts a tensor on the GPU. These learners must not touch
    a GPU that belongs to other jobs, so the check is skipped; nothing else differs from `torch.optim.Adam`."""
    import torch

    class CpuAdam(torch.optim.Adam):
        def _accelerator_graph_capture_health_check(self) -> None:      # recent torch
            return

        def _cuda_graph_capture_health_check(self) -> None:             # older torch
            return

    return CpuAdam(params, lr=lr, eps=eps)
