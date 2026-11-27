"""Fresh process: build a model from a weights-only export and print the policy distribution on the held-out probe.

    python int2_export_child.py EXPORT_DIR
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))


def main() -> int:
    import torch
    from cam_live.agent import register_agent_heads
    from cam_live.credit.checkpoint import load_exported_model
    from cam_live.validation import HeldOutProbe
    holder = {}
    ex = load_exported_model(sys.argv[1], register_heads=lambda m: holder.setdefault("heads", register_agent_heads(m)),
                             strict=True)
    model, heads = ex.model, holder["heads"]
    probe = HeldOutProbe(None)
    with torch.no_grad():
        b = model.core_forward(model.tokenize([list(probe.items)]), None).b
        e_a, _ = model.encode_actions([list(probe.candidates)])
        e_g = model.encode_goals([probe.goal])
        z = model.decision_latent(b, e_a, e_g)
        mask = torch.ones(1, len(probe.candidates), dtype=torch.bool, device=z.device)
        p = heads.policy(z, mask, b).probs[0].double().cpu().tolist()
    print(json.dumps({"probs": p, "missing": list(getattr(ex.report, "missing", []) or []),
                      "unexpected": list(getattr(ex.report, "unexpected", []) or []),
                      "group_hashes": model.group_hashes()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
