"""Evidence-based capacity-expansion proposals (never applied automatically)."""
from __future__ import annotations

from typing import Optional


class ExpansionAdvisor:
    """Reads the trainer's run state and proposes expansion only when the evidence suggests the
    current capacity is the limitation:

    * plateau: validation success on training families has not improved by ``min_gain`` over the last
      ``window`` promotion gates, while there is still headroom (< ``ceiling``) and the policy is still
      changing (non-trivial LoRA updates) — learning continues but stops transferring;
    * interference: a family's validation success dropped by ≥ ``drop`` from its best while other
      domains were being trained (forgetting / competition for shared capacity);
    * the proposal names the mechanism to test (rank expansion for a global plateau, an expert for
      domain-specific interference) — adoption requires a matched benchmark that shows a held-out gain.
    """

    def __init__(self, window: int = 4, min_gain: float = 0.02, ceiling: float = 0.95, drop: float = 0.25,
                 min_change: float = 1e-3):
        self.window, self.min_gain, self.ceiling, self.drop, self.min_change = window, min_gain, ceiling, drop, min_change

    def assess(self, state: dict) -> dict:
        vals = [e for e in state.get("evals", []) if e.get("split") == "val" and e.get("mode", "direct") == "direct"
                and e.get("train_families")]
        out: dict = {"propose": False, "reasons": [], "evidence": {}, "suggestion": None}
        if len(vals) < self.window + 1:
            out["reasons"].append(f"not enough validation points ({len(vals)} < {self.window + 1})")
            return out
        rates = [v["train_families"]["rate"] for v in vals]
        recent, before = max(rates[-self.window:]), max(rates[:-self.window])
        cyc = [c for c in state.get("cycles", [])][-self.window * 3:]
        moving = sum((c.get("ppo") or {}).get("policy_lora_change_l2", 0.0) or 0.0 for c in cyc)
        train_trend = [c["success_rate"] for c in cyc]
        out["evidence"] = {"val_best_recent": recent, "val_best_before": before, "lora_change_recent": moving,
                           "train_success_recent": train_trend[-6:]}
        plateau = recent - before < self.min_gain and recent < self.ceiling and moving > self.min_change
        # per-family forgetting / interference
        fam_best: dict = {}
        drops: list = []
        for v in vals:
            for k, r in (v["summary"].get("by_family") or {}).items():
                b = fam_best.get(k)
                if b is not None and b - r["rate"] >= self.drop:
                    drops.append((k, b, r["rate"], v.get("cycle")))
                fam_best[k] = max(b or 0.0, r["rate"])
        last_cycle = vals[-1].get("cycle")
        recent_drops = [d for d in drops if d[3] == last_cycle]
        out["evidence"]["recent_family_drops"] = recent_drops
        if plateau:
            out["propose"] = True
            out["reasons"].append(f"validation plateau: best {recent:.3f} vs {before:.3f} before the last "
                                  f"{self.window} gates while the policy keeps changing")
            out["suggestion"] = {"mechanism": "expand_rank", "detail": "LoRA r 16 -> 32 on all layers (function-preserving)"}
        if len({d[0].split('/')[0] for d in recent_drops}) >= 1 and len(recent_drops) >= 2:
            out["propose"] = True
            doms = sorted({d[0].split('/')[0] for d in recent_drops})
            out["reasons"].append(f"interference/forgetting in {doms}: {recent_drops[:4]}")
            out["suggestion"] = {"mechanism": "experts", "detail": f"bounded expert adapters routed by domain; protect {doms}"}
        if not out["propose"]:
            out["reasons"].append("no evidence that capacity is the limitation")
        return out


def capacity_status(state: dict, report: Optional[dict] = None) -> dict:
    return {"mode": (state.get("config") or {}).get("capacity_mode", "fixed"), "advisor": ExpansionAdvisor().assess(state),
            "params": report}
