"""Measure per-group parameter counts of the integrated agent (DOC / DL-02).

Builds CamLiveAgent exactly as build_agent does (every head the agent registers: predict heads, R_psi, credit heads,
...) and calls the model's own CamLiveModel.count_parameters(). Default configuration: configs/cam_live.yaml
(Qwen3-4B at the pinned revision, LoRA r32 q/v, bridge, 50m core) built on CPU (float32 on CPU; counts do not depend
on dtype), so no GPU and no GPU lock is used. Also the debug configuration (tiny random LM + debug core).
"""
import json, os, sys, tempfile, time, subprocess
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from cam_live.config import CamLiveConfig
from cam_live.agent import CamLiveAgent
from cam_live.nn.model import build_model
from cam_live.nn.api import GROUPS


def measure(cfg, label):
    t0 = time.time()
    model = build_model(cfg)
    before = model.count_parameters()
    with tempfile.TemporaryDirectory(dir=os.environ.get("CAM_LIVE_HOME")) as d:
        agent = CamLiveAgent(cfg, d, model=model)
        c = model.count_parameters()
        trainable = {g: sum(p.numel() for p in ps if p.requires_grad) for g, ps in model.param_groups().items()}
        heads = {n: {"params": sum(p.numel() for p in m.parameters()), "group": c["head_groups"][n],
                     "type": type(m).__name__} for n, m in model.heads.items()}
        info = model.backbone.info()
        try:
            agent.close()
        except Exception as e:  # noqa: BLE001
            print("close:", e, file=sys.stderr)
    return {"label": label, "build_s": round(time.time() - t0, 1), "device": str(model.device),
            "counts_model_only_before_agent": {k: before[k] for k in ("frozen", "experiential", "adapters",
                                                                       "adapters_lora", "adapters_bridge", "reward",
                                                                       "heads_total", "total_persistent_trainable")},
            "count_parameters": c, "requires_grad_per_group": trainable, "heads": heads, "backbone_info": info}


if __name__ == "__main__":
    os.environ.setdefault("CAM_LIVE_HOME", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/DOC/arch/home")
    os.makedirs(os.environ["CAM_LIVE_HOME"], exist_ok=True)
    out = {"command": " ".join(["CAM_LIVE_HOME=<scratch>", sys.executable] + sys.argv),
           "commit": subprocess.run(["git", "-C", "/workspace/camlive", "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(),
           "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "torch": torch.__version__, "groups": list(GROUPS)}
    dbg = CamLiveConfig.debug()
    out["debug"] = measure(dbg, "CamLiveConfig.debug() (tiny random LM, debug core; CPU tests only)")
    if "--debug-only" not in sys.argv:
        cfg = CamLiveConfig.load()
        cfg.device = "cpu"
        out["default"] = measure(cfg, "configs/cam_live.yaml (Qwen/Qwen3-4B pinned revision + LoRA + bridge + 50m), "
                                      "built on CPU for counting")
    dest = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    s = json.dumps(out, indent=1, default=str)
    if dest:
        with open(dest, "w") as f:
            f.write(s + "\n")
    print(s)
