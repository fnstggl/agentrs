"""``python -m agentrs.live run|status`` — the live observational learner."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from agentrs.live.service import LiveConfig, LiveLearner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agentrs.live")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run until stopped (SIGTERM / Ctrl-C saves the weights)")
    r.add_argument("--config", help="YAML with LiveConfig fields")
    r.add_argument("--workdir")
    r.add_argument("--sources", help="comma-separated: hackernews,bluesky[,reddit]")
    r.add_argument("--bluesky-per-min", type=float)
    r.add_argument("--max-hours", type=float, help="stop after this long (default: run indefinitely)")
    r.add_argument("--threads", type=int)
    s = sub.add_parser("status", help="print the latest status / scores")
    s.add_argument("--workdir", default="runs/live")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        p = Path(a.workdir) / "status.json"
        if not p.exists():
            print("no status yet", file=sys.stderr)
            return 1
        st = json.loads(p.read_text())
        print(json.dumps({k: st[k] for k in ("uptime_s", "checkpoint", "optimizer_steps", "updates", "posts",
                                             "observations", "targets")} |
                         {"overall": st.get("scores_all_time", {}).get("overall")}, indent=1))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = LiveConfig(**((yaml.safe_load(Path(a.config).read_text()) or {}) if a.config else {}))
    upd = {}
    if a.workdir:
        upd["workdir"] = a.workdir
    if a.sources:
        upd["sources"] = [x.strip() for x in a.sources.split(",") if x.strip()]
    if a.bluesky_per_min:
        upd["bluesky_per_min"] = a.bluesky_per_min
    if a.threads:
        upd["threads"] = a.threads
    cfg = cfg.model_copy(update=upd)
    LiveLearner(cfg).run(max_seconds=a.max_hours * 3600 if a.max_hours else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
