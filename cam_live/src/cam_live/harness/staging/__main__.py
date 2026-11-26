"""Run the staging sites for manual inspection: ``python -m cam_live.harness.staging [--base-port 18600]``."""
import argparse
import json
import time

from . import StagingSites


def main() -> None:
    ap = argparse.ArgumentParser(description="CAM Live staging sites (localhost only)")
    ap.add_argument("--base-port", type=int, default=0, help="first port; 0 = pick free ports")
    a = ap.parse_args()
    names = ["structured", "canvas", "dynamic", "variants", "labsite"]
    ports = {n: a.base_port + i for i, n in enumerate(names)} if a.base_port else None
    with StagingSites(ports=ports) as s:
        print(json.dumps(s.capabilities(), indent=1), flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
