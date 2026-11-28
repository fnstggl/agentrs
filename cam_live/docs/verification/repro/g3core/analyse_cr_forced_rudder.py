"""G3-core analysis CR-1 (D-CR1): what differs between CR's `simple` and forced-`rudder` arms (CR.md section 3.1)?

Reads CR's raw integrated-ablation outputs (``cr_agent_eval`` JSON: abl1/, abl2/ under the CR scratch ``agent``
directory) and prints, per seed, the expected adopters by quarter of the run (paired rudder - simple) and, for one
seed, the credited rows of every policy update (``policy_updates``: rows, credit_rows, return_like_rows,
replayed_rows).

    python analyse_cr_forced_rudder.py --cr-agent <scratch>/CR/agent > out/cr_forced_rudder_by_quarter.txt
"""
import argparse
import json
import os
import statistics as st


def load(root, cond, s):
    for d in ("abl1", "abl2"):
        f = f"{root}/{d}/{cond}_s{s}.json"
        if os.path.exists(f):
            return json.load(open(f))
    return None


def quarters(d):
    rows = [(r.get("expected_adopters") or 0.0) for r in d["truth_rows"]]
    n, k = len(rows), len(rows) // 4
    return [sum(rows[i * k:(i + 1) * k if i < 3 else n]) for i in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cr-agent", required=True)
    a = ap.parse_args()
    diffs = []
    for s in range(9):
        x, y = load(a.cr_agent, "simple", s), load(a.cr_agent, "rudder", s)
        if x is None or y is None:
            continue
        qa, qb = quarters(x), quarters(y)
        diffs.append([b - c for c, b in zip(qa, qb)])
        print(f"seed {s}: simple by quarter {[round(v, 1) for v in qa]} rudder {[round(v, 1) for v in qb]}")
    for i in range(4):
        v = [r[i] for r in diffs]
        print(f"quarter {i + 1}: paired rudder - simple {st.mean(v):.2f} +- {st.stdev(v) / len(v) ** 0.5:.2f} "
              f"(n={len(v)}), rudder better in {sum(1 for t in v if t > 0)}")
    for cond in ("simple", "rudder"):
        d = load(a.cr_agent, cond, 1)
        print(cond, "seed 1 policy updates (index, rows, credit_rows, return_like_rows, replayed_rows):",
              [(i, p["rows"], p["credit_rows"], p["return_like_rows"], p["replayed_rows"])
               for i, p in enumerate(d["policy_updates"])][:8])


if __name__ == "__main__":
    main()
