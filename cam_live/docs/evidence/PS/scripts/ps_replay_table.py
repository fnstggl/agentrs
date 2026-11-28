"""Paired summary of ps_replay.py results: per source stream and model seed, prequential NLL on / off, by family
block, and retention (NLL of the first block's records after each later block).

    python ps_replay_table.py FILE_OR_DIR [...]
"""
import glob
import json
import math
import os
import statistics as st
import sys

runs = []
for arg in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(arg, "*.json"))) if os.path.isdir(arg) else [arg]
    for f in files:
        d = json.load(open(f))
        runs += d if isinstance(d, list) else [d]
pairs = {}
for r in runs:
    pairs.setdefault((r["source"], r["seed"], r.get("rho_init")), {})[r["cond"]] = r
rows, d_all, d_blocks, d_ret = [], [], {}, []
for (src, seed, rho0), pr in sorted(pairs.items(), key=lambda x: (x[0][0], x[0][1])):
    on, off = pr.get("on"), pr.get("off")
    if not on or not off:
        continue
    name = src.split("/")[-4] if "/abl/" in src else src.split("/")[-4]
    d = on["prequential_nll"] - off["prequential_nll"]
    d_all.append(d)
    blk = []
    for bo, bf in zip(on["by_block"], off["by_block"]):
        if bo["prequential_nll"] is not None and bf["prequential_nll"] is not None:
            key = f"{bo['block']}:{bo['family']}"
            d_blocks.setdefault(key, []).append(bo["prequential_nll"] - bf["prequential_nll"])
            blk.append(f"{bo['family']} {bo['prequential_nll']:.3f}/{bf['prequential_nll']:.3f}")
    ro = [x["first_block_nll"] for x in on["retention"]]
    rf = [x["first_block_nll"] for x in off["retention"]]
    if len(ro) > 1 and None not in ro + rf:
        d_ret.append((ro[-1] - ro[0]) - (rf[-1] - rf[0]))
    print(f"| {name} | {seed} | {on['n_scored']} | {on['prequential_nll']:.3f} | {off['prequential_nll']:.3f} | {d:+.3f} | "
          + "; ".join(blk) + f" | {ro[0]:.3f} -> {ro[-1]:.3f} | {rf[0]:.3f} -> {rf[-1]:.3f} | rho {on.get('rho') and round(sum(on['rho'])/len(on['rho']), 4)} |")
if d_all:
    se = st.pstdev(d_all) / math.sqrt(len(d_all)) if len(d_all) > 1 else float("nan")
    print(f"\nprequential NLL on - off: mean {st.mean(d_all):+.3f} (SE {se:.3f}) over {len(d_all)} pairs; "
          f"on better in {sum(1 for x in d_all if x < 0)} of {len(d_all)}")
    for k, v in d_blocks.items():
        print(f"  block {k}: mean {st.mean(v):+.3f} over {len(v)} (on better in {sum(1 for x in v if x < 0)})")
    if d_ret:
        print(f"forgetting of the first block (on minus off, NLL change first->last block): mean {st.mean(d_ret):+.3f} "
              f"over {len(d_ret)} (on forgets less in {sum(1 for x in d_ret if x < 0)})")
