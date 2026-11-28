"""G3-core analysis IMP-3 (D-IMP1): are IMP's B2 ablation arms paired ("draw the same random numbers, so with the same
seed they are identical until behaviour differs", predict/imp_eval.py:10-15)?

Reads the raw B2 rows (extract docs/evidence/IMP/ablation_B2_final_code_raw.tar.xz into a directory; it contains
B2/T1_{on,off}_s{0..9}.pjson) and compares lifetime 0, decision 0 of each seed pair: identical context and no learning
yet, so a paired design must give identical recorded inputs and — where imagination does not change the distribution —
the same executed action.

    tar -xJf docs/evidence/IMP/ablation_B2_final_code_raw.tar.xz -C RAW
    python analyse_imp_b2_pairing.py --raw RAW > out/imp_b2_pairing.txt
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    a = ap.parse_args()
    ps = []
    diverged = 0
    print("decision 0 of lifetime 0: executed action per arm, on-arm share_tv, max |difference| of recorded inputs")
    for s in range(10):
        on = json.load(open(f"{a.raw}/B2/T1_on_s{s}.pjson"))
        off = json.load(open(f"{a.raw}/B2/T1_off_s{s}.pjson"))
        x, y = on["lifetimes"][0]["decision_rows"][0], off["lifetimes"][0]["decision_rows"][0]
        dq = max(abs(c[2] - d[2]) for c, d in zip(x["cands"], y["cands"]))
        dc = max(abs(c[4] - d[4]) for c, d in zip(x["cands"], y["cands"]))
        same = (x["operation"], json.dumps(x["arguments"], sort_keys=True)) == \
               (y["operation"], json.dumps(y["arguments"], sort_keys=True))
        tv = float(x["imagination"]["share_tv"])
        ps.append(tv)
        diverged += int(not same)
        print(f"seed {s}: on {x['operation']} {json.dumps(x['arguments'])} | off {y['operation']} "
              f"{json.dumps(y['arguments'])} | executed differ: {not same} | share_tv(on) {tv:.3f} | "
              f"max|d one-step value| {dq:.1e} | max|d imagined continuation| {dc:.3f}")
    dist = [1.0]
    for q in ps:
        new = [0.0] * (len(dist) + 1)
        for k, v in enumerate(dist):
            new[k] += v * (1 - q)
            new[k + 1] += v * q
        dist = new
    print(f"diverged at decision 0: {diverged} of 10; expected if only imagination caused it: {sum(ps):.2f}; "
          f"P(>= {diverged}) = {sum(dist[diverged:]):.1e} (Poisson-binomial over share_tv)")


if __name__ == "__main__":
    main()
