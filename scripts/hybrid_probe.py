"""Benchmark frozen semantic interfaces before choosing one (Milestone 2).

For each frozen model and hidden-state layer choice: encode natural-language observations of real
environment states, train a small probe (on training-wording texts of training states) to recover every
observation field, and test it on *unseen states* in the training wording and in the held-out wording.
Also: throughput and GPU memory per model, and the accuracy of (a) the rule-based training-wording
parser and (b) frozen-LLM structured extraction (few-shot rewrite into CAM's structured tokens) on
held-out wording — the interfaces of arms C-parser and C-LLM.

    PYTHONPATH=src python scripts/hybrid_probe.py --out runs/cam_hybrid/probe.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from agentrs.cam.envs import ENVS
from agentrs.cam.text import render as R


def states(fam, n_eps, seed):
    r = random.Random(seed)
    out = []
    for s in range(n_eps):
        e = ENVS[fam]()
        e.reset(10_000 + s, r.choice(["train", "val", "test"]))
        g = e.grammar()
        while not e.done:
            pre = []
            while not pre or pre[-1] != "END":
                pre.append(r.choice(g.allowed(pre)))
            e.step(pre[:-1])
            out.append(e.obs_tokens())
    return out


def fields(fam, t) -> dict:
    if fam == "negotiation":
        f = R._neg_fields(t)
        d = {"round": f["round"], "resp": f["resp"], "best": f["best"], "mood": f["mood"]}
        for i, iss in enumerate(R.ISSUES):
            d[f"last_{iss}"] = f["last"][i] if f["last"] else -1
            d[f"know_{iss}"] = dict(f["know"]).get(iss, "-")
        return d
    if fam == "product":
        f = R._prod_fields(t)
        d = {k: f[k] for k in ("WEEK", "BUDGET", "ACTIVE", "SIGNUPS", "CHURN", "REVENUE", "PRICE", "ONB")}
        for i in range(6):
            d[f"built_F{i}"] = f"F{i}" in f["built"]
            d[f"building_F{i}"] = f"F{i}" in f["building"]
        for c in R.CHANNELS:
            d[f"ch_{c}"] = f["ch"][c]
        return d
    f = R._biz_fields(t)
    d = {"step": f["step"], "pool": f["pool"]}
    for p in f["p"]:
        d[f"{p['id']}_status"] = p["status"]
        d[f"{p['id']}_obj"] = p["obj"] or "-"
        d[f"{p['id']}_want"] = p["want"] or "-"
    return d


def probe(Xtr, ytr, Xte_list, epochs=300, seed=0):
    torch.manual_seed(seed)
    classes = sorted(set(ytr) | {y for _, ys in Xte_list for y in ys}, key=str)
    idx = {c: i for i, c in enumerate(classes)}
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-4
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor([idx[y] for y in ytr])
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, len(classes)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(xt), yt).backward()
        opt.step()
    out = []
    with torch.no_grad():
        for X, ys in Xte_list:
            p = net(torch.tensor((X - mu) / sd, dtype=torch.float32)).argmax(-1).numpy()
            out.append(float(np.mean([classes[i] == y for i, y in zip(p, ys)])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen3-0.6B,Qwen/Qwen3-4B")
    ap.add_argument("--eps", type=int, default=120)
    ap.add_argument("--extract_n", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from agentrs.cam.hybrid.semantic import FrozenSemanticEncoder

    torch.set_num_threads(8)
    data = {}
    for fam in ENVS:
        S = states(fam, a.eps, 0)
        random.Random(1).shuffle(S)
        S = S[:3000]
        n_tr = int(0.75 * len(S))
        data[fam] = {"tr": S[:n_tr], "te": S[n_tr:],
                     "tr_txt": [R.obs_text(fam, t, "train", key=i) for i, t in enumerate(S[:n_tr])],
                     "te_txt": [R.obs_text(fam, t, "train", key=10_000 + i) for i, t in enumerate(S[n_tr:])],
                     "te_ho": [R.obs_text(fam, t, "heldout", key=20_000 + i) for i, t in enumerate(S[n_tr:])]}
    rep = {"states": {f: (len(d["tr"]), len(d["te"])) for f, d in data.items()}, "models": {}}
    # rule-based parser on held-out wording (arm C-parser interface)
    rep["parser_heldout_exact"] = {}
    for fam, d in data.items():
        ok = 0
        for t, x in zip(d["te"], d["te_ho"]):
            try:
                ok += R.parse_obs(fam, x) == t
            except Exception:  # noqa: BLE001
                pass
        rep["parser_heldout_exact"][fam] = ok / len(d["te"])
    def wait_free(gb):
        while torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2**30 < gb:
            print(f"waiting for {gb} GB free GPU memory", flush=True)
            time.sleep(60)

    for mname in a.models.split(","):
        wait_free(14 if "4B" in mname else 4)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        enc = FrozenSemanticEncoder(mname, layers=(0.5, 0.75, 1.0), batch=64 if "0.6B" in mname else 16)
        fp0 = enc.fingerprint()
        res = {"describe": enc.describe(), "families": {}}
        t0 = time.time()
        E = {fam: {k: enc.encode(d[k]).astype(np.float32) for k in ("tr_txt", "te_txt", "te_ho")} for fam, d in data.items()}
        dt = time.time() - t0
        res["texts_per_s"] = enc.stats["texts"] / dt
        res["tokens_per_s"] = enc.stats["tokens"] / max(1e-9, enc.stats["forward_s"])
        res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
        res["fingerprint_unchanged"] = enc.fingerprint() == fp0
        H = enc.model.config.hidden_size
        del enc  # release the GPU before the (CPU) probes
        torch.cuda.empty_cache()
        for fam, d in data.items():
            F_tr = [fields(fam, t) for t in d["tr"]]
            F_te = [fields(fam, t) for t in d["te"]]
            names = list(F_tr[0])
            for lname, sl in (("mid", slice(0, H)), ("3q", slice(H, 2 * H)), ("last", slice(2 * H, 3 * H)), ("mid+last", None)):
                pick = (lambda X: np.concatenate([X[:, :H], X[:, 2 * H:]], 1)) if sl is None else (lambda X, s=sl: X[:, s])
                accs_tr, accs_ho, maj = [], [], []
                for n in names:
                    ytr = [f[n] for f in F_tr]
                    yte = [f[n] for f in F_te]
                    a1, a2 = probe(pick(E[fam]["tr_txt"]), ytr, [(pick(E[fam]["te_txt"]), yte), (pick(E[fam]["te_ho"]), yte)])
                    accs_tr.append(a1)
                    accs_ho.append(a2)
                    vals, cnt = np.unique([str(y) for y in ytr], return_counts=True)
                    maj.append(float(np.mean([str(y) == vals[cnt.argmax()] for y in yte])))
                res["families"].setdefault(fam, {})[lname] = {
                    "field_acc_unseen_states_train_wording": float(np.mean(accs_tr)),
                    "field_acc_unseen_states_heldout_wording": float(np.mean(accs_ho)),
                    "majority_baseline": float(np.mean(maj)), "n_fields": len(names),
                    "worst_fields_heldout": sorted(zip(names, np.round(accs_ho, 3)), key=lambda p: p[1])[:4]}
        rep["models"][mname] = res
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str))  # incremental results
    # frozen-LLM structured extraction on held-out wording (arm C-LLM interface)
    if a.extract_n:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mname = a.models.split(",")[-1]
        wait_free(14)
        tok = AutoTokenizer.from_pretrained(mname)
        lm = AutoModelForCausalLM.from_pretrained(mname, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        rep["llm_extraction_heldout"] = {"model": mname}
        for fam, d in data.items():
            shots = "".join(f"Text: {R.obs_text(fam, t, 'heldout', key=900 + i)}\nTokens: {' '.join(t)}\n\n" for i, t in enumerate(d["tr"][:3]))
            exact, tok_acc, secs = 0, [], 0.0
            for t, x in list(zip(d["te"], d["te_ho"]))[:a.extract_n]:
                prompt = ("Rewrite each observation into the structured token format.\n\n" + shots + f"Text: {x}\nTokens:")
                ids = tok(prompt, return_tensors="pt").to(lm.device)
                t0 = time.time()
                with torch.no_grad():
                    g = lm.generate(**ids, max_new_tokens=6 * len(t) + 10, do_sample=False)
                secs += time.time() - t0
                pred = tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].split()
                exact += pred == t
                tok_acc.append(np.mean([a_ == b_ for a_, b_ in zip(pred, t)]) if pred else 0.0)
            rep["llm_extraction_heldout"][fam] = {"exact": exact / a.extract_n, "token_acc": float(np.mean(tok_acc)),
                                                  "sec_per_obs": secs / a.extract_n}
        del lm
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
