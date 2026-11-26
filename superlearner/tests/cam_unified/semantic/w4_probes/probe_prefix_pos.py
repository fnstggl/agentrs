"""Probe: where a neutral prefix perturbs the decoder least (KL vs no prefix over several JSON-argument texts)."""
import json
import os

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from superlearner.cam_unified.config import make_preset
from superlearner.cam_unified.semantic.encoder import QwenSemantic
from superlearner.cam_unified.semantic.bridge import LatentBridge

cfg = make_preset("dom_dev", dec_xattn_every=0)
enc = QwenSemantic(cfg, dtype=torch.float32)
tok = enc.tok
dev = enc.device
cases = [("id.create_org", "Create an organisation you own.",
          '{"handle": "bakery-labs", "name": "Bakery Labs", "kind": "company"}'),
         ("mail.send", "Send an email to a participant.",
          '{"to": ["p:client-3"], "subject": "Invoice 17", "body": "Hello, please find the invoice for March attached."}'),
         ("web.search", "Search the web.", '{"query": "csv to json converter", "limit": 5}')]
n = cfg.n_prefix
print("n_prefix", n)
rows = {"mean_row_scaled": LatentBridge.init_rows_from(enc, n).to(dev),
        "phrase_old": LatentBridge.init_rows_from(enc, n, text=" Plan, evidence and intent from the controller:").to(dev)}


def run(head, tail, prefix, where):
    ids_h = torch.tensor(head, device=dev)
    ids_t = torch.tensor(tail, device=dev)
    eh, et = enc.embed_ids(ids_h).float(), enc.embed_ids(ids_t).float()
    P = 0 if prefix is None else prefix.shape[0]
    if prefix is None:
        emb = torch.cat([eh, et], 0)
    elif where == "after_assistant":
        emb = torch.cat([eh, prefix, et], 0)
    else:                                   # at the start of the sequence
        emb = torch.cat([prefix, eh, et], 0)
    att = torch.ones(1, emb.shape[0], dtype=torch.long, device=dev)
    pos = torch.arange(emb.shape[0], device=dev).unsqueeze(0)
    with torch.no_grad():
        h = enc.decoder_hidden(emb.unsqueeze(0), att, pos)
        lg = enc.lm_logits(h)[0, emb.shape[0] - len(tail) - 1: -1]
    return torch.log_softmax(lg, -1)


res = {}
for name, desc, args in cases:
    user = (f"Write the JSON arguments for one call of the tool below. Reply with the JSON object only.\nTool: {name}\n"
            f"Description: {desc}\n")
    head = tok("<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    tail = tok("<think>\n\n</think>\n\n" + args, add_special_tokens=False)["input_ids"]
    base = run(head, tail, None, None)
    for rn, r in rows.items():
        for where in ("after_assistant", "start"):
            lp = run(head, tail, r, where)
            kl = (base.exp() * (base - lp)).sum(-1)
            k = f"{rn}@{where}"
            res.setdefault(k, []).append((float(kl.mean()), float((base.argmax(-1) == lp.argmax(-1)).float().mean())))
for k, v in res.items():
    print(k, "kl_mean", round(sum(a for a, _ in v) / len(v), 4), "top1", round(sum(b for _, b in v) / len(v), 3))
