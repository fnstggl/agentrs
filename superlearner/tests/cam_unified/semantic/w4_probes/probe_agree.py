"""Top-1 agreement / KL of the conditioned decoder (dom_dev layers, random unit-scale CAM tokens) vs the pretrained
decoder on JSON tool-call texts, at several gate values. Qwen3-0.6B fp32, memory capped at 3.9 GB."""
import json
import os
import sys

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.cuda.set_per_process_memory_fraction(3.9e9 / torch.cuda.get_device_properties(0).total_memory)
from superlearner.cam_unified.config import make_preset
from superlearner.cam_unified.semantic.encoder import QwenSemantic

cfg = make_preset("dom_dev")
torch.manual_seed(0)
enc = QwenSemantic(cfg, dtype=torch.float32)
texts = [
    '{"handle": "bakery-labs", "name": "Bakery Labs", "kind": "company"}',
    '{"to": ["p:client-3"], "subject": "Invoice 17", "body": "Hello, please find the invoice for March attached."}',
    '{"query": "csv to json converter", "limit": 5}',
    '{"need": "gen.transform.667a8225", "sort": "price", "limit": 10}',
    '{"repo": "zovo-15-labs/zovo-15-decode", "target": "service:solve", "timeout_s": 10}',
    '{"name": "decode-service", "owner": "o:zovo15", "description": "Decode partner feeds into records."}',
    '{"account": "acct_000123", "to": "acct_000456", "amount_cents": 2500, "memo": "server rent"}',
    '{"provider": "cloud-1", "cores": 2, "hours": 24}',
    '{"title": "Transform 0.1: Normalise records from partner feeds", "price_cents": 4, "kind": "api"}',
    '{"text": "Thanks for the order! We will deliver 40 units by Friday.", "to": "cust_7"}',
]
out = {"layers": enc.conditioning.layer_ids}
Z = torch.randn(1, 48, cfg.d_model, device=enc.device)
for a in (0.0, 0.05, 0.1, 0.2):
    with torch.no_grad():
        for m in enc.conditioning.mods.values():
            m.alpha.fill_(a)
    n = agree = 0
    kl = 0.0
    for t in texts:
        ids = enc.tok("<|im_start|>assistant\n" + t, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
        emb = enc.embed_ids(ids)
        att = torch.ones_like(ids)
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        with torch.no_grad():
            l0 = torch.log_softmax(enc.lm_logits(enc.decoder_hidden(emb, att, pos))[0], -1)
            l1 = torch.log_softmax(enc.lm_logits(enc.decoder_hidden(emb, att, pos, cond=(Z, None)))[0], -1)
        n += l0.shape[0]
        agree += int((l0.argmax(-1) == l1.argmax(-1)).sum())
        kl += float((l0.exp() * (l0 - l1)).sum())
    out[f"alpha_{a}"] = {"tokens": n, "top1_agreement": agree / n, "kl_per_token": kl / n}
    print(a, out[f"alpha_{a}"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
