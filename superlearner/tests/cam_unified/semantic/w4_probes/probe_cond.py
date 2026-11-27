"""Probe: (1) prefix base options vs no prefix (KL of next-token distributions on JSON-argument contexts);
(2) decoder latent conditioning at alpha 0 (bit-identical) and alpha 0.05 (top-1 agreement, KL)."""
import json
import os

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from superlearner.cam_unified.config import make_preset
from superlearner.cam_unified.semantic.encoder import QwenSemantic
from superlearner.cam_unified.semantic.bridge import LatentBridge

cfg = make_preset("dom_dev")
enc = QwenSemantic(cfg, dtype=torch.float32)
print("conditioning layers", enc.conditioning.layer_ids, "params", enc.conditioning.n_params())
tok = enc.tok
schema = {"type": "object", "properties": {"handle": {"type": "string", "maxLength": 40},
                                           "name": {"type": "string", "maxLength": 80}}, "required": ["handle", "name"]}
user = ("Write the JSON arguments for one call of the tool below. Reply with the JSON object only.\nTool: id.create_org\n"
        "Description: Create an organisation you own.\nArguments schema: " + json.dumps(schema))
head = tok("<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
tail = tok("<think>\n\n</think>\n\n" + '{"handle": "bakery-labs", "name": "Bakery Labs", "kind": "company"}',
           add_special_tokens=False)["input_ids"]
dev = enc.device


def run(prefix, cond=None):
    ids = torch.tensor(head + tail, device=dev)
    emb = enc.embed_ids(ids).float()
    P = 0 if prefix is None else prefix.shape[0]
    if prefix is not None:
        emb = torch.cat([emb[: len(head)], prefix.to(emb.dtype), emb[len(head):]], 0)
    att = torch.ones(1, emb.shape[0], dtype=torch.long, device=dev)
    pos = torch.arange(emb.shape[0], device=dev).unsqueeze(0)
    with torch.no_grad():
        h = enc.decoder_hidden(emb.unsqueeze(0), att, pos, cond=cond)
        lg = enc.lm_logits(h)[0, len(head) + P - 1:]          # predictions of the tail tokens
    return torch.log_softmax(lg, -1)


out = {}
base = run(None)
n = cfg.n_prefix
w = enc.input_embeddings().weight[: len(tok)].float()
rms = enc.embedding_rms
opts = {
    "phrase_old": LatentBridge.init_rows_from(enc, n, text=" Plan, evidence and intent from the controller:"),
    "mean_row_scaled": LatentBridge.init_rows_from(enc, n),
    "zeros": torch.zeros(n, enc.sem_dim),
    "space_token": enc.embed_ids(torch.tensor(tok(" ", add_special_tokens=False)["input_ids"] * n, device=dev)).float().cpu()[:n],
    "newline_token": enc.embed_ids(torch.tensor(tok("\n", add_special_tokens=False)["input_ids"] * n, device=dev)).float().cpu()[:n],
}
for name, rows in opts.items():
    lp = run(rows.to(dev))
    kl = (base.exp() * (base - lp)).sum(-1)
    agree = (base.argmax(-1) == lp.argmax(-1)).float().mean()
    out[name] = {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "top1_agree": float(agree)}
    print(name, out[name])

# decoder conditioning
Z = torch.randn(1, 40, cfg.d_model, device=dev)
pre = opts["mean_row_scaled"].to(dev)
lp0 = run(pre)
lpc = run(pre, cond=(Z, None))
out["cond_alpha0_bit_identical"] = bool(torch.equal(lp0, lpc))
out["cond_alpha0_maxdiff"] = float((lp0 - lpc).abs().max())
for a in (0.05, 0.2, 1.0):
    with torch.no_grad():
        for m in enc.conditioning.mods.values():
            m.alpha.fill_(a)
    lpa = run(pre, cond=(Z, None))
    kl = (lp0.exp() * (lp0 - lpa)).sum(-1)
    out[f"cond_alpha{a}"] = {"top1_agree": float((lp0.argmax(-1) == lpa.argmax(-1)).float().mean()),
                             "kl_mean": float(kl.mean())}
    print("alpha", a, out[f"cond_alpha{a}"])
print(json.dumps(out, indent=1))
