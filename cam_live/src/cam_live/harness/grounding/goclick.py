"""GoClick-Base (HongxinLi/GoClick-Base, MIT): the lightweight grounding candidate (workstream EC).

GoClick is a Florence-2 fine-tune published with *custom model code* (``modeling_florence2.py`` etc., needing
``trust_remote_code``, ``timm``, ``einops`` and an old ``transformers``). That code was read (see
``docs/EC/MODEL_REVIEW.md``) and is **not executed**: the checkpoint is loaded into the Florence-2 implementation
that ships inside ``transformers`` by renaming its tensors (:func:`convert_state_dict`; every tensor must land on a
parameter of identical shape, otherwise loading fails). The pieces of the published processor that matter are
reimplemented here from the reviewed source: the 1024 extra tokens appended to the BART vocabulary in a fixed
order (``<loc_0>`` = id 50269), bicubic resize to 768x768 with ImageNet normalisation, and ``[image tokens, prompt
tokens]`` as encoder input. Output ``<loc_x>,<loc_y>`` addresses 1000 bins per axis of the original image.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from .base import GrounderUnavailable, GroundingError, NeuralGrounder, point_result

REPO = "HongxinLi/GoClick-Base"
REVISION = "05633a495f9d2ce823372d0c7fd9b0b4bb67c6fa"
LOC_BINS = 1000
#: tokens the published processor appends to the tokenizer, in this order (processing_florence2.py, __init__)
EXTRA_TOKENS: List[str] = (["<od>", "</od>", "<ocr>", "</ocr>"] + [f"<loc_{i}>" for i in range(LOC_BINS)]
                           + ["<cap>", "</cap>", "<ncap>", "</ncap>", "<dcap>", "</dcap>", "<grounding>",
                              "</grounding>", "<seg>", "</seg>", "<sep>", "<region_cap>", "</region_cap>",
                              "<region_to_desciption>", "</region_to_desciption>", "<proposal>", "</proposal>",
                              "<poly>", "</poly>", "<and>"])
#: prompt templates from the model card (description / intent / functionality grounding)
PROMPTS = {"description": "Where is the {target} element? (Output the center coordinates of the target)",
           "intent": "I want to {target}. Please locate the target element I should interact with. "
                     "(Output the center coordinates of the target)",
           "functionality": "Locate the element according to its detailed functionality description. {target} "
                            "(Output the center coordinates of the target)"}

_RULES: Tuple[Tuple[str, str], ...] = (
    (r"^language_model\.model\.", "model.language_model."),
    (r"^language_model\.lm_head\.", "lm_head."),
    (r"^vision_tower\.convs\.(\d+)\.proj\.", r"model.vision_tower.convs.\1.conv."),
    (r"^vision_tower\.", "model.vision_tower."),
    (r"^image_pos_embed\.", "model.multi_modal_projector.image_position_embed."),
    (r"^image_proj_norm\.", "model.multi_modal_projector.image_proj_norm."),
    (r"^visual_temporal_embed\.", "model.multi_modal_projector.visual_temporal_embed."),
)
_BLOCK_RULES: Tuple[Tuple[str, str], ...] = (
    (r"\.(conv[12])\.fn\.dw\.", r".\1."),
    (r"\.(window_attn|channel_attn)\.fn\.", r".\1."),
    (r"\.(window_attn|channel_attn)\.norm\.", ".norm1."),
    (r"\.ffn\.fn\.net\.", ".ffn."),
    (r"\.ffn\.norm\.", ".norm2."),
)


def convert_state_dict(original: Dict[str, Any]) -> Dict[str, Any]:
    """Rename a published Florence-2 / GoClick state dict to the layout of ``transformers``' Florence-2."""
    out: Dict[str, Any] = {}
    for key, tensor in original.items():
        if key == "language_model.final_logits_bias":
            continue                                    # kept separately by the caller (all zeros in practice)
        if key == "image_projection":                   # a (in, out) matrix used as x @ W -> Linear weight (out, in)
            out["model.multi_modal_projector.image_projection.weight"] = tensor.t().contiguous()
            continue
        new = key
        for pat, rep in _RULES:
            new, n = re.subn(pat, rep, new)
            if n:
                break
        if ".blocks." in new:
            for pat, rep in _BLOCK_RULES:
                new = re.sub(pat, rep, new)
        out[new] = tensor
    return out


class GoClickGrounder(NeuralGrounder):
    name = "goclick-base"
    repo, revision = REPO, REVISION
    license = "MIT"
    architecture = "Florence-2 base (DaViT + BART encoder-decoder), loaded into transformers' native Florence2"
    heavy = False

    def __init__(self, device: str = "auto", allow_inprocess: bool = False, prompt: str = "description",
                 num_beams: int = 3, max_new_tokens: int = 12, **kw: Any):
        super().__init__(device=device, allow_inprocess=allow_inprocess, **kw)
        if prompt not in PROMPTS:
            raise ValueError(f"prompt must be one of {sorted(PROMPTS)}")
        self.prompt, self.num_beams, self.max_new_tokens = prompt, int(num_beams), int(max_new_tokens)
        self._tok: Any = None
        self._bias: Any = None
        self._loc0 = 0

    # ------------------------------------------------------------------------------------------ loading
    def _load(self, snapshot: str, device: str) -> Any:
        import torch
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        from transformers import Florence2Config, Florence2ForConditionalGeneration

        with open(os.path.join(snapshot, "config.json")) as f:
            cfg = json.load(f)
        vc, tc = cfg["vision_config"], dict(cfg["text_config"])
        tc["model_type"] = "bart"
        tc["tie_word_embeddings"] = False
        # the published checkpoint stores an output head that differs from the input embedding (max |diff| 0.37):
        # the two must stay separate tensors, so weight tying is switched off
        config = Florence2Config(tie_word_embeddings=False, text_config=tc, vision_config=dict(
            depths=vc["depths"], patch_size=vc["patch_size"], patch_stride=vc["patch_stride"],
            patch_padding=vc["patch_padding"], patch_prenorm=vc["patch_prenorm"], embed_dim=vc["dim_embed"],
            num_heads=vc["num_heads"], num_groups=vc["num_groups"], window_size=vc["window_size"],
            drop_path_rate=vc["drop_path_rate"], projection_dim=vc["projection_dim"],
            max_temporal_embeddings=vc["visual_temporal_embedding"]["max_temporal_embeddings"],
            max_position_embeddings=vc["image_pos_embed"]["max_pos_embeddings"]))
        model = Florence2ForConditionalGeneration(config)
        original = load_file(os.path.join(snapshot, "model.safetensors"))       # safetensors only, no pickle
        converted = convert_state_dict(original)
        target = model.state_dict()
        unknown = sorted(k for k in converted if k not in target)
        wrong = sorted(k for k in converted if k in target and tuple(converted[k].shape) != tuple(target[k].shape))
        if unknown or wrong:
            raise GrounderUnavailable(f"GoClick checkpoint does not fit the native Florence-2 layout: "
                                      f"unknown={unknown[:5]} shape_mismatch={wrong[:5]}")
        missing = sorted(k for k in target if k not in converted)
        tied = ("model.language_model.encoder.embed_tokens.weight", "model.language_model.decoder.embed_tokens.weight")
        if any(k not in tied for k in missing):
            raise GrounderUnavailable(f"GoClick checkpoint leaves parameters uninitialised: "
                                      f"{[k for k in missing if k not in tied][:5]}")
        model.load_state_dict(converted, strict=False)
        if model.lm_head.weight.data_ptr() == model.model.language_model.shared.weight.data_ptr():
            raise GrounderUnavailable("output head and input embedding are tied; the checkpoint needs them separate")
        shared = model.model.language_model.shared.weight
        for emb in (model.model.language_model.encoder.embed_tokens, model.model.language_model.decoder.embed_tokens):
            if emb.weight.data_ptr() != shared.data_ptr():
                emb.weight = shared                                             # BART ties them to ``shared``
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = model.to(device=device, dtype=dtype)
        self._bias = original["language_model.final_logits_bias"].to(device=device, dtype=torch.float32)
        tok = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
        base_vocab = tok.get_vocab_size()
        tok.add_special_tokens(EXTRA_TOKENS)
        self._loc0 = tok.token_to_id("<loc_0>")
        if self._loc0 != base_vocab + 4 or tok.get_vocab_size() != config.text_config.vocab_size:
            raise GrounderUnavailable(f"tokenizer layout differs from the published processor: <loc_0>={self._loc0}, "
                                      f"vocab={tok.get_vocab_size()} (model {config.text_config.vocab_size})")
        self._tok = tok
        self._ids = (config.text_config.decoder_start_token_id, config.text_config.bos_token_id,
                     config.text_config.eos_token_id)
        return model

    def _drop(self) -> None:
        self._tok = self._bias = None

    # ------------------------------------------------------------------------------------------ inference
    def _pixels(self, image: Any) -> Any:
        import numpy as np
        import torch
        from PIL import Image
        arr = np.asarray(image.resize((768, 768), Image.BICUBIC), dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    def _encode(self, image: Any, prompt: str) -> Tuple[Any, Any]:
        import torch
        m = self._model
        dtype = next(m.parameters()).dtype
        feats = m.get_image_features(self._pixels(image).to(self.device, dtype))
        feats = getattr(feats, "pooler_output", feats)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        ids = torch.tensor([self._tok.encode(prompt).ids[:512]], device=self.device)
        embeds = torch.cat([feats, m.get_input_embeddings()(ids)], dim=1)       # [image tokens, prompt tokens]
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device)
        enc = m.model.language_model.encoder(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        return enc, mask

    def _step(self, enc: Any, mask: Any, seqs: Any) -> Any:
        import torch
        m = self._model
        n = seqs.shape[0]
        hid = m.model.language_model.decoder(input_ids=seqs, encoder_hidden_states=enc.expand(n, -1, -1),
                                             encoder_attention_mask=mask.expand(n, -1)).last_hidden_state[:, -1]
        logits = m.lm_head(hid).float() + self._bias
        return torch.log_softmax(logits, dim=-1)

    def _decode(self, enc: Any, mask: Any) -> Tuple[List[int], List[float]]:
        """Beam search as configured by the checkpoint (3 beams, forced BOS, stop at EOS); returns the best
        sequence and the log-probability of each of its tokens."""
        import torch
        start, bos, eos = self._ids
        beams = [([start, bos], [0.0], 0.0)]            # (tokens, per-token log-probabilities, sum)
        done: List[Tuple[List[int], List[float], float]] = []
        for _ in range(self.max_new_tokens):
            seqs = torch.tensor([b[0] for b in beams], device=self.device)
            logp = self._step(enc, mask, seqs)
            cand = []
            for bi, (toks, lps, total) in enumerate(beams):
                top = torch.topk(logp[bi], self.num_beams)
                for lp, tid in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((toks + [tid], lps + [lp], total + lp))
            cand.sort(key=lambda c: c[2], reverse=True)
            beams = []
            for toks, lps, total in cand[:self.num_beams]:
                if toks[-1] == eos:
                    done.append((toks, lps, total / max(1, len(toks) - 1)))
                else:
                    beams.append((toks, lps, total))
            if len(done) >= self.num_beams or not beams:
                break
        if not done:
            done = [(t, l, s / max(1, len(t) - 1)) for t, l, s in beams]
        best = max(done, key=lambda d: d[2])
        return best[0], best[1]

    def _ground(self, image: Any, instruction: str) -> Dict[str, Any]:
        import math
        prompt = PROMPTS[self.prompt].format(target=instruction)
        enc, mask = self._encode(image, prompt)
        toks, lps = self._decode(enc, mask)
        raw = self._tok.decode(toks, skip_special_tokens=False)
        locs = [(t - self._loc0, lp) for t, lp in zip(toks, lps) if self._loc0 <= t < self._loc0 + LOC_BINS]
        if len(locs) < 2:
            raise GroundingError(f"no coordinate in the model output: {raw[:120]!r}")
        (bx, lx), (by, ly) = locs[0], locs[1]
        x = (bx + 0.5) / LOC_BINS * image.width
        y = (by + 0.5) / LOC_BINS * image.height
        return point_result(min(x, image.width - 1), min(y, image.height - 1), image.width, image.height,
                            math.exp(lx + ly), self.name, self.revision, raw=raw, bins=[bx, by],
                            confidence_kind="probability of the two coordinate tokens")
