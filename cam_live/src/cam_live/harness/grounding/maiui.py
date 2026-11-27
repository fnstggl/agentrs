"""MAI-UI-2B (Tongyi-MAI/MAI-UI-2B, Apache-2.0): the optional, more capable GUI model (workstream EC).

A Qwen3-VL fine-tune. The repository holds weights and configuration only (no model code); it is loaded with the
``Qwen3VLForConditionalGeneration`` class that ships inside ``transformers``. The grounding prompt and the output
convention (``<answer>{"coordinate": [x, y]}</answer>`` on a 0..999 scale per axis) are the published ones
(github.com/Tongyi-MAI/MAI-UI, ``src/prompt.py`` and ``src/mai_grounding_agent.py``, read on 2026-11-26).

About 4.3 GB of weights in bfloat16: it needs more than 4 GB of GPU memory, so callers on the shared GPU take the
shared lock (``heavy = True``). It is used only for grounding here: any other content of its answer is ignored.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict

from .base import GrounderUnavailable, GroundingError, NeuralGrounder, point_result

REPO = "Tongyi-MAI/MAI-UI-2B"
REVISION = "503050934809558c8dfd2ddedaf9621fa74ac2de"
SCALE = 999
SYSTEM_PROMPT = """You are a GUI grounding agent. 
## Task
Given a screenshot and the user's grounding instruction. Your task is to accurately locate a UI element based on the user's instructions.
First, you should carefully examine the screenshot and analyze the user's instructions,  translate the user's instruction into a effective reasoning process, and then provide the final coordinate.
## Output Format
Return a json object with a reasoning process in <grounding_think></grounding_think> tags, a [x,y] format coordinate within <answer></answer> XML tags:
<grounding_think>...</grounding_think>
<answer>
{"coordinate": [x,y]}
</answer>"""


def parse_answer(text: str) -> Any:
    """``(x, y)`` on the model's 0..999 scale from its answer text, or None."""
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.DOTALL)
    body = m.group(1) if m else text
    try:
        coord = json.loads(body.strip()).get("coordinate")
    except (ValueError, AttributeError):
        c2 = re.search(r'"coordinate"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', body)
        coord = [float(c2.group(1)), float(c2.group(2))] if c2 else None
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None
    try:
        return float(coord[0]), float(coord[1])
    except (TypeError, ValueError):
        return None


class MaiUIGrounder(NeuralGrounder):
    name = "mai-ui-2b"
    repo, revision = REPO, REVISION
    license = "Apache-2.0"
    architecture = "Qwen3-VL 2B (transformers Qwen3VLForConditionalGeneration)"
    heavy = True

    def __init__(self, device: str = "auto", allow_inprocess: bool = False, max_new_tokens: int = 256,
                 max_pixels: int = 1344 * 32 * 32, **kw: Any):
        super().__init__(device=device, allow_inprocess=allow_inprocess, **kw)
        self.max_new_tokens, self.max_pixels = int(max_new_tokens), int(max_pixels)
        self._ip: Any = None
        self._tok: Any = None

    def _load(self, snapshot: str, device: str) -> Any:
        import torch
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        if not device.startswith("cuda"):
            raise GrounderUnavailable("MAI-UI-2B is only run on a GPU here (a 2B vision-language model on CPU is "
                                      "too slow for interactive grounding)")
        # The packaged Qwen3VLProcessor also builds a video processor, which needs torchvision (not installed in
        # this environment). Only still images are used here, so the two parts that matter are used directly:
        # the PIL image processor and the tokenizer with the repository's chat template.
        self._ip = Qwen2VLImageProcessorPil.from_pretrained(snapshot, local_files_only=True)
        self._tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        return Qwen3VLForConditionalGeneration.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16).to(device)

    _helpers = ("_ip", "_tok")

    def _inputs(self, image: Any, instruction: str) -> Dict[str, Any]:
        import torch
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "text", "text": instruction + "\n"}, {"type": "image"}]}]
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vision = self._ip(images=[image], return_tensors="pt", max_pixels=self.max_pixels)
        n = int(vision["image_grid_thw"][0].prod()) // (self._ip.merge_size ** 2)
        if text.count("<|image_pad|>") != 1:
            raise GroundingError("the chat template did not produce exactly one image placeholder")
        text = text.replace("<|image_pad|>", "<|image_pad|>" * n)         # as Qwen3VLProcessor.replace_image_token
        enc = self._tok(text, return_tensors="pt")
        dtype = next(self._model.parameters()).dtype
        ids = enc["input_ids"].to(self.device)
        # 0 = text, 1 = image token (ProcessorMixin.create_mm_token_type_ids). Without it this transformers version
        # gives image tokens plain 1-D positions: the model then loses the horizontal position of what it sees
        # (measured before this line existed: slope of predicted over true x = 0.50, of y = 1.07).
        mm = (ids == self._model.config.image_token_id).to(torch.long)
        if int(mm.sum()) != n:
            raise GroundingError("image placeholder count does not match the image grid")
        return {"input_ids": ids, "attention_mask": enc["attention_mask"].to(self.device), "mm_token_type_ids": mm,
                "pixel_values": vision["pixel_values"].to(self.device, dtype),
                "image_grid_thw": vision["image_grid_thw"].to(self.device)}

    def _decode_text(self, ids: Any) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def _ground(self, image: Any, instruction: str) -> Dict[str, Any]:
        import torch
        inputs = self._inputs(image, instruction)
        out = self._model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                   output_scores=True, return_dict_in_generate=True)
        new = out.sequences[0, inputs["input_ids"].shape[1]:]
        text = self._decode_text(new)
        coord = parse_answer(text)
        if coord is None:
            raise GroundingError(f"no coordinate in the model output: {text[-160:]!r}")
        # likelihood of the answer part of the generation (the tokens after "<answer>"), not a calibrated probability
        lps = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(out.scores, new.tolist())]
        pieces = [self._tok.decode([t]) for t in new.tolist()]
        acc, start = "", 0
        for i, piece in enumerate(pieces):
            acc += piece
            if "<answer>" in acc:
                start = i + 1
                break
        tail = lps[start:] or lps
        conf = math.exp(sum(tail) / len(tail)) if tail else 0.0
        x = coord[0] / SCALE * (image.width - 1)
        y = coord[1] / SCALE * (image.height - 1)
        return point_result(x, y, image.width, image.height, conf, self.name, self.revision, raw=text[-300:],
                            confidence_kind="mean token likelihood of the answer span")
