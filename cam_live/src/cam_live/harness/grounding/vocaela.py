"""Vocaela-2-500M-1024R2 (vocaela/Vocaela-2-500M-1024R2): optional third candidate (workstream EC).

A SmolVLM2-500M derivative; weights and configuration only, loaded with ``transformers``' own SmolVLM classes.
**Licence: CC BY-NC-SA 4.0 (non-commercial, share-alike).** It is therefore integrated for evaluation only: the
grounder refuses to load unless ``accept_noncommercial=True`` is passed, the registry marks it ``noncommercial`` and
the runtime never selects it by default. The system message and the JSON action format (coordinates scaled to
0..1) are the ones in its model card. Only the coordinate of the first action is used: whatever action the model
names is ignored, because a grounder proposes a point and nothing else.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict

from .base import GroundingError, NeuralGrounder, point_result

REPO = "vocaela/Vocaela-2-500M-1024R2"
REVISION = "b0280f2ce8e2f126a1c3c446c70e63ee7bbeaed2"
SYSTEM_MESSAGE = """You are an assistant trained to navigate the computer screen. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next actions and wait for the next observation. 

## Allowed ACTION_TYPEs and parameters:
1. `PRESS_KEY`: Press one specified key. Two parameters: `key`, string, the single key to press; `presses`, integer, the number of times to press the key (default is 1).
2. `TYPE`: Type a string into an element. Parameter: `text`, string, the text to type.
3. `MOUSE_MOVE`: Move the mouse cursor to a specified position. Parameter: `coordinate`, formatted as [x,y], the position to move the cursor to.
4. `CLICK`: Click left mouse button once on an element. Parameter: `coordinate`, formatted as [x,y], the position to click on.
5. `DRAG`: Drag the cursor with the left mouse button pressed, start and end positions are specified. Two parameters: `coordinate`, formatted as [x,y], the start position to drag from; `coordinate2`, formatted as [x2,y2], the end position to drag to.
6. `RIGHT_CLICK`: Click right mouse button once on an element. Parameter: `coordinate`, formatted as [x,y], the position to right click on.
7. `MIDDLE_CLICK`: Click middle mouse button once on an element. Parameter: `coordinate`, formatted as [x,y], the position to middle click on.
8. `DOUBLE_CLICK`: Click left mouse button twice on an element. Parameter: `coordinate`, formatted as [x,y], the position to double click on.
9. `SCROLL`: Scroll the screen (via mouse wheel). Parameter: `scroll_direction`, the direction (`up`/`down`/`left`/`right`) to scroll.
10. `HOTKEY`: Press a combination of keys simultaneously. Parameter: `hotkeys`, list of strings, the keys to press together.
11. `ANSWER`: Answer a specific question. Required parameter: `text`, string, the answer text.

* NOTE *: The `coordinate` and `coordinate2` parameters (formatted as [x,y]) are the relative coordinates on the screenshot scaled to range of 0-1, [0,0] is the top-left corner and [1,1] is the bottom-right corner.

## Format your response as
<Action>the next actions</Action>

`The next actions` can be one or multiple actions. Format `the next actions` as a JSON array of objects as below, each object is an action:
[{"action": "<ACTION_TYPE>", "key": "<key>", "presses": <presses>, "hotkeys": ["<hotkeys>"], "text": "<text>", "coordinate": [x,y], "coordinate2": [x2,y2], "scroll_direction": "<scroll_direction>"}]

If a parameter is not applicable, don't include it in the JSON object.
"""


def parse_action(text: str) -> Any:
    """First ``coordinate`` ([x, y] in 0..1) in the model's action list, or None."""
    m = re.search(r"<Action>(.*?)(?:</Action>|$)", text, re.DOTALL)
    body = (m.group(1) if m else text).strip()
    try:
        actions = json.loads(body)
        actions = actions if isinstance(actions, list) else [actions]
        for a in actions:
            coord = a.get("coordinate") if isinstance(a, dict) else None
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                return float(coord[0]), float(coord[1])
    except (ValueError, TypeError):
        pass
    c2 = re.search(r'"coordinate"\s*:\s*\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]', body)
    return (float(c2.group(1)), float(c2.group(2))) if c2 else None


class VocaelaGrounder(NeuralGrounder):
    name = "vocaela-2-500m"
    repo, revision = REPO, REVISION
    license = "CC BY-NC-SA 4.0"
    architecture = "SmolVLM2-500M with SigLIP-B/16-256 (transformers SmolVLMForConditionalGeneration)"
    heavy = False
    noncommercial = True

    def __init__(self, device: str = "auto", allow_inprocess: bool = False, accept_noncommercial: bool = False,
                 max_new_tokens: int = 64, **kw: Any):
        super().__init__(device=device, allow_inprocess=allow_inprocess, accept_noncommercial=accept_noncommercial, **kw)
        self.max_new_tokens = int(max_new_tokens)
        self._proc: Any = None

    def _load(self, snapshot: str, device: str) -> Any:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._proc = AutoProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self._dtype = torch.float16 if device.startswith("cuda") else torch.float32
        return AutoModelForImageTextToText.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                           dtype=self._dtype).to(device)

    def _drop(self) -> None:
        self._proc = None

    def _ground(self, image: Any, instruction: str) -> Dict[str, Any]:
        import torch
        text = instruction if re.match(r"(?i)\s*(click|tap|press|select|open)\b", instruction) else f"Click the {instruction}"
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        inputs = self._proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                return_tensors="pt").to(self.device, dtype=self._dtype)
        out = self._model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                   output_scores=True, return_dict_in_generate=True)
        new = out.sequences[0, inputs["input_ids"].shape[1]:]
        raw = self._proc.decode(new, skip_special_tokens=True)
        coord = parse_action(raw)
        if coord is None:
            raise GroundingError(f"no coordinate in the model output: {raw[:160]!r}")
        lps = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(out.scores, new.tolist())]
        conf = math.exp(sum(lps) / len(lps)) if lps else 0.0
        x = min(max(coord[0], 0.0), 1.0) * (image.width - 1) if 0.0 <= coord[0] <= 1.0 else float("nan")
        y = min(max(coord[1], 0.0), 1.0) * (image.height - 1) if 0.0 <= coord[1] <= 1.0 else float("nan")
        return point_result(x, y, image.width, image.height, conf, self.name, self.revision, raw=raw,
                            confidence_kind="mean token likelihood of the generated action")
