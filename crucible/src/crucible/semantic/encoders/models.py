"""Pretrained models behind the content encoders: pinned Hugging Face revisions, local loading, parameter counts,
weights hashes and forward-pass accounting.

Every model is pinned to a commit of its Hugging Face repository and loaded from the local cache (`$HF_HOME`, default
`/workspace/.hf_home`) with `local_files_only=True`, so a simulation never touches the network. `fetch()` (or
`python -m crucible.semantic.encoders fetch`) downloads exactly the files listed here, once.

| role | repository | params | file | why this one |
|---|---|---|---|---|
| text | sentence-transformers/all-MiniLM-L6-v2 | 22.7M | 91 MB safetensors | small, strong sentence embeddings (mean pooling) |
| image | openai/clip-vit-base-patch32 (safetensors conversion, refs/pr/62) | 151M (vision 88M) | 605 MB | general visual semantics, reads rendered text, image space shared with a text tower |
| audio | MIT/ast-finetuned-audioset-10-10-0.4593 | 86.6M | 346 MB safetensors | AudioSet-supervised (527 classes: speech, music, instruments, tones, environmental sound), mAP 0.459; half the size of CLAP and stored as safetensors; its speech posterior gates the transcript |
| speech | openai/whisper-tiny | 37.8M | 151 MB safetensors | the smallest Whisper; transcripts of speech in audio and video feed the text encoder |

Total weights on disk: about 1.2 GB (budget 3 GB).
"""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    role: str
    repo: str
    revision: str            # pinned commit
    files: tuple[str, ...]   # what fetch() downloads
    weights_file: str
    doc: str = ""


SPECS: dict[str, ModelSpec] = {
    "text": ModelSpec(
        "text", "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json",
         "model.safetensors", "modules.json", "sentence_bert_config.json", "1_Pooling/config.json"),
        "model.safetensors", "MiniLM-L6 sentence encoder; mean pooling over the last hidden state"),
    "image": ModelSpec(
        "image", "openai/clip-vit-base-patch32", "37000f5cdccec47a3f6f83142bff131370757470",
        ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
         "merges.txt", "special_tokens_map.json", "model.safetensors"),
        "model.safetensors", "CLIP ViT-B/32; image features from the projected pooled vision output"),
    "audio": ModelSpec(
        "audio", "MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902",
        ("config.json", "preprocessor_config.json", "model.safetensors"),
        "model.safetensors", "Audio Spectrogram Transformer fine-tuned on AudioSet"),
    "speech": ModelSpec(
        "speech", "openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
         "vocab.json", "merges.txt", "normalizer.json", "added_tokens.json", "special_tokens_map.json",
         "model.safetensors"),
        "model.safetensors", "Whisper tiny speech recogniser (greedy decoding)"),
}


def hf_home() -> str:
    return os.environ.get("HF_HOME") or "/workspace/.hf_home"


def fetch(roles: tuple[str, ...] | None = None) -> dict[str, str]:
    """Download the pinned files of each model into the HF cache (idempotent). Returns role -> snapshot dir."""
    from huggingface_hub import snapshot_download
    out = {}
    for role in roles or tuple(SPECS):
        s = SPECS[role]
        out[role] = snapshot_download(s.repo, revision=s.revision, allow_patterns=list(s.files),
                                      cache_dir=os.path.join(hf_home(), "hub"))
    return out


def snapshot_dir(role: str) -> str:
    """The local snapshot of a pinned model; raises FileNotFoundError (with the fix) if it was never fetched."""
    s = SPECS[role]
    base = os.path.join(hf_home(), "hub", "models--" + s.repo.replace("/", "--"), "snapshots", s.revision)
    if not os.path.exists(os.path.join(base, s.weights_file)):
        raise FileNotFoundError(
            f"{s.repo}@{s.revision[:12]} is not in the local Hugging Face cache ({hf_home()}); run "
            f"`PYTHONPATH=src python -m crucible.semantic.encoders fetch` once")
    return base


def weights_file_sha256(role: str) -> str:
    """SHA-256 of the weights file as stored by the hub (the blob name of an LFS file is its SHA-256)."""
    path = os.path.realpath(os.path.join(snapshot_dir(role), SPECS[role].weights_file))
    name = os.path.basename(path)
    if len(name) == 64 and all(c in "0123456789abcdef" for c in name):
        return name
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_hash(module_or_state_dict) -> str:
    """SHA-256 over sorted parameter tensors (name, dtype, shape, bytes). Same definition as
    `crucible.substrate.inference.weights_hash` (INTERFACE_CONTRACTS 1.7); that function is used when it exists."""
    try:
        from ...substrate import inference as _inf  # noqa: F401
        fn = getattr(_inf, "weights_hash", None)
    except ImportError:
        fn = None
    if fn is not None:
        return fn(module_or_state_dict)
    import torch
    sd = module_or_state_dict.state_dict() if hasattr(module_or_state_dict, "state_dict") else module_or_state_dict
    h = hashlib.sha256()
    for name in sorted(sd):
        t = sd[name].detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        h.update(f"{name}\0{str(t.dtype).replace('torch.', '')}\0{list(t.shape)}\0".encode())
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------- device and loading
def device() -> str:
    """`CRUCIBLE_ENCODER_DEVICE` (cpu|cuda|cuda:N|auto, default auto): CUDA when the slot gave us a GPU, else CPU."""
    want = os.environ.get("CRUCIBLE_ENCODER_DEVICE", "auto")
    if want != "auto":
        return want
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Loaded:
    role: str
    model: Any
    processor: Any
    params: int
    device: str
    forward_passes: int = 0
    weights_sha256: str = ""
    extra: dict = field(default_factory=dict)


_LOCK = threading.RLock()
_LOADED: dict[str, Loaded] = {}


def _count_forward(loaded: Loaded):
    def hook(_mod, _inp, _out):
        loaded.forward_passes += 1
    return hook


def load(role: str) -> Loaded:
    """Load (once per process) the pinned model for `role`, in eval mode, fp32, on `device()`."""
    with _LOCK:
        if role in _LOADED:
            return _LOADED[role]
        import torch
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        n_threads = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
        if n_threads > 0 and torch.get_num_threads() != n_threads:
            torch.set_num_threads(n_threads)
        path = snapshot_dir(role)
        dev = device()
        kw = dict(local_files_only=True)
        used = None                       # the sub-networks that actually run (default: the whole model)
        if role == "text":
            from transformers import AutoModel, AutoTokenizer
            proc = AutoTokenizer.from_pretrained(path, **kw)
            model = AutoModel.from_pretrained(path, dtype=torch.float32, **kw)
            top = model
        elif role == "image":
            from transformers import CLIPModel, CLIPProcessor
            proc = CLIPProcessor.from_pretrained(path, **kw)
            model = CLIPModel.from_pretrained(path, dtype=torch.float32, **kw)
            top = model.vision_model
            used = [model.vision_model, model.visual_projection]     # the text tower is loaded but never run
        elif role == "audio":
            import warnings

            from transformers import ASTFeatureExtractor, ASTForAudioClassification
            with warnings.catch_warnings():      # the numpy fbank's top mel filter is empty at 16 kHz (harmless)
                warnings.simplefilter("ignore")
                proc = ASTFeatureExtractor.from_pretrained(path, **kw)
            model = ASTForAudioClassification.from_pretrained(path, dtype=torch.float32, **kw)
            top = model.audio_spectrogram_transformer
        elif role == "speech":
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            proc = WhisperProcessor.from_pretrained(path, **kw)
            model = WhisperForConditionalGeneration.from_pretrained(path, dtype=torch.float32, **kw)
            top = model.model.encoder
        else:
            raise KeyError(role)
        model = model.to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        parts = used or [model]
        params = int(sum(p.numel() for part in parts for p in part.parameters()))   # parameters actually executed
        ld = Loaded(role, model, proc, params, dev, weights_sha256=weights_file_sha256(role),
                    extra={"params_loaded": int(sum(p.numel() for p in model.parameters()))})
        top.register_forward_hook(_count_forward(ld))
        _LOADED[role] = ld
        return ld


def loaded_roles() -> list[str]:
    return sorted(_LOADED)


def forward_passes() -> dict[str, int]:
    """Forward passes of each loaded model's main network since it was loaded in this process."""
    return {r: ld.forward_passes for r, ld in sorted(_LOADED.items())}


def unload(role: str | None = None) -> None:
    """Drop loaded models (frees GPU memory)."""
    with _LOCK:
        for r in ([role] if role else list(_LOADED)):
            _LOADED.pop(r, None)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def l2n(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x / max(n, eps)
