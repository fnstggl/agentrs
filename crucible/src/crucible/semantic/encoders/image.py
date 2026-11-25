"""Image encoder: CLIP ViT-B/32 image features (projected pooled vision output), L2-normalised, float32 [512].

Preprocessing: decode with Pillow under a pixel limit (first frame of animations, alpha composited on white), pad to a
square with the mean colour of the image border (so wide banners and rendered text are not cropped away), resize to
224x224 (bicubic), normalise with CLIP's channel mean and standard deviation.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .. import media
from . import models as M
from .base import Encoder, register

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SIZE = 224


def square_pixels(im) -> np.ndarray:
    """PIL RGB image -> float32 [3, 224, 224] CLIP input (pad to square with the border's mean colour)."""
    from PIL import Image
    a = np.asarray(im, dtype=np.uint8)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0).astype(np.float32)
    fill = tuple(int(round(v)) for v in border.mean(axis=0))
    side = max(im.width, im.height)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    canvas = canvas.resize((SIZE, SIZE), Image.BICUBIC)
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    x = (x - CLIP_MEAN) / CLIP_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


class ImageEncoder(Encoder):
    modality = "image"
    id = "crucible.image.clip-vit-b32"
    roles = ("image",)
    dim = 512
    batch_size = 32
    preprocessing = {"decode": "pillow rgb, first frame, alpha on white", "max_pixels": media.LIMITS.max_pixels,
                     "canvas": "pad to square with mean border colour", "size": SIZE, "resample": "bicubic",
                     "mean": CLIP_MEAN.tolist(), "std": CLIP_STD.tolist(), "features": "visual_projection(pooler)",
                     "norm": "l2"}

    def embed(self, items: Sequence[bytes]) -> np.ndarray:
        return self.embed_images([media.open_image(b) for b in items])

    def embed_images(self, images: Sequence) -> np.ndarray:
        """[n, 512] for PIL images or uint8 HxWx3 arrays."""
        import torch
        from PIL import Image
        ld = M.load("image")
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for s in range(0, len(images), self.batch_size):
            chunk = images[s:s + self.batch_size]
            px = np.stack([square_pixels(Image.fromarray(im) if isinstance(im, np.ndarray) else im) for im in chunk])
            with torch.inference_mode():
                pv = torch.from_numpy(px).to(ld.device)
                pooled = ld.model.vision_model(pixel_values=pv).pooler_output
                feats = torch.nn.functional.normalize(ld.model.visual_projection(pooled), dim=-1)
            out[s:s + len(chunk)] = feats.float().cpu().numpy()
        return out


IMAGE = register(ImageEncoder())
