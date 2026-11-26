"""Lightweight input-transformation defenses. All take/return a float tensor
(B,3,H,W) in [0,1]."""
import io

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def identity(x):
    return x


def jpeg(quality):
    """Real JPEG encode/decode (non-differentiable: blocks naive gradients)."""
    def _f(x):
        out = []
        for img in x.detach().cpu():
            arr = (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            dec = np.array(Image.open(buf).convert("RGB"))
            out.append(torch.from_numpy(dec).permute(2, 0, 1).float() / 255)
        return torch.stack(out).to(x.device)
    return _f


def gaussian(sigma=1.0, kernel_size=5):
    def _f(x):
        return TF.gaussian_blur(x, [kernel_size] * 2, [sigma] * 2)
    return _f


def median(k=3):
    """k x k median filter (replaces each pixel by the median of its window)."""
    pad = k // 2

    def _f(x):
        xp = F.pad(x, (pad,) * 4, mode="reflect")
        patches = xp.unfold(2, k, 1).unfold(3, k, 1)          # B,C,H,W,k,k
        flat = patches.reshape(*patches.shape[:4], k * k)
        return flat.median(dim=-1).values
    return _f


def bit_depth(bits=4):
    levels = 2 ** bits - 1
    return lambda x: torch.round(x * levels) / levels


REGISTRY = {
    "none": identity,
    "jpeg75": jpeg(75),
    "jpeg50": jpeg(50),
    "jpeg30": jpeg(30),
    "gaussian": gaussian(1.0, 5),
    "median3": median(3),
    "median5": median(5),
    "bitdepth4": bit_depth(4),
}


def get_defense(name):
    return REGISTRY[name]
