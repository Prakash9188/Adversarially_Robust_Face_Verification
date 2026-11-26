"""Transformation-consistency features for a verification pair.

For every transform t in the bank we embed both images before and after t. A clean pair
keeps a stable score; an adversarial probe tuned to the raw model tends to be brittle.

Feature vector for a pair (anchor A, probe P), T transforms:
    z = [ s0,  (s_t - s0,  1 - min(cos(A, t(A)), cos(P, t(P))))  for t in bank ]
with s0 = cos(f(A), f(P)) and s_t = cos(f(t(A)), f(t(P))). Dimension 1 + 2T.
"""
import torch

from .defenses import get_defense
from .model import embed

DET_TRANSFORMS = ["jpeg75", "jpeg30", "median3", "gaussian", "bitdepth4"]
BANK_NAMES = ["raw"] + DET_TRANSFORMS + ["median5"]   # median5 is only used as a baseline defense


def bank(model, xf, names=BANK_NAMES, batch_size=32):
    """Embeddings of a float image tensor under each transform: {name: (N,512)}."""
    return {n: embed(model, xf, None if n == "raw" else get_defense(n), batch_size) for n in names}


def cos(a, b):
    return (a * b).sum(1)


def consistency_features(A, P, det_names=DET_TRANSFORMS):
    s0 = cos(A["raw"], P["raw"])
    cols = [s0]
    for t in det_names:
        st = cos(A[t], P[t])
        d1 = cos(A["raw"], A[t])
        d2 = cos(P["raw"], P[t])
        cols += [st - s0, 1 - torch.minimum(d1, d2)]
    return torch.stack(cols, 1).numpy()


def feature_names(det_names=DET_TRANSFORMS):
    n = ["s0"]
    for t in det_names:
        n += [f"ds_{t}", f"shift_{t}"]
    return n
