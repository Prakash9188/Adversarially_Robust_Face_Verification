"""Frozen pretrained CNN used purely as a feature extractor (transfer learning).

Input : float tensor (B,3,160,160) in [0,1]  (so attacks work in pixel space)
Output: L2-normalised 512-d embedding. For unit vectors, dot product == cosine
        similarity, which is our verification score.
"""
import torch
import torch.nn as nn


class FaceEmbedder(nn.Module):
    def __init__(self, pretrained="vggface2"):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained=pretrained).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = (x * 255.0 - 127.5) / 128.0          # facenet's fixed standardization
        e = self.net(x)
        return nn.functional.normalize(e, dim=1)


@torch.no_grad()
def embed(model, x, defense=None, batch_size=32):
    """Embed a (N,3,H,W) tensor in batches, optionally after a defense."""
    outs = []
    for i in range(0, len(x), batch_size):
        b = x[i:i + batch_size]
        if defense is not None:
            b = defense(b)
        outs.append(model(b))
    return torch.cat(outs)


def score(e1, e2):
    """Cosine similarity of already-normalised embeddings."""
    return (e1 * e2).sum(dim=1)
