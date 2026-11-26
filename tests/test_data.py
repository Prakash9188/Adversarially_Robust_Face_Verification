"""Tests for the pair protocol using a tiny synthetic LFW-like folder (no downloads)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_images, make_pairs, scan_lfw


def _fake_lfw(root, n_people=60, per=3):
    rng = np.random.default_rng(0)
    for i in range(n_people):
        d = Path(root) / "lfw_funneled" / f"Person_{i:03d}"
        d.mkdir(parents=True)
        for j in range(per):
            Image.fromarray(rng.integers(0, 255, (250, 250, 3), dtype=np.uint8)).save(d / f"Person_{i:03d}_{j+1:04d}.jpg")


def test_pairs_are_identity_disjoint_balanced_and_deterministic(tmp_path):
    _fake_lfw(tmp_path)
    counts = {"train": 20, "val": 10, "test": 10}
    a = make_pairs(tmp_path, [0.7, 0.15, 0.15], counts, seed=0)
    b = make_pairs(tmp_path, [0.7, 0.15, 0.15], counts, seed=0)
    assert a == b                                           # reproducible
    ids = {s: {p[3] for p in v} | {p[4] for p in v} for s, v in a.items()}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    for s, v in a.items():
        assert len(v) == counts[s]
        assert sum(p[2] for p in v) == counts[s] // 2       # balanced genuine/impostor
        for p in v:
            assert (p[3] == p[4]) == (p[2] == 1)            # label matches identity
            assert p[0] != p[1]


def test_load_images_shape_dtype(tmp_path):
    _fake_lfw(tmp_path, n_people=2)
    paths = [v[0] for v in scan_lfw(tmp_path).values()]
    x = load_images(tmp_path, paths, crop=150, size=160)
    assert x.shape == (2, 3, 160, 160) and x.dtype.is_floating_point is False
