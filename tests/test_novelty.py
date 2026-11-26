"""Fast tests for Commit-2 components (no network / pretrained weights)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attacks import cw_l2, deepfool, pgd_bpda, square_attack, ste
from src.defenses import jpeg, median
from src.detector import make_detector, posterior
from src.features import consistency_features
from src.fusion import FLAG, bayes_tau, decide, fused_asr, tau_at_fpr


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(3 * 8 * 8, 16)

    def forward(self, x):
        return nn.functional.normalize(self.fc(x.flatten(1)), dim=1)


EPS = 8 / 255


def _setup():
    m = Tiny()
    x = torch.rand(4, 3, 8, 8)
    y = torch.tensor([1, 0, 1, 0])
    anchor = m(torch.rand(4, 3, 8, 8)).detach()   # real pipeline computes anchors under no_grad
    return m, x, y, anchor


@pytest.mark.parametrize("name", ["deepfool", "cw", "square"])
def test_new_attacks_respect_budget(name):
    m, x, y, anchor = _setup()
    if name == "deepfool":
        adv = deepfool(m, anchor, x, y, EPS, thr=0.3, steps=3)
    elif name == "cw":
        adv = cw_l2(m, anchor, x, y, EPS, thr=0.3, steps=3)
    else:
        adv = square_attack(m, anchor, x, y, EPS, queries=5)
    assert (adv - x).abs().max() <= EPS + 1e-5
    assert adv.min() >= 0 and adv.max() <= 1


def test_ste_forward_equals_defense_backward_identity():
    x = torch.rand(1, 3, 8, 8, requires_grad=True)
    f = ste(median(3))
    assert torch.allclose(f(x), median(3)(x.detach()), atol=1e-6)
    f(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_bpda_through_jpeg_runs_and_respects_budget():
    m, x, y, anchor = _setup()
    x = torch.rand(4, 3, 16, 16)
    class T16(Tiny):
        def __init__(self):
            super().__init__(); self.fc = nn.Linear(3 * 16 * 16, 16)
    m = T16(); anchor = m(torch.rand(4, 3, 16, 16))
    adv = pgd_bpda(m, [anchor], x, y, EPS, [jpeg(75)], steps=2)
    assert (adv - x).abs().max() <= EPS + 1e-5


def test_features_shape_and_detector_separates_synthetic():
    n = 50
    A = {k: nn.functional.normalize(torch.randn(n, 8), dim=1) for k in
         ["raw", "jpeg75", "jpeg30", "median3", "gaussian", "bitdepth4"]}
    P = {k: v.clone() for k, v in A.items()}
    Z = consistency_features(A, P)
    assert Z.shape == (n, 1 + 2 * 5)
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, (100, 11))
    adv = rng.normal(3, 1, (100, 11))
    X = np.concatenate([clean, adv]); y = np.r_[np.zeros(100), np.ones(100)]
    for kind in ["lda", "qda", "svm", "gboost"]:
        det = make_detector(kind).fit(X, y)
        p = posterior(det, X)
        assert p[:100].mean() < 0.3 and p[100:].mean() > 0.7


def test_fusion_logic():
    assert bayes_tau(5, 1) == pytest.approx(1 / 6)
    p_clean = np.linspace(0, 1, 101)
    assert tau_at_fpr(p_clean, 0.05) >= 0.94
    s = np.array([0.9, 0.1]); p = np.array([0.0, 0.9])
    d = decide(s, 0.5, p, 0.5)
    assert d.tolist() == [1, FLAG]
    y = np.array([1, 0])
    r = fused_asr(np.array([0.9, 0.1]), np.array([0.1, 0.9]), y, 0.5, np.array([0.9, 0.0]), 0.5)
    # pair 0: flagged -> not a success; pair 1: impostor accepted, unflagged -> success
    assert r["ASR_fused"] == 0.5 and r["adv_flag_rate"] == 0.5


def test_detector_aware_bpda_runs_and_respects_budget():
    class T16(Tiny):
        def __init__(self):
            super().__init__(); self.fc = nn.Linear(3 * 16 * 16, 16)
    m = T16()
    x = torch.rand(4, 3, 16, 16); y = torch.tensor([1, 0, 1, 0])
    ts = [lambda z: z, median(3)]
    anchors = [m(torch.rand(4, 3, 16, 16)).detach() for _ in ts]
    adv = pgd_bpda(m, anchors, x, y, EPS, ts, steps=2, consistency_weight=1.0)
    assert (adv - x).abs().max() <= EPS + 1e-5
