"""Fast unit tests: no network, no pretrained weights."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics as M
from src.attacks import fgsm, loss_sign, pgd, quantize8
from src.defenses import REGISTRY, median


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(3 * 8 * 8, 16)

    def forward(self, x):
        return nn.functional.normalize(self.fc(x.flatten(1)), dim=1)


def test_defenses_shape_and_range():
    x = torch.rand(2, 3, 32, 32)
    for name, d in REGISTRY.items():
        y = d(x)
        assert y.shape == x.shape, name
        assert y.min() >= -1e-6 and y.max() <= 1 + 1e-6, name


def test_median_matches_numpy():
    x = torch.rand(1, 1, 6, 6)
    y = median(3)(x)[0, 0]
    xp = np.pad(x[0, 0].numpy(), 1, mode="reflect")
    ref = np.array([[np.median(xp[i:i + 3, j:j + 3]) for j in range(6)] for i in range(6)])
    assert np.allclose(y.numpy(), ref, atol=1e-6)


def test_metrics_perfect_and_threshold():
    s = np.array([0.9, 0.8, 0.1, 0.0])
    y = np.array([1, 1, 0, 0])
    thr = M.best_threshold(s, y)
    assert M.accuracy(s, y, thr) == 1.0
    assert M.auc(s, y) == 1.0
    assert M.tar_at_far(s, y, 0.01) == 1.0
    assert 0.1 < M.bayes_gaussian_threshold(s, y) < 0.9


def test_asr_definition():
    y = np.array([1, 0])
    clean = np.array([0.9, 0.1])
    adv = np.array([0.1, 0.1])        # first pair flipped, second unchanged
    assert M.attack_success_rate(clean, adv, y, 0.5) == 0.5


@pytest.mark.parametrize("attack", [fgsm, pgd])
def test_attacks_respect_budget_and_hurt(attack):
    m, eps = Tiny(), 8 / 255
    x = torch.rand(4, 3, 8, 8)
    labels = torch.tensor([1, 0, 1, 0])
    anchor = m(torch.rand(4, 3, 8, 8))
    adv = attack(m, anchor, x, labels, eps)
    assert (adv - x).abs().max() <= eps + 1e-6
    assert adv.min() >= 0 and adv.max() <= 1
    sign = loss_sign(labels)
    before = (sign * (m(x) * anchor).sum(1)).sum()
    after = (sign * (m(adv) * anchor).sum(1)).sum()
    assert after >= before - 1e-6


def test_quantize8():
    q = quantize8(torch.rand(10) )
    assert torch.allclose(q * 255, torch.round(q * 255))
