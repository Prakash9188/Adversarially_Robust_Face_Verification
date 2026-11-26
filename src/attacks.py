"""Gradient attacks on a verification pair (anchor, probe).

Only the probe image is perturbed (L_inf ball of radius eps, pixels in [0,1]).
  * genuine pair  (label 1): DODGING       -> push similarity DOWN
  * impostor pair (label 0): IMPERSONATION -> push similarity UP
We maximise  L = sign * s,  sign = +1 (impostor), -1 (genuine),  s = cos-sim.
"""
import torch


def loss_sign(labels):
    return 1.0 - 2.0 * labels.float()


def pgd(model, anchor_emb, x, labels, eps, steps=10, alpha=None, random_start=True):
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    sign = loss_sign(labels).to(x.device)
    x_adv = x.clone()
    if random_start:
        x_adv = (x_adv + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        s = (model(x_adv) * anchor_emb).sum(dim=1)
        grad, = torch.autograd.grad((sign * s).sum(), x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    return x_adv.detach()


def fgsm(model, anchor_emb, x, labels, eps):
    """FGSM == one PGD step of full size, no random start."""
    return pgd(model, anchor_emb, x, labels, eps, steps=1, alpha=eps, random_start=False)


def quantize8(x):
    """Real images are 8-bit; make adversarial images valid 8-bit images."""
    return torch.round(x * 255.0) / 255.0


def make_attacked(model, x1, x2, labels, attack, eps, steps=10, batch_size=16, cache_path=None):
    """Attack the probe x2 of each pair (x1 = anchor). x1,x2 are uint8; eps is in /255
    units. Returns float [0,1] 8-bit-quantised adversarial probes. Cached on disk."""
    from pathlib import Path
    if cache_path is not None and Path(cache_path).exists():
        return torch.load(cache_path)
    eps = eps / 255.0
    outs = []
    for i in range(0, len(x1), batch_size):
        a = x1[i:i + batch_size].float() / 255.0
        p = x2[i:i + batch_size].float() / 255.0
        y = labels[i:i + batch_size]
        with torch.no_grad():
            anchor = model(a)
        if attack == "fgsm":
            adv = fgsm(model, anchor, p, y, eps)
        elif attack == "pgd":
            adv = pgd(model, anchor, p, y, eps, steps=steps)
        else:
            raise ValueError(attack)
        outs.append(quantize8(adv))
        print(f"  {attack} eps={eps*255:.0f}: {min(i + batch_size, len(x1))}/{len(x1)}", end="\r")
    print()
    res = torch.cat(outs)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, cache_path)
    return res
