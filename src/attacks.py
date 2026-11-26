"""Attacks on a verification pair (anchor, probe).

Only the probe image is perturbed (L_inf ball of radius eps, pixels in [0,1]).
  * genuine pair  (label 1): DODGING       -> push similarity DOWN
  * impostor pair (label 0): IMPERSONATION -> push similarity UP
We maximise  L = sign * s,  sign = +1 (impostor), -1 (genuine),  s = cos-sim.

White-box: FGSM, PGD, DeepFool (L2 steps, clipped to L_inf), CW-L2 (clipped to L_inf).
Black-box (score only): Square attack.
Adaptive: PGD with BPDA / straight-through gradients through the defenses (`pgd_bpda`).
"""
import random
from pathlib import Path

import torch


def loss_sign(labels):
    return 1.0 - 2.0 * labels.float()


def _project(x_adv, x, eps):
    return torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)


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
        x_adv = _project(x_adv.detach() + alpha * grad.sign(), x, eps)
    return x_adv.detach()


def fgsm(model, anchor_emb, x, labels, eps):
    """FGSM == one PGD step of full size, no random start."""
    return pgd(model, anchor_emb, x, labels, eps, steps=1, alpha=eps, random_start=False)


def deepfool(model, anchor_emb, x, labels, eps, thr, steps=10, overshoot=0.02):
    """DeepFool for a threshold decision: linearise s(x), take the minimal L2 step that
    crosses `thr`, then clip to the L_inf budget."""
    y = labels.bool().to(x.device)
    x_adv = x.clone()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        s = (model(x_adv) * anchor_emb).sum(dim=1)
        g, = torch.autograd.grad(s.sum(), x_adv)
        with torch.no_grad():
            done = torch.where(y, s < thr, s >= thr)
            gn = g.flatten(1).norm(dim=1) ** 2 + 1e-12
            gap = (s - thr).abs() + 1e-3
            direction = torch.where(y, -torch.ones_like(s), torch.ones_like(s))
            r = (direction * gap / gn).view(-1, 1, 1, 1) * g
            r = torch.where(done.view(-1, 1, 1, 1), torch.zeros_like(r), r)
            x_adv = _project(x_adv.detach() + (1 + overshoot) * r, x, eps)
    return x_adv.detach()


def cw_l2(model, anchor_emb, x, labels, eps, thr, steps=30, c=20.0, kappa=0.05, lr=0.02):
    """Carlini-Wagner style L2 attack (tanh space, Adam), clipped to the L_inf budget."""
    y = labels.bool().to(x.device)
    w = torch.atanh((x * 2 - 1).clamp(-1 + 1e-6, 1 - 1e-6)).clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        x_adv = 0.5 * (torch.tanh(w) + 1)
        s = (model(x_adv) * anchor_emb).sum(dim=1)
        adv_loss = torch.where(y, torch.relu(s - (thr - kappa)), torch.relu((thr + kappa) - s))
        l2 = ((x_adv - x) ** 2).flatten(1).sum(1)
        loss = (l2 + c * adv_loss).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return _project((0.5 * (torch.tanh(w) + 1)).detach(), x, eps)


@torch.no_grad()
def square_attack(model, anchor_emb, x, labels, eps, queries=100, p_init=0.05):
    """Black-box Square Attack (score-based, L_inf): random square patches of +/-eps,
    keep a patch only if the loss improves. Uses forward passes only."""
    sign = loss_sign(labels).to(x.device)
    B, C, H, W = x.shape
    stripes = torch.randint(0, 2, (B, C, 1, W), device=x.device).float() * 2 - 1
    x_adv = (x + eps * stripes.expand(B, C, H, W)).clamp(0, 1)
    best = sign * (model(x_adv) * anchor_emb).sum(1)
    for i in range(queries):
        p = p_init / (2 ** (i // max(1, queries // 5)))
        side = max(1, int(round((p * H * W) ** 0.5)))
        cand = x_adv.clone()
        for b in range(B):
            r0, c0 = random.randint(0, H - side), random.randint(0, W - side)
            val = (torch.randint(0, 2, (C, 1, 1), device=x.device).float() * 2 - 1) * eps
            cand[b, :, r0:r0 + side, c0:c0 + side] = (x[b, :, r0:r0 + side, c0:c0 + side] + val).clamp(0, 1)
        new = sign * (model(cand) * anchor_emb).sum(1)
        imp = new > best
        x_adv[imp] = cand[imp]
        best = torch.where(imp, new, best)
    return x_adv


def ste(fn):
    """Straight-through wrapper (BPDA with identity): forward = fn(x), backward = identity."""
    def _f(x):
        return x + (fn(x) - x).detach()
    return _f


def pgd_bpda(model, anchors, x, labels, eps, transforms, steps=10, alpha=None, consistency_weight=0.0):
    """ADAPTIVE PGD. The attacker knows the defenses. `transforms` is a list of callables
    and `anchors[k]` = embedding of transform_k(anchor). The loss averages the pair score
    over all transforms, using straight-through gradients for non-differentiable ones.
    DETECTOR-AWARE variant (consistency_weight > 0): `transforms[0]` must be the identity;
    the attacker additionally maximises cos(f(x), f(t(x))) for t != identity, i.e. tries to
    keep the embedding stable under the transforms (the cue our detector relies on)."""
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    sign = loss_sign(labels).to(x.device)
    fns = [ste(t) for t in transforms]
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        embs = [model(f(x_adv)) for f in fns]
        loss = sum((sign * (e * a).sum(1)).sum() for e, a in zip(embs, anchors)) / len(fns)
        if consistency_weight > 0 and len(embs) > 1:
            loss = loss + consistency_weight * sum((embs[0] * e).sum(1).sum() for e in embs[1:]) / (len(embs) - 1)
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = _project(x_adv.detach() + alpha * grad.sign(), x, eps)
    return x_adv.detach()


def quantize8(x):
    """Real images are 8-bit; make adversarial images valid 8-bit images."""
    return torch.round(x * 255.0) / 255.0


def _cached(cache_path):
    return cache_path is not None and Path(cache_path).exists()


def _save(res, cache_path):
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, cache_path)


def make_attacked(model, x1, x2, labels, attack, eps, steps=10, batch_size=16,
                  cache_path=None, thr=None, queries=100, cw_steps=30):
    """Attack the probe x2 of each pair (x1 = anchor). x1,x2 are uint8; eps is in /255
    units. Returns float [0,1] 8-bit-quantised adversarial probes. Cached on disk."""
    if _cached(cache_path):
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
        elif attack == "deepfool":
            adv = deepfool(model, anchor, p, y, eps, thr, steps=steps)
        elif attack == "cw":
            adv = cw_l2(model, anchor, p, y, eps, thr, steps=cw_steps)
        elif attack == "square":
            adv = square_attack(model, anchor, p, y, eps, queries=queries)
        else:
            raise ValueError(attack)
        outs.append(quantize8(adv))
        print(f"  {attack} eps={eps*255:.0f}: {min(i + batch_size, len(x1))}/{len(x1)}", end="\r")
    print()
    res = torch.cat(outs)
    _save(res, cache_path)
    return res


def make_adaptive(model, x1, x2, labels, eps, transforms, steps=10, batch_size=8, cache_path=None,
                  consistency_weight=0.0):
    """Adaptive PGD (BPDA / ensemble over `transforms`). x1,x2 uint8; eps in /255 units."""
    if _cached(cache_path):
        return torch.load(cache_path)
    eps = eps / 255.0
    outs = []
    for i in range(0, len(x1), batch_size):
        a = x1[i:i + batch_size].float() / 255.0
        p = x2[i:i + batch_size].float() / 255.0
        y = labels[i:i + batch_size]
        with torch.no_grad():
            anchors = [model(t(a)) for t in transforms]
        outs.append(quantize8(pgd_bpda(model, anchors, p, y, eps, transforms, steps,
                                      consistency_weight=consistency_weight)))
        print(f"  adaptive eps={eps*255:.0f}: {min(i + batch_size, len(x1))}/{len(x1)}", end="\r")
    print()
    res = torch.cat(outs)
    _save(res, cache_path)
    return res
