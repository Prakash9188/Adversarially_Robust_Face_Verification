"""Verify one face pair, optionally attack it, and show the consistency cues.

    python scripts/demo.py IMG1 IMG2 --same 1            # clean score + decision
    python scripts/demo.py IMG1 IMG2 --same 1 --attack   # also run PGD (eps=4) on IMG2

--same is the ground truth (1 = same person); the attacker needs it to choose the direction.
Threshold 0.455 is the validation-fitted threshold from our baseline run (configs/default.yaml data).
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attacks import pgd, quantize8
from src.defenses import get_defense
from src.features import DET_TRANSFORMS
from src.model import FaceEmbedder

THR = 0.455


def load(path, crop=150, size=160):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    l, t = (w - crop) // 2, (h - crop) // 2
    im = im.crop((l, t, l + crop, t + crop)).resize((size, size), Image.BILINEAR)
    return torch.from_numpy(__import__("numpy").asarray(im).copy()).permute(2, 0, 1).float().div(255)[None]


def report(model, a, p, title):
    with torch.no_grad():
        ea, ep = model(a), model(p)
        s0 = float((ea * ep).sum())
        print(f"\n{title}: score={s0:.3f} -> {'SAME' if s0 >= THR else 'DIFFERENT'} (thr {THR})")
        for t in DET_TRANSFORMS:
            f = get_defense(t)
            st = float((model(f(a)) * model(f(p))).sum())
            drift = float((ep * model(f(p))).sum())
            print(f"   {t:10s} score={st:.3f}  probe embedding drift cos={drift:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img1"); ap.add_argument("img2")
    ap.add_argument("--same", type=int, required=True, choices=[0, 1])
    ap.add_argument("--attack", action="store_true")
    ap.add_argument("--eps", type=float, default=4)
    args = ap.parse_args()
    model = FaceEmbedder()
    a, p = load(args.img1), load(args.img2)
    report(model, a, p, "CLEAN")
    if args.attack:
        with torch.no_grad():
            anchor = model(a)
        adv = quantize8(pgd(model, anchor, p, torch.tensor([args.same]), args.eps / 255))
        report(model, a, adv, f"ATTACKED (PGD eps={args.eps}/255)")


if __name__ == "__main__":
    main()
