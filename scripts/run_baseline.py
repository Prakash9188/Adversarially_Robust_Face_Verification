"""Commit-1 pipeline: clean verification + FGSM/PGD vs. the poster's baseline defenses.

Threat model here is NON-ADAPTIVE: the attacker crafts examples on the undefended
model; the defense is then applied at test time to both images of the pair.

    python scripts/run_baseline.py [--config configs/default.yaml] [--quick]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics as M
from src.attacks import make_attacked
from src.data import download_lfw, get_split, to_float
from src.defenses import get_defense
from src.model import FaceEmbedder, embed, score

OUT = Path("experiments/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    bs = cfg["batch_size"]
    n_att = 24 if a.quick else cfg["attack"]["n_attack_pairs"]
    eps_list = [4] if a.quick else cfg["attack"]["eps"]
    if a.quick:
        cfg["data"]["pairs"] = {"train": 40, "val": 80, "test": 80}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if a.quick else ""

    download_lfw(cfg["data"]["root"])
    model = FaceEmbedder(cfg["model"]["pretrained"])
    val = get_split(cfg, "val")
    test = get_split(cfg, "test")
    yv, yt = val[2].numpy(), test[2].numpy()

    # ---- 1. clean verification per defense; thresholds fitted on VAL only ----
    thr, clean_rows = {}, []
    for dn in cfg["defenses"]:
        d = get_defense(dn)
        sv = score(embed(model, to_float(val[0]), d, bs), embed(model, to_float(val[1]), d, bs)).numpy()
        st = score(embed(model, to_float(test[0]), d, bs), embed(model, to_float(test[1]), d, bs)).numpy()
        thr[dn] = M.best_threshold(sv, yv)
        row = {"defense": dn, "thr": round(thr[dn], 4), "thr_bayes": round(M.bayes_gaussian_threshold(sv, yv), 4),
               "val_acc": M.accuracy(sv, yv, thr[dn]), "test_acc": M.accuracy(st, yt, thr[dn]),
               "test_auc": M.auc(st, yt)}
        for far in cfg["far_levels"]:
            row[f"TAR@FAR={far}"] = M.tar_at_far(st, yt, far)
        clean_rows.append(row)
        print(row)
    clean = pd.DataFrame(clean_rows)
    clean.to_csv(OUT / f"baseline_clean{tag}.csv", index=False)

    # ---- 2. attacks on the undefended model, evaluated through each defense ----
    x1, x2, y = test[0][:n_att], test[1][:n_att], test[2][:n_att]
    x1f = to_float(x1)
    rows = []
    for atk in ["fgsm", "pgd"]:
        for eps in eps_list:
            cp = Path(cfg["data"]["root"]) / "attacks" / f"test_{atk}_eps{eps}_n{n_att}_s{cfg['seed']}.pt"
            adv = make_attacked(model, x1, x2, y, atk, eps, cfg["attack"]["pgd_steps"], 16, cp)
            for dn in cfg["defenses"]:
                d = get_defense(dn)
                e1 = embed(model, x1f, d, bs)
                s_clean = score(e1, embed(model, to_float(x2), d, bs)).numpy()
                s_adv = score(e1, embed(model, adv, d, bs)).numpy()
                yy = y.numpy()
                rows.append({"attack": atk, "eps": eps, "defense": dn,
                             "clean_acc": M.accuracy(s_clean, yy, thr[dn]),
                             "adv_acc": M.accuracy(s_adv, yy, thr[dn]),
                             "ASR": M.attack_success_rate(s_clean, s_adv, yy, thr[dn])})
            print(pd.DataFrame(rows).query("attack==@atk and eps==@eps").round(3).to_string(index=False))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"baseline_attacks{tag}.csv", index=False)

    # ---- 3. figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sub = res[res.eps == max(eps_list)].pivot(index="defense", columns="attack", values="ASR").loc[cfg["defenses"]]
    sub.plot.bar(ax=ax)
    ax.set_ylabel("Attack success rate"); ax.set_title(f"Non-adaptive attacks, eps={max(eps_list)}/255")
    ax.set_ylim(0, 1); plt.xticks(rotation=30); plt.tight_layout()
    fig.savefig(OUT / f"baseline_asr{tag}.png", dpi=150)
    print("saved results to", OUT)


if __name__ == "__main__":
    main()
