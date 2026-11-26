"""Commit-2 experiments: consistency detector + Bayes fusion vs. non-adaptive and adaptive attacks.

    python scripts/run_detector.py [--quick]

Protocol
  train split (first n_train pairs): fit detector on clean vs {FGSM, PGD} (eps=4)  [SEEN attacks]
  val   split (first n_val pairs)  : per-defense thresholds, detector choice, tau at fixed FPR
  test  split (first n_attack_pairs): evaluate on SEEN (FGSM, PGD) and UNSEEN (DeepFool, CW,
                                      Square) attacks, then ADAPTIVE attacks.
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
from src.attacks import make_adaptive, make_attacked
from src.data import download_lfw, get_split, to_float
from src.defenses import get_defense
from src.detector import make_detector, posterior, select_features
from src.features import DET_TRANSFORMS, bank, consistency_features, cos, feature_names
from src.fusion import bayes_tau, clean_stats, fused_asr, tau_at_fpr
from src.model import FaceEmbedder

OUT = Path("experiments/results")
EPS = 4
BASE_DEFS = ["none", "jpeg30", "median5"]   # defenses compared against the fused system


def cache(cfg, split, name, n):
    return Path(cfg["data"]["root"]) / "attacks" / f"{split}_{name}_n{n}_s{cfg['seed']}.pt"


def attack_set(cfg, model, split_data, split, atk, n, thr, quick, eps=EPS):
    a = cfg["attack"]
    x1, x2, y = (t[:n] for t in split_data)
    return make_attacked(model, x1, x2, y, atk, eps, a["pgd_steps"] if atk != "deepfool" else a["deepfool_steps"],
                         16, cache(cfg, split, f"{atk}_eps{eps}", n), thr=thr,
                         queries=a["square_queries"], cw_steps=a["cw_steps"])


def tpr_at_fpr(p_clean, p_adv, fpr=0.05):
    tau = np.quantile(p_clean, 1 - fpr, method="higher")
    return float((p_adv > tau).mean())


def det_auc(p_clean, p_adv):
    return M.auc(np.concatenate([p_clean, p_adv]), np.concatenate([np.zeros(len(p_clean)), np.ones(len(p_adv))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    d, a = cfg["detector"], cfg["attack"]
    n_tr, n_va, n_te, n_ad = d["n_train"], d["n_val"], a["n_attack_pairs"], a["n_adaptive"]
    if args.quick:
        cfg["data"]["pairs"] = {"train": 40, "val": 40, "test": 40}
        n_tr, n_va, n_te, n_ad = 24, 24, 24, 12
        a["pgd_steps"], a["cw_steps"], a["square_queries"], a["deepfool_steps"] = 3, 3, 10, 3
    tag = "_quick" if args.quick else ""
    OUT.mkdir(parents=True, exist_ok=True)
    download_lfw(cfg["data"]["root"])
    model = FaceEmbedder(cfg["model"]["pretrained"])
    tr, va, te = (get_split(cfg, s) for s in ("train", "val", "test"))
    bs = cfg["batch_size"]

    # ------------------------------------------------------------------ banks
    def anchor_bank(split_data, n):
        return bank(model, to_float(split_data[0][:n]), batch_size=bs)

    def probe_bank(xf):
        return bank(model, xf, batch_size=bs)

    # ---- val: thresholds per defense (clean), needed for baselines and DeepFool/CW
    Av = anchor_bank(va, n_va)
    Pv = probe_bank(to_float(va[1][:n_va]))
    yv = va[2][:n_va].numpy()
    thr = {dn: M.best_threshold(cos(Av["raw" if dn == "none" else dn], Pv["raw" if dn == "none" else dn]).numpy(), yv)
           for dn in BASE_DEFS}
    print("thresholds:", thr)

    # ---- train/val attacked sets (SEEN attacks: fgsm, pgd)
    seen = ["fgsm", "pgd"]
    Atr = anchor_bank(tr, n_tr)
    Ptr = probe_bank(to_float(tr[1][:n_tr]))
    Ztr_clean = consistency_features(Atr, Ptr)
    Ztr_adv = []
    for atk in seen:
        adv = attack_set(cfg, model, tr, "train", atk, n_tr, thr["none"], args.quick)
        Ztr_adv.append(consistency_features(Atr, probe_bank(adv)))
    Xtr = np.concatenate([Ztr_clean, Ztr_clean] + Ztr_adv)          # clean repeated x2 -> balanced
    ytr = np.concatenate([np.zeros(2 * n_tr)] + [np.ones(n_tr)] * len(seen))

    Zv_clean = consistency_features(Av, Pv)
    Zv_adv = []
    for atk in seen:
        adv = attack_set(cfg, model, va, "val", atk, n_va, thr["none"], args.quick)
        Zv_adv.append(consistency_features(Av, probe_bank(adv)))
    Zv_adv = np.concatenate(Zv_adv)

    # ---- choose detector on VAL (AUC on seen attacks)
    dets, val_rows = {}, []
    for kind in d["kinds"]:
        det = make_detector(kind, cfg["seed"]).fit(Xtr, ytr)
        dets[kind] = det
        val_rows.append({"detector": kind, "val_AUC": det_auc(posterior(det, Zv_clean), posterior(det, Zv_adv))})
    val_df = pd.DataFrame(val_rows)
    print(val_df.round(3).to_string(index=False))
    best = val_df.sort_values("val_AUC", ascending=False).iloc[0]["detector"]
    det = dets[best]
    tau = tau_at_fpr(posterior(det, Zv_clean), d["fpr"])
    print(f"chosen detector: {best}; tau@FPR={d['fpr']}: {tau:.4f}; Bayes tau (costs {d['c_miss']}:{d['c_fa']}): "
          f"{bayes_tau(d['c_miss'], d['c_fa']):.4f}")
    val_df.to_csv(OUT / f"detector_val{tag}.csv", index=False)

    # ------------------------------------------------------------------ test
    yt = te[2][:n_te].numpy()
    At = anchor_bank(te, n_te)
    Pt = probe_bank(to_float(te[1][:n_te]))
    Zt_clean = consistency_features(At, Pt)
    pc = {k: posterior(m, Zt_clean) for k, m in dets.items()}

    sets = {}   # name -> (probe bank, n)
    for atk in ["fgsm", "pgd", "deepfool", "cw", "square"]:
        thr_atk = thr["none"]
        adv = attack_set(cfg, model, te, "test", atk, n_te, thr_atk, args.quick)
        sets[atk] = (probe_bank(adv), n_te)

    # ---- BENIGN CONTROLS: perturbed but NOT adversarial. Flag rate here = false alarms on
    # ordinary noisy / compressed photos (is the detector just a "noise detector"?)
    from src.attacks import quantize8
    g = torch.Generator().manual_seed(cfg["seed"])
    pf = to_float(te[1][:n_te])
    noise = quantize8((pf + (EPS / 255.0) * torch.sign(torch.randn(pf.shape, generator=g))).clamp(0, 1))
    sets["control_noise"] = (probe_bank(noise), n_te)
    sets["control_jpeg75"] = (probe_bank(quantize8(get_defense("jpeg75")(pf))), n_te)

    # ---- other budgets (detector was trained at eps=4 only)
    for atk in ["fgsm", "pgd"]:
        for e in (2, 8):
            adv = attack_set(cfg, model, te, "test", atk, n_te, thr["none"], args.quick, eps=e)
            sets[f"{atk}_eps{e}"] = (probe_bank(adv), n_te)

    # ---- ADAPTIVE attacks (fewer pairs: expensive)
    x1a, x2a, ya = te[0][:n_ad], te[1][:n_ad], te[2][:n_ad]
    for dn in ["jpeg30", "median5", "gaussian"]:
        adv = make_adaptive(model, x1a, x2a, ya, EPS, [get_defense(dn)], a["pgd_steps"], 8,
                            cache(cfg, "test", f"bpda_{dn}_eps{EPS}", n_ad))
        sets[f"bpda_{dn}"] = (probe_bank(adv), n_ad)
    ens = [get_defense("none")] + [get_defense(t) for t in DET_TRANSFORMS]
    adv = make_adaptive(model, x1a, x2a, ya, EPS, ens, a["pgd_steps"], 4, cache(cfg, "test", f"adaptive_ens_eps{EPS}", n_ad))
    sets["adaptive_ens"] = (probe_bank(adv), n_ad)
    # attacker that ALSO keeps the embedding stable under the transforms (detector-aware)
    adv = make_adaptive(model, x1a, x2a, ya, EPS, ens, a["pgd_steps"], 4, cache(cfg, "test", f"adaptive_aware_eps{EPS}", n_ad),
                        consistency_weight=1.0)
    sets["adaptive_aware"] = (probe_bank(adv), n_ad)

    # ---- tables
    det_rows, asr_rows = [], []
    for name, (P, n) in sets.items():
        Z = consistency_features({k: v[:n] for k, v in At.items()}, P)
        yy = yt[:n]
        row_kind = ("seen" if name in seen else "adaptive" if name.startswith(("bpda", "adaptive"))
                    else "control" if name.startswith("control") else "other-eps" if "_eps" in name else "unseen")
        for kind, m in dets.items():
            p_adv = posterior(m, Z)
            det_rows.append({"attack": name, "type": row_kind, "detector": kind,
                             "AUC": det_auc(pc[kind], p_adv), "TPR@5%FPR": tpr_at_fpr(pc[kind], p_adv)})
        # ASR of single defenses
        p_adv_best = posterior(det, Z)
        for dn in BASE_DEFS:
            key = "raw" if dn == "none" else dn
            sc = cos(At[key][:n], Pt[key][:n]).numpy()
            sa = cos(At[key][:n], P[key]).numpy()
            asr_rows.append({"attack": name, "type": row_kind, "system": dn,
                             "ASR": M.attack_success_rate(sc, sa, yy, thr[dn]), "flag_rate": 0.0})
        sc = cos(At["raw"][:n], Pt["raw"][:n]).numpy()
        sa = cos(At["raw"][:n], P["raw"]).numpy()
        f = fused_asr(sc, sa, yy, thr["none"], p_adv_best, tau)
        asr_rows.append({"attack": name, "type": row_kind, "system": f"fused({best})",
                         "ASR": f["ASR_fused"], "flag_rate": f["adv_flag_rate"]})
    det_df, asr_df = pd.DataFrame(det_rows), pd.DataFrame(asr_rows)
    sc_clean = cos(At["raw"], Pt["raw"]).numpy()
    cs = clean_stats(sc_clean, yt, thr["none"], pc[best], tau)
    print("\nDETECTOR (test):\n", det_df.round(3).to_string(index=False))
    print("\nASR (test):\n", asr_df.round(3).to_string(index=False))
    print("\nclean cost of fusion:", cs)
    det_df.to_csv(OUT / f"detector_test{tag}.csv", index=False)
    asr_df.to_csv(OUT / f"fusion_asr{tag}.csv", index=False)
    pd.DataFrame([{**cs, "detector": best, "tau": tau, "clean_acc_no_flag": M.accuracy(sc_clean, yt, thr["none"])}]).to_csv(
        OUT / f"fusion_clean{tag}.csv", index=False)

    # ---- ablation: one transform at a time (LDA) + SFFS selection
    names = feature_names()
    abl = []
    for t in DET_TRANSFORMS:
        cols = [0] + [i for i, nm in enumerate(names) if nm.endswith("_" + t)]
        m = make_detector("lda").fit(Xtr[:, cols], ytr)
        aucs = [det_auc(posterior(m, Zt_clean[:, cols]),
                        posterior(m, consistency_features({k: v[:n] for k, v in At.items()}, sets[s][0])[:, cols]))
                for s in ["fgsm", "pgd", "deepfool", "cw", "square"] for n in [sets[s][1]]]
        abl.append({"features": f"s0 + {t}", "mean_AUC_nonadaptive": float(np.mean(aucs))})
    all_aucs = det_df[(det_df.detector == "lda") & det_df.type.isin(["seen", "unseen"])].AUC.mean()
    abl.append({"features": "all transforms (LDA)", "mean_AUC_nonadaptive": float(all_aucs)})
    sel = select_features(Xtr, ytr, 4)
    abl.append({"features": "SFFS top-4: " + ", ".join(names[i] for i in sel), "mean_AUC_nonadaptive": np.nan})
    pd.DataFrame(abl).to_csv(OUT / f"ablation{tag}.csv", index=False)
    print(pd.DataFrame(abl).round(3).to_string(index=False))

    # ---- timing (ms per pair): single embedding vs. full detector features
    import time
    xs = to_float(te[0][:16])
    t0 = time.time(); bank(model, xs, ["raw"], 16); bank(model, xs, ["raw"], 16); t1 = time.time()
    bank(model, xs, batch_size=16); bank(model, xs, batch_size=16); t2 = time.time()
    pd.DataFrame([{"baseline_ms_per_pair": 1000 * (t1 - t0) / 16, "detector_ms_per_pair": 1000 * (t2 - t1) / 16}]).to_csv(
        OUT / f"timing{tag}.csv", index=False)

    # figures/tables for the report are built from the CSVs by scripts/make_report_assets.py
    print("done")


if __name__ == "__main__":
    main()
