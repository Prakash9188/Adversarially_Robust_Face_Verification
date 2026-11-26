"""Bayes minimum-risk fusion: verification score + detector posterior -> Same/Different/Flag.

Minimum-risk rule: flag the pair if  P(adv|z) * C_miss  >  P(clean|z) * C_false_alarm,
i.e. P(adv|z) >= tau_bayes = C_fa / (C_fa + C_miss).
In practice we also report an operating point with a fixed false-alarm rate on clean
validation pairs (`tau_at_fpr`), which is easier to interpret.
Decisions: 1 = same, 0 = different, -1 = flagged (reject / manual review).
"""
import numpy as np

FLAG = -1


def bayes_tau(c_miss, c_fa):
    return c_fa / (c_fa + c_miss)


def tau_at_fpr(p_clean, fpr):
    """Smallest threshold such that at most `fpr` of clean pairs have p > tau."""
    return float(np.quantile(p_clean, 1 - fpr, method="higher"))


def decide(scores, thr, p_adv, tau):
    out = (scores >= thr).astype(int)
    out[p_adv > tau] = FLAG
    return out


def clean_stats(scores, labels, thr, p_clean, tau):
    d = decide(scores, thr, p_clean, tau)
    y = labels.astype(int)
    return {"clean_flag_rate": float((d == FLAG).mean()),
            "clean_error_rate": float(((d != FLAG) & (d != y)).mean())}


def fused_asr(clean_scores, adv_scores, labels, thr, p_adv, tau):
    """Among pairs correct when clean (and unattacked), fraction that end up WRONG and
    NOT flagged after the attack. Also returns the detection (flag) rate on attacked pairs."""
    y = labels.astype(int)
    ok = (clean_scores >= thr).astype(int) == y
    d = decide(adv_scores, thr, p_adv, tau)
    bad = (d != FLAG) & (d != y)
    return {"ASR_fused": float((ok & bad).sum() / max(ok.sum(), 1)),
            "adv_flag_rate": float((d == FLAG).mean())}
