"""Verification metrics. `scores` = cosine similarity, `labels` 1=same, 0=different."""
import numpy as np
from sklearn.metrics import roc_auc_score


def accuracy(scores, labels, thr):
    return float(((scores >= thr) == labels.astype(bool)).mean())


def best_threshold(scores, labels):
    """Minimum-error-rate threshold: maximise accuracy over candidate cut points."""
    s = np.sort(np.unique(scores))
    cands = np.concatenate([[s[0] - 1e-6], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-6]])
    accs = [accuracy(scores, labels, t) for t in cands]
    return float(cands[int(np.argmax(accs))])


def bayes_gaussian_threshold(scores, labels):
    """Fit N(mu,sigma^2) per class by Maximum Likelihood, equal priors, and return
    the point between the means where the two densities are equal."""
    g, i = scores[labels == 1], scores[labels == 0]
    mu1, s1, mu0, s0 = g.mean(), g.std() + 1e-9, i.mean(), i.std() + 1e-9
    xs = np.linspace(mu0, mu1, 2000)
    d = (-0.5 * ((xs - mu1) / s1) ** 2 - np.log(s1)) - (-0.5 * ((xs - mu0) / s0) ** 2 - np.log(s0))
    return float(xs[int(np.argmin(np.abs(d)))])


def tar_at_far(scores, labels, far):
    """True-accept-rate when the false-accept-rate is (at most) `far`."""
    imp = np.sort(scores[labels == 0])
    k = int(np.ceil((1 - far) * len(imp))) - 1
    thr = imp[min(max(k, 0), len(imp) - 1)]
    return float((scores[labels == 1] > thr).mean())


def auc(scores, labels):
    return float(roc_auc_score(labels, scores))


def attack_success_rate(clean_scores, adv_scores, labels, thr):
    """Among pairs the system got right when clean, fraction it gets wrong when attacked."""
    y = labels.astype(bool)
    ok = (clean_scores >= thr) == y
    flipped = (adv_scores >= thr) != y
    return float((ok & flipped).sum() / max(ok.sum(), 1))
