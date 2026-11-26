"""Build report/tables/*.tex and report/figures/*.pdf (+ PNG copies in experiments/results)
directly from experiments/results/*.csv, so paper numbers cannot drift from the data.

    python scripts/make_report_assets.py
"""
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

R = Path("experiments/results")
T = Path("report/tables")
F = Path("report/figures")
T.mkdir(parents=True, exist_ok=True)
F.mkdir(parents=True, exist_ok=True)

DEF_NAMES = {"none": "None", "jpeg75": "JPEG-75", "jpeg50": "JPEG-50", "jpeg30": "JPEG-30",
             "gaussian": "Gaussian", "median3": "Median 3$\\times$3", "median5": "Median 5$\\times$5",
             "bitdepth4": "4-bit depth"}
ATTACKS = [  # key, label, group
    ("fgsm", "FGSM", "seen"), ("pgd", "PGD", "seen"),
    ("deepfool", "DeepFool", "unseen"), ("cw", "CW-L2", "unseen"), ("square", "Square (black-box)", "unseen"),
    ("fgsm_eps2", "FGSM, $\\epsilon{=}2$", "budget"), ("fgsm_eps8", "FGSM, $\\epsilon{=}8$", "budget"),
    ("pgd_eps2", "PGD, $\\epsilon{=}2$", "budget"), ("pgd_eps8", "PGD, $\\epsilon{=}8$", "budget"),
    ("bpda_jpeg30", "BPDA via JPEG-30", "adaptive"), ("bpda_median5", "BPDA via Median-5", "adaptive"),
    ("bpda_gaussian", "BPDA via Gaussian", "adaptive"), ("adaptive_ens", "Adaptive (all transforms)", "adaptive"),
    ("adaptive_aware", "Detector-aware adaptive", "adaptive"),
]


def f(x, d=3):
    return f"{x:.{d}f}"


def write(name, text):
    (T / name).write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ tables
clean = pd.read_csv(R / "baseline_clean.csv").set_index("defense")
att = pd.read_csv(R / "baseline_attacks.csv")
rows = []
for dn, lab in DEF_NAMES.items():
    g = att[att.defense == dn].set_index(["attack", "eps"])["ASR"]
    rows.append(f"{lab} & {f(clean.loc[dn, 'test_acc'])} & {f(g[('fgsm', 4)])} & {f(g[('pgd', 2)])} & "
                f"{f(g[('pgd', 4)])} & {f(g[('pgd', 8)])} \\\\")
write("tab_baseline.tex", "\\begin{tabular}{lccccc}\n\\toprule\nDefense & Clean acc. & FGSM$_4$ & PGD$_2$ & PGD$_4$ & PGD$_8$ \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

det = pd.read_csv(R / "detector_test.csv")
asr = pd.read_csv(R / "fusion_asr.csv")
best = [s for s in asr.system.unique() if s.startswith("fused")][0]
g = det[det.detector == best[6:-1]].set_index("attack")
a = asr.pivot(index="attack", columns="system", values="ASR")
fl = asr[asr.system == best].set_index("attack")["flag_rate"]
lines, last = [], None
for key, lab, grp in ATTACKS:
    if last is not None and grp != last:
        lines.append("\\midrule")
    last = grp
    lines.append(f"{lab} & {f(a.loc[key, 'none'])} & {f(a.loc[key, 'jpeg30'])} & {f(a.loc[key, 'median5'])} & "
                 f"\\textbf{{{f(a.loc[key, best])}}} & {f(fl[key], 2)} & {f(g.loc[key, 'AUC'])} \\\\")
write("tab_main.tex", "\\begin{tabular}{lcccccc}\n\\toprule\nAttack ($\\epsilon{=}4$ unless stated) & None & JPEG-30 & Median-5 & Fused & Flag rate & Det.\\ AUC \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

grp_of = {k: g_ for k, _, g_ in ATTACKS}
det["grp"] = det.attack.map(grp_of)
m = det.dropna(subset=["grp"]).pivot_table(index="detector", columns="grp", values="AUC", aggfunc="mean")
val = pd.read_csv(R / "detector_val.csv").set_index("detector")["val_AUC"]
order = {"lda": "LDA", "qda": "Gaussian Bayes (QDA)", "svm": "RBF-SVM", "gboost": "Boosted trees"}
lines = []
for k, lab in order.items():
    lines.append(f"{lab} & {f(val[k])} & " + " & ".join(f(m.loc[k, c]) for c in ["seen", "unseen", "budget", "adaptive"]) + " \\\\")
write("tab_detectors.tex", "\\begin{tabular}{lccccc}\n\\toprule\nDetector & Val AUC & Seen & Unseen & Other $\\epsilon$ & Adaptive \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

ab = pd.read_csv(R / "ablation.csv")
lines = []
for _, r in ab.iterrows():
    if pd.isna(r["mean_AUC_nonadaptive"]):
        continue
    lab = r["features"].replace("_", "\\_").replace("s0", "$s_0$")
    lines.append(f"{lab} & {f(r['mean_AUC_nonadaptive'])} \\\\")
write("tab_ablation.tex", "\\begin{tabular}{lc}\n\\toprule\nLDA features & Mean AUC (5 non-adaptive attacks) \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# commits table (hashes + ISO timestamps). Commit 3 cannot contain its own hash: macros.
log = subprocess.run(["git", "log", "--reverse", "--date=iso-strict", "--pretty=format:%h|%ad|%s"],
                     capture_output=True, text=True, check=True).stdout.splitlines()[:2]
lines = []
for i, l in enumerate(log, 1):
    h, d, s = l.split("|", 2)
    msg = s.split(": ", 1)[1].replace("&", "\\&")
    ts = d.replace("T", " ")
    lines.append(f"{i} & \\texttt{{{h}}} & {ts} & {msg} \\\\")
lines.append("3 & \\texttt{\\hashthree} & \\timethree & Finalize project, evaluation, and documentation \\\\")
write("tab_commits.tex", "\\begin{tabular}{clll}\n\\toprule\n\\# & Hash & Timestamp & Message \\\\\n\\midrule\n"
      + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

cc = pd.read_csv(R / "fusion_clean.csv").iloc[0]
tm = pd.read_csv(R / "timing.csv").iloc[0]
ctrl = asr[(asr.system == best) & asr.attack.str.startswith("control")].set_index("attack")["flag_rate"]
(T / "numbers.tex").write_text(
    f"\\newcommand{{\\cleanflag}}{{{100*cc['clean_flag_rate']:.1f}}}\n"
    f"\\newcommand{{\\cleanerr}}{{{100*cc['clean_error_rate']:.1f}}}\n"
    f"\\newcommand{{\\noiseflag}}{{{100*ctrl['control_noise']:.1f}}}\n"
    f"\\newcommand{{\\jpegflag}}{{{100*ctrl['control_jpeg75']:.1f}}}\n"
    f"\\newcommand{{\\msbase}}{{{tm['baseline_ms_per_pair']:.0f}}}\n"
    f"\\newcommand{{\\msdet}}{{{tm['detector_ms_per_pair']:.0f}}}\n"
    f"\\newcommand{{\\mscratio}}{{{tm['detector_ms_per_pair']/tm['baseline_ms_per_pair']:.1f}}}\n", encoding="utf-8")

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: pipeline
fig, ax = plt.subplots(figsize=(7.2, 2.4))
ax.set_xlim(0, 100); ax.set_ylim(-5, 34); ax.axis("off")


def box(x, y, w, h, text, fc="#dbe9f6"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="#33506e", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=0.9, color="#33506e"))


box(1, 12, 13, 10, "Face pair\n(anchor $a$,\nprobe $p$)", "#f2f2f2")
box(19, 12, 14, 10, "Frozen CNN\n$f(\\cdot)$ (512-d,\nunit norm)")
box(38, 21, 17, 9, "Score\n$s_0=f(a)^\\top f(p)$")
box(38, 3, 20, 14, "Transform bank $t$:\nJPEG-75/30, median,\nGaussian, 4-bit\n$\\Rightarrow s_t,\\ \\mathrm{drift}_t$", "#fde9d2")
box(63, 3, 17, 14, "Detector\n$P(\\mathrm{adv}\\,|\\,z)$\n(boosted trees)", "#fde9d2")
box(84, 11, 15, 12, "Bayes fusion\nSame / Different\n/ $\\mathbf{Flag}$", "#d9f0d9")
arrow(14, 17, 19, 17); arrow(33, 17, 38, 25); arrow(33, 15, 38, 11)
arrow(58, 10, 63, 10); arrow(80, 12, 84, 15); arrow(55, 25, 84, 20)
ax.text(58, -2.5, "consistency features $z=[s_0,\\ s_t-s_0,\\ \\mathrm{drift}_t]$", ha="center", fontsize=7, style="italic")
fig.savefig(F / "pipeline.pdf", bbox_inches="tight")
plt.close(fig)

# Fig 2: ASR for key attacks
keys = ["fgsm", "pgd", "cw", "fgsm_eps8", "pgd_eps8", "bpda_median5", "adaptive_aware"]
labs = ["FGSM", "PGD", "CW", "FGSM\n$\\epsilon$=8", "PGD\n$\\epsilon$=8", "BPDA\nMedian-5", "Detector-\naware"]
systems = [("none", "No defense", "#b23a48"), ("median5", "Median-5", "#e0a458"), (best, "Fused (ours)", "#2a6f97")]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
w = 0.26
for i, (s, lab, c) in enumerate(systems):
    ax.bar(np.arange(len(keys)) + (i - 1) * w, [a.loc[k, s] for k in keys], w, label=lab, color=c)
ax.set_xticks(range(len(keys))); ax.set_xticklabels(labs, fontsize=6)
ax.set_ylabel("Attack success rate"); ax.set_ylim(0, 1.25); ax.set_yticks([0, .25, .5, .75, 1])
ax.legend(loc="upper left", ncol=3, fontsize=6, frameon=False, columnspacing=0.8, handlelength=1)
fig.savefig(F / "asr.pdf", bbox_inches="tight"); fig.savefig(R / "report_asr.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Fig 3: flag rate: clean / benign controls / attacks
items = [("Clean", cc["clean_flag_rate"], "#6c757d"), ("Noise\n$\\pm4/255$", ctrl["control_noise"], "#6c757d"),
         ("JPEG-75", ctrl["control_jpeg75"], "#6c757d")]
for k, lab in [("fgsm", "FGSM"), ("pgd", "PGD"), ("deepfool", "DeepFool"), ("cw", "CW"), ("square", "Square"),
               ("adaptive_aware", "Detector-\naware")]:
    items.append((lab, fl[k], "#2a6f97"))
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.bar(range(len(items)), [v for _, v, _ in items], color=[c for *_, c in items])
ax.set_xticks(range(len(items))); ax.set_xticklabels([l for l, *_ in items], fontsize=6, rotation=45, ha="right")
ax.axhline(0.05, color="k", lw=0.6, ls="--", label="5% false-alarm budget")
ax.legend(loc="upper center", fontsize=6, frameon=False, bbox_to_anchor=(0.3, 1.0))
ax.set_ylabel("Flag rate"); ax.set_ylim(0, 1.05)
fig.savefig(F / "flags.pdf", bbox_inches="tight"); fig.savefig(R / "report_flags.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", sorted(p.name for p in T.iterdir()), sorted(p.name for p in F.iterdir()))
