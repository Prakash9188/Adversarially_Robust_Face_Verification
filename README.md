# Adversarially Robust Face Verification (lightweight defenses)

IIT Jodhpur semester project (Machine Learning). Team: Om Dinesh Dhande, Prakash Singh, Ankit Kumar.

**Question:** can cheap input-transformation defenses reduce adversarial attacks on face
verification while preserving clean accuracy, including against attackers who know the defense?

Status: **Commit 2 (novelty + experiments)**. Commit 1 = baseline; Commit 3 = final cleanup,
report and full documentation.

## What Commit 2 adds
- Attacks: DeepFool, CW-L2 (both clipped to L_inf), Square (black-box), and ADAPTIVE PGD with
  BPDA / straight-through gradients (`pgd_bpda`), including a detector-aware variant.
- **Proposed method:** transformation-consistency features (`src/features.py`), clean-vs-attacked
  detectors LDA / Gaussian-Bayes (QDA) / RBF-SVM / boosted trees (`src/detector.py`), and a Bayes
  minimum-risk Same / Different / **Flag** rule (`src/fusion.py`).
- `scripts/run_detector.py` runs the full study; `tests/test_novelty.py` tests the new parts.

```powershell
.venv\Scripts\python scripts\run_detector.py --quick    # smoke test
.venv\Scripts\python scripts\run_detector.py            # full study (hours on CPU; attacks are cached)
```

## Results (test split, 300 attacked pairs, eps = 4/255, detector = gradient-boosted trees)
Detector is trained on FGSM + PGD only (train split); thresholds / model choice on val; flags set
at 5% false-alarm on clean val pairs. ASR = attack success rate (lower is better).

| Attack | none | JPEG-30 | Median-5 | **Fused (flag counts as stopped)** | Detector AUC |
|---|---|---|---|---|---|
| FGSM (seen) | 0.788 | 0.636 | 0.613 | **0.020** | 0.994 |
| PGD (seen) | 1.000 | 0.859 | 0.791 | **0.000** | 1.000 |
| DeepFool (unseen) | 0.040 | 0.017 | 0.007 | 0.000 | 0.991 |
| CW (unseen) | 0.663 | 0.168 | 0.121 | **0.003** | 0.998 |
| Square (unseen, black-box) | 0.030 | 0.030 | 0.027 | 0.024 | 0.580 |
| BPDA through JPEG-30 (adaptive, 150 pairs) | 0.973 | 0.899 | 0.779 | 0.000 | 1.000 |
| BPDA through Median-5 (adaptive) | 0.973 | 0.839 | 0.966 | 0.000 | 0.995 |
| Detector-aware adaptive (150 pairs) | 1.000 | 0.940 | 0.933 | 0.000 | 1.000 |
| FGSM eps=8 (other budget) | 0.882 | 0.811 | 0.822 | 0.293 | 0.928 |

Cost and controls (`experiments/results/`): 5.3% of clean pairs are flagged and 0.3% are wrong
(clean accuracy without flagging 99.0%); benign +/-4/255 noise is flagged 8.3% of the time and
plain JPEG-75 images 1.0%, so the detector is not just a noise detector. The detector needs
~486 ms/pair versus ~73 ms for the plain model (about 6.7x, CPU).

**Honest caveats:** Square and DeepFool barely succeed at this budget (ASR 3-4%), so the detector's
weak AUC on Square says little; the "detector-aware" attack is one specific attack (PGD that also
keeps the embedding stable under the transforms), not a worst-case adaptive attacker; single
dataset, single backbone, single seed, 150-300 pairs, so small differences are within noise.

## What Commit 1 contains
- LFW loading with an **identity-disjoint** 70/15/15 split and balanced genuine/impostor pairs.
- Frozen pretrained InceptionResnetV1 (VGGFace2) as embedding extractor; score = cosine similarity.
- FGSM and PGD attacks on the probe image (dodging for genuine pairs, impersonation for impostors).
- Baseline defenses: JPEG (75/50/30), Gaussian blur, median 3x3/5x5, 4-bit depth.
- Metrics: accuracy, AUC, TAR@FAR, attack success rate (ASR); thresholds fitted on validation only.

## Setup (Windows PowerShell; Linux/macOS: use `source .venv/bin/activate`)
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
# CPU build of torch (the lock file pins the +cpu wheels):
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install numpy scipy scikit-learn pandas matplotlib pillow pyyaml pytest requests tqdm
.venv\Scripts\python -m pip install --no-deps facenet-pytorch   # it pins an old torch; --no-deps is required
```
Exact versions used are in `requirements-lock.txt` (`pip freeze`) and `environment.yml`.

## Run
```powershell
.venv\Scripts\python -m pytest -q tests                 # unit tests (no network)
.venv\Scripts\python scripts\run_baseline.py --quick    # smoke test
.venv\Scripts\python scripts\run_baseline.py            # full baseline
```
The first run downloads LFW funneled (~230 MB, figshare mirror
`https://ndownloader.figshare.com/files/5976015`; original host
`http://vis-www.cs.umass.edu/lfw/`) into `data/` and the pretrained weights
(~107 MB, VGGFace2, downloaded by `facenet-pytorch` from its GitHub release) into the torch cache.
**Neither is stored in this repository.** Outputs: `experiments/results/`.

## Configuration
`configs/default.yaml` (seed, pair counts, eps list, PGD steps, defenses, number of attacked pairs).

## Layout
```
src/        data.py model.py attacks.py defenses.py metrics.py
scripts/    run_baseline.py  commit_log.py
tests/      test_core.py
configs/    default.yaml
docs/       WORKFLOW.md VIVA_QA.md REPRODUCIBILITY.md
experiments/results/   CSVs, figures, logs
```

## Checkpoints
No model is trained here. We use the publicly released VGGFace2 InceptionResnetV1 weights from
`facenet-pytorch` (https://github.com/timesler/facenet-pytorch). We do not re-upload the ~107 MB
file; it is fetched automatically on first run.

## Results (Commit 1)
See `experiments/results/baseline_clean.csv` and `baseline_attacks.csv`. Non-adaptive attacks
only: examples are crafted on the undefended model and the defense is applied afterwards.
Adaptive attackers are evaluated in Commit 2.

## Limitations
Centre-crop instead of face detection/alignment; CPU-limited attacked subset; one backbone and
one dataset; digital attacks only.
