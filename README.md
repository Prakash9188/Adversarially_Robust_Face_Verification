# Adversarially Robust Face Verification (lightweight defenses)

IIT Jodhpur semester project (Machine Learning). Team: Om Dinesh Dhande, Prakash Singh, Ankit Kumar.

**Question:** can cheap input-transformation defenses reduce adversarial attacks on face
verification while preserving clean accuracy, including against attackers who know the defense?

Status: **Commit 1 (baseline)**. The detector/fusion novelty and adaptive attacks arrive in
Commit 2; the final report and full documentation in Commit 3.

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
