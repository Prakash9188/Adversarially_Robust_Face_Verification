# Complete workflow of the project

**Title:** Bayes-Fused Transformation-Consistency Detection for Adversarially Robust Face Verification

## 0. One-minute explanation
A face-verification system says "same person / different person" by comparing two face
embeddings. An attacker can add an almost invisible perturbation to one photo so the system
makes the wrong call. We (1) measure how badly this works against a frozen pretrained model,
(2) test the cheap defenses from the poster (JPEG, blur, median, bit-depth), (3) show how they
behave against *non-adaptive* and *adaptive* attackers, and (4) propose a detector that
checks whether a pair's similarity stays *consistent* under several cheap transformations,
fusing those signals with a Bayes decision rule that can output **Same / Different / Flag**.

## 1. Data flow (end to end)

```
LFW images (250x250)                       data.py
  -> identity-disjoint split 70/15/15 (train/val/test), balanced genuine/impostor pairs
  -> centre-crop 150, resize 160, uint8 tensors (cached)
        |
        v
FaceEmbedder (frozen InceptionResnetV1, VGGFace2)      model.py
  x in [0,1] -> standardise -> 512-d -> L2-normalise
  score s(x1,x2) = cosine similarity = e1 . e2
        |
        +--> Clean verification: threshold tau fitted on VAL only         metrics.py
        |      (max-accuracy threshold; Gaussian-ML Bayes threshold)
        |
        +--> Attacks on the probe image x2 (anchor x1 untouched)          attacks.py
        |      genuine pair  -> dodging      (push s down)
        |      impostor pair -> impersonation (push s up)
        |      FGSM, PGD  [Commit 1];  DeepFool, CW, Square, adaptive BPDA [Commit 2]
        |
        +--> Defenses applied to BOTH images at test time                 defenses.py
        |      none | JPEG 75/50/30 | Gaussian | Median 3/5 | bit-depth 4
        |
        +--> Commit 2: consistency features + detector + Bayes fusion     features/detector/fusion
               z = [ s, s_t - s, max(d1_t, d2_t) for each transform t ]
               classifier (LDA / Gaussian-Bayes / SVM-RBF / boosting) -> P(adv | z)
               minimum-risk rule -> Same | Different | Flag
```

## 2. Stage by stage

| Stage | What happens | Course concept |
|---|---|---|
| Split | Identities (not images) are split, so no person appears in two splits. This prevents data leakage. | Hypothesis evaluation |
| Embedding | A frozen CNN maps a face to a 512-d vector. Nothing is trained by us. | CNN, transfer learning |
| Score | Cosine similarity of unit vectors. | Distance-based methods |
| Threshold | Chosen on VAL to minimise error. Also a Gaussian-ML version: fit N(mu,sigma^2) to genuine and impostor scores, take the point where densities are equal. | Bayes minimum-error rule, ML estimation |
| Attack | Gradient ascent on the score w.r.t. the input pixels, inside an L-inf ball of radius eps. | Backprop, gradient descent |
| Defense | Input smoothing/quantisation that removes high-frequency perturbation. | (taught from scratch) |
| Detector (C2) | Classify "clean vs attacked" from the consistency vector. | LDA, Bayes/QDA, SVM, boosting, PCA, SFFS |
| Fusion (C2) | Decide attack if P(adv given z) * C_miss > P(clean given z) * C_false-alarm. | Bayes minimum-risk |

## 3. Evaluation protocol
- **Clean:** accuracy, AUC, TAR@FAR = 1e-2 and 1e-3, per defense.
- **Attack success rate (ASR):** among pairs correct when clean, the fraction wrong after the attack.
- **Non-adaptive:** examples crafted on the undefended model, defense applied afterwards.
- **Adaptive (C2):** attacker sees the defense; BPDA / straight-through gradients and an ensemble-over-transforms objective.
- **Detector:** AUC and TPR at fixed FPR; trained on seen attacks, tested on **unseen** attacks.
- **Cost:** milliseconds per pair.

## 4. Commits
1. **Foundation:** data, model, FGSM/PGD, baseline defenses, metrics, runner, tests, README.
2. **Novelty:** detector, Bayes fusion, extra attacks, adaptive attacks, comparison plots.
3. **Finalisation:** tests, edge cases, final runs, full docs, CVPR 8-page report, viva material.

## 5. How to run (after setup)
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pytest -q tests
.venv\Scripts\python scripts\run_baseline.py --quick     # smoke test (~minutes)
.venv\Scripts\python scripts\run_baseline.py             # full Commit-1 baseline
.venv\Scripts\python scripts\run_detector.py --quick     # Commit-2 smoke test
.venv\Scripts\python scripts\run_detector.py             # full Commit-2 study (hours on CPU)
.venv\Scripts\python scripts\demo.py IMG1 IMG2 --same 1 --attack
```
Outputs appear in `experiments/results/`.
