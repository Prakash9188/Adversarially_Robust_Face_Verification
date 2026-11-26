# Viva question bank with answers (face-verification project)

> All numbers below come from `experiments/results/` (test split, eps = 4/255 unless stated;
> 300 attacked pairs, 150 for adaptive attacks; one seed). Never quote a number that is not in
> those files. Items marked **(C2)** are the Commit-2 components (detector, fusion, adaptive attacks).

---
## 1. Problem statement and project understanding
1. **What is the project about?** Defending face verification (same/different person) against adversarial perturbations using lightweight input transformations and a Bayes-fused consistency detector.
2. **Problem?** A tiny L-inf perturbation on one photo can flip the verification decision.
3. **Why this problem?** Face verification guards phones, banking and access control; fooling it is identity fraud.
4. **Motivation?** Existing defenses (adversarial training) are expensive; cheap filters are known to be bypassable. We study whether cheap defenses can be made reliable.
5. **Who benefits?** Researchers (honest evaluation), developers (cheap defence), organisations, end users.
6. **Objectives?** Measure vulnerability; benchmark JPEG/Gaussian/Median/bit-depth; evaluate against adaptive attackers; propose and test a detector.
7. **Scope?** Digital attacks on a frozen pretrained model on LFW. Not physical attacks, not training new face models.
8. **Input/output?** Input: two face images. Output: Same / Different (and Flag in C2).
9. **If unsolved?** Biometric systems remain spoofable; trust in them erodes.
10. **Existing approaches?** Adversarial training, input preprocessing (JPEG, filters), feature squeezing detection, purification.
11. **Their limits?** Training is costly; preprocessing breaks under adaptive attacks (BPDA); trade-off with clean accuracy.
12. **Central research question?** Can a lightweight defense reduce attack success while keeping clean accuracy, *including against an attacker who knows the defense*?

## 2. Literature and existing work
1. **Related work?** Shield (JPEG defense); Feature Distillation and BPDA results on JPEG; Keep It Real (2025) on under-evaluated compression defenses; Massoli et al. on detecting attacks on face recognition; a Turku thesis on FGSM/PGD vs preprocessing for verification.
2. **Gap?** I found no work evaluating a pre/post-transformation embedding-consistency detector for verification *pairs*. My search was shallow, so this is "not found", not "does not exist".
3. **Difference from state of the art?** We do not claim to beat state-of-the-art robust training. Our contribution is a cheap, verification-specific detector and an honest adaptive evaluation.
4. **Why this approach?** Cheap, no retraining, built from course concepts (Bayes theory, LDA, SVM).
5. **Alternatives considered?** Adversarial training (too costly on CPU), autoencoder purifier (known BPDA target, heavy on CPU).
6. **Is the improvement meaningful?** Only if it holds on *unseen* attacks and adaptive attacks; otherwise we report that honestly.

## 3. Dataset and preprocessing
1. **Dataset?** LFW (Labeled Faces in the Wild), funneled version: 13,233 images of 5,749 identities.
2. **Why?** Standard, public, small enough for CPU, widely used for verification.
3. **Pairs?** We sample balanced genuine/impostor pairs inside each split with a fixed seed (train 400 / val 600 / test 600 by default).
4. **Why not official pairs.txt?** Its pairs mix identities across folds; an identity-disjoint 70/15/15 split would leave only dozens of impostor pairs in test.
5. **Balanced?** Yes, 50/50 by construction.
6. **Preprocessing?** Centre crop 150, resize 160, scale to [0,1]; the model standardises internally. Needed because the network expects fixed size and normalised input.
7. **Missing values?** None (images).
8. **Data leakage?** Information from test data influencing training/tuning. We prevent it by splitting *identities*, fitting thresholds on VAL only, and asserting no identity overlap in code.
9. **Why not tune on test?** It would make test accuracy optimistic and unreliable.
10. **Dataset limits?** Mostly celebrities, frontal, Western-skewed; not representative of all demographics or conditions.
11. **Noisy data?** Defenses (filters) and detector are partly aimed at that; not otherwise handled.

## 4. Algorithms and theory
1. **Embedding model?** InceptionResnetV1 pretrained on VGGFace2, frozen (transfer learning). It maps a face to a 512-d unit vector.
2. **Why cosine similarity?** On unit vectors it equals the dot product, is scale-free, and matches how the model was trained.
3. **FGSM:** x' = x + eps * sign(grad_x L). One step in the direction that most increases the loss under an L-inf budget (linear approximation).
4. **PGD:** repeat small signed-gradient steps and project back into the eps-ball (and [0,1]); a stronger, iterative FGSM with random start.
5. **Our loss for verification?** L = sign * s, with sign = +1 (impostor, impersonation) or -1 (genuine, dodging).
6. **Threshold selection?** (a) maximise accuracy on VAL (minimum error rate); (b) Gaussian ML: fit N(mu,sigma^2) per class, pick where the densities cross (Bayes rule with equal priors).
7. **JPEG?** Block DCT, quantisation, inverse; removes high-frequency detail; the quantisation step is non-differentiable.
8. **Median filter?** Replace each pixel by the median of its window; removes isolated outliers.
9. **Gaussian blur / bit-depth?** Low-pass filter / reduce colour levels; both squeeze the space available to the attacker.
10. **Assumptions?** Cosine score separates classes; perturbations are small and high-frequency; (C2) transformed-score vector is roughly Gaussian per class for LDA/QDA.
11. **Hyperparameters?** eps, PGD steps/step size, JPEG quality, kernel sizes, thresholds; chosen on VAL.
12. **Complexity?** Embedding is one CNN forward pass per image; PGD costs steps x (forward+backward); the detector adds one forward per transform (T extra passes).
13. **Overfitting?** The backbone is frozen, so overfitting risk is in the detector (small feature vector, small classifiers, evaluated on unseen attacks) and in threshold tuning (hence a separate test split).

## 5. Architecture and methodology
1. **Architecture?** pair -> preprocess -> frozen CNN -> cosine score -> threshold; **(C2)** plus transform bank -> consistency features -> detector -> minimum-risk fusion -> Same/Different/Flag.
2. **Why this design?** Frozen backbone = cheap and reproducible; transforms are cheap; Bayes fusion is principled and uses the course.
3. **Which module matters most?** The embedding model for accuracy; the detector/fusion for robustness.
4. **Can any be removed?** Defense filters can (that is the "no defense" baseline); the detector can (that is the Commit-1 system).
5. **If a module fails?** Fall back to plain thresholding; (C2) a Flag output can route to manual review.
6. **Bottleneck?** CNN forward/backward passes on CPU.
7. **Scaling?** Batch + GPU; cache embeddings per transform; the detector is tiny.

## 6. Novelty and contribution
1. **What exactly did you do beyond existing implementations?** (a) A verification-pair-level consistency detector over a cheap transform bank; (b) Bayes minimum-risk fusion with a Flag output **(C2)**; (c) a side-by-side non-adaptive vs adaptive evaluation of the poster's defenses; (d) a reproducible, identity-disjoint protocol.
2. **Type of novelty?** Mostly experimental + methodological. Research-level novelty is **unverified** without a full literature survey.
3. **Baseline vs proposed?** Baseline: threshold on the raw (optionally filtered) score. Proposed: threshold plus detector/fusion over several transforms.
4. **Why should it work?** An attack tuned to the raw model tends to be "brittle": the embedding shifts when the input is transformed, producing inconsistent scores. A clean pair stays consistent.
5. **If it does not work?** Report it. An adaptive attacker who optimises over all transforms may remove the inconsistency; that finding is itself a result.
6. **Validation?** Unseen-attack generalisation, adaptive attacks, ablations over which transforms are used (SFFS).
7. **Limit of the improvement?** Costs extra forward passes; weaker vs attackers who include the detector in their objective.

## 7. Implementation
1. **Language/libraries?** Python; PyTorch (autograd for attacks), facenet-pytorch (pretrained weights), scikit-learn (LDA/QDA/SVM/boosting, metrics), pandas, matplotlib, PIL.
2. **Folder structure?** `src/` (data, model, attacks, defenses, metrics; C2: features, detector, fusion), `scripts/` (runners), `tests/`, `configs/`, `docs/`, `experiments/results/`.
3. **Implemented from scratch?** Attacks (FGSM/PGD), median filter, metrics (threshold, TAR@FAR, ASR, Bayes threshold), data protocol. **Library:** the pretrained CNN, JPEG codec (PIL), sklearn classifiers.
4. **Main function?** `pgd()` in `src/attacks.py` and `main()` in `scripts/run_baseline.py`.
5. **Reproducibility?** Fixed seed, pinned requirements, cached pair list, config file, declared in `docs/REPRODUCIBILITY.md`.
6. **Invalid input?** Defenses assert/handle tensor shapes; data asserts identity-disjoint splits.
7. **Errors met?** `facenet-pytorch` needed `requests`/`tqdm` after a `--no-deps` install (it pins an old torch); the LFW host did not resolve so we used the figshare mirror; a unit test failed because test anchors carried an autograd graph (fixed by detaching, as the real pipeline does); the first "adaptive" attack was not aimed at the detector, so we added a detector-aware variant and benign controls.

## 8. Experiments and results
1. **Metrics and why?** Accuracy/AUC/TAR@FAR (standard verification), ASR (security), clean degradation (cost of defense), ms/pair (lightweight claim).
2. **Why TAR@FAR, not just accuracy?** Security systems operate at low false-accept rates; accuracy hides that.
3. **Final results?** Clean: 98.8% test accuracy, TAR@FAR=1% 0.990. Undefended ASR: PGD 100%, FGSM 78.8%. Best single filter (median 5x5): PGD 79.1%, FGSM 61.3%; none survives eps=8 (PGD 97.6%). Under adaptive BPDA the matching filters stay weak (ASR 90% for JPEG-30, 97% for median-5). Proposed detector+fusion (gradient boosting, trained on FGSM+PGD only): ASR FGSM 2.0%, PGD 0.0%, CW 0.3% (unseen), all adaptive attacks 0.0% with detector AUC 0.995-1.0; costs 5.3% clean flags and 0.3% clean errors, ~455 ms vs ~69 ms per pair. Failures: Square (AUC 0.58) and FGSM at eps=8 (ASR 29.3%, AUC 0.93).
4. **Why might JPEG/median look good?** Non-adaptive attacks ignore the defense. Under adaptive attacks (BPDA) such defenses are known to weaken substantially.
5. **Reliability?** Fixed seed, identity-disjoint splits, thresholds/tau/model choice fixed on val before touching test, benign controls (random noise flagged 8.3%, JPEG-75 flagged 1.0%) to rule out a pure noise detector. Single seed only, so no confidence intervals: differences of a few points are not claimed.
6. **Cross-validation?** The backbone is not trained, so classic CV is not needed; the detector can be cross-validated.
7. **Ablation?** Which transforms; which classifier; eps; number of PGD steps.
8. **Statistical significance?** With 300 attacked pairs the 95% interval on a proportion near 50% is about +/- 5-6 points (+/- 8 for 150 pairs); we do not claim smaller differences. Not repeated over seeds.
9. **Error sources?** Small attacked subset, crop-based preprocessing instead of face detection/alignment, one backbone, one dataset.

## 9. Limitations and future scope
1. **Limitations?** CPU-limited attack strength/size; digital attacks only; single pretrained backbone; centre-crop instead of MTCNN alignment; LFW demographic bias.
2. **Real-world?** Partly; physical attacks, different cameras and bias would need further study.
3. **Ethics/bias?** LFW is skewed; a detector could produce more false flags on under-represented groups; we do not collect new face data.
4. **Distribution shift?** Thresholds and detector would need re-calibration on the new data.
5. **Future work?** ArcFace/larger backbones, learned purifier, ensembles, physical attacks, fairness analysis.

## 10. GitHub and teamwork
1. **Why Git/GitHub?** Version history, reproducibility, collaboration. Git is the local tool; GitHub is the hosting service.
2. **Commits?** (1) foundation+baseline, (2) novelty+experiments, (3) finalisation+report. Hashes and timestamps: `docs/commit_log.md` (generated by `scripts/commit_log.py`).
3. **Reproduce?** Clone, create venv, install `requirements-lock.txt`, download LFW, run scripts as in README.
4. **Team division?** *Fill in honestly who did what (Om Dinesh Dhande, Prakash Singh, Ankit Kumar).*

## 11. Difficult questions
1. **What if the main assumption is wrong?** If clean/attacked consistency does not differ, the detector gives AUC near 0.5, and we report that as a negative result.
2. **Why trust the results?** Fixed protocol, no test-set tuning, identity-disjoint splits, unit tests, reported counts.
3. **Weakest part?** Evaluation against a limited set of adaptive attacks; a stronger attacker may break it.
4. **Is it better than baseline, provably?** Not proved; shown empirically on tested attacks at fixed clean accuracy.
5. **Worth the cost?** Compare the extra ms/pair with ASR reduction.
6. **Baseline better than proposed?** Then we say so; the poster's success criterion is a trade-off, not a win.
7. **Solving the problem or a metric?** The ASR under adaptive attacks is the security-relevant number; clean-attack ASR alone can mislead.
8. **Explain to a non-technical person.** A lock that compares your face to a photo can be tricked by invisible noise. We add cheap "second opinions" that check whether the answer stays stable, and raise a flag when it does not.

## 12. Topic table (course concepts)
| Topic | Short answer |
|---|---|
| Bias-variance | Underfit (high bias) vs overfit (high variance); detect overfitting by train/val/test gap. |
| Bayes decision theory | Choose the class with minimum risk; with equal costs, maximum posterior. Used for the threshold and the Flag rule. |
| Decision tree / Random Forest / Boosting | Split by impurity reduction; forest averages many trees (lower variance); boosting adds weak learners sequentially to reduce bias, unlike parallel bagging. |
| PCA vs LDA | PCA keeps maximum variance (unsupervised); LDA maximises class separation (supervised). |
| K-means / GMM / EM | K from elbow/silhouette; EM alternates responsibilities (E) and parameter updates (M) for GMM. |
| SVM | Maximum-margin hyperplane; kernels implicitly map to higher dimensions. |
| MLP / backprop | Chain rule to propagate error gradients; our attacks use the same gradient w.r.t. the *input*. |
| CNN | Weight sharing and local filters exploit image structure; hierarchical features. |
| Autoencoder | Encoder compresses, decoder reconstructs; used for purification, which we listed as future work. |

## 13. Preparation strategy
Learn Level 1 (problem, data, architecture, results, novelty, limits) first, then Level 2 (math, hyperparameters, code), then Level 3 (what-ifs, failure cases). For practice, ask Claude to run a viva simulation at Basic, Intermediate or Advanced level, on General, ML or this project.

## 14. Extra tough questions specific to our results
1. **Your detector is ~100% on adaptive attacks. Isn't that too good?** It is the result we measured, but it is conditional: our adaptive attacks are specific (PGD through the transforms, plus a variant that also keeps the embedding stable). They are not a worst-case attacker (e.g. one that differentiates through the detector itself). We state this limitation and do not claim robustness in general.
2. **Is the detector just detecting noise?** Controls say mostly no: random +/-4/255 noise is flagged 8.3% (near the 5% operating point) and plain JPEG-75 1.0%, whereas attacks are flagged 96-100%.
3. **Why does it fail on Square?** Our 100-query Square attack barely succeeds (3% ASR) and produces patch-structured perturbations unlike the gradient attacks the detector was trained on; AUC 0.58 shows poor generalisation to that family, but with so few successes it is weak evidence either way.
4. **Is "flagged" a real defence?** Flagging denies service; we report the clean flag rate (5.3%) as its price. If flags counted as failures, security is traded for availability.
5. **Why ~6.6x slower and still "lightweight"?** It needs 6 extra embeddings per image on CPU; it is cheap compared with adversarial training but not free. A fairer wording is "training-free", and we say so.
6. **What if the attacker knows the threshold and detector?** Then they can optimise against the detector (not tested). This is the main open risk and the first item of future work.
