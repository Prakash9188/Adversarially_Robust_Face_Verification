"""Clean-vs-adversarial pair detectors built from course techniques.

lda    : Linear Discriminant Analysis (Fisher / shared-covariance Gaussians)
qda    : Gaussian Bayes classifier with class-specific covariances (normal-density discriminants)
svm    : RBF-kernel SVM (kernel trick)
gboost : boosted shallow decision trees
Equal class priors are used so the posterior is not driven by the training mix.
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

KINDS = ("lda", "qda", "svm", "gboost")


def make_detector(kind, seed=0):
    if kind == "lda":
        clf = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
    elif kind == "qda":
        clf = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5], reg_param=0.1)
    elif kind == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced",
                  random_state=seed)
    elif kind == "gboost":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=seed)
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), clf)


def posterior(det, Z):
    """P(adversarial | z)."""
    return det.predict_proba(Z)[:, 1]


def select_features(Z, y, k):
    """Forward sequential feature selection (SFFS-style) with an LDA scorer."""
    sel = SequentialFeatureSelector(make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
                                    n_features_to_select=k, direction="forward", cv=3)
    return sel.fit(Z, y).get_support(indices=True)
