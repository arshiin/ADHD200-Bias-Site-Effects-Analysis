"""
Helper functions shared by the analysis scripts:
Load the prepared data
Make cross-validation folds
Apply ComBat harmonization, choosing C, and
the statistical tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationApply, harmonizationLearn
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# The five input configurations compared in the paper
CONFIGURATIONS = {
    "demo": "Demographics (age, sex)",
    "fc": "Functional Connectivity (FC)",
    "demo_fc": "Demographics + FC",
    "fc_combat": "FC (ComBat harmonized)",
    "demo_fc_combat": "Demographics + FC (ComBat harmonized)",
}


# ----------------------------------------------------------------------------- data
def load_data(release):
    """Prepared FC features (n x 276) and subject information. Subjects without a diagnosis,
    sex or age are excluded (one training subject has no recorded sex)."""
    info = pd.read_csv(config.PREPARED_DIR / f"subjects_{release}.csv")
    fc = pd.read_csv(config.PREPARED_DIR / f"fc_networks_{release}.csv")

    # input checks (they stop the analysis if the prepared files are not as expected)
    assert info["subject"].is_unique, "repeated subject IDs (one scan per subject expected)"
    assert (info["subject"].values == fc["subject"].values).all(), "subject order differs between files"
    assert list(fc.columns) == ["subject"] + config.FC_FEATURES, "FC features missing or in the wrong order"

    keep = (info["adhd"].notna() & info["sex"].notna() & info["age"].notna()).to_numpy()
    info = info[keep].reset_index(drop=True)
    info["adhd"] = info["adhd"].astype(int)
    X = fc[keep].drop(columns="subject").to_numpy(dtype=np.float32).astype(np.float64)

    assert X.shape[1] == 276 and np.isfinite(X).all(), "FC features must be 276 finite numbers per subject"
    assert not pd.DataFrame(X).duplicated().any(), "two analysed subjects have identical FC (repeated scan)"
    assert set(info["adhd"]) == {0, 1}, "diagnosis must be 0 (TD) or 1 (ADHD)"
    assert set(info["sex"]) <= {0, 1}, "sex must be 0 (female) or 1 (male)"
    assert info["age"].between(5, 30).all(), "age outside the expected range (ADHD-200 ages are 7-27)"
    assert set(info["site"]) <= set(config.SITE_CODES.values()), "unknown site"
    return X, info


def demographics(info):
    """The two demographic features: sex (0 = female, 1 = male) and age (years)."""
    return info[["sex", "age"]].to_numpy(dtype=np.float64)


def make_folds(info):
    """Five-fold cross-validation, stratified by diagnosis and grouped by subject (one scan each)."""
    cv = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    folds = list(cv.split(np.zeros((len(info), 1)), info["adhd"], groups=info["subject"]))
    for train_idx, test_idx in folds:
        assert not set(info["subject"][train_idx]) & set(info["subject"][test_idx]), "subject in train and test"
        assert info["site"][train_idx].nunique() == info["site"].nunique(), "a site is missing from a training fold"
        assert info["adhd"][train_idx].nunique() == 2 and info["adhd"][test_idx].nunique() == 2, \
            "both diagnostic groups are needed in every training and test fold"
    return folds


def make_model(C):
    """L2-regularized logistic regression with class-balanced weights."""
    return LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=4000,
                              random_state=config.SEED)


# ----------------------------------------------------------------------------- ComBat
def fit_combat(X_train, info_train):
    """Fit ComBat on training data only: site is the batch, age and sex are preserved.
    The diagnosis is never used. Features that are constant within a training site are dropped."""
    keep = np.ones(X_train.shape[1], dtype=bool)
    for site in info_train["site"].unique():
        keep &= X_train[info_train["site"].values == site].std(axis=0) > 1e-8
    covariates = pd.DataFrame({"SITE": info_train["site"].values,
                               "age": info_train["age"].values,
                               "sex": info_train["sex"].values})
    model, X_train_harmonized = harmonizationLearn(X_train[:, keep], covariates)
    return model, keep, X_train_harmonized


def apply_combat(model, keep, X_new, info_new, X_train, info_train, use_own_age_sex=True):
    """Harmonize new (held-out) subjects with a ComBat model fitted on training data.

    ComBat removes each site's offset and scaling and keeps the age and sex effects estimated on the
    training data. Held-out subjects are always harmonized with their own site label (ComBat needs it).
      use_own_age_sex=True  (ADHD classification): each subject's own age and sex are used, as usual.
      use_own_age_sex=False (when site, sex or age is the prediction target): the training mean age and
          most common sex are used instead, so a subject's own age and sex never enter their features.
    The diagnosis is never used.

    neuroHarmonize expects every training site to be present in the new data. If a site is missing
    (WashU is not in the test release), one training subject from that site is added as a
    placeholder and removed again afterwards. Each subject is harmonized separately with the stored
    training estimates, so the placeholder does not change anyone else's values.
    """
    if use_own_age_sex:
        age, sex = info_new["age"].values, info_new["sex"].values
    else:
        age = np.full(len(info_new), info_train["age"].mean())
        sex = np.full(len(info_new), float(info_train["sex"].mode().iloc[0]))
    sites = info_new["site"].values

    missing_sites = sorted(set(info_train["site"]) - set(sites))
    placeholders = [np.where(info_train["site"].values == s)[0][0] for s in missing_sites]
    X_all = np.vstack([X_new[:, keep], X_train[placeholders][:, keep]])
    covariates = pd.DataFrame({"SITE": np.r_[sites, info_train["site"].values[placeholders]],
                               "age": np.r_[age, info_train["age"].values[placeholders]],
                               "sex": np.r_[sex, info_train["sex"].values[placeholders]]})
    return harmonizationApply(X_all, covariates, model)[:len(X_new)]


# ----------------------------------------------------------------------------- features and models
def build_features(configuration, X, info, train_idx, test_idx, use_own_age_sex=True):
    """Training and held-out feature matrices for one split. Everything learned from data
    (here: ComBat) uses the training subjects only."""
    demo = demographics(info)
    if configuration == "demo":
        return demo[train_idx], demo[test_idx]

    if configuration in ("fc", "demo_fc"):
        A, B = X[train_idx], X[test_idx]
    else:  # ComBat-harmonized FC
        model, keep, A = fit_combat(X[train_idx], info.iloc[train_idx])
        B = apply_combat(model, keep, X[test_idx], info.iloc[test_idx], X[train_idx], info.iloc[train_idx],
                         use_own_age_sex)

    if configuration in ("demo_fc", "demo_fc_combat"):
        A = np.hstack([A, demo[train_idx]])
        B = np.hstack([B, demo[test_idx]])
    return A, B


def choose_C(configuration, X, info, target, train_idx, use_own_age_sex=True):
    """Choose C by inner cross-validation on the training subjects only (mean balanced accuracy).
    Standardization and ComBat are refit inside every inner split. Ties go to the smaller C."""
    inner_cv = StratifiedKFold(n_splits=config.N_INNER_FOLDS, shuffle=True, random_state=config.SEED)
    mean_score = np.zeros(len(config.C_VALUES))
    for inner_train, inner_val in inner_cv.split(np.zeros(len(train_idx)), target[train_idx]):
        tr, val = train_idx[inner_train], train_idx[inner_val]
        assert set(target[tr]) == set(target), "every class must be present in each inner training split"
        A, B = build_features(configuration, X, info, tr, val, use_own_age_sex)
        scaler = StandardScaler().fit(A)
        A, B = scaler.transform(A), scaler.transform(B)
        for j, C in enumerate(config.C_VALUES):
            prediction = make_model(C).fit(A, target[tr]).predict(B)
            mean_score[j] += balanced_accuracy_score(target[val], prediction) / config.N_INNER_FOLDS
    return config.C_VALUES[int(np.argmax(mean_score))]


def fit_and_predict(configuration, X, info, target, train_idx, test_idx, use_own_age_sex=True):
    """Choose C on the training subjects, refit on all of them, and predict the held-out subjects.
    Returns class probabilities for the held-out subjects and the chosen C."""
    C = choose_C(configuration, X, info, target, train_idx, use_own_age_sex)
    A, B = build_features(configuration, X, info, train_idx, test_idx, use_own_age_sex)
    scaler = StandardScaler().fit(A)
    model = make_model(C).fit(scaler.transform(A), target[train_idx])
    # every class seen in training, so probability columns always mean the same classes
    assert list(model.classes_) == sorted(set(target[train_idx])) == sorted(set(target)), "class missing in training"
    return model.predict_proba(scaler.transform(B)), C


# ----------------------------------------------------------------------------- metrics and tests
def auc(target, probabilities):
    """AUC from predicted probabilities; for more than two classes, the average one-vs-rest AUC."""
    if probabilities.shape[1] == 2:
        return roc_auc_score(target, probabilities[:, 1])
    return roc_auc_score(target, probabilities, multi_class="ovr", average="macro")


def delong_test(y, score_a, score_b):
    """DeLong's test for two correlated AUCs (DeLong et al., 1988). Returns (AUC_a - AUC_b, p)."""
    def components(score):
        pos, neg = score[y == 1], score[y == 0]
        wins = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        return wins.mean(axis=1), wins.mean(axis=0), wins.mean()

    pos_a, neg_a, auc_a = components(score_a)
    pos_b, neg_b, auc_b = components(score_b)
    cov_pos = np.cov(np.vstack([pos_a, pos_b]))
    cov_neg = np.cov(np.vstack([neg_a, neg_b]))
    variance = ((cov_pos[0, 0] + cov_pos[1, 1] - 2 * cov_pos[0, 1]) / len(pos_a)
                + (cov_neg[0, 0] + cov_neg[1, 1] - 2 * cov_neg[0, 1]) / len(neg_a))
    z = (auc_a - auc_b) / np.sqrt(variance)
    return auc_a - auc_b, 2 * norm.sf(abs(z))


def paired_bootstrap_test(y, prediction_a, prediction_b, rng, n_boot=config.N_BOOT_TESTS):
    """Paired bootstrap over subjects for a difference in balanced accuracy (same resampled subjects
    for both models). Returns (difference, two-sided p)."""
    n = len(y)
    differences = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:                       # both classes needed
            differences.append(balanced_accuracy_score(y[idx], prediction_a[idx])
                               - balanced_accuracy_score(y[idx], prediction_b[idx]))
    differences = np.array(differences)
    observed = balanced_accuracy_score(y, prediction_a) - balanced_accuracy_score(y, prediction_b)
    p = min(1.0, 2 * min((differences <= 0).mean(), (differences >= 0).mean()))
    return observed, p
