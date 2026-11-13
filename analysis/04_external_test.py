"""
External test (not reported in the paper): train once on the whole training release and evaluate
once on the separate ADHD-200 test release.

Each configuration is trained on all 767 training subjects (C chosen by inner cross-validation on
them, as in Table I; ComBat fitted on them) and predicts the 171 labelled test subjects (the 26 Brown
subjects have no released diagnosis). Every test site also appears in training. WashU is not in the
test release; see utils.apply_combat.

Output: results/external_test.csv, results/external_test_by_sex.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import utils
from utils import config

X_train, info_train = utils.load_data("train")
X_test, info_test = utils.load_data("test")
assert not set(info_train["subject"]) & set(info_test["subject"]), "subject in both releases"
assert set(info_test["site"]) <= set(info_train["site"]), "test site not seen in training"
assert not pd.DataFrame(np.vstack([X_train, X_test])).duplicated().any(), "same scan in both releases"

# Stack both releases so the same functions can be used with index arrays
X = np.vstack([X_train, X_test])
info = pd.concat([info_train, info_test], ignore_index=True)
y = info["adhd"].to_numpy()
train_idx = np.arange(len(info_train))
test_idx = np.arange(len(info_train), len(info))
print(f"train {len(train_idx)} subjects, test {len(test_idx)} subjects ({y[test_idx].sum()} ADHD)")

rng = np.random.default_rng(config.SEED)
rows, by_sex = [], []
for name, label in utils.CONFIGURATIONS.items():
    probabilities, C = utils.fit_and_predict(name, X, info, y, train_idx, test_idx)
    score = probabilities[:, 1]
    prediction = (score >= 0.5).astype(int)
    y_test = y[test_idx]

    boot_bacc, boot_auc = [], []
    while len(boot_bacc) < config.N_BOOT_CI:
        idx = rng.integers(0, len(y_test), len(y_test))
        if len(np.unique(y_test[idx])) < 2:
            continue
        boot_bacc.append(balanced_accuracy_score(y_test[idx], prediction[idx]))
        boot_auc.append(roc_auc_score(y_test[idx], score[idx]))
    rows.append({"configuration": label, "C": C,
                 "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                 "ci_low": np.percentile(boot_bacc, 2.5), "ci_high": np.percentile(boot_bacc, 97.5),
                 "auc": roc_auc_score(y_test, score),
                 "auc_ci_low": np.percentile(boot_auc, 2.5), "auc_ci_high": np.percentile(boot_auc, 97.5)})

    sex = info_test["sex"].to_numpy()
    for sex_value, sex_name in [(0, "female"), (1, "male")]:
        mask = sex == sex_value
        by_sex.append({"configuration": label, "sex": sex_name, "n": int(mask.sum()),
                       "labelled_adhd": prediction[mask].mean(),
                       "sensitivity": prediction[mask & (y_test == 1)].mean(),
                       "specificity": 1 - prediction[mask & (y_test == 0)].mean()})
    print(f"  {label} done (C = {C})")

results = pd.DataFrame(rows)
results.to_csv(config.RESULTS_DIR / "external_test.csv", index=False)
pd.DataFrame(by_sex).to_csv(config.RESULTS_DIR / "external_test_by_sex.csv", index=False)
pd.set_option("display.width", 200)
print("\n", results.round(3).to_string(index=False))
print("\n", pd.DataFrame(by_sex).round(3).to_string(index=False))
