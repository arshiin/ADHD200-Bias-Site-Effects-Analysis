"""
Table I: ADHD classification for the five input configurations, and the significance tests.

Five-fold cross-validation on the training release (n = 767). In every fold, C is chosen by inner
cross-validation on the training folds, and the model, standardization and ComBat are fit on the
training folds only. So every subject gets one held-out ("out-of-fold") prediction.

Reports:
  Table I : balanced accuracy and AUC in each of the five test folds -> mean and SD across folds
  tests   : on the pooled out-of-fold predictions; DeLong's test (AUC) and a paired bootstrap
            (balanced accuracy). No correction for multiple comparisons.

Outputs: results/table1_classification.csv, results/table1_tests.csv,
         results/oof_predictions.csv (used by 02 and 03), results/chosen_C_classification.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import utils
from utils import config

X, info = utils.load_data("train")
y = info["adhd"].to_numpy()
folds = utils.make_folds(info)
print(f"{len(y)} subjects ({y.sum()} ADHD), {len(folds)} folds")

# ---------------------------------------------------------------- out-of-fold predictions
scores = {name: np.zeros(len(y)) for name in utils.CONFIGURATIONS}
fold_results = []
for fold, (train_idx, test_idx) in enumerate(folds):
    for name in utils.CONFIGURATIONS:
        probabilities, C = utils.fit_and_predict(name, X, info, y, train_idx, test_idx)
        score = probabilities[:, 1]
        scores[name][test_idx] = score
        fold_results.append({"configuration": name, "fold": fold, "C": C,
                             "balanced_accuracy": balanced_accuracy_score(y[test_idx], (score >= 0.5).astype(int)),
                             "auc": roc_auc_score(y[test_idx], score)})
    print(f"  fold {fold} done")
fold_results = pd.DataFrame(fold_results)
predictions = {name: (scores[name] >= 0.5).astype(int) for name in utils.CONFIGURATIONS}

# ---------------------------------------------------------------- Table I (mean and SD across folds)
table1 = []
for name, label in utils.CONFIGURATIONS.items():
    r = fold_results[fold_results["configuration"] == name]
    table1.append({"configuration": label,
                   "balanced_accuracy_mean": r["balanced_accuracy"].mean(),
                   "balanced_accuracy_sd": np.std(r["balanced_accuracy"]),      # SD across the 5 folds
                   "auc_mean": r["auc"].mean(),
                   "auc_sd": np.std(r["auc"])})
table1 = pd.DataFrame(table1)

# ---------------------------------------------------------------- significance tests (pooled out-of-fold)
comparisons = [("demo", "fc"), ("demo_fc", "demo"), ("demo_fc", "fc"),          # input configurations
               ("fc", "fc_combat"), ("demo_fc", "demo_fc_combat")]             # effect of ComBat
rng = np.random.default_rng(config.SEED)
tests = []
for a, b in comparisons:
    auc_difference, p_delong = utils.delong_test(y, scores[a], scores[b])
    bacc_difference, p_bootstrap = utils.paired_bootstrap_test(y, predictions[a], predictions[b], rng)
    tests.append({"comparison": f"{utils.CONFIGURATIONS[a]} vs {utils.CONFIGURATIONS[b]}",
                  "auc_difference": auc_difference, "p_delong": p_delong,
                  "balanced_accuracy_difference": bacc_difference, "p_bootstrap": p_bootstrap})
tests = pd.DataFrame(tests)

# ---------------------------------------------------------------- save
config.RESULTS_DIR.mkdir(exist_ok=True)
table1.to_csv(config.RESULTS_DIR / "table1_classification.csv", index=False)
tests.to_csv(config.RESULTS_DIR / "table1_tests.csv", index=False)
fold_results[["configuration", "fold", "C"]].to_csv(config.RESULTS_DIR / "chosen_C_classification.csv", index=False)
oof = info[["subject", "site", "sex", "age", "mean_fd", "adhd"]].copy()
for name in utils.CONFIGURATIONS:
    oof[f"score_{name}"] = scores[name]
oof.to_csv(config.RESULTS_DIR / "oof_predictions.csv", index=False)

pd.set_option("display.width", 200)
print("\nTable I (mean ± SD across folds)\n", table1.round(3).to_string(index=False))
print("\nTests (pooled out-of-fold predictions, uncorrected)\n", tests.round(4).to_string(index=False))
