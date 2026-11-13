"""
Table II: how much site, sex and age information can be predicted from FC alone, before and after
ComBat harmonization.

Uses the same folds, classifier and tuning as Table I. FC is the only input. The predicted variable is never used
as an input. Held-out subjects (inner validation and outer test) are harmonized with their site label, as
ComBat requires, but without their own age and sex (training mean age and most common sex are used
instead), so ComBat won't pass a subject's age or sex into their features.
The ADHD row reuses the Table I predictions for FC and FC (ComBat); held-out subjects were
harmonized with their own age and sex (standard ComBat use because age and sex are not the target).

Reports: AUC on the pooled out-of-fold predictions (site and age band: average one-vs-rest AUC), with
95% bootstrap confidence intervals, and a paired bootstrap test of the before-after change
(the same resampled subjects for both).

Run 01_classification.py first. Outputs: results/table2_decodability.csv, results/chosen_C_decodability.csv
"""
import numpy as np
import pandas as pd

import utils
from utils import config

X, info = utils.load_data("train")
folds = utils.make_folds(info)
n = len(info)

sites = sorted(info["site"].unique())
targets = {
    "site": info["site"].map({s: i for i, s in enumerate(sites)}).to_numpy(),
    "sex": info["sex"].astype(int).to_numpy(),
    "age_band": pd.cut(info["age"], config.AGE_BINS, labels=False, right=False).to_numpy(),
}

# ---------------------------------------------------------------- out-of-fold probabilities
probabilities = {}
chosen_C = []
for target_name, target in targets.items():
    n_classes = len(np.unique(target))
    for features, configuration in [("before", "fc"), ("after", "fc_combat")]:
        probabilities[(target_name, features)] = np.zeros((n, n_classes))
for fold, (train_idx, test_idx) in enumerate(folds):
    for target_name, target in targets.items():
        for features, configuration in [("before", "fc"), ("after", "fc_combat")]:
            p, C = utils.fit_and_predict(configuration, X, info, target, train_idx, test_idx, use_own_age_sex=False)
            probabilities[(target_name, features)][test_idx] = p
            chosen_C.append({"target": target_name, "features": features, "fold": fold, "C": C})
    print(f"  fold {fold} done")

# ADHD row: the Table I out-of-fold predictions
oof = pd.read_csv(config.RESULTS_DIR / "oof_predictions.csv")
assert (oof["subject"].values == info["subject"].values).all()
targets["adhd"] = oof["adhd"].to_numpy()
probabilities[("adhd", "before")] = np.column_stack([1 - oof["score_fc"], oof["score_fc"]])
probabilities[("adhd", "after")] = np.column_stack([1 - oof["score_fc_combat"], oof["score_fc_combat"]])

# ---------------------------------------------------------------- AUC, bootstrap CI and before-after test
rng = np.random.default_rng(config.SEED + 1)
rows = []
for target_name in ["site", "sex", "age_band", "adhd"]:
    target = targets[target_name]
    n_classes = len(np.unique(target))
    before, after = probabilities[(target_name, "before")], probabilities[(target_name, "after")]
    boot_before, boot_after = [], []
    while len(boot_before) < config.N_BOOT_CI:
        idx = rng.integers(0, n, n)
        if len(np.unique(target[idx])) < n_classes:      # every class must be present
            continue
        boot_before.append(utils.auc(target[idx], before[idx]))
        boot_after.append(utils.auc(target[idx], after[idx]))
    boot_before, boot_after = np.array(boot_before), np.array(boot_after)
    change = boot_before - boot_after
    rows.append({"target": target_name,
                 "auc_before": utils.auc(target, before),
                 "ci_before_low": np.percentile(boot_before, 2.5), "ci_before_high": np.percentile(boot_before, 97.5),
                 "auc_after": utils.auc(target, after),
                 "ci_after_low": np.percentile(boot_after, 2.5), "ci_after_high": np.percentile(boot_after, 97.5),
                 "p_change": min(1.0, 2 * min((change <= 0).mean(), (change >= 0).mean()))})
table2 = pd.DataFrame(rows)

table2.to_csv(config.RESULTS_DIR / "table2_decodability.csv", index=False)
pd.DataFrame(chosen_C).to_csv(config.RESULTS_DIR / "chosen_C_decodability.csv", index=False)
pd.set_option("display.width", 200)
print("\nTable II (pooled out-of-fold AUC, 95% bootstrap CI)\n", table2.round(3).to_string(index=False))
