"""
Fig. 1 and the subgroup checks: how performance and errors are distributed across groups.

Uses the out-of-fold predictions from 01_classification.py (threshold 0.5).
  1. Balanced accuracy within subgroups (site, sex, age band, head motion) with 95% bootstrap
     confidence intervals -> Fig. 1. The two controls-only sites are not shown (no ADHD cases).
  2. Who each model labels as ADHD, by sex and at KKI.
  3. Site dependence of FC: AUC pooled over all sites vs within each site, and how often children at
     the two controls-only sites are labelled typically developing (TD) compared with TD children elsewhere.

Outputs: results/fig1_subgroups.csv, results/errors_by_group.csv, results/site_dependence.csv,
         figures/fig1_subgroups.png and .pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import utils
from utils import config

oof = pd.read_csv(config.RESULTS_DIR / "oof_predictions.csv")
y = oof["adhd"].to_numpy()
site = oof["site"].to_numpy()
MIXED_SITES = ["KKI", "NYU", "NeuroIMAGE", "OHSU", "Peking"]       # sites with both ADHD and TD children
CONTROL_SITES = ["Pittsburgh", "WashU"]                              # controls only
SHOWN = {"demo": "Demographics", "fc": "FC", "demo_fc": "Combined (FC + Demo)"}
predicted = {name: (oof[f"score_{name}"] >= 0.5).astype(int).to_numpy() for name in utils.CONFIGURATIONS}

# ---------------------------------------------------------------- 1. subgroup balanced accuracy (Fig. 1)
age_band = pd.cut(oof["age"], config.AGE_BINS, labels=config.AGE_LABELS, right=False).astype(str).to_numpy()
motion = np.full(len(oof), "", dtype=object)
motion[oof["mean_fd"] <= config.FD_THRESHOLD] = "FD ≤0.2"
motion[oof["mean_fd"] > config.FD_THRESHOLD] = "FD >0.2"
groups = [("Overall", "All", np.ones(len(oof), dtype=bool))]
groups += [("Site", s, site == s) for s in MIXED_SITES]
groups += [("Sex", "Female", oof["sex"].to_numpy() == 0), ("Sex", "Male", oof["sex"].to_numpy() == 1)]
groups += [("Age", a, age_band == a) for a in config.AGE_LABELS]
groups += [("Motion", m, motion == m) for m in ["FD ≤0.2", "FD >0.2"]]

rng = np.random.default_rng(config.SEED)


def bootstrap_ci(y_true, y_pred):
    """95% percentile bootstrap CI of balanced accuracy, resampling subjects within the subgroup."""
    values = []
    for _ in range(config.N_BOOT_CI):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            values.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


rows = []
for section, label, mask in groups:
    for name, shown_name in SHOWN.items():
        low, high = bootstrap_ci(y[mask], predicted[name][mask])
        rows.append({"section": section, "subgroup": label, "configuration": shown_name, "n": int(mask.sum()),
                     "adhd_prevalence": y[mask].mean(),
                     "balanced_accuracy": balanced_accuracy_score(y[mask], predicted[name][mask]),
                     "ci_low": low, "ci_high": high})
subgroups = pd.DataFrame(rows)
subgroups.to_csv(config.RESULTS_DIR / "fig1_subgroups.csv", index=False)

# ---------------------------------------------------------------- 2. who is labelled ADHD
rows = []
for name, shown_name in SHOWN.items():
    for group_name, mask in [("female", oof["sex"].to_numpy() == 0), ("male", oof["sex"].to_numpy() == 1), ("KKI", site == "KKI")]:
        rows.append({"configuration": shown_name, "group": group_name, "n": int(mask.sum()),
                     "labelled_adhd": predicted[name][mask].mean(),
                     "sensitivity": predicted[name][mask & (y == 1)].mean(),
                     "specificity": 1 - predicted[name][mask & (y == 0)].mean()})
pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "errors_by_group.csv", index=False)

# ---------------------------------------------------------------- 3. site dependence
rows = []
mixed = np.isin(site, MIXED_SITES)
for name, label in utils.CONFIGURATIONS.items():
    score = oof[f"score_{name}"].to_numpy()
    within = [roc_auc_score(y[site == s], score[site == s]) for s in MIXED_SITES]
    sizes = [(site == s).sum() for s in MIXED_SITES]
    rows.append({"configuration": label,
                 "auc_all_sites": roc_auc_score(y, score),
                 "auc_within_sites_mean": np.average(within, weights=sizes),
                 "labelled_td_controls_only_sites": 1 - predicted[name][np.isin(site, CONTROL_SITES)].mean(),
                 "labelled_td_td_children_other_sites": 1 - predicted[name][mixed & (y == 0)].mean()})
pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "site_dependence.csv", index=False)

# ---------------------------------------------------------------- Fig. 1
COLORS = {"Demographics": "#7C8CEE", "FC": "#B73056", "Combined (FC + Demo)": "#FF8D4B"}
order = subgroups[["section", "subgroup"]].drop_duplicates().reset_index(drop=True)
labels = order["subgroup"].tolist()
x = np.arange(len(labels))
width = 0.26

fig, ax = plt.subplots(figsize=(12.5, 5.4))
for k, (shown_name, color) in enumerate(COLORS.items()):
    r = subgroups[subgroups["configuration"] == shown_name].set_index("subgroup").loc[labels]
    errors = [r["balanced_accuracy"] - r["ci_low"], r["ci_high"] - r["balanced_accuracy"]]
    ax.bar(x + (k - 1) * width, r["balanced_accuracy"], width, color=color, label=shown_name)
    ax.errorbar(x + (k - 1) * width, r["balanced_accuracy"], yerr=errors, fmt="none", ecolor="#555",
                elinewidth=0.9, capsize=2)

prevalence = subgroups.drop_duplicates("subgroup").set_index("subgroup").loc[labels, "adhd_prevalence"]
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n({p * 100:.0f}%)" for l, p in zip(labels, prevalence)], fontsize=8.5)
ax.axhline(0.5, color="#888781", ls="--", lw=1.3)
ax.text(1.005, 0.5, "chance", transform=ax.get_yaxis_transform(), fontsize=11, color="#888781",
        ha="left", va="center", clip_on=False)               # right of the plot area, never over the bars
sections = order["section"].tolist()
for i in range(1, len(sections)):                                   # separators and section titles
    if sections[i] != sections[i - 1]:
        ax.axvline(i - 0.5, color="#d9d7d0", lw=0.9, zorder=0)
for section in order["section"].unique():
    positions = np.where(order["section"] == section)[0]
    ax.text(positions.mean(), 0.825, section, ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, 0.82)
ax.set_yticks(np.arange(0, 0.81, 0.2))
ax.set_ylabel("Balanced Accuracy")
ax.spines[["top", "right"]].set_visible(False)
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=9, ncol=3,
                   title="Configuration")
fig.tight_layout()
fig.canvas.draw()
box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
fig.text(box.x0 + box.width / 2, box.y0 - 0.01, "Values in parentheses = % ADHD prevalence in each subgroup",
         ha="center", va="top", fontsize=10, color="#5E244E", fontweight="bold")
config.FIGURES_DIR.mkdir(exist_ok=True)
for ext in ["png", "pdf"]:
    fig.savefig(config.FIGURES_DIR / f"fig1_subgroups.{ext}", dpi=200, bbox_inches="tight")

pd.set_option("display.width", 200)
print(subgroups.round(3).to_string(index=False))
print("\n", pd.read_csv(config.RESULTS_DIR / "errors_by_group.csv").to_string(index=False))
print("\n", pd.read_csv(config.RESULTS_DIR / "site_dependence.csv").to_string(index=False))
