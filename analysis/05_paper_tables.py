"""
Print the rows of Tables I and II in LaTeX (as they appear in the paper).
Run 01_classification.py and 02_decodability.py first.
"""
import pandas as pd

from utils import config

table1 = pd.read_csv(config.RESULTS_DIR / "table1_classification.csv")
print("% Table I: mean ± SD across the five folds")
for i, r in table1.iterrows():
    cells = [f"{r.balanced_accuracy_mean:.3f} \\pm {r.balanced_accuracy_sd:.3f}", f"{r.auc_mean:.3f} \\pm {r.auc_sd:.3f}"]
    if i == 0:                                        # demographics row is shown in bold
        cells = [f"\\mathbf{{{c}}}" for c in cells]
    if i == 3:
        print("\\midrule")
    print(f"{r.configuration} & \\({cells[0]}\\) & \\({cells[1]}\\) \\\\")

table2 = pd.read_csv(config.RESULTS_DIR / "table2_decodability.csv")
names = {"site": "Site", "sex": "Sex", "age_band": "Age band", "adhd": "ADHD"}
print("\n% Table II: pooled out-of-fold AUC [95% bootstrap CI]")
for _, r in table2.iterrows():
    before = f"{r.auc_before:.2f}\\ [{r.ci_before_low:.2f}, {r.ci_before_high:.2f}]"
    after = f"{r.auc_after:.2f}\\ [{r.ci_after_low:.2f}, {r.ci_after_high:.2f}]"
    print(f"{names[r.target]} & \\({before}\\) & \\({after}\\) \\\\")
