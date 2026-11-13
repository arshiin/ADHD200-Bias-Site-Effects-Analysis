"""
Run the whole pipeline in order.

    python run_all.py            # preprocessing + analysis (you will need the raw ADHD-200 data)
    python run_all.py analysis   # analysis only (this needs data/prepared/ from a previous run)
"""
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent

PREPROCESSING = ["preprocessing/01_select_scans.py",
                 "preprocessing/02_compute_fc.py",
                 "preprocessing/03_head_motion.py"]
ANALYSIS = ["analysis/01_classification.py",
            "analysis/02_decodability.py",
            "analysis/03_subgroups.py",
            "analysis/04_external_test.py",
            "analysis/05_paper_tables.py"]

steps = ANALYSIS if "analysis" in sys.argv[1:] else PREPROCESSING + ANALYSIS
for step in steps:
    script = REPO_DIR / step
    print(f"\n========== {step}")
    subprocess.run([sys.executable, script.name], cwd=script.parent, check=True)
print("\nDone. Results are in results/ and the figure in figures/.")
