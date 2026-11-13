"""
These are settings shared by all scripts:

data locations
analysis choices
random seeds

Edit RAW_DATA_DIR to point to your local copy of the ADHD-200 data (see README).
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------- paths
REPO_DIR = Path(__file__).resolve().parent

# Folder with the downloaded ADHD-200 files (see README). Default: data/raw inside this repository;
# to use another location, set the environment variable ADHD200_DATA_DIR.
RAW_DATA_DIR = Path(os.environ.get("ADHD200_DATA_DIR", REPO_DIR / "data" / "raw"))

TRAIN_TIMESERIES_DIR = RAW_DATA_DIR / "ADHD200_CC200_TCs_filtfix"        # training release
TEST_TIMESERIES_DIR = RAW_DATA_DIR / "ADHD200_CC200_TCs_TestRelease"     # test release
PHENOTYPE_FILE = RAW_DATA_DIR / "adhd200_preprocessed_phenotypics.csv"
MRIQC_DIR = RAW_DATA_DIR / "mriqc_func"                                  # MRIQC functional QC (JSON)

REGION_MAP_FILE = REPO_DIR / "preprocessing" / "atlas" / "cc200_regions.csv"
PREPARED_DIR = REPO_DIR / "data" / "prepared"
RESULTS_DIR = REPO_DIR / "results"
FIGURES_DIR = REPO_DIR / "figures"

# ----------------------------------------------------------------------------- data
# The CC200 atlas has 200 labels, but 10 of them are absent, leaving 190 parcels.
ABSENT_CC200_PARCELS = {2, 5, 11, 25, 34, 68, 82, 95, 126, 191}
CC200_PARCELS = sorted(set(range(1, 201)) - ABSENT_CC200_PARCELS)

# Site folders in the data release -> site name (Peking has three sub-folders)
SITE_FOLDERS = {"Peking_1": "Peking", "Peking_2": "Peking", "Peking_3": "Peking", "Brown": "Brown",
                "KKI": "KKI", "NeuroIMAGE": "NeuroIMAGE", "NYU": "NYU", "OHSU": "OHSU",
                "Pittsburgh": "Pittsburgh", "WashU": "WashU"}

# 17 Yeo cortical networks + 6 subcortical/cerebellar regions (order defines the feature order)
REGIONS = ["VisCent", "VisPeri", "SomMotA", "SomMotB", "DorsAttnA", "DorsAttnB", "SalVentAttnA",
           "SalVentAttnB", "LimbicA", "LimbicB", "ContA", "ContB", "ContC", "DefaultA", "DefaultB",
           "DefaultC", "TempPar", "Brainstem", "Caudate", "Cerebellum", "Other-subcortical",
           "Putamen", "Thalamus"]

# The 276 FC features in the order used everywhere else: within-region and region-pair averages
FC_FEATURES = [f"within_{REGIONS[a]}" if a == b else f"{REGIONS[a]}_x_{REGIONS[b]}"
               for a in range(len(REGIONS)) for b in range(a, len(REGIONS))]

# Site codes in the phenotype file (used to check that each scan's folder matches its phenotype)
SITE_CODES = {1: "Peking", 2: "Brown", 3: "KKI", 4: "NeuroIMAGE", 5: "NYU", 6: "OHSU", 7: "Pittsburgh", 8: "WashU"}

AGE_BINS = [0, 10, 12, 15, 100]               # left-closed: [0,10), [10,12), [12,15), [15,100)
AGE_LABELS = ["<10", "10–12", "12–15", "15+"]
FD_THRESHOLD = 0.2                            # mm, mean framewise displacement (low <= 0.2 < high)

# ----------------------------------------------------------------------------- analysis
# SEED fixes every random step: the outer folds (StratifiedGroupKFold), the inner folds used to choose C
# (StratifiedKFold), the logistic regression solver, and the bootstrap generators
# (SEED: tests in 01, subgroup CIs in 03, external test in 04; SEED + 1: Table II in 02).
SEED = 42
N_FOLDS = 5                                   # outer cross-validation
N_INNER_FOLDS = 3                             # inner cross-validation for choosing C
C_VALUES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
N_BOOT_TESTS = 5000                           # paired bootstrap for balanced-accuracy differences
N_BOOT_CI = 2000                              # bootstrap confidence intervals
