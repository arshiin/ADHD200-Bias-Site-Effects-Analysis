"""
Step 1 of preprocessing: choose one resting-state scan per subject and attach the phenotypes.

For every subject we use the Athena-preprocessed CC200 time series of the first available scan
(lowest session, then lowest run) of the filtered, motion-regressed variant ("sfnwmrda").
Diagnosis is coded ADHD = 1 (any subtype) and typically developing (TD) = 0; "pending" diagnoses
(the Brown test-release subjects) are left missing. Nothing is imputed.

Output: data/prepared/subjects_train.csv and data/prepared/subjects_test.csv
        (one row per subject, sorted by site and subject ID)
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

FILE_PATTERN = re.compile(r"(sfnwmrda|snwmrda)(\d+)_session_(\d+)_rest_(\d+)_cc200_TCs\.1D$")


def find_scans(release_dir):
    """List every CC200 time-series file of the 'sfnwmrda' variant in one data release."""
    rows = []
    for site_dir in sorted(release_dir.iterdir()):
        if not site_dir.is_dir() or site_dir.name == "templates":
            continue
        for path in site_dir.rglob("*cc200_TCs.1D"):
            match = FILE_PATTERN.search(path.name)
            if match is None or match.group(1) != "sfnwmrda":
                continue
            rows.append({"subject": int(match.group(2)),
                         "session": int(match.group(3)),
                         "run": int(match.group(4)),
                         "site": config.SITE_FOLDERS[site_dir.name],
                         "path": str(path.relative_to(config.RAW_DATA_DIR))})
    return pd.DataFrame(rows)


def load_phenotypes():
    pheno = pd.read_csv(config.PHENOTYPE_FILE, encoding="utf-8-sig")
    pheno = pheno.rename(columns={"ScanDir ID": "subject"})
    pheno["subject"] = pd.to_numeric(pheno["subject"], errors="coerce")
    pheno = pheno.dropna(subset=["subject"])
    pheno["subject"] = pheno["subject"].astype(int)

    diagnosis = pd.to_numeric(pheno["DX"], errors="coerce")        # "pending" -> NaN
    pheno["adhd"] = diagnosis.map({0: 0, 1: 1, 2: 1, 3: 1})       # 1, 2, 3 = ADHD subtypes
    pheno["sex"] = pd.to_numeric(pheno["Gender"], errors="coerce")  # 0 = female, 1 = male
    pheno["age"] = pd.to_numeric(pheno["Age"], errors="coerce")
    pheno["phenotype_site"] = pd.to_numeric(pheno["Site"], errors="coerce").map(config.SITE_CODES)
    assert pheno["subject"].is_unique, "a subject appears twice in the phenotype file"
    return pheno[["subject", "adhd", "sex", "age", "phenotype_site"]]


def main():
    config.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    phenotypes = load_phenotypes()

    selected = {}
    for release, release_dir in [("train", config.TRAIN_TIMESERIES_DIR), ("test", config.TEST_TIMESERIES_DIR)]:
        scans = find_scans(release_dir)
        n_with_several_scans = (scans.groupby("subject").size() > 1).sum()
        # first scan per subject: lowest session, then lowest run
        scans = scans.sort_values(["subject", "session", "run"]).drop_duplicates("subject", keep="first")

        subjects = scans.merge(phenotypes, on="subject", how="left", validate="one_to_one")
        # checks: every scan has a phenotype record, and the scan's site folder matches the phenotype site
        assert subjects["phenotype_site"].notna().all(), "some scans have no phenotype record"
        assert (subjects["site"] == subjects["phenotype_site"]).all(), "site folder and phenotype site disagree"
        subjects = subjects.sort_values(["site", "subject"]).reset_index(drop=True)
        subjects = subjects[["subject", "site", "adhd", "sex", "age", "session", "run", "path"]]
        subjects.to_csv(config.PREPARED_DIR / f"subjects_{release}.csv", index=False)
        selected[release] = set(subjects["subject"])

        print(f"{release}: {len(subjects)} subjects from {subjects['site'].nunique()} sites; "
              f"labelled: {subjects['adhd'].notna().sum()} (ADHD {int(subjects['adhd'].sum())}); "
              f"missing sex: {subjects['sex'].isna().sum()}; "
              f"{n_with_several_scans} subjects had more than one scan (first one used)")

    assert not selected["train"] & selected["test"], "a subject appears in both the training and test release"


if __name__ == "__main__":
    main()
