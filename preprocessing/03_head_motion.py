"""
Preprocessing step 3: head motion (mean framewise displacement, FD) for the selected scan.

Mean FD is taken from MRIQC (fd_mean) when it is available and plausible (<= 5 mm). Otherwise it is
computed from Athena's realignment parameters (rp_*.1D): rotations are converted to millimetres on a
50 mm sphere and FD is the sum of absolute frame-to-frame changes (Power et al., 2012).
Motion is used only to describe subgroups (low <= 0.2 mm < high), never as a model input.

Output: adds a 'mean_fd' column to data/prepared/subjects_train.csv and subjects_test.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

MRIQC_PATTERN = re.compile(r"sub-(\d+)_ses-(\d+)_task-rest_run-(\d+)_bold")
RP_PATTERN = re.compile(r"rp_(\d+)_session_(\d+)_rest_(\d+)\.1D$")


def mriqc_mean_fd():
    """(subject, session, run) -> MRIQC mean FD."""
    values = {}
    for path in config.MRIQC_DIR.glob("sub-*/ses-*/func/*_task-rest_run-*_bold.json"):
        match = MRIQC_PATTERN.search(path.name)
        if match:
            key = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            values[key] = json.load(open(path)).get("fd_mean")
    return values


def athena_motion_files():
    """(subject, session, run) -> path of Athena's realignment parameters."""
    files = {}
    for path in config.RAW_DATA_DIR.rglob("rp_*.1D"):
        match = RP_PATTERN.search(path.name)
        if match:
            files.setdefault((int(match.group(1)), int(match.group(2)), int(match.group(3))), path)
    return files


def mean_fd_from_realignment(path):
    params = np.loadtxt(path)                         # 6 columns: 3 rotations (degrees) + 3 translations (mm)
    params[:, :3] *= np.pi / 180.0 * 50.0              # rotations -> mm on a 50 mm sphere
    return float(np.abs(np.diff(params, axis=0)).sum(axis=1).mean())


def main():
    mriqc = mriqc_mean_fd()
    athena = athena_motion_files()
    for release in ["train", "test"]:
        subjects = pd.read_csv(config.PREPARED_DIR / f"subjects_{release}.csv")
        mean_fd, source = [], []
        for _, s in subjects.iterrows():
            key = (int(s["subject"]), int(s["session"]), int(s["run"]))
            fd = mriqc.get(key)
            if fd is not None and fd <= 5.0:
                mean_fd.append(fd)
                source.append("MRIQC")
            elif key in athena:
                mean_fd.append(mean_fd_from_realignment(athena[key]))
                source.append("Athena")
            else:
                mean_fd.append(np.nan)
                source.append("missing")
        subjects["mean_fd"] = mean_fd
        subjects.to_csv(config.PREPARED_DIR / f"subjects_{release}.csv", index=False)
        print(f"{release}: mean FD from {pd.Series(source).value_counts().to_dict()} (of {len(subjects)} subjects)")


if __name__ == "__main__":
    main()
