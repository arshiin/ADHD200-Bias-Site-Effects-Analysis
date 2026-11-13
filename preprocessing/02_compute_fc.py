"""
Preprocessing step 2: functional connectivity (FC) features.

For each subject:
  1. read the CC200 time series (190 parcels) and put the parcels in a fixed order;
  2. compute the Pearson correlation between every pair of parcels and apply the Fisher z-transform;
  3. average the connections within and between 23 regions (17 Yeo cortical networks and 6
     subcortical/cerebellar regions; see preprocessing/atlas/cc200_regions.csv), giving
     23 * 24 / 2 = 276 region-pair features.

A parcel with a constant (zero-variance) signal has undefined correlations; its connections are set
to 0. Regions with a single parcel have no within-region connections, so those 4 features are 0 for
everyone (they are dropped later, before harmonization).

Output: data/prepared/fc_networks_train.csv and fc_networks_test.csv (subject + 276 features)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def read_timeseries(path):
    """Read a CC200 .1D file and return a (time points x 190 parcels) array in a fixed parcel order."""
    table = pd.read_csv(path, sep="\t", float_precision="round_trip")
    table.columns = table.columns.str.strip()             # headers are padded with spaces ("Mean_1  ")
    columns = [f"Mean_{parcel}" for parcel in config.CC200_PARCELS]
    assert set(columns) <= set(table.columns), f"missing CC200 parcels in {path}"
    timeseries = table[columns].to_numpy(dtype=np.float64)
    assert np.isfinite(timeseries).all(), f"non-finite values in {path}"
    return timeseries


def fisher_z_connectivity(timeseries):
    """Pearson correlation matrix -> Fisher z. Undefined values (constant parcels) and the diagonal are 0."""
    constant = timeseries.std(axis=0) < 1e-10
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(timeseries.T)
    z = np.arctanh(np.clip(r, -1 + 1e-7, 1 - 1e-7))
    z[constant, :] = 0.0
    z[:, constant] = 0.0
    z[~np.isfinite(z)] = 0.0
    np.fill_diagonal(z, 0.0)
    return z


def region_features(fc, region_of_parcel):
    """Average FC within each region and between each pair of regions (276 values)."""
    fc = fc.astype(np.float32)
    members = [np.where(region_of_parcel == region)[0] for region in config.REGIONS]
    values = []
    for a in range(len(config.REGIONS)):
        for b in range(a, len(config.REGIONS)):
            if a == b:
                if len(members[a]) < 2:                  # a single parcel has no within-region connections
                    values.append(np.float32(0.0))
                    continue
                block = fc[np.ix_(members[a], members[a])]
                values.append(block[np.triu_indices(len(members[a]), k=1)].mean())
            else:
                values.append(fc[np.ix_(members[a], members[b])].mean())
    return np.array(values, dtype=np.float32)


def main():
    region_map = pd.read_csv(config.REGION_MAP_FILE).set_index("roi_id")
    region_of_parcel = region_map.loc[config.CC200_PARCELS, "region"].to_numpy()
    assert set(region_of_parcel) == set(config.REGIONS), "region map does not match config.REGIONS"

    for release in ["train", "test"]:
        subjects = pd.read_csv(config.PREPARED_DIR / f"subjects_{release}.csv")
        rows = []
        for path in subjects["path"]:
            timeseries = read_timeseries(config.RAW_DATA_DIR / path)
            rows.append(region_features(fisher_z_connectivity(timeseries), region_of_parcel))

        values = np.vstack(rows)
        assert values.shape == (len(subjects), len(config.FC_FEATURES)) == (len(subjects), 276)
        assert np.isfinite(values).all(), "non-finite FC values"
        # Report subjects whose FC is identical to another subject's (repeated scans under different IDs).
        # In the test release, Brown 26002/26052 and 26022/26043 have identical time series; Brown subjects
        # have no released diagnosis and are never analysed. The analysis re-checks its own subjects.
        duplicated = pd.DataFrame(values).duplicated(keep=False).to_numpy()
        if duplicated.any():
            print(f"  note: identical FC for subjects {subjects['subject'][duplicated].tolist()}")
        features = pd.DataFrame(values, columns=config.FC_FEATURES)
        features.insert(0, "subject", subjects["subject"].to_numpy())
        features.to_csv(config.PREPARED_DIR / f"fc_networks_{release}.csv", index=False, float_format="%.9g")
        print(f"{release}: {features.shape[0]} subjects x {features.shape[1] - 1} FC features")


if __name__ == "__main__":
    main()
