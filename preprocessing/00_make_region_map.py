"""
Preprocessing step 0 (optional): assign each CC200 parcel to one of 23 regions.
The result is already provided in preprocessing/atlas/cc200_regions.csv. This script shows how it was
made and regenerates it. (Requires nibabel and three atlas images (see README)).

For each parcel, the voxels are mapped to MNI space and looked up in:
  1. the Yeo 2011 17-network cortical atlas -> most common network among the voxels
  2. otherwise the Harvard-Oxford subcortical atlas -> most common structure (left/right combined)
  3. otherwise by location: centroid below z = -25 mm -> Cerebellum, else Other-subcortical.

Output: preprocessing/atlas/cc200_regions.csv (roi_id, region, region_type, source)
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

CC200_ATLAS = config.TRAIN_TIMESERIES_DIR / "templates" / "ADHD200_parcellate_200.nii.gz"
YEO_ATLAS = config.RAW_DATA_DIR / "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz"
HO_ATLAS = (config.RAW_DATA_DIR / "Harvard-Oxford cortical and subcortical structural atlases"
            / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz")

YEO_NAMES = {1: "VisCent", 2: "VisPeri", 3: "SomMotA", 4: "SomMotB", 5: "DorsAttnA", 6: "DorsAttnB",
             7: "SalVentAttnA", 8: "SalVentAttnB", 9: "LimbicA", 10: "LimbicB", 11: "ContA", 12: "ContB",
             13: "ContC", 14: "DefaultA", 15: "DefaultB", 16: "DefaultC", 17: "TempPar"}
HO_NAMES = {4: "Thalamus", 15: "Thalamus", 5: "Caudate", 16: "Caudate", 6: "Putamen", 17: "Putamen",
            7: "Pallidum", 18: "Pallidum", 9: "Hippocampus", 19: "Hippocampus", 10: "Amygdala",
            20: "Amygdala", 11: "Accumbens", 21: "Accumbens", 8: "Brainstem"}


def labels_at(voxels_mm, atlas_image, atlas_data):
    """Atlas label at each world (mm) coordinate, using the nearest atlas voxel (0 outside the atlas)."""
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(atlas_image.affine), voxels_mm)).astype(int)
    inside = (ijk >= 0).all(axis=1) & (ijk < atlas_data.shape[:3]).all(axis=1)
    labels = np.zeros(len(ijk), dtype=int)
    labels[inside] = atlas_data[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
    return labels


def main():
    cc200 = nib.load(CC200_ATLAS)
    cc200_data = np.asarray(cc200.get_fdata()).astype(int)
    yeo = nib.load(YEO_ATLAS)
    yeo_data = np.asarray(yeo.get_fdata()).astype(int)
    if yeo_data.ndim == 4:
        yeo_data = yeo_data[..., 0]
    ho = nib.load(HO_ATLAS)
    ho_data = np.asarray(ho.get_fdata()).astype(int)

    rows = []
    for parcel in config.CC200_PARCELS:
        voxels_mm = nib.affines.apply_affine(cc200.affine, np.argwhere(cc200_data == parcel))
        yeo_labels = labels_at(voxels_mm, yeo, yeo_data)
        yeo_labels = yeo_labels[yeo_labels > 0]
        if len(yeo_labels):
            region, region_type, source = YEO_NAMES[np.bincount(yeo_labels).argmax()], "cortical", "Yeo17"
        else:
            ho_labels = [HO_NAMES[l] for l in labels_at(voxels_mm, ho, ho_data) if l in HO_NAMES]
            if ho_labels:
                names, counts = np.unique(ho_labels, return_counts=True)
                region, region_type, source = names[counts.argmax()], "subcortical", "HarvardOxford"
            elif round(voxels_mm[:, 2].mean(), 1) < -25:
                region, region_type, source = "Cerebellum", "cerebellar", "location"
            else:
                region, region_type, source = "Other-subcortical", "other", "location"
        rows.append({"roi_id": parcel, "region": region, "region_type": region_type, "source": source})

    regions = pd.DataFrame(rows)
    assert set(regions["region"]) == set(config.REGIONS)
    regions.to_csv(config.REGION_MAP_FILE, index=False)
    print(regions["region"].value_counts().to_string())


if __name__ == "__main__":
    main()
