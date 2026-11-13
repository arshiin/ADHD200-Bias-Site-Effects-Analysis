# Data Bias and Scanner Site Effects in Pediatric Neuroimaging: A Responsible Machine Learning Approach

Code for the SIPAIM 2026 paper by A. S. Bayazidi (1), T. Djavaherpour (2), A. Gavrilov (1), M. Wadhera (1) and M. Bento (1)
((1) University of Calgary, (2) McMaster University).

This program compares ADHD Machine Learning classifiers built from demographic data (age, sex), resting-state functional connectivity (FC), and both, with and without ComBat site harmonization. The target was to measure how much site, sex, and age information is carried by the brain data in FC maps, and to examine how performance and errors are distributed across site, sex, age, and head motion levels.

## Repository structure

```
config.py                       paths, analysis settings and random seeds
run_all.py                      runs the whole pipeline in order
preprocessing/                  raw ADHD-200 files  ->  data/prepared/
    00_make_region_map.py       (optional) CC200 parcel -> 23 regions, from atlas images
    01_select_scans.py          one resting-state scan per subject + phenotypes
    02_compute_fc.py            Fisher-z FC -> 276 region-pair features
    03_head_motion.py           mean framewise displacement (FD)
    atlas/cc200_regions.csv     parcel -> region map (provided; made by step 0)
analysis/                       data/prepared/  ->  results/ and figures/
    utils.py                    shared functions (data checks, folds, ComBat, choosing C, statistics)
    01_classification.py        Table I + significance tests
    02_decodability.py          Table II (site, sex, age information in FC before/after ComBat)
    03_subgroups.py             Fig. 1, errors by sex and at KKI, pooled vs within-site performance
    04_external_test.py         evaluation on the ADHD-200 test release (not reported in the paper)
    05_paper_tables.py          prints Tables I and II in LaTeX
data/                           raw and prepared data (not in the repository)
results/, figures/              outputs of the analysis
```

## Setup

Tested with Python 3.10 and the package versions in `requirements.txt`:

```
pip install -r requirements.txt
```

`neuroCombat` is listed separately because `neuroHarmonize` needs it but does not install it.

## Data

No data are included. The data are publicly available after agreeing to the ADHD-200 / INDI data usage
terms; please follow those terms (do not redistribute the data) and cite the data papers.

| Needed file or folder | Where to get it |
|---|---|
| `ADHD200_CC200_TCs_filtfix/` (training release) | [ADHD-200 Preprocessed](http://preprocessed-connectomes-project.org/adhd200/), Athena pipeline, CC200 time courses; download from [NITRC](http://www.nitrc.org/frs/?group_id=383) or the public S3 bucket ([instructions](http://preprocessed-connectomes-project.org/adhd200/download.html)) |
| `ADHD200_CC200_TCs_TestRelease/` (test release) | same, test release (only needed for `04_external_test.py`) |
| `adhd200_preprocessed_phenotypics.csv` | ADHD-200 Preprocessed phenotypic file (if you have the `.tsv` version, save it as `.csv`) |
| `mriqc_func/sub-*/ses-*/func/*_bold.json` | MRIQC (v0.14.2) functional image-quality metrics for ADHD-200, from the public FCP-INDI S3 bucket |
| `rp_*.1D` (anywhere inside the data folder) | Athena realignment parameters (we used the KKI and NeuroIMAGE preprocessed archives); give mean FD for 84 training subjects without MRIQC |
| Yeo 2011 17-network atlas (`Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz`) and Harvard-Oxford subcortical atlas (`HarvardOxford-sub-maxprob-thr25-1mm.nii.gz`) | only for the optional step 0 (the CC200 atlas is in `ADHD200_CC200_TCs_filtfix/templates/`) |

Put everything in `data/raw/` (or anywhere, and set the environment variable `ADHD200_DATA_DIR` to that folder).
Head motion is used only to describe subgroups; without the MRIQC and `rp_` files, only the motion panel of
Fig. 1 cannot be reproduced.

Data references: Bellec et al. (2017), *The Neuro Bureau ADHD-200 Preprocessed repository*, NeuroImage 144:275–286;
ADHD-200 Consortium (2012), Frontiers in Systems Neuroscience 6:62.

## How to run

```
python run_all.py            # preprocessing + analysis (about 10 minutes)
python run_all.py analysis   # analysis only, if data/prepared/ already exists
```

Scripts can also be run one by one in numbered order (analysis scripts from inside `analysis/`).

## Code functionality

- **Data.** Training release: 768 children from 7 sites (2 sites, Pittsburgh and Washington University contributed
  controls only). One subject without recorded sex is excluded: 767 subjects (279 ADHD). For each subject
  the first resting-state scan is used.
- **Features.** Fisher z-transformed Pearson correlations between the 190 CC200 parcels, averaged within and
  between 23 regions (17 Yeo cortical networks, 6 subcortical/cerebellar regions): 276 features. Four of
  which (within-region values of single-parcel regions) are 0 for everyone and are dropped before ComBat.
  Demographic features are sex and age.
- **Model.** L2-regularized logistic regression with class-balanced weights. The regularization strength C is
  chosen inside each training fold by inner 3-fold cross-validation (C in 1e-4, 1e-3, 0.01, 0.1, 1, 10;
  mean balanced accuracy; ties go to the smaller C). The same procedure is used for every model.
- **Evaluation.** Five-fold cross-validation, stratified by diagnosis and grouped by subject (one scan per
  subject). Everything learned from data (C, standardization, ComBat) is fit on training data only, inside
  each inner and outer split, so each subject is predicted once by a model that never saw them.
- **Harmonization.** ComBat (neuroHarmonize), fitted on training data with site as batch and age and sex
  as preserved covariates; the diagnosis is not used. Held-out subjects are harmonized with their site
  label (ComBat requires it). For ADHD classification they are also harmonized with their own age and sex
  (standard use). When site, sex or age is the prediction target (Table II), held-out subjects are
  harmonized with the training mean age and most common sex instead, so ComBat cannot pass a subject's own
  age or sex into their features.
- **Reporting.** Table I: balanced accuracy and AUC computed in each of the five test folds and reported as
  mean ± SD across folds (population SD). Table II: AUC on the pooled out-of-fold predictions (site and age
  band: average one-vs-rest AUC) with 95% percentile bootstrap intervals over subjects. Fig. 1: balanced
  accuracy within subgroups on the pooled out-of-fold predictions with 95% bootstrap intervals. Bootstrap
  intervals resample the held-out predictions without retraining, so they describe uncertainty given the
  trained models, not variability from training or data partitioning.
- **Tests.** DeLong's test (AUC) and a paired subject-level bootstrap (balanced accuracy, 5000 resamples)
  for five comparisons in Table I; a paired bootstrap for the before–after changes in Table II. All use the
  pooled out-of-fold predictions and no multiple-comparison correction, and they do not account for
  variability from model training, so p-values should be read as approximate.

## Outputs and the paper

| Paper | File |
|---|---|
| Table I | `results/table1_classification.csv` |
| Significance tests | `results/table1_tests.csv` |
| Table II | `results/table2_decodability.csv` |
| Fig. 1 | `figures/fig1_subgroups.pdf` (data: `results/fig1_subgroups.csv`) |
| Errors by sex and at KKI | `results/errors_by_group.csv` |
| Pooled vs within-site performance | `results/site_dependence.csv` |
| Chosen C values | `results/chosen_C_classification.csv`, `results/chosen_C_decodability.csv` |
| External test (not in the paper) | `results/external_test.csv`, `results/external_test_by_sex.csv` |

Expected Table I (balanced accuracy / AUC): Demographics 0.625 / 0.656, FC 0.597 / 0.629,
Demographics + FC 0.608 / 0.643, FC (ComBat) 0.576 / 0.603, Demographics + FC (ComBat) 0.590 / 0.608.

## Reproducibility

- All random steps use `config.SEED` (outer folds, inner folds, solver, bootstraps; Table II uses SEED + 1).
  A rerun gives identical numbers with the versions in `requirements.txt`.
- The scripts check their inputs and stop if something is wrong: unique subjects matching across files,
  site folder matching the phenotype file, no subject in both releases, 276 finite FC features in the
  expected order, valid diagnosis/sex/age/site values, no repeated scans among analysed subjects, both
  diagnostic groups in every fold, and every class present wherever a model is trained.
- Participant-level files (`data/`, `results/oof_predictions.csv`) are excluded from the repository.

## Notes

- In the test release, Brown subjects 26002/26052 and 26022/26043 have identical time series. Brown
  subjects have no released diagnosis and are not used in any analysis.
- The test release was looked at during earlier development of this project, although it is never used for
  any choice made by this code. It should therefore not be described as a fully untouched hold-out set.

## Citation

A. S. Bayazidi, T. Djavaherpour, A. Gavrilov, M. Wadhera and M. Bento, "Data Bias and Scanner Site Effects
in Pediatric Neuroimaging: A Responsible Machine Learning Approach," SIPAIM 2026.

## License

The code is released under the MIT License (see `LICENSE`). The license covers the code only, not the
ADHD-200 data, which remain subject to the ADHD-200 / INDI data usage terms.
