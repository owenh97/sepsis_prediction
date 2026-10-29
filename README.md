# sepsis_prediction FIT1061 

# Early Sepsis Warning from Hourly Vitals

A Naive Bayes early-warning model for sepsis, built on the [PhysioNet/CinC 2019 Challenge](https://physionet.org/content/challenge-2019/1.0.0/) dataset (~40k ICU patients, two hospitals). The goal: flag septic patients early while keeping false alarms low enough that clinicians don't tune the alerts out.

## Problem

Each patient contributes one row per ICU hour with vitals (HR, O₂Sat, Temp, SBP, MAP, DBP, Resp) and a sepsis label. Only **~1.8% of hours are positive**, so raw accuracy is meaningless — a model that never alarms is 98% accurate and useless. We evaluate on *patient-level* detection (did we alarm at or before onset?) and false-alarm burden.

## Approach

- **Features** — per patient-hour: forward-filled vitals, 6-hour rolling mean, 6-hour delta, and a per-vital "was it measured this hour" flag.
- **Split** — patient-level 60/20/20, stratified by sepsis, so no patient leaks across train/val/test.
- **Model** — Gaussian Naive Bayes, used as a *ranker*: the operating threshold is tuned on validation to hit a target catch rate.
- **Baselines** — SIRS clinical rule, Logistic Regression, HistGradientBoosting.

## Key results

At a matched catch rate (80% of septic patients detected), on the pooled test set:

| Model | ROC-AUC | PR-AUC (× base) | False-alarm hours (healthy patients) |
|---|---|---|---|
| SIRS rule | 0.62 | 0.028 (1.6×) | 52% |
| **Naive Bayes** | 0.66 | 0.036 (2.0×) | **17%** |
| LogReg | 0.66 | 0.037 (2.1×) | 20% |
| HistGB | 0.70 | 0.045 (2.5×) | 12% |

- Naive Bayes cuts false alarms ~**3×** vs the clinical SIRS rule at the same detection rate.
- **Cross-hospital (train A → test B):** the heavy model (HistGB) degrades most (PR-AUC −38%); Naive Bayes degrades least and even overtakes Logistic Regression. Fewer parameters → more robust to hospital shift.

## Repo layout

```
analysis/
  profile.py    # dataset profiling & missingness stats
  hourly.py     # builds per-patient-hourly feature table
  model.py      # baselines + models, splits, evaluation
training/
  training_setA/  training_setB/   # raw .psv files (not committed)
```

## Run

```bash
python analysis/profile.py AB        # dataset stats
python analysis/hourly.py A          # build hourly features (repeat for B)
python analysis/model.py             # train + evaluate all models
```

## Caveats

Absolute performance is modest (ROC ~0.65–0.70) — early sepsis prediction is genuinely hard, and these numbers are in line with the original challenge. The value is in *ranking* and *false-alarm reduction*, not raw predictive power. Naive Bayes scores are deliberately **not** treated as calibrated probabilities.

## Data

PhysioNet/CinC Challenge 2019, v1.0.0. Raw patient files are not included — download from PhysioNet and place under `training/`.
