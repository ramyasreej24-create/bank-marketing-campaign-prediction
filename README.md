
# Bank Marketing Campaign Prediction

Predicting whether a client will subscribe to a term deposit based on personal,
financial, and campaign-related information, using the UCI Bank Marketing dataset.

## Problem Statement

A bank runs telemarketing campaigns to sell term deposits. Calling every customer
is expensive and inefficient. This project builds a binary classification model
to predict whether a customer will subscribe (`y = yes/no`), so the bank can:

- Reduce unnecessary calls → save cost and time
- Target the right customers → increase conversion rate
- Use data-driven insights to improve campaign strategy

## Dataset

- **Source:** UCI Bank Marketing dataset (`bank-full.csv` / `bank.csv`)
- **Records:** 45,211 total (after removing duplicates between the provided
  `train.csv` and `test.csv` — see [Data Leakage Note](#data-leakage-note) below)
- **Features:** 16 input features + 1 target (`y`)
- **Class balance:** ~88.3% "no" / ~11.7% "yes" → imbalanced classification problem

| Category | Columns |
|---|---|
| Personal | `age`, `job`, `marital`, `education` |
| Financial | `default`, `balance`, `housing`, `loan` |
| Campaign contact | `contact`, `day`, `month`, `duration` |
| Campaign history | `campaign`, `pdays`, `previous`, `poutcome` |
| Target | `y` |

## Data Leakage Note

The provided `test.csv` is **not an independent holdout** — every one of its 4,521
rows is an exact duplicate of a row already in `train.csv` (this matches the
well-known UCI setup where the smaller file is a 10% sample drawn *from* the full
file, not a separate split). Using it as-is inflates results (a Random Forest
scored a suspicious 1.00 ROC-AUC by memorizing rows it had already seen).

**Fix applied:** the two files are combined, exact duplicates are dropped, and a
fresh stratified 80/20 train/test split is created before any modeling. All
results below are from this corrected split.

## Key Modeling Decision: `duration` Causes Leakage

`duration` (length of the last call, in seconds) is only known **after** the call
happens — it can't be used to decide whether to call someone in the first place.
Including it inflates model performance and produces a model that isn't usable
for real campaign targeting. Two experiments are run:

- **Without `duration`** → the realistic, deployable model for deciding who to call
- **With `duration`** → an upper-bound / post-call analysis model, for reference only

## Approach

1. **EDA** — target imbalance, distribution checks, `pdays = -1` sentinel handling
2. **Feature engineering** — `was_contacted_before` flag derived from `pdays`
3. **Preprocessing** — one-hot encoding for categoricals, standard scaling for numerics
4. **Models compared** — Logistic Regression, Random Forest, Gradient Boosting
5. **Evaluation metrics** — ROC-AUC and PR-AUC (not accuracy, given class imbalance),
   full classification report, confusion matrix
6. **Explainability** — feature importance for the best-performing model in each experiment

## Results

### Realistic model — without `duration` (usable before calling)

| Model | ROC-AUC | PR-AUC | Recall (yes) |
|---|---|---|---|
| Logistic Regression | 0.772 | 0.409 | 0.62 |
| Random Forest | 0.791 | 0.433 | 0.21 |
| **Gradient Boosting** | **0.802** | 0.460 | 0.20 |

**Top predictive features:** `poutcome = success` (outcome of a prior campaign is
by far the strongest signal), `age`, `pdays`, `day`, `contact` type, `month`.

### Upper-bound model — with `duration` (post-call analysis only)

| Model | ROC-AUC | PR-AUC | Recall (yes) |
|---|---|---|---|
| Logistic Regression | 0.908 | 0.538 | 0.81 |
| **Random Forest** | **0.929** | 0.619 | 0.33 |
| Gradient Boosting | 0.924 | 0.598 | 0.41 |

`duration` dominates feature importance here — it acts as a proxy for customer
engagement/interest, confirming why it can't be used for pre-call targeting.

## Business Takeaway

For actual campaign targeting, use the **without-duration model**. There is a
precision/recall trade-off across models:

- Gradient Boosting has the best overall ranking ability (ROC-AUC) but only
  catches ~20% of true subscribers (low recall).
- Logistic Regression catches more true subscribers (62% recall) at the cost of
  more wasted calls to non-subscribers.

Which model is "better" depends on the relative cost of a wasted call vs. a
missed subscriber — a business decision, not just a modeling one. The
classification threshold can be tuned once that cost trade-off is defined.

## Project Structure

```
├── bank_term_deposit_model.py   # Full training + evaluation pipeline
├── train.csv                     # Raw data (bank-full.csv equivalent)
├── test.csv                      # Raw data (10% sample, deduplicated before use)
├── output/
│   ├── best_model_without_duration.pkl
│   ├── best_model_with_duration.pkl
│   ├── feature_importance_without_duration.png
│   └── feature_importance_with_duration.png
└── README.md
```

## How to Run

```bash
pip install scikit-learn pandas matplotlib joblib

python bank_term_deposit_model.py \
    --train train.csv \
    --test test.csv \
    --out ./output
```

The script will:
1. Combine and de-duplicate the input files
2. Create a proper stratified train/test split
3. Train and evaluate Logistic Regression, Random Forest, and Gradient Boosting
   — both with and without `duration`
4. Save the best model from each experiment as a `.pkl` file
5. Save a feature-importance chart for each experiment

## Next Steps

- Hyperparameter tuning (grid/random search) on Gradient Boosting
- SHAP-based explainability for individual predictions
- Cost-based threshold tuning once call cost vs. missed-subscriber cost is defined
- Deployment as a scoring API to rank customers before each calling campaign
