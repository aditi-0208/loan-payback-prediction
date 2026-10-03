# Loan Payback Prediction

Predicting whether a loan applicant will repay or default, using Logistic Regression, Decision Tree, Random Forest, and LightGBM on the Kaggle Playground Series S5E11 dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-boosting-green.svg)
![Best ROC-AUC](https://img.shields.io/badge/Best%20ROC--AUC-0.9212-brightgreen.svg)

Final project for **Machine Learning, Fall 2025** .

## Overview

Accurate credit-risk prediction helps lenders make better decisions. This project builds and compares four classifiers of increasing complexity on roughly 594k labeled loan records, evaluated with **ROC-AUC** (the competition metric), with accuracy as a secondary measure.

**Competition:** [Kaggle Playground Series S5E11 – Predicting Loan Payback](https://www.kaggle.com/competitions/playground-series-s5e11)

## Dataset

| | |
|---|---|
| Train rows | 593,994 |
| Test rows | 254,569 |
| Features | 11 (5 numeric, 6 categorical) |
| Target | `loan_paid_back` (1 = repaid, 0 = defaulted) |
| Class balance | ~79.9% repaid / ~20.1% defaulted |
| Missing values | None |

- **Numeric:** `annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`
- **Categorical:** `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade`

The data is not included in this repo. Download it from Kaggle (see [Getting started](#getting-started)).

## Approach

1. **EDA:** class balance, feature types, correlation heatmap, boxplots. `credit_score` (+0.23) and `interest_rate` (-0.13) are the numeric features most correlated with repayment.
2. **Preprocessing:** drop `id`, label-encode categoricals, stratified 80/20 train/validation split (`random_state=42`).
3. **Models:**
   - Logistic Regression: linear baseline
   - Decision Tree (`max_depth=6`): simple nonlinear model
   - Random Forest (200 trees): bagging ensemble
   - LightGBM (`learning_rate=0.02`, up to 2000 rounds, early stopping): boosting ensemble
4. **Submission:** predicted probabilities for the test set, submitted to Kaggle.

## Results

Validation set (118,799 rows):

| Model | ROC-AUC | Accuracy |
|---|---|---|
| Logistic Regression | 0.8391 | 0.8896 |
| Decision Tree | 0.9089 | 0.9019 |
| Random Forest | 0.9106 | 0.9012 |
| **LightGBM** | **0.9212** | ~0.900 |

**Takeaways**

- LightGBM gave the best ROC-AUC. Tree ensembles clearly beat the linear baseline, which points to important nonlinear effects and feature interactions.
- Accuracy is nearly identical across the tree-based models, so ROC-AUC is the more informative metric here.
- With ~80% of loans repaid, accuracy is easy to inflate. All models catch most repayers (recall about 0.97 to 0.99) but find only about 56% to 59% of defaulters, which is the main weakness to improve.

## Repository structure

```
loan-payback-prediction/
├── notebooks/
│   └── ML_Project_TeamXA.ipynb   # EDA, preprocessing, modeling, Kaggle submission
├── report/
│   └── Loan_Payback_Prediction_Report.docx
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Getting started

The notebook was written for **Google Colab** with data stored on Google Drive.

### Option A: Google Colab (as originally run)

1. Upload `notebooks/ML_Project_TeamXA.ipynb` to Colab.
2. Put your `kaggle.json` API token in `MyDrive/Kaggle_S5E11_LoanPayback/` (Kaggle → Settings → Create New Token).
3. Run all cells. The notebook mounts Drive, downloads the competition data, trains the models, and writes submission CSVs.

### Option B: Run locally

```bash
git clone https://github.com/aditi-0208/loan-payback-prediction.git
cd loan-payback-prediction
pip install -r requirements.txt

# download data (requires Kaggle API credentials and accepting the competition rules)
kaggle competitions download -c playground-series-s5e11 -p data
unzip data/playground-series-s5e11.zip -d data
jupyter notebook notebooks/ML_Project_TeamXA.ipynb
```

Skip the Colab-only cells (Drive mount, `!kaggle` setup) and point `train_path` / `test_path` at your local `data/` folder.

## Limitations and future work

- **Test-set encoding:** the test set is label-encoded with a fresh `LabelEncoder` per column instead of reusing the encoders fitted on the training data. This only gives consistent codes if every category appears in both sets. Fitting encoders on train and reusing them (or using an `OrdinalEncoder` / LightGBM's native categorical handling) would be safer.
- **Hyperparameters:** values were set by hand; no cross-validated search was run.
- **Ideas to improve:** feature engineering (for example income-to-loan ratios, splitting `grade_subgrade` into grade and subgrade), k-fold cross-validation, Optuna tuning, XGBoost/CatBoost, stacking, and threshold tuning or class weighting to improve default recall.

## Team

Team XA: *Aditi Malik, Xeniya Kyrychenko*

## References

1. Kaggle Playground Series S5E11 – Predicting Loan Payback
2. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR (2011)
3. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS (2017)

## License

Released under the MIT License. See [LICENSE](LICENSE).
