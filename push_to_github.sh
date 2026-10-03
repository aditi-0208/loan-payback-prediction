#!/usr/bin/env bash
# Requires GitHub CLI: https://cli.github.com  (run `gh auth login` once first)
set -e
git init -b main
git add .
git commit -m "Initial commit: loan payback prediction (ML Fall 2025)"
gh repo create loan-payback-prediction --public --source=. --push \
  --description "Predicting loan repayment with Logistic Regression, Decision Tree, Random Forest & LightGBM (Kaggle Playground S5E11, best ROC-AUC 0.9212)"
gh repo edit --add-topic machine-learning --add-topic lightgbm --add-topic scikit-learn \
  --add-topic kaggle --add-topic credit-risk --add-topic classification --add-topic random-forest
