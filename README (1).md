# Detecting Fraud in E-Commerce Transactions

A notebook-based exploration and modelling project for detecting fraudulent transactions in an e-commerce dataset of ~1.47 million transactions, of which ~5% are labelled fraudulent.

Online retail makes it effortless to buy almost anything with a few taps — and that same convenience makes e-commerce a prime target for fraud. Stolen cards, account takeovers, and "friendly fraud" chargebacks cost merchants billions every year, while every false decline costs a legitimate sale. This project builds and compares several classifiers to catch fraudulent transactions without over-flagging honest customers.

## Dataset

Each row describes a single transaction, including:
- Transaction amount, quantity, and product category
- Payment method and device used
- Customer age and account age (in days)
- Transaction hour, shipping/billing address, IP address
- Label: `Is Fraudulent` (1 = fraud, 0 = genuine)

> The raw CSV (`Fraudulent_E-Commerce_Transaction_Data.csv`) is not included in this repo. Place it in the project root before running the notebook.

## What's inside the notebook

1. **Data loading & cleaning** — sanity checks for missing values/duplicates, fixing impossible negative customer ages.
2. **Exploratory data analysis** — fraud rate broken down by payment method, product category, device, account age, and hour of day; transaction amount distributions; a check on the "shipping ≠ billing address" fraud heuristic (which turns out not to hold in this dataset).
3. **Feature engineering** — derived features including:
   - `Is New Account` — account age ≤ 30 days
   - `Is Night` — transaction between 00:00–05:59
   - `Amount Per Item` — transaction amount ÷ quantity
   - One-hot encoded categorical fields (payment method, product category, device)
4. **Visualization** — correlation heatmaps (overall and split by class), boxplots, 2D "fingerprint" strip plots, and a 3D scatter of amount / account age / hour colored by fraud label.
5. **Modelling** — four classifiers trained on a class-balanced sample (all fraud cases + a random sample of genuine cases), evaluated with precision/recall/F1 for the fraud class, ROC-AUC, and average precision (AUPRC), which is more informative than ROC-AUC under strong class imbalance:
   - Logistic Regression
   - K-Nearest Neighbors
   - Random Forest
   - XGBoost
6. **Model comparison** — ROC and precision-recall curves for all four models side by side, plus feature importance plots for the tree-based models.

## Key findings

- Payment method, product category, and device used are close to noise on their own — fraud rates for each sit within a fraction of a point of the overall 5% baseline.
- **Account age** is a strong signal: accounts younger than 30 days show a fraud rate around **22%**, roughly 7x higher than older accounts (~3%).
- Fraud rate roughly **triples overnight** (00:00–05:59, ~10%) compared to daytime (~3%).
- Fraudulent transactions average about **$548** vs **$210** for genuine ones, with a much heavier tail.
- Shipping/billing address mismatch, despite being a common fraud heuristic, does **not** separate fraud from genuine activity in this dataset.
- Tree-based ensembles (Random Forest, XGBoost) are expected to outperform the linear and distance-based baselines by best exploiting the engineered features.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
