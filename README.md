# Nigerian Banking Fraud Detection System
### Team 17 — DA204o: Data Science in Practice

## Team Members
- **Swarup E** — swarupe@iisc.ac.in — 13-19-02-19-52-25-1-26456
- **Novoneel C** — cnovoneel@iisc.ac.in — 13-19-02-19-52-25-1-26197
- **Sarthak S** — sarthak1@iisc.ac.in — 13-19-02-19-52-25-1-26177
- **Rakshit R** — rrakshit@iisc.ac.in — 13-19-02-19-52-25-1-26334

## Problem Statement
This project develops a machine-learning–based fraud detection system for Nigerian banking transactions using the Nigeria Inter-Bank Settlement System (NIBSS) dataset. The system combines LSTM-derived temporal sequence features with gradient boosting models (LightGBM, XGBoost, CatBoost), further optimized using Optuna. The goal is to achieve high detection accuracy while maintaining extremely low false positive rates to ensure smooth customer experience and regulatory compliance.

## Dataset Description
The project uses the NIBSS Fraud Dataset (Kaggle) — a synthetic dataset designed to realistically reflect Nigerian banking transaction patterns while preserving data privacy. It contains 1 million transaction records with a blend of numeric, categorical, and temporal attributes stored in CSV format.

Each transaction includes details such as:
- Transaction amount, type, and channel
- Available balance and merchant category
- Device and location information
- Customer demographics and account age
- Transaction frequency and behavioral patterns
- Historical fraud indicators and social engineering signals
- Ground-truth target label: “Is Fraud”

## High-Level Approach & Methods
The project follows an end-to-end fraud detection workflow:

1. **Exploratory Data Analysis (EDA):**
Understanding transaction patterns, temporal trends, and severe class imbalance.

2. **Feature Engineering:**
- Transformative features (categorical encodings, amount scaling, deviations)
- Temporal and velocity-based features
- Interaction and risk-composite features
- **LSTM-generated sequence embeddings** to model user transaction patterns

3. **Modeling:**
Training LightGBM, XGBoost, and CatBoost models to learn non-linear fraud patterns.

4. **Ensembling:**
Combining models to leverage complementary strengths and reduce variance.

5. **Handling Imbalance:**
Using weighted losses, scale_pos_weight, and threshold tuning to improve recall while maintaining extremely low false positives.

6. **Optimization:**
**Optuna** is used for hyperparameter tuning and learning optimal ensemble voting weights.

## Summary of Results
The final ensemble model delivers strong performance across all key metrics:

- **AUC-ROC:** 0.96+
- **F1-Score:** 0.8847 (98% of the project target)
- **False Positive Rate:** 0.001%
- **Precision:** Extremely high, ensuring minimal customer disruption

Incorporating **LSTM-based temporal embeddings** provides a modest improvement (~1%), while most predictive power comes from engineered behavioral, temporal, and interaction features. Overall, the system reliably detects high-risk transactions at scale and significantly outperforms traditional rule-based approaches, demonstrating strong applicability to Nigerian banking fraud detection.
