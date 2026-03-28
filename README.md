# 📊 Customer Churn Analysis — End-to-End ML Project

> Predicting telecom customer churn using machine learning — from raw data to a deployed Streamlit app.

---

## 🎯 Business Problem

A telecom company loses **~26% of its customers every year** to churn. Every churned customer represents roughly **$900 in lost annual revenue**. The company needs to know:

1. **Who** is likely to churn before they leave
2. **Why** they are at risk
3. **What** action to take for each customer

This project builds a complete ML pipeline that answers all three questions.

---

## 📊 Key Results

| Metric | Score |
|---|---|
| **ROC-AUC** | **0.821** |
| **PR-AUC** | **0.602** |
| **Recall** | **67.7%** |
| **Precision** | **55.1%** |
| **F1-Score** | **0.607** |

**Best model:** Logistic Regression with SMOTE oversampling  
**Decision threshold:** 0.484 (F1-optimal, selected on validation set)

The model **catches 68% of all churners** before they leave, with a false alarm rate of ~45% among flagged customers. At standard telco retention costs ($20/contact), the model generates a **positive ROI even at a 30% customer save rate**.

---

## 💡 Key Business Insights

| # | Insight | Implication |
|---|---|---|
| 1 | Month-to-Month customers churn at **42%** vs **3%** for Two Year | Contract upgrade is the single highest-impact retention lever |
| 2 | New customers (0–12 months) on Month-to-Month have **~60% churn rate** | First 90 days is the critical retention window |
| 3 | Customers inactive for **14+ days** churn at 2× the average | Automated re-engagement at day 7 and day 14 |
| 4 | Each add-on service subscribed reduces churn risk by **~5–8 pp** | Add-on bundling is a structural stickiness strategy |
| 5 | First late payment strongly predicts churn | Proactive payment support after first late bill |
| 6 | Customers with **all 4 risk factors** churn at **~80%** | Compound risk is non-linear — early intervention critical |

---

## 🗂 Project Structure

```
customer-churn-project/
│
├── notebooks/
│   ├── 01_data_collection.ipynb          # Load, simulate, inspect raw data
│   ├── 02_cleaning_merging.ipynb         # Clean, merge, validate master table
│   ├── 03_eda_feature_engineering.ipynb  # EDA, feature creation, scaling
│   ├── 04_modelling.ipynb                # Train, tune, evaluate, SHAP
│   └── 05_insights_recommendations.ipynb # Business insights + Power BI exports
│
├── app.py                                # Streamlit deployment app
├── requirements.txt                      # Python dependencies
│
├── data/
│   ├── raw/                              # Original unmodified datasets
│   ├── processed/                        # Cleaned master table + feature matrices
│
└── models/
    ├── best_model.pkl                    # Trained model
    ├── scaler_final.pkl                  # StandardScaler (fit on train only)
    ├── threshold.txt                     # Optimal decision threshold
    └── model_info.txt                    # Full metadata
```

---

## 📦 Data Sources

| Dataset | Source | Description |
|---|---|---|
| Telco Customer Churn | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | ~7,043 customers, 19 features, binary churn label |
| Online Retail II | [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | Used for EDA practice (different customer universe) |
| Usage logs | Simulated | Login sessions per Telco customer |
| Support tickets | Simulated | Support ticket history per customer |
| Billing history | Simulated | Monthly billing and payment records |

---

## ⚙️ Technical Pipeline

```
Raw Data (5 datasets)
    │
    ▼
Notebook 01: Data Collection
    • Load real Telco + Retail II datasets
    • Simulate usage, tickets, billing (same customer IDs)
    • Inject realistic messiness (nulls, casing, duplicates)
    │
    ▼
Notebook 02: Cleaning & Merging
    • Standardise customer_id to UPPERCASE across all tables (join key integrity)
    • Deduplicate, snake_case columns, fix dtypes
    • Grouped-median imputation with missing flags
    • Aggregate supporting tables → 1 row/customer
    • LEFT JOIN onto customers (preserves all 7,043 rows)
    │
    ▼
Notebook 03: EDA & Feature Engineering
    • Statistical tests: Mann-Whitney U (numeric) + Chi-squared + Cramér's V (categorical)
    • Spearman correlation heatmap
    • 5 composite features: avg_monthly_spend, charge_vs_tier_avg,
      logins_per_active_day, num_services, risk_score
    • Ordinal encoding (contract), one-hot (internet_service, payment_method)
    • StandardScaler (continuous + count features only)
    │
    ▼
Notebook 04: Modelling
    • 3-way stratified split: 64% train / 16% val / 20% test
    • Metrics: ROC-AUC + PR-AUC (primary), F1 + Recall (secondary)
    • 4 models via 5-fold CV: Logistic Regression, Random Forest, XGBoost, LightGBM
    • SMOTE vs class_weight comparison (best model only)
    • RandomizedSearchCV (50 iterations × 5 folds)
    • F1-optimal threshold on validation set
    • Test set opened exactly once
    • SHAP interpretation: beeswarm, bar, dependence, waterfall
    │
    ▼
Notebook 05: Insights & Recommendations
    • Revenue quantification ($revenue at risk)
    • 6 prioritised recommendations with business metrics
    • ROI estimate (conservative / moderate / optimistic)
    • Power BI-ready CSV exports (5 files)
```

---

## 🛠 How to Run

### Option A — Google Colab (recommended)

1. Upload notebooks to Google Colab
2. Mount your Google Drive
3. Download the Telco dataset from Kaggle and upload to `data/raw/`
4. Run notebooks 01 → 05 in order

### Option B — Local

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-analysis
cd customer-churn-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app (after running notebooks to generate model files)
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Fork this repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → connect your GitHub repo
4. Set main file path: `app.py`
5. Click Deploy

---

## 🔧 Feature Engineering Decisions

| Feature | Formula | Business Logic |
|---|---|---|
| `avg_monthly_spend` | `total_charges / tenure` | Normalises cumulative spend by tenure — separates high-recent-spend from legacy high-total |
| `charge_vs_tier_avg` | `monthly_charges − tier_mean` | Charges above peer average signal billing frustration |
| `logins_per_active_day` | `total_logins / login_days_active` | Separates habitual daily users from binge-then-disappear patterns |
| `num_services` | Count of add-on subscriptions | Proxy for switching cost and product stickiness |
| `risk_score` | Sum of 4 binary risk flags | Human-interpretable compound risk — directly usable for business triage |

---

## 📈 Why Logistic Regression Won

Logistic Regression outperformed all three gradient boosting models (Random Forest, XGBoost, LightGBM) on this dataset. This is a meaningful result, not a failure. The feature engineering in notebook 03 linearised the key relationships:

- `contract_encoded` captures the commitment gradient as an ordinal number
- `risk_score` combines the four strongest signals into one composite
- `charge_vs_tier_avg` removes absolute-price noise by normalising to peer group

Once those transformations are in place, the remaining signal is largely linear — which is exactly the regime where logistic regression is optimal and where gradient boosting's additional complexity adds overfitting rather than signal.

---

## 🚀 Deployment

The Streamlit app (`app.py`) provides:
- A form to input customer attributes
- Real-time churn probability prediction with a gauge chart
- Risk tier classification (High / Medium / Low)
- Personalised retention recommendations based on the specific risk factors detected
- Key driver tags showing *why* the customer was flagged


## 📋 Business Recommendations Summary

| Priority | Urgency | Segment | Action |
|---|---|---|---|
| 1 | IMMEDIATE | Month-to-Month, <12m tenure | Early Loyalty Programme — 15% discount to upgrade to 1-year |
| 2 | IMMEDIATE | Inactive 14+ days | Automated re-engagement at day 7 (warning) + day 14 (offer) |
| 3 | HIGH | Fiber optic, no add-ons | Proactive quality check at month 3 + free add-on bundle trial |
| 4 | HIGH | Any late payment | Flexible payment plan outreach after first late bill |
| 5 | MEDIUM | 2+ billing tickets | Escalate to senior support + <24h resolution SLA |
| 6 | MEDIUM | High-value ($80+/mo) | Dedicated account manager + quarterly check-in |

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Statistical tests | scipy (Mann-Whitney U, Chi-squared) |
| Machine learning | scikit-learn, XGBoost, LightGBM |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Interpretability | SHAP |
| Deployment | Streamlit |
| Environment | Google Colab / local Python |

---

## 📝 Licence

MIT Licence — see `LICENSE` for details.

---

*Built as a portfolio project demonstrating end-to-end data science: business understanding → data engineering → EDA → feature engineering → modelling → interpretation → deployment → business recommendations.*
