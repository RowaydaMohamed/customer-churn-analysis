import streamlit as st
import pandas as pd
import numpy  as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express       as px

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Customer Churn Predictor",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #1a1a2e; margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem; color: #6b6b6b; margin-top: 0;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1.2rem 1.5rem; border: 1px solid #e5e5e5;
    }
    .risk-high   { color: #E8654C; font-weight: 700; font-size: 1.3rem; }
    .risk-medium { color: #F5A623; font-weight: 700; font-size: 1.3rem; }
    .risk-low    { color: #4C9BE8; font-weight: 700; font-size: 1.3rem; }
    .driver-tag {
        background: #fff3f0; color: #E8654C;
        border-radius: 6px; padding: 2px 10px;
        font-size: 0.85rem; margin: 2px;
        display: inline-block;
    }
    .stButton > button {
        background: #E8654C; color: white;
        border-radius: 8px; border: none;
        font-weight: 600; font-size: 1rem;
        padding: 0.6rem 2rem; width: 100%;
    }
    .stButton > button:hover { background: #c9442b; }
</style>
""", unsafe_allow_html=True)

# ── Load model artefacts ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Load the trained model, scaler, threshold, and feature names.
    Looks for artefacts in ./models/ relative to this script.
    In production these would come from cloud storage (S3, GCS, etc).
    """
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        model     = joblib.load(os.path.join(base, "models", "best_model.pkl"))
        scaler    = joblib.load(os.path.join(base, "models", "scaler_final.pkl"))
        with open(os.path.join(base, "models", "threshold.txt")) as f:
            threshold = float(f.read().strip())
        with open(os.path.join(base, "processed", "feature_names.txt")) as f:
            feature_names = f.read().splitlines()
        with open(os.path.join(base, "models", "model_info.txt")) as f:
            model_info = dict(
                line.split(": ", 1)
                for line in f.read().splitlines()
                if ": " in line
            )
        return model, scaler, threshold, feature_names, model_info
    except FileNotFoundError:
        return None, None, 0.5, [], {}

model, scaler, THRESHOLD, FEATURE_NAMES, model_info = load_model()

# ── Header ────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 11])
with col_title:
    st.markdown('<p class="main-header">📊 Customer Churn Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter customer details below to predict churn probability and get personalised retention recommendations.</p>', unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar — about ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 About this tool")
    st.markdown("""
    This app predicts the probability that a telecom customer will churn
    within the next billing cycle.

    **How it works:**
    1. Fill in the customer details on the right
    2. Click **Predict Churn Risk**
    3. Review the risk score, probability, and recommendations

    **Model performance:**
    """)
    if model_info:
        st.metric("ROC-AUC",  model_info.get("test_roc_auc", "—"))
        st.metric("Recall",   f"{float(model_info.get('test_recall', 0)):.1%}")
        st.metric("Model",    model_info.get("model_name", "—"))
    else:
        st.warning("Model artefacts not found. Place model files in ./models/")


# ── Input form ────────────────────────────────────────────────────────
st.markdown("### Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Account Information**")
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    contract        = st.selectbox("Contract type",
                                   ["Month-to-Month", "One Year", "Two Year"])
    monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0, 0.5)
    payment_method  = st.selectbox("Payment method",
                                   ["Electronic check", "Mailed check",
                                    "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless_billing = st.checkbox("Paperless billing", value=True)

with col2:
    st.markdown("**Service Profile**")
    internet_service  = st.selectbox("Internet service",
                                     ["Fiber optic", "DSL", "No"])
    phone_service     = st.checkbox("Phone service", value=True)
    multiple_lines    = st.checkbox("Multiple lines", value=False)
    online_security   = st.checkbox("Online security", value=False)
    online_backup     = st.checkbox("Online backup", value=False)
    device_protection = st.checkbox("Device protection", value=False)
    tech_support      = st.checkbox("Tech support", value=False)
    streaming_tv      = st.checkbox("Streaming TV", value=False)
    streaming_movies  = st.checkbox("Streaming movies", value=False)

with col3:
    st.markdown("**Behavioural Signals**")
    senior_citizen        = st.checkbox("Senior citizen", value=False)
    partner               = st.checkbox("Has partner", value=False)
    dependents            = st.checkbox("Has dependents", value=False)
    total_logins          = st.slider("Total logins (lifetime)", 0, 500, 80)
    days_since_last_login = st.slider("Days since last login", 0, 365, 15)
    total_tickets         = st.slider("Support tickets (total)", 0, 20, 1)
    billing_tickets       = st.slider("Billing tickets", 0, 10, 0)
    late_payment_rate     = st.slider("Late payment rate", 0.0, 1.0, 0.0, 0.05)
    avg_payment_gap       = st.slider("Avg payment gap ($)", 0.0, 50.0, 0.0, 0.5)

st.markdown("---")

# ── Predict ────────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True)

if predict_btn:
    if model is None:
        st.error("Model files not found. Please place best_model.pkl, scaler_final.pkl, "
                 "threshold.txt, and feature_names.txt in the ./models/ directory.")
    else:
        # Build feature vector from inputs
        # This must match the feature engineering in notebook 03 exactly
        contract_enc = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}[contract]
        num_services = sum([online_security, online_backup, device_protection,
                            tech_support, streaming_tv, streaming_movies])

        # Derived features
        total_charges     = tenure * monthly_charges
        avg_monthly_spend = (total_charges / tenure) if tenure > 0 else monthly_charges
        login_days_active = max(1, total_logins // 5)
        logins_per_day    = total_logins / login_days_active if login_days_active > 0 else 0

        # Internet service and payment method dummies
        # (these must match the drop_first=True encoding from notebook 03)
        # Dropped references: internet_service=DSL, payment_method=Bank transfer
        is_fiber    = 1 if internet_service == "Fiber optic" else 0
        is_no_inet  = 1 if internet_service == "No" else 0
        is_mailed   = 1 if payment_method == "Mailed check" else 0
        is_elec     = 1 if payment_method == "Electronic check" else 0
        is_credit   = 1 if payment_method == "Credit card (automatic)" else 0

        # Charge vs tier average (approximation for app — exact values require full dataset)
        tier_avgs = {"Fiber optic": 80.0, "DSL": 55.0, "No": 22.0}
        charge_vs_tier = monthly_charges - tier_avgs.get(internet_service, 65.0)

        risk_score = (
            (1 if contract == "Month-to-Month" else 0) +
            (1 if tenure < 12 else 0) +
            (1 if late_payment_rate > 0 else 0) +
            (1 if total_tickets > 2 else 0)
        )

        # Assemble into a dict that covers all expected features
        # Note: this mapping must be verified against feature_names.txt
        feature_dict = {
            "gender":               0,   # neutral default
            "senior_citizen":       int(senior_citizen),
            "partner":              int(partner),
            "dependents":           int(dependents),
            "tenure":               tenure,
            "phone_service":        int(phone_service),
            "multiple_lines":       int(multiple_lines),
            "online_security":      int(online_security),
            "online_backup":        int(online_backup),
            "device_protection":    int(device_protection),
            "tech_support":         int(tech_support),
            "streaming_tv":         int(streaming_tv),
            "streaming_movies":     int(streaming_movies),
            "paperless_billing":    int(paperless_billing),
            "monthly_charges":      monthly_charges,
            "avg_session_mins":     total_logins * 2.5,  # approximation
            "avg_resolution_hours": 0,
            "max_resolution_hours": 0,
            "avg_payment_gap":      avg_payment_gap,
            "max_payment_gap":      avg_payment_gap * 1.5,
            "avg_amount_due":       monthly_charges,
            "total_logins":         total_logins,
            "login_days_active":    login_days_active,
            "days_since_last_login":days_since_last_login,
            "total_tickets":        total_tickets,
            "billing_tickets":      billing_tickets,
            "technical_tickets":    max(0, total_tickets - billing_tickets),
            "total_bills":          max(1, tenure),
            "total_late_payments":  int(late_payment_rate * max(1, tenure)),
            "late_payment_rate":    late_payment_rate,
            "billing_ticket_ratio": (billing_tickets / total_tickets) if total_tickets > 0 else 0,
            "avg_monthly_spend":    avg_monthly_spend,
            "charge_vs_tier_avg":   charge_vs_tier,
            "logins_per_active_day":logins_per_day,
            "num_services":         num_services,
            "risk_score":           risk_score,
            "contract_encoded":     contract_enc,
            # One-hot dummies
            "internet_service_Fiber Optic": is_fiber,
            "internet_service_No":          is_no_inet,
            "payment_method_Credit Card (Automatic)": is_credit,
            "payment_method_Electronic Check":        is_elec,
            "payment_method_Mailed Check":            is_mailed,
        }

        # Build input DataFrame aligned to feature_names
        if FEATURE_NAMES:
            row = pd.DataFrame([{k: feature_dict.get(k, 0) for k in FEATURE_NAMES}])
        else:
            row = pd.DataFrame([feature_dict])

        # Predict
        try:
            proba     = float(model.predict_proba(row)[0, 1])
            predicted = int(proba >= THRESHOLD)

            # Risk tier
            if proba >= 0.60:
                tier, tier_class = "HIGH", "risk-high"
            elif proba >= 0.35:
                tier, tier_class = "MEDIUM", "risk-medium"
            else:
                tier, tier_class = "LOW", "risk-low"

            # ── Result layout ────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Churn Probability", f"{proba:.1%}")
            with r2:
                st.markdown(f'<div class="metric-card">Risk Tier<br>'
                             f'<span class="{tier_class}">{tier}</span></div>',
                             unsafe_allow_html=True)
            with r3:
                st.metric("Decision Threshold", f"{THRESHOLD:.3f}")
            with r4:
                flag = "🔴 Flag for retention" if predicted else "🟢 Low priority"
                st.metric("Recommendation", flag)

            # ── Gauge chart ───────────────────────────────────────────
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = proba * 100,
                title = {"text": "Churn Probability (%)", "font": {"size": 16}},
                number= {"suffix": "%", "font": {"size": 28}},
                gauge = {
                    "axis":  {"range": [0, 100]},
                    "bar":   {"color": "#E8654C" if proba >= 0.60
                               else "#F5A623"   if proba >= 0.35
                               else "#4C9BE8"},
                    "steps": [
                        {"range": [0,  35], "color": "#EBF5FB"},
                        {"range": [35, 60], "color": "#FEF9E7"},
                        {"range": [60,100], "color": "#FDEDEC"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "value": THRESHOLD * 100
                    }
                }
            ))
            fig.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

            # ── Key risk drivers ──────────────────────────────────────
            st.markdown("#### 🔑 Key Risk Factors for this Customer")
            drivers = []
            if contract == "Month-to-Month":
                drivers.append("Month-to-Month contract (no commitment)")
            if tenure < 12:
                drivers.append(f"New customer ({tenure} months — high early churn period)")
            if days_since_last_login >= 14:
                drivers.append(f"Disengaged — {days_since_last_login} days since last login")
            if late_payment_rate > 0:
                drivers.append(f"Payment distress — {late_payment_rate:.0%} late payment rate")
            if total_tickets > 2:
                drivers.append(f"High support volume — {total_tickets} tickets")
            if billing_tickets > 0:
                drivers.append(f"Billing complaints — {billing_tickets} billing ticket(s)")
            if num_services == 0:
                drivers.append("Zero add-on services — low switching cost")
            if internet_service == "Fiber optic" and monthly_charges > 80:
                drivers.append("High-value Fiber optic customer with elevated charges")

            if drivers:
                driver_html = " ".join([f'<span class="driver-tag">⚠ {d}</span>'
                                         for d in drivers])
                st.markdown(driver_html, unsafe_allow_html=True)
            else:
                st.success("No significant risk factors identified.")

            # ── Personalised recommendations ─────────────────────────
            st.markdown("#### 💡 Recommended Actions")

            if tier == "HIGH":
                st.error("**Immediate action required.** Assign to retention team today.")
                if contract == "Month-to-Month":
                    st.markdown("→ Offer a **15% discount** to upgrade to a 1-year contract")
                if days_since_last_login >= 14:
                    st.markdown(f"→ Send a **personalised re-engagement email** — last active {days_since_last_login} days ago")
                if billing_tickets > 0:
                    st.markdown("→ **Escalate billing ticket** to senior support team")
                if num_services == 0:
                    st.markdown("→ Offer a **free 3-month trial** of Online Security or Backup")

            elif tier == "MEDIUM":
                st.warning("**Monitor closely.** Include in next retention campaign.")
                if days_since_last_login >= 7:
                    st.markdown("→ Add to automated **re-engagement email sequence**")
                if late_payment_rate > 0:
                    st.markdown("→ Proactive outreach to offer **flexible payment plan**")
                if num_services < 2:
                    st.markdown("→ Include in **add-on upsell campaign** to increase stickiness")

            else:
                st.success("**Low risk.** Include in standard loyalty programme.")
                st.markdown("→ Send **loyalty milestone reward** if tenure is a multiple of 12 months")
                st.markdown("→ Offer **referral programme** — loyal customers are best ambassadors")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Verify that feature_names.txt matches the model's expected input features.")

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#aaa'>Customer Churn Analysis Project | "
    "Built with Python, scikit-learn, Streamlit | "
    "Model: Logistic Regression + SMOTE | ROC-AUC: 0.821</small>",
    unsafe_allow_html=True
)
