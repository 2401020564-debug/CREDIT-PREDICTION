import streamlit as st
import pandas as pd
import xgboost as xgb

# --- 1. PAGE SETUP & PREMIUM CSS ---
st.set_page_config(page_title="Credit Risk AI", layout="wide")

# This block applies the background, glassmorphism, and custom result styles
st.markdown("""
<style>
    /* Premium Mesh Gradient Background */
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
        background-image: 
            radial-gradient(at 0% 0%, rgba(30, 58, 138, 0.5) 0, transparent 50%), 
            radial-gradient(at 50% 0%, rgba(15, 23, 42, 1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, rgba(30, 58, 138, 0.5) 0, transparent 50%);
        color: #f8fafc;
    }

    /* Glassmorphism Cards for Inputs */
    div[data-testid="stNumberInput"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        transition: transform 0.2s ease;
    }

    /* Gold "Predict" Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #d4af37 0%, #f1c40f 100%);
        color: #0f172a;
        font-weight: 800;
        border: none;
        padding: 15px;
        border-radius: 12px;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Attractive Prediction Box Styling */
    .prediction-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 15px;
        padding: 25px;
        margin-top: 25px;
        border: 2px solid;
        font-size: 22px;
        font-weight: bold;
        display: flex;
        align-items: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .prediction-icon {
        font-size: 35px;
        margin-right: 20px;
    }
    .prediction-text {
        color: #f8fafc;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
model = xgb.XGBClassifier()
model.load_model('credit_model.json')

# --- 3. SIDEBAR TECHNICAL SPECS ---
with st.sidebar:
    st.title("🛡️ Model Stats")
    st.info("**Algorithm:** XGBoost Classifier")
    st.write("Chosen for its superior handling of imbalanced financial data.")
    
    st.markdown("---")
    st.header("📊 Performance")
    st.metric(label="ROC-AUC Score", value="0.8686")
    st.metric(label="Balanced F1-Score", value="0.7361")
    
    st.markdown("---")
    st.header("🛠️ Techniques")
    st.write("- SMOTE Over-sampling")
    st.write("- Custom Thresholding")
    st.write("- Robust Feature Scaling")

# --- 4. MAIN UI CONTENT ---
st.title("🏦 Give Me Some Credit: Default Predictor")
st.write("Leveraging Gradient Boosting to analyze borrower financial reliability.")
st.markdown("---")

st.subheader("👤 Borrower Profile & Financials")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=1)

with col2:
    debt_ratio = st.number_input("Debt Ratio (e.g., 0.3 for 30%)", min_value=0.0, value=0.3)
    unsecured_lines = st.number_input("Credit Utilization", min_value=0.0, value=0.1)
    open_credit_lines = st.number_input("Open Credit Lines/Loans", min_value=0, value=5)

with col3:
    real_estate_loans = st.number_input("Real Estate Loans", min_value=0, value=1)
    late_30 = st.number_input("Times 30-59 Days Late", min_value=0, value=0)
    late_60 = st.number_input("Times 60-89 Days Late", min_value=0, value=0)

time_90_days_late = st.number_input("Times 90+ Days Late", min_value=0, value=0)

st.markdown("###")

# --- 5. PREDICTION LOGIC ---
if st.button("Predict Default Risk"):
    input_data = pd.DataFrame([[
        unsecured_lines, age, late_30, debt_ratio, monthly_income, 
        open_credit_lines, time_90_days_late, real_estate_loans, 
        late_60, dependents
    ]], columns=[
        'RevolvingUtilizationOfUnsecuredLines', 'age', 
        'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ])

    probability = model.predict_proba(input_data)[0][1]
    optimal_threshold = 0.26 

    st.markdown("---")
    if probability >= optimal_threshold:
        # Attractive Glowing High Risk Box
        st.markdown(f"""
        <div class="prediction-box" style="border-color: #ff4b4b; box-shadow: 0 0 20px rgba(255, 75, 75, 0.4);">
            <div class="prediction-icon">⚠️</div>
            <div class="prediction-text">High Risk! The model predicts a Default.<br><small>Probability: {probability:.2%}</small></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Attractive Glowing Low Risk Box
        st.markdown(f"""
        <div class="prediction-box" style="border-color: #2ecc71; box-shadow: 0 0 20px rgba(46, 204, 113, 0.4);">
            <div class="prediction-icon">✅</div>
            <div class="prediction-text">Low Risk. The model predicts No Default.<br><small>Probability: {probability:.2%}</small></div>
        </div>
        """, unsafe_allow_html=True)
