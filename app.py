import streamlit as st
import pandas as pd
import xgboost as xgb

# Replace the joblib.load line with this:
model = xgb.XGBClassifier()
model.load_model('credit_model.json')
# --- 1. PAGE SETUP & CSS ---
st.set_page_config(page_title="Credit Predictor", layout="wide")

custom_css = """
<style>
/* Gradient Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
}

/* Make the header transparent */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Style the Input Boxes to look like elevated cards */
div[data-testid="stNumberInput"] {
    background-color: #334155;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
    border-left: 4px solid #475569; 
}

/* Style the Predict Button */
div.stButton > button:first-child {
    background-color: #d4af37; /* Premium Gold */
    color: #111827;
    font-weight: bold;
    font-size: 18px;
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
    width: 100%;
}

/* Button Hover Effect */
div.stButton > button:first-child:hover {
    background-color: #f1c40f;
    transform: translateY(-2px);
    transition: 0.2s;
}
</style>
"""
# --- SIDEBAR: MODEL INFO ---
with st.sidebar:
    st.header("⚙️ Model Specifications")
    st.info("**Algorithm:** XGBoost Classifier")
    st.write("XGBoost was chosen for its ability to handle imbalanced financial data and non-linear relationships.")
    
    st.markdown("---")
    st.header("📊 Training Metrics")
    st.success("**ROC-AUC:** 0.8686")
    st.warning("**F1-Score:** 0.7361") # Using your balanced sweet-spot score
    
    st.markdown("---")
    st.header("🛠️ Techniques Used")
    st.write("- **Handling Imbalance:** SMOTE Over-sampling")
    st.write("- **Optimization:** Custom Probability Thresholding")
    st.write("- **Feature Scaling:** RobustScaler")
    
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. BUILD THE UI HEADER ---
st.title("🏦 Give Me Some Credit: Default Predictor")
st.write("Enter the borrower's details below to predict if they will experience financial distress in the next 2 years.")
st.markdown("---")

# --- 4. CREATE ORGANIZED INPUT CARDS ---
st.subheader("👤 Borrower Demographics")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
with col2:
    monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0)
with col3:
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=1)

st.markdown("###") # Spacing
st.subheader("📊 Financial Ratios")
col4, col5 = st.columns(2)
with col4:
    debt_ratio = st.number_input("Debt Ratio (e.g., 0.5 for 50%)", min_value=0.0, value=0.3)
with col5:
    unsecured_lines = st.number_input("Revolving Utilization of Unsecured Lines", min_value=0.0, value=0.1)

st.markdown("###") # Spacing
st.subheader("💳 Payment History & Open Lines")
col6, col7, col8 = st.columns(3)
with col6:
    open_credit_lines = st.number_input("Number of Open Credit Lines and Loans", min_value=0, value=5)
    real_estate_loans = st.number_input("Number of Real Estate Loans", min_value=0, value=1)
with col7:
    time_30_59_days_late = st.number_input("Times 30-59 Days Late", min_value=0, value=0)
    time_60_89_days_late = st.number_input("Times 60-89 Days Late", min_value=0, value=0)
with col8:
    time_90_days_late = st.number_input("Times 90+ Days Late", min_value=0, value=0)

st.markdown("###") # Spacing before the button

# --- 5. PREDICTION LOGIC ---
if st.button("Predict Default Risk"):
    # Group the inputs into a dataframe that matches our training data
    input_data = pd.DataFrame([[
        unsecured_lines, age, time_30_59_days_late, debt_ratio, monthly_income, 
        open_credit_lines, time_90_days_late, real_estate_loans, 
        time_60_89_days_late, dependents
    ]], columns=[
        'RevolvingUtilizationOfUnsecuredLines', 'age', 
        'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ])

    # Get the exact probability
    probability = model.predict_proba(input_data)[0][1]

    # Apply your winning Sweet Spot threshold!
    optimal_threshold = 0.26 

    # Show the result using the new smart logic!
    st.markdown("---")
    if probability >= optimal_threshold:
        st.error(f"⚠️ High Risk! The model predicts a Default. (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk. The model predicts No Default. (Probability: {probability:.2%})")




