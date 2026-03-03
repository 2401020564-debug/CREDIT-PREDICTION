import streamlit as st
import pandas as pd
import xgboost as xgb # Ensure you use xgboost for .json models

model = xgb.XGBClassifier()
model.load_model('credit_model.json')

# --- 1. PAGE SETUP & PREMIUM CSS ---
st.set_page_config(page_title="Credit Risk AI", layout="wide")

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
    div[data-testid="stNumberInput"]:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(212, 175, 55, 0.4); /* Gold glow on hover */
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

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR TECHNICAL SPECS ---
with st.sidebar:
    st.title("🛡️ Model Stats")
    st.markdown("### Architecture")
    st.code("XGBoost Classifier")
    
    st.markdown("### Performance")
    st.metric(label="ROC-AUC", value="0.8686")
    st.metric(label="F1-Score", value="0.4361")
    
    st.markdown("---")
    st.write("**Methodology:** Trained using SMOTE for class balance and custom probability thresholding to optimize Precision.")

# --- 3. MAIN UI CONTENT ---
st.title("🏦 Give Me Some Credit: Default Predictor")
st.write("Leveraging Gradient Boosting to analyze borrower reliability.")
st.markdown("---")

# Use columns to organize the "Glass" cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### 👤 Profile")
    age = st.number_input("Age", min_value=18, value=40)
    monthly_income = st.number_input("Monthly Income ($)", value=5000.0)
    dependents = st.number_input("Dependents", value=1)

with col2:
    st.markdown("#### 📊 Ratios")
    debt_ratio = st.number_input("Debt Ratio", value=0.3)
    unsecured_lines = st.number_input("Credit Utilization", value=0.1)
    open_lines = st.number_input("Open Credit Lines", value=5)

with col3:
    st.markdown("#### ⏳ History")
    late_30 = st.number_input("30-59 Days Late", value=0)
    late_60 = st.number_input("60-89 Days Late", value=0)
    late_90 = st.number_input("90+ Days Late", value=0)

# Prediction Button and Logic remains the same...
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





