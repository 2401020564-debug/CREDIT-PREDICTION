import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
model = joblib.load('credit_model.pkl')

# 2. Build the UI (Title and Description)
st.title("🏦 Give Me Some Credit: Default Predictor")
st.write("Enter the borrower's details below to predict if they will experience financial distress in the next 2 years.")

# 3. Create input boxes for the user to type in data
age = st.number_input("Age", min_value=18, max_value=100, value=40)
monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0)
dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=1)
debt_ratio = st.number_input("Debt Ratio (e.g., 0.5 for 50%)", min_value=0.0, value=0.3)
unsecured_lines = st.number_input("Revolving Utilization of Unsecured Lines", min_value=0.0, value=0.1)
open_credit_lines = st.number_input("Number of Open Credit Lines and Loans", min_value=0, value=5)
real_estate_loans = st.number_input("Number of Real Estate Loans", min_value=0, value=1)
time_30_59_days_late = st.number_input("Times 30-59 Days Late", min_value=0, value=0)
time_60_89_days_late = st.number_input("Times 60-89 Days Late", min_value=0, value=0)
time_90_days_late = st.number_input("Times 90+ Days Late", min_value=0, value=0)

# 4. Predict Button
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

    # Make the prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    # Show the result!
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk! The model predicts a Default. (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk. The model predicts No Default. (Probability: {probability:.2%})")