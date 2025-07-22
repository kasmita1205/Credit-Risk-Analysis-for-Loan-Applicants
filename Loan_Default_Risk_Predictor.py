#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd
# Load model
model = joblib.load('C:/Users/asmit\Credit_Risk_Analysis_for_Loan_Default_Applicants\model_xgb.pkl')


# Create all 33 features with default values
default_values = {
    'loan_amnt': 10000,
    'int_rate': 10,
    'annual_inc': 60000,
    'dti': 15.0,
    'fico_range_low': 700,
    'emp_length': 5,
    'term': 36,
    'credit_age':10,
    'revol_util': 30,
    'home_ownership_MORTGAGE': False,
    'home_ownership_NONE' : False,
    'home_ownership_OWN': False,
    'home_ownership_RENT': True,
    'verification_status_Source Verified': True,
    'verification_status_Verified': False,
    'purpose_credit_card': False,
    'purpose_debt_consolidation' : False,
 'purpose_home_improvement': False,
 'purpose_house': True,
 'purpose_major_purchase': False,
 'purpose_medical': False,
 'purpose_moving': False,
 'purpose_other': False,
 'purpose_renewable_energy': False,
 'purpose_small_business': False,
 'purpose_vacation': False,
 'purpose_wedding': False,
 'grade_B': False,
 'grade_C': False,
 'grade_D': False,
 'grade_E': False,
 'grade_F': True,
 'grade_G': False
}

# Create input form with key features
st.title('Loan Default Risk Predictor')

fico = st.slider('FICO Score', 300, 850, 700)
dti = st.slider('DTI Ratio', 0, 100, 15)
loan_amt = st.number_input('Loan Amount ($)', 1000, 40000, 15000)

# Create a DataFrame with ALL 33 features
input_data = pd.DataFrame([default_values])

# Update with user inputs
input_data['fico_range_low'] = fico
input_data['dti'] = dti
input_data['loan_amnt'] = loan_amt

# Predict Risk
if st.button('Predict'):
    risk = model.predict_proba(input_data)[0][1]
    risk = float(risk)
    st.success(f'Default Probability: {risk:.1%}')
    st.progress(risk)

