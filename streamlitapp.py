# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 12:50:53 2025

@author: sammy
"""


import streamlit as st
import pandas as pd
import joblib

# Load the trained model
with open("salary_bracket_logistic_model.pkl", "rb") as file:
    model = joblib.load(file)

# Title
st.title("💼💰 Salary Bracket Classifier")

# Description
st.write("🔍 This app predicts whether a data scientist’s salary is **Low**, **Medium**, or **High** based on your background and experience. Fill in your details below! 👇")

# Input fields
country = st.selectbox("🌎 Country", ["United States", "India", "United Kingdom", "Germany", "Canada", "Other"])
gender = st.selectbox("🧑‍🤝‍🧑 Gender", ["Man", "Woman", "Prefer not to say", "Other"])
education = st.selectbox("🎓 Education Level", [
    "Bachelor’s degree", 
    "Master’s degree", 
    "Doctoral degree", 
    "Some college/university study without earning a bachelor’s degree",
    "No formal education past high school",
    "Professional degree",
    "Other"
])
years_coding = st.selectbox("💻 Years of Coding Experience", [
    "< 1 year", "1-2 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"
])
job_title = st.selectbox("🧑‍💼 Job Title", [
    "Data Scientist", "Machine Learning Engineer", "Research Scientist",
    "Data Analyst", "Statistician", "Business Analyst", "Other"
])

# Prediction
if st.button("🔮 Predict Salary Midpoint"):
    input_data = pd.DataFrame({
        "Country": [country],
        "Gender": [gender],
        "Education_Level": [education],
        "Years_Coding": [years_coding],
        "Job_Title": [job_title]
    })

    # One-hot encode to match the model's training process
    input_enc = pd.get_dummies(input_data)

    # Ensure all features the model expects are present
    for feat in model.feature_names_in_:
        if feat not in input_enc.columns:
            input_enc[feat] = 0
    input_enc = input_enc[model.feature_names_in_]

    # Predict the numeric salary midpoint
    salary_pred = model.predict(input_enc)[0]

    # Display the result
    st.success(f"🎉 Predicted Salary Midpoint: **${int(salary_pred):,}** USD")
