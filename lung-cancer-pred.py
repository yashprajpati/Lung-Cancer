# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:30:12 2025

@author: Asus
"""

import numpy as np
import pickle
import streamlit as st

# Load trained model
loaded_model = pickle.load(open('lung.sav', 'rb'))

# Prediction function
def cancer_prediction(input_data):
    input_data_as_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "The person did NOT survive"
    else:
        return "The person SURVIVED"

# Streamlit UI
def main():
    st.title("Lung Cancer Survival Prediction App")

    age = st.text_input("Age")
    gender = st.text_input("Gender (0=Female, 1=Male)")
    cancer_stage = st.text_input("Cancer Stage (1=Stage I, 2=Stage II, 3=Stage III, 4=Stage IV)")
    family_history = st.text_input("Family History of Cancer (0=No, 1=Yes)")
    smoking_status = st.text_input("Smoking Status (0=Current Smoker, 1=Former Smoker, 2=Never Smoked, 3=Passive Smoker)")
    bmi = st.text_input("Body Mass Index (BMI)")
    cholesterol_level = st.text_input("Cholesterol Level (mg/dL)")
    hypertension = st.text_input("Hypertension (0=No, 1=Yes)")
    asthma = st.text_input("Asthma (0=No, 1=Yes)")
    cirrhosis = st.text_input("Cirrhosis (0=No, 1=Yes)")
    other_cancer = st.text_input("Other Cancer (0=No, 1=Yes)")
    treatment_type = st.text_input("Treatment Type (0=Chemotherapy, 1=Combined, 2=Radiation, 3=Surgery)")

    if st.button("Result of Cancer Prediction"):
        try:
            input_list = [
                float(age),
                float(gender),
                float(cancer_stage),
                float(family_history),
                float(smoking_status),
                float(bmi),
                float(cholesterol_level),
                float(hypertension),
                float(asthma),
                float(cirrhosis),
                float(other_cancer),
                float(treatment_type),
            ]

            prediction = cancer_prediction(input_list)
            st.success(f"Cancer prediction result: {prediction}")

        except ValueError:
            st.error("❗ Please enter valid numeric values in all fields.")

if __name__ == "__main__":
    main()
