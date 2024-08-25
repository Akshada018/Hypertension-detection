# app.py
import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
model = pickle.load(open('model/hypertension_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

st.title("Hypertension Detection App")

# User inputs
age = st.number_input('Age', min_value=1, max_value=100, value=25)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=22.0)
cholesterol = st.number_input('Cholesterol Level', min_value=100, max_value=300, value=150)
blood_sugar = st.number_input('Blood Sugar Level', min_value=50, max_value=200, value=100)
gender = st.selectbox('Gender', ('Male', 'Female'))

# Preprocess the inputs
gender_encoded = 0 if gender == 'Male' else 1
input_data = np.array([[age, bmi, cholesterol, blood_sugar, gender_encoded]])
input_data = scaler.transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.warning("High risk of Hypertension.")
    else:
        st.success("Low risk of Hypertension.")
