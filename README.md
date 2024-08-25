# Hypertension Detection App

## Overview

This project is a web application for detecting the risk of hypertension based on user inputs. The application uses a machine learning model to predict whether a user is at high or low risk of hypertension. The model is trained using data on age, BMI, cholesterol levels, blood sugar levels, and gender.

## Features

- **User Input:** Users can enter their age, BMI, cholesterol level, blood sugar level, and gender.
- **Prediction:** The application predicts the risk of hypertension based on the user inputs.
- **Results Display:** Users receive feedback on their risk level (high or low) based on the prediction.

## Prerequisites

Ensure you have the following installed:

- Python 3.12 or later
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Akshada018/Hypertension-detection.git
   cd Hypertension-detection

Set Up Virtual Environment:
- python -m venv .venv

Activate Virtual Environment:
 Windows:
  - .venv\Scripts\activate
  - source .venv/bin/activate
 
Install Dependencies:
  - pip install -r requirements.txt


If you encounter permission issues, try:
 - pip install --user -r requirements.txt

* Ensure that the trained model (hypertension_model.pkl) and scaler (scaler.pkl) are placed in the model directory.

Run the Application:
 - streamlit run app.py


# Access the Web App:
Open a browser and go to http://localhost:8501 to interact with the app.