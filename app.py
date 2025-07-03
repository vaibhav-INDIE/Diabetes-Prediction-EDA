import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Set Streamlit page configuration
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("Enter your health data to check your risk of diabetes.")

# Layout for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=21, max_value=81, value=33)
    bmi = st.slider("BMI (Body Mass Index)", min_value=0.0, max_value=67.1, value=32.1)
    insulin = st.slider("Insulin", min_value=0, max_value=846, value=82)

with col2:
    glucose = st.slider("Glucose Level", min_value=0, max_value=199, value=123)
    blood_pressure = st.slider("Blood Pressure", min_value=0, max_value=122, value=69)
    pregnancies = st.slider("Pregnancies", min_value=0, max_value=17, value=4)

# Hidden average values
skin_thickness = 20.78
dpf = 0.48  # Diabetes Pedigree Function

# Prediction
if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    result = model.predict(features)[0]
    st.success("✅ You are **Diabetic**." if result == 1 else "🎉 You are **Not Diabetic**.")
