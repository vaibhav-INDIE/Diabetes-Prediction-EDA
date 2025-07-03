import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Streamlit page setup
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("Enter basic health information to check your diabetes risk.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 10, 100, 30)
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 22.0)
    insulin = st.slider("Insulin", 0, 300, 85)

with col2:
    glucose = st.slider("Glucose Level", 50, 200, 100)
    blood_pressure = st.slider("Blood Pressure", 50, 130, 80)
    pregnancies = st.slider("Pregnancies", 0, 15, 1)

# Hidden average values
skin_thickness = 23
dpf = 0.47  # Diabetes Pedigree Function

if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    result = model.predict(features)[0]
    st.success("✅ You are **Diabetic**." if result == 1 else "🎉 You are **Not Diabetic**.")
