import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Dataset stats
feature_stats = {
    "Pregnancies":        {"min": 0,    "avg": 3.83,  "max": 17},
    "Glucose":            {"min": 0,    "avg": 122.94, "max": 199},
    "BloodPressure":      {"min": 0,    "avg": 69.33, "max": 122},
    "SkinThickness":      {"min": 0,    "avg": 20.78, "max": 99},
    "Insulin":            {"min": 0,    "avg": 82.37, "max": 846},
    "BMI":                {"min": 0.0,  "avg": 32.07, "max": 67.1},
    "DiabetesPedigreeFunction": {"min": 0.08, "avg": 0.48, "max": 2.42},
    "Age":                {"min": 21,   "avg": 33.22, "max": 81}
}

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("Enter your health data to check your diabetes risk. Dataset ranges shown below each input.")

# Input sliders with stats displayed
def input_with_stats(label, min_val, avg_val, max_val, step=1.0):
    value = st.slider(
        f"{label} (avg: {avg_val})",
        min_value=min_val,
        max_value=max_val,
        value=round(avg_val),
        step=step
    )
    st.caption(f"Min: {min_val} | Avg: {avg_val} | Max: {max_val}")
    return value

# Column layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = input_with_stats("Pregnancies", 0, 3.83, 17, step=1)
    glucose = input_with_stats("Glucose", 0, 122.94, 199, step=1)
    insulin = input_with_stats("Insulin", 0, 82.37, 846, step=1)
    dpf = input_with_stats("Diabetes Pedigree Function", 0.08, 0.48, 2.42, step=0.01)

with col2:
    blood_pressure = input_with_stats("Blood Pressure", 0, 69.33, 122, step=1)
    skin_thickness = input_with_stats("Skin Thickness", 0, 20.78, 99, step=1)
    bmi = input_with_stats("BMI", 0.0, 32.07, 67.1, step=0.1)
    age = input_with_stats("Age", 21, 33.22, 81, step=1)

# Prediction
if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    result = model.predict(features)[0]
    st.success("✅ You may be **Diabetic**." if result == 1 else "🎉 You are **Not Diabetic**.")
