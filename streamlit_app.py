import streamlit as st
import pandas as pd
import joblib

model = joblib.load("nutrisense_model.pkl")

st.title("NutriSense")
st.write("Prediksi status gizi anak berdasarkan berat badan, tinggi badan, dan usia.")

with st.form(key="input_form"):
    age_month = st.number_input("Usia (bulan)", min_value=0, max_value=60, value=24)
    weight = st.number_input("Berat badan (kg)", min_value=0.0, max_value=30.0, value=10.0)
    height = st.number_input("Tinggi badan (cm)", min_value=0.0, max_value=150.0, value=80.0)
    submit_button = st.form_submit_button("Prediksi")

if submit_button:
    age = int(round(age_month / 12))
    
    X_new = pd.DataFrame([[age, weight, height]], columns=["age", "weight", "height"])
    pred_encoded = model.predict(X_new)[0]
    pred_proba = model.predict_proba(X_new).max()
    
    label_map_rev = {0: "Normal", 1: "Berisiko", 2: "Stunted"}
    pred_label = label_map_rev[pred_encoded]
    
    st.subheader("Hasil Prediksi:")
    st.write(f"**Status gizi:** {pred_label}")
    st.write(f"**Confidence:** {pred_proba*100:.2f}%")
