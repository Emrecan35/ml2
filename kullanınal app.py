import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Dosya yolları
MODEL_PATH = "catboost_model.pkl"
SCALER_PATH = "scaler.pkl"

st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon="💧",
    layout="wide"
)

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def get_user_input():
    st.sidebar.header("Input Water Quality Features")

    ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0, step=0.1)
    hardness = st.sidebar.slider("Hardness", 0.0, 500.0, 150.0, step=1.0)
    solids = st.sidebar.slider("Solids (ppm)", 0.0, 50000.0, 20000.0, step=10.0)
    chloramines = st.sidebar.slider("Chloramines", 0.0, 20.0, 7.0, step=0.1)
    sulfate = st.sidebar.slider("Sulfate", 0.0, 500.0, 250.0, step=1.0)
    conductivity = st.sidebar.slider("Conductivity", 0.0, 1500.0, 300.0, step=1.0)
    organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 20.0, 5.0, step=0.1)
    trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 150.0, 40.0, step=0.1)
    turbidity = st.sidebar.slider("Turbidity", 0.0, 15.0, 3.0, step=0.1)

    data = {
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }

    input_df = pd.DataFrame([data])
    return input_df

def main():
    st.title("💧 Water Potability Prediction App")
    st.write("CatBoost model kullanılarak su içilebilirliği tahmini yapılmaktadır.")

    model, scaler = load_model_and_scaler()

    input_df = get_user_input()

    st.subheader("Girdiğiniz Özellikler")
    st.write(input_df)

    # Ölçekleme ve tahmin işlemi
    input_scaled = scaler.transform(input_df)

    if st.button("Tahmin Et"):
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0][1]
        st.write(f"💡 İçilebilir olasılığı: {prob:.3f}")
        result = "İÇİLEBİLİR SU 💧" if prediction[0] == 1 else "İÇİLEMEZ SU ❌"

        if prediction[0] == 1:
            st.success(f"Tahmin Sonucu: {result}")
        else:
            st.error(f"Tahmin Sonucu: {result}")

if __name__ == "__main__":
    main()