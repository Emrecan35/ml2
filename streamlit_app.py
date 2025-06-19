import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# Tema seçimi
st.set_page_config(
    page_title="Su İçilebilir mi Acaba?",
    page_icon="💧",
    layout="wide"
)

# Tema switcher
theme = st.sidebar.radio("🌗 Tema Seçimi", ["Açık Tema", "Koyu Tema"])
if theme == "Koyu Tema":
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: white; }
        .st-bf { background-color: #0e1117; }
        </style>
        """, unsafe_allow_html=True
    )

# Dosya yolları
MODEL_PATH = "aliemrecatboost_model.pkl"
SCALER_PATH = "scaler.pkl"
DEFAULTS_PATH = "impute_defaults.pkl"

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

defaults = joblib.load(DEFAULTS_PATH)

def get_user_input():
    st.sidebar.header("💧 Su Kalitesi Özelliklerini Girin")

    ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0, step=0.1)
    hardness = st.sidebar.slider("Hardness (Sertlik)", 0.0, 500.0, 150.0, step=1.0)
    solids = st.sidebar.slider("Solids (ppm)", 0.0, 50000.0, 20000.0, step=10.0)
    chloramines = st.sidebar.slider("Chloramines (ppm)", 0.0, 20.0, 7.0, step=0.1)
    sulfate = st.sidebar.slider("Sulfate (mg/L)", 0.0, 500.0, 250.0, step=1.0)
    conductivity = st.sidebar.slider("Conductivity", 0.0, 1500.0, 300.0, step=1.0)
    organic_carbon = st.sidebar.slider("Organic Carbon (ppm)", 0.0, 20.0, 5.0, step=0.1)
    trihalomethanes = st.sidebar.slider("Trihalomethanes (mg/L)", 0.0, 150.0, 40.0, step=0.1)
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

    df = pd.DataFrame([data])
    df.fillna(defaults, inplace=True)
    return df

def add_engineered_features(df):
    df["mineral_density"] = (df["Solids"] + df["Hardness"]) / (df["Conductivity"] + 0.01)
    df["ph_conductivity_interaction"] = df["ph"] * df["Conductivity"]
    df["ph_div_turbidity"] = df["ph"] / (df["Turbidity"] + 0.01)
    df["chloramine_ratio_total_chem"] = df["Chloramines"] / (df["Trihalomethanes"] + df["Organic_carbon"] + 0.01)
    df["tri_to_organic_ratio"] = df["Trihalomethanes"] / (df["Organic_carbon"] + 0.01)
    df["sulfate_to_total_dissolved"] = df["Sulfate"] / (df["Solids"] + df["Conductivity"] + 0.01)
    df["hardness_ratio"] = df["Hardness"] / (df["ph"] + df["Turbidity"] + 0.01)
    df["hard_ph_turb_mix"] = df["Hardness"] * df["ph"] * df["Turbidity"]
    df["chem_density_score"] = (
        df["Chloramines"]**0.5 +
        df["Trihalomethanes"]**0.5 +
        df["Organic_carbon"]**0.5
    )
    df["sulfate_minus_conductivity"] = df["Sulfate"] - df["Conductivity"]
    df["solids_minus_organic"] = df["Solids"] - df["Organic_carbon"]
    df["ph_minus_trihalo"] = df["ph"] - df["Trihalomethanes"]
    df["normalized_conductivity"] = df["Conductivity"] / df["Conductivity"].max()
    df["normalized_toxicity"] = (
        df["Chloramines"]/df["Chloramines"].max() +
        df["Trihalomethanes"]/df["Trihalomethanes"].max()
    )
    df["ph_x_inverse_turbidity"] = df["ph"] * (1 / (df["Turbidity"] + 0.01))
    df["sulfate_div_logsolids"] = df["Sulfate"] / (np.log1p(df["Solids"]))
    return df

def show_prediction_gauge(probability):
    with st.expander("🧪 Tahmin Olasılığı Grafiği"):
        fig, ax = plt.subplots(figsize=(4, 4))
        colors = ['red', 'green']
        labels = ['İçilemez', 'İçilebilir']
        ax.pie([1 - probability, probability], labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

def main():
    st.markdown("<h1 style='text-align: center; color: #0077b6;'>💧 Su İçilebilir mi Acaba?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:18px;'>CatBoost modeli ile suyun içilebilir olup olmadığını tahmin ediyoruz.</p>", unsafe_allow_html=True)

    model, scaler = load_model_and_scaler()
    input_df = get_user_input()

    st.subheader("🔍 Girdiğiniz Özellikler")
    st.write(input_df)

    input_with_features = add_engineered_features(input_df.copy())
    input_scaled = scaler.transform(input_with_features)

    if st.button("🚰 Tahmin Et"):
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]
        result = "İÇİLEBİLİR SU 💧" if prediction[0] == 1 else "İÇİLEMEZ SU ❌"

        if prediction[0] == 1:
            st.success(f"Tahmin Sonucu: {result}")
            st.info(f"💡 Güven Skoru: {probability:.2%} — Su büyük ihtimalle içilebilir.")
        else:
            st.error(f"Tahmin Sonucu: {result}")
            st.warning(f"⚠️ Güven Skoru: {probability:.2%} — Su içmeye uygun olmayabilir!")

        show_prediction_gauge(probability)

if __name__ == "__main__":
    main()
