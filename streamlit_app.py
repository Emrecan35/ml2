import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Dosya yolları
MODEL_PATH = "aliemrecatboost_model.pkl"
SCALER_PATH = "scaler.pkl"
DEFAULTS_PATH = "impute_defaults.pkl"

st.set_page_config(
    page_title="Su İçilebilir mi Acaba?",
    page_icon="💧",
    layout="wide"
)

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

def plot_user_inputs(input_df):
    st.subheader("📊 Girdi Değerleri Grafiği")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=input_df.columns, y=input_df.iloc[0], palette="Blues_d", ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel("Değer")
    plt.title("Kullanıcı Girdileri")
    st.pyplot(fig)

def show_line_chart(input_df):
    st.subheader("📈 Girdi Değerlerinin Line Chart Gösterimi")
    df_long = input_df.T.reset_index()
    df_long.columns = ['Özellik', 'Değer']
    df_long['İndeks'] = range(1, len(df_long) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df_long, x="İndeks", y="Değer", marker='o', palette="tab10", ax=ax)
    ax.set_xticks(df_long["İndeks"])
    ax.set_xticklabels(df_long["Özellik"], rotation=45, ha="right")
    plt.title("Line Chart ile Girdi Özellikleri")
    st.pyplot(fig)

def show_prediction_gauge(probability):
    fig, ax = plt.subplots(figsize=(4, 4))
    colors = ['red', 'green']
    labels = ['İçilemez', 'İçilebilir']
    ax.pie([1 - probability, probability], labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.subheader("🧪 Tahmin Güveni (Olasılık)")
    st.pyplot(fig)

def plot_feature_with_threshold(value, feature_name, safe_max):
    fig, ax = plt.subplots()
    ax.barh([feature_name], [value], color="green" if value <= safe_max else "red")
    ax.axvline(safe_max, color="gray", linestyle="--", label=f"Güvenli Sınır: {safe_max}")
    plt.xlabel("Değer")
    plt.title(f"{feature_name} ve Güvenli Sınır")
    plt.legend()
    st.pyplot(fig)

def main():
    st.markdown("<h1 style='text-align: center; color: #0077b6;'>💧 Su İçilebilir mi Acaba?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:18px;'>CatBoost modeli ile suyun içilebilir olup olmadığını tahmin ediyoruz.</p>", unsafe_allow_html=True)

    model, scaler = load_model_and_scaler()
    input_df = get_user_input()

    st.subheader("🔍 Girdiğiniz Özellikler")
    st.write(input_df)

    plot_user_inputs(input_df)
    show_line_chart(input_df)

    # Belirli özellikleri eşiklerle göster
    plot_feature_with_threshold(input_df["Turbidity"].values[0], "Turbidity", 5.0)
    plot_feature_with_threshold(input_df["Trihalomethanes"].values[0], "Trihalomethanes", 80.0)
    plot_feature_with_threshold(input_df["Sulfate"].values[0], "Sulfate", 250.0)
    plot_feature_with_threshold(input_df["Conductivity"].values[0], "Conductivity", 1000.0)

    # Özellik mühendisliği ve ölçekleme
    input_with_features = add_engineered_features(input_df.copy())
    input_scaled = scaler.transform(input_with_features)

    if st.button("🚰 Tahmin Et"):
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]
        result = "İÇİLEBİLİR SU 💧" if prediction[0] == 1 else "İÇİLEMEZ SU ❌"

        if prediction[0] == 1:
            st.success(f"Tahmin Sonucu: {result}\n\nGüven: {probability:.2%}")
        else:
            st.error(f"Tahmin Sonucu: {result}\n\nGüven: {probability:.2%}")

        show_prediction_gauge(probability)

if __name__ == "__main__":
    main()
