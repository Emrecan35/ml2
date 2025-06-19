import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from fpdf import FPDF
import base64
import io

from sklearn.preprocessing import StandardScaler

# Sayfa ayarı
st.set_page_config(
    page_title="Su İçilebilir mi Acaba?",
    page_icon="💧",
    layout="wide"
)

# Model ve scaler dosya yolları
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
    solids = st.sidebar.slider("Solids (ppm)", 0.0, 5000.0, 2000.0, step=10.0)
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

def create_pdf_report(input_data, prediction, probability):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Su İçilebilirlik Tahmini Raporu", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Girilen Su Kalitesi Özellikleri:", ln=True)
    for key, value in input_data.items():
        pdf.cell(0, 8, f"{key}: {value}", ln=True)

    pdf.ln(5)
    result_text = "İÇİLEBİLİR SU 💧" if prediction == 1 else "İÇİLEMEZ SU ❌"
    pdf.cell(0, 10, f"Tahmin Sonucu: {result_text}", ln=True)
    pdf.cell(0, 10, f"Güven Skoru: %{probability*100:.2f}", ln=True)

    # PDF verisini bytes olarak al
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

def plot_shap_summary(model, X_scaled):
    st.subheader("🔍 Model Özellik Önem Skoru (SHAP)")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        # shap_values catboost'ta liste dönebilir, direkt ilk elemana bakıyoruz:
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, features=X_scaled, feature_names=model.feature_names_in_, plot_type="bar", show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP grafiği oluşturulurken hata: {e}")

def main():
    st.markdown("<h1 style='text-align: center; color: #0077b6;'>💧 Su İçilebilir mi Acaba?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:18px;'>CatBoost modeli ile suyun içilebilir olup olmadığını tahmin ediyoruz.</p>", unsafe_allow_html=True)

    with st.expander("📘 Veri Seti Hakkında Bilgi"):
        st.markdown("""
        Bu uygulama, su içilebilirliğini tahmin etmek amacıyla oluşturulmuş bir makine öğrenimi modeline dayanmaktadır.

        **Veri Seti Özellikleri:**
        - Toplam 10 temel özellik (pH, sertlik, kloramin vs.)
        - İçilebilirlik: 0 = İçilemez, 1 = İçilebilir

        **Amaç:**
        - Kullanıcının girdiği değerlere göre suyun içilebilir olup olmadığını tahmin etmek

        **Kullanılan Model:**
        - CatBoostClassifier (dengelenmiş sınıflar ve yeni öznitelikler ile)

        **Ekstra Özellikler (Feature Engineering):**
        - Kimyasal yoğunluk skorları
        - Normalize toksisite skorları
        - Zıt etkili birleşimler
        """)

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
            with st.expander("🧾 İçilebilir Su Kriterleri"):
                st.markdown("""
                - **pH**: 6.5 - 8.5 arası  
                - **Sertlik**: < 300 mg/L  
                - **Turbidity (Bulanıklık)**: < 5 NTU 0-1 arası en uygun parametre 
                - **Trihalomethanes**: < 80 mg/L  
                - **Kloramin**: 1 - 3 mg/L arası  
                """)
        else:
            st.error(f"Tahmin Sonucu: {result}")
            st.warning(f"⚠️ Güven Skoru: {probability:.2%} — Su içmeye uygun olmayabilir!")
            with st.expander("🚱 Olası Sebepler"):
                st.markdown("""
                - pH seviyesi çok düşük veya çok yüksek olabilir.  
                - Kimyasal kalıntılar (kloramin, trihalometan) yüksek olabilir.  
                - İletkenlik veya bulanıklık sınırların dışında olabilir.  
                - Toplam toksisite riskli seviyede olabilir.  
                """)

        show_prediction_gauge(probability)
        plot_shap_summary(model, input_scaled)

        # PDF raporu oluşturup indirilebilir yap
        pdf_buffer = create_pdf_report(input_df.iloc[0].to_dict(), prediction[0], probability)

        st.download_button(
            label="📄 PDF Raporunu İndir",
            data=pdf_buffer,
            file_name="su_icerik_tahmin_raporu.pdf",
            mime="application/pdf"
        )

    st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 14px;'>
    Bu uygulama, su kalitesine göre içilebilirlik tahmini için geliştirilmiştir. | Geliştiren: Emrecan Karaslan © 2025
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
