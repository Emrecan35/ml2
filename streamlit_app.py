import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# Sayfa ayarı
st.set_page_config(
    page_title="Su İçilebilir mi Acaba?",
    page_icon="💧",
    layout="wide"
)

# Model ve scaler yükleme
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

def main():
    st.markdown("<h1 style='text-align: center; color: #0077b6;'>💧 Su İçilebilir mi Acaba?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:18px;'>CatBoost modeli ile suyun içilebilir olup olmadığını tahmin ediyoruz.</p>", unsafe_allow_html=True)

    with st.expander("📘 Veri Seti Hakkında Bilgi"):
        st.markdown("""
        Bu uygulama, su içilebilirliğini tahmin etmek amacıyla oluşturulmuş bir makine öğrenimi modeline dayanmaktadır.

        **Veri Seti Özellikleri:**
        - Toplam **9** temel özellik (pH, sertlik, kloramin vs.)
        - **İçilebilirlik:** 0 = İçilemez, 1 = İçilebilir

        **Amaç:**
        - Kullanıcının girdiği değerlere göre suyun **içilebilir** olup olmadığını tahmin etmek

        **Kullanılan Model:**
        - **CatBoostClassifier** (dengelenmiş sınıflar ve yeni öznitelikler ile)
        **Özellikler Hakkında Kısa Bilgiler**
        
        -**pH değeri**:
            PH, suyun asit-baz dengesinin değerlendirilmesinde önemli bir parametredir. Aynı zamanda su durumunun asidik veya alkali durumunun göstergesidir. DSÖ, izin verilen maksimum pH sınırını 6,5 ila 8,5 arasında önermiştir. Mevcut araştırma aralıkları DSÖ standartları aralığında olan 6,52–6,83 arasındaydı.
            
        -**Sertlik**:
            Sertliğe esas olarak kalsiyum ve magnezyum tuzları neden olur. Bu tuzlar, suyun içinden geçtiği jeolojik birikintilerden çözülür. Suyun sertlik üreten malzeme ile temas ettiği süre, ham suda ne kadar sertlik olduğunu belirlemeye yardımcı olur. Sertlik başlangıçta suyun Kalsiyum ve Magnezyumun neden olduğu sabunu çökeltme kapasitesi olarak tanımlandı.
       
        -**Katılar(Toplam çözünmüş katılar - TDS)**:
            Su, çok çeşitli inorganik ve potasyum, kalsiyum, sodyum, bikarbonatlar, klorürler, magnezyum, sülfatlar vb. Gibi bazı organik mineralleri veya tuzları çözme yeteneğine sahiptir. Bu mineraller su görünümünde istenmeyen tat ve seyreltilmiş renk üretti. Bu, su kullanımı için önemli parametredir. TDS değeri yüksek olan su, suyun yüksek oranda mineralize olduğunu gösterir. TDS için istenen sınır 500 mg / l'dir ve maksimum sınır, içme amacıyla öngörülen 1000 mg / l'dir.
        
        -**Kloraminler**:
            Klor ve kloramin, kamu su sistemlerinde kullanılan başlıca dezenfektanlardır. Kloraminler en yaygın olarak içme suyunu arıtmak için klora amonyak eklendiğinde oluşur. Litre başına 4 miligrama (mg / L veya milyonda 4 kısım (ppm)) kadar klor seviyeleri içme suyunda güvenli kabul edilir.
       
        -**Sülfat**:
            Sülfatlar, minerallerde, toprakta ve kayalarda bulunan doğal olarak oluşan maddelerdir. Ortam havasında, yeraltı suyunda, bitkilerde ve yiyeceklerde bulunurlar. Sülfatın başlıca ticari kullanımı kimya endüstrisindedir. Deniz suyundaki sülfat konsantrasyonu litre başına yaklaşık 2.700 miligramdır (mg / L). Bazı coğrafi bölgelerde çok daha yüksek konsantrasyonlar (1000 mg / L) bulunmasına rağmen, çoğu tatlı su kaynağında 3 ila 30 mg / L arasında değişir.
        
        -**İletkenlik**:
            Saf su, elektrik akımının iyi bir iletkeni değil, iyi bir yalıtkandır. İyon konsantrasyonundaki artış, suyun elektriksel iletkenliğini arttırır. Genel olarak, sudaki çözünmüş katıların miktarı elektriksel iletkenliği belirler. Elektriksel iletkenlik (EC) aslında akımı iletmesini sağlayan bir çözeltinin iyonik sürecini ölçer. DSÖ standartlarına göre EC değeri 400 µS/cm'yi geçmemelidir.
        -**Organik_karbon**:
        
            Kaynak sulardaki Toplam Organik Karbon (TOC), sentetik kaynakların yanı sıra çürüyen doğal organik maddeden (NOM) gelir. TOC, saf sudaki organik bileşiklerdeki toplam karbon miktarının bir ölçüsüdür. Usepa'ya göre arıtılmış / içme suyunda TOC olarak <2 mg / L ve arıtmada kullanılan kaynak suyunda <4 mg / L'dir.
        
        -**Trihalometanlar**:
            Thm'ler, klor ile arıtılmış suda bulunabilen kimyasallardır. İçme suyundaki THMs konsantrasyonu, sudaki organik madde seviyesine, suyu arıtmak için gereken klor miktarına ve arıtılan suyun sıcaklığına göre değişir. İçme suyunda 80 ppm'ye kadar olan THM seviyeleri güvenli kabul edilir.
       
        -**Bulanıklık**:
            Suyun bulanıklığı, askıda halde bulunan katı madde miktarına bağlıdır. Suyun ışık yayan özelliklerinin bir ölçüsüdür ve test, kolloidal maddeye göre atık deşarjının kalitesini belirtmek için kullanılır. Wondo Genet Kampüsü için elde edilen ortalama bulanıklık değeri (0,98 NTU) DSÖ tarafından önerilen 5,00 NTU değerinden düşüktür.
        -Hedef Değişkenimiz: İçilebilirlik:
        
            Suyun insan tüketimi için güvenli olup olmadığını gösterir, burada 1 içilebilir ve 0 içilemez anlamına gelir.
        Ekstra Özellikler (Feature Engineering):
        - Kimyasal yoğunluk skorları
        - Normalize toksisite skorları
        - Zıt etkili birleşimler yapılmıştır
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
                - **Turbidity (Bulanıklık)**: < 5 NTU  
                - **Trihalomethanes**: < 0.08 mg/L  
                - **Kloramin**: 3 - 4 mg/L arası  
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

    st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 14px;'>
    Bu uygulama, su kalitesine göre içilebilirlik tahmini için geliştirilmiştir. | Geliştiren: Emrecan Karaslan © 2025
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
