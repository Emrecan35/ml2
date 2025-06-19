
import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(page_title="Water Potability ML App", layout="wide")

st.title("💧 Water Potability - ML Model Dashboard")

# Veri Yükleme
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    return df

df = load_data()

st.subheader("🔍 İlk 5 Satır")
st.dataframe(df.head())

# Eksik verileri doldurma
df['ph'] = df['ph'].fillna(df.groupby('Potability')['ph'].transform('mean'))
df['Sulfate'] = df['Sulfate'].fillna(df.groupby('Potability')['Sulfate'].transform('mean'))
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df.groupby('Potability')['Trihalomethanes'].transform('mean'))

# Aykırı değer baskılama
def cap_outliers(df):
    for col in df.select_dtypes(include=np.number).columns[:-1]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower,
                           np.where(df[col] > upper, upper, df[col]))
    return df

df = cap_outliers(df)

# Model Seçimi
st.sidebar.header("⚙️ Model ve Eşik Ayarları")
model_name = st.sidebar.selectbox("Model Seç", [
    "Lojistik Regresyon", "Karar Ağacı", "KNN", "Random Forest", "XGBoost", "Voting (Hepsi)"
])

threshold = st.sidebar.slider("Sınıflandırma Eşiği", 0.1, 0.9, 0.5, 0.01)

# Train-Test bölme
X = df.drop("Potability", axis=1)
y = df["Potability"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Seçimi
if model_name == "Lojistik Regresyon":
    model = LogisticRegression()
elif model_name == "Karar Ağacı":
    model = DecisionTreeClassifier()
elif model_name == "KNN":
    model = KNeighborsClassifier()
elif model_name == "Random Forest":
    model = RandomForestClassifier()
elif model_name == "XGBoost":
    model = XGBClassifier()
else:
    model = VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('knn', KNeighborsClassifier())
    ], voting='soft')

model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > threshold).astype(int)

# Confusion Matrix ve Metrikler
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

st.subheader("📉 Sonuçlar")
col1, col2, col3 = st.columns(3)
col1.metric("Doğruluk (Accuracy)", f"{acc:.2f}")
col2.metric("Kesinlik (Precision)", f"{prec:.2f}")
col3.metric("Duyarlılık (Recall)", f"{rec:.2f}")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title(f"Confusion Matrix - {model_name}")
st.pyplot(fig)

