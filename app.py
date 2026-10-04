import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Prediksi Diabetes - Naive Bayes",
    page_icon="🩺",
    layout="centered"
)

DATA_FILE = Path(__file__).parent / "data skripsi.xlsx"


# ============================================================
# FUNGSI
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_FILE)
    df.columns = df.columns.str.strip()

    # Memisahkan tekanan darah menjadi sistolik dan diastolik
    tekanan = df["Tekanan Darah"].astype(str).str.split("/", expand=True)

    if tekanan.shape[1] != 2:
        raise ValueError(
            "Format kolom Tekanan Darah harus seperti 120/80."
        )

    df["Sistolik"] = pd.to_numeric(tekanan[0], errors="coerce")
    df["Diastolik"] = pd.to_numeric(tekanan[1], errors="coerce")
    df.drop(columns=["Tekanan Darah"], inplace=True)

    # Mengubah label target menjadi angka
    df["Label"] = df["Label"].map({
        "Tidak Diabetes": 0,
        "Diabetes": 1
    })

    if df["Label"].isna().any():
        raise ValueError(
            "Terdapat label yang tidak dikenali. "
            "Gunakan label 'Diabetes' dan 'Tidak Diabetes'."
        )

    return df


@st.cache_resource
def train_model():
    data = load_data()
    X = data.drop(columns=["Label"])
    y = data["Label"]

    categorical_features = ["Jenis Kelamin"]
    numeric_features = [
        "Usia",
        "Tinggi Badan",
        "Berat Badan",
        "Linkar Perut",
        "Sistolik",
        "Diastolik"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "num",
                StandardScaler(),
                numeric_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", GaussianNB())
        ]
    )

    # Model Naive Bayes dilatih menggunakan seluruh data penelitian
    model.fit(X, y)
    return model


# ============================================================
# HALAMAN UTAMA
# ============================================================

st.title("🩺 Prediksi Klasifikasi Diabetes")
st.subheader("Model Naive Bayes")

st.write(
    "Masukkan karakteristik individu, lalu tekan tombol "
    "**Lakukan Prediksi** untuk melihat hasil klasifikasi."
)

st.warning(
    "Aplikasi ini merupakan implementasi model klasifikasi untuk "
    "keperluan penelitian/edukasi dan bukan alat diagnosis medis."
)

try:
    data = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()


# ============================================================
# FORM INPUT DAN TOMBOL PREDIKSI
# ============================================================

st.header("Input Data Individu")

# st.form mencegah aplikasi memproses input setiap kali nilainya berubah.
# Prediksi hanya diproses setelah tombol submit ditekan.
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        jenis_kelamin = st.selectbox(
            "Jenis Kelamin",
            options=["LK", "PR"],
            format_func=lambda x: (
                "Laki-laki (LK)" if x == "LK" else "Perempuan (PR)"
            )
        )

        usia = st.number_input(
            "Usia (tahun)",
            min_value=1,
            max_value=120,
            value=int(data["Usia"].median()),
            step=1
        )

        tinggi_badan = st.number_input(
            "Tinggi Badan (cm)",
            min_value=50.0,
            max_value=250.0,
            value=float(data["Tinggi Badan"].median()),
            step=1.0
        )

    with col2:
        berat_badan = st.number_input(
            "Berat Badan (kg)",
            min_value=20.0,
            max_value=250.0,
            value=float(data["Berat Badan"].median()),
            step=0.5
        )

        lingkar_perut = st.number_input(
            "Lingkar Perut (cm)",
            min_value=30.0,
            max_value=200.0,
            value=float(data["Linkar Perut"].median()),
            step=1.0
        )

        st.write("**Tekanan Darah (mmHg)**")
        bp1, bp2 = st.columns(2)

        with bp1:
            sistolik = st.number_input(
                "Sistolik",
                min_value=50.0,
                max_value=250.0,
                value=120.0,
                step=1.0
            )

        with bp2:
            diastolik = st.number_input(
                "Diastolik",
                min_value=30.0,
                max_value=180.0,
                value=80.0,
                step=1.0
            )

    submitted = st.form_submit_button(
        "🔍 Lakukan Prediksi",
        type="primary",
        use_container_width=True
    )


# ============================================================
# HASIL PREDIKSI
# Hanya dijalankan setelah tombol ditekan
# ============================================================

if submitted:
    input_data = pd.DataFrame({
        "Jenis Kelamin": [jenis_kelamin],
        "Usia": [usia],
        "Tinggi Badan": [tinggi_badan],
        "Berat Badan": [berat_badan],
        "Linkar Perut": [lingkar_perut],
        "Sistolik": [sistolik],
        "Diastolik": [diastolik]
    })

    try:
        with st.spinner("Sedang memproses prediksi..."):
            model = train_model()
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]

        prob_tidak = probabilities[0]
        prob_diabetes = probabilities[1]

        st.divider()
        st.header("Hasil Prediksi")

        if prediction == 1:
            st.error("### Hasil: Diabetes")
            st.write(
                "Model mengklasifikasikan data input ke dalam kelas "
                "**Diabetes**."
            )
        else:
            st.success("### Hasil: Tidak Diabetes")
            st.write(
                "Model mengklasifikasikan data input ke dalam kelas "
                "**Tidak Diabetes**."
            )

        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Probabilitas Tidak Diabetes",
                f"{prob_tidak:.2%}"
            )

        with r2:
            st.metric(
                "Probabilitas Diabetes",
                f"{prob_diabetes:.2%}"
            )

        st.progress(
            float(prob_diabetes),
            text=f"Probabilitas Diabetes: {prob_diabetes:.2%}"
        )

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
