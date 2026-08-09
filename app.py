
import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate

# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Prediksi Diabetes - Naive Bayes",
    page_icon="🩺",
    layout="wide"
)

DATA_FILE = Path(__file__).parent / "data skripsi.xlsx"


# ============================================================
# FUNGSI
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_FILE)

    # Membersihkan spasi tersembunyi pada nama kolom
    df.columns = df.columns.str.strip()

    # Memisahkan tekanan darah
    tekanan = df["Tekanan Darah"].astype(str).str.split("/", expand=True)

    if tekanan.shape[1] != 2:
        raise ValueError(
            "Format kolom Tekanan Darah harus seperti 120/80."
        )

    df["Sistolik"] = pd.to_numeric(tekanan[0], errors="coerce")
    df["Diastolik"] = pd.to_numeric(tekanan[1], errors="coerce")

    df.drop(columns=["Tekanan Darah"], inplace=True)

    # Encode target
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

    # Model final untuk deployment dilatih menggunakan seluruh data
    model.fit(X, y)

    return model


@st.cache_data
def cross_validation_results():
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

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    metrics = {
        "Accuracy": results["test_accuracy"].mean(),
        "Precision": results["test_precision"].mean(),
        "Recall / Sensitivity": results["test_recall"].mean(),
        "F1-Score": results["test_f1"].mean(),
        "ROC-AUC": results["test_roc_auc"].mean(),
        "Accuracy SD": results["test_accuracy"].std()
    }

    return metrics


def parse_blood_pressure(systolic, diastolic):
    return float(systolic), float(diastolic)


# ============================================================
# LOAD
# ============================================================

try:
    data = load_data()
    model = train_model()
    metrics = cross_validation_results()
except Exception as e:
    st.error(f"Gagal memuat model/data: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Tentang Model")

    st.write(
        """
        **Naive Bayes** digunakan sebagai model klasifikasi
        berdasarkan hasil penelitian.

        Model menggunakan:
        - Jenis Kelamin
        - Usia
        - Tinggi Badan
        - Berat Badan
        - Tekanan Darah
        - Lingkar Perut
        """
    )

    st.divider()

    st.caption(
        "Model final dilatih menggunakan seluruh data penelitian "
        "setelah model Naive Bayes dipilih sebagai model terbaik."
    )


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Prediksi Klasifikasi Diabetes")
st.subheader("Model Naive Bayes")

st.write(
    """
    Masukkan karakteristik individu pada form di bawah untuk memperoleh
    hasil klasifikasi dari model Naive Bayes.
    """
)

st.warning(
    "Aplikasi ini merupakan implementasi model klasifikasi untuk "
    "keperluan penelitian/edukasi dan bukan alat diagnosis medis."
)


# ============================================================
# METRIK MODEL
# ============================================================

st.header("Performa Model")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
c2.metric("Precision", f"{metrics['Precision']:.2%}")
c3.metric("Recall", f"{metrics['Recall / Sensitivity']:.2%}")
c4.metric("F1-Score", f"{metrics['F1-Score']:.2%}")
c5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2%}")

st.caption(
    f"Evaluasi menggunakan Stratified 10-Fold Cross Validation. "
    f"Rata-rata accuracy = {metrics['Accuracy']:.2%} "
    f"± {metrics['Accuracy SD']:.2%}."
)


# ============================================================
# FORM INPUT
# ============================================================

st.header("Input Data Individu")

col1, col2 = st.columns(2)

with col1:
    jenis_kelamin = st.selectbox(
        "Jenis Kelamin",
        options=["LK", "PR"],
        format_func=lambda x: "Laki-laki (LK)" if x == "LK" else "Perempuan (PR)"
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


# ============================================================
# PREDIKSI
# ============================================================

if st.button(
    "🔍 Lakukan Prediksi",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Jenis Kelamin": [jenis_kelamin],
        "Usia": [usia],
        "Tinggi Badan": [tinggi_badan],
        "Berat Badan": [berat_badan],
        "Linkar Perut": [lingkar_perut],
        "Sistolik": [sistolik],
        "Diastolik": [diastolik]
    })

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    prob_tidak = probabilities[0]
    prob_diabetes = probabilities[1]

    st.divider()
    st.header("Hasil Prediksi")

    if prediction == 1:
        st.error("### Hasil: Diabetes")
        st.write(
            "Model mengklasifikasikan data input ke dalam kelas **Diabetes**."
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

    with st.expander("Lihat data input"):
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# INFORMASI DATA
# ============================================================

with st.expander("Informasi Dataset"):
    st.write(f"Jumlah observasi: **{len(data)}**")
    st.write(f"Jumlah variabel prediktor: **{len(data.columns) - 1}**")

    distribusi = data["Label"].map({
        0: "Tidak Diabetes",
        1: "Diabetes"
    }).value_counts()

    st.dataframe(
        distribusi.rename("Jumlah").to_frame(),
        use_container_width=True
    )
