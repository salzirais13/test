# -*- coding: utf-8 -*-
"""
Aplikasi Streamlit: Prediksi Diabetes
Perbandingan Naive Bayes vs Decision Tree

Alur data, preprocessing, dan konfigurasi model disamakan dengan
notebook Naive_vs_Decision_Tree.ipynb.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Prediksi Diabetes - Naive Bayes vs Decision Tree",
    page_icon="🩺",
    layout="wide"
)

DATA_FILE = Path(__file__).parent / "data skripsi.xlsx"

RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["Jenis Kelamin"]
NUMERIC_FEATURES = [
    "Usia",
    "Tinggi Badan",
    "Berat Badan",
    "Linkar Perut",
    "Sistolik",
    "Diastolik"
]

CLASS_NAMES = ["Tidak Diabetes", "Diabetes"]

# True  -> model untuk prediksi dilatih memakai seluruh data (200 observasi)
# False -> model untuk prediksi dilatih memakai data training 80% saja,
#          persis seperti model di notebook
TRAIN_ON_ALL_DATA = True


# ============================================================
# FUNGSI DATA & MODEL
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_FILE)

    # Membersihkan spasi tersembunyi pada nama kolom
    df.columns = df.columns.str.strip()

    # Acak baris (sama dengan notebook)
    df = df.sample(frac=1, random_state=RANDOM_STATE)

    # Memisahkan tekanan darah menjadi Sistolik dan Diastolik
    tekanan = df["Tekanan Darah"].astype(str).str.split("/", expand=True)

    if tekanan.shape[1] != 2:
        raise ValueError(
            "Format kolom Tekanan Darah harus seperti 120/80."
        )

    df["Sistolik"] = pd.to_numeric(tekanan[0], errors="coerce")
    df["Diastolik"] = pd.to_numeric(tekanan[1], errors="coerce")

    df = df.drop(columns=["Tekanan Darah"])

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


def encode_for_tree(X):
    """
    Encoding Decision Tree: One-hot Jenis Kelamin dengan drop_first=True
    (LK sebagai baseline -> kolom 'Jenis Kelamin_PR'), sama dengan
    pd.get_dummies(...) pada notebook. Dibuat manual supaya input satu baris
    dari form selalu menghasilkan kolom yang identik dengan data training.
    """
    X_enc = X.drop(columns=["Jenis Kelamin"]).copy()
    X_enc["Jenis Kelamin_PR"] = (X["Jenis Kelamin"] == "PR").astype(int)
    return X_enc


def build_nb_model():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES
            )
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", GaussianNB())
        ]
    )


def build_tree_model():
    return DecisionTreeClassifier(
        criterion="entropy",
        random_state=RANDOM_STATE
    )


@st.cache_resource
def train_models():
    data = load_data()

    X = data.drop(columns=["Label"])
    y = data["Label"]

    if not TRAIN_ON_ALL_DATA:
        X, _, y, _ = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )

    nb_model = build_nb_model()
    nb_model.fit(X, y)

    tree_model = build_tree_model()
    tree_model.fit(encode_for_tree(X), y)

    return nb_model, tree_model


def hitung_metrik(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall / Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }


@st.cache_data
def evaluasi_model():
    """
    Evaluasi mengikuti notebook:
    1) Hold-out: train/test split 80:20 (stratified, random_state=42)
    2) Stratified 10-Fold Cross Validation
    """
    data = load_data()

    X = data.drop(columns=["Label"])
    y = data["Label"]
    X_tree = encode_for_tree(X)

    # ---------------- Hold-out 80:20 ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train_tree, X_test_tree, _, _ = train_test_split(
        X_tree, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    nb = build_nb_model().fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_prob = nb.predict_proba(X_test)[:, 1]

    tree = build_tree_model().fit(X_train_tree, y_train)
    tree_pred = tree.predict(X_test_tree)
    tree_prob = tree.predict_proba(X_test_tree)[:, 1]

    holdout = pd.DataFrame([
        hitung_metrik(y_test, nb_pred, nb_prob),
        hitung_metrik(y_test, tree_pred, tree_prob)
    ], index=["Naive Bayes", "Decision Tree"])

    fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_prob)
    fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_prob)

    visual = {
        "cm_nb": confusion_matrix(y_test, nb_pred),
        "cm_tree": confusion_matrix(y_test, tree_pred),
        "roc_nb": (fpr_nb, tpr_nb, roc_auc_score(y_test, nb_prob)),
        "roc_tree": (fpr_tree, tpr_tree, roc_auc_score(y_test, tree_prob)),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }

    # ---------------- Stratified 10-Fold CV ----------------
    skf = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_nb = cross_validate(build_nb_model(), X, y, cv=skf, scoring=scoring)
    cv_tree = cross_validate(build_tree_model(), X_tree, y, cv=skf, scoring=scoring)

    def ringkas(cv):
        return {
            "Accuracy": cv["test_accuracy"].mean(),
            "Precision": cv["test_precision"].mean(),
            "Recall": cv["test_recall"].mean(),
            "F1-Score": cv["test_f1"].mean(),
            "ROC-AUC": cv["test_roc_auc"].mean(),
            "Accuracy SD": cv["test_accuracy"].std()
        }

    cv_summary = pd.DataFrame(
        [ringkas(cv_nb), ringkas(cv_tree)],
        index=["Naive Bayes", "Decision Tree"]
    )

    return holdout, cv_summary, visual


# ============================================================
# LOAD
# ============================================================

try:
    data = load_data()
    nb_model, tree_model = train_models()
    holdout, cv_summary, visual = evaluasi_model()
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
        Aplikasi ini membandingkan dua metode klasifikasi:

        - **Naive Bayes** (Gaussian NB, fitur numerik distandardisasi)
        - **Decision Tree** (kriteria *entropy*, tanpa pruning)

        Variabel prediktor:
        - Jenis Kelamin
        - Usia
        - Tinggi Badan
        - Berat Badan
        - Tekanan Darah (Sistolik & Diastolik)
        - Lingkar Perut
        """
    )

    st.divider()

    if TRAIN_ON_ALL_DATA:
        st.caption(
            "Model untuk prediksi dilatih menggunakan seluruh data "
            "penelitian. Angka performa dihitung dari hold-out 80:20 "
            "dan Stratified 10-Fold Cross Validation."
        )
    else:
        st.caption(
            "Model untuk prediksi dilatih menggunakan 80% data training "
            "(sama dengan notebook). Angka performa dihitung dari "
            "hold-out 80:20 dan Stratified 10-Fold Cross Validation."
        )


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Prediksi Klasifikasi Diabetes")
st.subheader("Naive Bayes vs Decision Tree")

st.write(
    """
    Masukkan karakteristik individu pada form di bawah untuk memperoleh
    hasil klasifikasi sekaligus probabilitasnya dari kedua model.
    """
)

st.warning(
    "Aplikasi ini merupakan implementasi model klasifikasi untuk "
    "keperluan penelitian/edukasi dan bukan alat diagnosis medis."
)


# ============================================================
# PERFORMA MODEL
# ============================================================

st.header("Performa Model")

tab_cv, tab_holdout = st.tabs([
    "Stratified 10-Fold CV",
    "Hold-out 80:20"
])

with tab_cv:
    tampil_cv = cv_summary.drop(columns=["Accuracy SD"]).T
    tampil_cv.index.name = "Metrik"

    st.dataframe(
        tampil_cv.style.format("{:.2%}").highlight_max(
            axis=1, color="#d4edda"
        ),
        use_container_width=True
    )

    st.caption(
        "Rata-rata 10 fold. Accuracy Naive Bayes = "
        f"{cv_summary.loc['Naive Bayes', 'Accuracy']:.2%} "
        f"± {cv_summary.loc['Naive Bayes', 'Accuracy SD']:.2%}; "
        "Decision Tree = "
        f"{cv_summary.loc['Decision Tree', 'Accuracy']:.2%} "
        f"± {cv_summary.loc['Decision Tree', 'Accuracy SD']:.2%}. "
        "Sel hijau menandai nilai tertinggi pada tiap metrik."
    )

with tab_holdout:
    tampil_ho = holdout.T
    tampil_ho.index.name = "Metrik"

    st.dataframe(
        tampil_ho.style.format("{:.2%}").highlight_max(
            axis=1, color="#d4edda"
        ),
        use_container_width=True
    )

    st.caption(
        f"Data training = {visual['n_train']}, data testing = "
        f"{visual['n_test']} (split stratified, random_state = "
        f"{RANDOM_STATE}). Sel hijau menandai nilai tertinggi."
    )

with st.expander("Confusion Matrix, ROC Curve, dan Struktur Decision Tree"):
    v1, v2 = st.columns(2)

    with v1:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))

        for ax, cm, judul, cmap in [
            (axes[0], visual["cm_nb"], "Naive Bayes", "Blues"),
            (axes[1], visual["cm_tree"], "Decision Tree", "Greens")
        ]:
            ax.imshow(cm, cmap=cmap)
            ax.set_title(f"Confusion Matrix - {judul}", fontsize=10)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(CLASS_NAMES, fontsize=8)
            ax.set_yticklabels(CLASS_NAMES, fontsize=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j, i, cm[i, j],
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=12
                    )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with v2:
        fig, ax = plt.subplots(figsize=(5, 4))

        fpr, tpr, auc_val = visual["roc_nb"]
        ax.plot(fpr, tpr, label=f"Naive Bayes (AUC = {auc_val:.3f})")

        fpr, tpr, auc_val = visual["roc_tree"]
        ax.plot(fpr, tpr, label=f"Decision Tree (AUC = {auc_val:.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Hold-out)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.write("**Struktur Decision Tree (model untuk prediksi)**")

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        tree_model,
        feature_names=encode_for_tree(
            data.drop(columns=["Label"])
        ).columns.tolist(),
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        ax=ax
    )
    ax.set_title("Decision Tree Visualization")
    st.pyplot(fig)
    plt.close(fig)


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

def tampilkan_hasil(judul, model, X_input):
    """Menampilkan kelas prediksi dan probabilitas untuk satu model."""
    prediction = int(model.predict(X_input)[0])
    probabilities = model.predict_proba(X_input)[0]

    prob_tidak = probabilities[0]
    prob_diabetes = probabilities[1]

    st.subheader(judul)

    if prediction == 1:
        st.error(
            f"### Hasil: Diabetes\n"
            f"Probabilitas Diabetes: **{prob_diabetes:.2%}**"
        )
    else:
        st.success(
            f"### Hasil: Tidak Diabetes\n"
            f"Probabilitas Tidak Diabetes: **{prob_tidak:.2%}**"
        )

    m1, m2 = st.columns(2)
    m1.metric("Probabilitas Tidak Diabetes", f"{prob_tidak:.2%}")
    m2.metric("Probabilitas Diabetes", f"{prob_diabetes:.2%}")

    st.progress(
        float(prob_diabetes),
        text=f"Probabilitas Diabetes: {prob_diabetes:.2%}"
    )

    return prediction


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

    st.divider()
    st.header("Hasil Prediksi")

    hasil_col1, hasil_col2 = st.columns(2)

    with hasil_col1:
        pred_nb = tampilkan_hasil(
            "Naive Bayes",
            nb_model,
            input_data
        )

    with hasil_col2:
        pred_tree = tampilkan_hasil(
            "Decision Tree",
            tree_model,
            encode_for_tree(input_data)
        )

    if pred_nb == pred_tree:
        st.info(
            "Kedua model memberikan klasifikasi yang **sama**: "
            f"**{CLASS_NAMES[pred_nb]}**."
        )
    else:
        st.warning(
            "Kedua model memberikan klasifikasi yang **berbeda**: "
            f"Naive Bayes → **{CLASS_NAMES[pred_nb]}**, "
            f"Decision Tree → **{CLASS_NAMES[pred_tree]}**."
        )

    st.caption(
        "Catatan: Decision Tree tanpa pruning memberi probabilitas berupa "
        "proporsi kelas pada daun (leaf) tempat data jatuh, sehingga sering "
        "bernilai 0% atau 100%. Probabilitas Naive Bayes bersifat lebih "
        "'halus' karena dihitung dari distribusi Gaussian tiap fitur."
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
