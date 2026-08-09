# Streamlit Naive Bayes - Prediksi Klasifikasi Diabetes

Aplikasi Streamlit interaktif untuk implementasi model **Naive Bayes**
berdasarkan dataset penelitian.

## Struktur project

```text
streamlit_naive_bayes/
├── app.py
├── data skripsi.xlsx
└── requirements.txt
```

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Buat repository baru di GitHub.
2. Upload:
   - `app.py`
   - `data skripsi.xlsx`
   - `requirements.txt`
3. Buka Streamlit Community Cloud.
4. Pilih repository GitHub tersebut.
5. Pilih file utama `app.py`.
6. Deploy.

## Catatan metodologi

Untuk deployment, model final Naive Bayes dilatih menggunakan seluruh
dataset setelah model dipilih berdasarkan evaluasi penelitian.

Evaluasi performa yang ditampilkan di aplikasi dihitung menggunakan
**Stratified 10-Fold Cross Validation** dengan `random_state=42`.

Aplikasi ini adalah implementasi model klasifikasi untuk keperluan
penelitian/edukasi dan bukan alat diagnosis medis.
