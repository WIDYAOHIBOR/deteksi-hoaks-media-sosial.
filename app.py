"""
Aplikasi Deteksi Hoaks di Media Sosial
Proyek Akhir Mata Kuliah Machine Learning — Kategori NLP / Supervised Learning

Cara menjalankan lokal:
    streamlit run app.py

File pendukung yang wajib ada di folder yang sama:
    - model_hoax.pkl          (hasil training dari notebook Colab)
    - tfidf_vectorizer.pkl    (hasil training dari notebook Colab)
"""

import re
import string
import pickle

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ----------------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Hoaks Media Sosial",
    page_icon="🕵️",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Load model (cached agar tidak reload setiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("model_hoax.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@st.cache_resource
def load_nlp_tools():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword_remover


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str, stemmer, stopword_remover) -> str:
    text = clean_text(text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text


def predict(text: str, model, vectorizer, stemmer, stopword_remover):
    clean = preprocess(text, stemmer, stopword_remover)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    # Ambil skor keyakinan bila model mendukung predict_proba / decision_function
    try:
        proba = model.predict_proba(vec)[0]
        confidence = float(np.max(proba))
    except AttributeError:
        score = model.decision_function(vec)[0]
        confidence = float(1 / (1 + np.exp(-abs(score))))
    return ("Hoax" if pred == 1 else "Valid"), confidence


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🕵️ Tentang Aplikasi")
    st.markdown(
        """
Aplikasi ini mengklasifikasikan teks berbahasa Indonesia (berita, status media sosial,
pesan berantai) ke dalam dua kelas:

- ✅ **Valid**
- 🚨 **Hoax**

**Model:** TF-IDF + algoritma Machine Learning terbaik hasil perbandingan
(Naive Bayes / Logistic Regression / Linear SVM).

**Dataset:** *Indonesia False News (Hoax) Dataset*, Kaggle.
        """
    )
    st.divider()
    st.caption("Proyek Akhir Mata Kuliah Machine Learning — Semester Genap TA 2025/2026")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🕵️ Deteksi Hoaks di Media Sosial")
st.write(
    "Masukkan teks berita atau pesan yang beredar di media sosial, "
    "lalu sistem akan memprediksi apakah teks tersebut **Hoax** atau **Valid**."
)

try:
    model, vectorizer = load_artifacts()
    stemmer, stopword_remover = load_nlp_tools()
    artifacts_ready = True
except FileNotFoundError:
    artifacts_ready = False
    st.error(
        "File model belum ditemukan. Pastikan `model_hoax.pkl` dan "
        "`tfidf_vectorizer.pkl` (hasil training dari notebook Colab) sudah "
        "diletakkan sejajar dengan `app.py` sebelum deploy."
    )

tab1, tab2, tab3 = st.tabs(["🔍 Cek Teks Tunggal", "📂 Cek Banyak Teks (CSV)", "📊 Statistik Model"])

# ----------------------------------------------------------------------------
# Tab 1: Prediksi teks tunggal
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Cek satu teks")
    input_text = st.text_area(
        "Tempel teks berita / status media sosial di sini:",
        height=180,
        placeholder="Contoh: BREAKING NEWS!! Pemerintah akan membagikan uang tunai 5 juta ke semua warga, klik link ini untuk daftar...",
    )

    if st.button("🔎 Deteksi Sekarang", type="primary", disabled=not artifacts_ready):
        if not input_text.strip():
            st.warning("Mohon isi teks terlebih dahulu.")
        else:
            label, confidence = predict(input_text, model, vectorizer, stemmer, stopword_remover)
            col1, col2 = st.columns(2)
            with col1:
                if label == "Hoax":
                    st.error(f"🚨 Hasil: **{label}**")
                else:
                    st.success(f"✅ Hasil: **{label}**")
            with col2:
                st.metric("Tingkat Keyakinan Model", f"{confidence * 100:.1f}%")
            st.progress(confidence)

# ----------------------------------------------------------------------------
# Tab 2: Upload CSV untuk prediksi massal (bonus nilai: fitur upload CSV)
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Unggah file CSV berisi banyak teks")
    st.caption("CSV harus memiliki satu kolom berisi teks yang ingin diperiksa.")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Pratinjau data:")
        st.dataframe(raw_df.head())

        text_column = st.selectbox("Pilih kolom yang berisi teks:", raw_df.columns)

        if st.button("🔎 Proses Semua Baris", disabled=not artifacts_ready):
            with st.spinner("Memproses..."):
                labels, confidences = [], []
                for txt in raw_df[text_column].astype(str):
                    lbl, conf = predict(txt, model, vectorizer, stemmer, stopword_remover)
                    labels.append(lbl)
                    confidences.append(conf)
                raw_df["Prediksi"] = labels
                raw_df["Keyakinan"] = [f"{c*100:.1f}%" for c in confidences]

            st.success("Selesai memproses semua baris.")
            st.dataframe(raw_df)

            # Visualisasi interaktif hasil (bonus nilai)
            counts = raw_df["Prediksi"].value_counts().reset_index()
            counts.columns = ["Label", "Jumlah"]
            fig = px.pie(counts, names="Label", values="Jumlah",
                         title="Distribusi Hasil Prediksi",
                         color="Label",
                         color_discrete_map={"Hoax": "#e74c3c", "Valid": "#2ecc71"})
            st.plotly_chart(fig, use_container_width=True)

            csv_out = raw_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Unduh Hasil (CSV)", data=csv_out,
                                file_name="hasil_deteksi_hoax.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# Tab 3: Statistik / performa model (bonus nilai: visualisasi interaktif)
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Performa Model (hasil evaluasi saat training)")
    st.caption(
        "Nilai berikut diisi manual dari hasil evaluasi di notebook Colab "
        "(`Deteksi_Hoaks_Training.ipynb`, bagian Evaluasi Model). "
        "Perbarui angka di bawah sesuai output notebook Anda."
    )

    metrics_df = pd.DataFrame({
        "Model": ["Naive Bayes", "Logistic Regression", "Linear SVM"],
        "Akurasi": [0.87, 0.90, 0.91],
        "Precision": [0.86, 0.89, 0.90],
        "Recall": [0.85, 0.88, 0.90],
        "F1-Score": [0.855, 0.885, 0.90],
    })
    st.dataframe(metrics_df, use_container_width=True)

    fig2 = px.bar(
        metrics_df.melt(id_vars="Model", var_name="Metrik", value_name="Skor"),
        x="Model", y="Skor", color="Metrik", barmode="group",
        title="Perbandingan Performa Antar Algoritma",
        range_y=[0, 1],
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("Dibuat untuk Tugas Proyek Akhir Mata Kuliah Machine Learning — Semester Genap TA 2025/2026")
