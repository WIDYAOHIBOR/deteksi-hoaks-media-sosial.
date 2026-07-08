# 🕵️ Deteksi Hoaks di Media Sosial

Proyek Akhir Mata Kuliah Machine Learning — Semester Genap TA 2025/2026
**Kategori:** Natural Language Processing (NLP) / Supervised Learning — Klasifikasi Teks Biner

## 📌 Deskripsi

Aplikasi ini mendeteksi apakah sebuah teks berita atau pesan yang beredar di media sosial
berbahasa Indonesia tergolong **Hoax** atau **Valid**, menggunakan pendekatan TF-IDF +
algoritma Machine Learning klasik (Naive Bayes, Logistic Regression, Linear SVM).

## 🗂️ Struktur Proyek

```
├── Deteksi_Hoaks_Training.ipynb   # Notebook Google Colab: preprocessing, training, evaluasi
├── app.py                         # Aplikasi Streamlit untuk deployment
├── requirements.txt                # Dependensi Python
├── model_hoax.pkl                  # Model terlatih (dihasilkan dari notebook)
├── tfidf_vectorizer.pkl             # Vectorizer TF-IDF terlatih (dihasilkan dari notebook)
└── README.md
```

## 📊 Dataset

**Indonesia False News (Hoax) Dataset** — Muhammad Ghazi Muharam, Kaggle
https://www.kaggle.com/datasets/muhammadghazimuharam/indonesiafalsenews

Berisi kumpulan teks/status berbahasa Indonesia berlabel hoax dan valid (memenuhi ketentuan
minimal 500 dokumen/teks untuk kategori NLP).

## ⚙️ Tahapan Machine Learning

1. **Studi Literatur** — lihat Laporan Akhir, bagian Tinjauan Pustaka.
2. **Pengumpulan Data** — dataset publik dari Kaggle.
3. **Preprocessing** — case folding, cleaning (URL/mention/angka/tanda baca), stopword removal, stemming (Sastrawi), TF-IDF vectorization.
4. **Pembangunan Model** — perbandingan 3 algoritma: Naive Bayes, Logistic Regression, Linear SVM.
5. **Evaluasi Model** — accuracy, precision, recall, F1-score, confusion matrix.
6. **Deployment** — Streamlit Community Cloud.

## 🚀 Menjalankan secara lokal

```bash
git clone <URL_REPO_INI>
cd <nama-folder>
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deployment ke Streamlit Community Cloud

1. Push seluruh isi folder ini (termasuk `model_hoax.pkl` dan `tfidf_vectorizer.pkl`) ke repository GitHub.
2. Buka https://share.streamlit.io, login dengan akun GitHub.
3. Klik **New app**, pilih repo ini, branch `main`, file utama `app.py`.
4. Klik **Deploy**. Tunggu proses build selesai — aplikasi akan memiliki URL publik
   berformat `https://<nama-app>.streamlit.app`.

## ✨ Fitur

- Deteksi hoax untuk satu teks langsung.
- Unggah file CSV untuk deteksi banyak teks sekaligus (batch prediction).
- Visualisasi interaktif distribusi hasil prediksi dan perbandingan performa model.
- Unduh hasil prediksi batch dalam format CSV.

## 👤 Kontributor

- Nama: _(isi nama Anda)_
- NIM: _(isi NIM Anda)_
- Kelas: _(isi kelas Anda)_

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik (Tugas Akhir Mata Kuliah Machine Learning).
