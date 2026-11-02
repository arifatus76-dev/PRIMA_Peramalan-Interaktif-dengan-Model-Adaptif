# 📊 PRIMA — Peramalan Interaktif dengan Model Adaptif

Aplikasi web berbasis **Streamlit** untuk peramalan data deret waktu (*time series forecasting*), dikembangkan oleh **BPS Kota Bontang**.

PRIMA menyediakan enam model peramalan yang dapat dijalankan dan dibandingkan secara simultan, lengkap dengan interval kepercayaan dan visualisasi interaktif.

---

## Fitur Utama

- **Multi-user** — Mendukung banyak pengguna dengan dua peran: user dan admin
- **Manajemen dataset** — Membuat, memilih, dan menghapus dataset per pengguna
- **Import data** — Unggah data dari file CSV atau Excel (.xlsx/.xls)
- **Variabel eksogen** — Mendukung prediktor eksternal untuk model SARIMAX dan MLPX
- **Auto-tuning** — Pemilihan parameter model secara otomatis berdasarkan MDAPE
- **Perbandingan model** — Membandingkan semua model secara simultan dalam satu tampilan
- **Confidence Interval** — Interval kepercayaan 80% dan 95% untuk setiap ramalan
- **Download hasil** — Grafik (PNG) dan data lengkap (Excel)
- **Uji stasioneritas** — ADF dan KPSS sebagai informasi pendukung
- **Estimasi GPH** — Estimasi orde differencing pecahan untuk model SARFIMA

## Model Peramalan

| Model | Deskripsi |
|-------|-----------|
| **Naïve** | Model baseline yang menggunakan nilai terakhir sebagai ramalan |
| **SARIMA** | Model statistik klasik dengan komponen musiman |
| **SARFIMA** | SARIMA dengan kemampuan menangkap memori jangka panjang |
| **SARIMAX** | SARIMA yang ditambah variabel eksogen (prediktor eksternal) |
| **MLP** | Neural network (jaringan saraf tiruan) untuk pola non-linier |
| **MLPX** | MLP yang ditambah variabel eksogen |

## Teknologi

- **Python 3.9+**
- **Streamlit** — framework antarmuka web
- **statsmodels** — model SARIMA/SARFIMA/SARIMAX
- **scikit-learn** — model MLP/MLPX
- **Altair** — grafik interaktif
- **matplotlib** — grafik untuk download (PNG)
- **SQLite** — penyimpanan data lokal (tanpa perlu setup database terpisah)

## Instalasi

### 1. Clone repository

```bash
git clone https://github.com/username/prima-forecasting.git
cd prima-forecasting
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada alamat `http://localhost:8501`.

## Akun Default

Saat pertama kali dijalankan, aplikasi membuat akun admin secara otomatis:

| Username | Password |
|----------|----------|
| `admin`  | `admin`  |

> ⚠️ **Penting:** Segera ganti password admin setelah login pertama.

## Struktur File

```
prima-forecasting/
├── app.py              # Aplikasi utama Streamlit
├── requirements.txt    # Daftar dependensi Python
├── README.md           # Dokumentasi ini
└── forecasting.db      # Database SQLite (dibuat otomatis saat pertama dijalankan)
```

## Lisensi

Dikembangkan oleh **Badan Pusat Statistik (BPS) Kota Bontang** © 2024.

---

> Untuk panduan penggunaan lengkap, lihat **Buku Panduan PRIMA** yang tersedia dalam format Word/PDF.
