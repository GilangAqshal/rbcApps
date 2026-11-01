# Deteksi dan Penghitungan Sel Darah Merah (RBC)

Aplikasi web pengolahan citra: Thresholding Otsu + Operasi Morfologi (+ watershed untuk sel menempel).
Dataset: BCCD (Blood Cell Count and Detection), https://www.kaggle.com/datasets/orvile/bccd-blood-cell-count-and-detection-dataset

## Menjalankan

```bash
pip install -r requirements.txt
python app.py
```

Buka http://localhost:5000, upload citra dari folder gambar BCCD (JPG), lalu klik **Proses citra**.
Isi kolom "Jumlah RBC sebenarnya" (hitung dari file anotasi) untuk melihat selisih dan akurasi.

## Struktur

- `rbc.py`  : pipeline pengolahan citra (bisa dipakai tanpa web)
- `app.py`  : aplikasi Flask
- `evaluate.py` : evaluasi otomatis terhadap anotasi BCCD (hasil: tabel + evaluasi.csv)
- `templates/index.html` : tampilan web

## Tahapan pengolahan citra

1. Citra asli
2. Grayscale (channel hijau atau grayscale standar)
3. Smoothing: Gaussian blur
4. Peningkatan kontras: CLAHE
5. Thresholding Otsu (inverse)
6. Morfologi: opening, closing, hole filling
6b. Deteksi dan penghapusan WBC berdasarkan warna (HSV: hue biru-ungu, saturasi tinggi)
7. Pemisahan sel menempel: distance transform, puncak lokal, watershed
8. Connected component, filter luas, penghitungan dan anotasi kontur

## Pengujian pada dataset (untuk laporan)

```bash
git clone https://github.com/Shenggan/BCCD_Dataset.git
python evaluate.py --root BCCD_Dataset/BCCD --n 15
```

Ambil minimal 10 citra BCCD, jalankan satu per satu, catat jumlah RBC hasil program
dan jumlah anotasi (file XML di dataset), lalu hitung selisih / MAE / akurasi.
