# RBC LAB — Deteksi dan Penghitungan Sel Darah Merah

Aplikasi Flask untuk analisis citra mikroskopis RBC menggunakan pipeline Otsu, morfologi, dan watershed.

## Menjalankan aplikasi

```bash
git clone "link gua"
cd rbcApps
pip install -r requirements.txt
python app.py
```

Buka `http://localhost:5000`.

## Fitur

- Upload beberapa gambar JPG/JPEG/PNG sekaligus (maksimal 30 gambar, total upload 80 MB).
- Ringkasan analisis dan tabel hasil untuk setiap gambar.
- Pipeline pengolahan citra ditampilkan secara lengkap dari atas ke bawah.
- Tampilan tab: Original, Preprocessing, Otsu Binary, Morfologi, dan Deteksi Final.
- Unduh tabel hasil batch dalam format CSV.
- Unduh masing-masing gambar tahapan.
- Parameter preprocessing dan segmentasi dapat disesuaikan.
- Kolom jumlah RBC sebenarnya tidak ditampilkan pada form utama.

## Catatan

Jumlah RBC merupakan estimasi dari algoritma segmentasi citra, bukan hasil diagnosis medis. Kualitas pewarnaan, pencahayaan, fokus, dan parameter dapat memengaruhi hasil. Periksa hasil deteksi secara visual sebelum digunakan untuk analisis lanjutan.

## Struktur proyek

- `app.py`: aplikasi Flask, upload multi-gambar, dan ekspor CSV.
- `rbc.py`: pipeline pengolahan citra inti.
- `evaluate.py`: evaluasi dataset BCCD.
- `templates/index.html`: antarmuka pengguna.
