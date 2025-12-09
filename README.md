# License Plate Recognition 🔍🚗

Proyek ini merupakan sistem pendeteksian kendaraan dan pembacaan plat nomor menggunakan model YOLO dan pipeline tracking SORT.  
Aplikasi membaca video `.mp4`, mendeteksi kendaraan, mendeteksi plat nomor, membaca teks plat, dan menyimpan hasilnya dalam format CSV.

---

## 🚀 Fitur Utama
- Deteksi kendaraan menggunakan YOLO (COCO model).
- Deteksi plat nomor menggunakan model YOLO khusus.
- Tracking kendaraan dengan SORT agar tidak terhitung dua kali.
- Pembacaan teks plat nomor.
- Ekspor hasil ke file CSV.
- Struktur project rapi dan modular.

---

## 📂 Struktur Folder

```text
license-plate-recognition/
├─ data/
│   ├─ input_videos/        # Video .mp4
│   └─ outputs/             # Hasil CSV
├─ models/                  # Model YOLO (.pt files)
├─ src/
│   ├─ main.py              # Program utama
│   ├─ util.py              # Helper functions
│   ├─ sort.py              # SORT tracker
│   ├─ add_missing_data.py  # Opsional
│   └─ visualize.py         # Visualisasi hasil
├─ .gitignore
├─ requirements.txt
└─ README.md
