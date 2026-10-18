"""
Klasifikasi Kematangan Buah Berbasis Warna (Pisang & Jeruk)
UTS Pengolahan Citra Digital - Topik 2

Label  : mentah (dominan hijau) | setengah_matang (hijau kekuningan) | matang (dominan kuning)

Struktur folder dataset (nama label = nama folder):
    dataset/
        mentah/           pisang__terang__01.jpg, jeruk__redup__02.jpg, ...
        setengah_matang/  ...
        matang/           ...
Format nama file : <buah>__<kondisi>__<nomor>.jpg  (huruf besar/kecil & satu/dua '_' ditoleransi)
    kondisi = bebas, mis. terang / redup / latar_ramai (dipakai untuk laporan per kondisi).
    Jika nama file tidak mengikuti format, kondisi dianggap "umum".

Cara menjalankan (satu perintah):
    python klasifikasi_kematangan_buah.py --data dataset --out hasil
    python klasifikasi_kematangan_buah.py --dummy          # uji coba dengan data sintetis

Pustaka: numpy, opencv-python, matplotlib

Pipeline (6 tahap):
  1. Pra-pemrosesan   : ubah ukuran -> Gaussian blur (reduksi derau)
  2. Perbaikan kontras: CLAHE pada kanal L (Lab)
  3. Konversi warna   : RGB -> HSV  
  4. Segmentasi       : 3 metode dibandingkan (Otsu-S [dari nol], Adaptive-S, K-means Lab)
  5. Pasca-proses     : morfologi opening + closing, ambil komponen terbesar, isi lubang
  6. Keputusan        : rasio piksel hijau vs kuning di dalam mask -> label (ambang disetel)

Dua operasi dari nol (NumPy murni): rgb_ke_hsv_scratch() dan otsu_scratch()
"""