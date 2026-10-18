"""
Klasifikasi Kematangan Buah Berbasis Warna (Pisang & Jeruk)
UTS Pengolahan Citra Digital - Topik 2

Label  : mentah (dominan hijau) | setengah_matang (hijau kekuningan) | matang (dominan kuning)

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

import argparse
import glob
import os
import random
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["mentah", "setengah_matang", "matang"]
EXT = (".jpg", ".jpeg", ".png", ".bmp")
SEED = 42
UKURAN_MAKS = 256          # sisi terpanjang setelah resize (alasan: cepat, detail warna tetap cukup)

# Hue (derajat, 0-360): kuning/jingga = [HUE_KUNING_MIN, batas), hijau = [batas, HUE_HIJAU_MAX].
# "batas" TIDAK ditebak: disetel per buah lewat grid search (lihat tuning_params) dan
# dapat dilihat pada histogram hue per buah (histogram_hue.png).
HUE_KUNING_MIN, HUE_HIJAU_MAX = 15.0, 170.0
HUE_GRID = list(range(45, 85, 5))
S_MIN, V_MIN = 0.25, 0.20  # abaikan piksel pucat/gelap (hue tidak stabil)


# =====================================================================
# 1. DUA OPERASI DARI NOL (hanya NumPy)
# =====================================================================
def rgb_ke_hsv_scratch(rgb_uint8):
    """RGB uint8 (H,W,3) -> HSV float32: H 0-360, S 0-1, V 0-1. Tanpa OpenCV/skimage."""
    x = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.max(x, axis=-1)
    mn = np.min(x, axis=-1)
    d = mx - mn
    d_aman = np.where(d == 0, 1.0, d)           

    h = np.zeros_like(mx)
    h = np.where(mx == r, ((g - b) / d_aman) % 6.0, h)
    h = np.where((mx == g) & (mx != r), (b - r) / d_aman + 2.0, h)
    h = np.where((mx == b) & (mx != r) & (mx != g), (r - g) / d_aman + 4.0, h)
    h = np.where(d == 0, 0.0, h) * 60.0

    s = np.where(mx > 0, d / np.where(mx == 0, 1.0, mx), 0.0)
    v = mx
    return np.stack([h, s, v], axis=-1).astype(np.float32)


def otsu_scratch(gray_uint8):
    """Thresholding Otsu dari nol. Mengembalikan ambang t; piksel > t = foreground."""
    hist = np.bincount(gray_uint8.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    i = np.arange(256, dtype=np.float64)
    omega = np.cumsum(p)                 
    mu = np.cumsum(p * i)                
    mu_t = mu[-1]
    pembagi = omega * (1.0 - omega)
    pembagi[pembagi == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / pembagi   
    return int(np.nanargmax(sigma_b))


# =====================================================================
# 2. PIPELINE
# =====================================================================
def praproses(bgr):
    """Tahap 1-2: resize, reduksi derau (Gaussian), perbaikan kontras (CLAHE pada L). Return RGB."""
    h, w = bgr.shape[:2]
    skala = UKURAN_MAKS / max(h, w)
    if skala < 1:
        bgr = cv2.resize(bgr, (int(w * skala), int(h * skala)), interpolation=cv2.INTER_AREA)
    bgr = cv2.GaussianBlur(bgr, (5, 5), 0)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)


def _orientasi(mask):
    """Objek = region yang tidak dominan di tepi citra (latar biasanya menyentuh tepi)."""
    tepi = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return (1 - mask) if tepi.mean() > 0.5 else mask


def seg_otsu_s(rgb, hsv):
    s8 = (hsv[..., 1] * 255).astype(np.uint8)
    t = otsu_scratch(s8)                                  # <- Otsu buatan sendiri
    return _orientasi((s8 > t).astype(np.uint8))


def seg_adaptive_s(rgb, hsv):
    s8 = (hsv[..., 1] * 255).astype(np.uint8)
    m = cv2.adaptiveThreshold(s8, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 51, -5)
    return _orientasi(m.astype(np.uint8))


def seg_kmeans_lab(rgb, hsv):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    z = lab[..., 1:].reshape(-1, 2).astype(np.float32)
    cv2.setRNGSeed(SEED)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lbl, _ = cv2.kmeans(z, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return _orientasi(lbl.reshape(rgb.shape[:2]).astype(np.uint8))


METODE_SEG = {
    "Otsu-S (scratch)": seg_otsu_s,
    "Adaptive-S": seg_adaptive_s,
    "KMeans-Lab(ab)": seg_kmeans_lab,
}


def pasca_morfologi(mask):
    """Tahap 5: opening (buang bintik) + closing (tutup celah), simpan kontur terbesar & isi lubang."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    kontur, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not kontur:
        return m
    out = np.zeros_like(m)
    cv2.drawContours(out, [max(kontur, key=cv2.contourArea)], -1, 1, thickness=-1)
    return out


def hitung_rasio(hsv, mask, hue_batas=60.0):
    """Tahap 6a: rasio hijau = hijau / (hijau + kuning) pada piksel valid dalam mask."""
    h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    valid = (mask > 0) & (sat > S_MIN) & (v > V_MIN)
    hijau = np.sum(valid & (h >= hue_batas) & (h <= HUE_HIJAU_MAX))
    kuning = np.sum(valid & (h >= HUE_KUNING_MIN) & (h < hue_batas))
    total = hijau + kuning
    rasio = hijau / total if total > 0 else 0.0
    frac_valid = valid.sum() / max(mask.sum(), 1)
    return float(rasio), float(frac_valid)


def keputusan(rasio, lo, hi):
    """Tahap 6b: ambang rasio hijau -> label."""
    if rasio >= hi:
        return "mentah"
    if rasio <= lo:
        return "matang"
    return "setengah_matang"


# =====================================================================
# 3. DATASET
# =====================================================================
def parse_nama(path):
    """'Jeruk__latar_ramai__03.jpeg' / 'pisang_terang_04.jpg' -> ('jeruk', 'latar_ramai')."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    tok = [t for t in stem.split("_") if t]
    if tok and tok[-1].isdigit():
        tok = tok[:-1]
    if len(tok) >= 2:
        return tok[0], "_".join(tok[1:])
    return (tok[0] if tok else "?"), "umum"


def muat_dataset(folder):
    data = []
    for label in LABELS:
        for path in sorted(glob.glob(os.path.join(folder, label, "*"))):
            if not path.lower().endswith(EXT):
                continue
            bgr = cv2.imread(path)
            if bgr is None:
                print(f"[PERINGATAN] gagal membaca {path}")
                continue
            buah, kondisi = parse_nama(path)
            rgb = praproses(bgr)
            data.append(dict(path=path, nama=os.path.basename(path), label=label,
                             buah=buah, kondisi=kondisi, rgb=rgb,
                             hsv=rgb_ke_hsv_scratch(rgb)))
    return data


def bagi_data(data, frac_dev=0.6):
    """Pembagian stratified per label: ~60% pengembangan, ~40% uji (tidak disentuh saat tuning)."""
    rng = random.Random(SEED)
    dev, uji = [], []
    for label in LABELS:
        item = [d for d in data if d["label"] == label]
        rng.shuffle(item)
        n = int(round(len(item) * frac_dev))
        dev += item[:n]
        uji += item[n:]
    return dev, uji


def proses_semua(items, fungsi_seg):
    """Tahap 4-5: segmentasi + morfologi."""
    for d in items:
        d["mask"] = pasca_morfologi(fungsi_seg(d["rgb"], d["hsv"]))
        d["area"] = float(d["mask"].mean())


def terapkan_params(items, params):
    """Tahap 6: hitung rasio & putuskan label memakai parameter per buah."""
    for d in items:
        p = params.get(d["buah"], params["*"])
        d["rasio"], d["frac_valid"] = hitung_rasio(d["hsv"], d["mask"], p["hue"])
        d["lo"], d["hi"], d["hue"] = p["lo"], p["hi"], p["hue"]
        d["pred"] = keputusan(d["rasio"], p["lo"], p["hi"])


# =====================================================================
# 4. EVALUASI
# =====================================================================
def matriks_konfusi(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[LABELS.index(t), LABELS.index(p)] += 1
    return cm


def akurasi(y_true, y_pred):
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0


def tabel(header, baris):
    lebar = [max(len(str(x)) for x in kol) for kol in zip(header, *baris)]
    garis = "+-" + "-+-".join("-" * w for w in lebar) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in lebar) + " |"
    print(garis); print(fmt.format(*header)); print(garis)
    for b in baris:
        print(fmt.format(*[str(x) for x in b]))
    print(garis)


def cari_param(items):
    """Grid search (batas hue, lo, hi). Seri -> pilih parameter yang paling 'tengah'."""
    y = [d["label"] for d in items]
    terbaik = None
    for hb in HUE_GRID:
        r = [hitung_rasio(d["hsv"], d["mask"], hb)[0] for d in items]
        for lo in np.arange(0.05, 0.55, 0.05):
            for hi in np.arange(lo + 0.10, 0.96, 0.05):
                acc = akurasi(y, [keputusan(x, lo, hi) for x in r])
                skor = (acc, -(abs(lo - 0.25) + abs(hi - 0.75) + abs(hb - 60) / 100))
                if terbaik is None or skor > terbaik[0]:
                    terbaik = (skor, dict(lo=float(lo), hi=float(hi), hue=float(hb)))
    return terbaik[1]


def tuning_params(items):
    """Parameter global ('*') + parameter khusus tiap buah (jika data pengembangannya cukup)."""
    params = {"*": cari_param(items)}
    for b in sorted({d["buah"] for d in items}):
        sub = [d for d in items if d["buah"] == b]
        if len(sub) >= 9 and len({d["label"] for d in sub}) == 3:
            params[b] = cari_param(sub)
    terapkan_params(items, params)
    return params, akurasi([d["label"] for d in items], [d["pred"] for d in items])


# =====================================================================
# 5. VERIFIKASI IMPLEMENTASI SENDIRI vs PUSTAKA
# =====================================================================
def verifikasi(data, n_min=10):
    print("\n" + "=" * 70)
    print("VERIFIKASI IMPLEMENTASI SENDIRI vs PUSTAKA (OpenCV)")
    print("=" * 70)
    sampel = data[:max(n_min, min(len(data), 20))]
    if len(sampel) < n_min:
        print(f"[PERINGATAN] hanya {len(sampel)} citra; ketentuan minimal {n_min}.")

    baris_hsv, baris_otsu = [], []
    for d in sampel:
        rgb = d["rgb"]
        # --- RGB -> HSV
        t0 = time.perf_counter(); h_own = rgb_ke_hsv_scratch(rgb); t_own = time.perf_counter() - t0
        t0 = time.perf_counter()
        h_lib = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)
        t_lib = time.perf_counter() - t0
        sel = np.abs(h_own - h_lib)
        sel[..., 0] = np.minimum(sel[..., 0], 360 - sel[..., 0])   # hue melingkar
        baris_hsv.append((d["nama"][:24], f"{sel[...,0].max():.4f}", f"{sel[...,1].max():.5f}",
                          f"{sel[...,2].max():.5f}", f"{t_own*1e3:.2f}", f"{t_lib*1e3:.2f}"))
        # --- Otsu pada kanal S
        s8 = (h_own[..., 1] * 255).astype(np.uint8)
        t0 = time.perf_counter(); t_o = otsu_scratch(s8); w_own = time.perf_counter() - t0
        t0 = time.perf_counter()
        t_l, m_lib = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        w_lib = time.perf_counter() - t0
        beda = int(np.sum((s8 > t_o) != (m_lib > 0)))
        baris_otsu.append((d["nama"][:24], t_o, int(t_l), beda, f"{w_own*1e3:.2f}", f"{w_lib*1e3:.2f}"))

    print("\n[1] RGB -> HSV  (selisih maksimum per kanal, waktu dalam ms)")
    tabel(["citra", "maxΔH(°)", "maxΔS", "maxΔV", "t_own", "t_lib"], baris_hsv)
    print("\n[2] Thresholding Otsu pada kanal S  (selisih piksel mask, waktu dalam ms)")
    tabel(["citra", "T_own", "T_lib", "Δpiksel", "t_own", "t_lib"], baris_otsu)
    tot = sum(b[3] for b in baris_otsu)
    print(f"Ringkasan: {len(sampel)} citra | total piksel mask berbeda (Otsu) = {tot}"
          f" | maks selisih H = {max(float(b[1]) for b in baris_hsv):.4f}°")
    print("Catatan: versi sendiri umumnya lebih lambat dari OpenCV (C++ teroptimasi); itu wajar.")


# =====================================================================
# 6. FIGUR & ANALISIS KEGAGALAN
# =====================================================================
def simpan_konfusi(cm, judul, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS, rotation=20); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya"); ax.set_title(judul)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def simpan_histogram_hue(items, params, path):
    """Histogram hue di dalam mask, satu panel per buah (pisang, jeruk), satu garis per label."""
    buah_list = sorted({d["buah"] for d in items})
    warna = {"mentah": "green", "setengah_matang": "olive", "matang": "goldenrod"}
    fig, axes = plt.subplots(1, len(buah_list), figsize=(6.2 * len(buah_list), 4), squeeze=False)
    baris = []
    for ax, b in zip(axes[0], buah_list):
        p = params.get(b, params["*"])
        for label in LABELS:
            sub = [d for d in items if d["buah"] == b and d["label"] == label]
            if not sub:
                continue
            acc, semua = np.zeros(36), []
            for d in sub:
                h, sat, v = d["hsv"][..., 0], d["hsv"][..., 1], d["hsv"][..., 2]
                valid = (d["mask"] > 0) & (sat > S_MIN) & (v > V_MIN)
                hist, _ = np.histogram(h[valid], bins=36, range=(0, 360))
                acc += hist / max(hist.sum(), 1)
                semua.append(h[valid])
            ax.plot(np.arange(36) * 10 + 5, acc / len(sub), label=f"{label} (n={len(sub)})",
                    color=warna[label], lw=2)
            semua = np.concatenate(semua)
            baris.append([b, label, len(sub), f"{np.median(semua):.1f}" if semua.size else "-"])
        ax.axvline(p["hue"], ls="--", c="k", label=f"batas hijau/kuning = {p['hue']:.0f}°")
        ax.set_xlim(0, 200)
        ax.set_xlabel("Hue (derajat)"); ax.set_ylabel("Proporsi piksel (rerata)")
        ax.set_title(f"Histogram hue - {b}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print("\nRingkasan hue (median, derajat) piksel valid dalam mask - citra pengembangan:")
    tabel(["buah", "label", "n", "median hue"], baris)


def sebab_kegagalan(d):
    """Heuristik: tebak tahap penyebab, lalu usulan perbaikan."""
    lo, hi = d["lo"], d["hi"]
    if d["area"] < 0.03 or d["area"] > 0.85:
        return ("Segmentasi",
                "mask tidak menangkap buah (terlalu kecil/besar). Usul: koreksi iluminasi, "
                "GrabCut/watershed, atau latar polos.")
    if d["frac_valid"] < 0.30:
        return ("Pra-pemrosesan / pencahayaan",
                "banyak piksel buah terlalu gelap/pucat (S/V rendah) sehingga hue tak stabil. "
                "Usul: normalisasi white-balance, gunakan ruang Lab (a*, b*).")
    if d["rasio"] == 0.0 and d["label"] != "matang":
        return ("Keputusan (batas hue hijau/kuning)",
                f"tidak ada piksel di atas batas hue {d['hue']:.0f}°, padahal buah belum matang "
                "(hijau kekuningan bergeser ke hue rendah). Usul: turunkan batas hue / setel "
                "terpisah per buah / tambah fitur a* Lab.")
    return ("Keputusan (ambang rasio)",
            f"rasio hijau {d['rasio']:.2f} dekat batas kelas (lo={lo:.2f}, hi={hi:.2f}); "
            "kelas bersebelahan saling tumpang tindih. Usul: tambah fitur (mean a* Lab, "
            "saturasi) atau pakai klasifikator sederhana (k-NN/Naive Bayes) pada histogram warna.")


def analisis_kegagalan(uji, folder_out, min_citra=3):
    print("\n" + "=" * 70)
    print("ANALISIS KEGAGALAN (citra uji)")
    print("=" * 70)
    salah = [d for d in uji if d["pred"] != d["label"]]
    tampil = list(salah)
    if len(tampil) < min_citra:
        print(f"Hanya {len(salah)} citra salah; ditambah kasus 'nyaris salah' (margin terkecil) "
              f"untuk melengkapi {min_citra} citra.")
        benar = [d for d in uji if d["pred"] == d["label"]]
        def margin(d):
            return min(abs(d["rasio"] - d["lo"]), abs(d["rasio"] - d["hi"]))
        tampil += sorted(benar, key=margin)[:min_citra - len(tampil)]
    tampil = tampil[:6]
    if not tampil:
        print("Tidak ada citra untuk dianalisis."); return

    fig, axes = plt.subplots(len(tampil), 2, figsize=(7, 3.2 * len(tampil)), squeeze=False)
    for i, d in enumerate(tampil):
        tahap, saran = sebab_kegagalan(d)
        status = "SALAH" if d["pred"] != d["label"] else "nyaris salah"
        print(f"\n[{i+1}] {d['nama']} ({status}) | benar={d['label']} | prediksi={d['pred']} "
              f"| rasio_hijau={d['rasio']:.2f} | kondisi={d['kondisi']}")
        print(f"    Tahap bermasalah : {tahap}")
        print(f"    Penjelasan/usul  : {saran}")
        overlay = d["rgb"].copy()
        kontur, _ = cv2.findContours(d["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, kontur, -1, (255, 0, 0), 2)
        axes[i, 0].imshow(d["rgb"]); axes[i, 0].set_title(f"{d['nama'][:26]}\nbenar: {d['label']}", fontsize=8)
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"pred: {d['pred']} | rasio={d['rasio']:.2f}\n{status}", fontsize=8)
        for a_ in axes[i]: a_.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(folder_out, "analisis_kegagalan.png"), dpi=120)
    plt.close(fig)


def analisis_benar(uji, folder_out, n_total=10):
    """Contoh prediksi BENAR pada citra uji: minimal n_total citra, dibagi rata per buah (50:50
    untuk dua buah), variasi label digilir, diurutkan dari yang paling yakin (margin terbesar)."""
    print("\n" + "=" * 70)
    print("CONTOH PREDIKSI BENAR (citra uji)")
    print("=" * 70)
    benar = [d for d in uji if d["pred"] == d["label"]]
    buah_list = sorted({d["buah"] for d in uji})
    kuota = {b: n_total // len(buah_list) for b in buah_list}
    for b in buah_list[:n_total % len(buah_list)]:
        kuota[b] += 1

    def margin(d):
        return min(abs(d["rasio"] - d["lo"]), abs(d["rasio"] - d["hi"]))

    terpilih = []
    for b in buah_list:
        pool = {l: sorted([d for d in benar if d["buah"] == b and d["label"] == l],
                          key=margin, reverse=True) for l in LABELS}
        ambil = []
        while len(ambil) < kuota[b] and any(pool.values()):
            for l in LABELS:
                if pool[l] and len(ambil) < kuota[b]:
                    ambil.append(pool[l].pop(0))
        if len(ambil) < kuota[b]:
            print(f"[PERINGATAN] citra uji benar untuk '{b}' hanya {len(ambil)} (target {kuota[b]}).")
        terpilih += ambil
    if not terpilih:
        print("Tidak ada prediksi benar untuk ditampilkan."); return
    terpilih.sort(key=lambda d: (d["buah"], LABELS.index(d["label"])))

    baris = [[i + 1, d["nama"][:28], d["buah"], d["kondisi"], d["label"], f"{d['rasio']:.2f}"]
             for i, d in enumerate(terpilih)]
    tabel(["no", "file", "buah", "kondisi", "label=prediksi", "rasio_hijau"], baris)
    for b in buah_list:
        print(f"  {b}: {sum(d['buah'] == b for d in terpilih)} citra "
              f"({sum(d['buah'] == b for d in terpilih) / len(terpilih) * 100:.0f}%)")

    n = len(terpilih)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 4, figsize=(13, 3.1 * rows), squeeze=False)
    for a_ in axes.ravel():
        a_.axis("off")
    for i, d in enumerate(terpilih):
        r, c = i // 2, (i % 2) * 2
        overlay = d["rgb"].copy()
        kontur, _ = cv2.findContours(d["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, kontur, -1, (255, 0, 0), 2)
        axes[r, c].imshow(d["rgb"])
        axes[r, c].set_title(f"{d['nama'][:24]}\n{d['buah']} | {d['kondisi']} | benar: {d['label']}", fontsize=8)
        axes[r, c + 1].imshow(overlay)
        axes[r, c + 1].set_title(f"pred: {d['pred']} | rasio={d['rasio']:.2f}\nBENAR", fontsize=8, color="green")
    fig.tight_layout(); fig.savefig(os.path.join(folder_out, "analisis_benar.png"), dpi=120)
    plt.close(fig)


# =====================================================================
# 7. DATA DUMMY (hanya untuk uji coba program)
# =====================================================================
def buat_dummy(folder, n_per_label=20):
    rng = np.random.default_rng(SEED)
    frac = {"mentah": (0.85, 1.0), "setengah_matang": (0.35, 0.65), "matang": (0.0, 0.15)}
    kondisi_list = ["terang", "redup", "latar_abu"]
    for label in LABELS:
        os.makedirs(os.path.join(folder, label), exist_ok=True)
        for n in range(n_per_label):
            kond = kondisi_list[n % 3]
            buah = "pisang" if n % 2 == 0 else "jeruk"
            bg = {"terang": 235, "redup": 150, "latar_abu": 190}[kond]
            img = np.full((300, 400, 3), bg, np.uint8)
            img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
            m = np.zeros((300, 400), np.uint8)
            if buah == "pisang":
                cv2.ellipse(m, (200, 150), (150, 55), int(rng.integers(-30, 30)), 0, 360, 1, -1)
            else:
                cv2.circle(m, (200, 150), 90, 1, -1)
            noise = cv2.GaussianBlur(rng.random((300, 400)).astype(np.float32), (0, 0), 12)
            f = rng.uniform(*frac[label])
            hijau = noise <= np.quantile(noise[m > 0], f)
            warna_h, warna_k = np.array([60, 170, 70]), np.array([50, 215, 235])   # BGR
            if buah == "jeruk":
                warna_k = np.array([40, 140, 245])
            skala = {"terang": 1.0, "redup": 0.65, "latar_abu": 0.9}[kond]
            buahimg = np.where(hijau[..., None], warna_h, warna_k) * skala
            img[m > 0] = np.clip(buahimg[m > 0] + rng.normal(0, 6, (int(m.sum()), 3)), 0, 255)
            cv2.imwrite(os.path.join(folder, label, f"{buah}__{kond}__{n:02d}.jpg"), img)
    print(f"[INFO] Data dummy dibuat di '{folder}' ({n_per_label*3} citra).")


# =====================================================================
# 8a. MENU INTERAKTIF 
# =====================================================================
def _baca_citra(path):
    """np.fromfile + imdecode: aman untuk path berisi spasi/Unicode (cv2.imread sering gagal di Windows)."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def _buka_gambar(path):
    """Buka gambar hasil dengan penampil bawaan OS (backend matplotlib 'Agg' tidak punya jendela)."""
    import subprocess
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        print(f"(Gambar tidak bisa dibuka otomatis, buka manual: {path})")


def klasifikasi_satu_citra(path, fungsi_seg, params):
    """Jalankan pipeline yang sama (tahap 1-6) pada satu citra. Return dict seperti item dataset, atau None."""
    bgr = _baca_citra(path)
    if bgr is None:
        return None
    buah, kondisi = parse_nama(path)
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    rgb = praproses(bgr)
    d = dict(path=path, nama=os.path.basename(path), label=folder if folder in LABELS else None,
             buah=buah, kondisi=kondisi, rgb=rgb, hsv=rgb_ke_hsv_scratch(rgb))
    proses_semua([d], fungsi_seg)
    terapkan_params([d], params)       # buah yang tak dikenal otomatis memakai parameter global ('*')
    return d


def tampilkan_hasil(d, folder_out):
    """Cetak hasil di terminal dan simpan/buka gambar: citra, kontur mask, histogram hue."""
    print("\n" + "-" * 60)
    print(f"File            : {d['nama']}")
    print(f"Buah / kondisi  : {d['buah']} / {d['kondisi']}")
    print(f"Rasio hijau     : {d['rasio']:.2f}  (batas hue {d['hue']:.0f}°, lo={d['lo']:.2f}, hi={d['hi']:.2f})")
    print(f"PREDIKSI        : {d['pred'].upper()}")
    if d["label"] in LABELS:
        print(f"Label sebenarnya: {d['label']}  ->  {'BENAR' if d['pred'] == d['label'] else 'SALAH'}")
    else:
        print("Label sebenarnya: (tidak diketahui)")
    if d["area"] < 0.03 or d["area"] > 0.85:
        print("[PERINGATAN] mask buah tidak wajar (terlalu kecil/besar); hasil kemungkinan tidak andal.")
    if d["frac_valid"] < 0.30:
        print("[PERINGATAN] banyak piksel buah terlalu gelap/pucat; hue tidak stabil.")
    print("-" * 60)

    overlay = d["rgb"].copy()
    kontur, _ = cv2.findContours(d["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, kontur, -1, (255, 0, 0), 2)
    h, sat, v = d["hsv"][..., 0], d["hsv"][..., 1], d["hsv"][..., 2]
    valid = (d["mask"] > 0) & (sat > S_MIN) & (v > V_MIN)
    hist, _ = np.histogram(h[valid], bins=36, range=(0, 360))

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(d["rgb"]); ax[0].set_title("Citra (setelah pra-proses)", fontsize=9); ax[0].axis("off")
    ax[1].imshow(overlay)
    ax[1].set_title(f"Segmentasi (kontur biru)\nprediksi: {d['pred']} | rasio={d['rasio']:.2f}", fontsize=9)
    ax[1].axis("off")
    ax[2].bar(np.arange(36) * 10 + 5, hist / max(hist.sum(), 1), width=9, color="goldenrod")
    ax[2].axvline(d["hue"], ls="--", c="k", label=f"batas hijau/kuning {d['hue']:.0f}°")
    ax[2].set_xlim(0, 200); ax[2].set_xlabel("Hue (derajat)"); ax[2].set_ylabel("Proporsi piksel")
    ax[2].set_title("Histogram hue dalam mask", fontsize=9); ax[2].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(folder_out, "hasil_menu.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Gambar hasil disimpan: {path}")
    _buka_gambar(path)


def _menu_daftar_uji(uji_urut, folder_out):
    tabel(["no", "file", "buah", "kondisi", "label"],
          [[i + 1, d["nama"][:30], d["buah"], d["kondisi"], d["label"]] for i, d in enumerate(uji_urut)])
    while True:
        try:
            p = input("Nomor citra (b = kembali ke menu): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if p in ("b", ""):
            return
        if p.isdigit() and 1 <= int(p) <= len(uji_urut):
            tampilkan_hasil(uji_urut[int(p) - 1], folder_out)
        else:
            print(f"Masukkan angka 1-{len(uji_urut)} atau 'b'.")


def menu_interaktif(uji, fungsi_seg, params, nama_metode, folder_out):
    """Menu terminal berulang sampai pengguna memilih 0 (selesai)."""
    uji_urut = sorted(uji, key=lambda d: (d["buah"], LABELS.index(d["label"]), d["nama"]))
    while True:
        print("\n" + "=" * 60)
        print(f"MENU KLASIFIKASI KEMATANGAN BUAH   (metode: {nama_metode})")
        print("=" * 60)
        print("  1. Pilih citra dari daftar citra uji")
        print("  2. Masukkan path citra sendiri")
        print("  3. Lihat ringkasan hasil citra uji")
        print("  0. Selesai")
        try:
            pilih = input("Pilih menu [0-3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar dari menu."); break

        if pilih == "0":
            print("Selesai. Terima kasih."); break
        elif pilih == "1":
            if uji_urut:
                _menu_daftar_uji(uji_urut, folder_out)
            else:
                print("Tidak ada citra uji.")
        elif pilih == "2":
            try:
                path = input("Path citra (.jpg/.png/.bmp): ").strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                print(); continue
            if not os.path.isfile(path):
                print("File tidak ditemukan."); continue
            d = klasifikasi_satu_citra(path, fungsi_seg, params)
            if d is None:
                print("File tidak bisa dibaca sebagai citra.")
            else:
                tampilkan_hasil(d, folder_out)
        elif pilih == "3":
            yt, yp = [d["label"] for d in uji], [d["pred"] for d in uji]
            print(f"\nAkurasi citra uji: {akurasi(yt, yp)*100:.1f}%  (n={len(uji)})")
            cm = matriks_konfusi(yt, yp)
            tabel(["sebenarnya \\ prediksi"] + LABELS, [[LABELS[i]] + list(cm[i]) for i in range(3)])
        else:
            print("Pilihan tidak valid, masukkan 0-3.")


# =====================================================================
# 8. MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Klasifikasi kematangan buah berbasis warna")
    ap.add_argument("--data", default="dataset", help="folder dataset (berisi mentah/, setengah_matang/, matang/)")
    ap.add_argument("--out", default="hasil", help="folder keluaran (gambar & csv)")
    ap.add_argument("--dummy", action="store_true", help="buat & pakai data sintetis untuk uji coba")
    ap.add_argument("--tanpa_menu", action="store_true", help="lewati menu interaktif")   # <-- TAMBAHAN
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED)
    if args.dummy:
        args.data = "dataset_dummy"
        buat_dummy(args.data)
    os.makedirs(args.out, exist_ok=True)

    data = muat_dataset(args.data)
    if len(data) < 6:
        sys.exit(f"Dataset kosong/terlalu kecil di '{args.data}'. Lihat struktur folder di bagian atas file.")
    if len(data) < 50 and not args.dummy:
        print(f"[PERINGATAN] {len(data)} citra < 50 (ketentuan minimal UTS).")
    dev, uji = bagi_data(data)
    print(f"Total {len(data)} citra | pengembangan {len(dev)} | uji {len(uji)}")
    for l in LABELS:
        print(f"  {l:16s}: {sum(d['label']==l for d in data)} citra")

    # --- Verifikasi 2 operasi dari nol
    verifikasi(data)

    # --- Perbandingan metode segmentasi (akurasi citra pengembangan; parameter disetel per metode & per buah)
    print("\n" + "=" * 70)
    print("PERBANDINGAN METODE SEGMENTASI (citra pengembangan, parameter disetel per metode)")
    print("=" * 70)
    hasil_metode = {}
    for nama, fungsi in METODE_SEG.items():
        proses_semua(dev, fungsi)
        params, acc = tuning_params(dev)
        area = np.mean([d["area"] for d in dev])
        hasil_metode[nama] = (acc, params)
        print(f"  {nama:18s} akurasi_dev={acc*100:5.1f}%  rerata_area_mask={area:.2f}")
    terbaik = max(hasil_metode, key=lambda k: hasil_metode[k][0])
    acc_dev, params = hasil_metode[terbaik]
    print(f"\n>> Metode terpilih (akurasi dev tertinggi): {terbaik}")
    print("   Parameter hasil tuning pada citra pengembangan:")
    tabel(["buah", "batas hue hijau/kuning", "lo", "hi"],
          [[k if k != "*" else "(global)", f"{v['hue']:.0f}°", f"{v['lo']:.2f}", f"{v['hi']:.2f}"]
           for k, v in params.items()])
    print("   Alasan: batas hue, lo, dan hi dicari lewat grid search pada citra pengembangan "
          "(bukan citra uji); lihat histogram_hue.png.")

    # --- Evaluasi akhir pada citra uji (hanya metode terpilih, parameter tetap)
    fungsi = METODE_SEG[terbaik]
    proses_semua(dev, fungsi); terapkan_params(dev, params)
    proses_semua(uji, fungsi); terapkan_params(uji, params)
    y_true = [d["label"] for d in uji]
    y_pred = [d["pred"] for d in uji]
    cm = matriks_konfusi(y_true, y_pred)

    print("\n" + "=" * 70)
    print(f"HASIL AKHIR PADA CITRA UJI  (n={len(uji)})")
    print("=" * 70)
    print(f"Akurasi keseluruhan: {akurasi(y_true, y_pred)*100:.1f}%")
    print("\nConfusion matrix (baris = sebenarnya, kolom = prediksi):")
    tabel(["sebenarnya \\ prediksi"] + LABELS,
          [[LABELS[i]] + list(cm[i]) for i in range(3)])

    print("\nPresisi / recall per kelas:")
    baris = []
    for i, l in enumerate(LABELS):
        tp = cm[i, i]; prec = tp / max(cm[:, i].sum(), 1); rec = tp / max(cm[i].sum(), 1)
        baris.append([l, f"{prec*100:.1f}%", f"{rec*100:.1f}%", cm[i].sum()])
    tabel(["kelas", "presisi", "recall", "jumlah"], baris)

    print("\nAkurasi per kondisi pengambilan citra:")
    baris = []
    for k in sorted({d["kondisi"] for d in uji}):
        sub = [d for d in uji if d["kondisi"] == k]
        baris.append([k, len(sub), f"{akurasi([d['label'] for d in sub], [d['pred'] for d in sub])*100:.1f}%"])
    tabel(["kondisi", "n", "akurasi"], baris)

    print("\nAkurasi per jenis buah:")
    baris = []
    for b in sorted({d["buah"] for d in uji}):
        sub = [d for d in uji if d["buah"] == b]
        baris.append([b, len(sub), f"{akurasi([d['label'] for d in sub], [d['pred'] for d in sub])*100:.1f}%"])
    tabel(["buah", "n", "akurasi"], baris)

    # --- Simpan keluaran
    simpan_konfusi(cm, f"Confusion matrix (uji) - {terbaik}", os.path.join(args.out, "confusion_matrix.png"))
    simpan_histogram_hue(dev, params, os.path.join(args.out, "histogram_hue.png"))
    with open(os.path.join(args.out, "prediksi_uji.csv"), "w", encoding="utf-8") as f:
        f.write("file,buah,kondisi,label_benar,prediksi,rasio_hijau,batas_hue,lo,hi\n")
        for d in uji:
            f.write(f"{d['nama']},{d['buah']},{d['kondisi']},{d['label']},{d['pred']},{d['rasio']:.4f},{d['hue']:.0f},{d['lo']:.2f},{d['hi']:.2f}\n")

    analisis_kegagalan(uji, args.out)
    analisis_benar(uji, args.out)
    print(f"\nSelesai. Gambar & CSV disimpan di folder '{args.out}/'.")

    if not args.tanpa_menu:                                                   
        menu_interaktif(uji, fungsi, params, terbaik, args.out)               


if __name__ == "__main__":
    main()