"""
Pipeline deteksi dan penghitungan sel darah merah (RBC)
pada citra mikroskopis menggunakan Thresholding Otsu + Operasi Morfologi.

Tahapan:
 1. Citra asli (BGR)
 2. Grayscale (atau channel hijau, kontras sel biasanya paling baik)
 3. Smoothing  : Gaussian blur
 4. Peningkatan kontras : CLAHE
 5. Segmentasi : Thresholding Otsu (inverse, sel lebih gelap dari latar)
 6. Morfologi  : opening (buang noise) -> closing (tutup celah) -> isi lubang
 7. Pemisahan sel yang menempel : distance transform + watershed
 8. Pelabelan komponen terhubung, filter luas, penghitungan
"""
import cv2
import numpy as np

DEFAULTS = {
    "channel": "green",      # "green" atau "gray"
    "blur_ksize": 5,         # ganjil
    "use_clahe": True,
    "morph_ksize": 5,        # ukuran kernel elips
    "use_watershed": True,
    "dist_ratio": 0.45,      # ambang sure-foreground terhadap maks distance transform
    "min_area": 300,         # px, di bawah ini dianggap noise / platelet
    "max_area_factor": 6.0,  # luas > faktor * median dianggap bukan satu sel tunggal
    "max_width": 800,        # resize agar proses konsisten
    "remove_wbc": True,      # keluarkan sel darah putih berdasarkan warna (HSV)
    "wbc_sat": 55,           # saturasi minimum inti WBC (0-255)
}


def detect_wbc(img_bgr, sat_min=55):
    """Deteksi sel darah putih (leukosit) dari warna inti sel.

    Inti WBC berwarna biru-ungu (hue OpenCV ~110-150) dengan saturasi tinggi,
    sedangkan RBC berwarna merah muda/cokelat (hue ~0-20 atau ~150-180, saturasi rendah).
    Mengembalikan (mask_inti, mask_wilayah_wbc, jumlah_wbc).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    nuc = ((h >= 108) & (h <= 150) & (s >= sat_min)).astype(np.uint8) * 255
    nuc = cv2.morphologyEx(nuc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    nuc = cv2.morphologyEx(nuc, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # buang bercak kecil; sisakan gumpalan besar (lobus inti)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(nuc)
    keep = np.zeros_like(nuc)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 500:
            keep[lab == i] = 255

    # gabungkan lobus yang berdekatan menjadi satu sel, lalu ambil convex hull
    grouped = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    n2, lab2 = cv2.connectedComponents(grouped)
    region = np.zeros_like(keep)
    count = 0
    for i in range(1, n2):
        pts = cv2.findNonZero(((lab2 == i) & (keep > 0)).astype(np.uint8))
        if pts is None:
            continue
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(region, hull, 255)
        count += 1
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    return keep, region, count


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _fill_holes(binary):
    """Isi lubang di dalam objek dengan menggambar ulang kontur luar."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def process(img_bgr, params=None):
    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})

    stages = []

    def add(title, desc, img):
        stages.append({"title": title, "desc": desc, "image": img})

    # --- resize -----------------------------------------------------------
    h, w = img_bgr.shape[:2]
    if w > p["max_width"]:
        scale = p["max_width"] / w
        img_bgr = cv2.resize(img_bgr, (p["max_width"], int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    add("1. Citra asli", "Citra input hasil upload (di-resize bila terlalu lebar).", img_bgr)

    # --- grayscale --------------------------------------------------------
    if p["channel"] == "green":
        gray = img_bgr[:, :, 1]
        desc = "Channel hijau dipakai karena sel darah merah tampak paling gelap di channel ini."
        title = "2. Grayscale (channel hijau)"
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        desc = "Konversi BGR ke grayscale (0.299 R + 0.587 G + 0.114 B)."
        title = "2. Grayscale"
    add(title, desc, gray)

    # --- smoothing --------------------------------------------------------
    k = _odd(p["blur_ksize"])
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    add("3. Smoothing (Gaussian blur)",
        f"Gaussian blur kernel {k}x{k} untuk meredam noise sebelum thresholding.", blur)

    # --- contrast ---------------------------------------------------------
    if p["use_clahe"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enh = clahe.apply(blur)
        add("4. Peningkatan kontras (CLAHE)",
            "CLAHE menyeragamkan kontras lokal sehingga threshold lebih stabil.", enh)
    else:
        enh = blur

    # --- otsu -------------------------------------------------------------
    otsu_t, binary = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    add("5. Thresholding Otsu",
        f"Nilai ambang otomatis Otsu = {otsu_t:.0f}. Mode inverse: sel (gelap) menjadi putih.", binary)

    # --- morphology -------------------------------------------------------
    mk = _odd(p["morph_ksize"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    filled = _fill_holes(closed)
    add("6. Operasi morfologi",
        f"Opening, closing (kernel elips {mk}x{mk}), lalu hole filling agar bagian tengah sel yang pucat ikut terisi.",
        filled)

    # --- sel darah putih (WBC): deteksi warna lalu keluarkan -------------
    wbc_region, wbc_count = np.zeros_like(filled), 0
    if p["remove_wbc"]:
        wbc_nuc, wbc_region, wbc_count = detect_wbc(img_bgr, p["wbc_sat"])
        vis = img_bgr.copy()
        vis[wbc_nuc > 0] = (255, 80, 0)
        cnts, _ = cv2.findContours(wbc_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 0, 255), 2)
        add("6b. Deteksi WBC berdasarkan warna",
            f"Segmentasi HSV (hue biru-ungu, saturasi >= {p['wbc_sat']}) menemukan inti sel darah putih "
            f"({wbc_count} WBC). Area WBC (kontur magenta) dikeluarkan dari hitungan RBC.", vis)
        filled = filled.copy()
        filled[wbc_region > 0] = 0
        add("6c. Mask RBC setelah WBC dihapus",
            "Piksel milik WBC dibuang dari mask sehingga tidak ikut terhitung sebagai RBC.", filled)

    # --- watershed / pemisahan -------------------------------------------
    if p["use_watershed"]:
        dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
        dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        add("7a. Distance transform",
            "Jarak tiap piksel sel ke tepi terdekat; pusat sel bernilai paling terang.", dist_vis)

        # Perkiraan jari-jari sel: median dari nilai maks distance transform
        # tiap komponen (sel tunggal mendominasi, jadi median cukup stabil).
        n_cc, cc = cv2.connectedComponents(filled)
        peaks_cc = [dist[cc == i].max() for i in range(1, n_cc)
                    if (cc == i).sum() >= p["min_area"]]
        radius = float(np.median(peaks_cc)) if peaks_cc else 10.0

        # Puncak lokal: titik yang maksimum dalam jendela ~ 0.8 x jari-jari
        # dan cukup dalam di dalam sel (dist_ratio x jari-jari).
        win = max(3, int(round(radius * 1.6)) | 1)
        local_max = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win)))
        sure_fg = ((dist >= local_max - 1e-6) & (dist >= p["dist_ratio"] * radius)).astype(np.uint8) * 255
        # gabungkan piksel puncak yang berdekatan (plateau) jadi satu marker
        sure_fg = cv2.dilate(sure_fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        sure_bg = cv2.dilate(filled, kernel, iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        n_fg, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        ws = cv2.watershed(img_bgr.copy(), markers.astype(np.int32))
        # label sel: 2..n ; batas = -1 ; latar = 1
        labels = ws.copy()
        labels[labels <= 1] = 0
        labels[filled == 0] = 0
        sep = (labels > 0).astype(np.uint8) * 255
        add("7b. Hasil watershed",
            "Sel yang saling menempel dipisahkan dengan garis batas watershed.", sep)
    else:
        n, labels = cv2.connectedComponents(filled)
        labels = labels.astype(np.int32)

    # --- counting ---------------------------------------------------------
    ids = [i for i in np.unique(labels) if i > 0]
    areas = {i: int((labels == i).sum()) for i in ids}
    valid = {i: a for i, a in areas.items() if a >= p["min_area"]}
    median_area = float(np.median(list(valid.values()))) if valid else 0.0
    max_area = median_area * p["max_area_factor"] if median_area else 1e9

    # objek sangat besar (mis. leukosit / gumpalan) tidak dihitung sebagai 1 RBC
    cells, ignored_big = [], 0
    for i, a in valid.items():
        if a > max_area:
            ignored_big += 1
        else:
            cells.append(i)

    result = img_bgr.copy()
    for n_idx, i in enumerate(sorted(cells), start=1):
        mask = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, cnts, -1, (0, 200, 0), 2)
        M = cv2.moments(mask, binaryImage=True)
        if M["m00"]:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.putText(result, str(n_idx), (cx - 8, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    if wbc_count:
        wc, _ = cv2.findContours(wbc_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, wc, -1, (255, 0, 255), 2)
        for c in wc:
            x, y, _, _ = cv2.boundingRect(c)
            cv2.putText(result, "WBC", (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(result, (0, 0), (210, 34), (255, 255, 255), -1)
    cv2.putText(result, f"Jumlah RBC: {len(cells)}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2, cv2.LINE_AA)
    add("8. Hasil deteksi dan penghitungan",
        "Kontur hijau = RBC terdeteksi, angka biru = nomor urut, kontur magenta = WBC (tidak dihitung).",
        result)

    cell_areas = [valid[i] for i in cells]
    return {
        "count": len(cells),
        "otsu_threshold": float(otsu_t),
        "median_area": float(np.median(cell_areas)) if cell_areas else 0.0,
        "ignored_small": len(areas) - len(valid),
        "ignored_big": ignored_big,
        "wbc_count": wbc_count,
        "stages": stages,
        "params": p,
    }
